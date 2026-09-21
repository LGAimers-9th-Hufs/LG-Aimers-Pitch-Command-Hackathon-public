# -*- coding: utf-8 -*-
"""제출 zip 빌더 v2 — 재캘리 레이어 + credibility 룩업 + β 사전등록.

    python submission/build_submission.py --tag new1 --model hgb82_reg --beta 0.04
    python submission/build_submission.py --tag new2 --model hgb82_reg --beta 0.12
    python submission/build_submission.py --tag off1 --model glm_offset_hgb --beta 0.04

v2 스택 (269점 부검 반영):
  [베이스]  BASE_MODELS 중 선택 (기본 hgb82_reg = 정칙화 HGB)
  [재캘리]  logit p* = a + b·logit p + c·(z_pitcher−z̄) + d·(prev5−t̄) + β
            (a,b,c,d)는 fit1(≤2023) 모델의 2024 예측에서 Brier 직접 최소화로 적합
            → isotonic 폐기(refinement −150~−190 실측 + 예측을 전년 수준에 고정)
  [레벨]    β = 사전 등록 스칼라(자유 절편). LB로 선택. train 유도식은 provenance에 기록
  [룩업]    z_pitcher = Bühlmann–Straub 신뢰도 수축 투수 로짓 (train 전체로 계산해 동봉 — §5 허용)

pickle 안전: 커스텀 클래스를 절대 pickle에 넣지 않는다. base.joblib = 순수 sklearn 객체들의
dict {"kind": "single"|"offset"|"avg", ...} — 서버에 우리 모듈이 없어도 언피클된다.
"""
from __future__ import annotations
import argparse, json, shutil, sys, time, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sweep"))
import real_data as rd                                    # noqa: E402
import recal as rc                                        # noqa: E402
from run_real import BASE_MODELS, THERMO, _OffsetBoost, _SeedAvg  # noqa: E402

SERVE_BEGIN = "# ================================ <<<SERVE:BEGIN>>> ================================"
SERVE_END = "# ================================= <<<SERVE:END>>> ================================="
REQUIREMENTS = "scikit-learn==1.8.0\njoblib==1.5.3\npandas==2.3.3\n"
TARGET_SEASON = 2025
CAL_SEASON = 2024


def extract_serve_block() -> str:
    src = (ROOT / "sweep" / "real_data.py").read_text(encoding="utf-8")
    i, j = src.index(SERVE_BEGIN), src.index(SERVE_END) + len(SERVE_END)
    return src[i:j] + "\n"


def build_fallback(X: pd.DataFrame, y: np.ndarray, sample=300_000, seed=42) -> dict:
    """순수 numpy 로지스틱 폴백 — base 언피클 실패 시의 보험."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    A = X.iloc[idx].to_numpy(dtype=np.float64)
    med = np.nanmedian(A, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    nz = np.where(np.isnan(A))
    A[nz] = np.take(med, nz[1])
    mu, sd = A.mean(axis=0), A.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    lr = LogisticRegression(max_iter=1000).fit((A - mu) / sd, y[idx])
    return {"median": med.tolist(), "mean": mu.tolist(), "scale": sd.tolist(),
            "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}


def bundle_model(est) -> dict:
    """학습된 estimator를 순수 sklearn 객체 dict로 분해 (커스텀 클래스 pickle 금지)."""
    if isinstance(est, _OffsetBoost):
        return {"kind": "offset", "glm": est.glm, "gbr": est.gbr, "glm_cols": est.GLM_COLS}
    if isinstance(est, _SeedAvg):
        return {"kind": "avg", "models": est.models}
    return {"kind": "single", "est": est}


def parse_blend(spec: str) -> list:
    """'glm_offset_hgb:0.5:-0.02,rf_official:0.5:0' → 멤버 사양 리스트 (모델:가중:β)."""
    out = []
    for tok in spec.split(","):
        parts = (tok.strip().split(":") + ["1", "0"])[:3]
        name = parts[0]
        if name not in BASE_MODELS:
            sys.exit(f"알 수 없는 모델: {name}")
        out.append({"name": name, "weight": float(parts[1]), "beta": float(parts[2])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="hgb82_reg", choices=list(BASE_MODELS))
    ap.add_argument("--blend", default=None,
                    help="확률 블렌드: '모델:가중:β,모델:가중:β' "
                         "(예: glm_offset_hgb:0.5:-0.02,rf_official:0.5:0). "
                         "지정 시 --model/--beta는 무시되고 재캘리는 자동 비활성.")
    ap.add_argument("--beta", type=float, default=0.04,
                    help="레벨 절편(로짓). 사전 등록 격자에서 선택 — docs/01_DECISIONS.md 기록 필수")
    ap.add_argument("--no-recal", action="store_true", help="재캘리 레이어 생략(K0+β만)")
    ap.add_argument("--outdir", default=str(ROOT / "submission" / "dist"))
    args = ap.parse_args()

    t0 = time.time()
    print(">> train.csv 로드")
    df = rd.load_train()
    X = rd.build_features(df)
    y = df[rd.TARGET].to_numpy()
    season = df[rd.SEASON].to_numpy()
    print(f"   {len(df):,}행 · 피처 {X.shape[1]}개")

    if args.blend:
        members = parse_blend(args.blend)
        args.no_recal = True                      # 블렌드는 K0 + 멤버별 β만 (D-11a 정합)
        calib = {"feature_order": list(rd.FEATURE_ORDER), "rep": "blend",
                 "clip": [0.001, 0.999], "thermo_col": THERMO, "beta": 0.0,
                 "model": "+".join(m["name"] for m in members), "tag": args.tag}
        prov = [f"train=train.csv 2019-2024 ({len(df)} rows)",
                "probability blend (row-wise weighted average of member probabilities). "
                "Rationale: Brier identity B(p̄) = mean(B) − mean((p_i−p_j)²)/4 makes the "
                "diversity gain structural; members' relative edge is regime-dependent "
                "(fold edge +245/−265, see docs/log/14 §2.2) so averaging hedges the regime bet."]
        built = []
        for mb in members:
            rep_m, est = BASE_MODELS[mb["name"]][1]()
            data_m = df[rd.CAT_COLS + rd.OFFICIAL_NUM] if rep_m == "raw47" else X
            print(f">> 블렌드 멤버 학습: {mb['name']} (rep={rep_m}, w={mb['weight']}, "
                  f"β={mb['beta']:+.3f})")
            est.fit(data_m, y)
            built.append({**bundle_model(est), "rep": rep_m, "weight": mb["weight"],
                          "beta": mb["beta"], "name": mb["name"]})
            prov.append(f"member {mb['name']} rep={rep_m} weight={mb['weight']} "
                        f"beta={mb['beta']:+.3f} — refit on all of train.csv (2019-2024)")
        m2 = None
        blend_bundle = {"kind": "blend", "members": built}
        calib["blend"] = [{k: mb[k] for k in ("name", "rep", "weight", "beta")} for mb in built]
    else:
        members = None
        blend_bundle = None

    tier, factory = BASE_MODELS[args.model]
    rep, _ = factory()
    data = df[rd.CAT_COLS + rd.OFFICIAL_NUM] if rep == "raw47" else X

    if not args.blend:
        calib = {"feature_order": list(rd.FEATURE_ORDER),
                 "rep": rep, "clip": [0.001, 0.999],
                 "thermo_col": THERMO, "beta": float(args.beta),
                 "model": args.model, "tag": args.tag}
        prov = [f"model={args.model} (rep={rep})",
                f"train=train.csv 2019-2024 ({len(df)} rows)",
                f"beta={args.beta:+.3f} logit — pre-registered level scalar, chosen by public-LB "
                f"selection among registered candidates (see docs/01_DECISIONS.md D-10)"]

    if not args.no_recal:
        # 풀링 OOF 적합 (RC2 구성): cal 시즌 {2022, 2023, 2024} 각각을 "그 이전 전체로 학습한
        # 모델"이 예측 → 세 배치를 합쳐 (a,b,c,d) 적합. 단일 cal 시즌(예: 2023 F 파단)이
        # 병리적일 때 Brier 최소화가 '모델을 버려라'로 퇴화하는 것을 시즌 희석으로 방어한다.
        batches = []
        for cs in (CAL_SEASON - 2, CAL_SEASON - 1, CAL_SEASON):
            fit_m = season < cs
            cal_m = season == cs
            print(f">> OOF 배치: fit(≤{cs-1}) → cal({cs}) 예측")
            _, mm = factory()
            mm.fit(data[fit_m], y[fit_m])
            p_c = mm.predict_proba(data[cal_m])[:, 1]
            tab_f, zdef_f, K_f, _ = rd.build_credibility(df[fit_m])
            pid_c = df["pitcher_id"][cal_m].to_numpy()
            batches.append({"p": p_c, "y": y[cal_m],
                            "z": np.array([tab_f.get(int(x), zdef_f) for x in pid_c]),
                            "t": df[THERMO][cal_m].to_numpy(dtype=float),
                            "zc": zdef_f,
                            "tc": float(np.nanmean(df[THERMO][fit_m].to_numpy(dtype=float)))})

        tab_all, zdef_all, K_all, m_all = rd.build_credibility(df)
        params = rc.fit_recal_pooled(batches)
        calib["recal"] = params
        calib["z_table"] = {str(k): float(v) for k, v in tab_all.items()}
        calib["z_default"] = zdef_all
        prov += [f"recal (a,b,c,d)=({params['a']:+.4f},{params['b']:.4f},"
                 f"{params['c']:+.4f},{params['d']:+.4f}) fit by direct Brier minimization on "
                 f"POOLED out-of-fold predictions for seasons "
                 f"{CAL_SEASON-2}/{CAL_SEASON-1}/{CAL_SEASON} (each predicted by a model trained "
                 f"strictly before that season; coefficients bounded b in [0.2,3])",
                 f"z_table: Buhlmann-Straub credibility-shrunk pitcher success logits from all "
                 f"train rows (K={K_all:.0f}, m={m_all:.4f}); fitting used per-batch fit-period "
                 f"tables to avoid cal leakage",
                 f"centers: z_center={params['z_center']:.4f} t_center={params['t_center']:.4f} "
                 f"(latest-batch fit-period constants, shipped verbatim)"]
        print(f"   recal(pooled {params['n_pool']}시즌): a={params['a']:+.3f} b={params['b']:.3f} "
              f"c={params['c']:+.3f} d={params['d']:+.3f} | z_table {len(tab_all)}명 K={K_all:.0f}")

    if blend_bundle is None:
        print(">> fit2(전체 2019-2024) 재적합 — 제출 모델")
        _, m2 = factory()
        m2.fit(data, y)
        prov.append("final model refit on all of train.csv (2019-2024)")

    print(">> 폴백 로지스틱 적합")
    calib["fallback"] = build_fallback(X, y)
    calib["provenance"] = prov

    # ---- 패키징 (최상위 = model/ · script.py · requirements.txt 고정) ----
    outdir = Path(args.outdir)
    stage = outdir / f"_stage_{args.tag}"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "model").mkdir(parents=True)
    joblib.dump(blend_bundle if blend_bundle is not None else bundle_model(m2),
                stage / "model" / "base.joblib", compress=3)
    (stage / "model" / "serve_features.py").write_bytes(extract_serve_block().encode("utf-8"))
    (stage / "model" / "calib.json").write_bytes(
        json.dumps(calib, ensure_ascii=False, indent=1).encode("utf-8"))
    shutil.copy2(ROOT / "submission" / "script_template.py", stage / "script.py")
    (stage / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")

    zpath = outdir / f"submit_{args.tag}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(stage).as_posix())
    shutil.rmtree(stage)
    print(f"\n생성: {zpath}  ({zpath.stat().st_size/1e6:.1f}MB, {time.time()-t0:.0f}s)")
    print("검증: python submission/verify_submission.py", zpath)


if __name__ == "__main__":
    main()
