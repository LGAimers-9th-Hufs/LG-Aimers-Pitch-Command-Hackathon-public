# -*- coding: utf-8 -*-
"""시대 다양화 레그(era-diversification leg) 제출 zip 빌더 — 분기 A(T2) 산출물.

    python submission/build_era_leg.py --tag era_all

무엇인가 (docs/log/37 §4 · results/tm_repr/era_bag.json)
--------------------------------------------------------
2019~2024 **전 시즌 균등 풀링**으로 학습한 LGBM 5시드 배깅 레그. 최신-시즌 레시피보다
솔로는 낮지만(게이트 fit≤2023→V24: 배깅 579.5 vs june풍 단일 645.7) D4 레그들과의
불일치가 d(D4)=0.0437로 기존 모든 레그(0.024~0.026 벽)를 크게 넘는다 — Brier 항등식
`Score(p̄) = 평균Score + C·E[(p_i−p̄)²]` 이 그 불일치를 블렌드 이득으로 바꾼다.
채택 산수(2024 프록시 30k행 정확 gain): E₅ = (4026.56+S_est)/5 + 181.55/1.4295 = 1083.9.

구성: XA(78) = X(is4 포함 57) + bis(6) + ebps(3) + cxp(12) — ENS-8과 같은 조립이라
서빙 템플릿 `script_lgbm_template.py`의 build_features를 **그대로** 쓴다(멤버만
lgbm×5, input="full"). 캘리 = 합법 레거시 (1.04, −0.01). 상수 전부 train 유래.

게이트: ① 피처 파리티(학습 vs 서빙 경로, ≤1e-9) ② 부스터 텍스트 왕복 파리티
③ 프록시 30k 캘리 후 d(D4)·정확 gain 리포트. 빌드 후 반드시:
    python submission/scan_probe_provenance.py submission/dist/submit_<tag>.zip   # BLOCK 0
    python submission/verify_submission.py submission/dist/submit_<tag>.zip --proxy 245789 --threads 6
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sweep"))
sys.path.insert(0, str(ROOT / "submission"))
import real_data as rd                    # noqa: E402
import lgbm_family as L                   # noqa: E402
import inseason as IS                     # noqa: E402
import inseason_full as IF                # noqa: E402
from tm_repr_leg import build_xa          # noqa: E402
import script_lgbm_template as SRV        # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REQUIREMENTS = "lightgbm==4.6.0\nscikit-learn==1.8.0\njoblib==1.5.3\npandas==2.3.3\n"
PARAMS = dict(objective="binary", metric="binary_logloss", learning_rate=0.05,
              num_leaves=15, min_data_in_leaf=1000, feature_fraction=0.85,
              bagging_fraction=0.8, bagging_freq=1, verbosity=-1)
SEEDS = (0, 1, 2, 3, 4)
ROUNDS = 600
CALIB = dict(center=0.5, slope=1.04, shift=-0.01)     # 합법 레거시 쌍(로컬 재적합 반올림)
P_SUCC = [("asof_pitcher_success_rate", "succ")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="era_all")
    ap.add_argument("--outdir", default=str(ROOT / "submission" / "dist"))
    args = ap.parse_args()
    t0 = time.time()

    print(">> train.csv 로드 + XA(78) 조립 (tm_repr_leg.build_xa — 게이트와 동일 경로)")
    df = rd.load_train()
    y = df[rd.TARGET].to_numpy(dtype=np.float32)
    season = df[rd.SEASON].to_numpy()
    XA = build_xa(df, season)                          # L.K_IS = 100 로 설정됨
    feat_names = list(XA.columns)
    assert len(feat_names) == 78, f"XA 폭이 78이 아니다: {len(feat_names)}"
    serve_season = int(season.max()) + 1               # 2025
    print(f"   XA {XA.shape} · 학습 = 전 시즌 균등 {len(df):,}행  [{time.time()-t0:.0f}s]")

    # ---- 5시드 학습 + 텍스트 부스터 저장 ------------------------------------------
    stage = Path(args.outdir) / f"_stage_{args.tag}"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "model").mkdir(parents=True)
    members, boosters = [], []
    for s in SEEDS:
        t1 = time.time()
        m = lgb.train(dict(PARAMS, seed=s), lgb.Dataset(XA, label=y),
                      num_boost_round=ROUNDS)
        fn = f"era_s{s}.txt"
        m.save_model(str(stage / "model" / fn))
        boosters.append(m)
        members.append({"type": "lgbm", "weight": 1.0 / len(SEEDS), "name": f"era_s{s}",
                        "scope": "all", "input": "full", "file": fn})
        mb = (stage / "model" / fn).stat().st_size / 1e6
        print(f"   era_s{s}: {ROUNDS}r × {PARAMS['num_leaves']}leaf  {mb:5.1f}MB  "
              f"[{time.time()-t1:.0f}s]")

    # ---- 서빙 룩업 (전부 train 유래, 마지막 기록 시즌 ≤2024 말) --------------------
    lut = IS.career_end_lookup(df)
    pid_all = df["pitcher_id"].to_numpy()
    serve_base = {}
    for p in np.unique(pid_all):
        ss = [s for (q, s) in lut if q == int(p) and s < serve_season]
        if ss:
            n_end, k_end = lut[(int(p), max(ss))]
            serve_base[str(int(p))] = [float(n_end), float(k_end)]

    b_ent, b_ncol, b_rc, K_BAT = next((e[1], e[2], e[3], e[4])
                                      for e in IF.BLOCKS if e[0] == "b")
    blut = IF.end_lookup(df, b_ent, b_ncol, b_rc)
    by_b = {}
    for (b_, s_), v_ in blut.items():
        by_b.setdefault(b_, {})[s_] = v_
    serve_batter, train_batter = {}, {}
    for b_, hist in by_b.items():
        n_e, ks_ = hist[max(hist)]
        serve_batter[str(int(b_))] = [float(n_e), float(ks_[0]), float(ks_[1])]
        prev_ss = [q for q in hist if q < 2024]
        if prev_ss:
            n_e, ks_ = hist[max(prev_ss)]
            train_batter[str(int(b_))] = [float(n_e), float(ks_[0]), float(ks_[1])]

    plut = IF.end_lookup(df, "pitcher_id", "asof_pitcher_n", P_SUCC)
    by_p = {}
    for (p_, s_), v_ in plut.items():
        by_p.setdefault(p_, {})[s_] = v_
    serve_peb, train_peb = {}, {}
    for p_, hist in by_p.items():
        n_e, ks_ = hist[max(hist)]
        serve_peb[str(int(p_))] = [float(n_e), float(ks_[0])]
        prev_ss = [q for q in hist if q < 2024]
        if prev_ss:
            n_e, ks_ = hist[max(prev_ss)]
            train_peb[str(int(p_))] = [float(n_e), float(ks_[0])]
    peb_league = float(np.nanmean(df["asof_pitcher_success_rate"].to_numpy(dtype=float)))
    print(f"   룩업: is4 {len(serve_base):,}투수 · bis {len(serve_batter):,}타자 · "
          f"pEB {len(serve_peb):,}투수 · league {peb_league:.6f}")

    # ---- 피처 파리티: 학습 경로(build_xa) vs 서빙 경로(script.py) -------------------
    # 2024 행 20k 표본. 서빙 함수에 **학습 빈티지**(2024행이 실제 조회한 s−1 말) 룩업을 넣어
    # 코드 동형성을 검증한다(빌더 표준 패턴 — build_lgbm_ensemble.py 697행과 동일).
    idx24 = np.where(season == 2024)[0][:20000]
    nb24, kb24 = IS.base_for(lut, pid_all[idx24], season[idx24])
    train_base = {}
    for p, n_, k_ in zip(pid_all[idx24], nb24, kb24):
        train_base.setdefault(str(int(p)), [float(n_), float(k_)])
    probe = {"category_maps": L.CAT_MAPS, "feature_names": feat_names,
             "inseason_base": train_base, "inseason_k": L.K_IS,
             "inseason_batter": train_batter, "inseason_batter_k": float(K_BAT),
             "pkg": {"peb_kappa": 100.0, "peb_league": peb_league,
                     "peb_base": train_peb}}
    df_s = df.iloc[idx24].reset_index(drop=True)
    serve_X = SRV.build_features(df_s, probe)
    train_X = XA.iloc[idx24].reset_index(drop=True)
    dmax = float(np.nanmax(np.abs(serve_X.to_numpy(dtype=np.float64)
                                  - train_X.to_numpy(dtype=np.float64))))
    nan_ok = bool((serve_X.isna().to_numpy() == train_X.isna().to_numpy()).all())
    print(f"   [피처 파리티] 최대 절대차 = {dmax:.3e} · 결측 패턴 일치 = {nan_ok}")
    if dmax > 1e-9 or not nan_ok:
        sys.exit("피처 파리티 실패 — 학습 경로와 서빙 경로가 어긋난다")

    # ---- 부스터 텍스트 왕복 파리티 -------------------------------------------------
    for mem, m in zip(members, boosters):
        m2 = lgb.Booster(model_file=str(stage / "model" / mem["file"]))
        d_ = float(np.abs(m2.predict(train_X) - m.predict(train_X)).max())
        if d_ > 1e-12:
            sys.exit(f"부스터 왕복 파리티 실패({mem['name']}): {d_:.3e}")
    print(f"   [부스터 파리티] 텍스트 왕복 5/5 · 최대차 ≤1e-12")

    # ---- 프록시 30k 리포트: 캘리 후(zip 출력 규약) d(D4)·정확 gain₅ ------------------
    # ⚠ 참고용 — 학습에 2024가 들어가 in-sample 수축이 있다(모든 캐시 레그와 같은 규약).
    #   채택 판정은 fit≤2023 게이트의 E₅=1083.9(era_bag.json)가 이미 내렸다.
    idx30 = np.where(season == 2024)[0][:30000]
    p_raw = np.mean([m.predict(XA.iloc[idx30]) for m in boosters], axis=0)
    p_cal = np.clip(0.5 + CALIB["slope"] * (p_raw - 0.5) + CALIB["shift"], 0.0, 1.0)
    legs = {}
    for nm in ("clookup", "cmoe", "physmix", "tm3L"):
        fs = sorted(glob.glob(str(ROOT / "results/leg_matrix" / f"{nm}_30000_*.npy")))
        if fs:
            legs[nm] = np.load(fs[-1])
    if len(legs) == 4:
        L4 = list(legs.values())
        D4 = np.mean(L4, axis=0)
        d4 = float(np.sqrt(np.mean((p_cal - D4) ** 2)))
        ps = L4 + [p_cal]
        pbar = np.mean(ps, axis=0)
        g5 = 4e5 * float(np.mean([np.mean((p - pbar) ** 2) for p in ps]))
        print(f"   [프록시 30k] 캘리 후 d(D4) = {d4:.5f} · 정확 gain₅ = {g5:.2f} "
              f"(in-sample 수축 포함 — 게이트 정직값은 0.0437/181.55)")

    # ---- metadata + zip ------------------------------------------------------------
    meta = {
        "version": f"era-{args.tag}",
        "training_seasons": "2019-2024 pooled uniformly",
        "n_train_rows": int(len(df)),
        "feature_names": feat_names,
        "category_maps": L.CAT_MAPS,
        "inseason_base": serve_base,
        "inseason_k": L.K_IS,
        "inseason_batter": serve_batter,
        "inseason_batter_k": float(K_BAT),
        "pkg": {"peb_kappa": 100.0, "peb_league": peb_league, "peb_base": serve_peb},
        "base_ncols": 57,
        "members": members,
        "calibration": CALIB,
        "fallback_constant": 0.5,
        "provenance": [
            f"trained on ALL train.csv seasons 2019-2024 pooled uniformly "
            f"({len(df)} rows; 5 LightGBM binary-objective boosters, seeds 0-4, equal "
            f"weight, 78-column input). This is a deliberate era-diversification leg for "
            f"the team probability-average blend: pooling old seasons is weaker solo than "
            f"the latest-season recipe on the fit<=2023 -> val-2024 gate, but its "
            f"disagreement with the recent-window legs is the point - the Brier identity "
            f"Score(mean p) = mean(Score) + C*E[(p_i - pbar)^2] converts squared "
            f"disagreement into a structural blend gain on any dataset. Adoption was "
            f"decided by that identity arithmetic on a train-2024 holdout sample "
            f"(results/tm_repr/era_bag.json); every bundled constant is derived from "
            f"train.csv alone.",
            "calibration (1.04, -0.01) applied after the equal-weight average: the same "
            "legal pair used by our other legs - it matches the fit-2023 -> val-2024 "
            "local-protocol refit (1.035270, -0.010359) within rounding; derived from "
            "training data only.",
            "the in-season block (is_logn/is_share/is_sm/is_delta) subtracts a bundled "
            "per-pitcher constant - that pitcher's cumulative (n, successes) at the end of "
            "their last recorded train season - from the official career-cumulative asof "
            "columns, isolating current-season form. Row-wise arithmetic plus a frozen "
            "train-derived lookup; no reference to other evaluation rows (rules 5).",
            "the batter in-season block (isb_*) and the pitcher EB-innovation block "
            "(ebps_*) subtract bundled per-entity season-end constants the same way; the "
            "cxp columns are row-wise products of columns already computed for that row. "
            "All features are row-wise transforms of that row's official columns plus "
            "frozen train-derived lookups; no other evaluation row is ever referenced "
            "(rules 5).",
        ],
    }
    (stage / "model" / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.copy2(ROOT / "submission" / "script_lgbm_template.py", stage / "script.py")
    (stage / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")

    zpath = Path(args.outdir) / f"submit_{args.tag}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(stage).as_posix())
    shutil.rmtree(stage)
    print(f"\n생성: {zpath}  ({zpath.stat().st_size/1e6:.1f}MB, {time.time()-t0:.0f}s)")
    print(f"다음: python submission/scan_probe_provenance.py {zpath}")
    print(f"      python submission/verify_submission.py {zpath} --proxy 245789 --threads 6")


if __name__ == "__main__":
    main()
