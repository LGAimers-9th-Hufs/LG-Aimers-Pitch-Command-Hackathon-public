# -*- coding: utf-8 -*-
"""Program A — 앙상블 멤버 대량 생산. **핵심은 멤버 수가 아니라 불일치 `s`다.**

    python sweep/member_factory.py --only algo          # A1: XGB/CatBoost만 (s의 핵심 미지수)
    python sweep/member_factory.py                      # 전체 풀
    python sweep/member_factory.py --val 2023           # 선택 낙관 측정용 재현 폴드

## 왜 이 모듈인가

블렌드 항등식 `이득 = 4e5·s²·(1−1/M)`에 실측을 넣으면(ENS-2: 멤버 가중평균 679.3 → 블렌드 773.5,
공식값 216 ⇒ 실현율 0.44):

| M | s=0.026 | s=0.030 | s=0.035 | s=0.040 |
|---|---|---|---|---|
| 30 | 로컬 794 | 832 | 888 | **951** |

**멤버 수만 늘리면 +20뿐이고 이득은 거의 전부 `s`에서 나온다.** 그래서 이 모듈의 목적은
"많이 만들기"가 아니라 **"같은 급인데 다르게 틀리는 것 찾기"**다.

## 지금까지 측정된 s (vs ENS-2)

| 멤버 | 단독 | s | 판정 |
|---|---|---|---|
| `all_raw`(전 시즌 원시) | 319 | **0.0410** | 저가중에서 +10.8 — **단독 최악인데 최대 기여** |
| `all_raw_d70` | 459 | 0.0347 | +9.6 |
| `ft_all2last` | 694 | 0.0187 | +4.0 — 단독 최고인데 기여 최소 |
| ET 계열 | 685 | 0.022 | (ENS-2 멤버) |
| `sk_hgb` | 692 | 0.009 | 무익 |
| `lg_dart` | −196 | 0.052 | **유해** (품질 붕괴) |

⇒ **단독 성능은 필요조건이 아니다(D-26). 단, 품질이 붕괴하면 항등식 첫 항이 이득을 먹는다.**

## A1이 왜 최우선인가

XGBoost·CatBoost는 원래 블루프린트의 M1~M3였는데 "로컬 미설치·설치 리스크"로 빠졌고
그 이유가 전부 무효가 됐다(서버는 설치 중 인터넷 가용, 설치 실패는 제출 횟수 미반영).
같은 GBDT라도 히스토그램 분할(LGBM) vs **ordered boosting·대칭 트리(CatBoost)**는
알고리즘 수준에서 다르게 틀리므로, 시드·파라미터 변형과는 **차원이 다른 s**를 기대할 수 있다.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import real_data as rd            # noqa: E402
import lgbm_family as L           # noqa: E402
import season_centering as SC     # noqa: E402
import phase2_ens_check as EC     # noqa: E402

OUT_DIR = HERE.parent / "results" / "programA"
NAN = -999.0
THREADS = 6          # ⚠ 서버는 6 vCPU. 시간 측정을 서버와 맞추기 위해 학습·예측 모두 6으로 고정한다.
WS = (0.05, 0.10, 0.15, 0.20, 0.30)


# ---------------------------------------------------------------- 후보 정의
def pool(group="all"):
    """(name, kind, spec). kind가 학습 방법을, spec이 파라미터를 정한다."""
    P = []

    # --- A1 알고리즘 다양성 (되살아난 M1~M3) --------------------------------
    if group in ("all", "algo"):
        for nm, sp in [
            ("xgb_d6",   dict(max_depth=6, min_child_weight=200, n_estimators=500,
                              learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                              reg_lambda=8.0)),
            ("xgb_d4",   dict(max_depth=4, min_child_weight=500, n_estimators=700,
                              learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
                              reg_lambda=8.0)),
            ("xgb_d10",  dict(max_depth=10, min_child_weight=100, n_estimators=300,
                              learning_rate=0.03, subsample=0.7, colsample_bytree=0.6,
                              reg_lambda=20.0)),
            # 결정적으로 다른 성장 방식: leaf-wise(=LGBM 유사) — 대조군
            ("xgb_lw",   dict(max_depth=0, grow_policy="lossguide", max_leaves=15,
                              min_child_weight=1000, n_estimators=400, learning_rate=0.03,
                              reg_lambda=8.0)),
        ]:
            P.append((nm, "xgb", sp))
        for nm, sp in [
            ("cat_d6",   dict(depth=6, iterations=600, learning_rate=0.05, l2_leaf_reg=8.0)),
            ("cat_d8",   dict(depth=8, iterations=500, learning_rate=0.05, l2_leaf_reg=20.0)),
            # langevin = 확률적 경사, 대칭 트리와 합쳐 LGBM과 가장 멀어지는 구성
            ("cat_lang", dict(depth=6, iterations=600, learning_rate=0.05, l2_leaf_reg=8.0,
                              langevin=True, diffusion_temperature=10000)),
        ]:
            P.append((nm, "cat", sp))

    # --- A2 랜덤 부분공간 (멤버마다 피처 집합을 **영구 고정**) ----------------
    if group in ("all", "sub"):
        for i, frac in enumerate([0.5, 0.5, 0.6, 0.6, 0.7, 0.7]):
            P.append((f"sub{int(frac*100)}_{i}", "lgbm_sub",
                      dict(frac=frac, seed=100 + i,
                           num_leaves=15, min_data_in_leaf=1000, num_iterations=400)))

    # --- A3 배깅 ------------------------------------------------------------
    if group in ("all", "bag"):
        for i in range(3):
            P.append((f"bag_{i}", "lgbm_bag",
                      dict(seed=200 + i, num_leaves=15, min_data_in_leaf=1000,
                           num_iterations=400)))

    # --- A4 목적함수·용량 혼합 ----------------------------------------------
    if group in ("all", "mix"):
        for nm, sp in [
            ("lg_bin",   dict(objective="binary", metric="binary_logloss",
                              num_leaves=15, min_data_in_leaf=1000, num_iterations=500, seed=19)),
            ("lg_huber", dict(objective="huber", alpha=0.5, num_leaves=15,
                              min_data_in_leaf=1000, num_iterations=500, seed=21)),
            ("lg_l31",   dict(num_leaves=31, min_data_in_leaf=500, num_iterations=400, seed=31)),
            ("lg_ff30",  dict(feature_fraction=0.3, num_leaves=31, num_iterations=600, seed=23)),
        ]:
            P.append((nm, "lgbm", sp))

    # --- A5 레짐 멤버 (season_centering arm 재사용) --------------------------
    if group in ("all", "regime"):
        for nm in ("all_raw", "all_raw_d70", "ft_all2last"):
            P.append((nm, "arm", dict(arm=nm)))

    # --- A7 **표현 다양성** (A1 실패에서 배운 방향) -------------------------
    # A1 실측: XGB/CatBoost는 s가 0.016~0.033인데 이득 ≈ 0. 같은 피처·같은 정칙화 체제에서
    # 알고리즘만 바꾸면 거의 같은 곳에 도달한다. 반면 `all_raw`(다른 시즌)와 `nn_lin`(원핫 선형)은
    # 단독이 나쁜데도 크게 기여했다 ⇒ **값을 하는 s는 "다른 정보 시각"에서 나온다.**
    # 트리는 상호작용 기계이므로, 그 반대편인 **가법(additive) 모델**이 구조적으로 가장 먼 시각이다.
    # 분위수 구간화 + 원핫 + Ridge = 상호작용이 **구조적으로 0인** GAM.
    if group in ("all", "repr"):
        for nb in (8, 16, 32):
            P.append((f"gam_b{nb}", "gam", dict(bins=nb, alpha=10.0)))
        P.append(("gam_b16_a300", "gam", dict(bins=16, alpha=300.0)))   # 강수축본

    # --- A6 ExtraTrees / RF 확장 (zip 10GB이므로 크기는 제약이 아니다) -------
    if group in ("all", "trees"):
        for nm, sp in [
            ("et_l400_f05", dict(n_estimators=200, min_samples_leaf=400, max_depth=20,
                                 max_features=0.5)),
            ("et_l50_d32",  dict(n_estimators=200, min_samples_leaf=50, max_depth=32,
                                 max_features=0.7)),
            ("et_l800_d24", dict(n_estimators=200, min_samples_leaf=800, max_depth=24,
                                 max_features=1.0)),
        ]:
            P.append((nm, "et", sp))
        P.append(("rf_l500_d12", "rf", dict(n_estimators=150, min_samples_leaf=500, max_depth=12)))
    return P


# ---------------------------------------------------------------- 학습
def train_one(kind, spec, X, y, fit, val, season, seasons, val_season):
    Xf, yf = X[fit], y[fit]
    if kind == "lgbm":
        m = L.train_lgbm(Xf, yf, {**spec, "num_threads": THREADS})
        return m.predict(X[val], num_threads=THREADS)

    if kind == "lgbm_sub":
        rng = np.random.default_rng(spec["seed"])
        k = max(8, int(round(spec["frac"] * X.shape[1])))
        cols = sorted(rng.choice(X.shape[1], size=k, replace=False).tolist())
        sp = {q: v for q, v in spec.items() if q not in ("frac",)}
        m = L.train_lgbm(Xf.iloc[:, cols], yf, {**sp, "num_threads": THREADS})
        return m.predict(X[val].iloc[:, cols], num_threads=THREADS)

    if kind == "lgbm_bag":
        rng = np.random.default_rng(spec["seed"])
        idx = rng.integers(0, len(yf), size=len(yf))
        m = L.train_lgbm(Xf.iloc[idx], yf[idx], {**spec, "num_threads": THREADS})
        return m.predict(X[val], num_threads=THREADS)

    if kind == "xgb":
        import xgboost as xgb
        m = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist",
                             n_jobs=THREADS, random_state=7, **spec).fit(Xf, yf)
        return m.predict(X[val])

    if kind == "cat":
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(loss_function="RMSE", thread_count=THREADS, random_seed=7,
                              verbose=False, allow_writing_files=False, **spec)
        m.fit(np.nan_to_num(Xf.to_numpy(dtype=np.float32), nan=NAN), yf)
        return m.predict(np.nan_to_num(X[val].to_numpy(dtype=np.float32), nan=NAN))

    if kind == "gam":
        # 분위수 구간화 → 원핫 → Ridge. **상호작용이 구조적으로 0**인 가법 모델이므로
        # 상호작용 기계인 트리와 오차 구조가 가장 멀다. 결측은 전용 구간으로 흡수한다.
        from sklearn.preprocessing import KBinsDiscretizer
        from sklearn.linear_model import Ridge
        from scipy import sparse
        med = Xf.median()
        # encode="onehot" 은 희소 행렬을 직접 준다 — 조밀본은 bins=32에서 수 GB가 된다.
        kb = KBinsDiscretizer(n_bins=spec["bins"], encode="onehot",
                              strategy="quantile", subsample=200_000, random_state=0)
        # 결측 지시자를 별도 컬럼으로 보존(트리의 결측 분기에 해당하는 정보)
        miss_f = Xf.isna().to_numpy(dtype=np.float32)
        miss_v = X[val].isna().to_numpy(dtype=np.float32)
        keep = miss_f.any(axis=0)
        Bf = sparse.hstack([kb.fit_transform(Xf.fillna(med)),
                            sparse.csr_matrix(miss_f[:, keep])], format="csr")
        Bv = sparse.hstack([kb.transform(X[val].fillna(med)),
                            sparse.csr_matrix(miss_v[:, keep])], format="csr")
        m = Ridge(alpha=spec["alpha"], solver="sparse_cg").fit(Bf, yf)
        return m.predict(Bv)

    if kind in ("et", "rf"):
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
        C = ExtraTreesRegressor if kind == "et" else RandomForestRegressor
        Xn = np.nan_to_num(Xf.to_numpy(dtype=np.float32), nan=NAN)
        m = C(n_jobs=THREADS, random_state=9, **spec).fit(Xn, yf)
        return m.predict(np.nan_to_num(X[val].to_numpy(dtype=np.float32), nan=NAN))

    if kind == "arm":
        arms = {a["name"]: a for a in SC.arms_for(val_season, seasons)}
        p, _ = SC.run_arm(arms[spec["arm"]], X, y, season, val)
        return p
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=int, default=2024)
    ap.add_argument("--only", default="all", help="all|algo|sub|bag|mix|regime|trees|repr")
    ap.add_argument("--inseason", action="store_true",
                    help="57피처(+is4) 기준선 = 실제 제출본 ENS-3(LB 953.06) 위에서 측정. "
                         "53피처 결과는 멤버 구성이 달라 그대로 옮길 수 없다")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(">> train.csv 로드")
    df = rd.load_train()
    if args.inseason:
        import inseason as IS
        lut = IS.career_end_lookup(df)
        nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy())
        X = L.build_features(df, base=(nb, kb))
        print(f"   57피처 기준선 (+is4) — val2024 ENS-3 기대 859.12")
    else:
        X = L.build_features(df)
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()
    seasons = sorted(int(s) for s in np.unique(season))
    fit = (season == args.val - 1)
    val = (season == args.val)
    yv = y[val]
    print(f"   fit {fit.sum():,} → val {val.sum():,}  r={yv.mean():.4f}  (threads={THREADS})")

    # ---- 기준 ENS-2 (정합성 앵커: val 2024에서 773.48이 나와야 한다)
    print(">> 기준 ENS-2 재구성")
    P5 = EC.train_members(X, y, fit, val)
    ens = sum(P5[n] * w for n, _, w, _ in EC.MEMBERS)
    ens_best = SC.best_cal(yv, ens)[0]
    print(f"   ENS-2 고정캘리 {L.score(yv, L.calibrate(ens, **EC.ENS2_CAL)):.2f} · "
          f"캘리재적합 {ens_best:.2f}   ← val2024 기대 773.48")

    rows, preds = [], {"_ens2": ens}
    for name, kind, spec in pool(args.only):
        t = time.time()
        try:
            p = np.clip(train_one(kind, spec, X, y, fit, val, season, seasons, args.val), 0, 1)
        except Exception as e:                      # 한 후보의 실패가 풀 전체를 죽이지 않게
            print(f"  {name:14s} FAILED: {type(e).__name__}: {e}")
            continue
        preds[name] = p
        s = float(np.sqrt(((p - ens) ** 2).mean()))
        gains = {w: SC.best_cal(yv, (1 - w) * ens + w * p)[0] - ens_best for w in WS}
        bw = max(gains, key=gains.get)
        rec = dict(name=name, kind=kind, solo=L.score(yv, L.calibrate(p)),
                   s=s, best_w=bw, gain=gains[bw], std=float(p.std()),
                   sec=round(time.time() - t, 1),
                   **{f"g{int(w*100):02d}": gains[w] for w in WS})
        rows.append(rec)
        print(f"  {name:14s} [{kind:8s}] 단독 {rec['solo']:8.2f}  "
              f"**s={s:.4f}**  최적w {bw:.2f} → {rec['gain']:+7.2f}   [{rec['sec']}s]")

    t = pd.DataFrame(rows).sort_values("s", ascending=False)
    print("\n" + "=" * 108)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n기준 ENS-2 캘리재적합 = {ens_best:.2f}")
    print("\n[s 상위 — 항등식상 이득은 s²에 비례한다]")
    print(t.head(8)[["name", "kind", "solo", "s", "best_w", "gain"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    np.savez_compressed(OUT_DIR / f"preds_val{args.val}_{args.only}.npz", y=yv, **preds)
    (OUT_DIR / f"pool_val{args.val}_{args.only}.json").write_text(
        json.dumps({"ens_best": ens_best, "rows": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
