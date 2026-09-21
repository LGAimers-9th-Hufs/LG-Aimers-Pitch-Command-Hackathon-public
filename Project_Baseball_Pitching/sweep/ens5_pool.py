# -*- coding: utf-8 -*-
"""ENS-5 후보 풀 — **ENS-4(LB 984.08) 기준선**에서 멤버를 넓히고 그리디로 다시 고른다.

    python sweep/ens5_pool.py --k 100 --vals 2024,2023

## 왜 이 방향인가

ENS-4의 이득(+31.02 LB)은 대부분 **가중 재최적화와 멤버 교체**에서 나왔다. 그리고 그리디가
가장 큰 가중을 준 멤버가 **`nn_lin` (.2475 → .3750)** 이다. 근거가 겹친다:

- is4가 선형 멤버를 가장 크게 올렸다 (644 → 805)
- 같은 입력에서 **깊은 NN은 143, 선형은 805** — 이 데이터의 신호는 거의 가법적이다(B1, D-30)
- 지금까지 값을 한 다양성은 전부 **"다른 정보 시각"** 이었다(`all_raw` = 다른 시즌, `nn_lin` = 다른 표현)

⇒ 두 축으로 넓힌다:
1. **선형 계열 확장** — 시드 변형은 무의미하다(시드 노이즈 0.0004). **피처 부분집합을 달리한
   선형 멤버**라야 진짜 다양성이 생긴다.
2. **전 시즌 학습 계열 확장** — `allraw_l15`가 .125를 받았다. 감쇠·용량 변형을 더 준다.

⚠ Program A/B(XGB·CatBoost·GAM·깊은 NN)는 53피처 기준선에서 전부 ≈0이었다(D-30).
멤버 구조가 바뀌었으니 **싼 것 몇 개만** 다시 넣어 확인하되 기대는 낮게 잡는다.

## 판정

`ens4_weights.py`와 같은 절차: val 2024에서 그리디 → **val 2023에 그대로 적용해 낙관 검사**.
두 폴드 모두 양수여야 채택한다. 투영은 **`LB = 0.6995·로컬 + 362.2` (잔차 ±12)** —
작은 증분에 점추정을 붙이지 않는다(D-32).
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from collections import Counter
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
import inseason as IS             # noqa: E402
import nn_member as NM            # noqa: E402
import season_centering as SC     # noqa: E402
from ens4_weights import affine_score, grid_calib   # noqa: E402

OUT_DIR = HERE.parent / "results" / "ens5"
NAN = -999.0
THREADS = 6

# ENS-4 동결 구성 (LB 984.08) — 기준선이자 그리디 초기값
ENS4_W = {"nn_lin": 0.375, "et_l100_d28": 0.300, "allraw_l15": 0.125,
          "june_l15": 0.100, "june_l7": 0.100}
ENS4_CAL = dict(center=0.5, slope=1.04, shift=-0.01)


def train_linear_sub(X, y, fit, val, frac, seed, wd=1e-4):
    """**피처 부분집합 선형 멤버.** 선형 계열은 시드 노이즈가 0.0004라 시드로는 다양성이 안 생긴다 —
    보는 피처를 영구히 다르게 해야 한다."""
    rng = np.random.default_rng(seed)
    k = max(12, int(round(frac * X.shape[1])))
    cols = sorted(rng.choice(X.shape[1], size=k, replace=False).tolist())
    Xf, Xv = X.iloc[:, cols][fit], X.iloc[:, cols][val]
    st, oh = NM.prep_fit(Xf), NM.onehot_fit(Xf)
    Z = np.concatenate([NM.prep_apply(Xf, st), NM.onehot_apply(Xf, oh)], axis=1)
    model, dev = NM.train_mlp(Z, y[fit], hidden=(), dropout=0.0, epochs=60,
                              lr=1e-3, wd=wd, seed=0)
    Zv = np.concatenate([NM.prep_apply(Xv, st), NM.onehot_apply(Xv, oh)], axis=1)
    return np.clip(NM.predict_mlp(model, dev, Zv), 0, 1)


def build_pool(X, y, season, v, seasons):
    from sklearn.ensemble import ExtraTreesRegressor
    fit, val = (season == v - 1), (season == v)
    yv = y[val]
    P = {}

    def add(name, p, t0):
        P[name] = np.clip(p, 0, 1)
        print(f"    {name:16s} {time.time()-t0:5.1f}s  단독 cal="
              f"{L.score(yv, L.calibrate(P[name])):8.2f}")

    # --- ENS-4 현행 5멤버 -----------------------------------------------
    import phase2_ens_check as EC
    t = time.time()
    base = EC.train_members(X, y, fit, val)
    for k_, p in base.items():
        P[k_] = np.clip(p, 0, 1)
    print(f"    [ENS-4 시즌멤버 5종] {time.time()-t:.0f}s")

    arms = {a["name"]: a for a in SC.arms_for(v, seasons)}
    fit_all = np.isin(season, arms["all_raw"]["fit"])

    # --- 축1: 선형 계열 확장 (피처 부분집합) ------------------------------
    for i, (frac, wd) in enumerate([(0.6, 1e-4), (0.6, 1e-3), (0.75, 1e-4), (0.85, 1e-3)]):
        t = time.time()
        add(f"lin_f{int(frac*100)}_{i}", train_linear_sub(X, y, fit, val, frac, 300 + i, wd), t)

    # --- 축2: 전 시즌 학습 계열 확장 -------------------------------------
    for nm, spec in (("allraw_l15", L.ORIGINAL["l15"]), ("allraw_l7", L.ORIGINAL["l7"]),
                     ("allraw_l31", dict(num_leaves=31, min_data_in_leaf=500,
                                         num_iterations=400, seed=77))):
        t = time.time()
        m = L.train_lgbm(X[fit_all], y[fit_all], spec)
        add(nm, m.predict(X[val], num_threads=THREADS), t)
    t = time.time()
    add("allraw_lin", train_linear_sub(X, y, fit_all, val, 1.0, 0), t)   # 전 시즌 × 선형
    for nm in ("all_raw_d70", "ft_all2last"):
        t = time.time()
        p, _ = SC.run_arm(arms[nm], X, y, season, val)
        add(nm, p, t)

    # --- 축3: 트리 다양성 (그리디가 고르게만 둔다) -------------------------
    Xn = np.nan_to_num(X[fit].to_numpy(dtype=np.float32), nan=NAN)
    Xvn = np.nan_to_num(X[val].to_numpy(dtype=np.float32), nan=NAN)
    for nm, kw in (("et_l200_d20", dict(min_samples_leaf=200, max_depth=20, max_features=1.0)),
                   ("et_l400_f05", dict(min_samples_leaf=400, max_depth=20, max_features=0.5)),
                   ("et_l50_d32", dict(min_samples_leaf=50, max_depth=32, max_features=0.7))):
        t = time.time()
        m = ExtraTreesRegressor(n_estimators=200, n_jobs=THREADS, random_state=9, **kw).fit(Xn, y[fit])
        add(nm, m.predict(Xvn), t)
    for nm, spec in (("lg_l31", dict(num_leaves=31, min_data_in_leaf=500,
                                     num_iterations=400, seed=31)),
                     ("lg_bin", dict(objective="binary", metric="binary_logloss",
                                     num_leaves=15, min_data_in_leaf=1000,
                                     num_iterations=500, seed=19))):
        t = time.time()
        m = L.train_lgbm(X[fit], y[fit], spec)
        add(nm, m.predict(X[val], num_threads=THREADS), t)
    return P, yv


def greedy(P, yv, n_iter=40, init=None):
    names = list(P)
    picks = list(init or [])
    cur = sum(P[n] for n in picks) if picks else np.zeros(len(yv))
    for _ in range(n_iter - len(picks)):
        best, bn = -1e18, None
        for nm in names:
            s = affine_score(yv, (cur + P[nm]) / (len(picks) + 1))
            if s > best:
                best, bn = s, nm
        picks.append(bn)
        cur = cur + P[bn]
    return {k: c / len(picks) for k, c in Counter(picks).items()}


def blend(P, w):
    return sum(P[k] * q for k, q in w.items() if k in P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, default=100.0)
    ap.add_argument("--vals", default="2024,2023")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f">> train.csv 로드 (K_IS = {args.k:g})")
    L.K_IS = args.k
    df = rd.load_train()
    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy())
    X = L.build_features(df, base=(nb, kb))
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()
    seasons = sorted(int(s) for s in np.unique(season))
    vals = [int(s) for s in args.vals.split(",")]

    P, YV = {}, {}
    for v in vals:
        print(f"\n[val {v}] 후보 풀 학습")
        P[v], YV[v] = build_pool(X, y, season, v, seasons)
    print(f"\n후보 {len(P[vals[0]])}종")

    main_v = vals[0]
    # 기준선: ENS-4 동결 구성
    print("\n" + "=" * 90)
    rows = []
    W_NEW = greedy(P[main_v], YV[main_v], init=["nn_lin", "et_l100_d28"])
    for nm, w in (("ENS-4 동결", ENS4_W), ("ENS-5 그리디", W_NEW)):
        line = {"cfg": nm}
        for v in vals:
            (sl, sh), s = grid_calib(YV[v], blend(P[v], w))
            line[f"val{v}"], line[f"cal{v}"] = s, f"({sl}, {sh})"
        rows.append(line)
        print(f"  {nm:14s} " + "  ".join(
            f"val{v} {line[f'val{v}']:8.2f} {line[f'cal{v}']}" for v in vals))

    print("\n  [ENS-5 그리디 가중]")
    for k, q in sorted(W_NEW.items(), key=lambda kv: -kv[1]):
        print(f"    {k:16s} {ENS4_W.get(k, 0):.4f} → {q:.4f}")

    d = {v: rows[1][f"val{v}"] - rows[0][f"val{v}"] for v in vals}
    worst = min(d.values())
    print("\n[판정]  " + " · ".join(f"val{v} {d[v]:+7.2f}" for v in vals)
          + f"   최악 {worst:+7.2f}")
    loc = rows[1][f"val{main_v}"]
    print(f"  ENS-5 로컬 {loc:.2f} → 투영 LB = 0.6995·{loc:.2f} + 362.2 = "
          f"{0.6995*loc+362.2:.1f}  (±12, D-32)")
    if worst <= 0:
        print("  ⚠ 한 폴드에서 이득이 사라졌다 = 선택 낙관. 채택하지 말 것.")

    (OUT_DIR / "pool.json").write_text(
        json.dumps({"ens4": ENS4_W, "greedy": W_NEW, "rows": rows},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    for v in vals:
        np.savez_compressed(OUT_DIR / f"preds_val{v}.npz", y=YV[v], **P[v])
    print(f"\n>> 저장 {OUT_DIR}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
