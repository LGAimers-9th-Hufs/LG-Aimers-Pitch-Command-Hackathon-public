# -*- coding: utf-8 -*-
"""M1 — LightGBM `linear_tree` (분할 + 잎 선형모델). PAINE과 같은 사상의 기성 대조군.

    python sweep/linear_tree_arm.py --vals 2024,2023,2022

킬리스트에 "보류(설치 리스크)"로 남아 있었으나 **현 lightgbm 4.6에 내장** — 리스크가 애초에 없었다.
신호가 거의 가법적(선형 805)이고 트리가 +23을 더하는 이 문제에서, "트리 분할 + 잎 선형"은
정확히 그 구조에 맞는 하이브리드다. ⚠ linear_tree는 NaN을 0 대치하므로 명시 중앙값 대치를 건다.
"""
from __future__ import annotations
import argparse
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
import inseason as IS             # noqa: E402
import ens5_pool as EP            # noqa: E402
from season_centering import best_cal   # noqa: E402

CACHE = HERE.parent / "results" / "ens5"
OUT_DIR = HERE.parent / "results" / "lintree"
WS = (0.05, 0.10, 0.15, 0.20, 0.30)

CONFIGS = [
    ("lt_t54",   dict(num_leaves=7, min_data_in_leaf=2500, learning_rate=0.0175,
                      num_iterations=726, lambda_l1=7.2, lambda_l2=0.77,
                      feature_fraction=0.6, bagging_fraction=0.73, bagging_freq=1,
                      linear_lambda=0.1, seed=57)),
    ("lt_l15",   dict(num_leaves=15, min_data_in_leaf=1000, learning_rate=0.03,
                      num_iterations=400, lambda_l1=0.5, lambda_l2=8.0,
                      feature_fraction=0.85, linear_lambda=0.1, seed=57)),
    ("lt_t54_ll1", dict(num_leaves=7, min_data_in_leaf=2500, learning_rate=0.0175,
                        num_iterations=726, lambda_l1=7.2, lambda_l2=0.77,
                        feature_fraction=0.6, bagging_fraction=0.73, bagging_freq=1,
                        linear_lambda=1.0, seed=57)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024,2023,2022")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    import lightgbm as lgb

    L.K_IS = 100.0
    df = rd.load_train()
    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy())
    X = L.build_features(df, base=(nb, kb)).reset_index(drop=True)
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()

    rows = []
    for v in [int(s) for s in args.vals.split(",")]:
        fit, val = (season == v - 1), (season == v)
        yv = y[val]
        z = np.load(CACHE / f"preds_val{v}.npz")
        A = sum(z[m] * w for m, w in EP.ENS4_W.items())
        sA = best_cal(yv, A)[0]
        med = X[fit].median()
        Xf, Xv = X[fit].fillna(med), X[val].fillna(med)     # linear_tree는 NaN을 0 대치 — 명시 대치
        print(f"\n[val {v}]  ENS-4 = {sA:.2f}")
        for name, spec in CONFIGS:
            t = time.time()
            pp = {**L.BASE_PARAMS, **{k: q for k, q in spec.items() if k != "num_iterations"},
                  "linear_tree": True, "num_threads": 6}
            ds = lgb.Dataset(Xf, label=y[fit], free_raw_data=False, params={"linear_tree": True})
            m = lgb.train(pp, ds, num_boost_round=spec["num_iterations"])
            p = np.clip(m.predict(Xv, num_threads=6), 1e-6, 1 - 1e-6)
            s = float(np.sqrt(((p - A) ** 2).mean()))
            gains = {w: best_cal(yv, (1 - w) * A + w * p)[0] - sA for w in WS}
            bw = max(gains, key=gains.get)
            rows.append(dict(val=v, arm=name, solo=best_cal(yv, p)[0], s=s,
                             best_w=bw, gain=gains[bw], sec=round(time.time() - t)))
            print(f"    {name:12s} 단독 {rows[-1]['solo']:8.2f}  s={s:.4f}  "
                  f"w{bw:.2f} → {gains[bw]:+.2f}  [{rows[-1]['sec']}s]")

    t = pd.DataFrame(rows)
    print("\n" + "=" * 84)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    g = t.groupby("arm").agg(solo=("solo", "mean"), s=("s", "mean"),
                             gain_mean=("gain", "mean"), gain_min=("gain", "min"))
    print("\n[요약 — 대조선: t54 828 · b_lgb82_a 830 · ENS-4 877.8]")
    print(g.sort_values("solo", ascending=False).to_string(float_format=lambda x: f"{x:.3f}"))
    (OUT_DIR / "gate.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    print(f">> 저장 {OUT_DIR}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
