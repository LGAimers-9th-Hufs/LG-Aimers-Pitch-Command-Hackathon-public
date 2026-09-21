# -*- coding: utf-8 -*-
"""Phase 2 후속 — 다시즌 멤버가 **ENS-2 위에서도** 이득인지 검사한다.

    python sweep/phase2_ens_check.py --vals 2024,2023

## 왜 필요한가

`season_centering.py --twofold`의 +18.7/+40.1은 **june 2종(base_1s) 대비**다. 우리 챔피언은
ENS-2(5멤버, 로컬 773.48)이고 거기엔 이미 ExtraTrees 2종·선형 1종의 다양성이 들어 있다.
새 멤버의 이득이 기존 다양성과 겹치면 0이 된다. **채택 판단은 반드시 ENS-2 대비로 해야 한다.**

ENS-2 구성은 `submission/build_lgbm_ensemble.py`의 `MEMBERS_FULL`을 그대로 옮겨 쓴다
(june_l15 .30 / nn_lin .275 / et_l100_d28 .225 / et_l200_d20 .175 / june_l7 .025, 캘리 1.02/−0.0075).

두 폴드(val 2024 = 안정 레짐 / val 2023 = 파단 레짐)에서 **고정 w**로 재고, 최악 폴드 이득으로 판정한다.
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

OUT_DIR = HERE.parent / "results" / "phase2"
ET_NAN = -999.0

MEMBERS = [
    ("june_l15",    "lgbm",   0.300, dict(num_leaves=15, min_data_in_leaf=1000,
                                          num_iterations=400, seed=57)),
    ("nn_lin",      "linear", 0.275, dict(hidden=(), dropout=0.0, epochs=60,
                                          lr=1e-3, wd=1e-4, seed=0)),
    ("et_l100_d28", "et",     0.225, dict(n_estimators=200, min_samples_leaf=100,
                                          max_depth=28, max_features=0.7)),
    ("et_l200_d20", "et",     0.175, dict(n_estimators=200, min_samples_leaf=200,
                                          max_depth=20, max_features=1.0)),
    ("june_l7",     "lgbm",   0.025, dict(num_leaves=7, min_data_in_leaf=1500,
                                          num_iterations=500, seed=49)),
]
ENS2_CAL = dict(slope=1.02, shift=-0.0075)
# 새 멤버 후보 — season_centering의 arm 정의를 재사용한다.
NEW = ["ft_all2last", "all_raw", "all_raw_d70"]
WS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)


def train_members(X, y, fit, val):
    from sklearn.ensemble import ExtraTreesRegressor
    Xf, yf = X[fit], y[fit]
    Xn = np.nan_to_num(Xf.to_numpy(dtype=np.float32), nan=ET_NAN)
    Xvn = np.nan_to_num(X[val].to_numpy(dtype=np.float32), nan=ET_NAN)
    out = {}
    for name, kind, _, spec in MEMBERS:
        t = time.time()
        if kind == "lgbm":
            m = L.train_lgbm(Xf, yf, spec)
            p = m.predict(X[val], num_threads=4)
        elif kind == "et":
            m = ExtraTreesRegressor(n_jobs=-1, random_state=9, **spec).fit(Xn, yf)
            p = m.predict(Xvn)
        else:
            import nn_member as NM
            st, oh = NM.prep_fit(Xf), NM.onehot_fit(Xf)
            Z = np.concatenate([NM.prep_apply(Xf, st), NM.onehot_apply(Xf, oh)], axis=1)
            model, dev = NM.train_mlp(Z, yf, **spec)
            Zv = np.concatenate([NM.prep_apply(X[val], st), NM.onehot_apply(X[val], oh)], axis=1)
            p = NM.predict_mlp(model, dev, Zv)
        out[name] = np.clip(p, 0, 1)
        print(f"    {name:14s} {time.time()-t:5.1f}s  단독 cal={L.score(y[val], L.calibrate(out[name])):8.2f}")
    return out


def run(df, val_season):
    X = L.build_features(df)
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()
    seasons = sorted(int(s) for s in np.unique(season))
    fit = (season == val_season - 1)
    val = (season == val_season)
    yv = y[val]
    print(f"\n[val {val_season}] fit {fit.sum():,} → val {val.sum():,}  r={yv.mean():.4f}")

    print("  ENS-2 멤버 학습")
    P = train_members(X, y, fit, val)
    ens = sum(P[n] * w for n, _, w, _ in MEMBERS)
    ens_fixed = L.score(yv, L.calibrate(ens, **ENS2_CAL))
    ens_best = SC.best_cal(yv, ens)[0]
    print(f"  ENS-2  고정캘리(1.02/−0.0075) {ens_fixed:8.2f}   캘리재적합 {ens_best:8.2f}")

    print("  새 멤버 후보 학습")
    arms = {a["name"]: a for a in SC.arms_for(val_season, seasons)}
    rows = []
    for nm in NEW:
        t = time.time()
        p, nfit = SC.run_arm(arms[nm], X, y, season, val)
        rms = float(np.sqrt(((p - ens) ** 2).mean()))
        print(f"    {nm:14s} {time.time()-t:5.1f}s  n={nfit:,}  "
              f"단독 cal={L.score(yv, L.calibrate(p)):8.2f}  rms_vs_ENS2={rms:.4f}")
        for w in WS:
            b = (1 - w) * ens + w * p
            rows.append(dict(val=val_season, arm=nm, w=w, rms=rms,
                             g_best=SC.best_cal(yv, b)[0] - ens_best,
                             g_fixed=L.score(yv, L.calibrate(b, **ENS2_CAL)) - ens_fixed))
    return rows, dict(val=val_season, ens_fixed=ens_fixed, ens_best=ens_best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024,2023")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(">> train.csv 로드")
    df = rd.load_train()
    rows, refs = [], []
    for v in [int(s) for s in args.vals.split(",")]:
        r, ref = run(df, v)
        rows += r
        refs.append(ref)
    t = pd.DataFrame(rows)
    piv = t.pivot_table(index=["arm", "w"], columns="val", values="g_best")
    piv["worst"] = piv.min(axis=1)
    print("\n[ENS-2 + 새 멤버] 캘리 재적합 기준 이득 (고정 w, 폴드별)")
    print(piv.sort_values("worst", ascending=False).to_string(float_format=lambda v: f"{v:.2f}"))
    print("\n기준: " + " · ".join(f"val{r['val']} ENS-2 재적합 {r['ens_best']:.2f}"
                                  f"(고정 {r['ens_fixed']:.2f})" for r in refs))
    (OUT_DIR / "ens_check.json").write_text(
        json.dumps({"rows": rows, "refs": refs}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
