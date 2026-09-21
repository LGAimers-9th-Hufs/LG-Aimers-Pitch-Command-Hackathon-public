# -*- coding: utf-8 -*-
"""ENS-7 조립 — nn_lin_bis(+15.9) + 가중 재최적화 + 트랙맨 보정기(+3.4) 결합 측정.

    python sweep/ens7_assemble.py --vals 2024,2023,2022

구성 후보를 사다리로 잰다 (전부 ENS 수준, V22는 선택 미사용):
  A0. ENS-4 동결 (기준)
  A1. nn_lin → nn_lin_bis 교체, 가중 동결
  A2. A1 + maximin 그리디 재가중 (선택 = 24·23)
  A3. A2 + 트랙맨 보정기 w_tm ∈ {0.25, 0.5} (커버 행만)
최종 캘리는 선택 폴드에서 재적합(grid_calib), V22로 확인.
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
import inseason as IS             # noqa: E402
import inseason_full as IF        # noqa: E402
import ens5_pool as EP            # noqa: E402
import nn_member as NM            # noqa: E402
import tm_member as TMM           # noqa: E402
from ens4_weights import affine_score, grid_calib   # noqa: E402
from ens5_select import greedy_multi   # noqa: E402
from nn_carrier import BIS, train_lin   # noqa: E402
from season_centering import best_cal   # noqa: E402

CACHE = HERE.parent / "results" / "ens5"
OUT_DIR = HERE.parent / "results" / "ens7"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024,2023,2022")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(">> 데이터·블록 준비 (K_IS=100)")
    L.K_IS = 100.0
    df = rd.load_train()
    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy())
    X = L.build_features(df, base=(nb, kb)).reset_index(drop=True)
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()
    # bis 블록 (타자 당해시즌)
    for tag, ent, ncol, rc, K in IF.BLOCKS:
        if tag == "b":
            lb = IF.end_lookup(df, ent, ncol, rc)
            B_bis = IF.build_block(df, tag, ent, ncol, rc, K, lb)[BIS].reset_index(drop=True)
    XB = pd.concat([X, B_bis], axis=1)
    prof = TMM.load_profile()

    vals = [int(s) for s in args.vals.split(",")]
    P, YV, CORR = {}, {}, {}
    for v in vals:
        fit, val = (season == v - 1), (season == v)
        yv = y[val]
        z = np.load(CACHE / f"preds_val{v}.npz")
        P[v] = {m: z[m] for m in EP.ENS4_W if m != "nn_lin"}
        t = time.time()
        P[v]["nn_lin_bis"] = train_lin(XB, y, fit, val)
        # 트랙맨 보정기 (커버 밖 0)
        resid = TMM.june_fit_resid(X, y, fit)
        Df, covf = TMM.design(df, X, prof, fit)
        Dv, covv = TMM.design(df, X, prof, val)
        A = Df[fit & covf]
        r = resid[covf[fit]]
        G = A.T @ A + 1000.0 * np.eye(A.shape[1])
        c = np.linalg.solve(G, A.T @ r)
        corr = Dv @ c
        corr[~covv] = 0.0
        CORR[v] = corr[val]
        YV[v] = yv
        print(f"   [val {v}] nn_lin_bis {best_cal(yv, P[v]['nn_lin_bis'])[0]:.2f} · "
              f"tm 커버 {covv[val].mean():.2f}  [{time.time()-t:.0f}s]")

    W4 = dict(EP.ENS4_W)
    W4_bis = {("nn_lin_bis" if k == "nn_lin" else k): w for k, w in W4.items()}

    def blend(v, w):
        return sum(P[v][m] * q for m, q in w.items() if m in P[v])

    sel = [v for v in vals if v in (2024, 2023)]
    ref = {v: affine_score(YV[v], blend(v, W4_bis)) for v in sel}
    W_G = greedy_multi(P, YV, sel, mode="maximin", ref=ref, init=["nn_lin_bis", "et_l100_d28"])

    print("\n[그리디(maximin) 재가중]")
    for k, q in sorted(W_G.items(), key=lambda kv: -kv[1]):
        print(f"    {k:14s} {W4_bis.get(k, 0):.4f} → {q:.4f}")

    rows = []
    # 기준 A0: ENS-4 캐시 (nn_lin 캐시 포함)
    for name, wcfg, wtm in (("A0 ENS-4 동결", None, 0.0),
                            ("A1 bis교체·가중동결", W4_bis, 0.0),
                            ("A2 +그리디재가중", W_G, 0.0),
                            ("A3 +tm w0.25", W_G, 0.25),
                            ("A3 +tm w0.5", W_G, 0.5)):
        line = {"cfg": name}
        for v in vals:
            if wcfg is None:
                z = np.load(CACHE / f"preds_val{v}.npz")
                p = sum(z[m] * w for m, w in EP.ENS4_W.items())
            else:
                p = blend(v, wcfg) + wtm * CORR[v]
            (sl, sh), s = grid_calib(YV[v], p)
            line[f"val{v}"] = s
            line[f"cal{v}"] = (sl, sh)
        rows.append(line)
        print(f"  {name:20s} " + "  ".join(f"V{v} {line[f'val{v}']:8.2f}" for v in vals))

    base = rows[0]
    print("\n[ENS-4 대비]  (V2022 = 선택 미사용)")
    for r_ in rows[1:]:
        d = {v: r_[f"val{v}"] - base[f"val{v}"] for v in vals}
        print(f"  {r_['cfg']:20s} " + " · ".join(f"V{v} {d[v]:+7.2f}" for v in vals))
    bestrow = max(rows[1:], key=lambda r_: min(r_[f"val{v}"] - base[f"val{v}"]
                                               for v in (2024, 2022) if v in vals))
    loc = bestrow["val2024"]
    print(f"\n  최종 후보: {bestrow['cfg']} · 로컬(V24) {loc:.2f} · 캘리 {bestrow['cal2024']}")
    print(f"  투영 LB = 0.6995·{loc:.2f} + 362.2 = {0.6995*loc+362.2:.1f} ± 12")

    (OUT_DIR / "assemble.json").write_text(
        json.dumps({"greedy": W_G, "rows": [{k: (v if not isinstance(v, tuple) else list(v))
                                             for k, v in r_.items()} for r_ in rows]},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    for v in vals:
        np.savez_compressed(OUT_DIR / f"preds_val{v}.npz", y=YV[v], corr=CORR[v],
                            **{m: P[v][m] for m in P[v]})
    print(f"\n>> 저장 {OUT_DIR}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
