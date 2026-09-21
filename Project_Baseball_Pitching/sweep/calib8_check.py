# -*- coding: utf-8 -*-
"""ENS-8(패키지) 구성의 캘리 재적합 확인 — V24에서 grid_calib 최적이 (1.04,−0.01)에서 움직였는가.

    python sweep/calib8_check.py
"""
from __future__ import annotations
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
from nn_carrier import BIS, train_lin   # noqa: E402
from eb_carrier import get_members, deploy_score, ENS7_W, TM_W, eb_block  # noqa: E402
from interaction_carrier import build_products   # noqa: E402
from ens4_weights import grid_calib   # noqa: E402

P_SUCC = [("asof_pitcher_success_rate", "succ")]


def main():
    t0 = time.time()
    print(">> 준비 (K_IS=100)")
    L.K_IS = 100.0
    df = rd.load_train()
    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy())
    X = L.build_features(df, base=(nb, kb)).reset_index(drop=True)
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()
    for tag_, ent_, ncol_, rc_, K_ in IF.BLOCKS:
        if tag_ == "b":
            lb = IF.end_lookup(df, ent_, ncol_, rc_)
            B_bis = IF.build_block(df, tag_, ent_, ncol_, rc_, K_, lb)[BIS].reset_index(drop=True)
    Bps, _, _ = eb_block(df, "pitcher_id", "asof_pitcher_n", P_SUCC, 100, "ebps")
    PR = build_products(df, X, B_bis)
    CXP = [c for c in PR.columns if c.startswith("cxp_")]
    XA = pd.concat([X, B_bis, Bps, PR[CXP]], axis=1)

    for v in (2024, 2023):
        fit, val = (season == v - 1), (season == v)
        P, corr, yv = get_members(df, X, y, season, v)
        others = sum(P[m] * w for m, w in ENS7_W.items() if m != "nn_lin_bis")
        w_lin = ENS7_W["nn_lin_bis"]
        p_lin = train_lin(XA, y, fit, val)
        ens8 = others + w_lin * p_lin + TM_W * corr
        s_frozen = deploy_score(yv, ens8)
        (sl, sh), s_best = grid_calib(yv, ens8)
        s7 = deploy_score(yv, others + w_lin * P["nn_lin_bis"] + TM_W * corr)
        print(f"[V{v}] ENS-7 동결 {s7:.2f} → ENS-8 동결(1.04,-0.01) {s_frozen:.2f} "
              f"(Δ {s_frozen-s7:+.2f}) · grid 최적 {s_best:.2f} @ ({sl:.2f},{sh:.4f}) "
              f"(동결 대비 +{s_best-s_frozen:.2f})")
    print(f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
