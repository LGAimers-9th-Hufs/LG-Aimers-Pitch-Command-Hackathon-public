# -*- coding: utf-8 -*-
"""ENS-9 결합 run — ENS-8(패키지 78컬럼 선형) + physmix 보정기 w0.5 를 한 번에 실측.

    python sweep/ens9_check.py --vals 2024,2023,2022,2021

규칙: 패키지 Δ는 개별 합산이 아니라 **결합 run 실측만 인정**. 기준 = ENS-7 동결.
V24에서 grid_calib 재확인(구성 변경 시 캘리 재적합 원칙).
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
import inseason_full as IF        # noqa: E402
import tm_member as TMM           # noqa: E402
from nn_carrier import BIS, train_lin   # noqa: E402
from eb_carrier import get_members, deploy_score, ENS7_W, TM_W, eb_block  # noqa: E402
from interaction_carrier import build_products   # noqa: E402
from is_corrector import ridge_corr   # noqa: E402
from tm_physmix import build_group_profile, row_z   # noqa: E402
from ens4_weights import grid_calib   # noqa: E402

OUT_DIR = HERE.parent / "results" / "dual"
P_SUCC = [("asof_pitcher_success_rate", "succ")]
W_PM = 0.5
LAM = 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024,2023,2022,2021")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(">> physmix 프로필")
    prof = build_group_profile()
    print(f"   {len(prof):,} 엔트리 [{time.time()-t0:.0f}s]")

    print(">> train 준비 (K_IS=100)")
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
    XA_pkg = pd.concat([X, B_bis, Bps, PR[CXP]], axis=1)

    Z_real, cov = row_z(df, prof)
    cc = (df["balls_before"].to_numpy() * 3 + df["strikes_before"].to_numpy()).astype(int)
    cn = cc / 11.0
    hand = (df["pitcher_hand"].to_numpy() == df["batter_hand"].to_numpy()).astype(float)
    cols = []
    for mi in range(4):
        for ctx in (np.ones(len(df)), cn, hand):
            cols.append(Z_real[:, mi] * ctx)
    D_pm = np.stack(cols, axis=1)
    prof_tm = TMM.load_profile()

    res = []
    for v in [int(s) for s in args.vals.split(",")]:
        fit, val = (season == v - 1), (season == v)
        P, corr, yv = get_members(df, X, y, season, v)
        others = sum(P[m] * w for m, w in ENS7_W.items() if m != "nn_lin_bis")
        w_lin = ENS7_W["nn_lin_bis"]
        ens7 = others + w_lin * P["nn_lin_bis"] + TM_W * corr
        s_ref = deploy_score(yv, ens7)

        residA = TMM.june_fit_resid(X, y, fit)
        Df_tm, covf_tm = TMM.design(df, X, prof_tm, fit)
        A_tm = Df_tm[fit & covf_tm]
        G = A_tm.T @ A_tm + 1000.0 * np.eye(A_tm.shape[1])
        c_tm = np.linalg.solve(G, A_tm.T @ residA[covf_tm[fit]])
        corr_tm_fit = Df_tm[fit] @ c_tm
        corr_tm_fit[~covf_tm[fit]] = 0.0
        residB = residA - TM_W * corr_tm_fit
        corr_pm = ridge_corr(D_pm, fit, val, residB, LAM)
        corr_pm[~cov[val]] = 0.0

        p_pkg = train_lin(XA_pkg, y, fit, val)
        ens8 = others + w_lin * p_pkg + TM_W * corr
        ens9 = ens8 + W_PM * corr_pm
        d8 = deploy_score(yv, ens8) - s_ref
        d7pm = deploy_score(yv, ens7 + W_PM * corr_pm) - s_ref
        d9 = deploy_score(yv, ens9) - s_ref
        line = dict(val=v, d_ens8=round(float(d8), 2), d_physmix_on7=round(float(d7pm), 2),
                    d_ens9=round(float(d9), 2))
        if v == 2024:
            (sl, sh), s_best = grid_calib(yv, ens9)
            line["grid"] = f"({sl:.2f},{sh:.4f}) +{s_best - deploy_score(yv, ens9):.2f}"
        res.append(line)
        print(f"[V{v}] ENS-8 {d8:+7.2f} · physmix(on7) {d7pm:+7.2f} · **ENS-9 {d9:+7.2f}**"
              + (f" · V24 grid {line.get('grid')}" if v == 2024 else ""))

    t = pd.DataFrame(res)
    core = t[t.val.isin([2024, 2023, 2022])]["d_ens9"]
    print(f"\n[ENS-9 결합] 3폴드 평균 {core.mean():+.2f} · SD {core.std():.2f} · "
          f"통과 {core.mean() > 0 and core.mean() > core.std()} · "
          f"V21 {t[t.val == 2021]['d_ens9'].iloc[0] if (t.val == 2021).any() else float('nan'):+.2f}")
    (OUT_DIR / "ens9_check.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    print(f">> 저장 {OUT_DIR/'ens9_check.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
