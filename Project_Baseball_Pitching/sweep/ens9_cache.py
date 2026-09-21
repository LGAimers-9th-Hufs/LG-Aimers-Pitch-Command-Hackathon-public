# -*- coding: utf-8 -*-
"""ENS-9 게이트 기준선 캐시 — `results/ens9/preds_val{v}.npz` 생성 (D-46 지시 이행).

    python sweep/ens9_cache.py --vals 2024,2023,2022,2021

챔피언이 ENS-9(LB 1009.26)로 교체됐는데 게이트 캐시는 ENS-7이라, 이후 후보를 재면
pkg·physmix 이득이 이중 계상된다. 이 스크립트가 ENS-9 구성의 멤버·보정기 예측을 폴드별로
저장한다. 로직은 `ens9_check.py`와 바이트 동일 경로(패키지 78컬럼 선형 + physmix residB 적합).

npz 키: june_l15 / june_l7 / allraw_l15 / et_l100_d28 / corr  (ens7 캐시에서 그대로)
        nn_lin_pkg  (X+bis+pEB+cxp 78컬럼 선형 멤버)
        corr_pm     (physmix 보정기, 비커버 0)
        y

검증: V24 재구성 절대점수가 ens9_check 실측(896.29 + 9.33 = 905.62)과 일치해야 한다.
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

CACHE9 = HERE.parent / "results" / "ens9"
P_SUCC = [("asof_pitcher_success_rate", "succ")]
W_PM = 0.5
LAM = 1000.0
V24_EXPECT = 905.62      # ens9_check.json 실측 (896.29 + 9.33)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024,2023,2022,2021")
    args = ap.parse_args()
    CACHE9.mkdir(parents=True, exist_ok=True)
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

    for v in [int(s) for s in args.vals.split(",")]:
        t1 = time.time()
        fit, val = (season == v - 1), (season == v)
        P, corr, yv = get_members(df, X, y, season, v)
        others = sum(P[m] * w for m, w in ENS7_W.items() if m != "nn_lin_bis")
        w_lin = ENS7_W["nn_lin_bis"]
        s_ref7 = deploy_score(yv, others + w_lin * P["nn_lin_bis"] + TM_W * corr)

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
        ens9 = others + w_lin * p_pkg + TM_W * corr + W_PM * corr_pm
        s9 = deploy_score(yv, ens9)

        np.savez_compressed(
            CACHE9 / f"preds_val{v}.npz", y=yv, corr=corr, corr_pm=corr_pm,
            nn_lin_pkg=p_pkg,
            **{m: P[m] for m in ENS7_W if m != "nn_lin_bis"})
        note = ""
        if v == 2024:
            note = f"  (기대 {V24_EXPECT:.2f}, 차 {s9 - V24_EXPECT:+.2f})"
        print(f"[V{v}] ENS-7 {s_ref7:8.2f} → ENS-9 {s9:8.2f} (Δ {s9-s_ref7:+.2f})"
              f"{note}  [{time.time()-t1:.0f}s]")

    print(f">> 저장 {CACHE9}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
