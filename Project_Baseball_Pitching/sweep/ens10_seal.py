# -*- coding: utf-8 -*-
"""ENS-10 Run C — CA_matrix 미측정 셀 봉인 실측 (전부 ENS-9 동결 기준).

    python sweep/ens10_seal.py --vals 2024,2023,2022,2021

셀 (CA_matrix §2 — 채택 기대 낮음, 빈칸 봉인 목적. TM6 타 form은 D-44에서 이미 봉인돼 제외):
1. **ism 구표현 × nn_lin** (§2-1, 매트릭스 최대 빈칸): 패키지 선형(nn_lin_pkg 78컬럼)에
   IF.BLOCKS "m" 블록(0.5-prior 구표현)을 추가해 멤버 교체 — ENS-9 직측.
2. **im 단독 × nn_lin** (§2-6): isp_mid_sm/isp_mid_d 2컬럼만 추가.
3. **E1 × 보정기** (§2-2): ps_delta·ps_sm ⊗ (1, count/11, hand) 잔차 보정기(rCpm 타깃),
   위약 = entity_consts shuffle_seed 경로 교환.
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
import entity_consts as EC        # noqa: E402
from nn_carrier import BIS, train_lin   # noqa: E402
from eb_carrier import (get_members9, ens9_pred, deploy_score,   # noqa: E402
                        ENS9_W, TM_W, W_PM, eb_block)
from interaction_carrier import build_products   # noqa: E402
from is_corrector import ridge_corr   # noqa: E402
from tm_physmix import build_group_profile, row_z   # noqa: E402

OUT_DIR = HERE.parent / "results" / "ens10"
P_SUCC = [("asof_pitcher_success_rate", "succ")]
LAM = 1000.0
WGRID = (0.25, 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024,2023,2022,2021")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(">> 준비 (K_IS=100)")
    L.K_IS = 100.0
    df = rd.load_train()
    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy())
    X = L.build_features(df, base=(nb, kb)).reset_index(drop=True)
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()

    print(">> 블록: 패키지(bis/pEB/cxp) + mix 구표현 + im2 + E1")
    blocks = {}
    for tag_, ent_, ncol_, rc_, K_ in IF.BLOCKS:
        lb = IF.end_lookup(df, ent_, ncol_, rc_)
        blocks[tag_] = IF.build_block(df, tag_, ent_, ncol_, rc_, K_, lb).reset_index(drop=True)
    B_bis = blocks["b"][BIS]
    Bps, _, _ = eb_block(df, "pitcher_id", "asof_pitcher_n", P_SUCC, 100, "ebps")
    PR = build_products(df, X, B_bis)
    CXP = [c for c in PR.columns if c.startswith("cxp_")]
    XA_pkg = pd.concat([X, B_bis, Bps, PR[CXP]], axis=1)
    MIX = list(blocks["m"].columns)
    B_mix = blocks["m"]
    B_im = blocks["p"][["isp_mid_sm", "isp_mid_d"]]
    print(f"   mix {len(MIX)}컬럼: {MIX}")

    ph = EC.season_history(df, "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate")
    E1r = EC.build_entity_cols(df, ph)
    E1p = EC.build_entity_cols(df, ph, shuffle_seed=13)
    prof_pm = build_group_profile()
    Z_pm, cov_pm = row_z(df, prof_pm)
    cc = (df["balls_before"].to_numpy() * 3 + df["strikes_before"].to_numpy()).astype(int)
    cn = cc / 11.0
    hand = (df["pitcher_hand"].to_numpy() == df["batter_hand"].to_numpy()).astype(float)
    pm_cols = []
    for mi in range(4):
        for ctx in (np.ones(len(df)), cn, hand):
            pm_cols.append(Z_pm[:, mi] * ctx)
    D_pm = np.stack(pm_cols, axis=1)
    prof_tm = TMM.load_profile()

    def e1_design(E1df):
        cols = []
        for c in ("ps_delta", "ps_sm"):
            v = np.nan_to_num(E1df[c].to_numpy(dtype=float), nan=(0.0 if c == "ps_delta" else 0.5))
            if c == "ps_sm":
                v = v - 0.5
            for ctx in (np.ones(len(df)), cn, hand):
                cols.append(v * ctx)
        return np.stack(cols, axis=1)

    D_e1, D_e1p = e1_design(E1r), e1_design(E1p)

    res = []
    for v in [int(s) for s in args.vals.split(",")]:
        t1 = time.time()
        fit, val = (season == v - 1), (season == v)
        P, corr, corr_pm_v, yv = get_members9(v)
        others = sum(P[m] * w for m, w in ENS9_W.items() if m != "nn_lin_pkg")
        w_lin = ENS9_W["nn_lin_pkg"]
        base_rest = others + TM_W * corr + W_PM * corr_pm_v
        s_ref = deploy_score(yv, ens9_pred(P, corr, corr_pm_v))

        # ---- ① nn_lin 스왑 사다리
        swap_arms = [("pkg재학습(파리티)", XA_pkg),
                     ("pkg+mix8", pd.concat([XA_pkg, B_mix], axis=1)),
                     ("pkg+im2", pd.concat([XA_pkg, B_im], axis=1))]
        print(f"\n[val {v}] ENS-9 동결 기준 {s_ref:.2f}")
        for name, XA in swap_arms:
            p_lin = train_lin(XA, y, fit, val)
            d = deploy_score(yv, base_rest + w_lin * p_lin) - s_ref
            res.append(dict(val=v, arm=name, w=np.nan, d=round(float(d), 3)))
            print(f"  {name:18s}  Δ {d:+7.2f}")

        # ---- ② E1 × 보정기 (rCpm 타깃)
        residA = TMM.june_fit_resid(X, y, fit)
        Df_tm, covf_tm = TMM.design(df, X, prof_tm, fit)
        A_tm = Df_tm[fit & covf_tm]
        G = A_tm.T @ A_tm + 1000.0 * np.eye(A_tm.shape[1])
        c_tm = np.linalg.solve(G, A_tm.T @ residA[covf_tm[fit]])
        corr_tm_fit = Df_tm[fit] @ c_tm
        corr_tm_fit[~covf_tm[fit]] = 0.0
        residB = residA - TM_W * corr_tm_fit
        corr_pm_fit = ridge_corr(D_pm, fit, fit, residB, LAM)
        corr_pm_fit[~cov_pm[fit]] = 0.0
        rCpm = residB - W_PM * corr_pm_fit
        ens9 = ens9_pred(P, corr, corr_pm_v)
        for name, Dm in (("E1×보정기", D_e1), ("E1×보정기·위약", D_e1p)):
            corr2 = ridge_corr(Dm, fit, val, rCpm, LAM)
            line = dict(val=v, arm=name)
            for w in WGRID:
                line[f"w{w:g}"] = round(float(
                    deploy_score(yv, ens9 + w * corr2) - s_ref), 3)
            res.append(line)
            print(f"  {name:18s}  " + "  ".join(f"w{w:g}: {line[f'w{w:g}']:+7.2f}" for w in WGRID))
        print(f"  [{time.time()-t1:.0f}s]")

    t = pd.DataFrame(res)
    (OUT_DIR / "seal_gate.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR/'seal_gate.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
