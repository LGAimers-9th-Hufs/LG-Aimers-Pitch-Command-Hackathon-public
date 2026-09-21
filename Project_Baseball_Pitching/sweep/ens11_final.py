# -*- coding: utf-8 -*-
"""ENS-11 결합 dry-run — 3주 프로그램 확보 재료 전체의 정직한 결합 총량 (Day-10 결정점 조기 실행).

    python sweep/ens11_final.py --vals 2024,2023,2022,2021

블록(순차 Gram 직교화, 단일 ridge λ=1000, 공유 절편, rC 타깃 = 설치 증분 전부 제거):
  denom(9) ⊥→ tilt-B2(4) ⊥→ condphys-core4(4) ⊥→ enc10(10)
위약 = denom 행셔플 + tilt/condphys/enc 동일 pid bijection 경로교환, 2시드.
셀: 주셀(전 4블록) · ablation(dn+tilt = ENS-10 재현 대조) · 위약. w ∈ {0.25, 0.5, 0.75}.
판정: 3폴드 평균 ≥ +10 & >SD & 위약 분리 & V21 — 미달 시 크기를 보고하고 다음 결정.
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
import prev_denom as PD           # noqa: E402
import trackman as TM             # noqa: E402
from nn_carrier import BIS, train_lin   # noqa: E402
from eb_carrier import (get_members9, ens9_pred, deploy_score,   # noqa: E402
                        ENS9_W, TM_W, W_PM, eb_block)
from interaction_carrier import build_products   # noqa: E402
from is_corrector import ridge_corr   # noqa: E402
from tm_physmix import build_group_profile, row_z   # noqa: E402
from ens10_pkg import orth, denom_design   # noqa: E402
from tm_intent import tier1_map, build_tilt, build_disp, build_block as ti_block, MIX_RC, K_MIX   # noqa: E402
from tm_condphys import build_condphys, row_block as cp_block   # noqa: E402
import commandnet as CN           # noqa: E402

OUT_DIR = HERE.parent / "results" / "ens11"
P_SUCC = [("asof_pitcher_success_rate", "succ")]
LAM = 1000.0
WGRID = (0.25, 0.5, 0.75)
CP_CORE4 = [0, 1, 2, 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024,2023,2022,2021")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(">> TM 상수 일괄 (프로필·tilt·condphys·인코더 데이터)")
    mmap = tier1_map()
    mmap_inv = {int(p): int(t) for t, p in mmap.items()}
    tm = TM.load_trackman()
    prof_pm = build_group_profile()
    tilt = build_tilt(tm, mmap)
    disp = build_disp(tm, mmap)
    cond = build_condphys(tm, mmap)
    d_tm, ctx_tm, gg_tm, Zt_tm, prefix_tm = CN.tm_ctx_targets(tm)
    ssn_tm = d_tm[rd.SEASON].to_numpy(int)
    pids_cov = sorted(mmap_inv)
    print(f"   [{time.time()-t0:.0f}s]")

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
    XA_bis = pd.concat([X, B_bis], axis=1)
    B_dn = PD.build_block(df)
    _, q_pre, q_cur = eb_block(df, "pitcher_id", "asof_pitcher_pitchmix_n", MIX_RC, K_MIX, "mixq")
    T_real, covT = ti_block(df, prof_pm, tilt, disp, q_pre, q_cur)
    CP_real, covCP = cp_block(df, cond, q_cur)
    Z_pm, cov_pm = row_z(df, prof_pm)
    cc = (df["balls_before"].to_numpy() * 3 + df["strikes_before"].to_numpy()).astype(int)
    cn = cc / 11.0
    hand = (df["pitcher_hand"].to_numpy() == df["batter_hand"].to_numpy()).astype(float)
    pm_cols = []
    for mi in range(4):
        for ctx_ in (np.ones(len(df)), cn, hand):
            pm_cols.append(Z_pm[:, mi] * ctx_)
    D_pm = np.stack(pm_cols, axis=1)
    prof_tm6 = TMM.load_profile()

    # 위약 블록 (pid bijection 동일 적용) 2시드
    pl_blocks = []
    for sd_ in range(2):
        rng = np.random.default_rng(71 + sd_)
        perm = rng.permutation(len(pids_cov))
        pmap = {pids_cov[i]: pids_cov[perm[i]] for i in range(len(pids_cov))}
        T_p, _ = ti_block(df, prof_pm, tilt, disp, q_pre, q_cur, pmap=pmap)
        CP_p, _ = cp_block(df, cond, q_cur, pmap=pmap)
        pl_blocks.append((pmap, T_p, CP_p))

    rng_perm = np.random.default_rng(23)
    res = []
    for v in [int(s) for s in args.vals.split(",")]:
        t1 = time.time()
        fit, val = (season == v - 1), (season == v)
        P, corr, corr_pm_v, yv = get_members9(v)
        ens9 = ens9_pred(P, corr, corr_pm_v)
        s_ref = deploy_score(yv, ens9)

        # rC 타깃 (설치 증분 직교화 — ens10_final과 동일)
        residA = TMM.june_fit_resid(X, y, fit)
        Df_tm, covf_tm = TMM.design(df, X, prof_tm6, fit)
        A_tm = Df_tm[fit & covf_tm]
        G = A_tm.T @ A_tm + 1000.0 * np.eye(A_tm.shape[1])
        c_tm = np.linalg.solve(G, A_tm.T @ residA[covf_tm[fit]])
        corr_tm_fit = Df_tm[fit] @ c_tm
        corr_tm_fit[~covf_tm[fit]] = 0.0
        residB = residA - TM_W * corr_tm_fit
        corr_pm_fit = ridge_corr(D_pm, fit, fit, residB, LAM)
        corr_pm_fit[~cov_pm[fit]] = 0.0
        residC0 = residB - W_PM * corr_pm_fit
        p_pkg_fit = train_lin(XA_pkg, y, fit, fit)
        p_bis_fit = train_lin(XA_bis, y, fit, fit)
        rC = residC0 - ENS9_W["nn_lin_pkg"] * (p_pkg_fit - p_bis_fit)

        # 폴드 인코더 (TM < v)
        tr_tm = ssn_tm < v
        ents_f = np.sort(d_tm.loc[tr_tm, "pitcher_trackman_id"].unique())
        eindex = {e: i for i, e in enumerate(ents_f)}
        eidx_f = d_tm.loc[tr_tm, "pitcher_trackman_id"].map(eindex).to_numpy(int)
        enc = CN.train_encoder(ctx_tm[tr_tm], eidx_f, gg_tm[tr_tm], Zt_tm[tr_tm], len(ents_f))
        E_real, covE = CN.encoder_features(enc, eindex, df, mmap_inv, prefix_tm)
        print(f"\n[val {v}] ENS-9 동결 기준 {s_ref:.2f} · 준비 [{time.time()-t1:.0f}s]")

        D_dn = denom_design(df, B_dn, fit)

        def stack(D1, T_b, CP_b, E_b):
            b2 = orth(D1, T_b[:, 4:8], fit)
            D12 = np.concatenate([D1, b2], axis=1)
            b3 = orth(D12, CP_b[:, CP_CORE4], fit)
            D123 = np.concatenate([D12, b3], axis=1)
            b4 = orth(D123, E_b[:, 0:10].astype(np.float64), fit)
            return np.concatenate([D123, b4], axis=1)

        D_main = stack(D_dn, T_real, CP_real, E_real)
        b2a = orth(D_dn, T_real[:, 4:8], fit)
        D_abl = np.concatenate([D_dn, b2a], axis=1)

        arms = [("주셀(dn+tilt+cp+enc)", D_main), ("ablation(dn+tilt)", D_abl)]
        for sd_, (pmap, T_p, CP_p) in enumerate(pl_blocks):
            perm = rng_perm.permutation(len(df))
            D_dnp = denom_design(df, B_dn, fit, perm=perm)
            E_p, _ = CN.encoder_features(enc, eindex, df, mmap_inv, prefix_tm, pl_map=pmap)
            arms.append((f"위약s{sd_}", stack(D_dnp, T_p, CP_p, E_p)))

        for name, Dm in arms:
            corrC = ridge_corr(Dm, fit, val, rC, LAM)
            line = dict(val=v, arm=name)
            for w in WGRID:
                line[f"w{w:g}"] = round(float(
                    deploy_score(yv, ens9 + w * corrC) - s_ref), 3)
            res.append(line)
            print(f"  {name:22s}  " + "  ".join(f"w{w:g}: {line[f'w{w:g}']:+7.2f}" for w in WGRID))

    t = pd.DataFrame(res)
    core = t[t.arm.str.startswith("주셀") & t.val.isin([2024, 2023, 2022])]
    if len(core) >= 3:
        for w in WGRID:
            d = core[f"w{w:g}"]
            print(f"\n[주셀 w{w:g}] 3폴드 평균 {d.mean():+.2f} · SD {d.std():.2f} · "
                  f"채택선(+10) {'통과' if d.mean() >= 10 and d.mean() > d.std() else '미달'}")
        v21 = t[t.arm.str.startswith("주셀") & (t.val == 2021)]
        if len(v21):
            print(f"[감사 V21] w0.5 {v21['w0.5'].iloc[0]:+.2f}")
    (OUT_DIR / "final_gate.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR/'final_gate.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
