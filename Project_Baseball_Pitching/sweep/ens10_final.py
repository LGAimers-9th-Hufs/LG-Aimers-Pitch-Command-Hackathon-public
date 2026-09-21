# -*- coding: utf-8 -*-
"""ENS-10 Phase 2 — **최종 결합 run** (사전등록 주셀, 성분 합산 금지·결합 실측만 판정).

    python sweep/ens10_final.py --vals 2024,2023,2022,2021
    python sweep/ens10_final.py --vals 2024,2023,2022,2021 --w75    # 보조셀: physmix w0.75

주셀 = ENS-9 + w·corrC where corrC = 단일 ridge[안1(14) ⊥→ denom(9) ⊥→ tilt-B2(4)], 타깃 rC
(설치 증분 전부 직교화 — ens10_pkg.py와 동일), λ=1000, 공유 절편, w=0.5.
위약 = 안1 is4 경로교환 + denom 행 셔플 + tilt pid 경로교환(커버 보존) 동시 적용 2시드.
채택선: 3폴드(V24/23/22) 평균 ≥ +10 & 평균 > SD & 위약 분리 & V21 붕괴 없음.
보조셀 --w75: 주셀 구성에서 physmix 가중만 0.5→0.75 (Brier 이차식 예상 +0.74, 타이브레이커).
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
from is_corrector import is_factors, build_design, ridge_corr   # noqa: E402
from tm_physmix import build_group_profile, row_z   # noqa: E402
from ens10_pkg import orth, denom_design, DN_M   # noqa: E402
from tm_intent import tier1_map, build_tilt, build_disp, build_block, MIX_RC, K_MIX   # noqa: E402
from ens4_weights import grid_calib   # noqa: E402

OUT_DIR = HERE.parent / "results" / "ens10"
P_SUCC = [("asof_pitcher_success_rate", "succ")]
LAM = 1000.0
W_C = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024,2023,2022,2021")
    ap.add_argument("--w75", action="store_true")
    ap.add_argument("--grid", action="store_true",
                    help="ablation(dn+tilt) 구성에서 (w_C × physmix 추가가중) 격자 실측")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(">> TM 상수 (프로필·tilt)")
    mmap = tier1_map()
    tm = TM.load_trackman()
    prof_pm = build_group_profile()
    tilt = build_tilt(tm, mmap)
    disp = build_disp(tm, mmap)

    print(">> train 준비 (K_IS=100)")
    L.K_IS = 100.0
    df = rd.load_train()
    lut = IS.career_end_lookup(df)
    pid = df["pitcher_id"].to_numpy()
    ssn = df[rd.SEASON].to_numpy()
    nb, kb = IS.base_for(lut, pid, ssn)
    X = L.build_features(df, base=(nb, kb)).reset_index(drop=True)
    X_pl = {sd_: L.build_features(df, base=IS.base_for(lut, pid, ssn, shuffle_seed=7 + sd_))
            .reset_index(drop=True) for sd_ in range(2)}
    y = df[rd.TARGET].to_numpy().astype(float)
    season = ssn

    print(">> 블록")
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
    T_real, covT = build_block(df, prof_pm, tilt, disp, q_pre, q_cur)
    pids_in = sorted({p for (p, _s) in prof_pm})
    T_pl = []
    for sd_ in range(2):
        rng = np.random.default_rng(71 + sd_)
        perm = rng.permutation(len(pids_in))
        pmap = {pids_in[i]: pids_in[perm[i]] for i in range(len(pids_in))}
        T_pl.append(build_block(df, prof_pm, tilt, disp, q_pre, q_cur, pmap=pmap))

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

    rng_perm = np.random.default_rng(23)
    res = []
    for v in [int(s) for s in args.vals.split(",")]:
        t1 = time.time()
        fit, val = (season == v - 1), (season == v)
        P, corr, corr_pm_v, yv = get_members9(v)
        ens9 = ens9_pred(P, corr, corr_pm_v)
        s_ref = deploy_score(yv, ens9)

        # rC 타깃 (ens10_pkg와 동일)
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
        residC0 = residB - W_PM * corr_pm_fit
        p_pkg_fit = train_lin(XA_pkg, y, fit, fit)
        p_bis_fit = train_lin(XA_bis, y, fit, fit)
        rC = residC0 - ENS9_W["nn_lin_pkg"] * (p_pkg_fit - p_bis_fit)
        print(f"\n[val {v}] ENS-9 동결 기준 {s_ref:.2f} · rC 준비 [{time.time()-t1:.0f}s]")

        def combined(D_a1, D_dn, T_blk):
            D2 = orth(D_a1, D_dn, fit)
            D12 = np.concatenate([D_a1, D2], axis=1)
            D3 = orth(D12, T_blk[:, 4:8], fit)
            return np.concatenate([D12, D3], axis=1)

        D_a1_full, n1 = build_design(df, *is_factors(X, fit))
        D_a1 = D_a1_full[:, :n1]
        D_dn = denom_design(df, B_dn, fit)
        D_main = combined(D_a1, D_dn, T_real)

        D3_dn = orth(D_dn, T_real[:, 4:8], fit)
        D_abl = np.concatenate([D_dn, D3_dn], axis=1)

        arms = [("주셀(안1+dn+tilt)", D_main, covT),
                ("ablation(dn+tilt)", D_abl, covT)]
        for sd_ in range(2):
            D_a1p = build_design(df, *is_factors(X_pl[sd_], fit))[0][:, :n1]
            perm = rng_perm.permutation(len(df))
            D_dnp = denom_design(df, B_dn, fit, perm=perm)
            arms.append((f"위약s{sd_}", combined(D_a1p, D_dnp, T_pl[sd_][0]), T_pl[sd_][1]))
            D3p = orth(D_dnp, T_pl[sd_][0][:, 4:8], fit)
            arms.append((f"위약dn+tilt s{sd_}",
                         np.concatenate([D_dnp, D3p], axis=1), T_pl[sd_][1]))

        for name, Dm, cvT in arms:
            corrC = ridge_corr(Dm, fit, val, rC, LAM)
            # tilt 블록 성분은 비커버 행에도 안1/denom 성분이 있으므로 전행 유지 (blk 자체가 비커버 0)
            line = dict(val=v, arm=name)
            ens10 = ens9 + W_C * corrC
            line["d"] = round(float(deploy_score(yv, ens10) - s_ref), 3)
            if args.grid and name.startswith("ablation"):
                for wc in (0.25, 0.5, 0.75, 1.0):
                    for pe in (0.0, 0.25):
                        line[f"wc{wc:g}_pe{pe:g}"] = round(float(
                            deploy_score(yv, ens9 + wc * corrC + pe * corr_pm_v) - s_ref), 3)
            if args.w75 and name.startswith("주셀"):
                ens10_w75 = ens10 + 0.25 * corr_pm_v
                line["d_w75"] = round(float(deploy_score(yv, ens10_w75) - s_ref), 3)
            if v == 2024 and name.startswith("주셀"):
                (sl, sh), s_best = grid_calib(yv, ens10)
                line["grid"] = f"({sl:.2f},{sh:.4f}) +{s_best - deploy_score(yv, ens10):.2f}"
            res.append(line)
            print(f"  {name:18s}  Δ {line['d']:+7.2f}"
                  + (f"  · w75 {line.get('d_w75'):+7.2f}" if "d_w75" in line else "")
                  + (f"  · V24 grid {line.get('grid')}" if "grid" in line else ""))
            if args.grid and name.startswith("ablation"):
                for wc in (0.25, 0.5, 0.75, 1.0):
                    print(f"      w_C={wc:<4g} " + "  ".join(
                        f"pm+{pe:g}: {line[f'wc{wc:g}_pe{pe:g}']:+7.2f}" for pe in (0.0, 0.25)))

    t = pd.DataFrame(res)
    core = t[t.arm.str.startswith("주셀") & t.val.isin([2024, 2023, 2022])]
    if len(core) >= 3:
        d = core["d"]
        print(f"\n[주셀 결합] 3폴드 평균 {d.mean():+.2f} · SD {d.std():.2f} · "
              f"채택선(+10) {'통과' if d.mean() >= 10 and d.mean() > d.std() else '미달'}")
        v21 = t[t.arm.str.startswith("주셀") & (t.val == 2021)]
        if len(v21):
            print(f"[감사 V21] Δ {v21['d'].iloc[0]:+.2f}")
    (OUT_DIR / "final_gate.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR/'final_gate.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
