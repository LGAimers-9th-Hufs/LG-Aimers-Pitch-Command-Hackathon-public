# -*- coding: utf-8 -*-
"""W2-E2 마무리 — **물리→성공 반응 모델**: 정렬 916k 투구쌍의 실물리로 β를 적합하고
서빙은 인코더 예측 물리에 β를 동결 적용한 스칼라 1개로 운반 (3주 프로그램 마지막 카드).

    python sweep/ens11_response.py --vals 2024,2023,2022,2021

## 왜 이게 enc10과 다른가
enc10 보정기는 [E[phys] 10컬럼]을 fit 폴드(25만 행)에서 ridge 재적합 — 계수가 예측물리의
수축·공선성에 오염된다. 반응 모델은 **실물리 916k 투구**에서 β를 적합(비선형 phys²·phys×ctx
포함)하므로 표본 효율과 구조가 다르다. 서빙 형태는 β·[E,E·cn,E·hh,E²] — E² 항은
Var(측정별 상수)가 절편에 흡수되므로 E[phys]²로 합법 계산 가능. 전부 행 산술 + 고정 상수(§5).

셀: 주셀4(D-54 재현) / +resp(스칼라 1) / +enc10ctx30(β 없는 같은 피처 재적합 — 투구수준
적합의 순기여 분리) / 위약 2종(β를 y-셔플로 적합 · pid 경로교환). w ∈ {0.25, 0.5}.
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
from tm_condphys import build_condphys, row_block as cp_block, GROUPS   # noqa: E402
from auxlabel_oracle import recover_events   # noqa: E402
import commandnet as CN           # noqa: E402

OUT_DIR = HERE.parent / "results" / "ens11"
P_SUCC = [("asof_pitcher_success_rate", "succ")]
LAM = 1000.0
LAM_R = 100.0          # 투구 수준 β ridge (표본 916k라 약하게)
WGRID = (0.25, 0.5)
CP_CORE4 = [0, 1, 2, 3]


# ---------------------------------------------------------------- 정렬 (pitch_align 재사용형)
def aligned_pairs(df, tm, mmap):
    """(tr_pos[], tm_pos_in_d[], season[]) — 상태열 일치 투구쌍. d = GROUPS 필터된 tm."""
    tr = TM.add_appearances(df).reset_index(drop=True)
    ev_order = df.sort_values(rd.ID).index.to_numpy()
    tm1 = tm[tm["pitcher_trackman_id"].isin(mmap)].copy()
    tm1["pid"] = tm1["pitcher_trackman_id"].map(mmap)
    tm1 = tm1[tm1["chan"] == "R"]
    tr1 = tr[tr["pitcher_id"].isin(set(mmap.values())) & (tr["game_type"] == "R")]
    app_tr = {}
    for (pid, s, app), g in tr1.groupby(["pitcher_id", rd.SEASON, "app_no"], sort=False):
        key = (int(pid), int(s), int(g["game_month"].iloc[0]),
               int(g["game_dayofweek"].iloc[0]), len(g))
        app_tr.setdefault(key, []).append(g)
    pos_tr, pos_tm, ssn_out = [], [], []
    for (pid, s, gid), g in tm1.groupby(["pid", rd.SEASON, "trackman_game_id"], sort=False):
        g = g.sort_values("pitch_no")
        key = (int(pid), int(s), int(g["game_month"].iloc[0]),
               int(g["game_dayofweek"].iloc[0]), len(g))
        lst = app_tr.get(key)
        if not lst or len(lst) != 1:
            continue
        a = lst[0]
        ok = ((a["balls_before"].to_numpy() == g["balls_before"].to_numpy())
              & (a["strikes_before"].to_numpy() == g["strikes_before"].to_numpy())
              & (a["outs_before"].to_numpy() == g["outs_before"].to_numpy())
              & (a["inning"].to_numpy() == g["inning"].to_numpy()))
        pos_tr.append(a.index.to_numpy()[ok])
        pos_tm.append(g.index.to_numpy()[ok])
        ssn_out.append(np.full(int(ok.sum()), int(s)))
    tr_pos = np.concatenate(pos_tr)
    tm_idx = np.concatenate(pos_tm)
    ssn = np.concatenate(ssn_out)
    return tr, tr_pos, tm_idx, ssn


def resp_design(E7, cn, hh):
    """[E7 | E7·cn | E7·hh | E7²] = 28컬럼 (β 적용/적합 공용 — 동형 보장)."""
    return np.concatenate([E7, E7 * cn[:, None], E7 * hh[:, None], E7 ** 2], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024,2023,2022,2021")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(">> TM 상수 일괄")
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
    tm_pos_of = pd.Series(np.arange(len(d_tm)), index=d_tm.index)   # 원 tm 인덱스 → d_tm 위치

    print(">> train 준비 + 정렬")
    L.K_IS = 100.0
    df = rd.load_train()
    tr, tr_pos, tm_idx, ssn_pair = aligned_pairs(df, tm, mmap)
    keep = pd.Series(tm_idx).isin(tm_pos_of.index).to_numpy()
    tr_pos, tm_idx, ssn_pair = tr_pos[keep], tm_idx[keep], ssn_pair[keep]
    tm_pos = tm_pos_of.loc[tm_idx].to_numpy()
    y_pair = tr.loc[tr_pos, rd.TARGET].to_numpy(dtype=float)
    cn_pair = ((tr.loc[tr_pos, "balls_before"].to_numpy() * 3
                + tr.loc[tr_pos, "strikes_before"].to_numpy()) / 11.0)
    hh_pair = (tr.loc[tr_pos, "batter_hand"].to_numpy(int) == 2).astype(float)
    Z_pair = Zt_tm[tm_pos]
    ok_z = ~np.isnan(Z_pair).any(axis=1)
    print(f"   정렬 투구쌍 {len(tr_pos):,} (물리 완전 {int(ok_z.sum()):,}) [{time.time()-t0:.0f}s]")

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
    cn_row = cc / 11.0
    hand = (df["pitcher_hand"].to_numpy() == df["batter_hand"].to_numpy()).astype(float)
    hh_row = (df["batter_hand"].to_numpy(int) == 2).astype(float)
    pm_cols = []
    for mi in range(4):
        for ctx_ in (np.ones(len(df)), cn_row, hand):
            pm_cols.append(Z_pm[:, mi] * ctx_)
    D_pm = np.stack(pm_cols, axis=1)
    prof_tm6 = TMM.load_profile()

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

        tr_tm = ssn_tm < v
        ents_f = np.sort(d_tm.loc[tr_tm, "pitcher_trackman_id"].unique())
        eindex = {e: i for i, e in enumerate(ents_f)}
        eidx_f = d_tm.loc[tr_tm, "pitcher_trackman_id"].map(eindex).to_numpy(int)
        enc = CN.train_encoder(ctx_tm[tr_tm], eidx_f, gg_tm[tr_tm], Zt_tm[tr_tm], len(ents_f))
        E_real, covE = CN.encoder_features(enc, eindex, df, mmap_inv, prefix_tm)

        # ---- 투구 수준 β (정렬쌍 season < v, 실물리)
        mfit = ok_z & (ssn_pair < v)
        Dr = resp_design(Z_pair[mfit], cn_pair[mfit], hh_pair[mfit])
        sd_r = Dr.std(axis=0)
        sd_r = np.where(sd_r < 1e-9, 1.0, sd_r)
        Ar = np.concatenate([Dr / sd_r, np.ones((len(Dr), 1))], axis=1)
        Gr = Ar.T @ Ar + LAM_R * np.eye(Ar.shape[1])
        beta = np.linalg.solve(Gr, Ar.T @ (y_pair[mfit] - y_pair[mfit].mean()))
        rng_y = np.random.default_rng(5)
        y_sh = y_pair[mfit].copy()
        rng_y.shuffle(y_sh)
        beta_pl = np.linalg.solve(Gr, Ar.T @ (y_sh - y_sh.mean()))

        def r_hat(E, b):
            D = resp_design(E[:, 3:10].astype(np.float64), cn_row, hh_row) / sd_r
            return (np.concatenate([D, np.ones((len(D), 1))], axis=1) @ b)

        rh = r_hat(E_real, beta)[:, None]
        rh_bpl = r_hat(E_real, beta_pl)[:, None]
        print(f"\n[val {v}] ENS-9 동결 기준 {s_ref:.2f} · β 표본 {int(mfit.sum()):,} "
              f"· r_hat std {rh.std():.4f} [{time.time()-t1:.0f}s]")

        D_dn = denom_design(df, B_dn, fit)

        def stack4(D1, T_b, CP_b, E_b):
            b2 = orth(D1, T_b[:, 4:8], fit)
            D12 = np.concatenate([D1, b2], axis=1)
            b3 = orth(D12, CP_b[:, CP_CORE4], fit)
            D123 = np.concatenate([D12, b3], axis=1)
            b4 = orth(D123, E_b[:, 0:10].astype(np.float64), fit)
            return np.concatenate([D123, b4], axis=1)

        D_main4 = stack4(D_dn, T_real, CP_real, E_real)
        E7ctx = resp_design(E_real[:, 3:10].astype(np.float64), cn_row, hh_row)

        arms = [
            ("주셀4(재현)", D_main4),
            ("주셀4+resp", np.concatenate([D_main4, orth(D_main4, rh, fit)], axis=1)),
            ("주셀4+encctx28", np.concatenate([D_main4, orth(D_main4, E7ctx, fit)], axis=1)),
            ("위약βsh", np.concatenate([D_main4, orth(D_main4, rh_bpl, fit)], axis=1)),
        ]
        pmap0, T_p0, CP_p0 = pl_blocks[0]
        E_p0, _ = CN.encoder_features(enc, eindex, df, mmap_inv, prefix_tm, pl_map=pmap0)
        rh_p0 = r_hat(E_p0, beta)[:, None]
        D_pl4 = stack4(denom_design(df, B_dn, fit, perm=rng_perm.permutation(len(df))),
                       T_p0, CP_p0, E_p0)
        arms.append(("위약경로", np.concatenate([D_pl4, orth(D_pl4, rh_p0, fit)], axis=1)))

        for name, Dm in arms:
            corrC = ridge_corr(Dm, fit, val, rC, LAM)
            line = dict(val=v, arm=name)
            for w in WGRID:
                line[f"w{w:g}"] = round(float(
                    deploy_score(yv, ens9 + w * corrC) - s_ref), 3)
            res.append(line)
            print(f"  {name:18s}  " + "  ".join(f"w{w:g}: {line[f'w{w:g}']:+7.2f}" for w in WGRID))

    t = pd.DataFrame(res)
    for arm in ("주셀4+resp", "주셀4+encctx28"):
        core = t[(t.arm == arm) & t.val.isin([2024, 2023, 2022])]
        base = t[(t.arm == "주셀4(재현)") & t.val.isin([2024, 2023, 2022])]
        if len(core) >= 3:
            d5 = core["w0.5"].to_numpy() - base["w0.5"].to_numpy()
            print(f"\n[{arm} 증분 w0.5] " + " ".join(f"{x:+.2f}" for x in d5)
                  + f" · 평균 {d5.mean():+.2f}")
    (OUT_DIR / "response_gate.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR/'response_gate.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
