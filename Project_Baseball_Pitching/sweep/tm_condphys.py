# -*- coding: utf-8 -*-
"""W1-D — **조건부 물리 변조(Δphysics) 보정기**: 같은 구종군 안에서 물리가 (카운트,타자손)에
따라 어떻게 변하는가 (3주 프로그램 Track D, docs/log/27 예정).

    python sweep/tm_condphys.py --vals 2024                    # V24 스크린 (real−placebo ≥ +2)
    python sweep/tm_condphys.py --vals 2024,2023,2022,2021

## 무엇이 새 정보인가

기존 physmix = 구종군 물리의 **레벨**(무조건부 평균), 기존 tilt = 구종 **선택**의 count×hand 조건부.
이 블록 = **같은 구종을 던질 때 물리가 맥락에 따라 변하는 정도** — 완전한 빈칸이었다.
사전 실측(라벨-프리): fastball z(구속) 3-0 −0.171 ↔ 0-2 +0.211 (0.38σ), 개인차 sd 0.285.
신규 물리축 3종(zone_speed 감속량·rel_height·extension)은 코드베이스 참조 0건이던 미사용 컬럼.

## 구성

    Δphys[p,g,cc,hh,m] = shrink_κ( E[z_m|p,g,cc,hh] − E[z_m|p,g] )   (career<s, κ=30)
    행 적분: E[Δ_m|row] = Σ_g q_g(row)·Δphys[p,g,cc(row),hh(row),m]   (q = EB mix κ200, 행 갱신)

z는 (시즌×구종군) 내 표준화(장비/레퍼토리 교락 소거 — trackman._measure_cells와 동일 규약).
운반 = ENS-9 잔차(rCpm) ridge 보정기(ens10 인프라 재사용), 위약 = pid 경로교환(커버 보존).
arms: velo1 / core4(velo·drag·relh·ext) / full7(+spin·ivb·hb).
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
import trackman as TM             # noqa: E402
import tm_member as TMM           # noqa: E402
from eb_carrier import (get_members9, ens9_pred, deploy_score,   # noqa: E402
                        TM_W, W_PM, eb_block)
from is_corrector import ridge_corr   # noqa: E402
from tm_physmix import build_group_profile, row_z, GROUPS   # noqa: E402
from tm_intent import tier1_map, MIX_RC, K_MIX   # noqa: E402

OUT_DIR = HERE.parent / "results" / "ens11"
MEAS7 = ["velo", "drag", "relh", "ext", "spin", "ivb", "hb"]
RAW_OF = {"velo": "rel_speed", "spin": "spin_rate", "ivb": "induced_vert_break",
          "hb": "horz_break", "ext": "extension", "relh": "rel_height"}
K_CELL = 30.0            # 조건부 셀 수축 (κ)
MIN_BASE = 30            # 군별 base 표본 하한 (physmix MIN_G_N과 통일)
LAM = 1000.0
WGRID = (0.25, 0.5, 1.0)
ARMS_COLS = {"velo1": [0], "core4": [0, 1, 2, 3], "full7(주셀)": list(range(7))}


def build_condphys(tm, mmap):
    """{(pid, qs): (D[3,12,2,7], avail[3])} — career<qs 접두, 셀 수축 Δ."""
    d = tm[tm["pitch_type_group"].isin(GROUPS)].copy()
    d = d[d["pitcher_trackman_id"].isin(mmap)]
    d["pid"] = d["pitcher_trackman_id"].map(mmap)
    d["drag"] = d["rel_speed"] - d["zone_speed"]
    d["cc"] = (d["balls_before"].astype(int) * 3 + d["strikes_before"].astype(int)).clip(0, 11)
    d["hh"] = d["batter_hand"].map({"Left": 0, "Right": 1}).fillna(1).astype(int)
    d["gg"] = d["pitch_type_group"].map({g: i for i, g in enumerate(GROUPS)})

    # (시즌×구종군) 내 z — 각 측정축 (trackman._measure_cells와 동일 규약)
    grp = d.groupby([rd.SEASON, "pitch_type_group"])
    for m in MEAS7:
        col = "drag" if m == "drag" else RAW_OF[m]
        mu = grp[col].transform("mean")
        sd = grp[col].transform("std")
        d[f"z_{m}"] = (d[col].astype(float) - mu) / sd
    # 셀 집계: (pid, season, gg, cc, hh) — n(측정축별)과 s(합)
    agg = {}
    for m in MEAS7:
        agg[f"n_{m}"] = (f"z_{m}", "count")
        agg[f"s_{m}"] = (f"z_{m}", "sum")
    cell = d.groupby(["pid", rd.SEASON, "gg", "cc", "hh"]).agg(**agg).reset_index()
    base = d.groupby(["pid", rd.SEASON, "gg"]).agg(**agg).reset_index()

    out = {}
    seasons = sorted(base[rd.SEASON].unique())
    by_p_cell = dict(tuple(cell.groupby("pid")))
    for pid_, b in base.groupby("pid"):
        c_all = by_p_cell.get(pid_)
        for qs in range(int(min(seasons)) + 1, 2025 + 1):
            bb = b[b[rd.SEASON] < qs]
            if not len(bb):
                continue
            # base 평균 (career<qs)
            nb = np.zeros((3, 7))
            sb = np.zeros((3, 7))
            for _, r in bb.iterrows():
                gi = int(r["gg"])
                for j, m in enumerate(MEAS7):
                    nb[gi, j] += r[f"n_{m}"]
                    sb[gi, j] += r[f"s_{m}"]
            avail = nb[:, 0] >= MIN_BASE
            if not avail.any():
                continue
            with np.errstate(invalid="ignore"):
                mu_b = np.where(nb > 0, sb / np.maximum(nb, 1), 0.0)
            cc_ = c_all[c_all[rd.SEASON] < qs]
            nc = np.zeros((3, 12, 2, 7))
            sc = np.zeros((3, 12, 2, 7))
            gg_a = cc_["gg"].to_numpy(int)
            cca = cc_["cc"].to_numpy(int)
            hha = cc_["hh"].to_numpy(int)
            for j, m in enumerate(MEAS7):
                np.add.at(nc, (gg_a, cca, hha, j), cc_[f"n_{m}"].to_numpy(float))
                np.add.at(sc, (gg_a, cca, hha, j), cc_[f"s_{m}"].to_numpy(float))
            with np.errstate(invalid="ignore"):
                mu_c = np.where(nc > 0, sc / np.maximum(nc, 1), 0.0)
            delta = (mu_c - mu_b[:, None, None, :]) * (nc / (nc + K_CELL))
            delta[~avail, :, :, :] = 0.0
            out[(int(pid_), int(qs))] = (delta.astype(np.float32), avail.copy())
    return out


def row_block(df, cond, q_cur, pmap=None):
    """행별 7컬럼: E[Δ_m|row] = Σ_g q̂_g·Δ[p,g,cc,hh,m]. 비커버 0."""
    n = len(df)
    pid = df["pitcher_id"].to_numpy().astype(int)
    ssn = df[rd.SEASON].to_numpy().astype(int)
    cc = (df["balls_before"].to_numpy() * 3 + df["strikes_before"].to_numpy()).astype(int).clip(0, 11)
    hh = (df["batter_hand"].to_numpy().astype(int) == 2).astype(int)
    keys = sorted(cond)
    kidx = {k: i for i, k in enumerate(keys)}
    if pmap is not None and keys:
        order = np.empty(len(keys), int)
        for i, k in enumerate(keys):
            p2 = pmap.get(k[0], k[0])
            order[i] = kidx.get((p2, k[1]), i)
    else:
        order = np.arange(len(keys))
    Darr = np.stack([cond[k][0] for k in keys])[order] if keys else np.zeros((0, 3, 12, 2, 7))
    Aarr = np.stack([cond[k][1] for k in keys])[order] if keys else np.zeros((0, 3), bool)
    idx = np.full(n, -1)
    cache = {}
    for i, (p, s) in enumerate(zip(pid, ssn)):
        j = cache.get((p, s), -2)
        if j == -2:
            j = kidx.get((p, s), -1)
            cache[(p, s)] = j
        idx[i] = j
    cov = idx >= 0
    qc = np.stack([q_cur[j] for j in range(3)], axis=1)
    out = np.zeros((n, 7))
    ii = np.where(cov)[0]
    if len(ii):
        Ai = Aarr[idx[ii]]
        qi = qc[ii] * Ai
        qi = qi / np.maximum(qi.sum(axis=1, keepdims=True), 1e-9)
        Di = Darr[idx[ii], :, cc[ii], hh[ii], :]          # (k,3,7) — 행별 (cc,hh) 슬라이스
        out[ii] = np.einsum("ng,ngm->nm", qi, Di)
    return out, cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(">> 조건부 물리 테이블 (Tier1, career<s, κ_cell=30)")
    mmap = tier1_map()
    tm = TM.load_trackman()
    cond = build_condphys(tm, mmap)
    print(f"   {len(cond):,} (pid,season) 엔트리 [{time.time()-t0:.0f}s]")

    print(">> train 준비 (K_IS=100)")
    L.K_IS = 100.0
    df = rd.load_train()
    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy())
    X = L.build_features(df, base=(nb, kb)).reset_index(drop=True)
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()
    _, q_pre, q_cur = eb_block(df, "pitcher_id", "asof_pitcher_pitchmix_n", MIX_RC, K_MIX, "mixq")
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

    print(">> 블록 (real + 위약 2시드)")
    t1 = time.time()
    B_real, covP = row_block(df, cond, q_cur)
    pids_in = sorted({p for (p, _s) in cond})
    B_pl = []
    for sd_ in range(2):
        rng = np.random.default_rng(91 + sd_)
        perm = rng.permutation(len(pids_in))
        pmap = {pids_in[i]: pids_in[perm[i]] for i in range(len(pids_in))}
        B_pl.append(row_block(df, cond, q_cur, pmap=pmap))
    nz = (np.abs(B_real) > 1e-12).mean(axis=0)
    print("   비영 비율: " + " ".join(f"{m}={v:.2f}" for m, v in zip(MEAS7, nz))
          + f"  [{time.time()-t1:.0f}s]")

    res = []
    for v in [int(s) for s in args.vals.split(",")]:
        fit, val = (season == v - 1), (season == v)
        P, corr, corr_pm_v, yv = get_members9(v)
        ens9 = ens9_pred(P, corr, corr_pm_v)
        s_ref = deploy_score(yv, ens9)

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
        print(f"\n[val {v}] ENS-9 동결 기준 {s_ref:.2f} · 커버 {covP[val].mean():.3f}")

        arms = [(nm, B_real, covP, cols) for nm, cols in ARMS_COLS.items()]
        for sd_, (Bp, covp) in enumerate(B_pl):
            arms.append((f"위약s{sd_}", Bp, covp, list(range(7))))
            arms.append((f"위약core4 s{sd_}", Bp, covp, [0, 1, 2, 3]))
        for name, B, cv, cols in arms:
            D = B[:, cols]
            corr2 = ridge_corr(D, fit, val, rCpm, LAM)
            corr2[~cv[val]] = 0.0
            line = dict(val=v, arm=name)
            for w in WGRID:
                line[f"w{w:g}"] = round(float(
                    deploy_score(yv, ens9 + w * corr2) - s_ref), 3)
            res.append(line)
            print(f"  {name:16s}  " + "  ".join(f"w{w:g}: {line[f'w{w:g}']:+7.2f}" for w in WGRID))

    t = pd.DataFrame(res)
    core = t[(t.arm == "full7(주셀)") & t.val.isin([2024, 2023, 2022])]
    if len(core) >= 3:
        for w in (0.25, 0.5):
            d = core[f"w{w:g}"]
            print(f"\n[주셀 w{w:g}] 평균 {d.mean():+.2f} · SD {d.std():.2f} · "
                  f"통과 {d.mean() > 0 and d.mean() > d.std()}")
    (OUT_DIR / "condphys_gate.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR/'condphys_gate.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
