# -*- coding: utf-8 -*-
"""연장 라운드 3 — **TM 구종군 물리 × asof mix** (Codex#2 처방3, 마지막 새 정보형 셀).

    python sweep/tm_physmix.py --vals 2024

z_m(row) = Σ_g mix_g(row) · profile[pid, career<s][g, m] — 구종군별 물리 평균(z, 시즌×군 내
표준화)을 행마다 갱신되는 공식 asof mix로 가중 → (투수,시즌) 안에서 행마다 변한다(판별 통과).
⚠ fastball 성분(tm_velo_fb 등 × fb_rate)은 기존 TM 보정기 design에 이미 존재 → 잔차 타깃은
**B = june 잔차 − 0.5·corr_tm(fit)** (순차 직교화)로 순수 증분만 잰다. A(원잔차)는 대조 arm.
위약 = 커버 보존 pid 경로 교환 2시드. 스크린 통과선 = V24 Δ ≥ +3 & 위약 분리.
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
from eb_carrier import get_members, deploy_score, ENS7_W, TM_W  # noqa: E402
from is_corrector import ridge_corr   # noqa: E402

OUT_DIR = HERE.parent / "results" / "dual"
MATCHES = HERE.parent / "results" / "trackman" / "matches.csv"
M4 = ["velo", "spin", "ivb", "hb"]
GROUPS = ["fastball", "breaking", "offspeed"]
MIN_G_N = 30
LAM = 1000.0
WGRID = (0.25, 0.5, 1.0)


def build_group_profile(tiers=(1,), mmap=None):
    """{(pid, season): (P[3,4] 군별 물리 z평균, avail[3])} — career < season 접두, 군별 n≥30.

    mmap 지정 시 matches.csv 대신 그 {tm_id: pid} 매핑을 쓴다(매칭 v2 A/B용, 기본 동작 불변).
    """
    if mmap is None:
        matches = pd.read_csv(MATCHES)
        mmap = dict(zip(matches[matches.tier.isin(tiers)].tm_id,
                        matches[matches.tier.isin(tiers)].pitcher_id))
    tm = TM.load_trackman()
    cells, _rows = TM._measure_cells(tm)
    cells = cells[cells["pitcher_trackman_id"].isin(mmap)].copy()
    cells["pid"] = cells["pitcher_trackman_id"].map(mmap)

    prof = {}
    seasons_all = sorted(cells[rd.SEASON].unique().tolist() + [2024])
    query_seasons = list(range(int(min(seasons_all)) + 1, 2025 + 1))   # 2025 = 서빙용(커리어 ≤2024)
    for (pid_, g), grp in cells.groupby(["pid", "pitch_type_group"]):
        gi = GROUPS.index(g)
        grp = grp.sort_values(rd.SEASON)
        ss = grp[rd.SEASON].to_numpy()
        cn = {m: np.cumsum(grp[f"n_{m}"].to_numpy(dtype=float)) for m in M4}
        cs = {m: np.cumsum(grp[f"s_{m}"].to_numpy(dtype=float)) for m in M4}
        for qs in query_seasons:
            k = int(np.searchsorted(ss, qs))          # 시즌 < qs 인 셀 개수
            if k == 0:
                continue
            if cn["velo"][k - 1] < MIN_G_N:
                continue
            key = (int(pid_), int(qs))
            P, avail = prof.get(key, (np.zeros((3, 4)), np.zeros(3, dtype=bool)))
            for mi, m in enumerate(M4):
                if cn[m][k - 1] > 0:
                    P[gi, mi] = cs[m][k - 1] / cn[m][k - 1]
            avail[gi] = True
            prof[key] = (P, avail)
    return prof


def row_z(df, prof, pmap=None):
    """행별 z4 = Σ_g mix_g·P[g,·] / Σ_g(avail) mix_g. pmap = 위약 경로 교환."""
    pid = df["pitcher_id"].to_numpy().astype(int)
    ssn = df[rd.SEASON].to_numpy().astype(int)
    mix = np.stack([np.nan_to_num(df[c].to_numpy(dtype=float), nan=1 / 3)
                    for c in ("asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
                              "asof_pitcher_offspeed_rate")], axis=1)
    keys = sorted(prof)
    kidx = {k: i for i, k in enumerate(keys)}
    Parr = np.stack([prof[k][0] for k in keys])           # (K,3,4)
    Aarr = np.stack([prof[k][1] for k in keys])           # (K,3)
    if pmap is not None:                                   # 커버 보존 경로 교환
        pids_in = sorted({p for (p, _s) in keys})
        remap = {}
        for k in keys:
            p2 = pmap.get(k[0], k[0])
            remap[k] = kidx.get((p2, k[1]), kidx[k])
        order = np.array([remap[k] for k in keys])
        Parr, Aarr = Parr[order], Aarr[order]
    idx = np.full(len(df), -1)
    cache = {}
    for i, (p, s) in enumerate(zip(pid, ssn)):
        j = cache.get((p, s), -2)
        if j == -2:
            j = kidx.get((p, s), -1)
            cache[(p, s)] = j
        idx[i] = j
    cov = idx >= 0
    Z = np.zeros((len(df), 4))
    ii = np.where(cov)[0]
    Pi, Ai = Parr[idx[ii]], Aarr[idx[ii]]                  # (n,3,4), (n,3)
    mi = mix[ii] * Ai
    denom = np.maximum(mi.sum(axis=1, keepdims=True), 1e-6)
    Z[ii] = np.einsum("ng,ngm->nm", mi / denom, Pi)
    return Z, cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(">> TM 구종군 프로필 구축 (Tier1)")
    prof = build_group_profile()
    print(f"   프로필 {len(prof):,} (pid,season) 엔트리 [{time.time()-t0:.0f}s]")

    print(">> train 준비 (K_IS=100)")
    L.K_IS = 100.0
    df = rd.load_train()
    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy())
    X = L.build_features(df, base=(nb, kb)).reset_index(drop=True)
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()
    prof_tm = TMM.load_profile()

    Z_real, cov = row_z(df, prof)
    pids_in = sorted({p for (p, _s) in prof})
    pl = []
    for sd_ in range(2):
        rng = np.random.default_rng(51 + sd_)
        perm = rng.permutation(len(pids_in))
        pmap = {pids_in[i]: pids_in[perm[i]] for i in range(len(pids_in))}
        pl.append(row_z(df, prof, pmap))
    cc = (df["balls_before"].to_numpy() * 3 + df["strikes_before"].to_numpy()).astype(int)
    cn = cc / 11.0
    hand = (df["pitcher_hand"].to_numpy() == df["batter_hand"].to_numpy()).astype(float)

    def design(Z):
        cols = []
        for mi in range(4):
            for ctx in (np.ones(len(df)), cn, hand):
                cols.append(Z[:, mi] * ctx)
        return np.stack(cols, axis=1)

    res = []
    for v in [int(s) for s in args.vals.split(",")]:
        fit, val = (season == v - 1), (season == v)
        P, corr, yv = get_members(df, X, y, season, v)
        ens_full = sum(P[m] * w for m, w in ENS7_W.items()) + TM_W * corr
        s_ref = deploy_score(yv, ens_full)
        residA = TMM.june_fit_resid(X, y, fit)
        Df_tm, covf_tm = TMM.design(df, X, prof_tm, fit)
        A_tm = Df_tm[fit & covf_tm]
        G = A_tm.T @ A_tm + 1000.0 * np.eye(A_tm.shape[1])
        c_tm = np.linalg.solve(G, A_tm.T @ residA[covf_tm[fit]])
        corr_tm_fit = Df_tm[fit] @ c_tm
        corr_tm_fit[~covf_tm[fit]] = 0.0
        residB = residA - TM_W * corr_tm_fit
        print(f"\n[val {v}] ENS-7 기준 {s_ref:.2f} · 커버 {cov[val].mean():.3f}")

        arms = [("physmix·B(주셀)", Z_real, cov, residB), ("physmix·A", Z_real, cov, residA),
                ("위약0·B", pl[0][0], pl[0][1], residB), ("위약1·B", pl[1][0], pl[1][1], residB)]
        for name, Z, cv, tgt in arms:
            D = design(Z)
            corr2 = ridge_corr(D, fit, val, tgt, LAM)
            corr2[~cv[val]] = 0.0
            line = dict(val=v, arm=name, cover=round(float(cv[val].mean()), 3))
            for w in WGRID:
                line[f"w{w:g}"] = round(float(
                    deploy_score(yv, ens_full + w * corr2) - s_ref), 3)
            res.append(line)
            print(f"  {name:16s} cover {line['cover']:.3f}  " +
                  "  ".join(f"w{w:g}: {line[f'w{w:g}']:+7.2f}" for w in WGRID))

    t = pd.DataFrame(res)
    (OUT_DIR / "tm_physmix_gate.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR/'tm_physmix_gate.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
