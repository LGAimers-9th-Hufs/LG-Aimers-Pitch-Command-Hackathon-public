# -*- coding: utf-8 -*-
"""ENS-10 Run B — **TM 의도믹스 블록**: 트랙맨의 미사용 조건부 통계 × 행 갱신 mix 가중.

    python sweep/tm_intent.py --vals 2024                    # V24 스크린 (real−placebo ≥ +2 통과선)
    python sweep/tm_intent.py --vals 2024,2023,2022,2021

## 세 성분 (Codex 자문 B1/B2/D — 전부 행 변수화 원리의 신규 적용)

B1 **당해 mix 혁신 × 구종군 물리**: q_is,g(row) = EB(Δk_g, Δn; κ=200, prior=직전시즌말 mix)
   → Δz_m = Σ_g [q_is,g − q_pre,g]·P[p,g,m].  physmix(커리어 mix 레벨)와 달리 **당해 레퍼토리
   변화의 물성**을 묻는다. 4컬럼.

B2 **count×타자손 구종 선택 성향(tilt)**: 트랙맨 원자료에서 tilt[p,c,h,g] =
   q_TM(g|p,c,h)/q_TM(g|p) (career<s, pseudocount 30 수축) → w_g(row) ∝ q_is,g·tilt →
   Δμ_m = Σ_g [w_g − q_is,g]·P[p,g,m]. 기존 physmix는 count별 game plan을 모르고, 기존 TM
   보정기의 물리×선형count는 구종군 배분을 바꾸지 못한다 — 새 정보×운반. 4컬럼.

B3 **구종군별 산포**: S_velo[p,g], S_move[p,g] (군내 z 표준편차, career<s) →
   Lσ = Σ_g q_is,g·S[p,g] (행 갱신 레벨), Δσ = Σ_g [w_g − q_is,g]·S[p,g]. 기존 tm_velostd/
   movestd는 투수 전체 상수 — 군별 산포×현재 mix는 행마다 변한다. 4컬럼.

## 타깃·게이트

타깃 = residB − 0.5·corr_pm(fit) — 설치된 TM 보정기·physmix에 순차 직교화(rCpm).
판정 = ENS-9 동결 기준 Δ(동결 배포 캘리), w∈{0.25,0.5,1.0}.
위약 = 프로필·tilt·산포의 투수 경로를 **같은 bijection**으로 통째 교환(커버 보존) 2시드.
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
from tm_physmix import build_group_profile, row_z, GROUPS, M4, MIN_G_N   # noqa: E402

OUT_DIR = HERE.parent / "results" / "ens10"
MATCHES = HERE.parent / "results" / "trackman" / "matches.csv"
MIX_RC = [("asof_pitcher_fastball_rate", "fb"), ("asof_pitcher_breaking_rate", "br"),
          ("asof_pitcher_offspeed_rate", "os")]
K_MIX = 200
TILT_PSEUDO = 30.0
LAM = 1000.0
WGRID = (0.25, 0.5, 1.0)
HAND_MAP = {"Left": 1, "Right": 2}      # 트랙맨 문자열 → train 정수 코드 (2=우, A-2 검증)


# ---------------------------------------------------------------- TM 상수 구축
def tier1_map():
    m = pd.read_csv(MATCHES)
    m = m[m.tier == 1]
    return dict(zip(m.tm_id, m.pitcher_id))


def build_tilt(tm, mmap):
    """{(pid, season): T[12,2,3] tilt, avail_g[3]} — career<season, pseudocount 수축.

    tilt[c,h,g] = (n[p,c,h,g] + 30·q_p,g) / (n[p,c,h,·] + 30) / q_p,g  (q_p,g = 투수 전체 믹스)
    """
    d = tm[tm["pitch_type_group"].isin(GROUPS)].copy()
    d = d[d["pitcher_trackman_id"].isin(mmap)]
    d["pid"] = d["pitcher_trackman_id"].map(mmap)
    d["cc"] = (d["balls_before"].astype(int) * 3 + d["strikes_before"].astype(int)).clip(0, 11)
    d["hh"] = d["batter_hand"].map(HAND_MAP).fillna(2).astype(int) - 1     # 0=좌, 1=우
    d["gg"] = d["pitch_type_group"].map({g: i for i, g in enumerate(GROUPS)})
    cell = (d.groupby(["pid", rd.SEASON, "cc", "hh", "gg"]).size()
            .rename("n").reset_index())

    tilt = {}
    for pid_, grp in cell.groupby("pid"):
        seasons = sorted(grp[rd.SEASON].unique())
        # 시즌 접두 누적 → query season qs 는 (< qs) 만 사용
        for qs in range(int(min(seasons)) + 1, 2025 + 1):
            g2 = grp[grp[rd.SEASON] < qs]
            if not len(g2):
                continue
            N = np.zeros((12, 2, 3))
            for cc_, hh_, gg_, n_ in zip(g2["cc"], g2["hh"], g2["gg"], g2["n"]):
                N[cc_, hh_, gg_] += n_
            tot = N.sum()
            if tot < MIN_G_N:
                continue
            q_pg = N.sum(axis=(0, 1)) / tot                       # 투수 전체 믹스 (3,)
            avail = q_pg > 0
            denom = N.sum(axis=2, keepdims=True) + TILT_PSEUDO
            post = (N + TILT_PSEUDO * q_pg[None, None, :]) / denom
            with np.errstate(divide="ignore", invalid="ignore"):
                T = np.where(avail[None, None, :], post / np.maximum(q_pg, 1e-9), 1.0)
            tilt[(int(pid_), int(qs))] = (T.astype(np.float32), avail)
    return tilt


def build_disp(tm, mmap):
    """{(pid, season): S[3,2]} — 군별 (velo, move) 산포(z 표준편차), career<season 풀링."""
    cells, _ = TM._measure_cells(tm)
    cells = cells[cells["pitcher_trackman_id"].isin(mmap)].copy()
    cells["pid"] = cells["pitcher_trackman_id"].map(mmap)

    def ssdof(sub, m):
        n, s_, q = (sub[f"n_{m}"].to_numpy(float), sub[f"s_{m}"].to_numpy(float),
                    sub[f"q_{m}"].to_numpy(float))
        ok = n >= 2
        ss = np.where(ok, q - s_ * s_ / np.maximum(n, 1), 0.0)
        dof = np.where(ok, n - 1, 0.0)
        return ss, dof

    disp = {}
    for (pid_, g), grp in cells.groupby(["pid", "pitch_type_group"]):
        gi = GROUPS.index(g)
        grp = grp.sort_values(rd.SEASON)
        ss_list = {}
        for m in ("velo", "ivb", "hb"):
            ss, dof = ssdof(grp, m)
            ss_list[m] = (np.cumsum(ss), np.cumsum(dof))
        ssn_arr = grp[rd.SEASON].to_numpy()
        for qs in range(int(ssn_arr.min()) + 1, 2025 + 1):
            k = int(np.searchsorted(ssn_arr, qs))
            if k == 0:
                continue
            sv, dv = ss_list["velo"][0][k - 1], ss_list["velo"][1][k - 1]
            si, di = ss_list["ivb"][0][k - 1], ss_list["ivb"][1][k - 1]
            sh, dh = ss_list["hb"][0][k - 1], ss_list["hb"][1][k - 1]
            if dv < MIN_G_N - 1:
                continue
            key = (int(pid_), int(qs))
            S = disp.get(key, np.full((3, 2), np.nan))
            S[gi, 0] = np.sqrt(sv / dv) if dv > 0 else np.nan
            S[gi, 1] = np.sqrt((si + sh) / (di + dh)) if (di + dh) > 0 else np.nan
            disp[key] = S
    return disp


# ---------------------------------------------------------------- 행 피처
def build_block(df, prof, tilt, disp, q_pre, q_cur, pmap=None):
    """12컬럼: dz4(B1) + dmu4(B2) + [Lσv, Lσm, Δσv, Δσm](B3). 비커버 0."""
    n = len(df)
    pid = df["pitcher_id"].to_numpy().astype(int)
    ssn = df[rd.SEASON].to_numpy().astype(int)
    cc = (df["balls_before"].to_numpy() * 3 + df["strikes_before"].to_numpy()).astype(int).clip(0, 11)
    hh = (df["batter_hand"].to_numpy().astype(int) == 2).astype(int)        # 0=좌, 1=우

    def make_idx(store):
        keys = sorted(store)
        kidx = {k: i for i, k in enumerate(keys)}
        if pmap is not None and keys:
            order = np.empty(len(keys), int)
            for i, k in enumerate(keys):
                p2 = pmap.get(k[0], k[0])
                order[i] = kidx.get((p2, k[1]), i)
        else:
            order = np.arange(len(keys))
        idx = np.full(n, -1)
        cache = {}
        for i, (p, s) in enumerate(zip(pid, ssn)):
            j = cache.get((p, s), -2)
            if j == -2:
                j = kidx.get((p, s), -1)
                cache[(p, s)] = j
            idx[i] = j
        return keys, order, idx

    keysP, ordP, idxP = make_idx(prof)
    Parr = np.stack([prof[k][0] for k in keysP])[ordP] if keysP else np.zeros((0, 3, 4))
    Aarr = np.stack([prof[k][1] for k in keysP])[ordP] if keysP else np.zeros((0, 3), bool)
    keysT, ordT, idxT = make_idx(tilt)
    Tarr = np.stack([tilt[k][0] for k in keysT])[ordT] if keysT else np.zeros((0, 12, 2, 3))
    keysS, ordS, idxS = make_idx(disp)
    Sarr = np.stack([disp[k] for k in keysS])[ordS] if keysS else np.zeros((0, 3, 2))

    dq = np.stack([q_cur[j] - q_pre[j] for j in range(3)], axis=1)          # (n,3) 혁신
    qc = np.stack([q_cur[j] for j in range(3)], axis=1)                     # (n,3) 당해 mix

    out = np.zeros((n, 12))
    covP = idxP >= 0
    ii = np.where(covP)[0]
    if len(ii):
        Pi, Ai = Parr[idxP[ii]], Aarr[idxP[ii]]
        # B1: Δz = Σ_g dq_g·P[g,m] (avail 군만)
        dqi = dq[ii] * Ai
        out[ii, 0:4] = np.einsum("ng,ngm->nm", dqi, Pi)
        # B2: w ∝ qc·tilt (tilt 있는 행만)
        jj = ii[idxT[ii] >= 0]
        if len(jj):
            Tj = Tarr[idxT[jj]]                                            # (k,12,2,3)
            tj = Tj[np.arange(len(jj)), cc[jj], hh[jj]]                     # (k,3)
            Aj = Aarr[idxP[jj]]
            qj = qc[jj] * Aj
            qj = qj / np.maximum(qj.sum(axis=1, keepdims=True), 1e-9)
            wj = qj * tj
            wj = wj / np.maximum(wj.sum(axis=1, keepdims=True), 1e-9)
            out[jj, 4:8] = np.einsum("ng,ngm->nm", wj - qj, Parr[idxP[jj]])
        # B3: Lσ/Δσ (산포 있는 행만)
        kk = ii[idxS[ii] >= 0]
        if len(kk):
            Sk = Sarr[idxS[kk]]                                            # (k,3,2)
            okS = np.isfinite(Sk).all(axis=2)                              # (k,3)
            Sk0 = np.nan_to_num(Sk, nan=0.0)
            qk = qc[kk] * okS
            qk = qk / np.maximum(qk.sum(axis=1, keepdims=True), 1e-9)
            out[kk, 8:10] = np.einsum("ng,ngm->nm", qk, Sk0)               # Lσ (velo, move)
            mask_kk = idxT[kk] >= 0
            kk2 = kk[mask_kk]
            if len(kk2):
                Tk = Tarr[idxT[kk2]]
                tk = Tk[np.arange(len(kk2)), cc[kk2], hh[kk2]]
                qk2 = qc[kk2] * okS[mask_kk]
                qk2 = qk2 / np.maximum(qk2.sum(axis=1, keepdims=True), 1e-9)
                wk = qk2 * tk
                wk = wk / np.maximum(wk.sum(axis=1, keepdims=True), 1e-9)
                out[kk2, 10:12] = np.einsum("ng,ngm->nm", wk - qk2,
                                            np.nan_to_num(Sarr[idxS[kk2]], nan=0.0))
    # Lσ는 레벨 컬럼이라 fit-평균 센터링을 ridge 표준화에 맡긴다(절편 공유)
    return out, covP


COLS = ["dz_velo", "dz_spin", "dz_ivb", "dz_hb",
        "dmu_velo", "dmu_spin", "dmu_ivb", "dmu_hb",
        "lsig_velo", "lsig_move", "dsig_velo", "dsig_move"]
SUB = {"B1(dz4)": list(range(0, 4)), "B2(dmu4)": list(range(4, 8)),
       "B3(sig4)": list(range(8, 12)), "B1+B2+B3(주셀)": list(range(12))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(">> TM 상수: 프로필 · tilt · 산포 (Tier1, career<s)")
    mmap = tier1_map()
    tm = TM.load_trackman()
    prof = build_group_profile()
    tilt = build_tilt(tm, mmap)
    disp = build_disp(tm, mmap)
    print(f"   prof {len(prof):,} · tilt {len(tilt):,} · disp {len(disp):,} [{time.time()-t0:.0f}s]")

    print(">> train 준비 (K_IS=100)")
    L.K_IS = 100.0
    df = rd.load_train()
    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy())
    X = L.build_features(df, base=(nb, kb)).reset_index(drop=True)
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()
    _, q_pre, q_cur = eb_block(df, "pitcher_id", "asof_pitcher_pitchmix_n", MIX_RC, K_MIX, "mixq")
    prof_tm = TMM.load_profile()
    Z_pm, cov_pm = row_z(df, prof)
    cc = (df["balls_before"].to_numpy() * 3 + df["strikes_before"].to_numpy()).astype(int)
    cn = cc / 11.0
    hand = (df["pitcher_hand"].to_numpy() == df["batter_hand"].to_numpy()).astype(float)
    pm_cols = []
    for mi in range(4):
        for ctx in (np.ones(len(df)), cn, hand):
            pm_cols.append(Z_pm[:, mi] * ctx)
    D_pm = np.stack(pm_cols, axis=1)

    print(">> 블록 계산 (real + 위약 2시드)")
    t1 = time.time()
    B_real, covP = build_block(df, prof, tilt, disp, q_pre, q_cur)
    pids_in = sorted({p for (p, _s) in prof})
    B_pl = []
    for sd_ in range(4):
        rng = np.random.default_rng(71 + sd_)
        perm = rng.permutation(len(pids_in))
        pmap = {pids_in[i]: pids_in[perm[i]] for i in range(len(pids_in))}
        B_pl.append(build_block(df, prof, tilt, disp, q_pre, q_cur, pmap=pmap))
    nz = (np.abs(B_real) > 1e-12).mean(axis=0)
    print(f"   비영 비율: " + " ".join(f"{c}={v:.2f}" for c, v in zip(COLS, nz))
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

        arms = [(nm, B_real, covP, cols) for nm, cols in SUB.items()]
        for sd_, (Bp, covp) in enumerate(B_pl):
            arms.append((f"위약s{sd_}", Bp, covp, list(range(12))))
            arms.append((f"위약B2s{sd_}", Bp, covp, list(range(4, 8))))
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
    core = t[(t.arm == "B1+B2+B3(주셀)") & t.val.isin([2024, 2023, 2022])]
    if len(core) >= 3:
        for w in (0.25, 0.5):
            d = core[f"w{w:g}"]
            print(f"\n[주셀 w{w:g}] 평균 {d.mean():+.2f} · SD {d.std():.2f} · "
                  f"통과 {d.mean() > 0 and d.mean() > d.std()}")
    (OUT_DIR / "tm_intent_gate.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR/'tm_intent_gate.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
