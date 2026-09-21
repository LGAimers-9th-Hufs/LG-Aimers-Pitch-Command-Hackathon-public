# -*- coding: utf-8 -*-
"""ENS-10 Run A — **ENS-9 재기준화 패키지** (안1 + denom [+donut]) 잔차 보정기 게이트.

    python sweep/ens10_pkg.py --vals 2024                    # V24 스크린
    python sweep/ens10_pkg.py --vals 2024,2023,2022,2021     # 전 폴드
    python sweep/ens10_pkg.py --vals 2024,2023,2022,2021 --donut

## 왜 재기준화인가 (Codex 자문, 2026-08-08)

is_corrector 안1의 +7.44는 ENS-7 기준이고, 15항 중 12항이 ENS-9에 이미 실린 cxp(카운트×is_delta)
와 동일하다. 타깃에서 **설치된 증분을 전부 빼야** 중복 계상이 없다:

    rC = (y − june_fit)                       ← 기존 보정기 계약(june 잔차)
         − 0.5·corr_tm(fit)                   ← 설치된 TM 보정기
         − 0.5·corr_pm(fit)                   ← 설치된 physmix 보정기
         − 0.475·(nn_pkg(fit) − nn_bis(fit))  ← 설치된 pEB+cxp 선형 증분

## 설계 (Codex 권고 그대로)

블록 순서 안1(14) → denom(9) → donut(1·조건부). 후행 블록은 선행 블록에 **fit-행 Gram 직교화**,
누적 블록을 **단일 ridge(λ=1000, 공유 절편 1개)** — 절편을 블록마다 두면 글로벌 shift를 여러 번
회수해 가산성이 과대평가된다. w=0.5 고정(0.25는 참고).

주셀 = **안1+denom**. donut은 주셀 생존 시에만 추가(사전등록). 위약 = 안1 is4 경로교환(2시드)
+ denom 행 셔플 + (donut 시) σa 테이블 pid 경로교환 — 결합 디자인 전체를 위약으로 재구성.
판정: ENS-9 동결 기준 Δ(동결 배포 캘리), 3폴드 평균>SD & 위약 분리.
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
import donut as DN                # noqa: E402
from nn_carrier import BIS, train_lin   # noqa: E402
from eb_carrier import (get_members9, ens9_pred, deploy_score,   # noqa: E402
                        ENS9_W, TM_W, W_PM, eb_block)
from interaction_carrier import build_products   # noqa: E402
from is_corrector import is_factors, build_design, ridge_corr   # noqa: E402
from tm_physmix import build_group_profile, row_z   # noqa: E402

OUT_DIR = HERE.parent / "results" / "ens10"
P_SUCC = [("asof_pitcher_success_rate", "succ")]
DN_M = ["pn1_log", "pr1_shrunk", "workload_delta"]
LAM = 1000.0
WGRID = (0.25, 0.5)


def orth(prev, blk, fit):
    """blk을 prev에 fit-행 기준 Gram 직교화(계수는 fit에서 추정, 전 행에 적용)."""
    A = prev[fit]
    G = A.T @ A + 1e-6 * np.eye(A.shape[1])
    W = np.linalg.solve(G, A.T @ blk[fit])
    return blk - prev @ W


def denom_design(df, B_dn, fit, perm=None):
    """denom M3 ⊗ (1, count/11, hand) = 9컬럼. NaN → fit 평균. perm = 위약 행 셔플."""
    cc = (df["balls_before"].to_numpy() * 3 + df["strikes_before"].to_numpy()).astype(int)
    cn = cc / 11.0
    hand = (df["pitcher_hand"].to_numpy() == df["batter_hand"].to_numpy()).astype(float)
    cols = []
    for c in DN_M:
        v = B_dn[c].to_numpy(dtype=float)
        if perm is not None:
            v = v[perm]
        mu = float(np.nanmean(v[fit]))
        v = np.nan_to_num(v, nan=mu)
        for ctx in (np.ones(len(df)), cn, hand):
            cols.append(v * ctx)
    return np.stack(cols, axis=1)


def donut_scalar(df, table, da, fit, pmap=None):
    """v = Psucc(σ_p, |a_p + δa_c|) 의 순수 상호작용 성분(이중 센터링, fit 통계)."""
    pid = df["pitcher_id"].to_numpy().astype(np.int64)
    ssn = df[rd.SEASON].to_numpy().astype(np.int64)
    if pmap is not None:
        keys_map = np.array([pmap.get(int(p), int(p)) for p in pid], dtype=np.int64) * 10000 + ssn
    else:
        keys_map = pid * 10000 + ssn
    cc = (df["balls_before"].to_numpy() * 3 + df["strikes_before"].to_numpy()).astype(int)
    sig = np.ones(len(df))
    aim = np.zeros(len(df))
    cov = np.zeros(len(df), dtype=bool)
    cache = {}
    for i, k in enumerate(keys_map):
        v = cache.get(k)
        if v is None:
            v = table.get(int(k), None)
            cache[k] = v if v is not None else False
        elif v is False:
            v = None
        if v:
            sig[i], aim[i] = v
            cov[i] = True
    dac = np.array([da.get(int(c), 0.0) for c in range(12)])[cc]
    _, _, _, ps = DN.p_parts(sig, np.abs(aim + dac))
    ps = np.where(cov, ps, np.nan)
    # 이중 센터링: v − 투수평균 − 카운트평균 + 총평균 (전부 fit 행 통계)
    d = pd.DataFrame({"v": ps, "k": keys_map, "c": cc})
    fmask = fit & cov
    gm = float(d.loc[fmask, "v"].mean())
    pmu = d.loc[fmask].groupby("k")["v"].mean()
    cmu = d.loc[fmask].groupby("c")["v"].mean()
    vv = (d["v"] - d["k"].map(pmu).fillna(gm) - d["c"].map(cmu).fillna(gm) + gm).to_numpy()
    return np.nan_to_num(np.where(cov, vv, 0.0), nan=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024")
    ap.add_argument("--donut", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(">> 준비 (K_IS=100)")
    L.K_IS = 100.0
    df = rd.load_train()
    lut = IS.career_end_lookup(df)
    pid = df["pitcher_id"].to_numpy()
    ssn = df[rd.SEASON].to_numpy()
    nb, kb = IS.base_for(lut, pid, ssn)
    X = L.build_features(df, base=(nb, kb)).reset_index(drop=True)
    X_pl = {sd_: L.build_features(df, base=IS.base_for(lut, pid, ssn, shuffle_seed=7 + sd_))
            .reset_index(drop=True) for sd_ in range(2)}       # 안1 위약: is4 경로 교환
    y = df[rd.TARGET].to_numpy().astype(float)
    season = ssn

    print(">> 블록: bis/pEB/cxp(패키지 재현) · denom · physmix")
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

    dn_table = da_by_fold = None
    if args.donut:
        print(">> donut (σ,a) 역산")
        dn_table = DN.build_sigma_a(df)
        pids_dn = sorted({k // 10000 for k in dn_table})

    rng_perm = np.random.default_rng(23)
    res = []
    for v in [int(s) for s in args.vals.split(",")]:
        t1 = time.time()
        fit, val = (season == v - 1), (season == v)
        P, corr, corr_pm_v, yv = get_members9(v)
        ens9 = ens9_pred(P, corr, corr_pm_v)
        s_ref = deploy_score(yv, ens9)

        # ---- rC 타깃 (설치 증분 전부 직교화)
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
        print(f"\n[val {v}] ENS-9 동결 기준 {s_ref:.2f} · rC 준비 [{time.time()-t1:.0f}s]"
              f" · pkg증분(fit) std {np.std(p_pkg_fit - p_bis_fit):.4f}")

        # ---- 디자인 블록
        D_a1_full, n1 = build_design(df, *is_factors(X, fit))
        D_a1 = D_a1_full[:, :n1]
        D_dn = denom_design(df, B_dn, fit)
        D_main = np.concatenate([D_a1, orth(D_a1, D_dn, fit)], axis=1)

        arms = [("안1·C", D_a1), ("안1+dn·C(주셀)", D_main)]
        if args.donut:
            da_f = DN.fit_da(df[season < v])
            vsc = donut_scalar(df, dn_table, da_f, fit)[:, None]
            D_full = np.concatenate([D_main, orth(D_main, vsc, fit)], axis=1)
            arms.append(("안1+dn+donut·C", D_full))
        for sd_ in range(2):
            D_a1p = build_design(df, *is_factors(X_pl[sd_], fit))[0][:, :n1]
            perm = rng_perm.permutation(len(df))
            D_dnp = denom_design(df, B_dn, fit, perm=perm)
            D_plc = np.concatenate([D_a1p, orth(D_a1p, D_dnp, fit)], axis=1)
            if args.donut:
                rngp = np.random.default_rng(31 + sd_)
                pperm = rngp.permutation(len(pids_dn))
                pmap = {pids_dn[i]: pids_dn[pperm[i]] for i in range(len(pids_dn))}
                vscp = donut_scalar(df, dn_table, da_f, fit, pmap=pmap)[:, None]
                D_plc = np.concatenate([D_plc, orth(D_plc, vscp, fit)], axis=1)
            arms.append((f"위약s{sd_}", D_plc))

        for name, Dm in arms:
            corr2 = ridge_corr(Dm, fit, val, rC, LAM)
            line = dict(val=v, arm=name)
            for w in WGRID:
                line[f"w{w:g}"] = round(float(
                    deploy_score(yv, ens9 + w * corr2) - s_ref), 3)
            res.append(line)
            print(f"  {name:16s}  " + "  ".join(f"w{w:g}: {line[f'w{w:g}']:+7.2f}" for w in WGRID))

    t = pd.DataFrame(res)
    core = t[(t.arm == "안1+dn·C(주셀)") & t.val.isin([2024, 2023, 2022])]
    if len(core) >= 3:
        for w in WGRID:
            d = core[f"w{w:g}"]
            print(f"\n[주셀 w{w:g}] 평균 {d.mean():+.2f} · SD {d.std():.2f} · "
                  f"통과 {d.mean() > 0 and d.mean() > d.std()}")
        v21 = t[(t.arm == "안1+dn·C(주셀)") & (t.val == 2021)]
        if len(v21):
            print(f"[감사 V21] w0.5 {v21['w0.5'].iloc[0]:+.2f}")
    (OUT_DIR / "pkg_gate.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR/'pkg_gate.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
