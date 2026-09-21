# -*- coding: utf-8 -*-
"""이중조사 — 제3 캐리어: is4×맥락 잔차 ridge 보정기 (C-C 스펙 구현).

    python sweep/is_corrector.py --vals 2024            # V24 스크린 (통과선 Δ≥+1.5 + 위약 분리)
    python sweep/is_corrector.py --vals 2024,2023,2022,2021

## 사전 등록 주 셀 (C-C 권고안)
안 2(is4 전면⊗맥락 24항) × 타깃 B(june 잔차 − 0.5·corr_tm : 순차 직교화, 배포식과 동형)
× λ=1000 × w2=0.5. 주 셀이 죽고 다른 셀만 살면 기각(격자 선택 낙관 방지).

- 안 1 = cxp 최소 이식(C12⊗is_delta + is_delta + hand·is_delta + 절편 = 15항).
- 안 2 = 안 1 + (is_smc, is_sharec, is_lognz)⊗(1, count/11, hand) 9항 = 24항.
- 타깃 A(june 잔차 그대로)는 비용 0의 대조 arm — Δ_A ≫ Δ_B면 TM 이중 운반 진단.
- 위약 P1(주) = is4 base 경로 교환(IS.base_for shuffle_seed) 2시드 · P2(보조) = is인자 행 셔플.
- 판정: deploy_score(ens_full + w2·corr2) − deploy_score(ens_full), w2∈{0.25,0.5}(1.0은 진단).
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
import tm_member as TMM           # noqa: E402
from eb_carrier import get_members, deploy_score, ENS7_W, TM_W  # noqa: E402

OUT_DIR = HERE.parent / "results" / "dual"
LAM = 1000.0
WGRID = (0.25, 0.5, 1.0)


def is_factors(X, fit):
    """is4 인자: 고정 리터럴 중심화(sm/share −0.5) + lognz(fit 통계 z) — fit 상수 반환."""
    pd_ = np.nan_to_num(X["is_delta"].to_numpy(dtype=float), nan=0.0)
    smc = np.nan_to_num(X["is_sm"].to_numpy(dtype=float), nan=0.5) - 0.5
    shc = np.nan_to_num(X["is_share"].to_numpy(dtype=float), nan=0.0) - 0.5
    ln = np.nan_to_num(X["is_logn"].to_numpy(dtype=float), nan=0.0)
    mu, sd = float(ln[fit].mean()), float(ln[fit].std() or 1.0)
    lnz = (ln - mu) / sd
    return pd_, smc, shc, lnz


def build_design(df, pd_, smc, shc, lnz):
    cc = (df["balls_before"].to_numpy() * 3 + df["strikes_before"].to_numpy()).astype(int)
    C = np.zeros((len(df), 12))
    C[np.arange(len(df)), cc] = 1.0
    hand = (df["pitcher_hand"].to_numpy() == df["batter_hand"].to_numpy()).astype(float)
    cn = cc / 11.0
    cols = [C[:, j] * pd_ for j in range(12)] + [pd_, hand * pd_]          # 안1 (절편 제외 14)
    n1 = len(cols)
    for f in (smc, shc, lnz):
        for ctx in (np.ones(len(df)), cn, hand):
            cols.append(f * ctx)                                            # +9 → 안2
    D = np.stack(cols, axis=1)
    return D, n1


def ridge_corr(D, fit, val, target_fit, lam):
    A = D[fit]
    sd = A.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    As = A / sd
    As = np.concatenate([As, np.ones((As.shape[0], 1))], axis=1)
    G = As.T @ As + lam * np.eye(As.shape[1])
    c = np.linalg.solve(G, As.T @ target_fit)
    Dv = np.concatenate([D[val] / sd, np.ones((int(val.sum()), 1))], axis=1)
    return Dv @ c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024")
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
            .reset_index(drop=True) for sd_ in range(2)}       # P1 경로 교환 2시드
    y = df[rd.TARGET].to_numpy().astype(float)
    season = ssn
    prof = TMM.load_profile()

    res = []
    for v in [int(s) for s in args.vals.split(",")]:
        fit, val = (season == v - 1), (season == v)
        P, corr, yv = get_members(df, X, y, season, v)
        ens_full = sum(P[m] * w for m, w in ENS7_W.items()) + TM_W * corr
        s_ref = deploy_score(yv, ens_full)
        t1 = time.time()
        residA = TMM.june_fit_resid(X, y, fit)                 # 타깃 A (fit 행 순서)
        Df_tm, covf = TMM.design(df, X, prof, fit)
        A_tm = Df_tm[fit & covf]
        G = A_tm.T @ A_tm + 1000.0 * np.eye(A_tm.shape[1])
        c_tm = np.linalg.solve(G, A_tm.T @ residA[covf[fit]])
        corr_tm_fit = Df_tm[fit] @ c_tm
        corr_tm_fit[~covf[fit]] = 0.0
        residB = residA - TM_W * corr_tm_fit                   # 타깃 B (순차 직교화)
        print(f"\n[val {v}] ENS-7 기준 {s_ref:.2f} · 타깃 준비 [{time.time()-t1:.0f}s]")

        pd_, smc, shc, lnz = is_factors(X, fit)
        D, n1 = build_design(df, pd_, smc, shc, lnz)
        rng = np.random.default_rng(3)
        perm = rng.permutation(len(df))

        arms = [("안1·B", D[:, :n1], residB), ("안2·B(주셀)", D, residB), ("안2·A", D, residA)]
        for sd_ in range(2):
            pf = is_factors(X_pl[sd_], fit)
            Dp, _ = build_design(df, *pf)
            arms.append((f"안2·B·위약P1s{sd_}", Dp, residB))
        arms.append(("안2·B·위약P2", D[perm], residB))

        for name, Dm, tgt in arms:
            corr2 = ridge_corr(Dm, fit, val, tgt, LAM)
            line = dict(val=v, arm=name)
            for w in WGRID:
                line[f"w{w:g}"] = round(float(
                    deploy_score(yv, ens_full + w * corr2) - s_ref), 3)
            res.append(line)
            print(f"  {name:16s}  " + "  ".join(f"w{w:g}: {line[f'w{w:g}']:+7.2f}" for w in WGRID))

    t = pd.DataFrame(res)
    core = t[(t.arm == "안2·B(주셀)") & t.val.isin([2024, 2023, 2022])]
    if len(core) >= 3:
        for w in (0.25, 0.5):
            d = core[f"w{w:g}"]
            print(f"\n[주셀 w{w:g}] 평균 {d.mean():+.2f} · SD {d.std():.2f} · "
                  f"통과 {d.mean() > 0 and d.mean() > d.std()}")
    (OUT_DIR / "is_corrector_gate.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR/'is_corrector_gate.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
