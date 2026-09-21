# -*- coding: utf-8 -*-
"""W1-A 다운스트림 — 매칭 v2의 physmix 채널 A/B (하드 정밀화 vs posterior 소프트 확장).

    python sweep/tm_v2_refit.py --vals 2024,2023,2022,2021

churn 0.5%가 말하는 것: v2 Tier1(281) ≈ v1 Tier1(407)의 **부분집합**(저 margin 126쌍 제거).
따라서 이 A/B는 정확히 "저신뢰 쌍이 physmix를 돕는가 해치는가"를 잰다(D-17 −12 반전의 재검).
posterior 소프트는 제거분·Tier2/3을 posterior 가중 평균으로 되살리는 세 번째 팔.

판정: ENS-9에서 corr_pm만 교체(다른 성분 동결) → Δ(동결 배포 캘리) vs 캐시 corr_pm.
- v1 재현 팔 = 파리티(≈0 기대, 하네스 검증)
- 채택 조건: 평균 ≥ +2 & >SD & V21 비붕괴 (Codex 다운스트림 kill 기준)
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
from eb_carrier import (get_members9, deploy_score,   # noqa: E402
                        ENS9_W, TM_W, W_PM)
from is_corrector import ridge_corr   # noqa: E402
from tm_physmix import build_group_profile, row_z   # noqa: E402

OUT_DIR = HERE.parent / "results" / "ens11"
TM_DIR = HERE.parent / "results" / "trackman"
LAM = 1000.0
POST_MIN = 0.10       # posterior 가중 하한 (그 미만 후보는 무시)


def soft_profile(post, prof_tm_by_id):
    """posterior 가중 프로필: P_soft(pid,qs) = Σj w_j·P(tm_j,qs) (가중 재정규화, avail=합집합)."""
    out = {}
    keys_by_tm = {}
    for (tmid, qs) in prof_tm_by_id:
        keys_by_tm.setdefault(tmid, []).append(qs)
    for pid_s, cand in post.items():
        pid = int(pid_s)
        cw = {int(k): v for k, v in cand.items() if v >= POST_MIN}
        if not cw:
            continue
        qs_all = set()
        for tmid in cw:
            qs_all.update(keys_by_tm.get(tmid, []))
        for qs in qs_all:
            num = np.zeros((3, 4))
            den = np.zeros(3)
            for tmid, w in cw.items():
                e = prof_tm_by_id.get((tmid, qs))
                if e is None:
                    continue
                P, av = e
                num += w * P * av[:, None]
                den += w * av
            av2 = den > 0.25          # 유효 가중 합 하한
            if not av2.any():
                continue
            P2 = np.zeros((3, 4))
            P2[av2] = num[av2] / den[av2, None]
            out[(pid, int(qs))] = (P2, av2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024,2023,2022,2021")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(">> 매핑 3종 구성")
    import json
    v1 = pd.read_csv(TM_DIR / "matches.csv")
    v2 = pd.read_csv(TM_DIR / "matches_v2.csv")
    post = json.loads((TM_DIR / "posterior.json").read_text(encoding="utf-8"))
    mm_v1 = dict(zip(v1[v1.tier == 1].tm_id, v1[v1.tier == 1].pitcher_id))
    mm_v2 = dict(zip(v2[v2.tier == 1].tm_id, v2[v2.tier == 1].pitcher_id))
    print(f"   v1 Tier1 {len(mm_v1)} · v2 Tier1 {len(mm_v2)} · posterior {len(post)}")

    profs = {"v1(파리티)": build_group_profile(mmap=mm_v1),
             "v2-hard": build_group_profile(mmap=mm_v2)}
    ident = build_group_profile(mmap={t: t for t in v1.tm_id})       # tm_id 자기 자신 키
    profs["v2-soft"] = soft_profile(post, ident)
    for k, p in profs.items():
        print(f"   {k:12s} 프로필 {len(p):,} 엔트리")

    print(">> train 준비 (K_IS=100)")
    L.K_IS = 100.0
    df = rd.load_train()
    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy())
    X = L.build_features(df, base=(nb, kb)).reset_index(drop=True)
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()
    cc = (df["balls_before"].to_numpy() * 3 + df["strikes_before"].to_numpy()).astype(int)
    cn = cc / 11.0
    hand = (df["pitcher_hand"].to_numpy() == df["batter_hand"].to_numpy()).astype(float)
    prof_tm = TMM.load_profile()

    D_of, cov_of = {}, {}
    for k, p in profs.items():
        Z, cov = row_z(df, p)
        cols = []
        for mi in range(4):
            for ctx in (np.ones(len(df)), cn, hand):
                cols.append(Z[:, mi] * ctx)
        D_of[k] = np.stack(cols, axis=1)
        cov_of[k] = cov
        print(f"   {k:12s} 행 커버 {cov.mean():.3f}")

    res = []
    for v in [int(s) for s in args.vals.split(",")]:
        fit, val = (season == v - 1), (season == v)
        P, corr, corr_pm_v, yv = get_members9(v)
        others = sum(P[m] * w for m, w in ENS9_W.items())
        base = others + TM_W * corr
        s_ref = deploy_score(yv, base + W_PM * corr_pm_v)

        residA = TMM.june_fit_resid(X, y, fit)
        Df_tm, covf_tm = TMM.design(df, X, prof_tm, fit)
        A_tm = Df_tm[fit & covf_tm]
        G = A_tm.T @ A_tm + 1000.0 * np.eye(A_tm.shape[1])
        c_tm = np.linalg.solve(G, A_tm.T @ residA[covf_tm[fit]])
        corr_tm_fit = Df_tm[fit] @ c_tm
        corr_tm_fit[~covf_tm[fit]] = 0.0
        residB = residA - TM_W * corr_tm_fit
        print(f"\n[val {v}] ENS-9 동결 기준 {s_ref:.2f}")
        for k in profs:
            corr_pm2 = ridge_corr(D_of[k], fit, val, residB, LAM)
            corr_pm2[~cov_of[k][val]] = 0.0
            d = deploy_score(yv, base + W_PM * corr_pm2) - s_ref
            res.append(dict(val=v, arm=k, cover=round(float(cov_of[k][val].mean()), 3),
                            d=round(float(d), 3)))
            print(f"  {k:12s} cover {res[-1]['cover']:.3f}  Δ {d:+7.2f}")

    t = pd.DataFrame(res)
    print("\n" + t.pivot_table(index="arm", columns="val", values="d")
          .to_string(float_format=lambda x: f"{x:+.2f}"))
    (OUT_DIR / "v2_refit_gate.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR/'v2_refit_gate.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
