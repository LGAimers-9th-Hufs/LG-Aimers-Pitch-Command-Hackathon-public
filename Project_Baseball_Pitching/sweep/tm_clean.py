# -*- coding: utf-8 -*-
"""반나절 1 — TM 증거 정화 (V24 폴드 최소판): **as-of 매칭 + 커버 보존 위약**.

    python sweep/tm_clean.py

## 무엇을 검증하나 (Codex 2차 심사 지적 A1·A2)

현행 `matches.csv`는 2019~2024 **전체**로 매칭됐다 → V24 폴드가 2024 정보로 매칭된 프로필을 쓴다.
2025에는 2025 TM이 없으므로 이는 배포 동형이 아니다. 여기서는:

1. **≤2023 데이터만으로 매칭을 재생성**(as-of) — 배포와 같은 조건.
2. career_rank 프로필도 ≤2023에서 재구축, (pid, 2023) 시점.
3. 보정기 파이프라인(V24 폴드)을 ①전역 매칭 ②as-of 매칭 ③**커버 보존 위약**(매칭된 투수 집합 안에서만
   프로필 재배정 — 커버 마스크 동일) 5시드로 비교.

판정: as-of 이득이 전역 이득과 비슷하고 위약과 분리되면 TM 성분은 배포 동형 검증 통과.
as-of에서 소멸하면 → 현 제출물의 TM 기여를 '미확정'으로 기록(제출물은 유지 — LB가 패키지를 검증).
"""
from __future__ import annotations
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
from eb_carrier import get_members, deploy_score, ENS7_W, TM_W   # noqa: E402

OUT_DIR = HERE.parent / "results" / "tm_clean"
CUT = 2023                         # V24 폴드의 as-of 경계
HAND = {2: "Right", 1: "Left"}     # 손 극성은 전역 검증에서 확정(2=우투) — 재검 불요


def asof_matches():
    """≤CUT 데이터만으로 커리어 Hungarian 매칭 → Tier1 근사 집합."""
    print(f">> as-of 매칭 재생성 (≤{CUT})")
    df = rd.load_train()
    df = df[df[rd.SEASON] <= CUT]
    tm = TM.load_trackman()
    tm = tm[tm[rd.SEASON] <= CUT]
    t = time.time()
    sig_tr = TM.season_signatures_train(TM.add_appearances(df))
    sig_tm = TM.season_signatures_tm(tm)
    print(f"   시그니처 [{time.time()-t:.0f}s]")
    ents_tr = TM._entity_tables(sig_tr)
    ents_tm = TM._entity_tables(sig_tm)
    C, n_co = TM.build_cost(sig_tr, sig_tm, HAND, ents=(ents_tr, ents_tm),
                            seasons=range(2019, CUT + 1))
    rows, cols = TM.assign(C)
    mr, mc = TM._margins(C, rows, cols)
    e_tr = ents_tr[0]
    e_tm = ents_tm[0]
    co = n_co[rows, cols]
    cost = C[rows, cols]
    ok = (co > 0) & (cost < TM.BIG / 2)
    thr = np.quantile(mr[ok], 0.25)
    tier1 = ok & (mr >= thr)
    m = pd.DataFrame({"pitcher_id": e_tr[rows[tier1]], "tm_id": e_tm[cols[tier1]]})
    print(f"   as-of Tier1 근사 {len(m)}쌍 (전역 Tier1은 407)")
    # 전역 매칭과의 합치율
    g = pd.read_csv(HERE.parent / "results" / "trackman" / "matches.csv")
    g1 = g[g.tier == 1][["pitcher_id", "tm_id"]]
    mg = m.merge(g1, on="pitcher_id", suffixes=("_asof", "_glob"))
    agree = float((mg.tm_id_asof == mg.tm_id_glob).mean()) if len(mg) else float("nan")
    print(f"   전역 Tier1과 공통 투수 {len(mg)}명 · **매칭 일치율 {agree:.1%}**")
    return m, tm


def career_rank_profile(m, tm, cuts=(CUT - 1, CUT)):
    """≤cut 트랙맨으로 (pid, cut) 시점 career_rank 6피처.
    fit 행(2023)은 (pid,2022)를, val 행(2024)은 (pid,2023)을 조회하므로 두 시점 다 만든다."""
    pid_of = dict(zip(m["tm_id"].astype(int), m["pitcher_id"].astype(int)))
    tm_sel = tm[tm["pitcher_trackman_id"].isin(pid_of)]
    cells, rows_ = TM._measure_cells(tm_sel)
    import tm_season_profile as TSP
    prof = {}
    for cut in cuts:
        f = TSP._features_from(cells[cells[rd.SEASON] <= cut], rows_[rows_[rd.SEASON] <= cut])
        f = f.dropna(how="all", subset=TSP.FEATS)
        rk = f[TSP.FEATS].rank(pct=True, method="average")
        for t in f.index:
            prof[(pid_of[int(t)], cut)] = [float(v) for v in rk.loc[t]]
    return prof


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    m_asof, tm_trunc = asof_matches()
    prof_asof = career_rank_profile(m_asof, tm_trunc)
    print(f"   as-of 프로필 {len(prof_asof)}투수")

    print(">> V24 폴드 보정기 비교 (K_IS=100)")
    L.K_IS = 100.0
    df = rd.load_train()
    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy())
    X = L.build_features(df, base=(nb, kb)).reset_index(drop=True)
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()
    v = 2024
    fit, val = (season == v - 1), (season == v)
    P, corr_glob, yv = get_members(df, X, y, season, v)
    base_no = sum(P[m_] * w for m_, w in ENS7_W.items())
    s_no = deploy_score(yv, base_no)                       # 보정 없음
    s_glob = deploy_score(yv, base_no + TM_W * corr_glob)  # 전역 매칭 (현 제출물)
    print(f"   보정없음 {s_no:.2f} · 전역매칭 Δ {s_glob-s_no:+.2f}")

    resid = TMM.june_fit_resid(X, y, fit)

    def corr_from(prof):
        Df, covf = TMM.design(df, X, prof, fit)
        Dv, covv = TMM.design(df, X, prof, val)
        A = Df[fit & covf]
        if (fit & covf).sum() < 5000:
            return None, 0.0
        G = A.T @ A + 1000.0 * np.eye(A.shape[1])
        c = np.linalg.solve(G, A.T @ resid[covf[fit]])
        cr = Dv @ c
        cr[~covv] = 0.0
        return cr[val], float(covv[val].mean())

    cr_asof, cov_a = corr_from(prof_asof)
    s_asof = deploy_score(yv, base_no + TM_W * cr_asof)
    print(f"   **as-of 매칭 Δ {s_asof-s_no:+.2f}** (커버 {cov_a:.2f})")

    # 커버 보존 위약: 매칭된 투수 집합 안에서 프로필만 재배정 (커버 마스크 동일)
    pids = sorted({p for (p, s_) in prof_asof})
    rng = np.random.default_rng(0)
    pls = []
    for sd in range(5):
        perm = rng.permutation(len(pids))
        pmap = {pids[i]: pids[perm[i]] for i in range(len(pids))}   # 경로 통째 교환
        prof_pl = {}
        for (p, s_), v in prof_asof.items():
            prof_pl[(p, s_)] = prof_asof.get((pmap[p], s_), v)
        cr_pl, _ = corr_from(prof_pl)
        if cr_pl is None:
            continue
        pls.append(deploy_score(yv, base_no + TM_W * cr_pl) - s_no)
    print(f"   위약(커버보존, 5시드) Δ 평균 {np.mean(pls):+.2f} (개별 "
          + " ".join(f"{d:+.2f}" for d in pls) + ")")

    dinfo = (s_asof - s_no) - np.mean(pls)
    print(f"\n[판정] as-of Δ {s_asof-s_no:+.2f} vs 전역 Δ {s_glob-s_no:+.2f} · "
          f"위약 {np.mean(pls):+.2f} · **Δinfo(as-of − 위약) {dinfo:+.2f}**")
    if s_asof - s_no > 0 and dinfo > 0:
        print("  ⇒ TM 성분이 배포 동형 조건에서 생존 — 현 제출물의 TM 기여 신뢰 가능.")
    else:
        print("  ⇒ as-of 조건에서 소멸 — 현 제출물의 TM 기여는 '미확정'으로 기록(제출물은 유지).")
    print(f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
