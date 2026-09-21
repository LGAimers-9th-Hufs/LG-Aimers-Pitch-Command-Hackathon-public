# -*- coding: utf-8 -*-
"""ENS-4 최종 가중·캘리 확정 — 7멤버(K=100 + all_raw) 그리디 + 선택 낙관 검사.

    python sweep/ens4_weights.py --k 100

## 왜 다시 잡는가

ENS-3는 ENS-2에서 **물려받은 가중**을 썼다. 그런데 is4가 들어가며 멤버 단독이 전부 +85~151 올랐고
(nn_lin 644→795가 최대) **상대 순위와 불일치 구조가 바뀌었다.** `ens3_weights.py` 실측:

| | val 2024(선택 폴드) | val 2023(검증 폴드) |
|---|---|---|
| 동결 가중 | 862.03 | −280.68 |
| 그리디 재최적화 | 866.99 (**+4.96**) | −272.21 (**+8.46**) |

**고르지 않은 폴드에서 이득이 더 크다 ⇒ 선택 낙관이 아니다.** 앙상블형이라 실현율 0.88.
여기서는 K=100과 `all_raw` 2종까지 포함한 **최종 멤버 집합**에서 다시 잡는다.

## 속도

이전 그리디는 매 단계 캘리 격자를 재적합해 1,524초가 걸렸다. 여기서는 **닫힌형 affine 적합**
(최소제곱 1차 회귀)으로 대체한다 — 격자와 사실상 같은 해를 주면서 비용이 무시할 수준이다.
최종 구성에만 격자 재적합을 한 번 돌려 빌더에 넣을 상수를 확정한다.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from collections import Counter
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
import phase2_ens_check as EC     # noqa: E402
import season_centering as SC     # noqa: E402

OUT_DIR = HERE.parent / "results" / "ens3"


def affine_score(y, p):
    """닫힌형 affine 캘리 후 점수 — 그리디 내부용 고속 채점."""
    x = np.asarray(p, dtype=float) - 0.5
    A = np.stack([x, np.ones_like(x)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y - 0.5, rcond=None)
    q = np.clip(0.5 + coef[0] * x + coef[1], 1e-6, 1 - 1e-6)
    return L.score(y, q)


def grid_calib(y, p):
    best = (None, -1e18)
    for sl in np.arange(0.55, 1.301, 0.01):    # ⚠ val 2023 최적 slope = 0.60. 0.85 시작은 경계 절단.
        q = 0.5 + sl * (np.asarray(p) - 0.5)
        for sh in np.arange(-0.03, 0.0201, 0.0025):
            v = L.score(y, np.clip(q + sh, 1e-6, 1 - 1e-6))
            if v > best[1]:
                best = ((round(float(sl), 3), round(float(sh), 4)), v)
    return best


def build_pool(X, y, season, v, seasons):
    """ENS-4 멤버 풀: 시즌 5종 + all_raw 2종(전 시즌 원시)."""
    fit, val = (season == v - 1), (season == v)
    P = EC.train_members(X, y, fit, val)
    arms = {a["name"]: a for a in SC.arms_for(v, seasons)}
    a = dict(arms["all_raw"])
    for nm, spec in (("allraw_l15", L.ORIGINAL["l15"]), ("allraw_l7", L.ORIGINAL["l7"])):
        t = time.time()
        fit_all = np.isin(season, a["fit"])
        m = L.train_lgbm(X[fit_all], y[fit_all], spec)
        P[nm] = np.clip(m.predict(X[val], num_threads=6), 0, 1)
        print(f"    {nm:14s} {time.time()-t:5.1f}s  단독 cal="
              f"{L.score(y[val], L.calibrate(P[nm])):8.2f}")
    return P, y[val]


def greedy(P, yv, n_iter=40, init=None):
    names = list(P)
    picks = list(init or [])
    cur = sum(P[n] for n in picks) if picks else np.zeros(len(yv))
    for _ in range(n_iter - len(picks)):
        best, bn = -1e18, None
        for nm in names:
            s = affine_score(yv, (cur + P[nm]) / (len(picks) + 1))
            if s > best:
                best, bn = s, nm
        picks.append(bn)
        cur = cur + P[bn]
    return {k: c / len(picks) for k, c in Counter(picks).items()}


def blend(P, w):
    return sum(P[k] * q for k, q in w.items() if k in P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, default=100.0)
    ap.add_argument("--vals", default="2024,2023")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f">> train.csv 로드 (K_IS = {args.k:g})")
    L.K_IS = args.k
    df = rd.load_train()
    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy())
    X = L.build_features(df, base=(nb, kb))
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()
    seasons = sorted(int(s) for s in np.unique(season))
    vals = [int(s) for s in args.vals.split(",")]

    # 후보 가중 3종을 같은 예측 위에서 비교한다
    W_FROZEN = {n: w for n, _, w, _ in EC.MEMBERS}                       # ENS-3 그대로
    W_PLUS = {**{k: v * 0.9 for k, v in W_FROZEN.items()},
              "allraw_l15": 0.05, "allraw_l7": 0.05}                     # ENS-4 보수본

    P, YV = {}, {}
    for v in vals:
        print(f"\n[val {v}] 멤버 풀 학습")
        P[v], YV[v] = build_pool(X, y, season, v, seasons)

    main_v = vals[0]
    W_GREEDY = greedy(P[main_v], YV[main_v], init=["june_l15", "nn_lin"])
    cands = {"ENS-3 동결": W_FROZEN, "ENS-4 보수(+allraw 0.10)": W_PLUS,
             "ENS-4 그리디": W_GREEDY}

    print("\n" + "=" * 88)
    rows = []
    for nm, w in cands.items():
        line = {"cfg": nm}
        for v in vals:
            (sl, sh), s = grid_calib(YV[v], blend(P[v], w))
            line[f"val{v}"] = s
            line[f"cal{v}"] = f"({sl}, {sh})"
        rows.append(line)
        print(f"  {nm:24s} " + "  ".join(
            f"val{v} {line[f'val{v}']:8.2f} {line[f'cal{v}']}" for v in vals))
    print("\n  [그리디 가중]")
    for k, q in sorted(W_GREEDY.items(), key=lambda kv: -kv[1]):
        print(f"    {k:14s} {W_PLUS.get(k, 0):.4f} → {q:.4f}")

    base = rows[0]
    print("\n[ENS-3 동결 대비]")
    for r in rows[1:]:
        d = {v: r[f"val{v}"] - base[f"val{v}"] for v in vals}
        worst = min(d.values())
        print(f"  {r['cfg']:24s} " + " · ".join(f"val{v} {d[v]:+7.2f}" for v in vals)
              + f"   최악 {worst:+7.2f}  → 2025 기대 {worst*0.879:+6.1f}")

    (OUT_DIR / "ens4_weights.json").write_text(
        json.dumps({"frozen": W_FROZEN, "plus": W_PLUS, "greedy": W_GREEDY, "rows": rows},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR/'ens4_weights.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
