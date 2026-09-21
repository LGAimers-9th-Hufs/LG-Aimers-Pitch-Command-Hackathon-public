# -*- coding: utf-8 -*-
"""ENS-3(LB 953.06) 위에서 **가중·캘리 재최적화**가 값을 하는가 — 앙상블형 이득 후보.

    python sweep/ens3_weights.py --vals 2024,2023

## 왜 이걸 재는가

3점으로 확정된 전이 계수는 **이득의 종류에 따라 다르다**([[lb-verified-scores]]):

| 이득의 성격 | 실현율 |
|---|---|
| 앙상블 다양성(분산 감소) | **87.9%** |
| 새 피처 신호 | 48.9% |

⇒ **같은 로컬 +1점이라도 앙상블형이 피처형의 1.8배로 전이된다.** 마지막 슬롯 후보 중
가중·캘리 재최적화는 순수 앙상블형이라 효율이 가장 좋다.

현재 ENS-3는 **ENS-2에서 물려받은 가중**(.300/.275/.225/.175/.025)과 캘리(1.02, −0.0075)를 쓴다.
그런데 is4가 들어가며 멤버 단독 점수가 전부 +85~151 올랐고(nn_lin 644→795가 최대)
**멤버 간 상대 순위와 불일치 구조가 바뀌었다.** 물려받은 가중이 더 이상 최적이 아닐 수 있다.

## 선택 낙관 측정 (이게 이 스크립트의 핵심)

가중을 val 2024에서 고르면 그 이득은 낙관을 포함한다. 그래서 **val 2024에서 고른 가중을
val 2023에 그대로 적용**해 이득이 살아남는지 본다. 살아남는 몫만 진짜다.
슬롯이 1회뿐이므로 이 확인 없이는 채택하지 않는다.
"""
from __future__ import annotations
import argparse
import json
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
import phase2_ens_check as EC     # noqa: E402
from season_centering import best_cal   # noqa: E402

OUT_DIR = HERE.parent / "results" / "ens3"


def greedy(preds: dict, yv, n_iter=40, init=None):
    """Caruana 그리디(복원 추출). 반환 (가중 dict, 점수)."""
    names = list(preds)
    picks = list(init or [])
    cur = np.zeros(len(yv))
    for nm in picks:
        cur = cur + preds[nm]
    for _ in range(n_iter - len(picks)):
        best, bn = -1e18, None
        for nm in names:
            s = best_cal(yv, (cur + preds[nm]) / (len(picks) + 1))[0]
            if s > best:
                best, bn = s, nm
        picks.append(bn)
        cur = cur + preds[bn]
    from collections import Counter
    w = {k: v / len(picks) for k, v in Counter(picks).items()}
    return w, best_cal(yv, sum(preds[k] * v for k, v in w.items()))[0]


def blend(preds, w):
    return sum(preds[k] * v for k, v in w.items() if k in preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024,2023")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(">> train.csv 로드")
    df = rd.load_train()
    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy())
    X = L.build_features(df, base=(nb, kb))
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()
    frozen = {n: w for n, _, w, _ in EC.MEMBERS}

    P, Y = {}, {}
    for v in [int(s) for s in args.vals.split(",")]:
        fit, val = (season == v - 1), (season == v)
        print(f"\n[val {v}] 멤버 5종 학습 (57피처)  fit {fit.sum():,} → val {val.sum():,}")
        P[v] = EC.train_members(X, y, fit, val)
        Y[v] = y[val]

    rows = []
    vals = list(P)
    main_v = vals[0]                      # val 2024 = 주 게이트(D-29 ①)
    print("\n" + "=" * 80)
    for v in vals:
        b_frozen = blend(P[v], frozen)
        s_frozen_fixed = L.score(Y[v], L.calibrate(b_frozen, **EC.ENS2_CAL))
        s_frozen_best, arg = best_cal(Y[v], b_frozen)
        print(f"[val {v}] 동결 가중: 고정캘리 {s_frozen_fixed:8.2f} · 재적합 {s_frozen_best:8.2f} {arg}")
        rows.append(dict(val=v, cfg="frozen", fixed=s_frozen_fixed, best=s_frozen_best))

    w_new, s_new = greedy(P[main_v], Y[main_v], init=["june_l15", "nn_lin"])
    print(f"\n[그리디 재최적화 — val {main_v}에서 선택]")
    for k, q in sorted(w_new.items(), key=lambda kv: -kv[1]):
        print(f"    {k:14s} {frozen.get(k, 0):.3f} → {q:.3f}")
    base_main = [r for r in rows if r["val"] == main_v][0]["best"]
    print(f"    val {main_v}: {base_main:.2f} → {s_new:.2f}  ({s_new - base_main:+.2f})")

    print(f"\n[선택 낙관 검사 — 위 가중을 다른 폴드에 그대로 적용]")
    surv = []
    for v in vals[1:]:
        b = blend(P[v], w_new)
        s = best_cal(Y[v], b)[0]
        b0 = [r for r in rows if r["val"] == v][0]["best"]
        print(f"    val {v}: {b0:.2f} → {s:.2f}  ({s - b0:+.2f})")
        surv.append(s - b0)
        rows.append(dict(val=v, cfg="greedy_from_main", best=s, delta=s - b0))
    rows.append(dict(val=main_v, cfg="greedy_from_main", best=s_new, delta=s_new - base_main))

    gain_main = s_new - base_main
    gain_other = float(np.mean(surv)) if surv else float("nan")
    print("\n[판정]")
    print(f"  주 폴드 이득 {gain_main:+.2f} · 다른 폴드에서 살아남은 몫 {gain_other:+.2f}")
    print(f"  → 앙상블형 실현율 0.879를 적용한 2025 기대 = {min(gain_main, gain_other)*0.879:+.1f}점")
    if gain_other <= 0:
        print("  ⚠ 다른 폴드에서 이득이 사라졌다 = 순수 선택 낙관. **채택하지 말 것.**")

    (OUT_DIR / "weights.json").write_text(
        json.dumps({"frozen": frozen, "greedy": w_new, "rows": rows},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
