# -*- coding: utf-8 -*-
"""Program F — 멤버 **선택 절차** 교정. 후보를 넓히는 게 아니라 덜 과적합하게 고른다.

    python sweep/ens5_select.py --k 100

## 무엇이 잘못됐나

`ens5_pool.py`는 후보 19종을 만들고 **val 2024 단독**으로 그리디를 돌렸다:

| | val 2024(선택) | val 2023(검증) |
|---|---|---|
| ENS-4 동결 | 877.82 | −128.46 |
| ENS-5 그리디 | 879.80 (**+1.98**) | −135.07 (**−6.61**) |

고른 폴드에서 +2, 안 고른 폴드에서 −6.6 ⇒ **순수 선택 낙관**. 후보가 나쁜 게 아니라
**절차가 한 폴드에 과적합**했다(ENS-4의 7후보 그리디는 검증 폴드에서 +52로 살아남았다).

## 교정

- 그리디 채점을 **val 2024 단독 → (val 2024 + val 2023) 평균**으로.
  두 레짐(안정 2024 / 파단 2023)에서 동시에 좋아야 뽑히므로 한 폴드 잡음을 좇지 못한다.
- **val 2022는 선택에 전혀 쓰지 않고 검증 폴드로 남긴다.**
- **채택 조건: val 2022에서 양수.** 아니면 ENS-4를 유지한다.

저장된 예측(`results/ens5/preds_val{2024,2023}.npz`)을 재사용하고 val 2022만 새로 만든다.
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
import ens5_pool as EP            # noqa: E402
from ens4_weights import affine_score, grid_calib   # noqa: E402

OUT_DIR = HERE.parent / "results" / "ens5"
SEL = (2024, 2023)        # 선택에 쓰는 폴드
HOLD = 2022               # 선택에 **절대** 쓰지 않는 검증 폴드


def greedy_multi(P, YV, folds, n_iter=40, init=None, mode="mean", ref=None):
    """다폴드 그리디.

    ⚠ mode="mean"(폴드 점수 평균)은 **스케일 지배로 깨진다** — 실측: 파단 연도(val 2023)의 점수
    가동 범위가 ±600으로 안정 연도(±10)보다 훨씬 넓어, 평균 최적화가 그 폴드 하나만 좇는다
    (lin_f60_0에 w=0.95를 몰아주고 val2024 −100 / val2023 +643 / val2022 −565로 폭주했다).

    mode="maximin"은 각 폴드에서 **기준 구성(ref) 대비 개선분**을 재고 **최악 폴드의 개선분**을
    최대화한다 — 한 폴드를 희생하는 선택이 원천 차단된다. ref는 {fold: 기준 점수} dict.
    """
    names = sorted(set.intersection(*[set(P[v]) for v in folds]))
    picks = list(init or [])
    cur = {v: (sum(P[v][n] for n in picks) if picks else np.zeros(len(YV[v]))) for v in folds}
    for _ in range(n_iter - len(picks)):
        best, bn = -1e18, None
        for nm in names:
            ss = {v: affine_score(YV[v], (cur[v] + P[v][nm]) / (len(picks) + 1)) for v in folds}
            if mode == "maximin":
                s = min(ss[v] - ref[v] for v in folds)
            else:
                s = float(np.mean(list(ss.values())))
            if s > best:
                best, bn = s, nm
        picks.append(bn)
        for v in folds:
            cur[v] = cur[v] + P[v][bn]
    return {k: c / len(picks) for k, c in Counter(picks).items()}


def blend(P, w):
    return sum(P[k] * q for k, q in w.items() if k in P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, default=100.0)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    L.K_IS = args.k
    P, YV = {}, {}
    need = [v for v in (*SEL, HOLD) if not (OUT_DIR / f"preds_val{v}.npz").exists()]
    for v in (*SEL, HOLD):
        f = OUT_DIR / f"preds_val{v}.npz"
        if f.exists():
            z = np.load(f)
            YV[v] = z["y"]
            P[v] = {k: z[k] for k in z.files if k != "y"}
            print(f"[val {v}] 저장된 예측 재사용 — 후보 {len(P[v])}종")

    if need:
        print(f">> train.csv 로드 (누락 폴드 {need})")
        df = rd.load_train()
        lut = IS.career_end_lookup(df)
        nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy())
        X = L.build_features(df, base=(nb, kb))
        y = df[rd.TARGET].to_numpy().astype(float)
        season = df[rd.SEASON].to_numpy()
        seasons = sorted(int(s) for s in np.unique(season))
        for v in need:
            print(f"\n[val {v}] 후보 풀 학습")
            P[v], YV[v] = EP.build_pool(X, y, season, v, seasons)
            np.savez_compressed(OUT_DIR / f"preds_val{v}.npz", y=YV[v], **P[v])

    common = sorted(set.intersection(*[set(P[v]) for v in (*SEL, HOLD)]))
    print(f"\n공통 후보 {len(common)}종 · 선택 폴드 {SEL} · **검증 폴드 {HOLD}(선택 미사용)**")

    # 기준: ENS-4 동결 구성의 폴드별 affine 점수 (maximin의 ref)
    ref = {v: affine_score(YV[v], blend(P[v], EP.ENS4_W)) for v in SEL}
    W_OLD = greedy_multi(P, YV, [SEL[0]], init=["nn_lin", "et_l100_d28"])   # 구 절차 재현
    W_NEW = greedy_multi(P, YV, list(SEL), init=["nn_lin", "et_l100_d28"],
                         mode="maximin", ref=ref)                            # 교정: maximin-delta

    rows = []
    for nm, w in (("ENS-4 동결", EP.ENS4_W), ("구 절차(val2024 단독)", W_OLD),
                  ("**교정(maximin)**", W_NEW)):
        line = {"cfg": nm}
        for v in (*SEL, HOLD):
            (sl, sh), s = grid_calib(YV[v], blend(P[v], w))
            line[f"val{v}"] = s
        rows.append(line)
    t = pd.DataFrame(rows)
    print("\n" + "=" * 84)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    base = rows[0]
    print(f"\n[ENS-4 동결 대비]  (val{HOLD}는 **선택에 쓰지 않은** 폴드)")
    for r in rows[1:]:
        d = {v: r[f"val{v}"] - base[f"val{v}"] for v in (*SEL, HOLD)}
        mark = "채택 가능" if d[HOLD] > 0 else "**기각 — 미사용 폴드에서 음수**"
        print(f"  {r['cfg']:22s} " + " · ".join(f"val{v} {d[v]:+7.2f}" for v in (*SEL, HOLD))
              + f"   → {mark}")

    print("\n  [교정 절차 가중]")
    for k, q in sorted(W_NEW.items(), key=lambda kv: -kv[1]):
        print(f"    {k:16s} {EP.ENS4_W.get(k, 0):.4f} → {q:.4f}")
    loc = rows[2][f"val{SEL[0]}"]
    print(f"\n  로컬(val 2024) {loc:.2f} → 투영 LB = 0.6995·{loc:.2f} + 362.2 = "
          f"{0.6995*loc+362.2:.1f} ± 12  (D-32: 점추정 금지, 구간으로만)")

    (OUT_DIR / "select.json").write_text(
        json.dumps({"old": W_OLD, "new": W_NEW, "rows": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR/'select.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
