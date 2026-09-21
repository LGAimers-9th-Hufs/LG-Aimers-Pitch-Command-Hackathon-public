# -*- coding: utf-8 -*-
"""Program C — "같은-시즌 상한 1185.9"를 **금지된 프로토콜 없이** 다시 잰다.

    python sweep/ceiling_protocol.py --val 2024

## 왜 이걸 다시 재는가

우리 전략 문서 전체가 한 숫자에 걸려 있다 —
**"현 53피처에는 1185.9만큼 신호가 있는데 우리는 773.5를 뽑는다. 남은 412점은 시즌 전이 비용"**
(`docs/log/17_channel_ceilings.md` §2~§4). 여기서 **"트랙맨이 원리적 필수"**라는 결론도 나왔다.

그런데 1185.9는 **`KFold(shuffle=True)`**로 측정됐다(`oracle_ceiling.py:122`) —
`CLAUDE.md`가 **첫 번째 비협상 원칙으로 금지한 프로토콜**이다
("forward-chaining by season + game-level grouping; **no random KFold**").
`asof_pitcher_*`는 **그 투수의 같은 시즌 다른 투구 라벨로 만든 누적 통계**이므로,
같은 투수의 다른 투구가 학습 폴드에 들어가면 편향은 **위쪽**으로 생긴다.

§3의 서명도 그 방향이다 — 같은 시즌이면 용량↑일수록 좋고(935 → 1185.9),
전이면 용량↑일수록 나쁘다(699.5 → 690 미만).

## 프로토콜 (같은 val 시즌 · 같은 피처 · 같은 용량 격자, 분할만 다르다)

| 코드 | 분할 | 제거하는 것 |
|---|---|---|
| **P-A** | `KFold(shuffle=True, 5)` | (없음 — 1185.9 재현 확인) |
| **P-B** | `GroupKFold(pitcher_id, 5)` | 같은 투수의 다른 투구가 학습에 들어가는 경로 |
| **P-C** | 시즌 내 forward `game_month ≤7` → `≥8` | 위 + 시간 방향 (실제 과제와 동형) |
| **P-D** | 전이 참조 `fit (val−1) → val` | (기존 체제, 773.5/699.5 계열) |

**⚠ 공정 비교를 위해 모든 프로토콜을 `game_month ≥ 8` 부분집합에서도 채점한다** —
P-C는 그 행들에만 예측이 있기 때문이다. 비교는 이 공통 열에서 한다.

**`std(p)`를 함께 본다**: 캘리된 예측에서 `score = 1e5·Var(p)/V`이므로 **누수는 std 팽창으로 나타난다.**

## 사전 등록 판정

- `P-C(leaves 31) ≳ 1100` 이고 P-A와 큰 차이 없음 ⇒ **412는 진짜.** 시즌 경계가 병목.
- `P-B/P-C ≈ P-D ≈ 700` ⇒ **1185.9는 프로토콜 산물.** "남은 412점"·"트랙맨 원리적 필수" 둘 다 철회.
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
import season_centering as SC     # noqa: E402

OUT_DIR = HERE.parent / "results" / "ceiling"
THREADS = 6
FWD_MONTH = 8                      # 시즌 내 forward 분할 경계 (2024: ≤7 = 69.7%, ≥8 = 30.3%)
GRID = [(7, 1500, 500), (15, 1000, 400), (31, 500, 600), (63, 300, 800), (127, 200, 400)]


def _fit_predict(Xtr, ytr, Xte, leaves, mind, iters):
    import lightgbm as lgb
    params = {**L.BASE_PARAMS, "num_leaves": leaves, "min_data_in_leaf": mind,
              "num_threads": THREADS}
    ds = lgb.Dataset(Xtr, label=ytr, free_raw_data=False)
    m = lgb.train(params, ds, num_boost_round=iters)
    return m.predict(Xte, num_threads=THREADS)


def oof(X, y, groups, kind, leaves, mind, iters, seed=0):
    """val 시즌 내부 out-of-fold 예측. 예측이 없는 행은 NaN."""
    from sklearn.model_selection import KFold, GroupKFold
    P = np.full(len(y), np.nan)
    if kind == "A":
        sp = KFold(n_splits=5, shuffle=True, random_state=seed).split(X)
    elif kind == "B":
        sp = GroupKFold(n_splits=5).split(X, y, groups=groups)
    else:
        raise ValueError(kind)
    for tr, te in sp:
        P[te] = _fit_predict(X.iloc[tr], y[tr], X.iloc[te], leaves, mind, iters)
    return P


def report(y, p, mask, tag, rows, note=""):
    m = mask & ~np.isnan(p)
    if m.sum() < 1000:
        return
    yy, pp = y[m], np.clip(p[m], 0, 1)
    raw = L.score(yy, pp)
    bc = SC.best_cal(yy, pp)[0]
    rows.append(dict(tag=tag, n=int(m.sum()), raw=raw, best_cal=bc,
                     std=float(pp.std()), note=note))
    print(f"    {tag:28s} n={m.sum():>7,}  raw {raw:9.1f}  캘리후 {bc:9.1f}  std {pp.std():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=int, default=2024)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(">> train.csv 로드")
    df = rd.load_train()
    Xall = L.build_features(df)
    yall = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()
    val = (season == args.val)
    prev = (season == args.val - 1)

    d = df[val]
    X = Xall[val].reset_index(drop=True)
    y = yall[val]
    pid = d["pitcher_id"].to_numpy()
    month = d["game_month"].to_numpy()
    late = month >= FWD_MONTH
    print(f"   val {args.val}: {val.sum():,}행 · 투수 {len(np.unique(pid))} · "
          f"month≥{FWD_MONTH} {late.mean():.1%} · r={y.mean():.4f} (threads={THREADS})")

    rows = []
    for leaves, mind, iters in GRID:
        cap = f"lv{leaves}/min{mind}"
        print(f"\n[용량 {cap}]")
        t = time.time()

        pa = oof(X, y, None, "A", leaves, mind, iters)
        report(y, pa, np.ones(len(y), bool), f"P-A 랜덤KFold  {cap}", rows, "전체")
        report(y, pa, late, f"P-A(≥{FWD_MONTH}월) {cap}", rows, "공통열")

        pb = oof(X, y, pid, "B", leaves, mind, iters)
        report(y, pb, np.ones(len(y), bool), f"P-B 투수그룹   {cap}", rows, "전체")
        report(y, pb, late, f"P-B(≥{FWD_MONTH}월) {cap}", rows, "공통열")

        pc = np.full(len(y), np.nan)
        pc[late] = _fit_predict(X[~late], y[~late], X[late], leaves, mind, iters)
        report(y, pc, late, f"P-C 시즌내forward {cap}", rows, "공통열")

        pd_ = _fit_predict(Xall[prev], yall[prev], X, leaves, mind, iters)
        report(y, pd_, np.ones(len(y), bool), f"P-D 전이       {cap}", rows, "전체")
        report(y, pd_, late, f"P-D(≥{FWD_MONTH}월) {cap}", rows, "공통열")
        print(f"    [{time.time()-t:.0f}s]")

    t = pd.DataFrame(rows)
    print("\n" + "=" * 96)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    print(f"\n[판정 표 — 공통열(month≥{FWD_MONTH})에서만 비교]")
    c = t[t.note == "공통열"].copy()
    c["cap"] = c.tag.str.extract(r"(lv\d+/min\d+)")
    c["proto"] = c.tag.str.extract(r"^(P-[ABCD])")
    piv = c.pivot_table(index="cap", columns="proto", values="best_cal")
    print(piv.to_string(float_format=lambda v: f"{v:.1f}"))

    pa_max = t[(t.tag.str.startswith("P-A")) & (t.note == "전체")]["best_cal"].max()
    pc_max = c[c.proto == "P-C"]["best_cal"].max()
    pd_max = c[c.proto == "P-D"]["best_cal"].max()
    print(f"\n  P-A 전체 최대 = {pa_max:.1f}   (기존 보고 1185.9 — 재현 여부 확인)")
    print(f"  P-C 최대 = {pc_max:.1f} · P-D 최대 = {pd_max:.1f}")
    if pc_max < pd_max + 150:
        print("  ⇒ **412는 프로토콜 산물이다.** 시즌 내 forward 상한이 전이 수준과 다르지 않다.\n"
              "     '남은 412점'·'트랙맨 원리적 필수' 두 결론을 철회해야 한다.")
    elif pc_max >= 1100:
        print("  ⇒ **412는 진짜다.** 시즌 경계가 병목 — 단조 제약 등 전이 회수 수단이 값을 한다.")
    else:
        print("  ⇒ 중간. 웅덩이는 있으나 보고된 412보다 작다 — 크기를 다시 계산할 것.")

    (OUT_DIR / f"protocol_val{args.val}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
