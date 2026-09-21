# -*- coding: utf-8 -*-
"""Phase 0 — 채널 상한 오라클. 460점이 **있을 수 있는 곳**부터 확정한다.

    python sweep/oracle_ceiling.py --all

## 왜 이걸 먼저 하는가

현재 팀 최고 = 911.18(로컬 773.48), 1위 = **1,371.40**. 격차 460점 = 로컬 **+524**(실현율 0.878).
동급 두 모델 블렌드로 460점을 얻으려면 불일치 RMS 0.068이 필요한데 실측이 0.026이다
⇒ **앙상블·튜닝으로는 원리적으로 불가. 새 정보가 어디에 있는지부터 재야 한다.**

캘리된 모델은 `score = 1e5·Var(p)/V` 이므로 목표는 예측 산포 **0.0477 → 0.0586 (+22.7%)**.

## 오라클의 의미와 한계

여기서 재는 값은 **val 시즌 라벨을 이미 안다고 가정**한 상한이다. 실전에서는 얻을 수 없다.
- **투수 오라클**: "각 투수의 그 시즌 실제 성공률을 안다면" → 투수 채널이 담을 수 있는 최대치.
  실전 모델은 이걸 **이전 시즌 데이터로 예측**해야 하므로 항상 그보다 낮다. 둘의 차이가
  "예측 난이도로 잃는 몫"이다.
- **교차적합 상한(C계열)**: val 안에서 5-fold로 학습/예측 → "같은 시즌 데이터가 충분할 때
  이 피처셋으로 뽑을 수 있는 최대치". **이게 1298 미만이면 현 53피처로는 1371이 원리적으로 불가**하고,
  트랙맨 같은 새 피처가 선택이 아니라 필수임이 증명된다.

자기 라벨 유입을 막기 위해 타깃 인코딩 오라클은 전부 **leave-one-out**으로 계산한다.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import real_data as rd        # noqa: E402
import lgbm_family as L       # noqa: E402

REF_LOCAL = {"june853": 707.90, "ENS-2": 773.48}
TARGET_LOCAL = 1298.0         # 1위 1371.40 → 실현율 0.878로 환산


def best_cal(y, p, coarse=False):
    """캘리 재적합 후 최고 점수 — 채널 간 비교를 레벨 운에서 분리한다."""
    sl = np.arange(0.5, 2.01, 0.05) if coarse else np.arange(0.5, 2.01, 0.01)
    sh = np.arange(-0.04, 0.041, 0.005)
    best = -1e18
    for s in sl:
        q = 0.5 + s * (p - 0.5)
        for h in sh:
            v = L.score(y, np.clip(q + h, 1e-6, 1 - 1e-6))
            if v > best:
                best = v
    return best


def loo_rate(key: np.ndarray, y: np.ndarray, K: float, m: float) -> np.ndarray:
    """leave-one-out 수축 타깃 인코딩: (k − yᵢ + K·m) / (n − 1 + K)."""
    df = pd.DataFrame({"k": key, "y": y})
    g = df.groupby("k")["y"]
    n = g.transform("size").to_numpy(dtype=float)
    s = g.transform("sum").to_numpy(dtype=float)
    return (s - y + K * m) / np.maximum(n - 1 + K, 1e-9)


def report(name, y, p, rows):
    sc = L.score(y, np.clip(p, 1e-6, 1 - 1e-6))
    bc = best_cal(y, p)
    rows.append({"오라클": name, "raw": sc, "캘리후": bc, "std(p)": float(np.std(p))})
    print(f"  {name:34s} raw={sc:9.1f}  캘리후={bc:9.1f}  std={np.std(p):.4f}")


def run(df, val_season=2024, fit_seasons=(2023,)):
    X = L.build_features(df)
    y = df[rd.TARGET].to_numpy().astype(float)
    val = (df[rd.SEASON] == val_season).to_numpy()
    yv, Xv = y[val], X[val]
    m = float(yv.mean())
    d = df[val]
    print(f"val={val_season}  {val.sum():,}행  r={m:.4f}")
    print(f"기준선: june853 로컬 {REF_LOCAL['june853']:.1f} · ENS-2 로컬 {REF_LOCAL['ENS-2']:.1f} · "
          f"목표(1위 환산) {TARGET_LOCAL:.0f}\n")

    rows = []
    print("[A] 타깃 인코딩 오라클 (leave-one-out, val 라벨 사용 — 상한)")
    for K in (0.0, 50.0, 200.0, 500.0):
        report(f"O1 투수 (K={K:g})", yv, loo_rate(d["pitcher_id"].to_numpy(), yv, K, m), rows)
    for K in (200.0,):
        report(f"O2 타자 (K={K:g})", yv, loo_rate(d["batter_id"].to_numpy(), yv, K, m), rows)

    ctx = (d["balls_before"].astype(str) + "_" + d["strikes_before"].astype(str) + "_"
           + d["outs_before"].astype(str) + "_" + d["base_state"].astype(str) + "_"
           + d["top_bottom"].astype(str)).to_numpy()
    report("O3 상황(카운트·아웃·주자·초말)", yv, loo_rate(ctx, yv, 200.0, m), rows)

    pc = (d["pitcher_id"].astype(str) + "|" + d["balls_before"].astype(str)
          + d["strikes_before"].astype(str)).to_numpy()
    report("O1×O3 투수×카운트", yv, loo_rate(pc, yv, 200.0, m), rows)

    # 투수 + 타자 결합 (로짓 가산, 리그 평균 기준)
    lg = lambda q: np.log(np.clip(q, 1e-4, 1 - 1e-4) / (1 - np.clip(q, 1e-4, 1 - 1e-4)))
    p1 = loo_rate(d["pitcher_id"].to_numpy(), yv, 200.0, m)
    p2 = loo_rate(d["batter_id"].to_numpy(), yv, 200.0, m)
    comb = 1 / (1 + np.exp(-(lg(p1) + lg(p2) - lg(np.full_like(p1, m)))))
    report("O1+O2 투수+타자(로짓 가산)", yv, comb, rows)

    print("\n[B] 교차적합 상한 — val 안에서 5-fold 학습/예측")
    print("    = '같은 시즌 데이터가 충분할 때 이 피처셋으로 뽑을 수 있는 최대치'")
    import lightgbm as lgb
    from sklearn.model_selection import KFold

    def xfit(cols_extra=(), tag="", **params):
        F = Xv.copy()
        for c in cols_extra:
            F[c] = d[c].to_numpy()
        P = np.zeros(len(F))
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        pp = {**L.BASE_PARAMS, **params}
        cat = [c for c in cols_extra]
        for tr, te in kf.split(F):
            ds = lgb.Dataset(F.iloc[tr], label=yv[tr],
                             categorical_feature=cat or "auto", free_raw_data=False)
            mdl = lgb.train(pp, ds, num_boost_round=params.get("num_iterations", 400))
            P[te] = mdl.predict(F.iloc[te], num_threads=4)
        report(tag, yv, np.clip(P, 0, 1), rows)
        return P

    # ⚠ 용량을 키우면 붕괴한다(leaves 255·min 100 → raw −6107, std 0.18). 신호가 약해서
    #   교차적합으로도 과적합을 막지 못한다 — 상한 측정은 **정칙화 격자**로 해야 의미가 있다.
    for lv, md, it in ((7, 1500, 500), (15, 1000, 400), (31, 500, 600), (63, 300, 800)):
        xfit((), f"C1 53피처 (leaves {lv}, min {md})", num_leaves=lv,
             min_data_in_leaf=md, num_iterations=it)
    xfit(("pitcher_id",), "C5 53피처 + 투수ID(범주형)", num_leaves=15,
         min_data_in_leaf=1000, num_iterations=400)
    xfit(("pitcher_id", "batter_id"), "C6 53피처 + 투수·타자ID", num_leaves=15,
         min_data_in_leaf=1000, num_iterations=400)

    t = pd.DataFrame(rows).sort_values("캘리후", ascending=False)
    print("\n" + "=" * 78)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    top = t.iloc[0]
    print("\n[판정]")
    c = t[t["오라클"].str.startswith("C")]["캘리후"].max()
    print(f"  현 피처셋 교차적합 상한(C계열 최대) = {c:.1f}  vs  목표 {TARGET_LOCAL:.0f}")
    if c < TARGET_LOCAL:
        print("  ⇒ **현 53피처로는 1371에 원리적으로 도달 불가.** 새 피처(트랙맨 등)가 필수다.")
    else:
        print("  ⇒ 현 피처셋 안에 여지가 있다 — 모델링(용량·정칙화·손실)이 병목이다.")
    o1 = t[t["오라클"].str.startswith("O1 투수")]["캘리후"].max()
    print(f"  투수 채널 상한 = {o1:.1f} (ENS-2 로컬 {REF_LOCAL['ENS-2']:.1f} 대비 "
          f"{o1 - REF_LOCAL['ENS-2']:+.1f})")
    print(f"  최고 오라클 = {top['오라클']} {top['캘리후']:.1f} (std {top['std(p)']:.4f})")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--val", type=int, default=2024)
    args = ap.parse_args()
    print(">> train.csv 로드")
    df = rd.load_train()
    run(df, val_season=args.val)


if __name__ == "__main__":
    main()
