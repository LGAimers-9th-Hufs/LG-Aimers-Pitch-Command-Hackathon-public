# -*- coding: utf-8 -*-
"""NOMADD 0슬롯 오프라인 검증 — 시즌별 리프값을 미래로 외삽해 전이 손실을 줄이나?

    python sweep/nomadd_gate.py --vals 2024,2023,2022

배경: 라운드6 각도2 유일 생존 후보(arXiv:2608.02845). 우리 약점은 정보가 아니라 2024→2025
전이(로컬 905 → LB 1058). NOMADD는 트리 **구조는 고정하고 리프값만** 시즌별로 refit해,
리프값의 시즌 궤적을 저랭크(선형) 외삽 → 다음(라벨 없는) 시즌으로 투영한다. future label·
test 배치 미사용 = §5 합법(is4와 같은 train-유도 계열). 행 독립 추론 유지(구조 불변 트리).

메커니즘:
  T   = LGBM(≤v−1 전시즌 풀)로 트리 구조 T 고정 (앵커)
  L_s = T.refit(X_s, y_s)의 리프값 벡터  (s ∈ {≤v−1 각 시즌})
  L_v̂ = 리프별 시즌 선형외삽(s → v)         ← NOMADD 투영
비교 arm(같은 T 위, val v):
  nomadd = T + L_v̂     ·  last = T + L_{v−1}(최신시즌 refit, june 정신)  ·  anchor = T + L_pooled
판정: nomadd − last (전이 개선분). 3폴드 평균>0 & 평균>SD & 위약(시즌 순서 셔플) 분리.
킬: 평균≤0 / 위약 미분리 / 리프드리프트 비단조(2025 룰 불연속 = 외삽 불가 신호).
⚠ 이건 다양성 레그가 아니라 솔로 리프트 — 통과해도 컷 1135 미달(d 벽 무관 축).
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import real_data as rd            # noqa: E402
import lgbm_family as L           # noqa: E402
import inseason as IS             # noqa: E402
from season_centering import best_cal, paired_se   # noqa: E402

OUT_DIR = HERE.parent / "results" / "nomadd"
SPEC = dict(num_leaves=15, min_data_in_leaf=1000, num_iterations=400, seed=57)


def leaf_frame(booster):
    """(tree_index, leaf_id) → value, 정렬된 리프값 벡터. 같은 구조면 순서 불변."""
    df = booster.trees_to_dataframe()
    lv = df[df["left_child"].isna() & df["right_child"].isna()].copy()
    lv["leaf_id"] = lv["node_index"].str.split("L").str[-1].astype(int)
    lv = lv.sort_values(["tree_index", "leaf_id"]).reset_index(drop=True)
    return lv[["tree_index", "leaf_id", "value"]]


def set_leaves(booster, tree_idx, leaf_id, values):
    for t, l, v in zip(tree_idx, leaf_id, values):
        booster.set_leaf_output(int(t), int(l), float(v))


def roundtrip_check(X, y):
    """set_leaf_output 왕복 무결성 — 리프값을 읽어 그대로 다시 쓰면 예측 불변."""
    b = L.train_lgbm(X.iloc[:20000], y[:20000], dict(SPEC, num_iterations=30))
    p0 = b.predict(X.iloc[:2000])
    lf = leaf_frame(b)
    set_leaves(b, lf.tree_index, lf.leaf_id, lf.value.to_numpy())
    p1 = b.predict(X.iloc[:2000])
    d = float(np.abs(p1 - p0).max())
    assert d < 1e-9, f"왕복 파리티 실패 {d:.2e}"
    print(f"   [왕복] set_leaf_output 무결성 OK (max Δ {d:.1e}, 리프 {len(lf)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024,2023,2022")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(">> train.csv 로드 (is4 K=100)")
    L.K_IS = 100.0
    df = rd.load_train()
    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy())
    X = L.build_features(df, base=(nb, kb)).reset_index(drop=True)
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()

    roundtrip_check(X, y)

    rows = []
    rng = np.random.default_rng(7)
    for v in [int(s) for s in args.vals.split(",")]:
        fit = season <= v - 1
        val = season == v
        yv = y[val]
        fit_seasons = sorted(np.unique(season[fit]))
        print(f"\n[val {v}]  앵커 전시즌 {fit_seasons} ({fit.sum():,}) → val {val.sum():,}")

        # 앵커 구조 T (전시즌 풀) — refit의 기준 트리
        T = L.train_lgbm(X[fit], y[fit], SPEC)
        L_pooled = leaf_frame(T)
        tkeys = (L_pooled.tree_index.to_numpy(), L_pooled.leaf_id.to_numpy())

        # 시즌별 리프값 (같은 T 구조로 refit)
        Ls = {}
        for s in fit_seasons:
            m = (season == s)
            Ts = T.refit(data=X[m].to_numpy(), label=y[m])
            Ls[s] = leaf_frame(Ts).value.to_numpy()
        Lmat = np.stack([Ls[s] for s in fit_seasons], axis=0)      # (n_seasons, n_leaves)

        # 리프별 시즌 선형외삽 → v
        ss = np.asarray(fit_seasons, dtype=float)
        A = np.vstack([ss, np.ones_like(ss)]).T
        coef, *_ = np.linalg.lstsq(A, Lmat, rcond=None)            # (2, n_leaves)
        L_hat = coef[0] * float(v) + coef[1]

        # 위약: 시즌 순서 셔플 후 같은 외삽(궤적 방향 파괴)
        perm = rng.permutation(len(fit_seasons))
        coef_sh, *_ = np.linalg.lstsq(A, Lmat[perm], rcond=None)
        L_hat_sh = coef_sh[0] * float(v) + coef_sh[1]

        def predict_with(vals):
            b = T.refit(data=X[season == fit_seasons[-1]].to_numpy(),
                        label=y[season == fit_seasons[-1]])   # 구조 유지용 refit
            set_leaves(b, tkeys[0], tkeys[1], vals)
            return np.clip(b.predict(X[val].to_numpy()), 1e-6, 1 - 1e-6)

        p_nomadd = predict_with(L_hat)
        p_last = predict_with(Ls[fit_seasons[-1]])
        p_anchor = predict_with(L_pooled.value.to_numpy())
        p_shuf = predict_with(L_hat_sh)

        s_nomadd = best_cal(yv, p_nomadd)[0]
        s_last = best_cal(yv, p_last)[0]
        s_anchor = best_cal(yv, p_anchor)[0]
        s_shuf = best_cal(yv, p_shuf)[0]
        rows.append(dict(val=v, nomadd=round(s_nomadd, 2), last=round(s_last, 2),
                         anchor=round(s_anchor, 2), shuf=round(s_shuf, 2),
                         d_vs_last=round(s_nomadd - s_last, 2),
                         d_vs_anchor=round(s_nomadd - s_anchor, 2),
                         edge=round(s_nomadd - s_shuf, 2),
                         se=round(paired_se(yv, L.calibrate(p_nomadd), L.calibrate(p_last)), 2)))
        print(f"    nomadd {s_nomadd:8.2f} · last {s_last:8.2f} · anchor {s_anchor:8.2f} · "
              f"위약 {s_shuf:8.2f}")
        print(f"    Δ(vs last) {s_nomadd-s_last:+7.2f}±{rows[-1]['se']:.1f} · "
              f"Δ(vs anchor) {s_nomadd-s_anchor:+7.2f} · 위약분리 {s_nomadd-s_shuf:+7.2f}")

    t = pd.DataFrame(rows)
    print("\n" + "=" * 84)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    dm, dsd, dmin = t.d_vs_last.mean(), t.d_vs_last.std(), t.d_vs_last.min()
    em = t.edge.to_numpy()
    print(f"\n[3폴드 요약] Δ(vs last) 평균 {dm:+.2f} · SD {dsd:.2f} · 최소 {dmin:+.2f} · "
          f"위약분리 평균 {em.mean():+.2f}/최소 {em.min():+.2f}")
    passed = (dm > 0) and (dm > dsd) and (em.min() > 0)
    print(f"  ⇒ 채택 게이트: {'통과' if passed else '미달'} "
          f"(평균>0 & 평균>SD & 위약분리>0)")
    if not passed:
        print("     = NOMADD 외삽이 최신시즌 refit을 못 이김(전이 리프트 없음/불안정).")
    (OUT_DIR / "gate.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR/'gate.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
