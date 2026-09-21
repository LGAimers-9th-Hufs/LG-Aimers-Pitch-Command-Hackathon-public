# -*- coding: utf-8 -*-
"""Pipeline B — "팀원이 만들었을" 두 번째 독립 파이프라인. 설계 축을 의도적으로 전부 뒤집는다.

    python sweep/pipeline_b.py --vals 2024,2023,2022

## 왜 이것이 마지막 열린 문인가

블렌드 항등식: A(984급)와 **레벨이 비슷하고 다르게 틀리는** B가 있으면
`이득 = w(S_B−S_A) + 1e5·w(1−w)·s²/V`. S_B가 A보다 −38 낮아도 s=0.025면 w=0.3에서 **+41 로컬**이다.
지금까지의 멤버 실험이 이걸 못 얻은 이유는 후보들의 **레벨이 너무 낮아서**였다(feat82 단독 565,
allraw 520 — 적자가 다양성 이득을 삼킨다). B의 과제는 "다르게"가 아니라 **"다르면서 840+"**다.

## 설계 대척점 체크리스트 (A와 공유하는 것은 정보원뿐이어야 한다)

| 축 | A (ENS-4) | **B** |
|---|---|---|
| 피처 | june 53 + is4 | **feat82클린**(폼추세·믹스엔트로피·LI변환 등 다른 파생) + is4 |
| 목적함수 | **regression(MSE)** 확률평균 | **binary logloss** (진짜 분류기) |
| 라이브러리 | LGBM+sklearn ET+torch 선형 | LGBM-binary + **CatBoost-Logloss** |
| 캘리 | (1.04, −0.01) | B 자체 재적합 |

공유가 불가피한 결정(정보 기반이라 스타일이 아님): 최신 1시즌 학습 · is4 · K=100.

⚠ feat82에서 `season`·`era_abs` 제거(시즌 드리프트 — z_asof/−192와 같은 실패 형태).

## 판정

① B 자체 레벨(3폴드) ② s(A,B) ③ 블렌드 w 격자를 maximin(Δ24,Δ23)으로 → **val2022(미사용) 확인**
④ 채택 기준 = 기존 그대로(maximin +10 이상이어야 zip 빌드).
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
import ens5_pool as EP            # noqa: E402
from ens4_weights import grid_calib   # noqa: E402
from season_centering import best_cal   # noqa: E402

CACHE = HERE.parent / "results" / "ens5"
OUT_DIR = HERE.parent / "results" / "pipeline_b"
DROP82 = ["season", "era_abs"]                 # 시즌 드리프트 컬럼
WGRID = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)

# B 멤버 — 전부 분류기(binary/logloss). 정칙화는 t54 교훈(강하게)을 잇되 값은 독립 선택.
B_MEMBERS = [
    ("b_lgb82_a", "lgb82", dict(objective="binary", metric="binary_logloss",
                                num_leaves=7, min_data_in_leaf=2500, learning_rate=0.0175,
                                num_iterations=800, lambda_l1=5.0, lambda_l2=1.0,
                                feature_fraction=0.6, bagging_fraction=0.75, bagging_freq=1,
                                seed=101)),
    ("b_lgb82_b", "lgb82", dict(objective="binary", metric="binary_logloss",
                                num_leaves=15, min_data_in_leaf=1200, learning_rate=0.025,
                                num_iterations=500, lambda_l1=0.5, lambda_l2=8.0,
                                feature_fraction=0.85, seed=103)),
    ("b_cat82",   "cat82", dict(depth=6, iterations=800, learning_rate=0.03, l2_leaf_reg=8.0)),
    ("b_lgb57",   "lgb57", dict(objective="binary", metric="binary_logloss",
                                num_leaves=15, min_data_in_leaf=1000, learning_rate=0.03,
                                num_iterations=500, seed=19)),      # ens5 풀의 lg_bin(786) 재현
]


def build_feat82(df, nb, kb):
    X = rd.build_features(df).reset_index(drop=True)
    X = X.drop(columns=[c for c in DROP82 if c in X.columns])
    x = df[["asof_pitcher_n", "asof_pitcher_success_rate"]].copy()
    x = L.add_inseason(x, nb, kb)
    for c in L.IS_COLS:
        X[c] = x[c].to_numpy()
    return X


def train_b(name, kind, spec, X82, X57, y, fit, val):
    import lightgbm as lgb
    if kind in ("lgb82", "lgb57"):
        X = X82 if kind == "lgb82" else X57
        pp = {**{k: v for k, v in spec.items() if k != "num_iterations"},
              "num_threads": 6, "verbosity": -1, "force_col_wise": True}
        ds = lgb.Dataset(X[fit], label=y[fit], free_raw_data=False)
        m = lgb.train(pp, ds, num_boost_round=spec["num_iterations"])
        return np.clip(m.predict(X[val], num_threads=6), 1e-6, 1 - 1e-6)
    if kind == "cat82":
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(loss_function="Logloss", thread_count=6, random_seed=7,
                               verbose=False, allow_writing_files=False, **spec)
        Xn = np.nan_to_num(X82[fit].to_numpy(dtype=np.float32), nan=-999.0)
        Xv = np.nan_to_num(X82[val].to_numpy(dtype=np.float32), nan=-999.0)
        m.fit(Xn, y[fit].astype(int))
        return np.clip(m.predict_proba(Xv)[:, 1], 1e-6, 1 - 1e-6)
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024,2023,2022")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(">> train.csv 로드 (K_IS=100)")
    L.K_IS = 100.0
    df = rd.load_train()
    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy())
    X57 = L.build_features(df, base=(nb, kb)).reset_index(drop=True)
    X82 = build_feat82(df, nb, kb)
    print(f"   B 피처: feat82클린+is4 = {X82.shape[1]}컬럼 · A 피처 57컬럼")
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()

    res, save = [], {}
    for v in [int(s) for s in args.vals.split(",")]:
        fit, val = (season == v - 1), (season == v)
        yv = y[val]
        z = np.load(CACHE / f"preds_val{v}.npz")
        A = sum(z[m] * w for m, w in EP.ENS4_W.items())
        sA = best_cal(yv, A)[0]
        print(f"\n[val {v}]  fit {fit.sum():,} → val {val.sum():,}  A(ENS-4) = {sA:.2f}")

        PB = {}
        for name, kind, spec in B_MEMBERS:
            t = time.time()
            PB[name] = train_b(name, kind, spec, X82, X57, y, fit, val)
            print(f"    {name:10s} 단독 {best_cal(yv, PB[name])[0]:8.2f}  "
                  f"s_vs_A={np.sqrt(((PB[name]-A)**2).mean()):.4f}  [{time.time()-t:.0f}s]")
        B = np.mean(list(PB.values()), axis=0)                     # B 앙상블 = 등가중(설계 단순성)
        sB = best_cal(yv, B)[0]
        sAB = float(np.sqrt(((B - A) ** 2).mean()))
        print(f"    B(등가중 4멤버)  레벨 {sB:8.2f} (A와 차 {sB-sA:+.1f}) · s(A,B) = {sAB:.4f}")

        row = dict(val=v, sA=sA, sB=sB, s=sAB)
        for w in WGRID:
            row[f"w{int(w*100):02d}"] = best_cal(yv, (1 - w) * A + w * B)[0] - sA
        res.append(row)
        save[v] = dict(A=A, B=B, y=yv)
        print("    블렌드 Δ: " + "  ".join(f"w{w:.2f}:{row[f'w{int(w*100):02d}']:+.1f}"
                                           for w in WGRID))

    t = pd.DataFrame(res)
    print("\n" + "=" * 96)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    sel = [r for r in res if r["val"] in (2024, 2023)]
    if len(sel) == 2:
        print("\n[maximin(Δ24, Δ23) — 선택 폴드]")
        best_w, best_mm = None, -1e18
        for w in WGRID:
            k = f"w{int(w*100):02d}"
            mm = min(r[k] for r in sel)
            print(f"  w={w:.2f}  maximin {mm:+7.2f}  (Δ24 {sel[0][k]:+.2f} · Δ23 {sel[1][k]:+.2f})")
            if mm > best_mm:
                best_mm, best_w = mm, w
        hold = [r for r in res if r["val"] == 2022]
        if hold:
            k = f"w{int(best_w*100):02d}"
            print(f"\n  최적 w={best_w:.2f} (maximin {best_mm:+.2f}) → "
                  f"**val2022(미사용) Δ {hold[0][k]:+.2f}** "
                  f"{'✅' if hold[0][k] > 0 else '❌ 채택 금지'}")
            print(f"  채택 기준: maximin ≥ +10 {'충족' if best_mm >= 10 else '**미달**'}")

    for v, d in save.items():
        np.savez_compressed(OUT_DIR / f"preds_val{v}.npz", **d)
    (OUT_DIR / "gate.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
