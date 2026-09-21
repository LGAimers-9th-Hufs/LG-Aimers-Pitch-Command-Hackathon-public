# -*- coding: utf-8 -*-
"""K2 — CatBoost + **ID 범주형** 멤버. Phase 0의 "ID 유해" 판정을 올바른 도구로 재심한다.

    python sweep/catboost_ids.py --vals 2024,2023

## 왜 재심인가

Phase 0 C5는 "53피처+투수ID(1142.7) < 53피처(1185.9) ⇒ ID 유해"라 판정했다. 그런데 그 측정은
**LGBM 정수 범주형 + 랜덤 KFold 교차적합**이었다 — ID를 다루는 데 가장 나쁜 도구다.

CatBoost의 **ordered target statistics**는 정확히 이 문제(고카디널리티 범주 + 타깃 누수 + 미등장
카테고리)를 위해 설계됐다: 순열 순서상 앞 행들의 타깃 통계만 쓰므로 누수가 없고, 미등장 ID는
prior로 자연 폴백된다. **다시즌 학습**과 결합하면 ID별 이력이 길어져 통계가 정밀해진다.

주의: 오라클상 투수 채널은 포화(723.9)다. 이게 이길 경로는 "투수 식별 자체"가 아니라
**ordered TS가 만드는 시간 순서 통계가 is4와 다른 형태의 당해 적응**을 하는 경우뿐이다.
prior는 낮게 잡되, 크라우드 표준이므로 측정으로 닫는다.

## 게이트

멤버 후보 기준: ENS-4 캐시 블렌드 대비 단독·불일치 s·저가중 이득 (2폴드 + 미사용 val2022 확인).
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
from season_centering import best_cal   # noqa: E402

CACHE = HERE.parent / "results" / "ens5"
OUT_DIR = HERE.parent / "results" / "catboost_ids"
WS = (0.05, 0.10, 0.15, 0.20)

CONFIGS = [
    # (이름, 학습 스코프, CatBoost 파라미터)
    ("cat_id_1s",  "season", dict(depth=6, iterations=800, learning_rate=0.03, l2_leaf_reg=8.0)),
    ("cat_id_all", "all",    dict(depth=6, iterations=1200, learning_rate=0.03, l2_leaf_reg=8.0)),
    ("cat_id_all_d8", "all", dict(depth=8, iterations=1000, learning_rate=0.03, l2_leaf_reg=20.0)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024,2023")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    from catboost import CatBoostRegressor, Pool
    print(">> train.csv 로드 (K_IS=100)")
    L.K_IS = 100.0
    df = rd.load_train()
    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy())
    X = L.build_features(df, base=(nb, kb)).reset_index(drop=True)
    # ID를 **문자열 범주형**으로 덧붙인다 (CatBoost가 ordered TS로 처리)
    XI = X.copy()
    XI["pid_cat"] = df["pitcher_id"].astype(str).to_numpy()
    XI["bid_cat"] = df["batter_id"].astype(str).to_numpy()
    cat_idx = [XI.columns.get_loc("pid_cat"), XI.columns.get_loc("bid_cat")]
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()

    rows = []
    for v in [int(s) for s in args.vals.split(",")]:
        val = season == v
        yv = y[val]
        z = np.load(CACHE / f"preds_val{v}.npz")
        ens = sum(z[m] * w for m, w in EP.ENS4_W.items())
        base = best_cal(yv, ens)[0]
        print(f"\n[val {v}]  ENS-4 기준 {base:.2f}")
        # CatBoost는 결측 NaN 그대로 지원 — 수치는 그대로, 범주만 문자열
        Xv = XI[val]
        pool_v = Pool(Xv, cat_features=cat_idx)
        for name, scope, params in CONFIGS:
            fit = (season == v - 1) if scope == "season" else (season < v)
            t = time.time()
            m = CatBoostRegressor(loss_function="RMSE", thread_count=6, random_seed=7,
                                  verbose=False, allow_writing_files=False, **params)
            m.fit(Pool(XI[fit], y[fit], cat_features=cat_idx))
            p = np.clip(m.predict(pool_v), 0, 1)
            s = float(np.sqrt(((p - ens) ** 2).mean()))
            gains = {w: best_cal(yv, (1 - w) * ens + w * p)[0] - base for w in WS}
            bw = max(gains, key=gains.get)
            rows.append(dict(val=v, arm=name, scope=scope, n_fit=int(fit.sum()),
                             solo=L.score(yv, L.calibrate(p)), s=s,
                             best_w=bw, gain=gains[bw], sec=round(time.time() - t)))
            print(f"    {name:14s} [{scope:6s} {fit.sum():>9,}행] 단독 {rows[-1]['solo']:8.2f}  "
                  f"s={s:.4f}  최적w {bw:.2f} → {gains[bw]:+7.2f}  [{rows[-1]['sec']}s]")

    t = pd.DataFrame(rows)
    print("\n" + "=" * 88)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    g = t.groupby("arm").agg(solo=("solo", "mean"), s=("s", "mean"),
                             gain_mean=("gain", "mean"), gain_min=("gain", "min"))
    print("\n[요약 — ENS-4 대비]")
    print(g.sort_values("gain_mean", ascending=False).to_string(float_format=lambda v: f"{v:.3f}"))
    (OUT_DIR / "gate.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR/'gate.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
