# -*- coding: utf-8 -*-
"""is4 계열 값싼 확장 — **차분 피처**와 **수축 상수 K**.

    python sweep/is_tuning.py --vals 2024,2023,2022

## 왜 이 둘인가

is4가 성공한 원리는 두 가지다:
1. **트리가 원리적으로 만들 수 없는 값**을 준다 — `pitcher_id`가 피처에 없어 투수별 baseline을 못 뺀다.
2. **차분(당해 − 커리어)** 형태다. 축 정렬 분할로는 두 컬럼의 차를 표현하기가 매우 비효율적이다.

②는 **룩업 없이도** 쓸 수 있다. `prev{1,3,5}_fill`과 `p_sm500`은 **둘 다 이미 53피처 안에 있는데
그 차는 없다.** 트리에게는 "최근 5경기가 커리어보다 좋은가"가 두 컬럼을 동시에 쪼개야 하는 문제고,
`is_delta`가 같은 형태로 크게 값을 했으므로 짧은 윈도우 버전도 볼 가치가 있다.
(D-14/15/16이 "율 유래 피처 증분 0"을 냈지만 그건 **폐기된 feat82 트랙**이고 차분 형태가 아니었다.)

K=200은 `inseason.py`에서 **근거 없이 고른 값**이다. june의 p_sm500이 K=500을 쓰므로 당해 시즌
표본(수십~수천)에는 더 작은 값이 맞을 수 있다. 다만 val 2024에서 고르면 낙관이 붙으므로 3폴드로 본다.

## 게이트

기준선 = **57피처(is4, K=200)** = 실제 제출본 ENS-3. 3폴드 + D-25 규칙.
`prevdelta`는 룩업이 필요 없어 서빙 비용이 0이다.
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
from season_centering import best_cal, paired_se, fit_pair, predict_pair   # noqa: E402

OUT_DIR = HERE.parent / "results" / "inseason"


def inseason_cols(df, nb, kb, K):
    n = df["asof_pitcher_n"].astype("float64").to_numpy()
    rate = df["asof_pitcher_success_rate"].fillna(0.0).astype("float64").to_numpy()
    k = np.rint(rate * n)
    is_n = np.maximum(n - nb, 0.0)
    is_k = np.clip(k - kb, 0.0, is_n)
    is_sm = (is_k + K * 0.5) / (is_n + K)
    car_sm = (rate * n + 250.0) / (n + 500.0)
    return pd.DataFrame({
        "is_logn": np.log1p(is_n).astype("float32"),
        "is_share": (is_n / np.maximum(n, 1.0)).astype("float32"),
        "is_sm": is_sm.astype("float32"),
        "is_delta": (is_sm - car_sm).astype("float32")})


def prev_delta_cols(df):
    """룩업 없는 순수 행 단위 차분 — 두 컬럼 모두 이미 53피처 안에 있지만 그 **차**는 없다."""
    n = df["asof_pitcher_n"].astype("float64").to_numpy()
    rate = df["asof_pitcher_success_rate"].fillna(0.0).astype("float64").to_numpy()
    car_sm = (rate * n + 250.0) / (n + 500.0)
    out = {}
    for w in (1, 3, 5):
        p = (df[f"asof_pitcher_prev{w}_game_success_rate"]
             .fillna(df["asof_pitcher_success_rate"]).fillna(0.5).astype("float64").to_numpy())
        out[f"pd{w}"] = (p - car_sm).astype("float32")
    out["pd15"] = (out["pd1"] - out["pd5"]).astype("float32")     # 초단기 − 단기 = 모멘텀
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024,2023,2022")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(">> train.csv 로드")
    df = rd.load_train()
    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy())
    X = L.build_features(df).reset_index(drop=True)
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()

    ISK = {K: inseason_cols(df, nb, kb, K) for K in (50, 100, 200, 500)}
    PD = prev_delta_cols(df)
    print(f"   is4 K격자 {list(ISK)} · prevdelta {list(PD.columns)}")

    def frame(K, extra=None):
        parts = [X, ISK[K]]
        if extra is not None:
            parts.append(extra)
        return pd.concat(parts, axis=1)

    arms = [("is4_K200(기준)", lambda: frame(200)),
            ("is4_K50", lambda: frame(50)),
            ("is4_K100", lambda: frame(100)),
            ("is4_K500", lambda: frame(500)),
            ("+prevdelta", lambda: frame(200, PD)),
            ("+pd_only15", lambda: frame(200, PD[["pd1", "pd5", "pd15"]]))]

    res = []
    for v in [int(s) for s in args.vals.split(",")]:
        fit, val = (season == v - 1), (season == v)
        yv = y[val]
        print(f"\n[val {v}]  fit {fit.sum():,} → val {val.sum():,}")
        base_p = None
        for name, mk in arms:
            t = time.time()
            XA = mk()
            p = np.clip(predict_pair(fit_pair(XA[fit], y[fit], None), XA[val]), 0, 1)
            if base_p is None:
                base_p = p
            pc, bpc = L.calibrate(p), L.calibrate(base_p)
            rec = dict(val=v, arm=name, ncol=XA.shape[1], fixed=L.score(yv, pc),
                       best=best_cal(yv, p)[0], d=L.score(yv, pc) - L.score(yv, bpc),
                       se=paired_se(yv, pc, bpc), sec=round(time.time() - t, 1))
            res.append(rec)
            print(f"  {name:16s} ({XA.shape[1]:2d}col) 고정 {rec['fixed']:8.2f} "
                  f"재적합 {rec['best']:8.2f}  Δ {rec['d']:+7.2f}±{rec['se']:.1f} [{rec['sec']}s]")

    t = pd.DataFrame(res)
    print("\n" + "=" * 92)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\n[폴드 요약 — D-25 규칙, 기준 = is4_K200]")
    g = t[~t.arm.str.contains("기준")].groupby("arm").agg(
        ncol=("ncol", "max"), d_mean=("d", "mean"), d_sd=("d", "std"), d_min=("d", "min"),
        pos=("d", lambda s: int((s > 0).sum())), n=("d", "size"))
    g["통과"] = (g.d_mean > 0) & (g.d_mean > g.d_sd)
    print(g.sort_values("d_mean", ascending=False).to_string(float_format=lambda v: f"{v:.2f}"))
    (OUT_DIR / "tuning.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR/'tuning.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
