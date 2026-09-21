# -*- coding: utf-8 -*-
"""학습창(era window) 레짐 베팅의 d 최종 판정 — "정직한 d의 벽"이 보편인지 결정한다.

    python sweep/era_window_d.py

배경(docs/log/36 §4): 유능한 같은-데이터 모델의 d(D4평균)가 0.024~0.026에 4중 수렴.
마지막 미측정 d-원천 = 학습창 베팅("최신 1시즌만"은 ±400~750점급 예측 변화 — june853 메모).
전시즌/가중 학습 LGBM의 d가 이것마저 ≤0.026이면 벽은 보편 ⇒ 1100 추구 공식 종료.

측정: fit ≤2023 변형 4종(2023단독 / 2022-23 / 2019-23 균등 / 2019-23 recency 0.5^Δ)을
같은 XA(78) 위에서 학습 → ① 2024 프록시 30k행에서 캐시 레그들과 d ② V24 솔로(leg_score).
D4 레그들도 2023-빈티지(p23)로 잰 값이므로 fit≤2023이 공정 비교다.
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import real_data as rd                     # noqa: E402
from tm_repr_leg import build_xa, leg_score   # noqa: E402

OUT = HERE.parent / "results" / "tm_repr"
PARAMS = dict(objective="binary", metric="binary_logloss", learning_rate=0.05,
              num_leaves=15, min_data_in_leaf=1000, feature_fraction=0.85,
              bagging_fraction=0.8, bagging_freq=1, verbosity=-1, seed=0)


def main():
    t0 = time.time()
    df = rd.load_train()
    y = df[rd.TARGET].to_numpy(dtype=np.float32)
    season = df[rd.SEASON].to_numpy()
    XA = build_xa(df, season)
    val = season == 2024
    yv = y[val]
    print(f">> XA {XA.shape}  [{time.time()-t0:.0f}s]")

    legs = {}
    for nm in ("clookup", "cmoe", "physmix", "tm3L", "cregime", "ysy_mlp"):
        fs = sorted(glob.glob(str(HERE.parent / "results/leg_matrix" / f"{nm}_30000_*.npy")))
        if fs:
            legs[nm] = np.load(fs[-1])
    D4 = np.mean([legs[k] for k in ("clookup", "cmoe", "physmix", "tm3L")], axis=0)

    variants = {
        "w23":  (season == 2023, None),
        "w2223": ((season >= 2022) & (season <= 2023), None),
        "wALL": (season <= 2023, None),
        "wREC": (season <= 2023, "rec"),
    }
    rep = {}
    preds = {}
    for name, (fit, wmode) in variants.items():
        t1 = time.time()
        sw = None
        if wmode == "rec":
            sw = (0.5 ** (2023 - season[fit])).astype(np.float64)
        m = lgb.train(PARAMS, lgb.Dataset(XA[fit], label=y[fit], weight=sw),
                      num_boost_round=600)
        pv = m.predict(XA[val])
        p30 = pv[:30000]
        preds[name] = p30
        solo = leg_score(yv, pv)
        d4 = float(np.sqrt(np.mean((p30 - D4) ** 2)))
        dd = {k: round(float(np.sqrt(np.mean((p30 - v) ** 2))), 5) for k, v in legs.items()}
        rep[name] = {"n_fit": int(fit.sum()), "solo_v24": round(solo, 2),
                     "d_D4": round(d4, 5), "d_legs": dd,
                     "train_s": round(time.time() - t1, 1)}
        print(f"[{name:6s}] fit {fit.sum():>9,}  solo(V24) {solo:8.2f}  d(D4) {d4:.5f}  "
              f"[{time.time()-t1:.0f}s]")

    # 변형 간 상호 d (학습창이 실제로 예측을 얼마나 바꾸나)
    cross = {}
    names = list(variants)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            cross[f"{names[i]}|{names[j]}"] = round(
                float(np.sqrt(np.mean((preds[names[i]] - preds[names[j]]) ** 2))), 5)
    rep["_cross_d"] = cross
    print("상호 d:", cross)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "era_window_d.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f">> saved results/tm_repr/era_window_d.json  [{time.time()-t0:.0f}s]")
    dmax = max(v["d_D4"] for k, v in rep.items() if not k.startswith("_"))
    print(f"\n판정: 학습창 변형의 최대 d(D4) = {dmax:.4f}"
          + ("  →  벽(0.026) 돌파 — T2 재평가" if dmax > 0.027 else
         "  →  벽 이내 — d 벽은 보편, 1100 추구 공식 종료 권고"))


if __name__ == "__main__":
    main()
