# -*- coding: utf-8 -*-
"""ENS-4 구성 확정 — K=100 + `all_raw` 저가중 멤버, 캘리 상수 재적합.

    python sweep/ens4_config.py --vals 2024,2023

## 구성 근거

| 변경 | 성격 | 로컬 Δ | 실현율 | 기대 LB | 근거 |
|---|---|---|---|---|---|
| **K 200 → 100** | 피처형 | +13.0(V24) · 3폴드 전부 양수 | 0.47 | +6.1 | `is_tuning.py`, D-25 통과 |
| **`all_raw` w=0.10** | 앙상블형 | +5.97 | 0.88 | +5.2 | `member_factory --inseason --only regime` |

**⚠ 구성이 바뀌면 캘리 상수는 무효다**([[june853-recipe]] 명시 규칙 — 블렌드가 분산을 줄이므로
최적 slope가 달라진다). ENS-3의 (1.02, −0.0075)를 그대로 쓰면 안 되고 여기서 다시 잡는다.

**⚠ 가중 재조정**: 기존 5멤버를 0.90배로 축소하고 `all_raw`에 0.10을 준다.
그리디로 재최적화하지 않는다 — `ens3_weights.py`의 선택 낙관 검사를 통과한 경우에만 별도로 적용한다.

`all_raw` = june l7+l15 레시피를 **전 시즌 원시 타깃**으로 학습한 것. 단독은 502로 나쁘지만
ENS와의 불일치가 0.0349로 우리가 가진 것 중 최대라 저가중에서 값을 한다(D-26의 채택 기준 ①완화).
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
W_ALLRAW = 0.10


def fit_calib(y, p):
    """(slope, shift) 격자 재적합 — 빌더에 동결할 상수를 여기서 정한다."""
    best = (None, -1e18)
    for sl in np.arange(0.55, 1.31, 0.01):     # ⚠ 파단 연도(val 2023)는 최적 slope가 0.60이다.
                                                #   격자를 0.85에서 시작하면 경계에서 잘려 −460짜리 가짜 손실이 나온다.
        q = 0.5 + sl * (np.asarray(p) - 0.5)
        for sh in np.arange(-0.03, 0.0201, 0.0025):
            v = L.score(y, np.clip(q + sh, 1e-6, 1 - 1e-6))
            if v > best[1]:
                best = ((round(float(sl), 3), round(float(sh), 4)), v)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024,2023")
    ap.add_argument("--k", type=float, default=100.0)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f">> train.csv 로드 (K_IS = {args.k:g})")
    L.K_IS = args.k                       # ← 빌더도 이 상수를 읽어 metadata에 넣는다
    df = rd.load_train()
    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy())
    X = L.build_features(df, base=(nb, kb))
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()
    seasons = sorted(int(s) for s in np.unique(season))

    out = []
    for v in [int(s) for s in args.vals.split(",")]:
        fit, val = (season == v - 1), (season == v)
        yv = y[val]
        print(f"\n[val {v}]  fit {fit.sum():,} → val {val.sum():,}")

        P = EC.train_members(X, y, fit, val)
        ens3 = sum(P[n] * w for n, _, w, _ in EC.MEMBERS)
        s3_fixed = L.score(yv, L.calibrate(ens3, **EC.ENS2_CAL))
        s3_best, a3 = best_cal(yv, ens3)
        print(f"    ENS-3(K={args.k:g}) 고정캘리(1.02,−0.0075) {s3_fixed:8.2f} · 재적합 {s3_best:8.2f} {a3}")

        # all_raw = 전 시즌 원시 풀링, june 2종
        import season_centering as SC
        arms = {a["name"]: a for a in SC.arms_for(v, seasons)}
        t = time.time()
        p_all, n_all = SC.run_arm(arms["all_raw"], X, y, season, val)
        p_all = np.clip(p_all, 0, 1)
        print(f"    all_raw 학습 {n_all:,}행 [{time.time()-t:.0f}s] "
              f"단독 {L.score(yv, L.calibrate(p_all)):.2f} · "
              f"불일치 {np.sqrt(((p_all-ens3)**2).mean()):.4f}")

        ens4 = (1 - W_ALLRAW) * ens3 + W_ALLRAW * p_all
        (sl, sh), s4 = fit_calib(yv, ens4)
        s4_old = L.score(yv, L.calibrate(ens4, **EC.ENS2_CAL))
        print(f"    ENS-4 = 0.90·ENS-3 + 0.10·all_raw")
        print(f"      구 캘리(1.02,−0.0075) {s4_old:8.2f}  ·  **재적합 ({sl}, {sh}) {s4:8.2f}**")
        print(f"      ⇒ ENS-3 대비 {s4 - s3_best:+.2f}")
        out.append(dict(val=v, ens3_fixed=s3_fixed, ens3_best=s3_best,
                        ens4_oldcal=s4_old, ens4_best=s4, calib=[sl, sh],
                        delta=s4 - s3_best))

    print("\n" + "=" * 78)
    t = pd.DataFrame(out)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    main_v = out[0]
    print(f"\n[빌더에 동결할 값]  K_IS = {args.k:g} · calib = {main_v['calib']} "
          f"· all_raw 가중 {W_ALLRAW}")
    print(f"[val {main_v['val']} 기준] ENS-3 {main_v['ens3_best']:.2f} → ENS-4 {main_v['ens4_best']:.2f}")
    (OUT_DIR / "ens4.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR/'ens4.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
