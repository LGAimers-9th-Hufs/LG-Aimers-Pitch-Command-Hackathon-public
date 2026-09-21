# -*- coding: utf-8 -*-
"""C2 — 당해시즌 벡터의 **저섭동 운반**: nn_lin 멤버만 교체, 다른 4멤버 고정 → ENS 직측.

    python sweep/nn_carrier.py --vals 2024,2023,2022

## 왜 이 운반인가 (Codex 지적 ③)

`is4+타자+실패유형`은 is4 대비 **3/3 폴드 양수**(+3.2/+73.2/+12.3)였는데, 전부 june 트리에
컬럼 직삽으로 쟀다 — 그 방식은 ±13 이상의 구조 섭동이 실증돼 있다(D-25 위약). 저섭동 대안:
**nn_lin만 확장 입력으로 재학습**하고 다른 4멤버는 캐시를 쓰면 트리 재배치가 원천 차단되고,
판정이 곧 ENS 증분이다. is4가 nn_lin을 가장 크게 올렸다는 실측(644→805)과도 정합 —
선형 멤버가 이 계열(수축률·비율 벡터) 정보의 최적 수용체다.

## 사다리

base(57) → +타자 당해분(bis 4) → +투수 실패유형 당해분(8) → +둘 다 → +둘 다+분모(C1)
각 arm: 3폴드 ENS 증분 + 최상 arm에 엔티티 셔플 위약 2시드.
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
import inseason_full as IF        # noqa: E402
import ens5_pool as EP            # noqa: E402
import nn_member as NM            # noqa: E402
import prev_denom as PD           # noqa: E402
from season_centering import best_cal   # noqa: E402

CACHE = HERE.parent / "results" / "ens5"
OUT_DIR = HERE.parent / "results" / "nn_carrier"

BIS = ["isb_logn", "isb_share", "isb_succ_sm", "isb_succ_d", "isb_mid_sm", "isb_mid_d"]
PFAIL = ["isp_rev_sm", "isp_rev_d", "isp_mid_sm", "isp_mid_d",
         "isp_ball_sm", "isp_ball_d", "isp_strk_sm", "isp_strk_d"]


def train_lin(XA, y, fit, val):
    st, oh = NM.prep_fit(XA[fit]), NM.onehot_fit(XA[fit])
    Z = np.concatenate([NM.prep_apply(XA[fit], st), NM.onehot_apply(XA[fit], oh)], axis=1)
    model, dev = NM.train_mlp(Z, y[fit], hidden=(), dropout=0.0, epochs=60,
                              lr=1e-3, wd=1e-4, seed=0)
    Zv = np.concatenate([NM.prep_apply(XA[val], st), NM.onehot_apply(XA[val], oh)], axis=1)
    return np.clip(NM.predict_mlp(model, dev, Zv), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024,2023,2022")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(">> train.csv 로드 + 블록 생성 (K_IS=100)")
    L.K_IS = 100.0
    df = rd.load_train()
    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy())
    X = L.build_features(df, base=(nb, kb)).reset_index(drop=True)
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()

    blocks = {}
    for tag, ent, ncol, rc, K in IF.BLOCKS:
        lut2 = IF.end_lookup(df, ent, ncol, rc)
        blocks[tag] = IF.build_block(df, tag, ent, ncol, rc, K, lut2)
    B_bis = blocks["b"][BIS].reset_index(drop=True)
    B_pf = blocks["p"][PFAIL].reset_index(drop=True)
    B_dn = PD.build_block(df)[PD.DENOM_COLS].reset_index(drop=True)
    print(f"   bis {B_bis.shape[1]} · pfail {B_pf.shape[1]} · denom {B_dn.shape[1]}")

    arms = [("base", None),
            ("+bis", B_bis),
            ("+pfail", B_pf),
            ("+bis+pfail", pd.concat([B_bis, B_pf], axis=1)),
            ("+전부", pd.concat([B_bis, B_pf, B_dn], axis=1))]

    rng = np.random.default_rng(11)
    res = []
    for v in [int(s) for s in args.vals.split(",")]:
        fit, val = (season == v - 1), (season == v)
        yv = y[val]
        z = np.load(CACHE / f"preds_val{v}.npz")
        others = sum(z[m] * w for m, w in EP.ENS4_W.items() if m != "nn_lin")
        w_lin = EP.ENS4_W["nn_lin"]
        s_ref = best_cal(yv, others + w_lin * z["nn_lin"])[0]
        print(f"\n[val {v}]  ENS-4 캐시 기준 {s_ref:.2f}")
        for name, B in arms:
            XA = X if B is None else pd.concat([X, B], axis=1)
            p_lin = train_lin(XA, y, fit, val)
            s_ens = best_cal(yv, others + w_lin * p_lin)[0]
            rec = dict(val=v, arm=name, lin=best_cal(yv, p_lin)[0],
                       d_ens=s_ens - s_ref, d_pl=np.nan)
            if name == "+bis+pfail":                       # 최상 후보에만 셔플 위약 2시드
                ds = []
                for sd in range(2):
                    Bs = B.sample(frac=1, random_state=sd).reset_index(drop=True)
                    ps = train_lin(pd.concat([X, Bs], axis=1), y, fit, val)
                    ds.append(best_cal(yv, others + w_lin * ps)[0] - s_ref)
                rec["d_pl"] = float(np.mean(ds))
            res.append(rec)
            print(f"  {name:12s} nn_lin {rec['lin']:8.2f}  **ΔENS {rec['d_ens']:+7.2f}**"
                  + (f"  (위약 {rec['d_pl']:+.2f})" if not np.isnan(rec["d_pl"]) else ""))

    t = pd.DataFrame(res)
    print("\n" + t.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    g = t[t.arm != "base"].groupby("arm").agg(
        d_mean=("d_ens", "mean"), d_min=("d_ens", "min"),
        pos=("d_ens", lambda s: int((s > 0).sum())))
    print("\n[ENS 증분 요약 — V23은 스트레스 테스트, 판정은 V24·V22 중심]")
    print(g.sort_values("d_mean", ascending=False).to_string(float_format=lambda x: f"{x:.2f}"))
    (OUT_DIR / "gate.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    print(f"\n({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
