# -*- coding: utf-8 -*-
"""반나절 3 — **구조화된 상호작용 캐리어**: nn_lin에 명시적 곱 기저를 준다.

    python sweep/interaction_carrier.py --vals 2024,2023,2022,2021

## 근거

- "트리 우위 +23은 전부 상호작용"(D-38 해부). 트리는 min_data 2500 granularity로 **강한** 상호작용만
  잡는다 — 약하고 매끄러운 곱 구조는 선형 캐리어에 **명시적 곱 컬럼**으로 줘야 한다.
- 상태 벡터 = **검증된 기존 표현**(EB 재표현은 전 폴드 기각됨 — eb_carrier 결과):
  투수 당해 `is_sm/is_delta`(is4) × 타자 당해 `isb_succ_d/isb_mid_d`(bis) × 카운트/손.
- nn_lin은 현재 카운트 원핫과 상태값을 **따로** 가진다 — 곱은 처음이다.

## 게이트 (D-42 개정판 유지)

ENS-7 가중·캘리 동결(nn_lin_bis 교체 증분만) · 주 판정 = 동결 배포 캘리 ·
폴드 V24/V23/V22/V21(캐시 재사용) · 위약 = 곱 컬럼 행 셔플 1시드(섭동 크기 확인용).
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
import nn_member as NM            # noqa: E402
from nn_carrier import BIS, train_lin   # noqa: E402
from eb_carrier import get_members, deploy_score, ENS7_W, TM_W   # noqa: E402

OUT_DIR = HERE.parent / "results" / "interaction"


def build_products(df, X, B_bis):
    """행 단위 곱 기저. 전부 이미 있는 컬럼들의 곱 — 서빙도 같은 산술."""
    cc = (df["balls_before"].to_numpy() * 3 + df["strikes_before"].to_numpy()).astype(int)
    C = np.zeros((len(df), 12), dtype=np.float32)
    C[np.arange(len(df)), cc] = 1.0
    p_d = np.nan_to_num(X["is_delta"].to_numpy(dtype=float), nan=0.0)
    p_sm = np.nan_to_num(X["is_sm"].to_numpy(dtype=float), nan=0.5) - 0.5
    b_d = np.nan_to_num(B_bis["isb_succ_d"].to_numpy(dtype=float), nan=0.0)
    b_md = np.nan_to_num(B_bis["isb_mid_d"].to_numpy(dtype=float), nan=0.0)
    hand = (df["pitcher_hand"].to_numpy() == df["batter_hand"].to_numpy()).astype(float)

    out = {}
    for j in range(12):                                    # count × 투수 당해혁신
        out[f"cxp_{j}"] = (C[:, j] * p_d).astype("float32")
    for j in range(12):                                    # count × 타자 당해혁신
        out[f"cxb_{j}"] = (C[:, j] * b_d).astype("float32")
    out["pxb"] = (p_d * b_d).astype("float32")             # 투수 × 타자 혁신
    out["pxb_mid"] = (p_d * b_md).astype("float32")
    out["psm_x_bd"] = (p_sm * b_d).astype("float32")
    out["hand_x_pd"] = (hand * p_d).astype("float32")
    out["hand_x_bd"] = (hand * b_d).astype("float32")
    out["hand_x_pxb"] = (hand * p_d * b_d).astype("float32")
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024,2023,2022,2021")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(">> 준비 (K_IS=100)")
    L.K_IS = 100.0
    df = rd.load_train()
    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy())
    X = L.build_features(df, base=(nb, kb)).reset_index(drop=True)
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()
    for tag_, ent_, ncol_, rc_, K_ in IF.BLOCKS:
        if tag_ == "b":
            lb = IF.end_lookup(df, ent_, ncol_, rc_)
            B_bis = IF.build_block(df, tag_, ent_, ncol_, rc_, K_, lb)[BIS].reset_index(drop=True)
    PR = build_products(df, X, B_bis)
    CXP = [c for c in PR.columns if c.startswith("cxp_")]
    CXB = [c for c in PR.columns if c.startswith("cxb_")]
    PXB = ["pxb", "pxb_mid", "psm_x_bd", "hand_x_pd", "hand_x_bd", "hand_x_pxb"]
    print(f"   곱 기저 {PR.shape[1]}컬럼")

    rng = np.random.default_rng(5)
    PR_pl = PR.iloc[rng.permutation(len(PR))].reset_index(drop=True)   # 위약(행 셔플)

    arms = [("bis(기준)", None),
            ("+cxp", CXP), ("+cxb", CXB), ("+pxb계열", PXB),
            ("+전부", CXP + CXB + PXB),
            ("+위약(전부셔플)", "PL")]

    res = []
    for v in [int(s) for s in args.vals.split(",")]:
        fit, val = (season == v - 1), (season == v)
        P, corr, yv = get_members(df, X, y, season, v)
        others = sum(P[m] * w for m, w in ENS7_W.items() if m != "nn_lin_bis")
        w_lin = ENS7_W["nn_lin_bis"]
        s_ref = deploy_score(yv, others + w_lin * P["nn_lin_bis"] + TM_W * corr)
        print(f"\n[val {v}] ENS-7 동결캘리 기준 {s_ref:.2f}")
        for name, cols in arms:
            if cols is None:
                XA = pd.concat([X, B_bis], axis=1)
            elif cols == "PL":
                XA = pd.concat([X, B_bis, PR_pl[CXP + CXB + PXB]], axis=1)
            else:
                XA = pd.concat([X, B_bis, PR[cols]], axis=1)
            p_lin = train_lin(XA, y, fit, val)
            s = deploy_score(yv, others + w_lin * p_lin + TM_W * corr)
            res.append(dict(val=v, arm=name, ncol=XA.shape[1], d=s - s_ref))
            print(f"  {name:16s} ({XA.shape[1]:3d}col)  Δ {s-s_ref:+7.2f}")

    t = pd.DataFrame(res)
    print("\n" + t.pivot_table(index="arm", columns="val", values="d")
          .to_string(float_format=lambda x: f"{x:+.2f}"))
    g = t[~t.arm.str.contains("기준|위약")].groupby("arm").agg(
        d_mean=("d", "mean"), d_min=("d", "min"), pos=("d", lambda s: int((s > 0).sum())))
    print("\n[요약 — V24 주 판정 · V21 감사]")
    print(g.sort_values("d_mean", ascending=False).to_string(float_format=lambda x: f"{x:.2f}"))
    (OUT_DIR / "gate.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    print(f"\n({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
