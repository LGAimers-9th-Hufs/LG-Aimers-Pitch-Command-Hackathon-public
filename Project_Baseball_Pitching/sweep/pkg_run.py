# -*- coding: utf-8 -*-
"""연장 라운드 1 — 검증된 양수 조각의 **패키지 결합 run**: bis + pEB + cxp 를 한 번에.

    python sweep/pkg_run.py --vals 2024,2023,2022,2021

개별 실측(전부 +10 미달이나 위약 분리 확인된 것): pEB 추가분 +2.51/+0.62/+3.39/+2.30 (Codex#2
차분 재해석) · cxp +2.12/+7.23/+2.94/−0.40 (D-43). 규칙 "패키지 Δ는 결합 run 실측만 인정"에 따라
합산 가능액을 직접 잰다. 위약 = pEB(투수 경로 교환) + cxp(행 셔플) 동시, 컬럼 수 매칭 15컬럼, 2시드.
"""
from __future__ import annotations
import argparse
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
from nn_carrier import BIS, train_lin   # noqa: E402
from eb_carrier import (get_members, deploy_score, ENS7_W, TM_W,   # noqa: E402
                        eb_block, entity_traj_map)
from interaction_carrier import build_products   # noqa: E402

OUT_DIR = HERE.parent / "results" / "dual"
P_SUCC = [("asof_pitcher_success_rate", "succ")]


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

    Bps, _, _ = eb_block(df, "pitcher_id", "asof_pitcher_n", P_SUCC, 100, "ebps")
    PR = build_products(df, X, B_bis)
    CXP = [c for c in PR.columns if c.startswith("cxp_")]
    PKG = pd.concat([Bps, PR[CXP]], axis=1)
    print(f"   pEB {Bps.shape[1]} + cxp {len(CXP)} = 패키지 {PKG.shape[1]}컬럼")

    PL = {}
    for sd_ in range(2):
        emap = entity_traj_map(df, "pitcher_id", 31 + sd_)
        Bps_pl, _, _ = eb_block(df, "pitcher_id", "asof_pitcher_n", P_SUCC, 100, "ebps", emap)
        rng = np.random.default_rng(41 + sd_)
        PR_pl = PR[CXP].iloc[rng.permutation(len(PR))].reset_index(drop=True)
        PL[sd_] = pd.concat([Bps_pl, PR_pl], axis=1)

    arms = [("bis(기준)", None), ("+pEB", Bps), ("+cxp", PR[CXP]), ("+패키지", PKG),
            ("+패키지_위약0", PL[0]), ("+패키지_위약1", PL[1])]

    res = []
    for v in [int(s) for s in args.vals.split(",")]:
        fit, val = (season == v - 1), (season == v)
        P, corr, yv = get_members(df, X, y, season, v)
        others = sum(P[m] * w for m, w in ENS7_W.items() if m != "nn_lin_bis")
        w_lin = ENS7_W["nn_lin_bis"]
        s_ref = deploy_score(yv, others + w_lin * P["nn_lin_bis"] + TM_W * corr)
        print(f"\n[val {v}] ENS-7 기준 {s_ref:.2f}")
        for name, B in arms:
            XA = pd.concat([X, B_bis], axis=1) if B is None else pd.concat([X, B_bis, B], axis=1)
            p_lin = train_lin(XA, y, fit, val)
            d = deploy_score(yv, others + w_lin * p_lin + TM_W * corr) - s_ref
            res.append(dict(val=v, arm=name, ncol=XA.shape[1], d=round(float(d), 3)))
            print(f"  {name:14s} ({XA.shape[1]:3d}col)  Δ {d:+7.2f}")

    t = pd.DataFrame(res)
    core = t[(~t.arm.str.contains("기준|위약")) & t.val.isin([2024, 2023, 2022])]
    g = core.groupby("arm").agg(d_mean=("d", "mean"), d_sd=("d", "std"), d_min=("d", "min"),
                                pos=("d", lambda s: int((s > 0).sum())))
    g["통과"] = (g.d_mean > 0) & (g.d_mean > g.d_sd)
    print("\n[요약 — V24/23/22 · V21은 감사]")
    print(g.sort_values("d_mean", ascending=False).to_string(float_format=lambda x: f"{x:.2f}"))
    (OUT_DIR / "pkg_run.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR/'pkg_run.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
