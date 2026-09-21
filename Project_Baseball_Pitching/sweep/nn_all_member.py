# -*- coding: utf-8 -*-
"""연장 라운드 2 — **전시즌 학습 nn_lin_bis**를 신규 멤버로 (학습 스코프 다양성의 선형판).

    python sweep/nn_all_member.py --vals 2024,2023,2022,2021

근거: 다양성이 실제로 값을 한 유일한 축 = 학습 스코프(all_raw 단독 319인데 ENS +5.97, 불일치
0.041 최대). 선형 캐리어의 전시즌판(nn_all)은 미측정 셀. 판정 = 동결 ENS-7과의 볼록 블렌드
`(1-w)·ens + w·p_all`, w 격자. 신규 멤버라 위약 불요(귀속이 아니라 블렌드 이득 측정) —
대신 불일치 RMS·단독 점수를 병기해 블렌드 항등식으로 정합 확인.
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
from eb_carrier import get_members, deploy_score, ENS7_W, TM_W  # noqa: E402

OUT_DIR = HERE.parent / "results" / "dual"
WGRID = (0.05, 0.10, 0.15, 0.25)


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
    XA = pd.concat([X, B_bis], axis=1)

    res = []
    for v in [int(s) for s in args.vals.split(",")]:
        fit1, fit_all, val = (season == v - 1), (season < v), (season == v)
        P, corr, yv = get_members(df, X, y, season, v)
        ens_full = sum(P[m] * w for m, w in ENS7_W.items()) + TM_W * corr
        s_ref = deploy_score(yv, ens_full)
        t1 = time.time()
        p_all = train_lin(XA, y, fit_all, val)
        s_solo = deploy_score(yv, p_all)
        srms = float(np.sqrt(np.mean((p_all - ens_full) ** 2)))
        line = dict(val=v, solo=round(s_solo, 2), s_rms=round(srms, 4))
        print(f"\n[val {v}] 기준 {s_ref:.2f} · nn_all 단독 {s_solo:.2f} · 불일치 {srms:.4f} "
              f"[{time.time()-t1:.0f}s]")
        for w in WGRID:
            d = deploy_score(yv, (1 - w) * ens_full + w * p_all) - s_ref
            line[f"w{w:g}"] = round(float(d), 3)
            print(f"  w={w:g}  Δ {d:+7.2f}")
        res.append(line)

    t = pd.DataFrame(res)
    core = t[t.val.isin([2024, 2023, 2022])]
    print("\n[요약 — V24/23/22]")
    for w in WGRID:
        d = core[f"w{w:g}"]
        print(f"  w={w:g}: 평균 {d.mean():+.2f} · SD {d.std():.2f} · min {d.min():+.2f} · "
              f"통과 {d.mean() > 0 and d.mean() > d.std()}")
    (OUT_DIR / "nn_all_member.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR/'nn_all_member.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
