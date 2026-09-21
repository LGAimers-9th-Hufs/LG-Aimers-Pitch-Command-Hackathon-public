# -*- coding: utf-8 -*-
"""bis 성분 증거 완결 — V21 감사 폴드에서 nn_lin(구, bis 없음) 대비 증분 측정.

    python sweep/bis_v21_check.py

Codex #2 결함 지적 ①: bis 채택 증거(V24 +19.15 / V23 +70.28 / V22 +3.41, 평균 30.94 <
SD 35.1)가 평균>SD 미달 — V21 비교군(+24 이상이면 4폴드 역전)이 없다. 이 스크립트가 채운다.
ENS-7 가중·캘리·타 멤버 전부 동결, nn_lin_bis ↔ nn_lin(57컬럼)만 교체.
"""
from __future__ import annotations
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
from nn_carrier import train_lin  # noqa: E402
from eb_carrier import get_members, deploy_score, ENS7_W, TM_W  # noqa: E402

OUT_DIR = HERE.parent / "results" / "dual"


def main():
    t0 = time.time()
    print(">> 준비 (K_IS=100)")
    L.K_IS = 100.0
    df = rd.load_train()
    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy())
    X = L.build_features(df, base=(nb, kb)).reset_index(drop=True)
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()

    res = []
    for v in (2021, 2022, 2023, 2024):
        fit, val = (season == v - 1), (season == v)
        P, corr, yv = get_members(df, X, y, season, v)
        others = sum(P[m] * w for m, w in ENS7_W.items() if m != "nn_lin_bis")
        w_lin = ENS7_W["nn_lin_bis"]
        s_bis = deploy_score(yv, others + w_lin * P["nn_lin_bis"] + TM_W * corr)
        p_old = train_lin(X, y, fit, val)          # 57컬럼, bis 없음
        s_old = deploy_score(yv, others + w_lin * p_old + TM_W * corr)
        res.append(dict(val=v, bis=round(s_bis, 2), old=round(s_old, 2),
                        d_bis=round(s_bis - s_old, 2)))
        print(f"[V{v}] bis {s_bis:9.2f}  vs nn_lin(구) {s_old:9.2f}  → bis 증분 {s_bis-s_old:+7.2f}")

    t = pd.DataFrame(res)
    core = t[t.val.isin([2024, 2023, 2022])]["d_bis"]
    all4 = t["d_bis"]
    print(f"\n3폴드(24/23/22): 평균 {core.mean():+.2f} · SD {core.std():.2f} · "
          f"통과 {core.mean() > 0 and core.mean() > core.std()}")
    print(f"4폴드(+21):      평균 {all4.mean():+.2f} · SD {all4.std():.2f} · "
          f"통과 {all4.mean() > 0 and all4.mean() > all4.std()}")
    (OUT_DIR / "bis_v21_check.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    print(f">> 저장 {OUT_DIR/'bis_v21_check.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
