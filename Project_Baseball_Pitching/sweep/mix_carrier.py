# -*- coding: utf-8 -*-
"""이중조사 스크린 1 — 구종믹스 당해분해(승리 인코딩 0.5-sm)를 nn_lin 캐리어로.

    python sweep/mix_carrier.py --vals 2024            # V24 스크린
    python sweep/mix_carrier.py --vals 2024,2023,2022,2021   # 확정 게이트

## 근거 (사전 등록: 이중조사 계획, Claude 독립 시드 ④ — Codex 결과 수신 전 동결분)

- 구종믹스 당해 블록("m", 성분별 0.5-sm 인코딩)은 **트리 캐리어에서만** 기각됐다
  (inseason_full.py `is4+mix` arm은 fit_pair = june 트리 쌍 — D-28 확장 기각의 실체).
- bis 전례: 같은 성격의 블록이 트리 직삽 3연속 실패 → nn_lin 캐리어에서 +15.9 (C2, D-39).
- eb_carrier의 "ebB100+mix"는 **Dirichlet 재표현**을 (자체가 더 나쁜) ebB100 배경 위에서 잰 것.
  D-43 정산: "0.5-수축 sm은 비율+표본수 결합 인코딩이라 선형 캐리어에 유리" — 그 승리 인코딩
  × nn_lin × bis_old 기준 셀은 미측정.
- 판별 질문 통과: asof mix 3률·pitchmix_n은 (투수,시즌) 안에서 행마다 갱신된다.

## 게이트 (D-42 개정판)

ENS-7 가중·캘리 동결(nn_lin_bis 교체 증분만) · 주 판정 = 동결 배포 캘리(deploy_score) ·
위약 = **투수 entity 경로 교환**(mix LUT 재배정 — 행 셔플 아님, 컬럼 수 매칭) 2시드 ·
요약에 d_sd 포함(D-25 규칙: 3폴드 평균>0 & 평균>폴드간 SD).
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
from eb_carrier import get_members, deploy_score, ENS7_W, TM_W, entity_traj_map  # noqa: E402

OUT_DIR = HERE.parent / "results" / "dual"

MIX8 = ["ism_logn", "ism_share", "ism_fb_sm", "ism_fb_d",
        "ism_br_sm", "ism_br_d", "ism_os_sm", "ism_os_d"]
MIX6 = ["ism_fb_sm", "ism_fb_d", "ism_br_sm", "ism_br_d", "ism_os_sm", "ism_os_d"]
MIXD3 = ["ism_fb_d", "ism_br_d", "ism_os_d"]


def permute_lut(lut, emap):
    """entity 경로 교환: e2가 e의 전 시즌 경로를 받는다 (eb_carrier.eb_block과 동형)."""
    by = {}
    for (e, s_), v in lut.items():
        by.setdefault(e, {})[s_] = v
    out = {}
    for e, hist in by.items():
        e2 = emap.get(e, e)
        for s_, v in hist.items():
            out[(e2, s_)] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024")
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

    B_bis = None
    B_mix = None
    B_pl = {}
    for tag_, ent_, ncol_, rc_, K_ in IF.BLOCKS:
        if tag_ == "b":
            lb = IF.end_lookup(df, ent_, ncol_, rc_)
            B_bis = IF.build_block(df, tag_, ent_, ncol_, rc_, K_, lb)[BIS].reset_index(drop=True)
        if tag_ == "m":
            lm = IF.end_lookup(df, ent_, ncol_, rc_)
            B_mix = IF.build_block(df, tag_, ent_, ncol_, rc_, K_, lm).reset_index(drop=True)
            for sd in range(2):                       # 위약: 투수 경로 교환 (컬럼 수 매칭)
                emap = entity_traj_map(df, ent_, sd)
                B_pl[sd] = IF.build_block(df, tag_, ent_, ncol_, rc_, K_,
                                          permute_lut(lm, emap)).reset_index(drop=True)
    print(f"   bis {B_bis.shape[1]} · mix {B_mix.shape[1]} (+위약 2시드) [{time.time()-t0:.0f}s]")

    arms = [("bis(기준)", None),
            ("+mix8", ("R", MIX8)),
            ("+mix6", ("R", MIX6)),
            ("+mixd3", ("R", MIXD3)),
            ("+mix8_위약0", ("P", 0)),
            ("+mix8_위약1", ("P", 1))]

    res = []
    for v in [int(s) for s in args.vals.split(",")]:
        fit, val = (season == v - 1), (season == v)
        P, corr, yv = get_members(df, X, y, season, v)
        others = sum(P[m] * w for m, w in ENS7_W.items() if m != "nn_lin_bis")
        w_lin = ENS7_W["nn_lin_bis"]
        s_ref = deploy_score(yv, others + w_lin * P["nn_lin_bis"] + TM_W * corr)
        print(f"\n[val {v}] ENS-7 동결캘리 기준 {s_ref:.2f}")
        for name, spec in arms:
            t1 = time.time()
            if spec is None:
                XA = pd.concat([X, B_bis], axis=1)
            elif spec[0] == "R":
                XA = pd.concat([X, B_bis, B_mix[spec[1]]], axis=1)
            else:
                XA = pd.concat([X, B_bis, B_pl[spec[1]][MIX8]], axis=1)
            p_lin = train_lin(XA, y, fit, val)
            s = deploy_score(yv, others + w_lin * p_lin + TM_W * corr)
            res.append(dict(val=v, arm=name, ncol=XA.shape[1], d=s - s_ref))
            print(f"  {name:14s} ({XA.shape[1]:3d}col)  Δ {s-s_ref:+7.2f}  [{time.time()-t1:.0f}s]")

    t = pd.DataFrame(res)
    print("\n" + t.pivot_table(index="arm", columns="val", values="d")
          .to_string(float_format=lambda x: f"{x:+.2f}"))
    core = t[(~t.arm.str.contains("기준|위약")) & t.val.isin([2024, 2023, 2022])]
    g = core.groupby("arm").agg(
        d_mean=("d", "mean"), d_sd=("d", "std"), d_min=("d", "min"),
        pos=("d", lambda s: int((s > 0).sum())), n=("d", "size"))
    if g["n"].max() >= 3:
        g["통과"] = (g.d_mean > 0) & (g.d_mean > g.d_sd)
    print("\n[요약 — V24/23/22 (V21은 감사 폴드, 평균 제외)]")
    print(g.sort_values("d_mean", ascending=False).to_string(float_format=lambda x: f"{x:.2f}"))
    (OUT_DIR / "mix_gate.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR/'mix_gate.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
