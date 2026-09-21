# -*- coding: utf-8 -*-
"""이중조사 오라클 — train 연속 행 차분으로 **행별 실제 이벤트(구종군·실패유형) 복원** 검증
+ "현재 구종군을 안다면" 상한 측정 (진단 전용 — 오라클 피처는 §5·§6 금지 정보).

    python sweep/auxlabel_oracle.py --vals 2024

## 검증 대상 주장 (Codex #1, 미검증)
같은 (투수,시즌) 연속 행에서 `k_t = rint(rate_t·n_t)` 차분으로 각 투구의 이벤트가 복원된다:
success 이벤트 = 제공 y와 100% 일치 · 구종군(fb/br/os) 이벤트 합 = 정확히 1 · 커버 ~99.9%.

## 오라클 판정 규칙 (사전 등록)
TYPE3(실제 구종군 원핫)을 nn_lin 캐리어에 추가한 ENS 증분 < +20 이면
"지도된 구종 MoE/expert" 채널 전체를 닫는다 (단순 MoE ≈0은 이미 실측됨).
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


def recover_events(df):
    """(투수,시즌) 내 asof_pitcher_n 정렬 → 다음 행과의 k 차분으로 현재 투구 이벤트 복원."""
    n_all = df["asof_pitcher_n"].to_numpy(dtype=float)
    nmix = df["asof_pitcher_pitchmix_n"].to_numpy(dtype=float)
    y = df[rd.TARGET].to_numpy(dtype=float)

    k = {}
    for c, nm, base in [("asof_pitcher_success_rate", "succ", "n"),
                        ("asof_pitcher_reverse_rate", "rev", "n"),
                        ("asof_pitcher_middle_rate", "mid", "n"),
                        ("asof_pitcher_ball_rate", "ball", "n"),
                        ("asof_pitcher_strike_rate", "strk", "n"),
                        ("asof_pitcher_fastball_rate", "fb", "m"),
                        ("asof_pitcher_breaking_rate", "br", "m"),
                        ("asof_pitcher_offspeed_rate", "os", "m")]:
        base_n = n_all if base == "n" else nmix
        k[nm] = np.rint(df[c].fillna(0.0).to_numpy(dtype=float) * base_n)

    order = df.sort_values(["pitcher_id", rd.SEASON, "asof_pitcher_n"],
                           kind="stable").index.to_numpy()
    pos = np.empty(len(df), dtype=np.int64)
    pos[order] = np.arange(len(df))
    pid_o = df["pitcher_id"].to_numpy()[order]
    ssn_o = df[rd.SEASON].to_numpy()[order]
    n_o = n_all[order]
    nmix_o = nmix[order]
    same_next = (pid_o[1:] == pid_o[:-1]) & (ssn_o[1:] == ssn_o[:-1])
    dn = n_o[1:] - n_o[:-1]
    dm = nmix_o[1:] - nmix_o[:-1]
    consec = same_next & (dn == 1)

    ev = {}
    for nm, arr in k.items():
        a = arr[order]
        d = a[1:] - a[:-1]
        e = np.full(len(df), np.nan)
        e[:-1][consec] = d[consec]
        out = np.full(len(df), np.nan)
        out[order] = e
        ev[nm] = out
    cover = np.zeros(len(df), dtype=bool)
    tmp = np.zeros(len(df), dtype=bool)
    tmp[:-1] = consec
    cover[order] = tmp
    mix_consec = np.zeros(len(df), dtype=bool)
    tmp2 = np.zeros(len(df), dtype=bool)
    tmp2[:-1] = same_next & (dm == 1)
    mix_consec[order] = tmp2
    return ev, cover, mix_consec, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(">> 준비 (K_IS=100)")
    L.K_IS = 100.0
    df = rd.load_train()
    ev, cover, mix_ok, y_all = recover_events(df)
    n_cov = int(cover.sum())
    print(f"\n[복원 검증] 연속(Δn=1) 행 {n_cov:,} / {len(df):,} = {n_cov/len(df):.4%}")
    m = cover
    succ_match = float((ev["succ"][m] == y_all[m]).mean())
    print(f"  success 이벤트 vs 제공 y 일치율: {succ_match:.6f}")
    frac_01 = {nm: float(np.isin(ev[nm][m], [0.0, 1.0]).mean())
               for nm in ["succ", "rev", "mid", "ball", "strk"]}
    print("  0/1 비율:", {k2: round(v, 5) for k2, v in frac_01.items()})
    mm = cover & mix_ok
    mix_sum = ev["fb"][mm] + ev["br"][mm] + ev["os"][mm]
    print(f"  구종군 이벤트 합=1 비율: {float((mix_sum == 1.0).mean()):.6f} "
          f"(mix 연속 행 {int(mm.sum()):,})")

    if succ_match < 0.999:
        print("🚫 success 일치율 미달 — 복원 주장 기각, 오라클 중단")
        return

    print("\n>> 오라클 arm 준비")
    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy())
    X = L.build_features(df, base=(nb, kb)).reset_index(drop=True)
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()
    for tag_, ent_, ncol_, rc_, K_ in IF.BLOCKS:
        if tag_ == "b":
            lb = IF.end_lookup(df, ent_, ncol_, rc_)
            B_bis = IF.build_block(df, tag_, ent_, ncol_, rc_, K_, lb)[BIS].reset_index(drop=True)

    TYPE3 = pd.DataFrame({f"orc_{nm}": np.nan_to_num(ev[nm], nan=0.0).astype("float32")
                          for nm in ["fb", "br", "os"]})
    FTYPE = pd.DataFrame({f"orc_{nm}": np.nan_to_num(ev[nm], nan=0.0).astype("float32")
                          for nm in ["rev", "mid", "ball", "strk"]})

    res = []
    for v in [int(s) for s in args.vals.split(",")]:
        fit, val = (season == v - 1), (season == v)
        P, corr, yv = get_members(df, X, y, season, v)
        others = sum(P[m2] * w for m2, w in ENS7_W.items() if m2 != "nn_lin_bis")
        w_lin = ENS7_W["nn_lin_bis"]
        s_ref = deploy_score(yv, others + w_lin * P["nn_lin_bis"] + TM_W * corr)
        print(f"\n[val {v}] ENS-7 동결캘리 기준 {s_ref:.2f}")
        arms = [("bis(기준)", None), ("+TYPE3(오라클)", TYPE3),
                ("+FTYPE4(오라클)", FTYPE), ("+둘다(오라클)", pd.concat([TYPE3, FTYPE], axis=1))]
        for name, B in arms:
            t1 = time.time()
            XA = pd.concat([X, B_bis], axis=1) if B is None else pd.concat([X, B_bis, B], axis=1)
            p_lin = train_lin(XA, y, fit, val)
            s = deploy_score(yv, others + w_lin * p_lin + TM_W * corr)
            res.append(dict(val=v, arm=name, ncol=XA.shape[1], d=s - s_ref))
            print(f"  {name:16s} ({XA.shape[1]:3d}col)  Δ {s-s_ref:+7.2f}  [{time.time()-t1:.0f}s]")

    t = pd.DataFrame(res)
    (OUT_DIR / "auxlabel_oracle.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    print(f"\n판정 규칙: TYPE3 오라클 Δ < +20 → 지도 MoE/expert 채널 폐쇄")
    print(f">> 저장 {OUT_DIR/'auxlabel_oracle.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
