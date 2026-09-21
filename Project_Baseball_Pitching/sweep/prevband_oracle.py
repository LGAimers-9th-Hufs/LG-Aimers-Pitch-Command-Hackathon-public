# -*- coding: utf-8 -*-
"""이중조사 오라클 2 — prev 밴드 분해의 **진짜 분모 상한** (C1 공동분모 트랙 킬-테스트).

    python sweep/prevband_oracle.py --vals 2024

## 사전 등록 판정 규칙
train 등판 복원으로 만든 **진짜** 밴드 통계(직전 1 / 2~3 / 4~5 등판 분리 + 진짜 분모)를
nn_lin 캐리어에 얹은 ENS 증분이 **+10 미만이면 C1 공동분모 solver 트랙 전체 폐쇄**
(실현 버전은 여기에 복원 오차가 얹히므로 항상 이보다 나쁘다).

- arm A "+truedenom6": 기존 C1 피처 형태(pn{1,3,5}_log·pr{1,3,5}_shrunk)를 진짜 n으로
  → 82.5% 복원 solver의 상한 (실측 ENS +0.5/+1.7/−2.5와 비교).
- arm B "+bands": 밴드 분해(마지막 1 / 2~3 / 4~5 등판 각각 n·수축률) + 추세·워크로드
  → Codex #1 deconvolution 제안의 상한.
※ 오라클은 진단 전용이 아니라 **§5 합법 정보의 상한**이다(train 시점 등판 통계는 원리상
  prev1/3/5 6비율에서 부분 복원 가능한 값) — 단 실현하려면 solver가 필요.
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
import trackman as TM             # noqa: E402
from nn_carrier import BIS, train_lin   # noqa: E402
from eb_carrier import get_members, deploy_score, ENS7_W, TM_W  # noqa: E402

OUT_DIR = HERE.parent / "results" / "dual"


def band_features(df):
    """등판 복원 → (pid, season, app_no)별 진짜 (n, k) → 행별 밴드 피처."""
    tr = TM.add_appearances(df)
    app = tr.groupby(["pitcher_id", rd.SEASON, "app_no"]).agg(
        n=("row_id", "size"), k=(rd.TARGET, "sum")).reset_index()
    app = app.sort_values(["pitcher_id", rd.SEASON, "app_no"], kind="stable")

    g = app.groupby(["pitcher_id", rd.SEASON])
    n_sh = {i: g["n"].shift(i) for i in range(1, 6)}
    k_sh = {i: g["k"].shift(i) for i in range(1, 6)}

    def band(idxs, K):
        n = sum(n_sh[i] for i in idxs)
        k = sum(k_sh[i] for i in idxs)
        return (np.log1p(n).astype("float32"),
                ((k + K * 0.5) / (n + K)).astype("float32"), n)

    F = pd.DataFrame({c: app[c] for c in ["pitcher_id", rd.SEASON, "app_no"]})
    n1l, r1, n1 = band([1], 20.0)
    n23l, r23, n23 = band([2, 3], 40.0)
    n45l, r45, n45 = band([4, 5], 40.0)
    # arm A: 기존 C1 형태 (누적 윈도우 1 / 1~3 / 1~5, 진짜 분모)
    for w, idxs in ((1, [1]), (3, [1, 2, 3]), (5, [1, 2, 3, 4, 5])):
        nl, rr, nn = band(idxs, 20.0 * w)
        F[f"tp{w}_log"] = nl
        F[f"tr{w}_shrunk"] = rr
    # arm B: 밴드 분해
    F["b1_log"], F["b1_r"] = n1l, r1
    F["b23_log"], F["b23_r"] = n23l, r23
    F["b45_log"], F["b45_r"] = n45l, r45
    F["b_trend"] = (r1 - r45).astype("float32")
    F["b_workload"] = (n1 - (n1 + n23 + n45) / 5.0).astype("float32")

    cols = [c for c in F.columns if c not in ("pitcher_id", rd.SEASON, "app_no")]
    M = tr[["pitcher_id", rd.SEASON, "app_no"]].merge(
        F, on=["pitcher_id", rd.SEASON, "app_no"], how="left")
    B = M[cols].reset_index(drop=True)
    # tr은 df와 같은 행 순서(add_appearances가 순서 보존한다고 가정 — 검증)
    assert len(B) == len(df)
    return B


TRUED = ["tp1_log", "tr1_shrunk", "tp3_log", "tr3_shrunk", "tp5_log", "tr5_shrunk"]
BANDS = ["b1_log", "b1_r", "b23_log", "b23_r", "b45_log", "b45_r", "b_trend", "b_workload"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(">> 준비 (K_IS=100)")
    L.K_IS = 100.0
    df = rd.load_train()
    B_bands = band_features(df)
    print(f"   밴드 블록 {B_bands.shape[1]}컬럼 · b1_r 결측 {B_bands['b1_r'].isna().mean():.1%} "
          f"· b45_r 결측 {B_bands['b45_r'].isna().mean():.1%} [{time.time()-t0:.0f}s]")
    # 검증: 진짜 prev1 성공률(비수축, K=0으로 재계산)이 공식 prev1과 일치하는가
    tr1_raw_n = np.expm1(B_bands["tp1_log"].to_numpy(dtype=float))
    off = df["asof_pitcher_prev1_game_success_rate"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        raw_r1 = (B_bands["tr1_shrunk"].to_numpy(dtype=float) * (tr1_raw_n + 20.0) - 10.0) / \
            np.maximum(tr1_raw_n, 1e-9)
    both = ~(np.isnan(off) | np.isnan(raw_r1)) & (tr1_raw_n > 0)
    match = float((np.abs(raw_r1[both] - off[both]) < 1e-4).mean())
    print(f"   [정의 검증] 진짜 직전등판 성공률 vs 공식 prev1: 일치 {match:.2%} (표본 {both.sum():,})")

    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy())
    X = L.build_features(df, base=(nb, kb)).reset_index(drop=True)
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()
    for tag_, ent_, ncol_, rc_, K_ in IF.BLOCKS:
        if tag_ == "b":
            lb = IF.end_lookup(df, ent_, ncol_, rc_)
            B_bis = IF.build_block(df, tag_, ent_, ncol_, rc_, K_, lb)[BIS].reset_index(drop=True)

    res = []
    for v in [int(s) for s in args.vals.split(",")]:
        fit, val = (season == v - 1), (season == v)
        P, corr, yv = get_members(df, X, y, season, v)
        others = sum(P[m] * w for m, w in ENS7_W.items() if m != "nn_lin_bis")
        w_lin = ENS7_W["nn_lin_bis"]
        s_ref = deploy_score(yv, others + w_lin * P["nn_lin_bis"] + TM_W * corr)
        print(f"\n[val {v}] ENS-7 동결캘리 기준 {s_ref:.2f}")
        for name, cols in [("bis(기준)", None), ("+truedenom6", TRUED),
                           ("+bands8", BANDS), ("+둘다", TRUED + BANDS)]:
            t1 = time.time()
            XA = pd.concat([X, B_bis], axis=1) if cols is None \
                else pd.concat([X, B_bis, B_bands[cols]], axis=1)
            p_lin = train_lin(XA, y, fit, val)
            s = deploy_score(yv, others + w_lin * p_lin + TM_W * corr)
            res.append(dict(val=v, arm=name, ncol=XA.shape[1], d=s - s_ref))
            print(f"  {name:14s} ({XA.shape[1]:3d}col)  Δ {s-s_ref:+7.2f}  [{time.time()-t1:.0f}s]")

    t = pd.DataFrame(res)
    (OUT_DIR / "prevband_oracle.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    print(f"\n판정 규칙: 진짜-분모 오라클 Δ < +10 → C1 공동분모 solver 트랙 폐쇄")
    print(f">> 저장 {OUT_DIR/'prevband_oracle.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
