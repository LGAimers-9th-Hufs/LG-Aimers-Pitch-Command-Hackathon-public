# -*- coding: utf-8 -*-
"""C1 — prev1/3/5 rate의 **숨은 분모 복원** + 홈팀/매치업 블록 (Codex 감사 발견의 검증·구현).

    python sweep/prev_denom.py --audit                 # Codex 주장 재현 (분모 복원율 ≈83%?)
    python sweep/prev_denom.py --gate                  # june 캐리어 3폴드 + 위약
    python sweep/prev_denom.py --ens                   # nn_lin 캐리어 교체 → ENS 증분

## 발상 (Codex, 2026-08-08 상담)

공식 컬럼은 직전 1/3/5경기의 성공률·한복판률을 주지만 **투구 수는 주지 않는다**. 그런데 같은
윈도우의 두 비율은 **같은 정수 분모를 공유**한다:

    n̂ = min{ n ≥ 1 : n·succ_rate 와 n·middle_rate 가 동시에 정수(±ε) }

n̂이 실분모이거나 그 약수다. **추론은 그 행의 rate 6개만 쓰므로 §5 완전 적법**이고,
최근 워크로드(직전 등판 투구수 = 선발/불펜)·prev율 신뢰도는 **행마다 변하는 값**이며
현재 모델이 전혀 못 보는 정보다.

## 감사 설계 (Codex 수치는 재현 전 신뢰하지 않는다)

`trackman.add_appearances`로 등판 경계 복원 → 각 (투수, 등판 k)의 실제 투구수·성공률 계산 →
다음 등판 첫 행의 `prev1_game_success_rate`와 대조(윈도우 정의 검증) → 분모 복원 정확도 측정.
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
import ens5_pool as EP            # noqa: E402
import nn_member as NM            # noqa: E402
from ens4_weights import grid_calib   # noqa: E402
from season_centering import best_cal, fit_pair, predict_pair   # noqa: E402

CACHE = HERE.parent / "results" / "ens5"
OUT_DIR = HERE.parent / "results" / "prev_denom"
NMAX = 160            # 한 등판 최대 투구수 상한(감사 실측 max 151) — prev3/5는 3·5배
EPS = 1e-6


def denom_hat(r1: np.ndarray, r2: np.ndarray, nmax: int, chunk=100_000) -> np.ndarray:
    """두 비율이 동시에 정수가 되는 최소 분모. (N,)×(nmax,) 행렬을 청크로 처리
    (w=5는 nmax 800 × 1.47M행 = 9GB라 통짜 벡터화가 불가)."""
    n = np.arange(1, nmax + 1, dtype=np.float64)
    tol = EPS * n[None, :]
    out = np.empty(len(r1))
    for i in range(0, len(r1), chunk):
        a = r1[i:i + chunk, None] * n[None, :]
        b = r2[i:i + chunk, None] * n[None, :]
        both = (np.abs(a - np.rint(a)) < tol) & (np.abs(b - np.rint(b)) < tol)
        has = both.any(axis=1)
        out[i:i + chunk] = np.where(has, both.argmax(axis=1) + 1, np.nan)
    return out


# ---------------------------------------------------------------- 감사
def audit():
    import trackman as TM
    print(">> train.csv 로드 + 등판 복원")
    df = rd.load_train()
    tr = TM.add_appearances(df)
    # 등판 단위 실측: 투구수 · 성공률 · 한복판률(라벨은 success만 있으므로 middle은 rate 역산 불가 —
    # 대신 **다음 등판의 prev1 rate가 이 등판의 실측 성공률과 일치하는지**로 윈도우 정의를 검증한다)
    app = tr.groupby(["pitcher_id", "app_no"]).agg(
        n=("row_id", "size"), succ=(rd.TARGET, "mean"),
        season=(rd.SEASON, "first")).reset_index()
    print(f"   등판 {len(app):,}개 (시즌 경계 무시 전 기준)")

    # 다음 등판 첫 행의 prev1 rate 추출
    first_rows = tr.groupby(["pitcher_id", "app_no"]).first().reset_index()
    first_rows = first_rows[["pitcher_id", "app_no",
                             "asof_pitcher_prev1_game_success_rate",
                             "asof_pitcher_prev1_game_middle_rate", rd.SEASON]]
    app["app_next"] = app["app_no"] + 1
    m = app.merge(first_rows, left_on=["pitcher_id", "app_next"],
                  right_on=["pitcher_id", "app_no"], suffixes=("", "_nx"))
    same_season = m[rd.SEASON] == m[f"{rd.SEASON}_nx"]
    m = m[same_season & m["asof_pitcher_prev1_game_success_rate"].notna()]
    # ⚠ CSV가 유효숫자 6자리로 반올림 저장('0.52381'=11/21 실측) → 허용오차 1e-5
    match = np.abs(m["asof_pitcher_prev1_game_success_rate"] - m["succ"]) < 1e-5
    print(f"   [윈도우 검증] 다음 등판 prev1 == 이번 등판 실측 성공률: "
          f"{match.mean():.1%}  ({match.sum():,}/{len(m):,})")

    # 분모 복원: prev1 (succ_rate, middle_rate) → n̂ vs 실제 n
    r1 = m["asof_pitcher_prev1_game_success_rate"].to_numpy(dtype=np.float64)
    r2 = m["asof_pitcher_prev1_game_middle_rate"].to_numpy(dtype=np.float64)
    ok = ~np.isnan(r2)
    nh = denom_hat(r1[ok], r2[ok], NMAX)
    ntrue = m["n"].to_numpy(dtype=float)[ok]
    exact = np.nanmean(nh == ntrue)
    divisor = np.nanmean((ntrue % np.where(np.isnan(nh), 1, nh)) == 0)
    print(f"   [분모 복원 prev1] 정확 {exact:.1%} · 약수 관계 {divisor:.1%} · "
          f"복원실패(NaN) {np.isnan(nh).mean():.1%}  (Codex 주장: 정확 ≈82.5%)")
    print(f"   실제 n 분포: 중앙값 {np.median(ntrue):.0f} · q95 {np.quantile(ntrue, .95):.0f} · "
          f"최대 {ntrue.max():.0f}")
    return exact


# ---------------------------------------------------------------- 피처 블록
def build_block(df) -> pd.DataFrame:
    """전부 그 행의 값만으로 계산(§5 적법). NaN(데뷔 초기) → n̂ NaN 유지."""
    out = {}
    for w, nmax in ((1, NMAX), (3, 3 * NMAX), (5, 5 * NMAX)):
        r1 = df[f"asof_pitcher_prev{w}_game_success_rate"].to_numpy(dtype=np.float64)
        r2 = df[f"asof_pitcher_prev{w}_game_middle_rate"].to_numpy(dtype=np.float64)
        valid = ~(np.isnan(r1) | np.isnan(r2))
        nh = np.full(len(df), np.nan)
        if valid.any():
            nh[valid] = denom_hat(r1[valid], r2[valid], nmax)
        out[f"pn{w}_log"] = np.log1p(nh).astype("float32")
        # 표본수 반영 수축률 (K=20·w): n̂이 작을수록 prev율을 0.5로 끌어당김
        K = 20.0 * w
        out[f"pr{w}_shrunk"] = np.where(
            np.isnan(nh) | np.isnan(r1), np.nan,
            (r1 * nh + K * 0.5) / (nh + K)).astype("float32")
    n1 = np.expm1(out["pn1_log"].astype(np.float64))
    n5 = np.expm1(out["pn5_log"].astype(np.float64))
    out["workload_delta"] = (n1 - n5 / 5.0).astype("float32")     # 직전 등판 − 평균 등판
    out["starter_proxy"] = (n1 >= 60).astype("float32")           # 선발 프록시(감사 분포로 조정)
    B = pd.DataFrame(out)

    # 홈팀/매치업 (행 단위 산술 — top_bottom, 팀 ID만 사용)
    is_top = (df["top_bottom"].astype(str) == "T").to_numpy()
    pt = df["pitcher_team_id"].to_numpy()
    bt = df["batter_team_id"].to_numpy()
    B["home_team"] = np.where(is_top, pt, bt).astype("float32")
    B["matchup"] = (pt * 30 + bt).astype("float32")
    return B


DENOM_COLS = ["pn1_log", "pn3_log", "pn5_log", "pr1_shrunk", "pr3_shrunk", "pr5_shrunk",
              "workload_delta", "starter_proxy"]
HOME_COLS = ["home_team", "matchup"]


# ---------------------------------------------------------------- 게이트
def gate(args, df, X, B, y, season):
    rng = np.random.default_rng(7)
    arms = [("base", []), ("denom", DENOM_COLS), ("home", HOME_COLS),
            ("denom+home", DENOM_COLS + HOME_COLS)]
    res = []
    for v in [int(s) for s in args.vals.split(",")]:
        fit, val = (season == v - 1), (season == v)
        yv = y[val]
        print(f"\n[val {v}]  fit {fit.sum():,} → val {val.sum():,}")
        base_p = None
        for name, cols in arms:
            XA = X if not cols else pd.concat([X, B[cols]], axis=1)
            p = np.clip(predict_pair(fit_pair(XA[fit], y[fit], None), XA[val]), 0, 1)
            if base_p is None:
                base_p = p
            d_tot = L.score(yv, L.calibrate(p)) - L.score(yv, L.calibrate(base_p))
            d_pl = np.nan
            if cols:                                            # 위약: 블록 행 셔플 3시드
                ds = []
                for sd in range(3):
                    Bs = B[cols].sample(frac=1, random_state=sd).reset_index(drop=True)
                    XS = pd.concat([X, Bs], axis=1)
                    ps = np.clip(predict_pair(fit_pair(XS[fit], y[fit], None), XS[val]), 0, 1)
                    ds.append(L.score(yv, L.calibrate(ps)) - L.score(yv, L.calibrate(base_p)))
                d_pl = float(np.mean(ds))
            res.append(dict(val=v, arm=name, ncol=XA.shape[1], d_total=d_tot, d_placebo=d_pl,
                            d_info=d_tot - d_pl if cols else 0.0))
            print(f"  {name:12s} ({XA.shape[1]:2d}col)  Δtotal {d_tot:+8.2f}  "
                  f"Δplacebo {d_pl:+8.2f}  **Δinfo {d_tot-d_pl if cols else 0:+8.2f}**")
    t = pd.DataFrame(res)
    print("\n" + t.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    g = t[t.arm != "base"].groupby("arm").agg(
        info_mean=("d_info", "mean"), info_min=("d_info", "min"),
        pos=("d_info", lambda s: int((s > 0).sum())))
    print("\n[Δinfo 요약 — 채택 후보 조건: 3폴드 중 2+ 양수]")
    print(g.to_string(float_format=lambda x: f"{x:.2f}"))
    (OUT_DIR / "gate.json").write_text(t.to_json(orient="records"), encoding="utf-8")


def ens_swap(args, df, X, B, y, season):
    """nn_lin 캐리어: nn_lin만 확장 입력으로 재학습, 다른 4멤버는 캐시 → ENS 증분."""
    arms = [("nn_lin(기준)", []), ("+denom", DENOM_COLS), ("+home", HOME_COLS),
            ("+둘다", DENOM_COLS + HOME_COLS)]
    res = []
    for v in [int(s) for s in args.vals.split(",")]:
        fit, val = (season == v - 1), (season == v)
        yv = y[val]
        z = np.load(CACHE / f"preds_val{v}.npz")
        others = {m: z[m] for m in EP.ENS4_W if m != "nn_lin"}
        ens_ref = sum(z[m] * w for m, w in EP.ENS4_W.items())
        s_ref = best_cal(yv, ens_ref)[0]
        print(f"\n[val {v}]  ENS-4 캐시 기준 {s_ref:.2f}")
        for name, cols in arms:
            XA = X if not cols else pd.concat([X, B[cols]], axis=1)
            st, oh = NM.prep_fit(XA[fit]), NM.onehot_fit(XA[fit])
            Z = np.concatenate([NM.prep_apply(XA[fit], st), NM.onehot_apply(XA[fit], oh)], axis=1)
            model, dev = NM.train_mlp(Z, y[fit], hidden=(), dropout=0.0, epochs=60,
                                      lr=1e-3, wd=1e-4, seed=0)
            Zv = np.concatenate([NM.prep_apply(XA[val], st), NM.onehot_apply(XA[val], oh)], axis=1)
            p_lin = np.clip(NM.predict_mlp(model, dev, Zv), 0, 1)
            ens = sum(others[m] * w for m, w in EP.ENS4_W.items() if m != "nn_lin") \
                + EP.ENS4_W["nn_lin"] * p_lin
            s_ens = best_cal(yv, ens)[0]
            res.append(dict(val=v, arm=name, lin_solo=best_cal(yv, p_lin)[0],
                            ens=s_ens, d_ens=s_ens - s_ref))
            print(f"  {name:14s} nn_lin 단독 {res[-1]['lin_solo']:8.2f}  "
                  f"ENS {s_ens:8.2f}  **ΔENS {s_ens-s_ref:+7.2f}**")
    t = pd.DataFrame(res)
    print("\n" + t.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    (OUT_DIR / "ens_swap.json").write_text(t.to_json(orient="records"), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--ens", action="store_true")
    ap.add_argument("--vals", default="2024,2023,2022")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.audit:
        audit()
        print(f"({time.time()-t0:.0f}s)")
        return

    print(">> train.csv 로드 (K_IS=100)")
    L.K_IS = 100.0
    df = rd.load_train()
    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy())
    X = L.build_features(df, base=(nb, kb)).reset_index(drop=True)
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()
    t = time.time()
    B = build_block(df)
    print(f"   블록 {B.shape[1]}컬럼 생성 [{time.time()-t:.0f}s]")
    for c in DENOM_COLS:
        v_ = B[c].to_numpy()
        print(f"     {c:16s} 결측 {np.isnan(v_).mean():5.1%} · 중앙값 {np.nanmedian(v_):+.3f}")

    if args.gate:
        gate(args, df, X, B, y, season)
    if args.ens:
        ens_swap(args, df, X, B, y, season)
    print(f"\n({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
