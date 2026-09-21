# -*- coding: utf-8 -*-
"""롱샷 일괄 시험 — "예상 밖의 것"을 놓치지 않기 위한 저비용 전수 소진.

    python sweep/longshots.py --vals 2024,2023,2022

사용자 원칙: **어떤 후보도 하드 제외 금지, 롱샷에도 탐색 예산**. 지금까지 안 재본 것을 전부 잰다.

| arm | 내용 | 왜 미시험인가 |
|---|---|---|
| **L1 월 recency 가중** | fit 시즌 안에서 늦은 달에 가중(decay^(10−월)) | "최신 1시즌"이 +530이었는데 **시즌 내부** recency는 아무도 안 봄. 2025는 2024년 9월 다음이다 |
| **L2 R전용 학습** | 퓨처스(F)·포스트(P) 행 제외 | F가 2023에 −0.236 파단(세그먼트 오염원). test가 R 위주면 이득 |
| **L3 교차계열 잔차 부스팅** | june 예측의 **잔차**를 원핫+ridge로 적합해 가산 | 병렬 블렌드만 해봤지 **순차(부스팅)** 결합은 미시험. 선형이 이 데이터에서 강함 |
| **L4 구 82피처 멤버** | `real_data.build_features`(82피처)로 june 파라미터 학습 | **다른 피처 엔지니어링** = 다른 정보 시각. 새 프로토콜에서 잰 적 없음 |
| **L6 로짓 아핀 캘리** | 확률이 아니라 로짓 공간에서 slope/shift | 꼬리 늘림 방식이 다르다. 확률 아핀만 써왔음 |
| **L7 카운트별 아핀 캘리** | count_code 12구간별 slope/shift(교차 폴드 학습) | 블렌드의 카운트 조건부 miscalibration은 안 본 축 |

L1·L2 = 학습 방식(기준: june 2종) · L3·L4 = 멤버 후보(기준: **ENS-4 캐시 블렌드**, 불일치 s + 저가중 이득)
· L6·L7 = 캘리(교차 폴드: 다른 폴드에서 상수를 배워 적용 — 서빙과 동형).
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
from season_centering import best_cal   # noqa: E402

CACHE = HERE.parent / "results" / "ens5"
OUT_DIR = HERE.parent / "results" / "longshots"
MEMBERS = list(EP.ENS4_W)
WS = (0.05, 0.10, 0.15, 0.20)


def june_pair(Xf, yf, Xv, weights=None):
    ps, pf = [], []
    for name, sp in L.ORIGINAL.items():
        import lightgbm as lgb
        params = {**L.BASE_PARAMS, **{k: v for k, v in sp.items() if k != "num_iterations"},
                  "num_threads": 6}
        ds = lgb.Dataset(Xf, label=yf, weight=weights, free_raw_data=False)
        m = lgb.train(params, ds, num_boost_round=sp["num_iterations"])
        ps.append(m.predict(Xv, num_threads=6))
        pf.append(m.predict(Xf, num_threads=6))
    return np.clip(np.mean(ps, axis=0), 0, 1), np.clip(np.mean(pf, axis=0), 0, 1)


def ens4_cached(v):
    z = np.load(CACHE / f"preds_val{v}.npz")
    P = {m: z[m] for m in MEMBERS}
    return sum(P[m] * w for m, w in EP.ENS4_W.items()), z["y"]


def member_report(name, p, ens, yv, rows, v):
    s = float(np.sqrt(((p - ens) ** 2).mean()))
    base = best_cal(yv, ens)[0]
    gains = {w: best_cal(yv, (1 - w) * ens + w * p)[0] - base for w in WS}
    bw = max(gains, key=gains.get)
    rows.append(dict(val=v, arm=name, solo=L.score(yv, L.calibrate(p)), s=s,
                     best_w=bw, gain=gains[bw]))
    print(f"    {name:14s} 단독 {rows[-1]['solo']:8.2f}  s={s:.4f}  "
          f"최적w {bw:.2f} → {gains[bw]:+7.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024,2023,2022")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(">> train.csv 로드 (K_IS=100)")
    L.K_IS = 100.0
    df = rd.load_train()
    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy())
    X = L.build_features(df, base=(nb, kb)).reset_index(drop=True)
    X82 = rd.build_features(df).reset_index(drop=True)          # 구 82피처 (z_asof 제거본)
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()
    month = df["game_month"].to_numpy()
    gtype = df["game_type"].astype(str).to_numpy()
    count_code = (df["balls_before"].to_numpy() * 3 + df["strikes_before"].to_numpy())

    vals = [int(s) for s in args.vals.split(",")]
    rows_t, rows_m, rows_c = [], [], []

    for v in vals:
        fit, val = (season == v - 1), (season == v)
        yv = y[val]
        print(f"\n[val {v}]  fit {fit.sum():,} → val {val.sum():,}")
        p_base, p_fit = june_pair(X[fit], y[fit], X[val])
        base = best_cal(yv, p_base)[0]
        print(f"    기준 june쌍 재적합 {base:8.2f}")

        # --- L1 월 recency 가중 --------------------------------------------
        for decay in (0.85, 0.70):
            w_ = np.power(decay, (10 - month[fit]).clip(0))
            p, _ = june_pair(X[fit], y[fit], X[val], weights=w_)
            d = best_cal(yv, p)[0] - base
            rows_t.append(dict(val=v, arm=f"L1 월가중 d={decay}", d=d))
            print(f"    L1 d={decay}  Δ {d:+7.2f}")

        # --- L2 R전용 학습 --------------------------------------------------
        fitR = fit & (gtype == "R")
        p, _ = june_pair(X[fitR], y[fitR], X[val])
        d = best_cal(yv, p)[0] - base
        rows_t.append(dict(val=v, arm="L2 R전용", d=d))
        mR = val & (gtype == "R")
        dR = best_cal(y[mR], p[(gtype == "R")[val]])[0] - best_cal(y[mR], p_base[(gtype == "R")[val]])[0]
        print(f"    L2 R전용({fitR.sum():,}행)  Δ전체 {d:+7.2f} · ΔR층 {dR:+7.2f}")

        # --- 멤버 후보는 ENS-4 캐시 대비 -------------------------------------
        ens, y_chk = ens4_cached(v)
        assert np.allclose(y_chk, yv)
        # L3 교차계열 잔차 부스팅: june 잔차를 원핫+ridge로
        st, oh = NM.prep_fit(X[fit]), NM.onehot_fit(X[fit])
        Zf = np.concatenate([NM.prep_apply(X[fit], st), NM.onehot_apply(X[fit], oh)], axis=1)
        Zv = np.concatenate([NM.prep_apply(X[val], st), NM.onehot_apply(X[val], oh)], axis=1)
        resid = y[fit] - p_fit
        for lam in (100.0, 10000.0):
            G = Zf.T @ Zf + lam * np.eye(Zf.shape[1])
            c = np.linalg.solve(G, Zf.T @ resid)
            p3 = np.clip(p_base + Zv @ c, 0, 1)
            member_report(f"L3 잔차부스팅 λ={lam:g}", p3, ens, yv, rows_m, v)
        # L4 구 82피처 멤버
        p4, _ = june_pair(X82[fit], y[fit], X82[val])
        member_report("L4 feat82", p4, ens, yv, rows_m, v)

        # --- L6/L7 캘리 (교차 폴드: 다른 폴드에서 상수 학습) ------------------
        others = [u for u in vals if u != v]
        if others:
            u = others[0]
            ens_u, y_u = ens4_cached(u)
            base_v = best_cal(yv, ens)[0]
            # L6 로짓 아핀: u에서 (a,b) 격자 최적 → v에 적용
            lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
            best_u, arg = -1e18, None
            zu, zv_ = lg(ens_u), lg(ens)
            for a in np.arange(0.7, 1.61, 0.05):
                for b in np.arange(-0.15, 0.151, 0.025):
                    s_ = L.score(y_u, 1 / (1 + np.exp(-(a * zu + b))))
                    if s_ > best_u:
                        best_u, arg = s_, (a, b)
            a, b = arg
            d6 = L.score(yv, 1 / (1 + np.exp(-(a * zv_ + b)))) - base_v
            rows_c.append(dict(val=v, arm=f"L6 로짓아핀(←{u})", d=d6))
            # L7 카운트별 아핀: u에서 count별 (slope, shift) → v에 적용
            cc_u = count_code[season == u]
            cc_v = count_code[val]
            pv7 = ens.copy()
            for c_ in range(12):
                mu_, mv_ = cc_u == c_, cc_v == c_
                if mu_.sum() < 2000 or mv_.sum() == 0:
                    continue
                bu, au = -1e18, (1.0, 0.0)
                for sl in np.arange(0.7, 1.41, 0.05):
                    q = 0.5 + sl * (ens_u[mu_] - 0.5)
                    for sh in np.arange(-0.02, 0.021, 0.005):
                        s_ = L.score(y_u[mu_], np.clip(q + sh, 1e-6, 1 - 1e-6))
                        if s_ > bu:
                            bu, au = s_, (sl, sh)
                pv7[mv_] = np.clip(0.5 + au[0] * (ens[mv_] - 0.5) + au[1], 1e-6, 1 - 1e-6)
            d7 = L.score(yv, pv7) - base_v
            rows_c.append(dict(val=v, arm=f"L7 카운트캘리(←{u})", d=d7))
            print(f"    L6 로짓아핀 Δ {d6:+7.2f} · L7 카운트캘리 Δ {d7:+7.2f}")

    print("\n" + "=" * 90)
    for tag, rr, key in (("학습 방식 (기준 june쌍)", rows_t, "d"),
                         ("캘리 (기준 ENS-4, 교차폴드)", rows_c, "d")):
        if not rr:
            continue
        t = pd.DataFrame(rr)
        g = t.groupby("arm").agg(d_mean=(key, "mean"), d_sd=(key, "std"), d_min=(key, "min"),
                                 pos=(key, lambda s: int((s > 0).sum())), n=(key, "size"))
        g["통과"] = (g.d_mean > 0) & (g.d_mean > g.d_sd.fillna(0))
        print(f"\n[{tag}]")
        print(g.sort_values("d_mean", ascending=False).to_string(float_format=lambda v: f"{v:.2f}"))
    if rows_m:
        t = pd.DataFrame(rows_m)
        g = t.groupby("arm").agg(solo=("solo", "mean"), s=("s", "mean"),
                                 gain_mean=("gain", "mean"), gain_min=("gain", "min"),
                                 pos=("gain", lambda s: int((s > 0).sum())), n=("gain", "size"))
        print("\n[멤버 후보 (기준 ENS-4 캐시 블렌드, 저가중 이득)]")
        print(g.sort_values("gain_mean", ascending=False).to_string(float_format=lambda v: f"{v:.3f}"))

    (OUT_DIR / "gate.json").write_text(
        json.dumps({"train": rows_t, "member": rows_m, "calib": rows_c},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR/'gate.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
