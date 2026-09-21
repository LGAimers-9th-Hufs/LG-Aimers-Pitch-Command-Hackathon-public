# -*- coding: utf-8 -*-
"""Step 2A-1 — 학습창 레그(wALL·wREC) 5시드 배깅 재확인 + 정확 채택 산수.

    python sweep/era_bag.py

era_window_d.py의 단일시드 판정(wALL d=0.0457, 분기 A)을 배깅으로 재확인하고,
근사식 대신 **캐시 예측으로 정확 gain**을 계산한다:
  gain_M = 4e5 · mean_i mean_rows[(p_i − p̄_M)²]   (Brier 항등식, 프록시 30k행)
  E_M    = mean(S_legs)/M + gain_M/ρ,  ρ = 1.4295
S_est(신규 레그) = 0.7541·로컬 + 320.9 (전이선, RMS 8.0 — wALL은 지지범위 밖 외삽 주의).
민감도(ρ↑·S↓)까지 표로 출력. 산출: results/tm_repr/era_bag.json + 배깅 예측 npy.
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import real_data as rd                     # noqa: E402
from tm_repr_leg import build_xa, leg_score   # noqa: E402

OUT = HERE.parent / "results" / "tm_repr"
PARAMS = dict(objective="binary", metric="binary_logloss", learning_rate=0.05,
              num_leaves=15, min_data_in_leaf=1000, feature_fraction=0.85,
              bagging_fraction=0.8, bagging_freq=1, verbosity=-1)
SEEDS = (0, 1, 2, 3, 4)
RHO = 1.4295
D4_SUM = 4026.56          # S(clookup)+S(cmoe)+S(physmix)+S(tm3L), 역산 포함
BEST = 1058.6047851923    # blendD4 실측


def transfer(local):
    return 0.7541 * local + 320.9


def main():
    t0 = time.time()
    df = rd.load_train()
    y = df[rd.TARGET].to_numpy(dtype=np.float32)
    season = df[rd.SEASON].to_numpy()
    XA = build_xa(df, season)
    val = season == 2024
    yv = y[val]
    print(f">> XA {XA.shape}  [{time.time()-t0:.0f}s]")

    legs = {}
    for nm in ("clookup", "cmoe", "physmix", "tm3L"):
        fs = sorted(glob.glob(str(HERE.parent / "results/leg_matrix" / f"{nm}_30000_*.npy")))
        legs[nm] = np.load(fs[-1])
    L4 = [legs[k] for k in ("clookup", "cmoe", "physmix", "tm3L")]
    D4 = np.mean(L4, axis=0)
    gain4 = 4e5 * float(np.mean([np.mean((p - D4) ** 2) for p in L4]))
    print(f">> 정확 gain₄(캐시) = {gain4:.2f}  (참조 74.28)")

    fitmask = season <= 2023
    variants = {"wALL": None,
                "wREC": (0.5 ** (2023 - season[fitmask])).astype(np.float64)}
    rep = {"gain4_exact": round(gain4, 2)}
    bag30 = {}
    for name, sw in variants.items():
        pv_seeds = []
        t1 = time.time()
        for s in SEEDS:
            m = lgb.train(dict(PARAMS, seed=s),
                          lgb.Dataset(XA[fitmask], label=y[fitmask], weight=sw),
                          num_boost_round=600)
            pv_seeds.append(m.predict(XA[val]))
        pv_bag = np.mean(pv_seeds, axis=0)
        solos = [leg_score(yv, p) for p in pv_seeds]
        solo_bag = leg_score(yv, pv_bag)
        p30 = pv_bag[:30000]
        bag30[name] = p30
        d4 = float(np.sqrt(np.mean((p30 - D4) ** 2)))
        np.save(OUT / f"era_bag_{name}_val24.npy", pv_bag)
        rep[name] = {"solo_seeds_mean": round(float(np.mean(solos)), 2),
                     "solo_seeds_sd": round(float(np.std(solos)), 2),
                     "solo_bag": round(solo_bag, 2), "d_D4_bag": round(d4, 5),
                     "S_est": round(transfer(solo_bag), 1),
                     "train_s": round(time.time() - t1, 1)}
        print(f"[{name}] 시드 {np.mean(solos):7.2f}±{np.std(solos):.2f}  "
              f"배깅 {solo_bag:7.2f}  d(D4) {d4:.5f}  [{time.time()-t1:.0f}s]")

    def exact_gain(ps):
        pbar = np.mean(ps, axis=0)
        return 4e5 * float(np.mean([np.mean((p - pbar) ** 2) for p in ps]))

    def table(name, ps, s_new):
        g = exact_gain(ps)
        M = len(ps)
        rows = {}
        for rho in (RHO, 1.6, 1.7):
            for ds in (0, -50, -100):
                e = (D4_SUM + sum(x + ds for x in s_new)) / M + g / rho
                rows[f"rho{rho}_dS{ds}"] = round(e, 1)
        base = rows[f"rho{RHO}_dS0"]
        print(f"[{name}] M={M}  정확 gain = {g:.2f}  E = {base:.1f} ({base-BEST:+.1f})  "
              f"| 민감도 ρ1.6:{rows['rho1.6_dS0']:.1f} ρ1.7:{rows['rho1.7_dS0']:.1f} "
              f"S−50:{rows[f'rho{RHO}_dS-50']:.1f} S−100:{rows[f'rho{RHO}_dS-100']:.1f}")
        return {"gain_exact": round(g, 2), "E": rows}

    sA = rep["wALL"]["S_est"]
    sR = rep["wREC"]["S_est"]
    rep["blendF5_wALL"] = table("D4+wALL (5)", L4 + [bag30["wALL"]], [sA])
    rep["blendF5_wREC"] = table("D4+wREC (5)", L4 + [bag30["wREC"]], [sR])
    rep["blendF6_both"] = table("D4+wALL+wREC (6)", L4 + [bag30["wALL"], bag30["wREC"]],
                                [sA, sR])

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "era_bag.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f">> saved results/tm_repr/era_bag.json  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
