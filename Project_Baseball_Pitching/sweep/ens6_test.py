# -*- coding: utf-8 -*-
"""ENS-6 실증 — HPO 최적 파라미터를 멤버에 넣고 **앙상블 수준에서** 이득을 잰다.

    python sweep/ens6_test.py --vals 2024,2023,2022

## 왜 별도 실증인가

HPO의 +37은 **단일 모델 june 기준**이다. 정제형 개선은 앙상블이 흡수한 전례가 있다
(K=100: june 기준 +13.0 → ENS 기준 +3.2). 채택 판단은 반드시 ENS 수준에서 한다.

구성: ENS-4의 5멤버 중 **LGBM 2종의 파라미터만** HPO 최적(t54/t57)으로 교체.
- june_l15 역할 → t54: lr 0.0175 · it 726 · leaves 7 · min 2500 (seed 57)
- june_l7 역할 → t57: lr 0.0211 · it 514 · leaves 11 · min 1800 (seed 49)
- allraw_l15 역할 → allraw HPO 결과가 나오면 교체(그 전엔 기존 스펙)
- nn_lin · et_l100_d28 불변, K_IS = 100 (두 최적 구성 모두 K=100 — 일관)

판정: ①고정 ENS-4 가중 + 아핀 재적합으로 Δ ②그리디(maximin) 재최적화 ③val2022 확인.
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
from ens4_weights import affine_score, grid_calib, greedy as greedy_single   # noqa: E402
from ens5_select import greedy_multi   # noqa: E402
from season_centering import best_cal   # noqa: E402

OUT_DIR = HERE.parent / "results" / "ens6"

# HPO trials_june.jsonl 상위 (val2022 검증 통과분만 쓸 것 — --verify 결과 확인 후)
HPO_L15 = dict(num_leaves=7, min_data_in_leaf=2500, num_iterations=726,
               learning_rate=0.0175, lambda_l1=None, lambda_l2=None,
               feature_fraction=None, bagging_fraction=None, bagging_freq=1, seed=57)
HPO_L7 = dict(num_leaves=11, min_data_in_leaf=1800, num_iterations=514,
              learning_rate=0.0211, lambda_l1=None, lambda_l2=None,
              feature_fraction=None, bagging_fraction=None, bagging_freq=1, seed=49)


def load_trial_params(target, trial_no):
    log = HERE.parent / "results" / "hpo" / f"trials_{target}.jsonl"
    rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
    r = [x for x in rows if x["trial"] == trial_no][0]
    return r["params"], r["k_is"]


def train_lgbm_full(X, y, fit, val, params):
    import lightgbm as lgb
    pp = {**L.BASE_PARAMS, **{k: v for k, v in params.items()
                              if v is not None and k != "num_iterations"}, "num_threads": 6}
    ds = lgb.Dataset(X[fit], label=y[fit], free_raw_data=False)
    m = lgb.train(pp, ds, num_boost_round=params["num_iterations"])
    return np.clip(m.predict(X[val], num_threads=6), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024,2023,2022")
    ap.add_argument("--t_l15", type=int, default=54, help="june_l15 역할의 HPO 트라이얼 번호")
    ap.add_argument("--t_l7", type=int, default=57)
    ap.add_argument("--t_allraw", type=int, default=-1, help="-1 = 기존 스펙 유지")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    p15, k15 = load_trial_params("june", args.t_l15)
    p7, k7 = load_trial_params("june", args.t_l7)
    p7 = {**p7, "seed": 49}                       # 다양성용 시드 분리
    assert k15 == k7 == 100, f"K 불일치: {k15}/{k7} — K_IS 통일 필요"
    if args.t_allraw >= 0:
        pall, _ = load_trial_params("allraw", args.t_allraw)
    else:
        pall = dict(L.ORIGINAL["l15"])
    print(f"t{args.t_l15}(l15역): {p15}")
    print(f"t{args.t_l7}(l7역):  {p7}")
    print(f"allraw:      {pall}")

    print(">> train.csv 로드 (K_IS=100)")
    L.K_IS = 100.0
    df = rd.load_train()
    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy())
    X = L.build_features(df, base=(nb, kb))
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()

    out = []
    P_all, Y_all = {}, {}
    for v in [int(s) for s in args.vals.split(",")]:
        fit, val = (season == v - 1), (season == v)
        fit_all = season < v
        yv = y[val]
        print(f"\n[val {v}]  fit {fit.sum():,} → val {val.sum():,}")

        P = {}
        t = time.time()
        P["june_l15"] = train_lgbm_full(X, y, fit, val, p15)
        P["june_l7"] = train_lgbm_full(X, y, fit, val, p7)
        P["allraw_l15"] = train_lgbm_full(X, y, fit_all, val, pall)
        # nn_lin / ET — 기존 그대로
        from sklearn.ensemble import ExtraTreesRegressor
        Xn = np.nan_to_num(X[fit].to_numpy(dtype=np.float32), nan=-999.0)
        Xvn = np.nan_to_num(X[val].to_numpy(dtype=np.float32), nan=-999.0)
        P["et_l100_d28"] = np.clip(ExtraTreesRegressor(
            n_estimators=200, min_samples_leaf=100, max_depth=28, max_features=0.7,
            n_jobs=6, random_state=9).fit(Xn, y[fit]).predict(Xvn), 0, 1)
        st, oh = NM.prep_fit(X[fit]), NM.onehot_fit(X[fit])
        Z = np.concatenate([NM.prep_apply(X[fit], st), NM.onehot_apply(X[fit], oh)], axis=1)
        model, dev = NM.train_mlp(Z, y[fit], hidden=(), dropout=0.0, epochs=60,
                                  lr=1e-3, wd=1e-4, seed=0)
        Zv = np.concatenate([NM.prep_apply(X[val], st), NM.onehot_apply(X[val], oh)], axis=1)
        P["nn_lin"] = np.clip(NM.predict_mlp(model, dev, Zv), 0, 1)
        print(f"    멤버 5종 학습 [{time.time()-t:.0f}s]  단독: " + " ".join(
            f"{m}={best_cal(yv, P[m])[0]:.1f}" for m in EP.ENS4_W))
        P_all[v], Y_all[v] = P, yv

        ens_new = sum(P[m] * w for m, w in EP.ENS4_W.items())
        (sl, sh), s_new = grid_calib(yv, ens_new)
        # 기준: ENS-4 캐시 (구 파라미터 멤버들)
        z = np.load(HERE.parent / "results" / "ens5" / f"preds_val{v}.npz")
        ens_old = sum(z[m] * w for m, w in EP.ENS4_W.items())
        s_old = best_cal(yv, ens_old)[0]
        out.append(dict(val=v, old=s_old, new=s_new, d=s_new - s_old, cal=[sl, sh]))
        print(f"    ENS(고정가중)  구 {s_old:8.2f} → 신 {s_new:8.2f}   Δ {s_new-s_old:+7.2f}  ({sl},{sh})")

    t_ = pd.DataFrame(out)
    print("\n" + "=" * 80)
    print(t_.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    # 그리디 재최적화 (선택 = 2024+2023 maximin, 2022는 확인 전용)
    sel = [v for v in P_all if v in (2024, 2023)]
    if len(sel) == 2:
        ref = {v: affine_score(Y_all[v], sum(P_all[v][m] * w for m, w in EP.ENS4_W.items()))
               for v in sel}
        Wn = greedy_multi(P_all, Y_all, sel, mode="maximin", ref=ref,
                          init=["nn_lin", "june_l15"])
        print("\n[그리디(maximin) 재최적화]")
        for k_, q in sorted(Wn.items(), key=lambda kv: -kv[1]):
            print(f"    {k_:14s} {EP.ENS4_W.get(k_, 0):.4f} → {q:.4f}")
        for v in P_all:
            b = sum(P_all[v][m] * w for m, w in EP.ENS4_W.items())
            g = sum(P_all[v][m] * w for m, w in Wn.items() if m in P_all[v])
            (sl, sh), sg = grid_calib(Y_all[v], g)
            sb = best_cal(Y_all[v], b)[0]
            tag = " (선택 미사용 폴드)" if v == 2022 else ""
            print(f"    val {v}: 고정 {sb:8.2f} → 그리디 {sg:8.2f}  Δ {sg-sb:+6.2f}  ({sl},{sh}){tag}")
        (OUT_DIR / "greedy.json").write_text(json.dumps(Wn, ensure_ascii=False, indent=1),
                                             encoding="utf-8")

    (OUT_DIR / "gate.json").write_text(t_.to_json(orient="records"), encoding="utf-8")
    for v, P in P_all.items():
        np.savez_compressed(OUT_DIR / f"preds_val{v}.npz", y=Y_all[v], **P)
    print(f"\n>> 저장 {OUT_DIR}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
