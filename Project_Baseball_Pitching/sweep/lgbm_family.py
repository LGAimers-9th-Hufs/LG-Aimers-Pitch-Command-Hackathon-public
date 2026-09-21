# -*- coding: utf-8 -*-
"""june853(LB 853.57) 레시피 재현 + 다양성 변형 — 1000점 경로의 기반 모듈.

    python sweep/lgbm_family.py --validate           # 원본 재현 확인 (목표: raw 683.1 / cal 712.79)
    python sweep/lgbm_family.py --variants           # 다양성 변형 학습 → (점수, june853과의 불일치)

## 왜 이 모듈인가

팀 최고 제출 `aimers_sub_june_verzip.zip`(853.57)을 해체해 얻은 사실:

- **LightGBM 2종(num_leaves 7·15) 확률 평균**, `objective=regression`(=Brier 직접 최소화)
- 피처 53개 = 공식 44 + 범주형 3(정수 매핑) + 파생 6
  (`count_code`, `p_logn`, `b_logn`, **`p_sm500`/`b_sm500`**, `prev{1,3,5}_fill`, `hand_match`)
- **`season`·`pitcher_id`·`batter_id`를 버린다** ← 우리가 −192로 데인 out-of-support 문제를 구조적으로 회피
- **`p_sm500 = (rate·n + 250)/(n + 500)`** = 신뢰도 수축 K=500을 **행 단위로** 계산.
  우리 `z_asof`는 같은 수축(K=500이 최적이라는 스캔 결과까지 동일)을 **train 유래 룩업**으로 만들어
  시즌 누적이 커지는 분포 이동을 그대로 맞았다. **같은 통계를 행 단위로 만들면 드리프트가 없다.** 이 대비가 이 대회의 핵심 교훈이다.
- 강정칙: `lambda_l2=8`, `min_data_in_leaf` 1000~1500, `num_leaves` 7~15
  (우리 HGB는 leaf 31·min 200·l2 1.0으로 **훨씬 약했다**)
- 후처리 = **분산 수축** `p* = 0.5 + 0.91·(p − 0.5) − 0.01`
- 자체 검증(fit 2023 → val 2024): raw 683.1 → 캘리 712.79. **2025 실측 853.57(+141 전이 이득)**

## 1000점 산수 (memory/lb-verified-scores.md §블렌드 경제학)

853급 두 모델을 0.5:0.5로 섞을 때: RMS 0.030 → 943.6(1위 931 돌파) · 0.040 → 1013.6.
**정확도를 더 짜내는 문제가 아니라 "같은 급인데 다르게 틀리는 모델"을 만드는 문제다.**
현재 우리 모델들의 june853 대비 불일치는 0.019~0.027로 전부 손익분기 미달이다.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import real_data as rd  # noqa: E402

CAT_MAPS = {
    "top_bottom": {"B": 0, "T": 1},
    "game_type": {"P": 0, "R": 1},      # 원본 그대로 — 실제 값 'F'는 -1로 떨어진다(일관되므로 무해)
    "base_state": {"123": 0, "12_": 1, "1_3": 2, "1__": 3,
                   "_23": 4, "_2_": 5, "__3": 6, "___": 7},
}

DROP = ["row_id", "control_success", "season", "pitcher_id", "batter_id"]

FEATURES = [
    "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
    "balls_before", "strikes_before", "outs_before",
    "run_top_before", "run_bot_before", "run_total_before",
    "score_diff_home", "score_diff_pitcher_team",
    "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on", "base_state",
    "home_win_expectancy", "away_win_expectancy", "li",
    "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id",
    "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
    "count_code", "p_logn", "b_logn", "p_sm500", "b_sm500",
    "prev1_fill", "prev3_fill", "prev5_fill", "hand_match",
]

# 원본 두 모델의 파라미터 (model/*.txt의 parameters 블록에서 추출)
BASE_PARAMS = dict(objective="regression", metric="l2", boosting="gbdt",
                   learning_rate=0.03, feature_fraction=0.85,
                   lambda_l1=0.5, lambda_l2=8.0, num_threads=4,
                   force_col_wise=True, verbosity=-1)
ORIGINAL = {
    "l7":  dict(num_leaves=7,  min_data_in_leaf=1500, num_iterations=500, seed=49),
    "l15": dict(num_leaves=15, min_data_in_leaf=1000, num_iterations=400, seed=57),
}
CAL = dict(center=0.5, slope=0.91, shift=-0.01)


# ---------------------------------------------------------------- 당해 시즌 분해 (D-28)
# `asof_*`는 커리어 누적이고 2025 서버 안에서도 갱신된다(실측: pid 21813이 train 2024말 3085 →
# test 3465). train으로 만든 "직전 시즌 말 누적"을 빼면 그 투수의 **당해 시즌 성적**만 남는다.
# `pitcher_id`가 피처에서 빠져 있어 트리는 이 값을 원리적으로 만들 수 없다.
# 3폴드 +123(위약 −4.9) · ENS-2 위에서 +85.6. 상세 `docs/log/21_inseason.md`.
IS_COLS = ["is_logn", "is_share", "is_sm", "is_delta"]     # ← 순서가 서빙 계약이다. 바꾸지 말 것
K_IS = 200.0


def add_inseason(x: pd.DataFrame, n_base, k_base) -> pd.DataFrame:
    """행 단위 산술만 쓴다(§5). `n_base/k_base`는 동봉 룩업에서 온 그 행의 상수다."""
    n = x["asof_pitcher_n"].astype("float64").to_numpy()
    rate = x["asof_pitcher_success_rate"].fillna(0.0).astype("float64").to_numpy()
    k = np.rint(rate * n)
    is_n = np.maximum(n - np.asarray(n_base, dtype=float), 0.0)
    is_k = np.clip(k - np.asarray(k_base, dtype=float), 0.0, is_n)
    is_sm = (is_k + K_IS * 0.5) / (is_n + K_IS)
    car_sm = (rate * n + 250.0) / (n + 500.0)          # = p_sm500 (동일 산식)
    x["is_logn"] = np.log1p(is_n).astype("float32")
    x["is_share"] = (is_n / np.maximum(n, 1.0)).astype("float32")
    x["is_sm"] = is_sm.astype("float32")
    x["is_delta"] = (is_sm - car_sm).astype("float32")
    return x


def build_features(df: pd.DataFrame, base=None) -> pd.DataFrame:
    """june853 script.py의 build_features를 그대로 옮긴 것 — 전부 행 단위 변환(§5 준수).

    `base=(n_base, k_base)`를 주면 당해 시즌 분해 4컬럼(IS_COLS)을 덧붙인다.
    """
    x = df.copy()
    for col, mapping in CAT_MAPS.items():
        x[col] = x[col].astype("string").map(mapping).fillna(-1).astype("int8")

    x["count_code"] = (x["balls_before"] * 3 + x["strikes_before"]).astype("int8")
    x["p_logn"] = np.log1p(x["asof_pitcher_n"].astype("float32")).astype("float32")
    x["b_logn"] = np.log1p(x["asof_batter_n"].astype("float32")).astype("float32")

    p_n = x["asof_pitcher_n"].astype("float32")
    b_n = x["asof_batter_n"].astype("float32")
    p_rate = x["asof_pitcher_success_rate"].fillna(0.5).astype("float32")
    b_rate = x["asof_batter_success_rate"].fillna(0.5).astype("float32")
    # 신뢰도 수축 K=500 — 룩업이 아니라 **그 행의 값만으로** 계산되므로 시즌 드리프트가 없다.
    x["p_sm500"] = ((p_rate * p_n + 250.0) / (p_n + 500.0)).astype("float32")
    x["b_sm500"] = ((b_rate * b_n + 250.0) / (b_n + 500.0)).astype("float32")

    for w in (1, 3, 5):
        x[f"prev{w}_fill"] = (x[f"asof_pitcher_prev{w}_game_success_rate"]
                              .fillna(x["asof_pitcher_success_rate"])
                              .fillna(0.5).astype("float32"))

    x["hand_match"] = (x["pitcher_hand"] == x["batter_hand"]).astype("int8")
    cols = FEATURES
    if base is not None:
        x = add_inseason(x, base[0], base[1])
        cols = FEATURES + IS_COLS
    x = x.drop(columns=[c for c in DROP if c in x.columns])
    return x.loc[:, cols]


def calibrate(p, center=CAL["center"], slope=CAL["slope"], shift=CAL["shift"]):
    return np.clip(center + slope * (np.asarray(p) - center) + shift, 0.0, 1.0)


def score(y, p):
    y = np.asarray(y, dtype=float)
    r = y.mean()
    return 1e5 * (1.0 - float(np.mean((np.asarray(p) - y) ** 2)) / (r * (1 - r)))


def train_lgbm(X, y, spec: dict):
    import lightgbm as lgb
    params = {**BASE_PARAMS, **{k: v for k, v in spec.items() if k != "num_iterations"}}
    ds = lgb.Dataset(X, label=np.asarray(y, dtype=float), free_raw_data=False)
    return lgb.train(params, ds, num_boost_round=spec.get("num_iterations", 400))


def _folds(df, fit_seasons, val_season):
    fit = df[rd.SEASON].isin(fit_seasons).to_numpy()
    val = (df[rd.SEASON] == val_season).to_numpy()
    return fit, val


def validate(df, fit_seasons, val_season=2024, specs=None, verbose=True):
    """원본 2종 앙상블을 fit_seasons로 학습해 val_season에서 채점 — 재현 확인용."""
    specs = specs or ORIGINAL
    X = build_features(df)
    y = df[rd.TARGET].to_numpy()
    fit, val = _folds(df, fit_seasons, val_season)
    ps = []
    for name, sp in specs.items():
        m = train_lgbm(X[fit], y[fit], sp)
        p = m.predict(X[val], num_threads=4)
        ps.append(p)
        if verbose:
            print(f"    {name:>6s} 단독 score={score(y[val], np.clip(p, 0, 1)):8.2f}")
    raw = np.mean(ps, axis=0)
    return raw, calibrate(raw), y[val], val


def variant_models(X, y, fit):
    """다양성 후보. **점수는 비슷하되 다르게 틀리는** 모델을 찾는 것이 목적이다.
    같은 계열의 seed 변형은 불일치가 0.01 수준이라 쓸모없다 — 계열 자체를 바꿔야 0.03이 나온다."""
    from sklearn.ensemble import (HistGradientBoostingRegressor, RandomForestRegressor,
                                  ExtraTreesRegressor)
    Xf, yf = X[fit], y[fit]
    out = {}

    def lgbm(name, **sp):
        out[name] = train_lgbm(Xf, yf, {**dict(num_leaves=15, min_data_in_leaf=1000,
                                               num_iterations=400, seed=7), **sp})

    lgbm("lg_dart", boosting="dart", num_iterations=500, drop_rate=0.1, seed=11)
    lgbm("lg_extra", extra_trees=True, num_leaves=63, min_data_in_leaf=500,
         num_iterations=600, seed=13)
    lgbm("lg_deep", num_leaves=127, min_data_in_leaf=200, lambda_l2=1.0,
         num_iterations=300, seed=17)
    lgbm("lg_binary", objective="binary", metric="binary_logloss", num_iterations=500, seed=19)
    lgbm("lg_ff30", feature_fraction=0.3, num_leaves=31, num_iterations=600, seed=23)

    Xn = np.nan_to_num(Xf.to_numpy(dtype=np.float32), nan=-999.0)
    out["sk_hgb"] = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=1000,
        l2_regularization=8.0, early_stopping=False, random_state=3).fit(Xf, yf)
    out["sk_rf"] = RandomForestRegressor(
        n_estimators=120, max_depth=12, min_samples_leaf=500, n_jobs=-1,
        random_state=5).fit(Xn, yf)
    out["sk_et"] = ExtraTreesRegressor(
        n_estimators=200, max_depth=16, min_samples_leaf=400, n_jobs=-1,
        random_state=9).fit(Xn, yf)
    return out


def predict_variant(name, model, X):
    if name.startswith("sk_rf") or name.startswith("sk_et"):
        return model.predict(np.nan_to_num(X.to_numpy(dtype=np.float32), nan=-999.0))
    if name.startswith("sk_"):
        return model.predict(X)
    return model.predict(X, num_threads=4)


def run_variants(df, fit_seasons=(2023,), val_season=2024):
    X = build_features(df)
    y = df[rd.TARGET].to_numpy()
    fit, val = _folds(df, list(fit_seasons), val_season)
    yv = y[val]

    print(f"\n[기준] june 레시피(l7+l15) fit={list(fit_seasons)} → val={val_season}")
    ref_ps = []
    for name, sp in ORIGINAL.items():
        m = train_lgbm(X[fit], y[fit], sp)
        ref_ps.append(m.predict(X[val], num_threads=4))
    ref = np.mean(ref_ps, axis=0)
    print(f"    raw {score(yv, np.clip(ref,0,1)):.2f} · cal {score(yv, calibrate(ref)):.2f}")

    print("\n[변형] 학습 중...")
    models = variant_models(X, y, fit)
    rows = []
    for name, m in models.items():
        p = np.clip(predict_variant(name, m, X[val]), 0, 1)
        rms = float(np.sqrt(((p - ref) ** 2).mean()))
        rows.append({"variant": name, "raw": score(yv, p), "cal": score(yv, calibrate(p)),
                     "rms_vs_ref": rms, "mean": p.mean()})
    t = pd.DataFrame(rows).sort_values("cal", ascending=False)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # 기준과 2원 블렌드했을 때 val에서의 실현 점수 (항등식이 아니라 직접 계산)
    print("\n[기준 + 변형 블렌드] val 실현 점수 (w = 변형 가중)")
    ref_cal_score = score(yv, calibrate(ref))
    best = []
    for name, m in models.items():
        p = np.clip(predict_variant(name, m, X[val]), 0, 1)
        row = []
        for w in (0.2, 0.3, 0.4, 0.5):
            s = score(yv, calibrate((1 - w) * ref + w * p))
            row.append(f"w{w}:{s:7.1f}")
        gain = max(score(yv, calibrate((1 - w) * ref + w * p)) for w in (0.2, 0.3, 0.4, 0.5)) \
            - ref_cal_score
        best.append((gain, name))
        print(f"  {name:10s} " + "  ".join(row) + f"   최대이득 {gain:+7.1f}")
    print(f"\n  기준 단독 = {ref_cal_score:.2f}")
    for g, n in sorted(best, reverse=True)[:3]:
        print(f"  ▶ {n}: +{g:.1f}")
    return t


def greedy_ensemble(preds: dict, yv, cal=True, n_iter=40, init=None):
    """Caruana 그리디 앙상블 선택(복원 추출) — 가중을 격자로 훑는 대신 멤버를 반복 추가한다.
    각 단계에서 val 점수를 최대화하는 멤버 1개를 뽑아 누적 평균에 더한다."""
    names = list(preds)
    cur = np.zeros(len(yv))
    picks = list(init or [])
    for nm in picks:
        cur += preds[nm]
    best_hist = []
    for _ in range(n_iter - len(picks)):
        best, bn = -1e18, None
        for nm in names:
            cand = (cur + preds[nm]) / (len(picks) + 1)
            s = score(yv, calibrate(cand) if cal else np.clip(cand, 0, 1))
            if s > best:
                best, bn = s, nm
        picks.append(bn)
        cur += preds[bn]
        best_hist.append((len(picks), bn, best))
    from collections import Counter
    w = {k: v / len(picks) for k, v in Counter(picks).items()}
    final = sum(preds[k] * v for k, v in w.items())
    return w, score(yv, calibrate(final) if cal else np.clip(final, 0, 1)), best_hist, final


def push(df, fit_seasons=(2023,), val_season=2024):
    """ET 계열을 축으로 다양성 멤버를 넓히고 그리디 앙상블로 결합."""
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    X = build_features(df)
    y = df[rd.TARGET].to_numpy()
    fit, val = _folds(df, list(fit_seasons), val_season)
    yv = y[val]
    Xf, yf = X[fit], y[fit]
    Xn, Xvn = (np.nan_to_num(a.to_numpy(dtype=np.float32), nan=-999.0) for a in (Xf, X[val]))

    preds = {}
    print(f"[기준] june l7+l15  fit={list(fit_seasons)} → val={val_season}")
    for name, sp in ORIGINAL.items():
        m = train_lgbm(Xf, yf, sp)
        preds[f"june_{name}"] = np.clip(m.predict(X[val], num_threads=4), 0, 1)
    ref = 0.5 * (preds["june_l7"] + preds["june_l15"])
    ref_s = score(yv, calibrate(ref))
    print(f"    기준 앙상블 cal = {ref_s:.2f}")

    print("[탐색] ExtraTrees / RF 격자")
    grid = [("et_l200_d20", dict(min_samples_leaf=200, max_depth=20, max_features=1.0)),
            ("et_l400_d16", dict(min_samples_leaf=400, max_depth=16, max_features=1.0)),
            ("et_l800_d24", dict(min_samples_leaf=800, max_depth=24, max_features=1.0)),
            ("et_l400_f05", dict(min_samples_leaf=400, max_depth=20, max_features=0.5)),
            ("et_l100_d28", dict(min_samples_leaf=100, max_depth=28, max_features=0.7)),
            ("et_l1500_d12", dict(min_samples_leaf=1500, max_depth=12, max_features=1.0))]
    for nm, kw in grid:
        m = ExtraTreesRegressor(n_estimators=200, n_jobs=-1, random_state=9, **kw).fit(Xn, yf)
        preds[nm] = np.clip(m.predict(Xvn), 0, 1)
        print(f"    {nm:14s} cal={score(yv, calibrate(preds[nm])):8.2f}  "
              f"rms_vs_ref={np.sqrt(((preds[nm]-ref)**2).mean()):.4f}")
    for nm, kw in [("rf_l500_d12", dict(min_samples_leaf=500, max_depth=12)),
                   ("rf_l200_d18", dict(min_samples_leaf=200, max_depth=18))]:
        m = RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=5, **kw).fit(Xn, yf)
        preds[nm] = np.clip(m.predict(Xvn), 0, 1)
        print(f"    {nm:14s} cal={score(yv, calibrate(preds[nm])):8.2f}  "
              f"rms_vs_ref={np.sqrt(((preds[nm]-ref)**2).mean()):.4f}")

    print("\n[결합] Caruana 그리디 앙상블 (val 2024 기준)")
    w, s, hist, final = greedy_ensemble(preds, yv, init=["june_l7", "june_l15"])
    for k, v in sorted(w.items(), key=lambda kv: -kv[1]):
        print(f"    {k:14s} w={v:.3f}")
    print(f"    → cal score {s:.2f}   (기준 {ref_s:.2f}, 이득 {s-ref_s:+.2f})")

    print("\n[캘리 재적합] 블렌드에 맞는 slope/shift 재탐색")
    best = (None, -1e18)
    for slope in np.arange(0.80, 1.06, 0.01):
        for shift in np.arange(-0.03, 0.011, 0.005):
            sc = score(yv, calibrate(final, slope=slope, shift=shift))
            if sc > best[1]:
                best = ((round(slope, 3), round(shift, 4)), sc)
    print(f"    최적 (slope, shift) = {best[0]}  → {best[1]:.2f} "
          f"(원본 0.91/−0.01 대비 {best[1]-s:+.2f})")
    return preds, w, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="원본 재현 확인 (목표 raw 683.1 / cal 712.79)")
    ap.add_argument("--variants", action="store_true",
                    help="다양성 변형 학습 → 점수·불일치·블렌드 이득")
    ap.add_argument("--push", action="store_true",
                    help="ET/RF 격자 + 그리디 앙상블 + 캘리 재적합")
    ap.add_argument("--fit", default="2023", help="학습 시즌(쉼표 구분)")
    ap.add_argument("--val", type=int, default=2024)
    args = ap.parse_args()

    print(">> train.csv 로드")
    df = rd.load_train()

    if args.validate:
        for label, fits in (("2023 단독", [2023]),
                            ("≤2023 전체", [2019, 2020, 2021, 2022, 2023])):
            print(f"\n[fit={label} → val=2024]")
            raw, cal, yv, _ = validate(df, fits)
            print(f"    앙상블 raw       score = {score(yv, np.clip(raw,0,1)):8.2f}   "
                  f"(원본 보고 683.10)")
            print(f"    앙상블 calibrated score = {score(yv, cal):8.2f}   "
                  f"(원본 보고 712.79)")
            print(f"    예측평균 raw {raw.mean():.4f} → cal {cal.mean():.4f} "
                  f"(실제 r={yv.mean():.4f})")

    fits = [int(s) for s in args.fit.split(",")]
    if args.variants:
        run_variants(df, fit_seasons=fits, val_season=args.val)
    if args.push:
        push(df, fit_seasons=fits, val_season=args.val)


if __name__ == "__main__":
    main()