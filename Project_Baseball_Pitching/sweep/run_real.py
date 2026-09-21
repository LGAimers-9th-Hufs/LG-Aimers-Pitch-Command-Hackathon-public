# -*- coding: utf-8 -*-
"""실데이터 시뮬레이션 러너 v2 — refinement 게이트 + 재캘리 레이어 + 역방향 폴드.

    python sweep/run_real.py                              # 전량: V22·V23·V24 + R2019(역방향)
    python sweep/run_real.py --folds 2024,R2019           # 폴드 선택 (R2019 = 상승 레짐 리허설)
    python sweep/run_real.py --sample 300000 --no-record  # 코드 정합성 리허설 (의사결정 금지)

v2 설계 (269점 부검의 결론):
  1) **게이트 = refinement** = score + 1e5·d²/(r(1−r)) — 앵커 운을 제거한 순수 조건부 성능.
     269 사태의 근본 원인은 앵커에 오염된 게이트였다(K4가 앵커 운으로 1위 → 앵커가 반대로 뒤집힘).
     레벨(절편)은 별도 축: 사전 등록 β 격자에서 LB로 결정한다.
  2) **isotonic 제거**: refinement를 −150~−190 파괴하고 예측을 전년 수준에 고정해
     asof_prev* 적응 채널(주최가 2025 안에서 갱신해 주는 합법 온도계)과 싸운다. 비교용 arm만 유지.
  3) **재캘리 레이어(RC)**: logit p* = a + b·logit p + c·(z_pitcher−z̄) + d·(prev5−t̄) —
     cal 시즌에서 Brier 직접 최소화로 (a,b,c,d) 적합, 상수만 동봉(recal.py 참조).
  4) **역방향 폴드 R2019**: 로컬 유일의 상승 레짐 리허설(2025가 상승 반전했으므로 필수 guard).
"""
from __future__ import annotations
import argparse, dataclasses, hashlib, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor, RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# Windows 콘솔 cp949 → utf-8 (reconfigure: import한 쪽 stdout을 닫지 않는다)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_data as rd
import recal as rc
import reporting
from validation import competition_score

EPS = 1e-6
METRIC_KEYS = ("score", "refine", "refine_R", "refine_F", "brier", "logloss", "auc",
               "pred_mean", "d", "score_cold", "score_m1", "se_vs_anchor")
THERMO = "asof_pitcher_prev5_game_success_rate"   # 온도계 컬럼(시즌 수준 추적 기울기 0.83)


# ------------------------------------------------------------ 지표
def refinement(y, p):
    """score에서 앵커(평균 오차) 성분을 제거한 순수 조건부 성능. 클수록 좋음."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    r = y.mean()
    d = p.mean() - r
    return competition_score(y, p) + 1e5 * d * d / (r * (1 - r))


def metrics(y, p, name, role, seg=None, extra=None):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y)
    row = {"name": name, "role": role,
           "score": competition_score(y, p),
           "refine": refinement(y, p),
           "brier": float(brier_score_loss(y, p)),
           "logloss": float(log_loss(y, p, labels=[0, 1])),
           "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
           "pred_mean": float(p.mean()),
           "d": float(p.mean() - y.mean())}
    for code, key in ((0, "refine_R"), (1, "refine_F")):     # 세그먼트별 분해
        row[key] = float("nan")
        if seg is not None:
            m = seg == code
            if m.sum() > 1000 and len(np.unique(y[m])) > 1:
                row[key] = refinement(y[m], p[m])
    row.update(extra or {})
    return row


def paired_se(y, p_a, p_b):
    """두 예측의 ΔScore 표준오차(행 쌍 비교). 고정 임계 대신 비교마다 계산."""
    y = np.asarray(y, dtype=float)
    r = y.mean()
    diff = (np.asarray(p_a) - y) ** 2 - (np.asarray(p_b) - y) ** 2
    return float(1e5 * diff.std(ddof=1) / np.sqrt(len(diff)) / (r * (1 - r)))


# ------------------------------------------------------------ 베이스 모델
def _hgb(**kw):
    params = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                  min_samples_leaf=200, l2_regularization=1.0,
                  early_stopping=False, random_state=42)
    params.update(kw)
    return HistGradientBoostingClassifier(**params)


def _rf_official():
    """주최 베이스라인 레시피 그대로(원본 47컬럼). 앵커·비교 기준."""
    return ("raw47", Pipeline([
        ("pre", ColumnTransformer([
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
             rd.CAT_COLS),
            ("num", SimpleImputer(strategy="median"), rd.OFFICIAL_NUM)])),
        ("clf", RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=200,
                                       n_jobs=-1, random_state=42))]))


def _rf_est(**kw):
    """주최 RF와 동일 하이퍼파라미터의 분류기(표현만 바꿔 끼우기 위한 헬퍼)."""
    p = dict(n_estimators=100, max_depth=10, min_samples_leaf=200, n_jobs=-1, random_state=42)
    p.update(kw)
    return RandomForestClassifier(**p)


# --- 전이 손실 요인설계 (계획 Phase 2) -------------------------------------------------
# 로컬 게이트가 2025 순위를 뒤집으므로(우리 −248 / baseline +57), 원인을 LB 실측으로 가른다.
# 2×2: {raw47 공식 47컬럼, feat82 파생 포함} × {RF 배깅, HGB 부스팅}.
#   raw47×RF = rf_official (549.51 실측) · feat82×HGB = hgb82_reg · raw47×HGB = hgb47 · feat82×RF = rf82
# 여기에 glm_offset_hgb(500.61 실측, feat82×HGB+GLM오프셋)를 더하면 오프셋 구조의 효과까지 분리된다.
def _hgb47(**kw):
    """raw47 × HGB. HGB는 NaN을 분기로 직접 처리하므로 수치는 passthrough(대치하지 않는다)."""
    return ("raw47", Pipeline([
        ("pre", ColumnTransformer([
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
             rd.CAT_COLS),
            ("num", "passthrough", rd.OFFICIAL_NUM)])),
        ("clf", _hgb(**kw))]))


def _rf82():
    """feat82 × RF. RF는 결측 처리가 트리와 다르므로 rf_official과 동일하게 중앙값 대치를 건다."""
    return ("feat82", Pipeline([("imp", SimpleImputer(strategy="median")),
                                ("clf", _rf_est())]))


class _SeedAvg:
    """시드 평균 앙상블. Brier 이득 = 4e5·s²·(1−1/M) (ambiguity 분해) — 실현 이득을 직접 측정."""

    def __init__(self, seeds=(42, 43, 44), **kw):
        self.models = [_hgb(random_state=s, **kw) for s in seeds]
        self.spread_ = float("nan")

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return self

    def predict_proba(self, X):
        ps = np.stack([m.predict_proba(X)[:, 1] for m in self.models])
        self.spread_ = float(ps.std(axis=0).mean())          # 교차 시드 SD s
        p = ps.mean(axis=0)                                   # 확률 평균(볼록성 → Brier 비악화 보장)
        return np.column_stack([1 - p, p])


class _OffsetBoost:
    """GLM 오프셋 부스팅 (보험 GBM-with-offset 표준형, metric-native).

    1단: 로지스틱 GLM이 레벨 + 온도계(prev5) + 커리어·타자율을 **소유** → p0
    2단: HGBRegressor(squared_error)가 잔차 y−p0만 학습(상호작용 담당)
    → GLM 계수가 부스팅에 침식되지 않아 온도계의 시프트 추적 기울기가 보존된다.
    지표가 곧 제곱오차라 잔차 학습이 대회 손실을 직접 최적화한다.
    """
    GLM_COLS = [THERMO, "asof_pitcher_success_rate", "asof_pitcher_prev1_game_success_rate",
                "asof_batter_success_rate", "game_type", "balls_before", "strikes_before"]

    def __init__(self):
        self.glm = Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("lr", LogisticRegression(max_iter=1000))])
        self.gbr = HistGradientBoostingRegressor(
            loss="squared_error", max_iter=150, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=200, l2_regularization=1.0, random_state=42)

    def fit(self, X, y):
        p0 = self._p0_fit(X, y)
        self.gbr.fit(X, np.asarray(y, dtype=float) - p0)
        return self

    def _p0_fit(self, X, y):
        self.glm.fit(X[self.GLM_COLS], y)
        return self.glm.predict_proba(X[self.GLM_COLS])[:, 1]

    def predict_proba(self, X):
        p = np.clip(self.glm.predict_proba(X[self.GLM_COLS])[:, 1]
                    + self.gbr.predict(X), 1e-3, 1 - 1e-3)
        return np.column_stack([1 - p, p])


class _DropCols:
    """짝 대조용 래퍼 — 새 DERIVED 컬럼을 fit/predict 직전에 떨궈 '피처 추가 전' 모델을
    같은 런·같은 행에서 재현한다(DERIVED 변경이 전 arm의 X를 바꾸므로 paired SE에 필수).
    사용 예: BASE_MODELS에 ("feat82", _DropCols(_OffsetBoost(), 새컬럼목록)) 대조 arm 등록.
    (S1.5에서 이 패턴으로 cpk/spc 9피처의 −15를 검출 — D-15.)"""

    def __init__(self, inner, cols):
        self.inner, self.cols = inner, list(cols)

    def fit(self, X, y):
        self.inner.fit(X.drop(columns=self.cols), y)
        return self

    def predict_proba(self, X):
        return self.inner.predict_proba(X.drop(columns=self.cols))


class _OffsetBoostZ(_OffsetBoost):
    """N11: as-of 투수 z(SERVE 리터럴 Z_ASOF, K=500·ρ=0.85 — z_asof.py 스캔으로 동결)를
    GLM 항으로 소유시킨다. 재캘리 레이어로 기각된 z(D-11a)의 '레이어가 아니라 GLM 안' 재시도.
    bundle_model이 est.GLM_COLS를 그대로 담아 서빙에 자동 전파된다."""
    GLM_COLS = _OffsetBoost.GLM_COLS + ["z_asof"]


BASE_MODELS = {
    "rf_official":    ("T0", _rf_official),
    "hgb82":          ("T1", lambda: ("feat82", _hgb())),
    "hgb82_reg":      ("T1", lambda: ("feat82", _hgb(max_iter=150))),      # +39 ref 실측
    "hgb82_reg_avg3": ("T1", lambda: ("feat82", _SeedAvg(max_iter=150))),
    "glm_offset_hgb": ("T2", lambda: ("feat82", _OffsetBoost())),
    "glm_offset_hgb_z": ("T2", lambda: ("feat82", _OffsetBoostZ())),       # N11 (S1) — D-14 기각
    # Phase 2 요인설계 측정 후보 — 전부 β=0·재캘리 없음으로 제출해 2025 점수를 정확히 잰다.
    "hgb47":        ("T1", lambda: _hgb47(max_iter=150)),                  # raw47 × HGB
    "rf82":         ("T1", lambda: _rf82()),                               # feat82 × RF
    "hgb82_strong": ("T1", lambda: ("feat82", _hgb(max_iter=60, max_depth=4,   # 강정칙(과적합 축)
                                                   max_leaf_nodes=None,
                                                   min_samples_leaf=2000,
                                                   l2_regularization=10.0))),
    # (N13 대조 arm 2종은 D-17 미채택으로 제거 — DERIVED에 tm_* 컬럼이 없으면 _DropCols가 죽는다.
    #  재실험 시: trackman.py --profile 로 리터럴 복원 후 아래 두 줄을 되살린다.
    #    "glm_offset_hgb_notm": ("T2", lambda: ("feat82", _DropCols(_OffsetBoost(), rd.TM_COLS))),
    #    "hgb82_reg_notm":      ("T1", lambda: ("feat82", _DropCols(_hgb(max_iter=150), rd.TM_COLS))),)
}


# ------------------------------------------------------------ 캐시
def _fit_predict(model_name, factory, fold, key, df, X, y, cache_dir, verbose=True):
    tag = f"{model_name}_{fold['val_season']}_{fold['mode']}_{key}"
    cache = (cache_dir / f"{tag}.npz") if cache_dir else None
    if cache and cache.exists():
        z = np.load(cache)
        return z["p_cal"], z["p_val_m1"], z["p_val"], 0.0

    rep, model = factory()
    data = df[rd.CAT_COLS + rd.OFFICIAL_NUM] if rep == "raw47" else X

    t0 = time.time()
    model.fit(data.iloc[fold["fit1"]], y[fold["fit1"]])
    p_cal = model.predict_proba(data.iloc[fold["cal"]])[:, 1]
    p_val_m1 = model.predict_proba(data.iloc[fold["val"]])[:, 1]
    rep2, model2 = factory()
    model2.fit(data.iloc[fold["fit2"]], y[fold["fit2"]])
    p_val = model2.predict_proba(data.iloc[fold["val"]])[:, 1]
    dt = time.time() - t0
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, p_cal=p_cal, p_val_m1=p_val_m1, p_val=p_val)
    if verbose:
        print(f"  {model_name:<15} fit+predict {dt:>5.0f}s"
              f"  cal_mean={p_cal.mean():.4f} val_mean={p_val.mean():.4f}")
    return p_cal, p_val_m1, p_val, dt


# ------------------------------------------------------------ 폴드 실행
def run_fold(df, X, y, fold, model_names, cache_dir, feat_key, pool_ctx=None, verbose=True):
    """pool_ctx: {model: [cal 배치, ...]} — 시간순으로 누적되는 과거 cal 시즌 OOF.
    RC2(풀링 적합)가 여기서 나온다: 단일 cal 시즌(예: 2023 F 파단)이 병리적이면
    Brier 최소화가 '모델을 버려라'(b≈0)로 퇴화하는데, 시즌을 합치면 희석된다."""
    vs, mode = fold["val_season"], fold["mode"]
    val_i, cal_i = fold["val"], fold["cal"]
    y_val, y_cal = y[val_i], y[cal_i]
    r_val = float(y_val.mean())

    seg_val = df[rd.SEGMENT].iloc[val_i].map(rd.CAT_MAPS[rd.SEGMENT]).to_numpy()
    cold_val = X["p_coldstart"].iloc[val_i].to_numpy() > 0.5
    career_val = df["asof_pitcher_success_rate"].iloc[val_i].to_numpy()

    # 투수 신뢰도 테이블 — fit1 기간으로 적합용(누수 0), fit2 기간으로 적용용(제출과 동형)
    tab1, zdef1, K1, m1 = rd.build_credibility(df.iloc[fold["fit1"]])
    tab2, zdef2, K2, m2 = rd.build_credibility(df.iloc[fold["fit2"]])
    pid_cal = df["pitcher_id"].iloc[cal_i].to_numpy()
    pid_val = df["pitcher_id"].iloc[val_i].to_numpy()
    z_cal = np.array([tab1.get(int(p), zdef1) for p in pid_cal])
    z_val = np.array([tab2.get(int(p), zdef2) for p in pid_val])
    t_cal = df[THERMO].iloc[cal_i].to_numpy(dtype=float)
    t_val = df[THERMO].iloc[val_i].to_numpy(dtype=float)
    t_center = float(np.nanmean(df[THERMO].iloc[fold["fit1"]].to_numpy(dtype=float)))

    if verbose:
        arrow = "↑역방향" if mode == "reverse" else "→"
        print(f"\n{'='*88}\n[fold {arrow}] fit1={fold['fit1_seasons']} cal={df[rd.SEASON].iloc[cal_i].iloc[0]} "
              f"fit2={fold['fit2_seasons']} val={vs}  (실제 r={r_val:.4f})"
              f"  credibility K={K2:.0f} m={m2:.4f}")

    def M(p, name, role, p_m1=None):
        extra = {"score_cold": refinement(y_val[cold_val], np.clip(p, EPS, 1 - EPS)[cold_val])
                 if cold_val.sum() > 100 and len(np.unique(y_val[cold_val])) > 1 else float("nan"),
                 "score_m1": competition_score(y_val, np.clip(p_m1, EPS, 1 - EPS))
                 if p_m1 is not None else float("nan")}
        return metrics(y_val, p, name, role, seg=seg_val, extra=extra)

    rows, preds = [], {}
    for nm, tgt in (("const_prev", float(y_cal.mean())), ("const_ORACLE", r_val)):
        p = np.full(len(val_i), tgt)
        rows.append(M(p, nm, "oracle" if "ORACLE" in nm else "reference"))
        preds[nm] = p

    anchor_p = None
    for mname in model_names:
        tier, factory = BASE_MODELS[mname]
        p_cal, p_val_m1, p_val, _ = _fit_predict(mname, factory, fold, feat_key,
                                                 df, X, y, cache_dir, verbose)
        variants = [("K0", p_val, p_val_m1)]

        # 비교용 isotonic (제출 스택에서는 폐기 — refinement 세금 실측 기록용)
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal, y_cal)
        variants.append(("K1_iso", iso.predict(np.clip(p_val, 0, 1)),
                         iso.predict(np.clip(p_val_m1, 0, 1))))

        # RC — 재캘리 레이어 (v2 핵심). cal에서 (a,b,c,d) 적합 → val에 행 단위 적용, β=0
        params = rc.fit_recal(p_cal, y_cal, z_cal, t_cal, zdef1, t_center)
        p_rc = rc.apply_recal(p_val, z_val, t_val, params)
        p_rc_m1 = rc.apply_recal(p_val_m1, z_val, t_val, params)
        variants.append(("RC", p_rc, p_rc_m1))

        # RC2 — 풀링 적합: 과거 cal 시즌들 + 현재 cal을 합쳐 (a,b,c,d) 적합 (제출 구성과 동형)
        batch = {"p": p_cal, "y": y_cal, "z": z_cal, "t": t_cal, "zc": zdef1, "tc": t_center}
        if pool_ctx is not None:
            hist = pool_ctx.setdefault(mname, [])
            if hist:                                          # 배치 2개 이상일 때만 의미
                params2 = rc.fit_recal_pooled(hist + [batch])
                variants.append(("RC2", rc.apply_recal(p_val, z_val, t_val, params2),
                                 rc.apply_recal(p_val_m1, z_val, t_val, params2)))
                if verbose and mname in ("hgb82_reg", "glm_offset_hgb"):
                    print(f"    {mname}+RC2({params2['n_pool']}시즌 풀) 계수: "
                          f"a={params2['a']:+.3f} b={params2['b']:.3f} "
                          f"c={params2['c']:+.3f} d={params2['d']:+.3f}")
            hist.append(batch)
        if verbose and mname in ("hgb82_reg", "glm_offset_hgb"):
            print(f"    {mname}+RC 계수: a={params['a']:+.3f} b={params['b']:.3f} "
                  f"c={params['c']:+.3f} d={params['d']:+.3f}")
            ts, ps_, ratio = rc.dispersion_report(y_val, p_val, career_val)
            ts2, ps2, ratio2 = rc.dispersion_report(y_val, p_rc, career_val)
            print(f"    십분위 스프레드 실측 {ts:.4f} | K0 예측 {ps_:.4f} (비율 {ratio:.2f})"
                  f" → RC 예측 {ps2:.4f} (비율 {ratio2:.2f})")
            dec = rc.brier_decomposition(y_val, p_rc)
            print(f"    Brier 분해(RC): REL={dec['REL']*1e5:.1f} RES={dec['RES']*1e5:.1f} "
                  f"UNC={dec['UNC']:.4f} (점수 스케일 ×1e5/UNC: REL {1e5*dec['REL']/dec['UNC']:.0f} "
                  f"/ RES {1e5*dec['RES']/dec['UNC']:.0f})")

        for suffix, p, p_m1 in variants:
            nm = f"{mname}+{suffix}"
            rows.append(M(p, nm, "candidate", p_m1))
            preds[nm] = p
        if isinstance(factory()[1], _SeedAvg):
            pass  # spread는 캐시 경로에선 측정 불가(기록 생략)

        if mname == "rf_official":
            anchor_p = p_val

    for r in rows:
        r["val_season"] = vs
        r["se_vs_anchor"] = (paired_se(y_val, preds[r["name"]], anchor_p)
                             if anchor_p is not None else float("nan"))
    return rows


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="2022,2023,2024,R2019",
                    help="전진 시즌 목록 + 'R2019'(역방향 상승 리허설)")
    ap.add_argument("--models", default="rf_official,hgb82,hgb82_reg,hgb82_reg_avg3,glm_offset_hgb")
    ap.add_argument("--mode", default="refit", choices=["refit", "gap"])
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--cache-dir", default="results/cache")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args()

    fw_seasons, reverse = [], []
    for tok in args.folds.split(","):
        tok = tok.strip()
        if tok.upper().startswith("R"):
            reverse.append(int(tok[1:]))
        elif tok:
            fw_seasons.append(int(tok))
    model_names = []
    for m in args.models.split(","):
        if m in BASE_MODELS:
            model_names.append(m)
        elif m:
            print(f"경고: 알 수 없는 모델 '{m}' 무시")
    if not model_names:
        sys.exit("실행할 모델이 없습니다.")

    t0 = time.time()
    print(">> [load] train.csv")
    df = rd.load_train()
    if args.sample:
        per = max(1000, args.sample // df[rd.SEASON].nunique())
        idx = np.concatenate([g.sample(min(len(g), per), random_state=42).index.to_numpy()
                              for _, g in df.groupby(rd.SEASON)])
        df = df.loc[np.sort(idx)].reset_index(drop=True)
        print(f"   ⚠ 서브샘플 {len(df):,}행 — 의사결정 금지")
    print(f"   {df.shape[0]:,}행  ({time.time()-t0:.0f}s)")
    for s in [None] + sorted(df[rd.SEGMENT].dropna().unique()):
        mm = rd.season_means(df, s)
        lbl = "전체" if s is None else f"{rd.SEGMENT}={s}"
        print(f"   {lbl}: " + " ".join(f"{int(k)}:{v:.4f}" for k, v in mm.items()))

    X = rd.build_features(df)
    y = df[rd.TARGET].to_numpy()
    # SERVE_VERSION을 캐시 키에 섞는다 — 리터럴(Z_ASOF 등) 값만 바뀌고 컬럼명이 그대로일 때
    # 낡은 npz가 재사용되는 거짓 판독을 차단(docs/research/13 §5 함정 3).
    feat_key = hashlib.md5(("|".join(X.columns) + f"|{len(df)}|"
                            + getattr(rd, "SERVE_VERSION", "0")).encode()).hexdigest()[:8]
    cache_dir = None if args.no_cache else Path(args.cache_dir)

    folds = sorted(rd.make_folds(df, val_seasons=fw_seasons, mode=args.mode),
                   key=lambda f: f["val_season"])
    folds += [rd.make_reverse_fold(df, vs) for vs in reverse]

    # 역방향 폴드용 X: as-of z는 2019 라벨이 fit 행 집계에 새므로(R2019의 val이 2019),
    # 2019 제외로 재계산한 테이블로 교체 — refR2019 게이트의 낙관 오염 차단.
    X_rev = X
    if reverse and "z_asof" in X.columns and X["z_asof"].notna().any():
        import z_asof as za
        meta = getattr(rd, "Z_ASOF_META", {})
        tab_rev = za.build_z_asof(df, K=meta.get("K", "moment"),
                                  rho=meta.get("rho", 1.0), exclude=(2019,))
        X_rev = X.copy()
        zkey = df["pitcher_id"].astype("int64") * 10000 + df[rd.SEASON].astype("int64")
        X_rev["z_asof"] = zkey.map(tab_rev).astype("float32")

    all_rows, pool_ctx = [], {}
    for f in folds:
        # 전진 폴드는 cal 시즌 오름차순으로 돌아 pool_ctx가 과거→현재로 누적된다.
        # 역방향 폴드는 풀링 제외(시간 방향이 반대라 섞으면 안 됨).
        ctx = pool_ctx if f["mode"] != "reverse" else None
        Xf = X_rev if f["mode"] == "reverse" else X
        all_rows += run_fold(df, Xf, y, f, model_names, cache_dir, feat_key, pool_ctx=ctx)

    # ---- 집계: meta/vals 분리 (구버전 크래시 방지 구조 유지)
    agg = {}
    for r in all_rows:
        a = agg.setdefault(r["name"], {"meta": {"name": r["name"], "role": r["role"]},
                                       "vals": {k: [] for k in METRIC_KEYS},
                                       "by_fold": {}})
        for k in METRIC_KEYS:
            a["vals"][k].append(r.get(k, float("nan")))
        a["by_fold"][r["val_season"]] = r

    leaderboard = []
    for name, a in agg.items():
        row = dict(a["meta"])
        for k, v in a["vals"].items():
            row[k] = float(np.nanmean(v)) if len(v) else float("nan")
        row["bss"] = row["score"] / 1e5
        row["n_folds"] = len(a["vals"]["score"])
        # 주 게이트: V24 refinement / guard: 전 폴드 최소 refinement
        bf = a["by_fold"]
        row["refine_v24"] = bf.get(2024, {}).get("refine", float("nan"))
        row["refine_min"] = float(np.nanmin([x["refine"] for x in bf.values()])) if bf else float("nan")
        row["refine_r2019"] = bf.get(2019, {}).get("refine", float("nan"))
        row["d_r2019"] = bf.get(2019, {}).get("d", float("nan"))
        leaderboard.append(row)

    anchor = next((r for r in leaderboard if r["name"] == "rf_official+K0"), None)
    anchor_ref = anchor["refine_v24"] if anchor else float("nan")
    for r in leaderboard:
        r["delta_vs_anchor"] = r["refine_v24"] - anchor_ref
        r["beats_anchor"] = bool(r["role"] == "candidate" and r["refine_v24"] > anchor_ref)
    cands = sorted([r for r in leaderboard if r["role"] == "candidate"],
                   key=lambda r: -(r["refine_v24"] if np.isfinite(r["refine_v24"]) else -1e9))
    refs = sorted([r for r in leaderboard if r["role"] != "candidate"], key=lambda r: -r["score"])

    print(f"\n{'='*104}\n리더보드 — 1차 정렬 = V24 refinement (앵커 운 제거). 레벨은 β로 별도 통제.\n{'='*104}")
    print(f"{'#':<3}{'후보':<24}{'refV24':>9}{'refMIN':>9}{'refR2019':>10}{'d@R2019':>9}"
          f"{'refR':>8}{'refF':>8}{'AUC':>7}{'Cold':>8}")
    for i, r in enumerate(cands[:30], 1):
        print(f"{i:<3}{r['name']:<24}{r['refine_v24']:>9.1f}{r['refine_min']:>9.1f}"
              f"{r['refine_r2019']:>10.1f}{r['d_r2019']:>9.4f}{r['refine_R']:>8.1f}"
              f"{r['refine_F']:>8.1f}{r['auc']:>7.4f}{r['score_cold']:>8.1f}")
    print(f"\n앵커 rf_official+K0 refinement@V24 = {anchor_ref:.1f}")
    print("refMIN = 전 폴드 최소(강건성 guard) · refR2019 = 상승 레짐 리허설 · Cold = 콜드스타트 층 refinement")

    if not args.no_record:
        @dataclasses.dataclass
        class RealSpec:
            target: str = rd.TARGET
            season: str = rd.SEASON
            segment: str = rd.SEGMENT
            n_features: int = 0
            val_seasons: str = ""
            fold_mode: str = "refit"
            gate: str = "refine_v24"
        spec = RealSpec(n_features=X.shape[1], val_seasons=args.folds, fold_mode=args.mode)
        meta = reporting.collect_meta(spec, seed=42, data_source="real", tag=args.tag)
        report = {
            "harness": {"anchor": "rf_official+K0", "anchor_score": anchor["score"] if anchor else None,
                        "anchor_refine_v24": anchor_ref,
                        "metric": "brier_skill_score_x100000", "gate": "refine_v24",
                        "higher_is_better": True, "official_baseline_lb": 549.51,
                        "lb_facts": {"ours_s1": 269.04861, "ours_new1": 457.2942506222,
                                     "ours_new3": 500.6103395903,
                                     "baseline": 549.51193, "top1": 931.0,
                                     # β쌍(new1/new3, 동일 모델) 연립 — refinement 상쇄로 가장 정밀
                                     "r2025_backsolve": 0.499}},
            "candidates": [{"name": n} for n in agg],
            "leaderboard": cands + refs,
            "per_fold": all_rows,
            "season_means": {int(k): float(v) for k, v in rd.season_means(df).items()},
            "segment_means": {str(s): {int(k): float(v) for k, v in rd.season_means(df, s).items()}
                              for s in sorted(df[rd.SEGMENT].dropna().unique())},
        }
        run_dir = reporting.record_run(report, {"n_rows": int(len(df))}, meta)
        print(f"\n기록: {run_dir}")
    print(f"총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
