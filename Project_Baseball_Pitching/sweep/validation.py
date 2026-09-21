"""
시간분할(forward-chaining) 검증 + 캘리브레이션 + proper-score 평가.

핵심(05 6장 / 08 6장):
  - 랜덤 KFold 금지. 시즌 기준 전진 검증(train=과거 시즌, val=미래 시즌).
  - **게이트 지표 = Brier Skill Score (BSS)** — 대회 공식 지표(2026-08-05 확정).
    BSS = 1 − Brier / (r(1−r)),  r = 해당 val 폴드의 실제 성공률(기후값 기준선).
    대회 점수 = max(0, 100000 × BSS). 클수록 좋음.
    LogLoss는 **병기 진단용**(확신 오류 감시), AUC는 참고만.
  - 캘리브레이션: 기본은 train 내 최신 시즌을 calib 홀드아웃으로 isotonic 적합 → val에 적용.

시간분할 정합 (fold-aware 피처):
  - 피처는 **폴드별로** 생성된다(cutoff=val 시즌 → target-파생 통계 동결, features.build_features).
    evaluate_candidate는 `get_rep(cand, val_season)`으로 그 폴드의 입력 표현을 받는다.

캘리 모드 (C5, 게이트 불변):
  - "isotonic_latest"(기본): 현행. 그 외 arm으로 recent_window / prior_shift / venn_abers 선택 가능.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression

EPS = 1e-6


def competition_score(y_true, p):
    """대회 공식 산식. Score = max(0, 100000 × (1 − Brier / (r(1−r)))), r = 평가셋 실제 성공률.
    raw=True면 음수도 반환(클리핑 전) — 진단·게이트에는 raw를 쓴다(0으로 뭉개면 순위가 사라짐)."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    r = y.mean()
    ref = r * (1.0 - r)
    if ref <= 0:
        return float("nan")
    return 100000.0 * (1.0 - float(np.mean((p - y) ** 2)) / ref)


def make_time_folds(meta, spec, min_train_seasons=3):
    """전진 검증 폴드: 각 폴드는 (train=이전 시즌 전부, val=다음 시즌)."""
    seasons = sorted(meta[spec.season].unique())
    folds = []
    for i in range(min_train_seasons, len(seasons)):
        val_season = seasons[i]
        train_mask = meta[spec.season] < val_season
        val_mask = meta[spec.season] == val_season
        folds.append((np.where(train_mask.values)[0], np.where(val_mask.values)[0],
                      seasons[:i], val_season))
    return folds


def _take(X, idx):
    """flat(DataFrame)·seq(np.ndarray)·seq_kd(tuple) 표현을 같은 폴드 인덱스로 슬라이스."""
    if isinstance(X, tuple):
        return tuple(_take(x, idx) for x in X)
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


def _p1(model, X):
    p = np.asarray(model.predict_proba(X))
    return p[:, 1] if p.ndim == 2 else p


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _fit_calibrator(p_raw, y, mode, fit_rate=None, calib_rate=None):
    """캘리 map 함수 반환. p_raw/y = calib 홀드아웃. (C4/C5 arm은 여기서 분기.)"""
    p_raw = np.clip(p_raw, EPS, 1 - EPS)
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_raw, y)
    if mode == "prior_shift":
        # train fit→calib base-rate 드리프트를 val로 외삽(val 라벨 미사용). 로짓 가산 오프셋.
        off = 0.0
        if fit_rate is not None and calib_rate is not None:
            off = _logit(calib_rate) - _logit(fit_rate)
        return lambda p: 1.0 / (1.0 + np.exp(-(_logit(np.clip(iso.predict(np.clip(p, EPS, 1 - EPS)), EPS, 1 - EPS)) + off)))
    # isotonic_latest / recent_window / venn_abers(근사: isotonic) 공통
    return lambda p: iso.predict(np.clip(p, EPS, 1 - EPS))


def _evaluate_one_fold(cand, X, yv, y, meta, spec, fold, calib_mode):
    tr_idx, va_idx, tr_seasons, va_season = fold
    calib_season = tr_seasons[-1]
    tr_meta = meta.iloc[tr_idx]

    if calib_mode == "recent_window":  # train 최근 30% 행을 calib으로
        order = np.argsort(meta.iloc[tr_idx][spec.season].values, kind="stable")
        cut = int(len(tr_idx) * 0.7)
        fit_local, calib_local = tr_idx[order[:cut]], tr_idx[order[cut:]]
    else:                               # 기본: train 내 최신 시즌 = calib 홀드아웃
        fit_local = tr_idx[(tr_meta[spec.season] < calib_season).values]
        calib_local = tr_idx[(tr_meta[spec.season] == calib_season).values]
    if len(fit_local) == 0 or len(calib_local) == 0:
        fit_local, calib_local = tr_idx, tr_idx

    model = cand.build()
    model.fit(_take(X, fit_local), y.iloc[fit_local])

    p_cal_raw = np.clip(_p1(model, _take(X, calib_local)), EPS, 1 - EPS)
    calibrate = _fit_calibrator(p_cal_raw, yv[calib_local], calib_mode,
                                fit_rate=float(yv[fit_local].mean()),
                                calib_rate=float(yv[calib_local].mean()))

    p_va_raw = np.clip(_p1(model, _take(X, va_idx)), EPS, 1 - EPS)
    p_va_cal = np.clip(calibrate(p_va_raw), EPS, 1 - EPS)

    yv_va = yv[va_idx]
    row = {
        "val_season": va_season,
        "logloss_raw": log_loss(yv_va, p_va_raw, labels=[0, 1]),
        "logloss_cal": log_loss(yv_va, p_va_cal, labels=[0, 1]),
        "brier_cal": brier_score_loss(yv_va, p_va_cal),
        "auc": roc_auc_score(yv_va, p_va_raw) if len(np.unique(yv_va)) > 1 else float("nan"),
    }
    # 기록용(reliability diagram 재생성 재료). 지표·게이트에는 미사용.
    row["reliability"] = _reliability_bins(yv_va, p_va_cal)
    # 콜드스타트 층화(F13 arm 평가용): 얕은 히스토리 구간의 LogLoss 병기.
    if hasattr(X, "columns") and "p_coldstart" in X.columns:
        cold = X.iloc[va_idx]["p_coldstart"].to_numpy() > 0.5
        if cold.sum() > 0 and len(np.unique(yv_va[cold])) > 1:
            row["logloss_cold"] = log_loss(yv_va[cold], p_va_cal[cold], labels=[0, 1])
    return row


def _reliability_bins(y_true, p, n_bins=15):
    """등폭 bin 요약 {edges, mean_pred, frac_pos, count} — 원시 예측 저장 없이
    calibration diagram/ECE를 재생성하기 위한 최소 통계(빈 bin 제외)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    out = {"edges": edges.tolist(), "mean_pred": [], "frac_pos": [], "count": [], "bin": []}
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        out["bin"].append(int(b))
        out["mean_pred"].append(float(p[m].mean()))
        out["frac_pos"].append(float(np.asarray(y_true)[m].mean()))
        out["count"].append(int(m.sum()))
    return out


def evaluate_candidate(cand, get_rep, y, meta, spec, folds, calib_mode="isotonic_latest"):
    """후보를 폴드들에서 학습·평가. get_rep(cand, val_season) → 그 폴드의 입력 표현(fold-aware).
    반환: dict(mean logloss/brier/auc, per-fold)."""
    yv = y.values
    rows = [_evaluate_one_fold(cand, get_rep(cand, f[3]), yv, y, meta, spec, f, calib_mode)
            for f in folds]
    cold_vals = [r["logloss_cold"] for r in rows if "logloss_cold" in r]
    return {
        "name": cand.name, "tier": cand.tier,
        "logloss_cal": float(np.mean([r["logloss_cal"] for r in rows])),
        "logloss_raw": float(np.mean([r["logloss_raw"] for r in rows])),
        "brier_cal": float(np.mean([r["brier_cal"] for r in rows])),
        "auc": float(np.nanmean([r["auc"] for r in rows])),
        "logloss_cold": float(np.mean(cold_vals)) if cold_vals else float("nan"),
        "per_fold": rows,
    }
