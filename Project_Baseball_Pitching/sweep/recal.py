# -*- coding: utf-8 -*-
"""GLM 재캘리브레이션 레이어 + Brier 진단 유틸 (파이프라인 v2의 심장).

구조 (CTR field-aware calibration / 임상 모델 업데이트의 표준형):

    logit p* = a + b·logit(p_gbdt) + c·(z_pitcher − z_c) + d·(t_prev5 − t_c) + β

  - (a,b,c,d)는 cal 시즌 OOF 예측에서 **Brier를 직접 최소화**해 적합 (지표가 곧 목적함수 — EMOS 원칙)
  - b>1 이 전역 과수축 교정, c>0 가 투수(field) 단위 잔여 편향 주입,
    d 가 "온도계"(prev5 — 시즌 수준을 기울기 0.83로 추적하는 행 단위 합법 신호)를 제 기울기로 복원
    → d 항이 곧 **합법적 label-shift 보정**이다 (2025에서 t가 오르면 예측이 따라 오른다)
  - 절편 β는 자유 스칼라로 **분리**(임상 문헌의 "절편만 갱신" 원칙): 레벨은 500그루 트리 속이 아니라
    통제 가능한 숫자 1개에 둔다. β는 사전 등록 격자에서 LB로 선택.
  - 상수 4~6개만 동봉 → isotonic(수백 절점, refinement −150~−190 실측)과 달리 과적합 여지가 작다.

전부 행 단위 적용 = 규정 §5 안전.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# 계수 경계 — cal 시즌이 병리적(예: 2023 F 파단)일 때 "모델을 버려라"(b≤0)로 퇴화하는 것을 막는다.
# b ∈ [0.2, 3]: 모델 로짓을 최소한 20%는 신뢰. 정상 연도에서는 경계에 닿지 않는다(실측 b≈1.0~1.1).
BOUNDS = [(-0.5, 0.5), (0.2, 3.0), (-2.0, 2.0), (-3.0, 3.0)]


def _design(p, z, t, z_center, t_center):
    x0 = _logit(p)
    xz = np.nan_to_num(np.asarray(z, dtype=float), nan=z_center) - z_center
    xt = np.nan_to_num(np.asarray(t, dtype=float), nan=t_center) - t_center
    return np.column_stack([np.ones_like(x0), x0, xz, xt])


def _fit_design(X, y, bounds):
    y = np.asarray(y, dtype=float)

    def obj(theta):
        p = _sigmoid(X @ theta)
        r = p - y
        return float(np.mean(r * r)), 2.0 * (X.T @ (r * p * (1 - p))) / len(y)

    return minimize(obj, x0=np.array([0.0, 1.0, 0.0, 0.0]), jac=True,
                    method="L-BFGS-B", bounds=bounds)


def fit_recal(p_cal, y_cal, z_cal, t_cal, z_center, t_center, bounds=BOUNDS):
    """(a,b,c,d)를 cal 표본에서 Brier 직접 최소화로 적합(해석적 기울기 + 경계).

    p_cal: fit1 모델의 cal 시즌 예측 / z_cal: 투수 신뢰도 로짓(fit1 기간 테이블 → cal 누수 없음)
    t_cal: prev5 (NaN은 t_center로 대치) / centers: fit1 기간 상수(동봉되어 서빙에서도 동일 사용)
    """
    X = _design(p_cal, z_cal, t_cal, z_center, t_center)
    res = _fit_design(X, y_cal, bounds)
    a, b, c, d = (float(v) for v in res.x)
    return {"a": a, "b": b, "c": c, "d": d,
            "z_center": float(z_center), "t_center": float(t_center),
            "cal_brier": float(res.fun)}


def fit_recal_pooled(batches, bounds=BOUNDS):
    """여러 cal 시즌의 OOF를 **합쳐서** (a,b,c,d) 적합 — 단일 cal 시즌이 병리적 연도일 때의 방어.

    batches: [{"p","y","z","t","zc","tc"}, ...] — 각 배치는 자기 기간 상수로 센터링(경계 시즌의
    p는 그 시즌 이전 모델의 예측이라 누수 0). CTR 캘리 레이어를 롤링 윈도로 적합하는 관행과 동형.
    반환 params의 center는 **마지막(최신) 배치** 기준 — 적용 시 그 상수를 쓴다.
    """
    X = np.vstack([_design(b["p"], b["z"], b["t"], b["zc"], b["tc"]) for b in batches])
    y = np.concatenate([np.asarray(b["y"], dtype=float) for b in batches])
    res = _fit_design(X, y, bounds)
    a, bb, c, d = (float(v) for v in res.x)
    last = batches[-1]
    return {"a": a, "b": bb, "c": c, "d": d,
            "z_center": float(last["zc"]), "t_center": float(last["tc"]),
            "cal_brier": float(res.fun), "n_pool": len(batches)}


def apply_recal(p, z, t, params, beta=0.0):
    """행 단위 적용. z/t의 NaN은 동봉된 center 상수로 대치."""
    zl = np.nan_to_num(np.asarray(z, dtype=float), nan=params["z_center"]) - params["z_center"]
    td = np.nan_to_num(np.asarray(t, dtype=float), nan=params["t_center"]) - params["t_center"]
    return _sigmoid(params["a"] + params["b"] * _logit(p)
                    + params["c"] * zl + params["d"] * td + beta)


# ---------------------------------------------------------------- 진단
def brier_decomposition(y, p, n_bins=20):
    """Murphy 분해: Brier = REL − RES + UNC. (BSS = (RES−REL)/UNC)
    REL 낮을수록·RES 높을수록 좋다. 남은 격차가 reliability(싸게 고침)인지
    resolution(새 정보 필요)인지를 가르는 대시보드."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    ybar = y.mean()
    unc = ybar * (1 - ybar)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges[1:-1], p), 0, n_bins - 1)
    rel = res = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        w = m.mean()
        pb, yb = p[m].mean(), y[m].mean()
        rel += w * (pb - yb) ** 2
        res += w * (yb - ybar) ** 2
    return {"REL": float(rel), "RES": float(res), "UNC": float(unc),
            "brier": float(np.mean((p - y) ** 2))}


def dispersion_report(y, p, career_rate, n_dec=10):
    """투수 이력(커리어 성공률) 십분위별 실측 vs 예측 스프레드 — 과수축 진단.
    반환: (true_spread, pred_spread, ratio). ratio<1 = 여전히 과수축."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    cr = np.asarray(career_rate, dtype=float)
    ok = ~np.isnan(cr)
    if ok.sum() < 1000:
        return float("nan"), float("nan"), float("nan")
    q = np.quantile(cr[ok], np.linspace(0, 1, n_dec + 1))
    idx = np.clip(np.searchsorted(q[1:-1], cr[ok]), 0, n_dec - 1)
    ty = np.array([y[ok][idx == b].mean() for b in range(n_dec) if (idx == b).any()])
    tp = np.array([p[ok][idx == b].mean() for b in range(n_dec) if (idx == b).any()])
    ts, ps = float(ty.max() - ty.min()), float(tp.max() - tp.min())
    return ts, ps, (ps / ts if ts > 0 else float("nan"))
