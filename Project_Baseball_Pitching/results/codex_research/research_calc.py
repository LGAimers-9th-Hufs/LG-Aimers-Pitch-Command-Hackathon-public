#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""남은 LB 프로브 연구의 재현 계산.

이 스크립트는 읽기 입력으로 train.csv와 results/ens9/preds_val*.npz만
사용하고, 출력은 이 파일과 같은 디렉터리의 calculations.json에만 쓴다.

실행:
    python results/codex_research/research_calc.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "calculations.json"
TRAIN = ROOT / "data" / "data" / "train.csv"
ENS9 = ROOT / "results" / "ens9"

C_2025 = 402_463.285680
V_2025 = 1e5 / C_2025
S_A_2025 = 1025.5511136843
RAW_SLOPE_CURVATURE_2025 = 786.2
TRAIN_W_F = 0.109148
TRUE_W_F_2025 = 0.113674589546  # b=40557.5와 train 중심화 상수를 사용한 정확식
TRUE_B_F_2025 = 40_557.5
# 블렌드된 최종확률 자체에 다시 affine slope를 거는 방향의 곡률.
# 사용자 제공 |score cross coefficient|=2287=2|H_gA|, corr(g_F,A)=-0.184,
# H_gg=40557.5를 결합한 추론값이다. raw-slope의 786.2와 구분한다.
POST_BLEND_SLOPE_CURVATURE_2025 = (2287 / 2 / (0.184 * np.sqrt(TRUE_B_F_2025))) ** 2
ENS9_W = {
    "nn_lin_pkg": 0.475,
    "et_l100_d28": 0.250,
    "allraw_l15": 0.125,
    "june_l15": 0.100,
    "june_l7": 0.050,
}


def score_raw(y: np.ndarray, p: np.ndarray) -> float:
    r = float(np.mean(y))
    return float(1e5 * (1.0 - np.mean((p - y) ** 2) / (r * (1.0 - r))))


def ens9_raw(npz: np.lib.npyio.NpzFile) -> np.ndarray:
    p = sum(ENS9_W[k] * np.asarray(npz[k], float) for k in ENS9_W)
    return p + 0.5 * np.asarray(npz["corr"], float) + 0.5 * np.asarray(npz["corr_pm"], float)


def residualize(v: np.ndarray, cols: list[np.ndarray]) -> np.ndarray:
    X = np.column_stack(cols)
    return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]


def joint_gain(y: np.ndarray, q: np.ndarray, Z: np.ndarray) -> float:
    """q에 Z의 최적 선형 가산을 했을 때 raw score 증가."""
    e = q - y
    G = (Z.T @ Z) / len(y)
    c = (Z.T @ e) / len(y)
    return float((1e5 / (y.mean() * (1 - y.mean()))) * (c @ np.linalg.pinv(G) @ c))


def partition_labels(df: pd.DataFrame) -> dict[str, np.ndarray]:
    count12 = (df["balls_before"].astype(str) + "-" + df["strikes_before"].astype(str)).to_numpy()
    month = df["game_month"].clip(upper=10)
    return {
        "count12": count12,
        "month": ("m" + month.astype(str)).to_numpy(),
        # 2023에는 3월 행이 0이라 생기는 월축 곡률 특이성을 제거하는 보수적 7셀판.
        "month_m34": ("m" + month.replace({3: 4}).astype(str)).to_numpy(),
        "inning": df["inning"].clip(upper=10).astype(str).to_numpy(),
        "hand4": ("P" + df["pitcher_hand"].astype(str) + "-B" + df["batter_hand"].astype(str)).to_numpy(),
        "game_type": df["game_type"].astype(str).to_numpy(),
    }


def partition_oracle(e: np.ndarray, labels: np.ndarray, K: float) -> dict:
    u, inv, n = np.unique(labels, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=e)
    bias = sums / n
    gain = K * float(np.sum((n / len(e)) * bias**2))
    return {
        "cells": int(len(u)),
        "gain": gain,
        "bias_max_abs": float(np.max(np.abs(bias))),
    }


def nullspace_of_row(w: np.ndarray) -> np.ndarray:
    # 열들이 w^T alpha=0인 contrast 공간의 직교기저다.
    _, _, vh = np.linalg.svd(w.reshape(1, -1), full_matrices=True)
    return vh[1:].T


def generalized_extremes(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    ev, U = np.linalg.eigh(A)
    keep = ev > 1e-14
    W = U[:, keep] @ np.diag(ev[keep] ** -0.5) @ U[:, keep].T
    z = np.linalg.eigvalsh(W @ B @ W)
    z = z[z > -1e-10]
    return float(max(0.0, z.min())), float(z.max())


def weight_stability(labels: np.ndarray, seasons: np.ndarray, ref_mask: np.ndarray | None = None) -> dict:
    cells = np.unique(labels)
    cell_to_i = {c: i for i, c in enumerate(cells)}

    def weights(mask: np.ndarray) -> np.ndarray:
        idx = np.fromiter((cell_to_i[x] for x in labels[mask]), dtype=int)
        return np.bincount(idx, minlength=len(cells)).astype(float) / mask.sum()

    if ref_mask is None:
        ref_mask = np.ones(len(labels), dtype=bool)
    w = weights(ref_mask)
    B = nullspace_of_row(w)
    H0 = B.T @ np.diag(w) @ B
    out = {"cells": int(len(cells)), "weights_reference": dict(zip(map(str, cells), map(float, w))), "seasons": {}}
    for s in sorted(np.unique(seasons)):
        v = weights(seasons == s)
        H_fixed = B.T @ np.diag(v) @ B
        H_profile = B.T @ (np.diag(v) - np.outer(v, v)) @ B
        f_lo, f_hi = generalized_extremes(H0, H_fixed)
        p_lo, p_hi = generalized_extremes(H0, H_profile)
        out["seasons"][str(int(s))] = {
            "max_abs_weight_error": float(np.max(np.abs(v - w))),
            "fixed_level_rho_min": f_lo,
            "fixed_level_rho_max": f_hi,
            "profiled_level_rho_min": p_lo,
            "profiled_level_rho_max": p_hi,
        }
    return out


def invsqrt_psd(A: np.ndarray) -> np.ndarray:
    ev, U = np.linalg.eigh(A)
    z = np.zeros_like(ev)
    z[ev > 1e-12] = ev[ev > 1e-12] ** -0.5
    return U @ np.diag(z) @ U.T


def canonical_partition_corr(a: np.ndarray, b: np.ndarray) -> float:
    ca, ia = np.unique(a, return_inverse=True)
    cb, ib = np.unique(b, return_inverse=True)
    A = np.eye(len(ca))[ia]
    B = np.eye(len(cb))[ib]
    A -= A.mean(axis=0)
    B -= B.mean(axis=0)
    Caa = A.T @ A / len(a)
    Cbb = B.T @ B / len(a)
    Cab = A.T @ B / len(a)
    M = invsqrt_psd(Caa) @ Cab @ invsqrt_psd(Cbb)
    return float(np.linalg.svd(M, compute_uv=False)[0])


def shape_fold(y: np.ndarray, raw: np.ndarray, game_type: np.ndarray) -> dict:
    n = len(y)
    K = 1e5 / (y.mean() * (1 - y.mean()))
    x = raw - raw.mean()
    gf = (game_type == "F").astype(float)
    gf -= gf.mean()
    base = np.column_stack([np.ones(n), x, gf])
    q = base @ np.linalg.lstsq(base, y, rcond=None)[0]
    e = q - y

    q2_raw = x**2
    q2 = residualize(q2_raw, [np.ones(n), x, gf])
    q3_raw = x**3
    q3_aff = residualize(q3_raw, [np.ones(n), x, gf])
    q3_unique = residualize(q3_raw, [np.ones(n), x, gf, q2])

    p = np.clip(raw, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    lh0 = residualize(p * (1 - p), [np.ones(n), x, gf])
    lh1 = residualize(p * (1 - p) * logit, [np.ones(n), x, gf])
    L = np.column_stack([lh0, lh1])
    P = np.column_stack([q2, q3_unique])

    def axis(g: np.ndarray) -> dict:
        vg = float(np.mean(g**2))
        ve = float(np.mean(e**2))
        cov = float(np.mean(e * g))
        corr = cov / np.sqrt(vg * ve)
        return {
            "mean_g2": vg,
            "curvature": K * vg,
            "corr_residual": corr,
            "oracle_gain": K * cov**2 / vg,
            "max_abs_g": float(np.max(np.abs(g))),
        }

    # logit 잔여 2차원과 polynomial 2차원의 canonical correlations.
    Ql, _ = np.linalg.qr(L)
    Qp, _ = np.linalg.qr(P)
    can = np.linalg.svd(Ql.T @ Qp, compute_uv=False)
    return {
        "base_score_affine_plus_game_type": score_raw(y, q),
        "residual_brier": float(np.mean(e**2)),
        "quadratic": axis(q2),
        "cubic_after_affine": axis(q3_aff),
        "cubic_unique_after_quadratic": axis(q3_unique),
        "logit_intercept_tangent": axis(lh0),
        "logit_slope_tangent": axis(lh1),
        "logit_joint_gain": joint_gain(y, q, L),
        "poly_joint_gain": joint_gain(y, q, P),
        "logit_vs_poly_canonical": [float(v) for v in can],
        "axis_correlations": {
            "q2_q3_aff": float(np.corrcoef(q2, q3_aff)[0, 1]),
            "q2_logit_intercept": float(np.corrcoef(q2, lh0)[0, 1]),
            "q3_unique_logit_slope": float(np.corrcoef(q3_unique, lh1)[0, 1]),
        },
    }


def blend_rows() -> list[dict]:
    """두 레그가 각각 level+slope 최적일 때 post-affine 닫힌형 표."""
    rows = []
    var_a = POST_BLEND_SLOPE_CURVATURE_2025 / C_2025
    for sb in (850.0, 900.0, 1000.0):
        for s in (0.0134, 0.020, 0.030, 0.045):
            D = sb - S_A_2025
            k = C_2025 * s * s
            w_old = float(np.clip(0.5 + D / (2 * k), 0, 1))
            gain_old = float(w_old * D + k * w_old * (1 - w_old))
            # 두 레그가 각각 affine-optimal이면 h=<x_A,d>가 D,s로 정해진다.
            h = (D / C_2025 - s * s) / 2
            s_perp2 = s * s - h * h / var_a
            gain_new = 0.0
            beta = 1.0
            w_new = 0.0
            bcoef = 0.0
            valid = s_perp2 > 0
            if valid:
                bcoef = (D + k) / (2 * C_2025 * s_perp2)
                acoef = -(h / var_a) * bcoef
                beta = 1 + acoef
                w_new = bcoef / beta if beta > 0 else np.nan
                valid = bool(bcoef >= 0 and beta > 0 and 0 <= w_new <= 1)
                if valid:
                    gain_new = float((D + k) ** 2 / (4 * C_2025 * s_perp2))
                else:
                    gain_new, w_new, beta = (max(0.0, D), 1.0 if D > 0 else 0.0, 1.0)
            rows.append({
                "S_B_affine": sb,
                "s": s,
                "breakeven_s": float(np.sqrt(max(S_A_2025 - sb, 0) / C_2025)),
                "pre_recal_w": w_old,
                "pre_recal_gain": gain_old,
                "shape_s_perp": float(np.sqrt(max(s_perp2, 0))),
                "post_recal_blend_w": float(w_new),
                "post_recal_slope_multiplier": float(beta),
                "post_recal_gain": gain_new,
                "interior_solution": valid,
            })
    return rows


def main() -> None:
    use = [
        "season", "game_month", "inning", "game_type", "balls_before", "strikes_before",
        "pitcher_hand", "batter_hand", "control_success",
    ]
    df = pd.read_csv(TRAIN, usecols=use)
    seasons = df["season"].to_numpy(int)
    y_train = df["control_success"].to_numpy(float)
    labels = partition_labels(df)

    out: dict = {
        "constants": {
            "C_2025": C_2025,
            "V_2025": V_2025,
            "S_A_2025": S_A_2025,
            "raw_slope_curvature_2025": RAW_SLOPE_CURVATURE_2025,
            "post_blend_slope_curvature_2025_inferred": POST_BLEND_SLOPE_CURVATURE_2025,
        },
        "game_type_postmortem": {},
        "partition_weight_stability": {},
        "partition_weight_stability_ref2024": {},
        "partition_cross_canonical_2024": {},
        "folds": {},
        "blend_table": blend_rows(),
    }

    b_train = C_2025 * TRAIN_W_F * (1 - TRAIN_W_F)
    b_true_fixed_center = C_2025 * (
        TRUE_W_F_2025 * (1 - TRAIN_W_F) ** 2 + (1 - TRUE_W_F_2025) * TRAIN_W_F**2
    )
    rho = TRUE_B_F_2025 / b_train
    tstar = 0.0057024
    delta = 0.010
    beta_probe = delta / (2 * tstar)
    realized_ratio = 1 - ((rho - 1) * (1 - beta_probe)) ** 2
    out["game_type_postmortem"] = {
        "b_train": b_train,
        "b_true_from_fixed_train_center": b_true_fixed_center,
        "b_true_LB": TRUE_B_F_2025,
        "rho": rho,
        "probe_beta_delta_over_2tstar": beta_probe,
        "one_sided_realized_gain_ratio": realized_ratio,
        "failure_rho_upper": 1 + 1 / abs(1 - beta_probe),
    }

    for name, lab in labels.items():
        out["partition_weight_stability"][name] = weight_stability(lab, seasons)
        out["partition_weight_stability_ref2024"][name] = weight_stability(lab, seasons, seasons == 2024)

    m24 = seasons == 2024
    pairs = [("count12", "month"), ("count12", "inning"), ("count12", "hand4"),
             ("month", "inning"), ("month", "hand4"), ("inning", "hand4"),
             ("game_type", "count12"), ("game_type", "month"),
             ("game_type", "inning"), ("game_type", "hand4")]
    for a, b in pairs:
        out["partition_cross_canonical_2024"][f"{a}__{b}"] = canonical_partition_corr(
            labels[a][m24], labels[b][m24]
        )

    for v in (2021, 2022, 2023, 2024):
        m = seasons == v
        z = np.load(ENS9 / f"preds_val{v}.npz")
        y = np.asarray(z["y"], float)
        assert len(y) == int(m.sum()) and np.array_equal(y, y_train[m])
        raw = ens9_raw(z)
        sf = shape_fold(y, raw, labels["game_type"][m])

        # 각 폴드에서 level+slope+game_type를 먼저 profile한 잔차의 분할 오라클.
        x = raw - raw.mean()
        gf = (labels["game_type"][m] == "F").astype(float)
        gf -= gf.mean()
        X = np.column_stack([np.ones(len(y)), x, gf])
        q = X @ np.linalg.lstsq(X, y, rcond=None)[0]
        e = q - y
        K = 1e5 / (y.mean() * (1 - y.mean()))
        por = {name: partition_oracle(e, lab[m], K) for name, lab in labels.items() if name != "game_type"}

        # partition-slope cross: one-hot subspace가 raw slope를 설명하는 canonical corr.
        slope_cross = {}
        for name, lab in labels.items():
            u, inv = np.unique(lab[m], return_inverse=True)
            means = np.bincount(inv, weights=x) / np.bincount(inv)
            fitted = means[inv]
            slope_cross[name] = float(np.sqrt(np.mean(fitted**2) / np.mean(x**2)))
        g = gf
        corr_gx = float(np.corrcoef(g, x)[0, 1])
        h_gx = float(K * np.mean(g * x))

        out["folds"][str(v)] = {
            "n": int(len(y)), "r": float(y.mean()), "raw_mean": float(raw.mean()),
            "shape": sf, "partition_oracle_after_profile": por,
            "partition_slope_canonical": slope_cross,
            "game_type_slope_corr": corr_gx,
            "game_type_slope_H_cross": h_gx,
            "required_abs_corr_for_gain20": float(np.sqrt(20 / (K * sf["residual_brier"]))),
        }

    # 현재 LB에서 +20에 필요한 상관. Brier*K = 1e5-S (level 최적이면 Var≈MSE).
    out["required_corr_2025"] = float(np.sqrt(20 / (1e5 - S_A_2025)))
    out["shape_probe_design"] = {
        "target_no_signal_cost": 4.0,
        "quadratic": {
            "curvature": 3.72,
            "delta": float(np.sqrt(4 / 3.72)),
            "tstar_if_gain20": float(np.sqrt(20 / 3.72)),
            "max_move_at_tstar_using_given_range": float(np.sqrt(20 / 3.72) * 0.02319),
        },
        "cubic": {
            "curvature": 0.0540,
            "delta": float(np.sqrt(4 / 0.0540)),
            "tstar_if_gain20": float(np.sqrt(20 / 0.0540)),
        },
    }

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
