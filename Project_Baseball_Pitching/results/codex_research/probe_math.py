#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LB 점수 도착 뒤 손계산 없이 1축 또는 5슬롯 블렌드 대수를 푼다.

1축(해석 곡률을 둔 one-sided probe):
  python results/codex_research/probe_math.py axis \
    --s0 1025.5511136843 --sprobe SCORE --delta 1.0369517 --curvature 3.72

블렌드 4 probe + 꼭짓점 1회의 앞 4점 역산:
  P1: A+t1(B-A)
  P2: A+t2(B-A)
  P0: P1+u (상수 shift)
  PS: P1+v*x_A (x_A = 현재 최종확률 A의 평균보존 slope 방향)

  python results/codex_research/probe_math.py blend5 \
    --s0 1025.5511136843 --p1 SCORE --p2 SCORE --pshift SCORE --pslope SCORE \
    --t1 .5 --t2 1 --shift .0025 --slope .06
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def solve_axis(a: argparse.Namespace) -> dict:
    dscore = a.sprobe - a.s0
    linear = (dscore + a.curvature * a.delta**2) / a.delta
    tstar = linear / (2 * a.curvature)
    gain = linear**2 / (4 * a.curvature)
    return {
        "delta_score": dscore,
        "linear_a_hat": linear,
        "t_star_assumed_curvature": tstar,
        "gain_assumed_curvature": gain,
        "predicted_vertex_score": a.s0 + gain,
        "stop": bool(a.curvature <= 0 or not np.isfinite(gain)),
    }


def solve_blend5(a: argparse.Namespace) -> dict:
    # S(t)-S0 = ell*t - h*t^2
    M = np.array([[a.t1, -a.t1**2], [a.t2, -a.t2**2]], float)
    ell, hdd = np.linalg.solve(M, np.array([a.p1 - a.s0, a.p2 - a.s0]))
    h0d = -(a.pshift - a.p1 + a.c0 * a.shift**2) / (2 * a.shift * a.t1)
    hsd = -(a.pslope - a.p1 + a.cs * a.slope**2) / (2 * a.slope * a.t1)
    H = np.array([[a.c0, 0.0, h0d], [0.0, a.cs, hsd], [h0d, hsd, hdd]])
    eig = np.linalg.eigvalsh(H)
    linear = np.array([0.0, 0.0, ell])
    theta = 0.5 * np.linalg.solve(H, linear) if eig.min() > 0 else np.full(3, np.nan)
    gain = float(linear @ theta - theta @ H @ theta) if eig.min() > 0 else float("nan")
    return {
        "unknowns": {"ell_d": float(ell), "H_dd": float(hdd), "H_level_d": float(h0d), "H_slope_d": float(hsd)},
        "H": H.tolist(),
        "H_eigenvalues": eig.tolist(),
        "theta_star_additive_level_slope_blend": theta.tolist(),
        "gain": gain,
        "predicted_vertex_score": a.s0 + gain,
        "guards": {
            "positive_definite": bool(eig.min() > 0),
            "finite": bool(np.isfinite(gain)),
            "no_clipping_required": True,
            "basis_contract": "q=A+theta0+theta1*x_A+theta2*(B-A); x_A must match prior slope probe",
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    x = sub.add_parser("axis")
    x.add_argument("--s0", type=float, required=True)
    x.add_argument("--sprobe", type=float, required=True)
    x.add_argument("--delta", type=float, required=True)
    x.add_argument("--curvature", type=float, required=True)

    b = sub.add_parser("blend5")
    for name in ("s0", "p1", "p2", "pshift", "pslope"):
        b.add_argument(f"--{name}", type=float, required=True)
    b.add_argument("--t1", type=float, default=0.5)
    b.add_argument("--t2", type=float, default=1.0)
    b.add_argument("--shift", type=float, default=0.0025)
    b.add_argument("--slope", type=float, default=0.06)
    b.add_argument("--c0", type=float, default=402463.285680)
    # 최종확률 A에 post-blend slope를 거는 방향의 추론 곡률. underlying raw 방향이면
    # 반드시 --cs 786.2를 명시해야 한다(두 basis를 섞으면 대수가 틀린다).
    b.add_argument("--cs", type=float, default=952.2818676068)
    a = p.parse_args()
    ans = solve_axis(a) if a.mode == "axis" else solve_blend5(a)
    print(json.dumps(ans, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
