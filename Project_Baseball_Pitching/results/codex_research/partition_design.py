#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train 비중으로 partition의 H-직교 one-sided LB probe 설계를 만든다.

예:
  python results/codex_research/partition_design.py --axis month_m34 --cost 1
  python results/codex_research/partition_design.py --axis month_m34 --cost 1 \
    --s0 1025.5511136843 --scores S1,S2,S3,S4,S5,S6

생성 JSON은 이 디렉터리 안에만 기록된다. `scores`가 없으면 각 probe가 셀마다
더해야 할 확률 offset을, 있으면 해석 H를 가정한 꼭짓점 offset까지 출력한다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRAIN = ROOT / "data" / "data" / "train.csv"
C = 402_463.285680


def labels(df, axis):
    if axis == "count12":
        return (df.balls_before.astype(str)+"-"+df.strikes_before.astype(str)).to_numpy()
    if axis in ("month", "month_m34"):
        x = df.game_month.clip(upper=10)
        if axis == "month_m34":
            x = x.replace({3: 4})
        return ("m"+x.astype(str)).to_numpy()
    if axis == "inning":
        return df.inning.clip(upper=10).astype(str).to_numpy()
    if axis == "hand4":
        return ("P"+df.pitcher_hand.astype(str)+"-B"+df.batter_hand.astype(str)).to_numpy()
    raise ValueError(axis)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--axis", choices=["count12", "month", "month_m34", "inning", "hand4"], required=True)
    p.add_argument("--reference", choices=["all", "2024"], default="all")
    p.add_argument("--cost", type=float, default=1.0, help="방향당 무신호 점수 비용")
    p.add_argument("--s0", type=float)
    p.add_argument("--scores", help="basis 열 순서 one-sided LB 점수의 쉼표 목록")
    a = p.parse_args()
    use = ["season", "balls_before", "strikes_before", "game_month", "inning", "pitcher_hand", "batter_hand"]
    df = pd.read_csv(TRAIN, usecols=use)
    lab = labels(df, a.axis)
    cells, inv = np.unique(lab, return_inverse=True)
    mask = np.ones(len(df), bool) if a.reference == "all" else df.season.to_numpy() == 2024
    n = np.bincount(inv[mask], minlength=len(cells)).astype(float)
    w = n/n.sum()

    # w^T alpha=0 contrast 기저를 H=diag(w)로 whitening한다.
    _, _, vh = np.linalg.svd(w.reshape(1, -1), full_matrices=True)
    B = vh[1:].T
    G = B.T @ np.diag(w) @ B
    ev, U = np.linalg.eigh(G)
    Z = B @ U @ np.diag(ev**-0.5) @ U.T
    eta = float(np.sqrt(a.cost/C))
    offsets = eta*Z
    out = {
        "axis": a.axis, "reference": a.reference, "cells": list(map(str, cells)),
        "weights": dict(zip(map(str, cells), map(float, w))),
        "dimension": int(len(cells)-1), "cost_per_probe": a.cost, "eta": eta,
        "cell_offsets_per_probe": {
            f"d{j+1}": dict(zip(map(str, cells), map(float, offsets[:, j])))
            for j in range(offsets.shape[1])
        },
        "checks": {
            "max_abs_weighted_mean": float(np.max(np.abs(w@offsets))),
            "max_abs_gram_error": float(np.max(np.abs(Z.T@np.diag(w)@Z-np.eye(Z.shape[1])))),
            "max_abs_probe_move": float(np.max(np.abs(offsets))),
        },
    }
    if a.scores:
        if a.s0 is None:
            raise SystemExit("--scores에는 --s0가 필요")
        scores = np.array([float(x) for x in a.scores.split(",")])
        if len(scores) != Z.shape[1]:
            raise SystemExit(f"score {len(scores)}개, 필요 {Z.shape[1]}개")
        linear = (scores-a.s0+a.cost)/eta
        tstar = linear/(2*C)
        cell_star = Z@tstar
        gain = float(np.sum(linear**2)/(4*C))
        out["solution_assumed_H"] = {
            "linear": linear.tolist(), "basis_coefficients": tstar.tolist(),
            "cell_offsets": dict(zip(map(str, cells), map(float, cell_star))),
            "gain": gain, "predicted_vertex_score": a.s0+gain,
        }
    fn = HERE/f"partition_design_{a.axis}.json"
    fn.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {fn}")


if __name__ == "__main__":
    main()
