#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Codex 파트너 CSV가 도착한 만큼 읽어 affine 재캘리 후 블렌드 기하를 측정한다.

출력은 results/codex_research/partner_analysis.json 하나뿐이다. 파트너 디렉터리는
읽기만 하며 수정하지 않는다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "partner_analysis.json"
PARTNER = ROOT / "results" / "codex_partner"
TRAIN = ROOT / "data" / "data" / "train.csv"
ENS9 = ROOT / "results" / "ens9"
W = {"nn_lin_pkg": .475, "et_l100_d28": .25, "allraw_l15": .125, "june_l15": .1, "june_l7": .05}
C25 = 402_463.285680
SA25 = 1025.5511136843
HAA25 = 952.2818676068


def raw_a(z):
    return sum(W[k] * np.asarray(z[k], float) for k in W) + .5*z["corr"] + .5*z["corr_pm"]


def affine(y, p):
    X = np.column_stack([np.ones(len(p)), p])
    return X @ np.linalg.lstsq(X, y, rcond=None)[0]


def score(y, p):
    r = y.mean()
    return float(1e5 * (1 - np.mean((p-y)**2)/(r*(1-r))))


def geometry(y, A, B):
    K = 1e5 / (y.mean() * (1-y.mean()))
    Ac, Bc = affine(y, A), affine(y, B)
    sa, sb = score(y, Ac), score(y, Bc)
    d = Bc - Ac
    s = float(np.sqrt(np.mean(d*d)))
    D = sb-sa
    w = float(np.clip(.5 + D/(2*K*s*s), 0, 1))
    pre_gain = score(y, Ac+w*d)-sa
    xa = Ac-Ac.mean()
    h = float(np.mean(xa*d))
    sperp2 = float(np.mean(d*d) - h*h/np.mean(xa*xa))
    L = D + K*s*s
    bcoef = L/(2*K*sperp2)
    acoef = -(h/np.mean(xa*xa))*bcoef
    beta = 1+acoef
    wpost = bcoef/beta
    X = np.column_stack([np.ones(len(y)), Ac, Bc])
    joint = X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return {
        "S_A_aff": sa, "S_B_raw": score(y, B), "S_B_aff": sb,
        "mean_A_aff": float(Ac.mean()), "mean_B_raw": float(B.mean()), "mean_B_aff": float(Bc.mean()),
        "s_aff": s, "s_perp": float(np.sqrt(sperp2)),
        "pre_recal_w": w, "pre_recal_gain": float(pre_gain),
        "post_recal_w": float(wpost), "post_recal_slope": float(beta),
        "post_recal_gain": float(score(y, joint)-sa),
        "identity_formula_post_gain": float(L*L/(4*K*sperp2)),
    }


def projected_2025(local):
    sb = .7541*local["S_B_aff"] + 320.9
    s = local["s_aff"]
    D = sb-SA25
    k = C25*s*s
    oldw = float(np.clip(.5+D/(2*k), 0, 1))
    oldg = float(oldw*D+k*oldw*(1-oldw))
    va = HAA25/C25
    h = (D/C25-s*s)/2
    sp2 = s*s-h*h/va
    if sp2 <= 0 or D+k <= 0:
        return {"S_B_aff_projected": sb, "pre_gain": oldg, "post_gain": 0.0, "post_w": 0.0}
    b = (D+k)/(2*C25*sp2)
    a = -(h/va)*b
    beta = 1+a
    w = b/beta
    valid = beta > 0 and 0 <= w <= 1
    return {
        "S_B_aff_projected": sb, "pre_gain": oldg,
        "post_gain": float((D+k)**2/(4*C25*sp2)) if valid else max(0.0, D),
        "post_w": float(w) if valid else (1.0 if D > 0 else 0.0),
        "post_slope": float(beta) if valid else 1.0,
        "s_perp_projected": float(np.sqrt(max(sp2, 0))),
        "breakeven_s": float(np.sqrt(max(SA25-sb, 0)/C25)),
    }


def choose_file(v):
    p = PARTNER / f"team_pred_{v}.csv"
    if p.exists():
        return p
    if v == 2024:
        for q in (PARTNER/"exp_mech"/"team_pred_2024.csv", PARTNER/"r1"/"team_pred_2024.csv"):
            if q.exists():
                return q
    return None


def main():
    df = pd.read_csv(TRAIN, usecols=["season", "row_id", "control_success"])
    out = {"folds": {}, "note": "top-level partner CSV preferred; while unfinished, V2024 exp_mech fallback is used"}
    for v in (2021, 2022, 2023, 2024):
        fn = choose_file(v)
        if fn is None:
            continue
        m = df.season.to_numpy() == v
        ids = df.loc[m, "row_id"].to_numpy()
        y = df.loc[m, "control_success"].to_numpy(float)
        tb = pd.read_csv(fn)
        assert np.array_equal(ids, tb.row_id.to_numpy())
        z = np.load(ENS9/f"preds_val{v}.npz")
        assert np.array_equal(y, z["y"])
        g = geometry(y, raw_a(z), tb.pred.to_numpy(float))
        g["file"] = str(fn.relative_to(ROOT)).replace("\\", "/")
        g["projected_2025"] = projected_2025(g)
        out["folds"][str(v)] = g
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}; folds={list(out['folds'])}")


if __name__ == "__main__":
    main()
