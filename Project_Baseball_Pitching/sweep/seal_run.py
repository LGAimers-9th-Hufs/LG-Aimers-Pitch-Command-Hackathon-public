# -*- coding: utf-8 -*-
"""이중조사 봉인 런 — C-A 매트릭스의 마지막 미측정 셀 3개를 한 번에 닫는다.

    python sweep/seal_run.py --vals 2024,2023,2022,2021

셀: ①E1(직전 완결 시즌 투수 성적)×nn_lin ②E1×잔차 보정기 ③denom(C1 분모복원)×잔차 보정기.
전부 C-A 감사의 EV 앵커 0~+3 — 채택 기대가 아니라 **매트릭스 완전 봉인**이 목적.
게이트: ENS-7 동결·deploy_score·위약(E1=경로 교환 / denom=행 셔플) 각 1시드.
보정기 잔차 타깃 = A(june_fit_resid) — is_corrector 실측에서 A≈B(Δ<0.1) 확인됨.
(donut 스칼라×보정기는 의도적 미실행: 율 유래 물리 역산 계열은 D-16에서 -7.8 실측 +
 E1급 앵커라 EV/시간이 최하 — 미측정 셀로 기록만 남긴다.)
"""
from __future__ import annotations
import argparse
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
import inseason_full as IF        # noqa: E402
import prev_denom as PD           # noqa: E402
import tm_member as TMM           # noqa: E402
from nn_carrier import BIS, train_lin   # noqa: E402
from eb_carrier import get_members, deploy_score, ENS7_W, TM_W, entity_traj_map  # noqa: E402

OUT_DIR = HERE.parent / "results" / "dual"
LAM = 1000.0
WGRID = (0.25, 0.5)


def e1_seasonal(df, emap=None):
    """직전 완결 시즌의 시즌 단위 성공률·표본 (per-투수 상수). emap = 위약 경로 교환."""
    rc = [("asof_pitcher_success_rate", "succ")]
    lut = IF.end_lookup(df, "pitcher_id", "asof_pitcher_n", rc)
    by = {}
    for (e, s), (n_end, ks) in lut.items():
        e2 = emap.get(e, e) if emap else e
        by.setdefault(e2, {})[s] = (n_end, ks[0])
    pid = df["pitcher_id"].to_numpy().astype(int)
    ssn = df[rd.SEASON].to_numpy().astype(int)
    rate = np.full(len(df), np.nan)
    nlog = np.zeros(len(df))
    cache = {}
    for i, (e, s) in enumerate(zip(pid, ssn)):
        v = cache.get((e, s))
        if v is None:
            hist = by.get(e, {})
            prevs = sorted(q for q in hist if q < s)
            v = (np.nan, 0.0)
            if prevs:
                q = prevs[-1]
                n1, k1 = hist[q]
                n0, k0 = hist[prevs[-2]] if len(prevs) >= 2 else (0.0, 0.0)
                dn, dk = n1 - n0, k1 - k0
                if dn > 0:
                    v = (dk / dn, np.log1p(dn))
            cache[(e, s)] = v
        rate[i], nlog[i] = v
    lg = float(np.nanmean(rate))
    return np.where(np.isnan(rate), lg, rate), nlog


def ridge_corr(D, fit, val, target_fit, lam=LAM):
    A = D[fit]
    sd = A.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    As = np.concatenate([A / sd, np.ones((A.shape[0], 1))], axis=1)
    G = As.T @ As + lam * np.eye(As.shape[1])
    c = np.linalg.solve(G, As.T @ target_fit)
    Dv = np.concatenate([D[val] / sd, np.ones((int(val.sum()), 1))], axis=1)
    return Dv @ c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024,2023,2022,2021")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(">> 준비 (K_IS=100)")
    L.K_IS = 100.0
    df = rd.load_train()
    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy())
    X = L.build_features(df, base=(nb, kb)).reset_index(drop=True)
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()
    for tag_, ent_, ncol_, rc_, K_ in IF.BLOCKS:
        if tag_ == "b":
            lb = IF.end_lookup(df, ent_, ncol_, rc_)
            B_bis = IF.build_block(df, tag_, ent_, ncol_, rc_, K_, lb)[BIS].reset_index(drop=True)

    e1r, e1n = e1_seasonal(df)
    emap = entity_traj_map(df, "pitcher_id", 21)
    e1r_pl, e1n_pl = e1_seasonal(df, emap)
    B_dn = PD.build_block(df)[PD.DENOM_COLS].reset_index(drop=True)
    Dn = np.nan_to_num(B_dn.to_numpy(dtype=float), nan=0.0)
    rng = np.random.default_rng(9)
    Dn_pl = Dn[rng.permutation(len(Dn))]

    cc = (df["balls_before"].to_numpy() * 3 + df["strikes_before"].to_numpy()).astype(int)
    cn = cc / 11.0
    hand = (df["pitcher_hand"].to_numpy() == df["batter_hand"].to_numpy()).astype(float)

    def e1_design(r):
        rc_ = r - 0.5
        return np.stack([rc_, rc_ * cn, rc_ * hand], axis=1)

    res = []
    for v in [int(s) for s in args.vals.split(",")]:
        fit, val = (season == v - 1), (season == v)
        P, corr, yv = get_members(df, X, y, season, v)
        others = sum(P[m] * w for m, w in ENS7_W.items() if m != "nn_lin_bis")
        w_lin = ENS7_W["nn_lin_bis"]
        ens_full = others + w_lin * P["nn_lin_bis"] + TM_W * corr
        s_ref = deploy_score(yv, ens_full)
        residA = TMM.june_fit_resid(X, y, fit)
        print(f"\n[val {v}] ENS-7 기준 {s_ref:.2f}")

        # ① E1 × nn_lin (2컬럼 추가 스왑)
        for name, rr, nn_ in [("E1xnn_lin", e1r, e1n), ("E1xnn_lin_위약", e1r_pl, e1n_pl)]:
            E = pd.DataFrame({"e1_rate": rr.astype("float32"), "e1_nlog": nn_.astype("float32")})
            p_lin = train_lin(pd.concat([X, B_bis, E], axis=1), y, fit, val)
            d = deploy_score(yv, others + w_lin * p_lin + TM_W * corr) - s_ref
            res.append(dict(val=v, arm=name, d=round(float(d), 3)))
            print(f"  {name:16s} Δ {d:+7.2f}")

        # ②③ 보정기 셀 (w 격자)
        for name, Dm in [("E1x보정기", e1_design(e1r)), ("E1x보정기_위약", e1_design(e1r_pl)),
                         ("denomx보정기", Dn), ("denomx보정기_위약", Dn_pl)]:
            c2 = ridge_corr(Dm, fit, val, residA)
            line = dict(val=v, arm=name)
            for w in WGRID:
                line[f"w{w:g}"] = round(float(deploy_score(yv, ens_full + w * c2) - s_ref), 3)
            res.append(line)
            print(f"  {name:16s} " + "  ".join(f"w{w:g}: {line[f'w{w:g}']:+7.2f}" for w in WGRID))

    t = pd.DataFrame(res)
    (OUT_DIR / "seal_run.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR/'seal_run.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
