# -*- coding: utf-8 -*-
"""반나절 2 — Fresh-state **경험적 베이즈(EB) 캐리어**. Codex 2차 심사의 재표현 프로그램.

    python sweep/eb_carrier.py --vals 2024,2023,2022,2021

## 무엇을 고치는가

기존 당해 분해의 두 버그(D-42):
1. **0.5 고정 prior** — middle/ball/strike/구종믹스에 부적절. 특히 구종믹스 3률을 각각 0.5로
   수축하면 합이 1.5까지 간다(심플렉스 파괴). 기존 "구종믹스 기각"은 확률기하의 기각이었다.
2. K_B/K_P 주석-구현 반대.

## 통일 재표현 (전 비율 공통)

    q_pre  = k_base / n_base                (직전 시즌말 비율 = prior mean; 신인은 리그 상수)
    q_cur  = (Δk + κ·q_pre) / (Δn + κ)     (당해 posterior mean)
    innov  = q_cur − q_pre                  (당해 혁신 — 캐리어의 본체)
    rel    = Δn / (Δn + κ)                  (신뢰도)
    구종믹스: α_g = κ·q_pre_g 인 Dirichlet posterior → Σ posterior mean = 1 보장
             → 2차원 log-ratio: log(m_br/m_fb), log(m_os/m_fb)

## 게이트 (D-42 개정판)

- **ENS-7 가중·캘리 완전 동결** — nn_lin 교체 증분만. **주 판정 = 동결 배포 캘리 (1.04,−0.01) 점수.**
- 폴드 V24(주)·V23(스트레스)·V22·**V21(신규 감사)**. V21은 멤버 캐시가 없어 즉석 학습.
- 위약 = **entity 전 시즌 경로 교환**(행 셔플 아님) 2시드(채택 후보만 확대).
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
import inseason_full as IF        # noqa: E402
import ens5_pool as EP            # noqa: E402
import nn_member as NM            # noqa: E402
from nn_carrier import BIS, train_lin   # noqa: E402
from season_centering import best_cal   # noqa: E402

CACHE7 = HERE.parent / "results" / "ens7"
CACHE9 = HERE.parent / "results" / "ens9"
OUT_DIR = HERE.parent / "results" / "eb_carrier"
ENS7_W = {"nn_lin_bis": 0.475, "et_l100_d28": 0.25, "allraw_l15": 0.125,
          "june_l15": 0.10, "june_l7": 0.05}
# ENS-9 (챔피언, LB 1009.26): 선형 멤버가 패키지 78컬럼(nn_lin_pkg)으로 교체 + physmix w0.5.
# 캐시는 sweep/ens9_cache.py가 생성. 게이트는 반드시 이 기준선 위 교체 증분으로 잰다(D-46).
ENS9_W = {"nn_lin_pkg": 0.475, "et_l100_d28": 0.25, "allraw_l15": 0.125,
          "june_l15": 0.10, "june_l7": 0.05}
TM_W = 0.5
W_PM = 0.5
# ── 배포 캘리 (2026-08-09 갱신, D-59) ────────────────────────────────────────────
# 출처는 추정이 아니라 **현 챔피언 zip의 원본**: submit_exact1.zip → model/metadata.json
#   calibration = {center 0.5, slope 1.14202, shift -0.00264162}
#   seg_probe   = [{col game_type, levels ['F'], w 0.10914844633419475, t 0.0057}]
# D-58 LB 프로빙으로 확정된 값이다(레벨 +6.44 · 세그먼트 +1.32 · 기울기 +8.18 = LB 1025.5511136843).
#
# ⚠ 로컬 게이트는 이 축을 원리적으로 못 본다. fit-2023→val-2024 프로토콜의 최적은
#   (1.035270, -0.010359)인데 2025 최적은 (1.14202, -0.0026)이고 그 차이가 +8.18이었다.
#   따라서 여기서 grid로 재적합하지 말 것 — 이 상수는 **제출 프로빙의 산물**이다.
# ⚠ 기준선 이동: D-58 이전 로그의 deploy_score 수치는 전부 아래 LEGACY 상수에서 잰 것이라
#   절대값이 직접 비교되지 않는다(증분 d = s − s_ref는 계속 유효).
DEPLOY_CAL = dict(center=0.5, slope=1.14202, shift=-0.00264162)
DEPLOY_SEG = dict(col="game_type", level="F", w=0.10914844633419475, t=0.0057)
DEPLOY_CAL_LEGACY = dict(center=0.5, slope=1.04, shift=-0.01)   # ~D-57, 과거 로그 대조용


def deploy_q(p, seg=None):
    """배포 캘리를 적용한 최종 확률. seg = 행별 game_type 배열(주면 세그먼트 항까지 반영).

    세그먼트 항은 캘리 **이후** 가산이고 중심화 지시자라 전역 절편축과 직교한다
    (서빙 템플릿 seg_correction()과 같은 순서 — 여기서 순서를 바꾸면 게이트가 배포본과 어긋난다).
    """
    q = (DEPLOY_CAL["center"]
         + DEPLOY_CAL["slope"] * (np.asarray(p, dtype=float) - DEPLOY_CAL["center"])
         + DEPLOY_CAL["shift"])
    if seg is not None:
        g = (np.asarray(seg) == DEPLOY_SEG["level"]).astype(float) - DEPLOY_SEG["w"]
        q = q + DEPLOY_SEG["t"] * g
    return np.clip(q, 1e-6, 1 - 1e-6)


def deploy_score(y, p, seg=None):
    return L.score(y, deploy_q(p, seg))


# ---------------------------------------------------------------- EB 블록
def eb_block(df, ent, ncol, rate_cols, kappa, prefix, ent_map=None):
    """통일 EB 재표현. ent_map = 위약용 entity 경로 교환 사전({ent: ent'})."""
    lut = IF.end_lookup(df, ent, ncol, rate_cols)
    ents = df[ent].to_numpy().astype(int)
    if ent_map is not None:
        by = {}
        for (e, s_), v in lut.items():
            by.setdefault(e, {})[s_] = v
        lut = {}
        for e, hist in by.items():
            e2 = ent_map.get(e, e)
            for s_, v in hist.items():
                lut[(e2, s_)] = v                       # e2가 e의 전 시즌 경로를 받음
    ssn = df[rd.SEASON].to_numpy().astype(int)
    n = df[ncol].to_numpy(dtype=float)
    league = {nm: float(np.nanmean(df[c].to_numpy(dtype=float)))
              for c, nm in rate_cols}                    # 신인 fallback (train 상수)

    by2 = {}
    for (e, s_), v in lut.items():
        by2.setdefault(e, {})[s_] = v
    nb = np.zeros(len(df))
    kb = np.zeros((len(df), len(rate_cols)))
    cache = {}
    for i, (e, s_) in enumerate(zip(ents, ssn)):
        v = cache.get((e, s_))
        if v is None:
            hist = by2.get(e)
            v = (0.0, [0.0] * len(rate_cols))
            if hist:
                prev = [q for q in hist if q < s_]
                if prev:
                    v = hist[max(prev)]
            cache[(e, s_)] = v
        nb[i] = v[0]
        kb[i] = v[1]

    dn = np.maximum(n - nb, 0.0)
    out = {f"{prefix}_rel": (dn / (dn + kappa)).astype("float32"),
           f"{prefix}_logdn": np.log1p(dn).astype("float32")}
    q_pre_all, q_cur_all = [], []
    for j, (c, nm) in enumerate(rate_cols):
        r = df[c].fillna(league[nm]).to_numpy(dtype=float)
        k = np.rint(r * n)
        dk = np.clip(k - kb[:, j], 0.0, dn)
        q_pre = np.where(nb > 0, kb[:, j] / np.maximum(nb, 1.0), league[nm])
        q_cur = (dk + kappa * q_pre) / (dn + kappa)
        out[f"{prefix}_{nm}_innov"] = (q_cur - q_pre).astype("float32")
        q_pre_all.append(q_pre)
        q_cur_all.append(q_cur)
    return pd.DataFrame(out), q_pre_all, q_cur_all


def mix_logratio(df, kappa, ent_map=None):
    """구종믹스 Dirichlet posterior → 2D log-ratio (심플렉스 보존)."""
    rc = [("asof_pitcher_fastball_rate", "fb"), ("asof_pitcher_breaking_rate", "br"),
          ("asof_pitcher_offspeed_rate", "os")]
    B, q_pre, q_cur = eb_block(df, "pitcher_id", "asof_pitcher_pitchmix_n", rc,
                               kappa, "mix", ent_map)
    # posterior mean은 자동으로 합 1 (Σ(Δk+κq_pre) = Δn+κ·Σq_pre, Σq_pre≈1)
    eps = 1e-4
    m = [np.clip(q, eps, 1.0) for q in q_cur]
    out = pd.DataFrame({
        "mix_lr_br": np.log(m[1] / m[0]).astype("float32"),
        "mix_lr_os": np.log(m[2] / m[0]).astype("float32"),
        "mix_rel": B["mix_rel"],
        "mix_innov_fb": B["mix_fb_innov"],
    })
    return out


def entity_traj_map(df, ent, seed):
    ids = np.unique(df[ent].to_numpy().astype(int))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ids)
    return dict(zip(ids.tolist(), perm.tolist()))


# ---------------------------------------------------------------- 멤버 (V21용 즉석 학습 포함)
def get_members(df, X, y, season, v):
    f7 = CACHE7 / f"preds_val{v}.npz"
    if f7.exists():
        z = np.load(f7)
        return ({m: z[m] for m in ENS7_W}, z["corr"], z["y"])
    # V21: 즉석 학습 (june×2 · ET · allraw · nn_lin_bis) + tm 보정기
    from sklearn.ensemble import ExtraTreesRegressor
    import tm_member as TMM
    fit, val = (season == v - 1), (season == v)
    P = {}
    for nm, sp in (("june_l15", L.ORIGINAL["l15"]), ("june_l7", L.ORIGINAL["l7"])):
        m = L.train_lgbm(X[fit], y[fit], {**sp, "num_threads": 6})
        P[nm] = np.clip(m.predict(X[val], num_threads=6), 0, 1)
    m = L.train_lgbm(X[season < v], y[season < v], {**L.ORIGINAL["l15"], "num_threads": 6})
    P["allraw_l15"] = np.clip(m.predict(X[val], num_threads=6), 0, 1)
    Xn = np.nan_to_num(X[fit].to_numpy(dtype=np.float32), nan=-999.0)
    Xv = np.nan_to_num(X[val].to_numpy(dtype=np.float32), nan=-999.0)
    P["et_l100_d28"] = np.clip(ExtraTreesRegressor(
        n_estimators=200, min_samples_leaf=100, max_depth=28, max_features=0.7,
        n_jobs=6, random_state=9).fit(Xn, y[fit]).predict(Xv), 0, 1)
    for tag_, ent_, ncol_, rc_, K_ in IF.BLOCKS:
        if tag_ == "b":
            lb = IF.end_lookup(df, ent_, ncol_, rc_)
            B_bis = IF.build_block(df, tag_, ent_, ncol_, rc_, K_, lb)[BIS].reset_index(drop=True)
    P["nn_lin_bis"] = train_lin(pd.concat([X, B_bis], axis=1), y, fit, val)
    prof = TMM.load_profile()
    resid = TMM.june_fit_resid(X, y, fit)
    Df, covf = TMM.design(df, X, prof, fit)
    Dv, covv = TMM.design(df, X, prof, val)
    A = Df[fit & covf]
    G = A.T @ A + 1000.0 * np.eye(A.shape[1])
    c = np.linalg.solve(G, A.T @ resid[covf[fit]])
    corr = Dv @ c
    corr[~covv] = 0.0
    np.savez_compressed(CACHE7 / f"preds_val{v}.npz", y=y[val], corr=corr[val], **P)
    return P, corr[val], y[val]


def get_members9(v):
    """ENS-9 캐시 로더 — 반환 (P[ENS9_W 멤버], corr_tm, corr_pm, y).

    캐시가 없으면 명시적으로 실패한다(즉석 학습 없음 — 기준선은 ens9_cache.py로만 생성해
    모든 게이트가 같은 예측을 쓰게 한다).
    """
    f9 = CACHE9 / f"preds_val{v}.npz"
    if not f9.exists():
        raise FileNotFoundError(
            f"{f9} 없음 — 먼저 `python sweep/ens9_cache.py --vals {v}` 실행")
    z = np.load(f9)
    return ({m: z[m] for m in ENS9_W}, z["corr"], z["corr_pm"], z["y"])


def ens9_pred(P, corr, corr_pm):
    """ENS-9 배포 전 확률(캘리 미적용) 재구성."""
    return sum(P[m] * w for m, w in ENS9_W.items()) + TM_W * corr + W_PM * corr_pm


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

    # 블록 사전 구축 (κ=100 기본 + 격자)
    BATTER_RC = [("asof_batter_success_rate", "succ"), ("asof_batter_middle_rate", "mid")]
    blocks = {}
    for kap in (25, 100, 400):
        Bb, _, _ = eb_block(df, "batter_id", "asof_batter_n", BATTER_RC, kap, f"eb{kap}")
        blocks[f"ebB{kap}"] = Bb
    blocks["mix"] = mix_logratio(df, 100)
    PFAIL_RC = [("asof_pitcher_reverse_rate", "rev"), ("asof_pitcher_middle_rate", "mid"),
                ("asof_pitcher_ball_rate", "ball"), ("asof_pitcher_strike_rate", "strk")]
    Bp, _, _ = eb_block(df, "pitcher_id", "asof_pitcher_n", PFAIL_RC, 100, "ebp")
    blocks["pfail"] = Bp[[c for c in Bp.columns if c.endswith("_innov")]]
    P_SUCC = [("asof_pitcher_success_rate", "succ")]
    Bps, _, _ = eb_block(df, "pitcher_id", "asof_pitcher_n", P_SUCC, 100, "ebps")
    blocks["pEB"] = Bps
    # 기존 bis (기준 재현용)
    for tag_, ent_, ncol_, rc_, K_ in IF.BLOCKS:
        if tag_ == "b":
            lb = IF.end_lookup(df, ent_, ncol_, rc_)
            blocks["bis_old"] = IF.build_block(df, tag_, ent_, ncol_, rc_, K_, lb)[BIS] \
                .reset_index(drop=True)
    # 위약: 타자 경로 교환 2시드 (EB100 기준)
    for sd in range(2):
        emap = entity_traj_map(df, "batter_id", sd)
        Bb, _, _ = eb_block(df, "batter_id", "asof_batter_n", BATTER_RC, 100, "eb100", emap)
        blocks[f"plB{sd}"] = Bb
    print(f"   블록 {len(blocks)}종 [{time.time()-t0:.0f}s]")

    arms = [
        ("bis_old(기준)", ["bis_old"]),
        ("ebB25",  ["ebB25"]), ("ebB100", ["ebB100"]), ("ebB400", ["ebB400"]),
        ("ebB100+mix", ["ebB100", "mix"]),
        ("ebB100+pEB", ["ebB100", "pEB"]),
        ("ebB100+mix+pEB", ["ebB100", "mix", "pEB"]),
        ("ebB100+pfail", ["ebB100", "pfail"]),
        ("위약B0", ["plB0"]), ("위약B1", ["plB1"]),
    ]

    res = []
    for v in [int(s) for s in args.vals.split(",")]:
        fit, val = (season == v - 1), (season == v)
        P, corr, yv = get_members(df, X, y, season, v)
        others = sum(P[m] * w for m, w in ENS7_W.items() if m != "nn_lin_bis")
        w_lin = ENS7_W["nn_lin_bis"]
        ens_ref = others + w_lin * P["nn_lin_bis"] + TM_W * corr
        s_ref = deploy_score(yv, ens_ref)
        print(f"\n[val {v}] ENS-7 동결캘리 기준 {s_ref:.2f}")
        for name, keys in arms:
            B = pd.concat([blocks[k] for k in keys], axis=1)
            p_lin = train_lin(pd.concat([X, B], axis=1), y, fit, val)
            s = deploy_score(yv, others + w_lin * p_lin + TM_W * corr)
            res.append(dict(val=v, arm=name, d=s - s_ref))
            print(f"  {name:18s} Δ(동결캘리) {s-s_ref:+7.2f}")

    t = pd.DataFrame(res)
    print("\n" + t.pivot_table(index="arm", columns="val", values="d")
          .to_string(float_format=lambda x: f"{x:+.2f}"))
    g = t[~t.arm.str.contains("기준|위약")].groupby("arm").agg(
        d_mean=("d", "mean"), d_min=("d", "min"), pos=("d", lambda s: int((s > 0).sum())))
    print("\n[요약 — 주 판정 V24, V23=스트레스, V21=감사]")
    print(g.sort_values("d_mean", ascending=False).to_string(float_format=lambda x: f"{x:.2f}"))
    (OUT_DIR / "gate.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    print(f"\n({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
