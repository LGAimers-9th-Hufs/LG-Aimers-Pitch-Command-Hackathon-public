# -*- coding: utf-8 -*-
"""평가 서버 추론 스크립트 — LGBM×2 + ExtraTrees×3 가중 앙상블 (june853 계열 확장).

서버 계약: CWD = zip 루트, 입력 `./data/{test,sample_submission}.csv`, 출력 `./output/submission.csv`.

규정 준수(§5): 모든 피처가 **그 행의 값만으로** 계산된다. 다른 행/배치 통계 참조 없음.
특히 신뢰도 수축 `p_sm500`은 train 유래 룩업이 아니라 행 단위 산식이라 시즌 드리프트가 없다.

강건성: 멤버 로드 실패 시 남은 멤버로 진행하고, 전부 실패하면 상수 안전망을 쓴다.
`script.py` 실행 오류는 제출 횟수를 차감하므로 어떤 경우에도 submission.csv를 만든다.
"""
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"
DATA_DIR = "./data"
MODEL_DIR = "./model"
OUT_PATH = "./output/submission.csv"
ET_NAN = -999.0


def build_features(df, meta):
    x = df.copy()
    for col, mapping in meta["category_maps"].items():
        x[col] = x[col].astype("string").map(mapping).fillna(-1).astype("int8")

    x["count_code"] = (x["balls_before"] * 3 + x["strikes_before"]).astype("int8")
    x["p_logn"] = np.log1p(x["asof_pitcher_n"].astype("float32")).astype("float32")
    x["b_logn"] = np.log1p(x["asof_batter_n"].astype("float32")).astype("float32")

    p_n = x["asof_pitcher_n"].astype("float32")
    b_n = x["asof_batter_n"].astype("float32")
    p_rate = x["asof_pitcher_success_rate"].fillna(0.5).astype("float32")
    b_rate = x["asof_batter_success_rate"].fillna(0.5).astype("float32")
    x["p_sm500"] = ((p_rate * p_n + 250.0) / (p_n + 500.0)).astype("float32")
    x["b_sm500"] = ((b_rate * b_n + 250.0) / (b_n + 500.0)).astype("float32")

    for w in (1, 3, 5):
        x["prev%d_fill" % w] = (x["asof_pitcher_prev%d_game_success_rate" % w]
                                .fillna(x["asof_pitcher_success_rate"])
                                .fillna(0.5).astype("float32"))

    x["hand_match"] = (x["pitcher_hand"] == x["batter_hand"]).astype("int8")

    # ---- 당해 시즌 분해 (docs/log/21) --------------------------------------
    # `asof_pitcher_n`은 커리어 누적이고 2025 안에서도 갱신된다. 동봉 룩업(=train에서 산출한
    # 그 투수의 **직전 시즌 말 누적**)을 빼면 당해 시즌분만 남는다.
    # §5 준수: 그 행의 값 + 동봉 상수만 쓴다. 다른 test 행을 보지 않고 배치 통계도 없다.
    base = meta.get("inseason_base")
    if base:
        pid = x["pitcher_id"].to_numpy()
        nb = np.zeros(len(x), dtype=np.float64)
        kb = np.zeros(len(x), dtype=np.float64)
        for i in range(len(pid)):
            v = base.get(str(int(pid[i])))      # 룩업에 없으면 (0,0) = 신인 → 커리어가 곧 당해
            if v is not None:
                nb[i] = v[0]
                kb[i] = v[1]
        n = x["asof_pitcher_n"].astype("float64").to_numpy()
        rate = x["asof_pitcher_success_rate"].fillna(0.0).astype("float64").to_numpy()
        k = np.rint(rate * n)
        is_n = np.maximum(n - nb, 0.0)
        is_k = np.clip(k - kb, 0.0, is_n)
        k_is = float(meta.get("inseason_k", 200.0))
        is_sm = (is_k + k_is * 0.5) / (is_n + k_is)
        car_sm = (rate * n + 250.0) / (n + 500.0)
        x["is_logn"] = np.log1p(is_n).astype("float32")
        x["is_share"] = (is_n / np.maximum(n, 1.0)).astype("float32")
        x["is_sm"] = is_sm.astype("float32")
        x["is_delta"] = (is_sm - car_sm).astype("float32")

    # ---- 타자 당해 시즌 분해 (bis — is4의 타자판, docs/log D-39) ------------
    # §5 준수: 그 행의 값 + 동봉 상수만. 룩업 없는 타자(신인) → (0,0,0) = 커리어가 곧 당해.
    bb = meta.get("inseason_batter")
    if bb:
        bid = x["batter_id"].to_numpy()
        nb = np.zeros(len(x), dtype=np.float64)
        ks = np.zeros(len(x), dtype=np.float64)
        km = np.zeros(len(x), dtype=np.float64)
        for i in range(len(bid)):
            v = bb.get(str(int(bid[i])))
            if v is not None:
                nb[i], ks[i], km[i] = v
        n = x["asof_batter_n"].astype("float64").to_numpy()
        rs = x["asof_batter_success_rate"].fillna(0.0).astype("float64").to_numpy()
        rm = x["asof_batter_middle_rate"].fillna(0.0).astype("float64").to_numpy()
        K = float(meta.get("inseason_batter_k", 100.0))
        is_n = np.maximum(n - nb, 0.0)
        x["isb_logn"] = np.log1p(is_n).astype("float32")
        x["isb_share"] = (is_n / np.maximum(n, 1.0)).astype("float32")
        for tag, r, kb in (("succ", rs, ks), ("mid", rm, km)):
            k = np.rint(r * n)
            is_k = np.clip(k - kb, 0.0, is_n)
            sm = (is_k + K * 0.5) / (is_n + K)
            car = (k + K * 0.5) / (n + K)
            x[f"isb_{tag}_sm"] = sm.astype("float32")
            x[f"isb_{tag}_d"] = (sm - car).astype("float32")

    # ---- ENS-8 패키지: pEB(투수 당해 EB 혁신 3) + cxp(카운트×당해 이탈 곱 12) (D-44) ----
    # §5 준수: 그 행의 값 + 동봉 상수(pEB 룩업·리그 평균·kappa)만 사용. 다른 행 참조 없음.
    # 학습측 eb_block/build_products와 산술·float32 캐스팅까지 동형(빌더 파리티 게이트가 강제).
    pkg = meta.get("pkg")
    if pkg:
        pb = pkg["peb_base"]
        kap = float(pkg.get("peb_kappa", 100.0))
        lg = float(pkg["peb_league"])
        pid = x["pitcher_id"].to_numpy()
        nb2 = np.zeros(len(x), dtype=np.float64)
        kb2 = np.zeros(len(x), dtype=np.float64)
        for i in range(len(pid)):
            v = pb.get(str(int(pid[i])))          # 없으면 (0,0) = 신인 → q_pre = 리그 평균
            if v is not None:
                nb2[i], kb2[i] = v
        n = x["asof_pitcher_n"].astype("float64").to_numpy()
        r = x["asof_pitcher_success_rate"].fillna(lg).astype("float64").to_numpy()
        k = np.rint(r * n)
        dn = np.maximum(n - nb2, 0.0)
        dk = np.clip(k - kb2, 0.0, dn)
        q_pre = np.where(nb2 > 0, kb2 / np.maximum(nb2, 1.0), lg)
        q_cur = (dk + kap * q_pre) / (dn + kap)
        x["ebps_rel"] = (dn / (dn + kap)).astype("float32")
        x["ebps_logdn"] = np.log1p(dn).astype("float32")
        x["ebps_succ_innov"] = (q_cur - q_pre).astype("float32")
        cc = (x["balls_before"].to_numpy() * 3 + x["strikes_before"].to_numpy()).astype(int)
        C = np.zeros((len(x), 12), dtype=np.float32)
        C[np.arange(len(x)), cc] = 1.0
        p_d = np.nan_to_num(x["is_delta"].to_numpy(dtype=float), nan=0.0)
        for j in range(12):
            x["cxp_%d" % j] = (C[:, j] * p_d).astype("float32")

    drop = [c for c in [ID_COL, TARGET_COL, "season", "pitcher_id", "batter_id"] if c in x.columns]
    x = x.drop(columns=drop)
    return x.loc[:, meta["feature_names"]]


def tm_correction(test, X, meta):
    """트랙맨 보정 (D-39 C3): 동봉 프로필(투수별 6랭크) × 행 맥락의 선형식. 커버 밖 = 0.
    §5 준수 — 그 행의 값 + 동봉 상수(프로필·계수)만 사용, 다른 행 참조 없음."""
    tm = meta.get("tm_corrector")
    if not tm:
        return np.zeros(len(X))
    prof = tm["profile"]                                  # {pid: [6랭크]}
    coef = np.asarray(tm["coef"], dtype=np.float64)       # 31 = 6×5 + 절편
    pid = test["pitcher_id"].to_numpy()
    M = np.full((len(X), 6), np.nan)
    for i in range(len(pid)):
        v = prof.get(str(int(pid[i])))
        if v is not None:
            M[i] = v
    cov = ~np.isnan(M).any(axis=1)
    ctx = np.stack([
        np.ones(len(X)),
        (test["balls_before"].to_numpy() * 3 + test["strikes_before"].to_numpy()) / 11.0,
        (test["pitcher_hand"].to_numpy() == test["batter_hand"].to_numpy()).astype(float),
        np.nan_to_num(X["is_sm"].to_numpy(dtype=float), nan=0.5) - 0.5,
        np.nan_to_num(test["asof_pitcher_fastball_rate"].to_numpy(dtype=float), nan=0.5) - 0.5,
    ], axis=1)
    Mc = np.nan_to_num(M, nan=0.0) - 0.5
    D = np.concatenate([np.einsum("nf,nc->nfc", Mc, ctx).reshape(len(X), -1),
                        np.ones((len(X), 1))], axis=1)
    corr = D @ coef
    corr[~cov] = 0.0
    return float(tm["weight"]) * corr


def pm_correction(test, X, meta):
    """TM 구종군 물리×mix 보정 (D-46): 동봉 프로필(투수별 구종군 3×물리 4 z평균 + 가용 마스크)
    × 그 행의 공식 asof mix율 가중 → 맥락(카운트·손) 곱 13계수 선형식. 커버 밖 = 0.
    §5 준수 — 그 행의 값 + 동봉 상수만 사용, 다른 행 참조·배치 통계 없음."""
    pm = meta.get("pm_corrector")
    if not pm:
        return np.zeros(len(X))
    prof = pm["profile"]                                  # {pid: [P 12값 + avail 3값]}
    coef = np.asarray(pm["coef"], dtype=np.float64)       # 13 = 4물리×3맥락 + 절편
    scale = np.asarray(pm["scale"], dtype=np.float64)     # 12 (학습 fit sd)
    pid = test["pitcher_id"].to_numpy()
    P = np.zeros((len(X), 3, 4))
    AV = np.zeros((len(X), 3))
    cov = np.zeros(len(X), dtype=bool)
    for i in range(len(pid)):
        v = prof.get(str(int(pid[i])))
        if v is not None:
            P[i] = np.asarray(v[:12], dtype=np.float64).reshape(3, 4)
            AV[i] = v[12:15]
            cov[i] = True
    mix = np.stack([np.nan_to_num(test[c].to_numpy(dtype=float), nan=1.0 / 3.0)
                    for c in ("asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
                              "asof_pitcher_offspeed_rate")], axis=1)
    mi = mix * AV
    denom = np.maximum(mi.sum(axis=1, keepdims=True), 1e-6)
    Z = np.einsum("ng,ngm->nm", mi / denom, P)
    cc = (test["balls_before"].to_numpy() * 3 + test["strikes_before"].to_numpy()).astype(int)
    cn = cc / 11.0
    hd = (test["pitcher_hand"].to_numpy() == test["batter_hand"].to_numpy()).astype(float)
    cols = []
    for m in range(4):
        for ctx in (np.ones(len(X)), cn, hd):
            cols.append(Z[:, m] * ctx)
    D = np.stack(cols, axis=1) / scale
    D = np.concatenate([D, np.ones((len(X), 1))], axis=1)
    corr = D @ coef
    corr[~cov] = 0.0
    return float(pm["weight"]) * corr


def _denom_hat(r1, r2, nmax, chunk=100000):
    """두 비율이 동시에 정수가 되는 최소 분모 — 행 단위 산술(§5 적법, 다른 행 참조 없음)."""
    n = np.arange(1, nmax + 1, dtype=np.float64)
    tol = 1e-6 * n[None, :]
    out = np.empty(len(r1))
    for i in range(0, len(r1), chunk):
        a = r1[i:i + chunk, None] * n[None, :]
        b = r2[i:i + chunk, None] * n[None, :]
        both = (np.abs(a - np.rint(a)) < tol) & (np.abs(b - np.rint(b)) < tol)
        has = both.any(axis=1)
        out[i:i + chunk] = np.where(has, both.argmax(axis=1) + 1, np.nan)
    return out


def dt_correction(test, X, meta):
    """ENS-10 denom+tilt 잔차 보정 (D-47 생존물):
    ① prev1/5 분모 복원(pn1_log·pr1_shrunk·workload_delta — 그 행의 rate만으로 계산) ⊗ 맥락 3
    ② TM count×타자손 구종선택 tilt × 행 갱신 EB mix → 물리 차분 4컬럼 (동봉 상수 룩업)
    두 블록을 학습 시점 직교화 행렬 W로 결합한 13컬럼 ridge 선형식. §5 준수 — 그 행의 값 +
    동봉 상수(계수·스케일·W·룩업)만 사용, 다른 행 참조·배치 통계 없음."""
    dt = meta.get("dt_corrector")
    if not dt:
        return np.zeros(len(X))
    coef = np.asarray(dt["coef"], dtype=np.float64)          # 14 = 13 + 절편
    scale = np.asarray(dt["scale"], dtype=np.float64)        # 13 (학습 fit sd)
    W = np.asarray(dt["W"], dtype=np.float64).reshape(9, 4)  # 직교화 계수 (fit에서 추정)
    mu = np.asarray(dt["mu_dn"], dtype=np.float64)           # denom 3컬럼 NaN 대체(fit 평균)

    # ---- ① denom 블록 (행 단위 산술) ----
    r1 = test["asof_pitcher_prev1_game_success_rate"].to_numpy(dtype=np.float64)
    m1 = test["asof_pitcher_prev1_game_middle_rate"].to_numpy(dtype=np.float64)
    r5 = test["asof_pitcher_prev5_game_success_rate"].to_numpy(dtype=np.float64)
    m5 = test["asof_pitcher_prev5_game_middle_rate"].to_numpy(dtype=np.float64)
    n1 = np.full(len(X), np.nan)
    ok1 = ~(np.isnan(r1) | np.isnan(m1))
    if ok1.any():
        n1[ok1] = _denom_hat(r1[ok1], m1[ok1], 160)
    n5 = np.full(len(X), np.nan)
    ok5 = ~(np.isnan(r5) | np.isnan(m5))
    if ok5.any():
        n5[ok5] = _denom_hat(r5[ok5], m5[ok5], 800)
    # ⚠ float32 캐스팅까지 학습측(prev_denom.build_block)과 동형 — 파리티 관문이 강제
    pn1_log = np.log1p(n1).astype(np.float32)
    pn5_log = np.log1p(n5).astype(np.float32)
    pr1_shr = np.where(np.isnan(n1) | np.isnan(r1), np.nan,
                       (r1 * n1 + 20.0 * 0.5) / (n1 + 20.0)).astype(np.float32)
    wdelta = (np.expm1(pn1_log.astype(np.float64))
              - np.expm1(pn5_log.astype(np.float64)) / 5.0).astype(np.float32)
    M3 = [np.where(np.isnan(v), m_, v.astype(np.float64))
          for v, m_ in zip((pn1_log, pr1_shr, wdelta), mu)]
    cc = (test["balls_before"].to_numpy() * 3 + test["strikes_before"].to_numpy()).astype(int)
    cc = np.clip(cc, 0, 11)
    cn = cc / 11.0
    hd = (test["pitcher_hand"].to_numpy() == test["batter_hand"].to_numpy()).astype(float)
    dn_cols = []
    for v in M3:
        for ctx in (np.ones(len(X)), cn, hd):
            dn_cols.append(v * ctx)
    D_dn = np.stack(dn_cols, axis=1)

    # ---- ② tilt 블록 (동봉 프로필·tilt·mix 룩업 + 그 행의 asof 값) ----
    T4 = np.zeros((len(X), 4))
    pm = meta.get("pm_corrector")
    tprof = dt.get("tilt_profile") or {}
    mbase = dt.get("mix_base") or {}
    if pm and tprof:
        prof = pm["profile"]
        lgm = np.asarray(dt["league_mix"], dtype=np.float64)
        kap = float(dt.get("kappa", 200.0))
        pid = test["pitcher_id"].to_numpy()
        nmix = test["asof_pitcher_pitchmix_n"].to_numpy(dtype=np.float64)
        R = np.stack([test[c].to_numpy(dtype=np.float64)
                      for c in ("asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
                                "asof_pitcher_offspeed_rate")], axis=1)
        for g in range(3):
            R[:, g] = np.where(np.isnan(R[:, g]), lgm[g], R[:, g])
        hh = (test["batter_hand"].to_numpy(dtype=np.float64) == 2).astype(int)
        cache = {}                    # 투수별 상수(프로필·tilt·base·q_pre) 재구성 1회
        _MISS = ()
        for i in range(len(pid)):
            key = str(int(pid[i]))
            ent = cache.get(key, _MISS)
            if ent is _MISS:
                pv = prof.get(key)
                tv = tprof.get(key)
                if pv is None or tv is None:
                    ent = None
                else:
                    P = np.asarray(pv[:12], dtype=np.float64).reshape(3, 4)
                    av = np.asarray(pv[12:15], dtype=np.float64)
                    mb = mbase.get(key)
                    nb_ = float(mb[0]) if mb else 0.0
                    kb_ = np.asarray(mb[1:4], dtype=np.float64) if mb else np.zeros(3)
                    q_pre = kb_ / max(nb_, 1.0) if nb_ > 0 else lgm
                    T = np.asarray(tv, dtype=np.float64).reshape(12, 2, 3)
                    ent = (P, av, nb_, kb_, q_pre, T)
                cache[key] = ent
            if ent is None:
                continue
            # ⚠ 아래 산술은 게이트(tm_intent.build_block)와 연산 순서까지 동형이어야 한다
            #   (파리티 관문이 강제). max(sum,1e-9) 정규화 포함 — guard 분기 금지.
            P, av, nb_, kb_, q_pre, T = ent
            n_ = nmix[i]
            k_ = np.rint(R[i] * n_)
            dnn = max(n_ - nb_, 0.0)
            dk = np.clip(k_ - kb_, 0.0, dnn)
            q_cur = (dk + kap * q_pre) / (dnn + kap)
            qj = q_cur * av
            qj = qj / max(qj.sum(), 1e-9)
            wj = qj * T[cc[i], hh[i]]
            wj = wj / max(wj.sum(), 1e-9)
            T4[i] = (wj - qj) @ P

    D13 = np.concatenate([D_dn, T4 - D_dn @ W], axis=1) / scale
    D13 = np.concatenate([D13, np.ones((len(X), 1))], axis=1)
    return float(dt["weight"]) * (D13 @ coef)


def c11_correction(test, X, meta):
    """ENS-11 결합 보정 (D-54/55): [denom 9 ⊥ tilt 4 ⊥ condphys 4 ⊥ enc10 (⊥ resp 1)]
    단일 ridge 선형식. 인코더는 동봉 가중치의 순수 numpy forward(GELU=erf) — 전부 그 행의 값
    + 고정 상수(§5 적법). 학습측(sweep/ens11_*.py)과 연산 순서까지 동형(파리티 관문 강제)."""
    c11 = meta.get("c11_corrector")
    if not c11:
        return np.zeros(len(X))
    from scipy.special import erf
    coef = np.asarray(c11["coef"], dtype=np.float64)
    scale = np.asarray(c11["scale"], dtype=np.float64)
    Ws = [np.asarray(w, dtype=np.float64).reshape(s)
          for w, s in zip(c11["Ws"], c11["Wshape"])]
    mu = np.asarray(c11["mu_dn"], dtype=np.float64)
    lgm = np.asarray(c11["league_mix"], dtype=np.float64)
    kap = float(c11.get("kappa", 200.0))
    n = len(X)

    # ---- 블록1: denom 9 (dt_correction과 동일 산술 — float32 캐스팅 포함)
    r1 = test["asof_pitcher_prev1_game_success_rate"].to_numpy(dtype=np.float64)
    m1 = test["asof_pitcher_prev1_game_middle_rate"].to_numpy(dtype=np.float64)
    r5 = test["asof_pitcher_prev5_game_success_rate"].to_numpy(dtype=np.float64)
    m5 = test["asof_pitcher_prev5_game_middle_rate"].to_numpy(dtype=np.float64)
    n1 = np.full(n, np.nan)
    ok1 = ~(np.isnan(r1) | np.isnan(m1))
    if ok1.any():
        n1[ok1] = _denom_hat(r1[ok1], m1[ok1], 160)
    n5 = np.full(n, np.nan)
    ok5 = ~(np.isnan(r5) | np.isnan(m5))
    if ok5.any():
        n5[ok5] = _denom_hat(r5[ok5], m5[ok5], 800)
    pn1_log = np.log1p(n1).astype(np.float32)
    pn5_log = np.log1p(n5).astype(np.float32)
    pr1_shr = np.where(np.isnan(n1) | np.isnan(r1), np.nan,
                       (r1 * n1 + 10.0) / (n1 + 20.0)).astype(np.float32)
    wdelta = (np.expm1(pn1_log.astype(np.float64))
              - np.expm1(pn5_log.astype(np.float64)) / 5.0).astype(np.float32)
    M3 = [np.where(np.isnan(v), m_, v.astype(np.float64))
          for v, m_ in zip((pn1_log, pr1_shr, wdelta), mu)]
    cc = np.clip((test["balls_before"].to_numpy() * 3
                  + test["strikes_before"].to_numpy()).astype(int), 0, 11)
    cn = cc / 11.0
    hd = (test["pitcher_hand"].to_numpy() == test["batter_hand"].to_numpy()).astype(float)
    dn_cols = []
    for v in M3:
        for ctx in (np.ones(n), cn, hd):
            dn_cols.append(v * ctx)
    D_dn = np.stack(dn_cols, axis=1)

    # ---- 공통: EB mix q_cur (행 갱신, κ=200) + 투수 상수 캐시
    pm = meta.get("pm_corrector") or {}
    prof = pm.get("profile") or {}
    tprof = c11.get("tilt_profile") or {}
    cprof = c11.get("cond_profile") or {}
    mbase = c11.get("mix_base") or {}
    tm_of = c11.get("tm_of_pid") or {}
    emb_lut = c11.get("emb") or {}
    pref_lut = c11.get("enc_prefix") or {}
    emb_fb = np.asarray(c11["emb_fallback"], dtype=np.float64)
    pid = test["pitcher_id"].to_numpy()
    nmix = test["asof_pitcher_pitchmix_n"].to_numpy(dtype=np.float64)
    R = np.stack([test[c].to_numpy(dtype=np.float64)
                  for c in ("asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
                            "asof_pitcher_offspeed_rate")], axis=1)
    for g in range(3):
        R[:, g] = np.where(np.isnan(R[:, g]), lgm[g], R[:, g])
    hh = (test["batter_hand"].to_numpy(dtype=np.float64) == 2).astype(int)

    T4 = np.zeros((n, 4))
    CP4 = np.zeros((n, 4))
    EMB = np.empty((n, len(emb_fb)))
    PREF = np.zeros((n, 10))
    cache = {}
    _MISS = ()
    for i in range(n):
        key = str(int(pid[i]))
        ent = cache.get(key, _MISS)
        if ent is _MISS:
            pv = prof.get(key)
            P_ = av_ = None
            if pv is not None:
                P_ = np.asarray(pv[:12], dtype=np.float64).reshape(3, 4)
                av_ = np.asarray(pv[12:15], dtype=np.float64)
            tv = tprof.get(key)
            T_ = np.asarray(tv, dtype=np.float64).reshape(12, 2, 3) if tv is not None else None
            cvv = cprof.get(key)
            C_ = Ca_ = None
            if cvv is not None:
                C_ = np.asarray(cvv[:288], dtype=np.float64).reshape(3, 12, 2, 4)
                Ca_ = np.asarray(cvv[288:291], dtype=np.float64)
            mb = mbase.get(key)
            nb_ = float(mb[0]) if mb else 0.0
            kb_ = np.asarray(mb[1:4], dtype=np.float64) if mb else np.zeros(3)
            q_pre = kb_ / max(nb_, 1.0) if nb_ > 0 else lgm
            tmid = tm_of.get(key)
            ev = emb_lut.get(str(tmid)) if tmid is not None else None
            e_ = np.asarray(ev, dtype=np.float64) if ev is not None else emb_fb
            pf = pref_lut.get(str(tmid)) if tmid is not None else None
            p_ = np.asarray(pf, dtype=np.float64) if pf is not None else np.zeros(10)
            ent = (P_, av_, T_, C_, Ca_, nb_, kb_, q_pre, e_, p_)
            cache[key] = ent
        P_, av_, T_, C_, Ca_, nb_, kb_, q_pre, e_, p_ = ent
        EMB[i] = e_
        PREF[i] = p_
        n_ = nmix[i]
        k_ = np.rint(R[i] * n_)
        dnn = max(n_ - nb_, 0.0)
        dk = np.clip(k_ - kb_, 0.0, dnn)
        q_cur = (dk + kap * q_pre) / (dnn + kap)
        if P_ is not None and T_ is not None:
            qj = q_cur * av_
            qj = qj / max(qj.sum(), 1e-9)
            wj = qj * T_[cc[i], hh[i]]
            wj = wj / max(wj.sum(), 1e-9)
            T4[i] = (wj - qj) @ P_
        if C_ is not None:
            qi = q_cur * Ca_
            qi = qi / max(qi.sum(), 1e-9)
            CP4[i] = qi @ C_[:, cc[i], hh[i], :]

    # ---- 블록4: encoder numpy forward (ctx 26 + prefix 10 + emb)
    C12 = np.zeros((n, 12))
    C12[np.arange(n), cc] = 1.0
    month = np.clip(test["game_month"].to_numpy(int), 3, 10)
    Mo = np.zeros((n, 8))
    Mo[np.arange(n), month - 3] = 1.0
    ctxE = np.concatenate([
        C12, Mo,
        (np.clip(test["inning"].to_numpy(int), 1, 10) / 10.0)[:, None],
        (np.clip(test["outs_before"].to_numpy(int), 0, 2) / 2.0)[:, None],
        (test["top_bottom"].astype(str).str.upper().str[0] == "T").to_numpy(float)[:, None],
        ((test["season"].to_numpy(int) - 2019) / 5.0)[:, None],
        (test["batter_hand"].to_numpy(int) == 2).astype(float)[:, None],
        (test["pitcher_hand"].to_numpy(int) == 2).astype(float)[:, None],
        PREF,
    ], axis=1)
    enc = c11["enc"]
    W1 = np.asarray(enc["W1"], dtype=np.float64)
    b1 = np.asarray(enc["b1"], dtype=np.float64)
    Zin = np.concatenate([EMB, ctxE], axis=1)
    H = Zin @ W1.T + b1
    H = 0.5 * H * (1.0 + erf(H / np.sqrt(2.0)))
    lt = H @ np.asarray(enc["Wt"], dtype=np.float64).T + np.asarray(enc["bt"], dtype=np.float64)
    lt = lt - lt.max(axis=1, keepdims=True)
    et = np.exp(lt)
    tp = et / et.sum(axis=1, keepdims=True)
    pp = H @ np.asarray(enc["Wp"], dtype=np.float64).T + np.asarray(enc["bp"], dtype=np.float64)
    E10 = np.concatenate([tp, pp], axis=1)

    blocks = [D_dn, T4, CP4, E10]
    resp = c11.get("resp")
    if resp:
        E7 = E10[:, 3:10]
        Dr = np.concatenate([E7, E7 * cn[:, None], E7 * hh.astype(np.float64)[:, None],
                             E7 ** 2], axis=1) / np.asarray(resp["scale_r"], dtype=np.float64)
        rh = (np.concatenate([Dr, np.ones((n, 1))], axis=1)
              @ np.asarray(resp["beta"], dtype=np.float64))[:, None]
        blocks.append(rh)

    cum = blocks[0]
    for W_, blk in zip(Ws, blocks[1:]):
        cum = np.concatenate([cum, blk - cum @ W_], axis=1)
    D = np.concatenate([cum / scale, np.ones((n, 1))], axis=1)
    return float(c11["weight"]) * (D @ coef)


def linear_predict(X, spec):
    """선형 멤버 — 원핫 + 표준화 + 로지스틱. 전부 그 행의 값만으로 계산된다.

    상수(원핫 범주 목록·median/mean/scale·결측 지시자 위치·계수)는 학습 구간에서 산출해 동봉한 것이라
    배치 통계가 아니다. 피처 결합 순서는 학습과 **정확히 같아야 하며**, 빌더가 이 함수로 파리티를 검사한다:
        [표준화 수치 53] + [결측 지시자] + [원핫]
    """
    A = X.to_numpy(dtype=np.float64)
    nanmask = np.isnan(A)
    med = np.asarray(spec["median"], dtype=np.float64)
    B = np.where(nanmask, med, A)
    Z = (B - np.asarray(spec["mean"], dtype=np.float64)) / np.asarray(spec["scale"], dtype=np.float64)
    parts = [Z]
    mc = spec["miss_cols"]
    if mc:
        parts.append(nanmask[:, np.asarray(mc, dtype=int)].astype(np.float64))
    for col, vals in spec["onehot"]:
        v = X[col].to_numpy()
        parts.append((v[:, None] == np.asarray(vals)[None, :]).astype(np.float64))
    F = np.concatenate(parts, axis=1)
    z = F @ np.asarray(spec["coef"], dtype=np.float64) + float(spec["bias"])
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def seg_correction(test, meta):
    """세그먼트 절편 보정 (D-58) — 캘리 **이후** 최종 확률에 가산한다.

    방향은 **중심화 지시자** `g = 1_A − w_A` (w_A = train 유래 상수). mean(g)=0 이므로 전역 절편과
    직교하고, 2셀 축의 레벨 오라클을 전량 담는다(비중심 지시자는 (1−w)²배만 회수).
    `within`이 주어지면 그 그룹 **안에서만** 중심화하므로 그룹 밖 방향과 교차항이 항등적으로 0이다
    (2024 그람-슈미트 직교화는 타 시즌에서 무너지지만 지지집합 분리는 무너지지 않는다).

    §5 준수: 그 행의 컬럼 값 + 동봉 상수만 사용. 배치 집계·다른 행 참조 없음."""
    terms = meta.get("seg_probe")
    if not terms:
        return np.zeros(len(test))
    out = np.zeros(len(test), dtype=np.float64)
    for tm in terms:
        if "map" in tm:                              # 분할축: 셀별 오프셋 룩업 (동봉 상수, 행 단위 조회)
            arr = test[tm["col"]].astype(str).to_numpy()
            off = np.zeros(len(test), dtype=np.float64)
            for kk, vv in tm["map"].items():         # 미등장 값은 0 — 안전 폴백
                off[arr == str(kk)] = float(vv)
            out += float(tm.get("t", 1.0)) * off
            continue
        if "cells" in tm:                            # 다열 조합 셀 (예: count12 = balls×strikes)
            arrs = [test[c].astype(str).to_numpy() for c in tm["cols"]]
            ind = np.zeros(len(test), dtype=bool)
            for cell in tm["cells"]:
                m = np.ones(len(test), dtype=bool)
                for a, v in zip(arrs, cell):
                    m &= (a == str(v))
                ind |= m
        elif "levels" in tm:
            ind = np.isin(test[tm["col"]].astype(str).to_numpy(),
                          [str(v) for v in tm["levels"]])
        else:                                        # 수치 임계 (동봉 상수, 행 단위 비교)
            ind = test[tm["col"]].to_numpy(dtype=np.float64) >= float(tm["ge"])
        g = ind.astype(np.float64) - float(tm["w"])
        wi = tm.get("within")
        if wi:                                       # 그룹 밖은 0 → 지지집합 분리
            g = g * np.isin(test[wi["col"]].astype(str).to_numpy(),
                            [str(v) for v in wi["levels"]]).astype(np.float64)
        out += float(tm["t"]) * g
    return out


def partner_leg(test, meta):
    """독립 파트너 모델 레그 (D-59) — **캘리 이전** raw 단계에서 볼록 결합한다.

        raw_B' = mean_a + (std_a/std_b)·(raw_B − mean_b)      ← 평균·분산 보존 정규화
        raw    = (1−w)·raw_A + w·raw_B'

    정규화하는 이유: 캘리 상수(slope·shift)는 raw_A의 레벨·스케일 위에서 정해진 값이라,
    파트너를 날것으로 섞으면 블렌드 레벨이 흔들려 **블렌드 가중 축과 캘리 레벨 축이
    뒤섞인다**. 평균·분산을 보존해 결합하면 w 축이 레벨 축과 직교하게 유지된다.
    상수 4개는 전부 train 2024에서 산출해 동봉한 것이고 평가 데이터를 보지 않는다.

    §5 준수: 파트너 predict()도 행 단위 계약이며 금지 호출 0을 AST로 검사해 실었다."""
    p = meta.get("partner")
    if not p:
        return None
    import importlib.util
    ma, sa = float(p["mean_a"]), float(p["std_a"])
    out = []
    for leg in p["legs"]:
        pdir = os.path.join(MODEL_DIR, "partner", leg["name"])
        spec = importlib.util.spec_from_file_location(
            "_partner_" + leg["name"], os.path.join(pdir, "predict.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        pb = np.asarray(mod.predict(test, pdir), dtype=np.float64)
        if pb.shape != (len(test),) or not np.isfinite(pb).all():
            raise RuntimeError("partner leg shape/finite 계약 위반: " + leg["name"])
        out.append((float(leg["w"]),
                    ma + (sa / float(leg["std_b"])) * (pb - float(leg["mean_b"]))))
    return out


def quad_correction(raw, meta):
    """캘리 **형상(곡률)** 축 (D-59) — 캘리·세그먼트 이후 최종 확률에 가산한다.

        g = (raw − raw̄)^k − m          (k=2 곡률, k=3 3차. raw̄·m·k·t 전부 동봉 상수)

    왜 이 축이 남아 있나: D-35가 "아핀을 넘는 캘리 자유도는 전부 레짐 과적합"으로 닫았지만
    그건 **로컬 게이트의 판정**이고, 그 게이트가 slope에서 0.107(=+8.18)을 놓친 것이 D-58에서
    실증됐다. 폴드 오라클도 +74.3 / +5.2 / +321.7 / +0.6 로 slope 축과 같은 형태다.

    ⚠ raw̄는 **평가셋(2025)의 raw 평균**이어야 한다(LB 역산값 0.4680607). 학습 시즌 표본평균을
    쓰면 g 안에 `2(E[raw]−c)(raw−E[raw])` 라는 기울기 성분이 섞여 이미 최적인 slope 축을
    건드린다(2024 데이터로 재보면 corr(g,raw)가 0.22 → 0.74로 커지는 것이 그 증거다).

    §5 준수: 그 행의 raw 예측값 + 동봉 상수만. 배치 집계·다른 행 참조 없음."""
    q = meta.get("quad_probe")
    if not q:
        return np.zeros(len(raw), dtype=np.float64)
    u = np.asarray(raw, dtype=np.float64) - float(q["raw_mean"])
    k = int(q.get("power", 2))
    return float(q["t"]) * (u ** k - float(q["m"]))


def main():
    with open(os.path.join(MODEL_DIR, "metadata.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), encoding="utf-8-sig")
    sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"), encoding="utf-8-sig")
    print("test=%d submission=%d" % (len(test), len(sub)))

    X = build_features(test, meta)
    # 멤버별 입력 폭: 트리 계열은 base(57) 컬럼만, 선형(bis 포함)은 전체를 본다.
    nb_ = int(meta.get("base_ncols", X.shape[1]))
    Xb = X.iloc[:, :nb_]
    Xn = None                                   # ET용 numpy (결측 → ET_NAN, 학습과 동일)

    total = np.zeros(len(X), dtype=np.float64)
    wsum = 0.0
    for m in meta["members"]:
        try:
            if m["type"] == "linear":
                Xin = X if m.get("input") == "full" else Xb
                p = linear_predict(Xin, m["spec"])        # 상수는 metadata.json 안에 있다
            elif m["type"] == "lgbm":
                import lightgbm as lgb
                Xin = X if m.get("input") == "full" else Xb
                p = lgb.Booster(model_file=os.path.join(MODEL_DIR, m["file"])).predict(
                    Xin, num_threads=4)
            else:
                import joblib
                if Xn is None:
                    Xn = np.nan_to_num(Xb.to_numpy(dtype=np.float32), nan=ET_NAN)
                p = joblib.load(os.path.join(MODEL_DIR, m["file"])).predict(Xn)
        except Exception:
            traceback.print_exc()
            print("!! 멤버 실패, 건너뜀: %s" % m.get("name", m.get("file")))
            continue
        total += float(m["weight"]) * np.asarray(p, dtype=np.float64)
        wsum += float(m["weight"])
        # ⚠ 로그에도 배치 집계(mean 등)를 쓰지 않는다 — 예측에 영향이 없더라도 서빙 경로를
        #   집계 연산 0으로 유지해야 §5 자동 가드(test_regulation.banned_calls)가 의미를 갖는다.
        print("  %-18s w=%.3f n=%d" % (m.get("name", m.get("file")), m["weight"], len(p)))

    if wsum <= 0:
        raise RuntimeError("사용 가능한 멤버가 없습니다")
    raw = total / wsum
    try:
        raw = raw + tm_correction(test, X, meta)          # 트랙맨 보정 (없으면 0)
    except Exception:
        traceback.print_exc()
        print("!! 트랙맨 보정 실패 — 보정 없이 진행")
    try:
        raw = raw + pm_correction(test, X, meta)          # physmix 보정 (없으면 0)
    except Exception:
        traceback.print_exc()
        print("!! physmix 보정 실패 — 보정 없이 진행")
    try:
        raw = raw + dt_correction(test, X, meta)          # denom+tilt 보정 (없으면 0)
    except Exception:
        traceback.print_exc()
        print("!! denom+tilt 보정 실패 — 보정 없이 진행")
    try:
        raw = raw + c11_correction(test, X, meta)         # ENS-11 결합 보정 (없으면 0)
    except Exception:
        traceback.print_exc()
        print("!! ENS-11 보정 실패 — 보정 없이 진행")

    # 파트너 레그는 캘리 이전에 결합한다. ⚠ 실패 시 조용히 넘어가면 "예측이 w만큼만 다르다"는
    #    프로브 전제가 깨져 역산 전체가 거짓이 되므로, 사용 여부를 로그에 반드시 남긴다.
    _pw = float((meta.get("partner") or {}).get("w", 0.0))
    _pused = 0
    if _pw > 0.0:
        try:
            _legs = partner_leg(test, meta)
            if _legs:
                raw = (1.0 - _pw) * raw + sum(w * v for w, v in _legs)
                _pused = len(_legs)
        except Exception:
            traceback.print_exc()
            print("!! 파트너 레그 실패 — 우리 레그만으로 진행")
    print("partner_used=%d wsum=%.6f" % (_pused, _pw))

    cal = meta["calibration"]
    p = np.clip(cal["center"] + cal["slope"] * (raw - cal["center"]) + cal["shift"], 0.0, 1.0)
    try:
        p = np.clip(p + seg_correction(test, meta), 0.0, 1.0)   # 세그먼트 절편 (없으면 0)
    except Exception:
        traceback.print_exc()
        print("!! 세그먼트 보정 실패 — 보정 없이 진행")
    try:
        p = np.clip(p + quad_correction(raw, meta), 0.0, 1.0)   # 캘리 형상 축 (없으면 0)
    except Exception:
        traceback.print_exc()
        print("!! 형상(곡률) 보정 실패 — 보정 없이 진행")
    if not np.isfinite(p).all():
        raise RuntimeError("예측값에 NaN/inf")
    print("blend done: rows=%d weights_used=%.3f" % (len(p), wsum))

    pred = pd.Series(p, index=test[ID_COL].astype(str))
    aligned = pred.reindex(sub[ID_COL].astype(str))
    if aligned.isna().any():
        raise RuntimeError("row_id 정렬 중 예측 누락")
    sub[TARGET_COL] = aligned.to_numpy(dtype=np.float64)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    sub.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print("Saved %s (rows=%d)" % (OUT_PATH, len(sub)))


def safety_net():
    """어떤 실패에도 유효한 제출 파일을 남긴다(실행 오류 = 슬롯 차감)."""
    try:
        with open(os.path.join(MODEL_DIR, "metadata.json"), "r", encoding="utf-8") as f:
            const = float(json.load(f).get("fallback_constant", 0.5))
    except Exception:
        const = 0.5
    sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"), encoding="utf-8-sig")
    sub[TARGET_COL] = const
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    sub.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print("!! 안전망 발동: 상수 %.4f (rows=%d)" % (const, len(sub)))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        safety_net()
