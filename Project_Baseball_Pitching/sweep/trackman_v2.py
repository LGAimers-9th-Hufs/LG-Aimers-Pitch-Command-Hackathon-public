# -*- coding: utf-8 -*-
"""W1-A — 매칭 v2: 설계(문서 13 §3)에 있었으나 미구현이던 5레버 + posterior linkage.

    python sweep/trackman_v2.py --match2                 # v2 매칭 + 4중 검증 + 라벨-프리 kill
    python sweep/trackman_v2.py --match2 --nperm 2000

v1(`trackman.py run_match`) 불변 보존 — 이 모듈은 v1 함수를 임포트해 재사용하고
결과를 `results/trackman/{matches_v2.csv, self_consistency_v2.json, posterior.json}`에 따로 쓴다.

## v2 레버 (근거 = 2026-08-08 탐사 실측)

1. **등판 분위수 5축 추가** — `app_q10..q90_R`은 `_app_profile`이 이미 계산하는데
   `SCALAR_AXES`에 빠져 있었다(등판 분포의 모양이 전부 버려지고 평균 1개만 사용).
2. **KIA 볼륨 보정** — tm 쪽 KIA는 홈(광주) 결측으로 절반 볼륨. n_R·n_app_R ×2.
3. **등판 튜플 집합 거리** — (월,요일,투구수) 결합의 투수-시즌 내 유일성 99.1% 실측.
   부분집합 정합(train ⊇ tm 가정: tm 등판이 train에서 얼마나 회수되는가)을 새 비용 축으로.
   시즌×쌍 행렬을 1회 사전계산(가중 지터는 가중치만 흔든다).
4. **재정규화 2회 루프** — 1차 할당의 고신뢰 상호쌍을 랭크 앵커 모집단으로 삼아
   양측 랭크를 재계산(모집단 불일치 왜곡 완화, §3.2 원설계).
5. **τ = margin 이봉 antimode 실측** — 현 τ=q25(0.72)는 첫 봉우리 정중앙(실측 antimode ≈1.2).
   커버 바닥(엔티티 35% / 행가중 50%) 미달 시 q25 폴백.

+ **posterior linkage**: Tier1 밖은 hard match 금지 — 지터 20회 + 시즌별 투표의 후보 분포를
  posterior로 저장(`P(tm_j|pid)`), 다운스트림(tm_v2_refit)이 posterior 가중 프로필로 소비.

## 라벨-프리 kill (다운스트림 점수 판독 전 필수 통과)

Tier1 churn(v1 대비 교체 비율) ≤5% — 초과 시 요일 perm z·팀 순도가 v1보다 개선돼야 함 ·
요일 permutation p<0.01 · 방법합치 ≥80% · 손 극성비 ≥1.5×.
여기서 매칭을 command 점수로 튜닝하지 않는다(이후 전 TM 결과의 선택 낙관 오염 방지).
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
import trackman as TM             # noqa: E402

OUT_DIR = HERE.parent / "results" / "trackman"
APP_Q_AXES = [("app_q10_R", 0.4), ("app_q25_R", 0.4), ("app_q50_R", 0.5),
              ("app_q75_R", 0.4), ("app_q90_R", 0.4)]
W_APP_SET = 1.2                   # 등판 집합 거리 가중 (히스토그램 mR_ 1.0과 동급+)
APP_NP_SCALE = 10.0               # |Δ투구수| 소프트 매칭 스케일
KIA_PREFIX = "KIA"


# ---------------------------------------------------------------- 등판 튜플
def app_tuples(df, ent, app_col, chan_col):
    """{(ent, season): ndarray[k,3] = (월, 요일, 투구수)} — R 채널만."""
    sub = df[df[chan_col] == "R"]
    ap = sub.groupby([ent, rd.SEASON, app_col]).agg(
        mo=("game_month", "first"), dw=("game_dayofweek", "first"),
        np_=("inning", "size")).reset_index()
    out = {}
    for (e, s), g in ap.groupby([ent, rd.SEASON]):
        out[(e, int(s))] = g[["mo", "dw", "np_"]].to_numpy(dtype=np.int32)
    return out


def _set_score(Ta, Tb):
    """부분집합 정합 점수 = tm 등판의 train 회수율(소프트). train ⊇ tm 가정."""
    if Ta is None or Tb is None or not len(Tb):
        return np.nan
    used = np.zeros(len(Ta), dtype=bool)
    tot = 0.0
    key_a = Ta[:, 0] * 8 + Ta[:, 1]
    key_b = Tb[:, 0] * 8 + Tb[:, 1]
    order_a = {}
    for i, k in enumerate(key_a):
        order_a.setdefault(int(k), []).append(i)
    for j in range(len(Tb)):
        cand = order_a.get(int(key_b[j]))
        if not cand:
            continue
        best, bq = -1, 0.0
        for i in cand:
            if used[i]:
                continue
            q = max(0.0, 1.0 - abs(float(Ta[i, 2]) - float(Tb[j, 2])) / APP_NP_SCALE)
            if q > bq:
                best, bq = i, q
        if best >= 0 and bq > 0:
            used[best] = True
            tot += bq
    return tot / len(Tb)


def app_set_matrices(tup_tr, tup_tm, e_tr, e_tm, seasons):
    """{season: (n_tr×n_tm) 거리행렬 = 1 − 정합점수}. 1회 계산 후 캐시(지터는 가중만 변경)."""
    mats = {}
    for s in seasons:
        A = [tup_tr.get((e, s)) for e in e_tr]
        B = [tup_tm.get((e, s)) for e in e_tm]
        ia = [i for i, t in enumerate(A) if t is not None]
        jb = [j for j, t in enumerate(B) if t is not None]
        M = np.full((len(e_tr), len(e_tm)), np.nan, dtype=np.float32)
        for i in ia:
            Ta = A[i]
            for j in jb:
                M[i, j] = 1.0 - _set_score(Ta, B[j])
        mats[s] = M
    return mats


# ---------------------------------------------------------------- 재정규화 랭크
def _rank_ref(v, mask_eval, mask_ref):
    """mask_ref 모집단을 앵커로 mask_eval 값들의 분위수 랭크."""
    out = np.full(len(v), np.nan)
    ref = np.sort(v[mask_ref & ~np.isnan(v)])
    if len(ref) < 2:
        out[mask_eval] = 0.5
        return out
    ev = np.where(mask_eval)[0]
    pos = np.searchsorted(ref, v[ev], side="left") + np.searchsorted(ref, v[ev], side="right")
    out[ev] = pos / (2.0 * len(ref))
    return out


def season_arrays_anchored(sig, ents, pos, season, hand_key, anchor=None):
    """TM._season_arrays와 동일하되, anchor(엔티티 bool 마스크) 제공 시 스칼라 랭크의
    모집단을 present∩anchor∩같은손 으로 앵커링한다."""
    base = TM._season_arrays(sig, ents, pos, season, hand_key)
    if anchor is None:
        return base
    n = len(ents)
    present = base["present"]
    raw = {}
    try:
        s = sig.xs(season, level=1)
    except KeyError:
        return base
    idx = np.array([pos[e] for e in s.index if e in pos])
    keep = np.array([e in pos for e in s.index])
    s = s[keep]
    for c, _ in TM.SCALAR_AXES:
        v = np.full(n, np.nan)
        if len(idx) and c in s:
            v[idx] = pd.to_numeric(s[c], errors="coerce").to_numpy()
        raw[c] = np.nan_to_num(v, nan=0.0)
    sides = pd.unique(base["hand"][present]) if present.any() else []
    for c, _ in TM.SCALAR_AXES:
        r = np.full(n, np.nan)
        for sd in sides:
            m_eval = present & (base["hand"] == sd)
            m_ref = m_eval & anchor
            if m_ref.sum() < 5:
                m_ref = m_eval
            rr = _rank_ref(raw[c], m_eval, m_ref)
            r[m_eval] = rr[m_eval]
        base[c] = r
    return base


def build_cost_v2(sig_tr, sig_tm, hand_map, ents, weights=None, seasons=None,
                  extra_mats=None, anchors=None):
    """TM.build_cost 동형 + 등판 집합 축 + 앵커 랭크. anchors=(mask_tr, mask_tm) or None."""
    seasons = seasons or TM.SEASONS
    (e_tr, p_tr, h_tr), (e_tm, p_tm, h_tm) = ents
    w = weights or {}
    C = np.zeros((len(e_tr), len(e_tm)), dtype=np.float32)
    n_co = np.zeros_like(C)
    a_tr = anchors[0] if anchors else None
    a_tm = anchors[1] if anchors else None
    for s in seasons:
        A = season_arrays_anchored(sig_tr, e_tr, p_tr, s, h_tr, anchor=a_tr)
        B = season_arrays_anchored(sig_tm, e_tm, p_tm, s, h_tm, anchor=a_tm)
        D, both = TM._pair_cost_season(A, B, weights)
        if extra_mats is not None and s in extra_mats:
            M = extra_mats[s]
            wt = np.float32(w.get("app_set", W_APP_SET))
            D = D + wt * np.nan_to_num(M, nan=0.6)      # 결측 = 중립보다 약간 벌점
        C += np.where(both, D, 0.0).astype(np.float32)
        n_co += both
        C += np.float32(TM.LAM_TM) * (~A["present"][:, None] & B["present"][None, :])
        C += np.float32(TM.LAM_TR) * (A["present"][:, None] & ~B["present"][None, :])
    hand_tr = np.array([hand_map.get(h_tr.get(e)) for e in e_tr], dtype=object)
    hand_tm = np.array([h_tm.get(e) for e in e_tm], dtype=object)
    C = np.where(hand_tr[:, None] != hand_tm[None, :], np.float32(TM.BIG), C)
    return C, n_co


# ---------------------------------------------------------------- τ antimode
def antimode_tau(margins, lo=0.2, hi=3.0):
    """margin 분포의 이봉 antimode. 실패 시 None."""
    from scipy.stats import gaussian_kde
    v = margins[(margins > lo) & (margins < hi)]
    if len(v) < 50:
        return None
    kde = gaussian_kde(v, bw_method=0.18)
    xs = np.linspace(lo, hi, 281)
    ys = kde(xs)
    peaks = [i for i in range(1, len(xs) - 1) if ys[i] > ys[i - 1] and ys[i] > ys[i + 1]]
    if len(peaks) < 2:
        return None
    top2 = sorted(sorted(peaks, key=lambda i: -ys[i])[:2])
    i0, i1 = top2
    j = i0 + int(np.argmin(ys[i0:i1 + 1]))
    return float(xs[j])


# ---------------------------------------------------------------- v2 드라이버
def run_match_v2(nperm=2000, seed=0):
    t0 = time.time()
    print(">> 시그니처 로드/생성")
    f_tr, f_tm = OUT_DIR / "sig_train.pkl", OUT_DIR / "sig_tm.pkl"
    if f_tr.exists() and f_tm.exists():
        sig_tr = pd.read_pickle(f_tr)
        sig_tm = pd.read_pickle(f_tm)
    else:
        df = rd.load_train()
        tr = TM.add_appearances(df)
        sig_tr = TM.season_signatures_train(tr)
        tm = TM.load_trackman()
        sig_tm = TM.season_signatures_tm(tm)

    # ---- 레버 ①: 등판 분위수 5축 추가 (전역 축 확장 — v2 프로세스 한정)
    for ax in APP_Q_AXES:
        if ax not in TM.SCALAR_AXES:
            TM.SCALAR_AXES.append(ax)
    print(f"   SCALAR_AXES {len(TM.SCALAR_AXES)}축 (app_q 5축 추가)")

    # ---- 레버 ②: KIA 볼륨 보정 (tm 쪽 ×2)
    sig_tm = sig_tm.copy()
    kia = sig_tm["team_raw"].astype(str).str.startswith(KIA_PREFIX)
    for c in ("n_R", "n_app_R"):
        sig_tm.loc[kia, c] = sig_tm.loc[kia, c] * 2.0
    sig_tm.loc[kia, "n_all"] = sig_tm.loc[kia, "n_R"] + sig_tm.loc[kia, "n_F"]
    print(f"   KIA 보정: {int(kia.sum())} 투수-시즌 볼륨 ×2")

    ents_tr = TM._entity_tables(sig_tr)
    ents_tm = TM._entity_tables(sig_tm)
    ents = (ents_tr, ents_tm)
    e_tr, e_tm = ents_tr[0], ents_tm[0]

    # ---- 레버 ③: 등판 튜플 집합 거리 (1회 사전계산)
    print(">> 등판 튜플 집합 거리 사전계산")
    t1 = time.time()
    df = rd.load_train()
    tr = TM.add_appearances(df)
    tr["chan"] = tr["game_type"]
    tup_tr = app_tuples(tr, "pitcher_id", "app_no", "chan")
    tmr = TM.load_trackman()
    tup_tm = app_tuples(tmr, "pitcher_trackman_id", "trackman_game_id", "chan")
    mats = app_set_matrices(tup_tr, tup_tm, e_tr, e_tm, TM.SEASONS)
    nn = sum(int((~np.isnan(M)).sum()) for M in mats.values())
    print(f"   행렬 6시즌 · 유효 셀 {nn:,} [{time.time()-t1:.0f}s]")

    # ---- 손 극성 자가검증
    print("\n[1] 손 극성 자가검증 (v2 비용)")
    pol = {}
    for name, hmap in (("2=Right", {2: "Right", 1: "Left"}), ("2=Left", {2: "Left", 1: "Right"})):
        C, _ = build_cost_v2(sig_tr, sig_tm, hmap, ents, extra_mats=mats)
        r, c = TM.assign(C)
        pol[name] = (float(C[r, c].sum()), hmap)
        print(f"    {name}: 총비용 {pol[name][0]:,.1f}")
    best_pol = min(pol, key=lambda k: pol[k][0])
    ratio = max(pol[k][0] for k in pol) / max(min(pol[k][0] for k in pol), 1e-9)
    hand_map = pol[best_pol][1]
    print(f"    → {best_pol} (비 {ratio:.2f}×)")

    # ---- 레버 ④: 재정규화 2회 루프
    print("\n[2] 커리어 Hungarian + 재정규화 루프")
    anchors = None
    rows = cols = None
    for it in range(3):
        C, n_co = build_cost_v2(sig_tr, sig_tm, hand_map, ents,
                                extra_mats=mats, anchors=anchors)
        rows, cols = TM.assign(C)
        mr, mc = TM._margins(C, rows, cols)
        co = n_co[rows, cols]
        hi = (co > 0) & (C[rows, cols] < TM.BIG / 2) & (mr >= np.quantile(mr[co > 0], 0.5))
        a_tr = np.zeros(len(e_tr), dtype=bool)
        a_tm = np.zeros(len(e_tm), dtype=bool)
        a_tr[rows[hi]] = True
        a_tm[cols[hi]] = True
        anchors = (a_tr, a_tm)
        print(f"    반복 {it}: 고신뢰 앵커 {int(hi.sum())}쌍 · "
              f"cost/season 중앙값 {np.median(C[rows, cols][co > 0] / np.maximum(co[co > 0], 1)):.3f}")
    m = pd.DataFrame({"pitcher_id": e_tr[rows], "tm_id": e_tm[cols],
                      "cost": C[rows, cols], "n_co": co,
                      "cost_per_season": C[rows, cols] / np.maximum(co, 1),
                      "margin_row": mr, "margin_col": mc})
    m["forced"] = m["cost"] >= TM.BIG / 2

    # ---- 요일 permutation (예약 차원, v1 로직 재사용 형태)
    print("\n[3] 요일 permutation (팀×손 블록)")
    rng = np.random.default_rng(seed)
    R, R_co = TM.build_cost(sig_tr, sig_tm, hand_map, ents=ents, reserved_only=True)
    ok = (co > 0) & (~m.forced.to_numpy())
    ri, cj = rows[ok], cols[ok]
    stat = lambda a, b: float(np.mean(R[a, b] / np.maximum(R_co[a, b], 1)))
    obs = stat(ri, cj)
    hand_tr = np.array([hand_map.get(ents_tr[2].get(e)) for e in e_tr], dtype=object)
    team_of = (sig_tr.reset_index().sort_values("n_all")
               .groupby("pitcher_id")["team_raw"].last())
    tm_team = np.array([str(team_of.get(e, -1)) for e in e_tr], dtype=object)
    blk = np.array([f"{a}|{b}" for a, b in zip(tm_team[ri], hand_tr[ri])], dtype=object)
    null = np.empty(nperm)
    for t in range(nperm):
        perm_c = cj.copy()
        for sd in np.unique(blk):
            idx = np.where(blk == sd)[0]
            if len(idx) > 1:
                perm_c[idx] = cj[rng.permutation(idx)]
        null[t] = stat(ri, perm_c)
    p = (1 + int((null <= obs).sum())) / (1 + nperm)
    z = (np.mean(null) - obs) / (np.std(null) + 1e-12)
    print(f"    관측 {obs:.4f} · 귀무 {null.mean():.4f}±{null.std():.4f} · z={z:.1f} · p={p:.5f}")

    # ---- 방법합치 (시즌별 독립) + posterior 재료
    print("\n[4] 방법합치 + 시즌 투표")
    per_season = {}
    for s in TM.SEASONS:
        Cs, _ = build_cost_v2(sig_tr, sig_tm, hand_map, ents,
                              extra_mats={s: mats[s]}, seasons=[s], anchors=anchors)
        A = TM._season_arrays(sig_tr, e_tr, ents_tr[1], s, ents_tr[2])["present"]
        B = TM._season_arrays(sig_tm, e_tm, ents_tm[1], s, ents_tm[2])["present"]
        ia, ib = np.where(A)[0], np.where(B)[0]
        if not len(ia) or not len(ib):
            continue
        rr, cc = TM.assign(Cs[np.ix_(ia, ib)])
        for a, b in zip(ia[rr], ib[cc]):
            per_season.setdefault(e_tr[a], []).append(e_tm[b])
    mode_map = {}
    for pid, lst in per_season.items():
        vals, cnts = np.unique(lst, return_counts=True)
        mode_map[pid] = vals[np.argmax(cnts)]
    m["season_mode"] = m["pitcher_id"].map(mode_map)
    m["n_season_obs"] = m["pitcher_id"].map({k: len(v) for k, v in per_season.items()}).fillna(0)
    multi = m[m.n_season_obs >= 2]
    rate = float((multi.season_mode == multi.tm_id).mean()) if len(multi) else float("nan")
    m["agree"] = (m.season_mode == m.tm_id) | (m.n_season_obs < 2)
    print(f"    다시즌 일치율 {rate:.1%}")

    # ---- 지터 + posterior linkage
    print("\n[5] 가중치 지터 + posterior")
    post = {int(pid): {} for pid in e_tr}
    for pid, lst in per_season.items():
        for q in lst:
            post[int(pid)][int(q)] = post[int(pid)].get(int(q), 0) + 1
    surv = np.zeros(len(m))
    NJIT = 20
    axes_w = ([(a, b) for a, b in TM.SCALAR_AXES + TM.DIRECT_AXES]
              + [(a, w0) for a, _, w0 in TM.HIST_AXES] + [("app_set", W_APP_SET)])
    for t in range(NJIT):
        w = {k: v * float(np.exp(rng.normal(0, 0.25))) for k, v in axes_w}
        Cj, _ = build_cost_v2(sig_tr, sig_tm, hand_map, ents, weights=w,
                              extra_mats=mats, anchors=anchors)
        rr, cc = TM.assign(Cj)
        pair = dict(zip(e_tr[rr], e_tm[cc]))
        surv += np.array([1.0 if pair.get(pq) == q else 0.0
                          for pq, q in zip(m.pitcher_id, m.tm_id)])
        for pq, q in pair.items():
            post[int(pq)][int(q)] = post[int(pq)].get(int(q), 0) + 1
    m["stability"] = surv / NJIT
    for pid, q in zip(m.pitcher_id, m.tm_id):                     # 커리어 할당 2표
        post[int(pid)][int(q)] = post[int(pid)].get(int(q), 0) + 2
    posterior = {}
    for pid, d in post.items():
        tot = sum(d.values())
        if tot:
            top = sorted(d.items(), key=lambda kv: -kv[1])[:5]
            posterior[str(pid)] = {str(k): round(v / tot, 4) for k, v in top}
    print(f"    지터 생존율 중앙값 {m.stability.median():.2f} · posterior {len(posterior)}투수")

    # ---- 팀 대조표 (독립 검증)
    tt = sig_tr.reset_index()[["pitcher_id", "season", "team_raw", "n_R"]]
    mt = sig_tm.reset_index()[["pitcher_trackman_id", "season", "team_raw", "n_R"]].rename(
        columns={"pitcher_trackman_id": "tm_id", "team_raw": "tm_team", "n_R": "n_R_tm"})
    j = (tt[tt.n_R > 0].merge(m[["pitcher_id", "tm_id"]], on="pitcher_id")
         .merge(mt[mt.n_R_tm > 0], on=["tm_id", "season"]))
    ct = pd.crosstab(j["team_raw"], j["tm_team"])
    purity = float((ct.max(axis=1) / ct.sum(axis=1)).mean())
    print(f"\n[6] 팀 순도 {purity:.1%}")

    # ---- 레버 ⑤: τ antimode + Tier + churn
    tau_am = antimode_tau(m.loc[m.n_co > 0, "margin_row"].to_numpy())
    tau_q25 = float(np.quantile(m.loc[m.n_co > 0, "margin_row"], 0.25))
    n_rows_map = sig_tr.groupby(level=0)["n_all"].sum()
    m["train_rows"] = m["pitcher_id"].map(n_rows_map)

    def tier_with(tau):
        t_ = pd.Series(3, index=m.index)
        t_[(m.n_co > 0) & (~m.forced) & (m.stability >= 0.6)] = 2
        t_[(m.n_co > 0) & (~m.forced) & m.agree & (m.stability >= 0.9)
           & (m.margin_row >= tau)] = 1
        return t_

    tau = tau_am if tau_am is not None else tau_q25
    t_try = tier_with(tau)
    cov_e = float((t_try == 1).mean())
    cov_r = float(m.loc[t_try == 1, "train_rows"].sum() / m["train_rows"].sum())
    if cov_e < 0.35 and cov_r < 0.50:
        print(f"    ⚠ antimode τ={tau:.2f} 커버 바닥 미달({cov_e:.1%}/{cov_r:.1%}) → q25 폴백")
        tau = tau_q25
        t_try = tier_with(tau)
    m["tier"] = t_try
    cov_e = float((m.tier == 1).mean())
    cov_r = float(m.loc[m.tier == 1, "train_rows"].sum() / m["train_rows"].sum())

    v1 = pd.read_csv(OUT_DIR / "matches.csv")
    v1t1 = v1[v1.tier == 1][["pitcher_id", "tm_id"]]
    mg = v1t1.merge(m[["pitcher_id", "tm_id"]], on="pitcher_id", suffixes=("_v1", "_v2"))
    churn = float((mg.tm_id_v1 != mg.tm_id_v2).mean())
    new_t1 = m[m.tier == 1].pitcher_id
    added = len(set(new_t1) - set(v1t1.pitcher_id))
    print(f"\n[7] τ={tau:.3f}{'(antimode)' if tau == tau_am else '(q25)'} · "
          f"Tier1 {int((m.tier==1).sum())} (v1 407) · 엔티티 {cov_e:.1%} · 행가중 {cov_r:.1%}")
    print(f"    v1 Tier1 churn {churn:.1%} [kill >5%] · 신규 Tier1 +{added}")

    kills = []
    if not (p < 0.01):
        kills.append(f"요일 perm p={p:.4f}")
    if not (rate >= 0.80):
        kills.append(f"방법합치 {rate:.1%}")
    if ratio < 1.5:
        kills.append(f"손 극성비 {ratio:.2f}")
    if churn > 0.05 and (z < 26.2 or purity < 0.6945):
        kills.append(f"churn {churn:.1%} + 검증 미개선(z {z:.1f} vs 26.2, 순도 {purity:.1%} vs 69.5%)")
    verdict = {"perm_p": p, "perm_z": float(z), "method_agree": rate,
               "polarity_ratio": float(ratio), "team_purity": purity,
               "tau": float(tau), "tau_antimode": tau_am, "tau_q25": tau_q25,
               "tier1": int((m.tier == 1).sum()), "tier1_ent": cov_e, "tier1_row": cov_r,
               "churn_vs_v1": churn, "new_tier1": added,
               "kills": kills, "go": not kills}
    print("\nGO/NO-GO: " + ("**GO**" if not kills else "**NO-GO** — " + " / ".join(kills)))

    m.to_csv(OUT_DIR / "matches_v2.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "self_consistency_v2.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "posterior.json").write_text(
        json.dumps(posterior, ensure_ascii=False), encoding="utf-8")
    print(f"저장: matches_v2.csv · self_consistency_v2.json · posterior.json ({time.time()-t0:.0f}s)")
    return m, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match2", action="store_true")
    ap.add_argument("--nperm", type=int, default=2000)
    args = ap.parse_args()
    if args.match2:
        run_match_v2(nperm=args.nperm)


if __name__ == "__main__":
    main()
