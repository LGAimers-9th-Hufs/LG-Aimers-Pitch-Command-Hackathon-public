# -*- coding: utf-8 -*-
"""S3 — trackman_history × train 투수-시즌 엔티티 매칭 (설계: docs/research/13 §3).

    python sweep/trackman.py --sig        # P0: 로더 + 양측 시그니처 생성/캐시 + 프로파일 리포트
    python sweep/trackman.py --match      # P1: 팀 매칭 → 손 극성 → 커리어 Hungarian → 4중 검증

서빙 코드가 아니다(AST 제약 없음). 출력물은 `results/trackman/`에 캐시된다.

행 단위 결합은 정량 기각됐다(문서 13 §1.2: 양쪽 유일 블록 1.9%). 유일한 활로는
**투수-시즌 분포 시그니처**로 엔티티(792 train pid ↔ 906 tm id)를 맞추는 것.

⚠ 문서 13 대비 P0에서 갱신된 사실 (2026-08-06 실측):
  1. **등판 구조 복원 성공(문서에 없던 최강 축)**: train의 row_id 순서는 투수별 투구 시간순이다
     (투수 내 `asof_pitcher_n` 증분이 100% +1). 등판 경계는 prev1/3/5 게임 통계 변화 + 이닝 감소
     + top_bottom 변화 + 득점 감소로 복원된다 → 시즌 등판수·등판당 투구수 분위수·역할(선발/불펜)이
     트랙맨의 `trackman_game_id` 집계와 **같은 정의로** 비교 가능해진다.
  2. **train F(퓨처스)는 퓨처스 전체가 아니다**: 특정 한 팀(id 13)의 퓨처스 경기만 기록돼 있고
     나머지 팀 F행은 그 상대팀 몫(팀당 시즌 0.8~2.7k)뿐이다. → F 채널의 워크로드는 매칭 비용에서
     **저가중**(문서 13 §3.4의 λ_tr=0 조항 발동). 주 매칭 축은 R(1군) 채널.
  3. **퓨처스 전용 엔티티 앵커**: train 25(F 292행, 2024만) ↔ tm MIN_HAW(292행, 2024만) — 행수 완전 일치.
     train 22(676행, 2019만) ↔ KBO_POL(902행, 2019만), train 23(6시즌) ↔ KBO_ARM(6시즌).
  4. tm 1군 커버리지 = train R의 약 90%(팀별 123~128k vs 132~134k), **KIA만 ~50%**(홈 광주 결측).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import real_data as rd  # noqa: E402

OUT_DIR = HERE.parent / "results" / "trackman"

# ---------------------------------------------------------------- 팀 계층 (실측 확정)
TM_FARM_PREFIX = "MIN_"
TM_FARM_EXTRA = {"KBO_ARM", "KBO_POL"}          # 상무·경찰 = 퓨처스 전용 엔티티
TM_EXCLUDE = {"ACE_MEX"}                        # 152행, 대응 없음 (문서 13 §1.3-6)
TM_FRANCHISE = {"SK_WYV": "SSG_LAN"}            # 리브랜드 통합 (2019–20 → 2021–)

TAGGED_TYPO = {"Changeup": "ChangeUp", "SInker": "Sinker",
               "Undefind": "Undefined", "Undefined#": "Undefined"}

MONTHS = list(range(3, 11))     # 3~10 (tm의 11월은 10월에 병합)
DOWS = list(range(7))
INNINGS = list(range(1, 11))    # 10 = 10회 이상
COUNTS = [(b, s) for b in range(4) for s in range(3)]
APP_Q = np.array([0.1, 0.25, 0.5, 0.75, 0.9])   # 등판당 투구수 분위수


def _chan_tm(team: pd.Series) -> pd.Series:
    """트랙맨 팀명 → 채널 'R'(1군) / 'F'(퓨처스) / None(제외)."""
    farm = team.str.startswith(TM_FARM_PREFIX) | team.isin(TM_FARM_EXTRA)
    out = pd.Series(np.where(farm, "F", "R"), index=team.index, dtype=object)
    out[team.isin(TM_EXCLUDE)] = None
    return out


# ---------------------------------------------------------------- 로딩 (§1.4 전처리 계약)
def load_trackman(path=None, downcast=True) -> pd.DataFrame:
    p = Path(path or rd.DATA_DIR / "trackman_history.csv")
    df = pd.read_csv(p, encoding="utf-8-sig")

    # game_date: 2019–21 MM/DD/YYYY(0패딩 불규칙) / 2022–24 YYYY-MM-DD 혼재
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")

    df["tagged_pitch_type"] = df["tagged_pitch_type"].replace(TAGGED_TYPO)
    df.loc[df["extension"] <= 0, "extension"] = np.nan      # min −0.387, 물리 불가
    df["game_month"] = df["game_month"].clip(upper=10)      # 11월(포스트시즌) → 10월 병합
    df["inning"] = df["inning"].clip(lower=1, upper=10)
    df["balls_before"] = df["balls_before"].clip(upper=3)
    df["strikes_before"] = df["strikes_before"].clip(upper=2)
    df["outs_before"] = df["outs_before"].clip(upper=2)

    df["chan"] = _chan_tm(df["pitcher_team"])
    df["franchise"] = df["pitcher_team"].replace(TM_FRANCHISE)
    df = df[df["chan"].notna()].copy()

    if downcast:
        for c in df.select_dtypes("float64").columns:
            df[c] = df[c].astype("float32")
    return df


# ---------------------------------------------------------------- 등판 복원 (train 전용)
def add_appearances(tr: pd.DataFrame) -> pd.DataFrame:
    """row_id 순 = 투수별 투구 시간순이라는 실측 사실 위에서 등판(경기) 경계를 복원한다.

    경계 규칙(투수 내 연속 행 비교, OR 결합):
      - prev1/prev3/prev5 게임 성공률·한복판률 6종 중 하나라도 변화 (다음 경기부터 값이 갱신)
      - 이닝 감소 / top_bottom 변화 / 팀·game_type·월·요일 변화 / 누적 득점 감소
    prev* 가 NaN(데뷔 초기)이거나 값이 우연히 같아 놓치는 경계를 나머지 규칙이 보완한다.
    """
    tr = tr.sort_values(rd.ID, kind="mergesort").reset_index(drop=True)
    g = tr.groupby("pitcher_id", sort=False)

    chg = np.zeros(len(tr), dtype=bool)
    for c in ["asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
              "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
              "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate"]:
        d = g[c].diff()
        chg |= (d.abs() > 1e-9).to_numpy()
        # NaN → 값 있음 전이(데뷔 후 첫 갱신)도 경계
        chg |= (g[c].shift(1).isna() & tr[c].notna()).to_numpy()

    for c in ["season", "game_month", "game_dayofweek", "pitcher_team_id"]:
        chg |= (g[c].diff().abs() > 0).to_numpy()
    chg |= (tr["game_type"] != g["game_type"].shift(1)).to_numpy() & g.cumcount().gt(0).to_numpy()
    chg |= (tr["top_bottom"] != g["top_bottom"].shift(1)).to_numpy() & g.cumcount().gt(0).to_numpy()
    chg |= (g["inning"].diff() < 0).to_numpy()
    chg |= (g["run_total_before"].diff() < 0).to_numpy()

    chg |= (g.cumcount() == 0).to_numpy()
    # ⚠ 행 순서는 **양팀 투수가 교차하는 전역 투구 순서**다 — 한 등판의 행은 연속하지 않는다.
    #   따라서 전역 cumsum은 두 투수를 한 세그먼트에 섞는다(실측 14,223건). 투수 내 누적이 정답.
    tr["app_no"] = pd.Series(chg, index=tr.index).groupby(tr["pitcher_id"]).cumsum().astype("int32")
    return tr


# ---------------------------------------------------------------- 시그니처
def _hist(df, key_cols, col, values, prefix):
    """(엔티티, 시즌)×범주 카운트 → 확률 히스토그램 (컬럼명 prefix+값)."""
    ct = (df.assign(_v=df[col])
            .pivot_table(index=key_cols, columns="_v", values=rd.ID if rd.ID in df else col,
                         aggfunc="size", fill_value=0))
    ct = ct.reindex(columns=values, fill_value=0).astype("float64")
    tot = ct.sum(axis=1).replace(0, np.nan)
    ct = ct.div(tot, axis=0).fillna(0.0)
    ct.columns = [f"{prefix}{v}" for v in values]
    return ct


def _app_profile(df, key_cols, app_col):
    """등판 단위 요약: 등판수 · 등판당 투구수 분위수 · 선발 비율 · 등판당 이닝수."""
    ap = df.groupby(key_cols + [app_col]).agg(np=("inning", "size"),
                                              inn=("inning", "nunique"),
                                              first_inn=("inning", "min"))
    g = ap.groupby(key_cols)
    out = pd.DataFrame({"n_app": g.size().astype(float)})
    q = g["np"].quantile(APP_Q).unstack()
    q.columns = [f"app_q{int(v*100)}" for v in APP_Q]
    out = out.join(q)
    out["app_mean"] = g["np"].mean()
    out["start_share"] = g["np"].apply(lambda s: float((s >= 50).mean()))
    out["inn_mean"] = g["inn"].mean()
    out["first_inn_mean"] = g["first_inn"].mean()
    return out


def _side_signature(df, ent, season_col, app_col, chan_col, tag):
    """한쪽(train/tm)의 (엔티티, 시즌) 시그니처. 채널 R/F를 접미사로 분리해 병렬 보관."""
    keys = [ent, season_col]
    parts = []
    for chan in ("R", "F"):
        sub = df[df[chan_col] == chan]
        if not len(sub):
            continue
        n = sub.groupby(keys).size().rename(f"n_{chan}").astype(float)
        blk = [n.to_frame()]
        blk.append(_hist(sub, keys, "game_month", MONTHS, f"m{chan}_"))
        blk.append(_hist(sub, keys, "game_dayofweek", DOWS, f"w{chan}_"))
        blk.append(_hist(sub, keys, "inning", INNINGS, f"i{chan}_"))
        blk.append(_hist(sub.assign(_cnt=sub["balls_before"].astype(int) * 3
                                    + sub["strikes_before"].astype(int)),
                         keys, "_cnt", list(range(12)), f"c{chan}_"))
        top = sub.groupby(keys)["top_bottom"].apply(
            lambda s: float((s.astype(str).str.upper().str[0] == "T").mean())).rename(f"top_{chan}")
        blk.append(top.to_frame())
        ap = _app_profile(sub, keys, app_col)
        ap.columns = [f"{c}_{chan}" for c in ap.columns]
        blk.append(ap)
        parts.append(pd.concat(blk, axis=1))

    sig = parts[0].join(parts[1:], how="outer") if len(parts) > 1 else parts[0]
    for chan in ("R", "F"):
        if f"n_{chan}" not in sig:
            sig[f"n_{chan}"] = 0.0
        sig[f"n_{chan}"] = sig[f"n_{chan}"].fillna(0.0)
    sig["n_all"] = sig["n_R"] + sig["n_F"]
    sig["side"] = tag
    return sig


def season_signatures_train(tr: pd.DataFrame) -> pd.DataFrame:
    if "app_no" not in tr:
        tr = add_appearances(tr)
    tr = tr.copy()
    tr["chan"] = tr["game_type"]
    sig = _side_signature(tr, "pitcher_id", rd.SEASON, "app_no", "chan", "train")
    hand = tr.groupby(["pitcher_id", rd.SEASON])["pitcher_hand"].agg(
        lambda s: int(s.mode().iloc[0]))
    team = tr.groupby(["pitcher_id", rd.SEASON])["pitcher_team_id"].agg(
        lambda s: int(s.mode().iloc[0]))
    sig["hand_raw"] = hand
    sig["team_raw"] = team
    sig = sig.join(_train_mix(tr), how="left")
    return sig


def season_signatures_tm(tm: pd.DataFrame) -> pd.DataFrame:
    sig = _side_signature(tm, "pitcher_trackman_id", rd.SEASON, "trackman_game_id", "chan", "tm")
    keys = ["pitcher_trackman_id", rd.SEASON]
    sig["hand_raw"] = tm.groupby(keys)["pitcher_hand"].agg(lambda s: s.mode().iloc[0])
    sig["team_raw"] = tm.groupby(keys)["franchise"].agg(lambda s: s.mode().iloc[0])
    mix = _hist(tm, keys, "pitch_type_group", ["fastball", "breaking", "offspeed"], "mix_")
    sig = sig.join(mix, how="left")
    return sig


def _train_mix(tr: pd.DataFrame) -> pd.DataFrame:
    """train의 시즌 구종믹스 복원 — asof_pitchmix는 커리어 누적 단조(문서 13 §1.3-1).

    시즌말 커리어 (n, 3률)에서 전 시즌말 값을 빼 **시즌 증분 믹스**를 만든다.
    trackman의 `pitch_type_group` 시즌 비율과 같은 정의가 된다.
    """
    keys = ["pitcher_id", rd.SEASON]
    cols = ["asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]
    last = tr.sort_values(rd.ID).groupby(keys)[cols].last()
    n = last["asof_pitcher_pitchmix_n"].astype(float)
    cnt = pd.DataFrame({k: last[f"asof_pitcher_{k}_rate"].astype(float) * n
                        for k in ("fastball", "breaking", "offspeed")})
    cnt["n"] = n
    prev = cnt.groupby(level=0).shift(1).fillna(0.0)
    inc = cnt - prev
    denom = inc["n"].where(inc["n"] > 0)
    out = pd.DataFrame({f"mix_{k}": (inc[k] / denom).clip(0, 1)
                        for k in ("fastball", "breaking", "offspeed")})
    out["mix_n_inc"] = inc["n"]
    return out


# ---------------------------------------------------------------- P0 리포트
def profile_report(sig_tr, sig_tm):
    print("\n" + "=" * 78)
    print("P0 시그니처 프로파일")
    print("=" * 78)
    for tag, sig in (("train", sig_tr), ("tm", sig_tm)):
        ent = sig.index.get_level_values(0)
        print(f"\n[{tag}] 투수-시즌 {len(sig):,} · 고유 투수 {ent.nunique():,}")
        print(f"  n_R>0: {(sig.n_R > 0).sum():,}  n_F>0: {(sig.n_F > 0).sum():,}  "
              f"둘 다: {((sig.n_R > 0) & (sig.n_F > 0)).sum():,}")
        cols = [c for c in ("n_R", "n_F", "n_app_R", "app_mean_R", "start_share_R",
                            "mix_fastball") if c in sig]
        print(sig[cols].describe().round(3).to_string())

    print("\n[손] train:", sig_tr.hand_raw.value_counts(normalize=True).round(4).to_dict())
    print("     tm   :", sig_tm.hand_raw.value_counts(normalize=True).round(4).to_dict())

    print("\n[1군 투수-시즌 천장] train R:", int((sig_tr.n_R > 0).sum()),
          " tm R:", int((sig_tm.n_R > 0).sum()))
    print("[등판 축 정합] train R 등판당 투구수 중앙값:",
          round(float(sig_tr.loc[sig_tr.n_R > 0, "app_mean_R"].median()), 2),
          " tm:", round(float(sig_tm.loc[sig_tm.n_R > 0, "app_mean_R"].median()), 2))
    print("[믹스 정합] fastball 평균 — train:",
          round(float(sig_tr.mix_fastball.mean()), 4),
          " tm:", round(float(sig_tm.mix_fastball.mean()), 4))


# ================================================================ P1 매칭
# 비용 축과 가중치. **요일(wR_)은 의도적으로 제외** — §3.6-1 permutation 검정의 예약 차원이다.
SCALAR_AXES = [("n_R", 1.0), ("n_app_R", 1.0), ("app_mean_R", 1.2),
               ("inn_mean_R", 0.5), ("first_inn_mean_R", 0.5),
               ("mix_fastball", 0.6), ("mix_breaking", 0.4), ("mix_offspeed", 0.4),
               ("n_F", 0.2)]
DIRECT_AXES = [("start_share_R", 0.8), ("top_R", 0.3)]      # 이미 [0,1] — 랭크 불필요
HIST_AXES = [("mR_", MONTHS, 1.0), ("iR_", INNINGS, 0.8), ("cR_", list(range(12)), 0.4)]
# ⓑ 등판단위 joint(mnR_ 월×투구수)는 실측 역효과(NO-GO 45.2%, research/15 §7) — 요일 없는
# 안전버전은 노이즈만 추가. 각도3 원안(요일 joint + 예약차원 재지정)은 다음 세션 다일 작업.
RESERVED_HIST = ("wR_", DOWS)                                # 검증 전용
LAM_TM, LAM_TR = 3.0, 1.0                                    # 시즌 존재 비대칭 페널티
BIG = 1e6
SEASONS = list(range(2019, 2025))


def _entity_tables(sig, ent_level=0):
    """(엔티티, 시즌) 시그니처 → 엔티티 목록 + 커리어 손 + 시즌별 정렬 배열 접근자."""
    ents = np.array(sorted(sig.index.get_level_values(ent_level).unique()))
    pos = {e: i for i, e in enumerate(ents)}
    hand = pd.Series(index=ents, dtype=object)
    hw = sig.groupby(level=ent_level).apply(
        lambda d: d.loc[d["n_all"].idxmax(), "hand_raw"] if len(d) else None)
    hand.loc[hw.index] = hw.values
    return ents, pos, hand


def _rank01(v, present):
    """존재 엔티티 내 분위수 랭크 [0,1] — 계통 오프셋·커버리지 결손을 자동 소거(§3.2)."""
    out = np.full(len(v), np.nan)
    idx = np.where(present)[0]
    if len(idx) <= 1:
        out[idx] = 0.5
        return out
    r = pd.Series(v[idx]).rank(pct=True).to_numpy()
    out[idx] = r
    return out


def _season_arrays(sig, ents, pos, season, hand_key):
    """한 시즌의 (엔티티×축) 배열 묶음. 랭크는 **손-시즌 내**에서 매긴다."""
    try:
        s = sig.xs(season, level=1)
    except KeyError:
        s = sig.iloc[0:0]
    n = len(ents)
    idx = np.array([pos[e] for e in s.index if e in pos])
    keep = np.array([e in pos for e in s.index])
    s = s[keep]

    present = np.zeros(n, dtype=bool)
    if len(idx):
        present[idx] = s["n_R"].to_numpy() > 0

    out = {"present": present, "hand": np.array([hand_key.get(e) for e in ents], dtype=object)}
    raw = {}
    for c in [a for a, _ in SCALAR_AXES] + [a for a, _ in DIRECT_AXES]:
        v = np.full(n, np.nan)
        if len(idx) and c in s:
            v[idx] = pd.to_numeric(s[c], errors="coerce").to_numpy()
        raw[c] = v

    # 스칼라 축: 손-시즌 내 분위수 랭크
    sides = pd.unique(out["hand"][present]) if present.any() else []
    for c, _ in SCALAR_AXES:
        r = np.full(n, np.nan)
        for sd in sides:
            m = present & (out["hand"] == sd)
            r_side = _rank01(np.nan_to_num(raw[c], nan=0.0), m)
            r[m] = r_side[m]
        out[c] = r
    for c, _ in DIRECT_AXES:
        out[c] = raw[c]

    for pre, vals, _ in HIST_AXES + [(RESERVED_HIST[0], RESERVED_HIST[1], 0.0)]:
        cols = [f"{pre}{v}" for v in vals]
        M = np.zeros((n, len(cols)))
        if len(idx) and all(c in s for c in cols):
            M[idx] = s[cols].to_numpy(dtype=float)
        out[pre] = np.sqrt(np.clip(M, 0, None))     # Hellinger용 √확률
    return out


def _pair_cost_season(A, B, weights=None, include_hist=True):
    """한 시즌의 (train n_a × tm n_b) 거리행렬 + 존재 마스크."""
    w = weights or {}
    na, nb = len(A["present"]), len(B["present"])
    D = np.zeros((na, nb), dtype=np.float32)
    for c, w0 in SCALAR_AXES + DIRECT_AXES:
        wt = w.get(c, w0)
        if wt == 0:
            continue
        a = np.nan_to_num(A[c], nan=0.5)[:, None]
        b = np.nan_to_num(B[c], nan=0.5)[None, :]
        D += np.float32(wt) * np.abs(a - b).astype(np.float32)
    if include_hist:
        for pre, _, w0 in HIST_AXES:
            wt = w.get(pre, w0)
            if wt == 0:
                continue
            bc = np.clip(A[pre] @ B[pre].T, 0.0, 1.0)
            D += np.float32(wt) * np.sqrt(1.0 - bc).astype(np.float32)
    both = A["present"][:, None] & B["present"][None, :]
    return D, both


def build_cost(sig_tr, sig_tm, hand_map, weights=None, seasons=SEASONS,
               ents=None, reserved_only=False):
    """커리어 대 커리어 비용행렬(§3.3 2단). 손 불일치는 하드 블록(BIG)."""
    (e_tr, p_tr, h_tr), (e_tm, p_tm, h_tm) = ents
    C = np.zeros((len(e_tr), len(e_tm)), dtype=np.float32)
    n_co = np.zeros_like(C)
    for s in seasons:
        A = _season_arrays(sig_tr, e_tr, p_tr, s, h_tr)
        B = _season_arrays(sig_tm, e_tm, p_tm, s, h_tm)
        if reserved_only:
            bc = np.clip(A[RESERVED_HIST[0]] @ B[RESERVED_HIST[0]].T, 0.0, 1.0)
            D = np.sqrt(1.0 - bc).astype(np.float32)
            both = A["present"][:, None] & B["present"][None, :]
        else:
            D, both = _pair_cost_season(A, B, weights)
        C += np.where(both, D, 0.0).astype(np.float32)
        n_co += both
        if not reserved_only:
            C += np.float32(LAM_TM) * (~A["present"][:, None] & B["present"][None, :])
            C += np.float32(LAM_TR) * (A["present"][:, None] & ~B["present"][None, :])

    if not reserved_only:
        hand_tr = np.array([hand_map.get(h_tr.get(e)) for e in e_tr], dtype=object)
        hand_tm = np.array([h_tm.get(e) for e in e_tm], dtype=object)
        mismatch = hand_tr[:, None] != hand_tm[None, :]
        C = np.where(mismatch, np.float32(BIG), C)
    return C, n_co


def assign(C):
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(C)
    return r, c


def _margins(C, rows, cols):
    """행·열 양방향 margin = (2위 − 1위) — Tier 컷 근거(§3.6-3)."""
    mr, mc = np.zeros(len(rows)), np.zeros(len(rows))
    for k, (i, j) in enumerate(zip(rows, cols)):
        row = C[i].copy()
        best = row[j]
        row[j] = np.inf
        mr[k] = np.min(row) - best
        col = C[:, j].copy()
        col[i] = np.inf
        mc[k] = np.min(col) - best
    return mr, mc


def _gmm_antimode(mr, fallback_q=0.25):
    """margin_row 이봉성의 antimode(두 성분 사이 밀도 골)를 Tier1 컷으로 (research/15 각도3 ⓒ).

    D-17에서 Tier1+2 확장이 refV24를 −12.0으로 뒤집은 원인 = 저마진 오매칭이 Tier1에 섞임.
    고정 25분위 대신 "확실/애매" 이봉의 골을 실측해 오매칭을 배제한다. 이봉이 뚜렷하지
    않으면(단봉·분리 약함) 기존 분위수로 안전 폴백한다.
    """
    mr = np.asarray(mr, dtype=float)
    mr = mr[np.isfinite(mr)]
    if len(mr) < 50:
        return float(np.quantile(mr, fallback_q)) if len(mr) else 0.0
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(2, random_state=0, n_init=3).fit(mr.reshape(-1, 1))
    mu = np.sort(gm.means_.ravel())
    sd = float(np.sqrt(gm.covariances_.ravel()).mean())
    if mu[1] - mu[0] < sd:                       # 분리 약함 → 이봉 아님, 분위수 폴백
        return float(np.quantile(mr, fallback_q))
    grid = np.linspace(mu[0], mu[1], 400)
    dens = gm.score_samples(grid.reshape(-1, 1))
    tau = float(grid[int(np.argmin(dens))])
    q = float((mr < tau).mean())
    print(f"    [GMM Tier컷] 이봉 μ={mu[0]:.3f}/{mu[1]:.3f} σ̄={sd:.3f} → "
          f"antimode τ={tau:.4f} (하위 {q:.1%} 배제) vs 분위수 {np.quantile(mr, fallback_q):.4f}")
    return tau


# ================================================================ P2 프로필 (N13)
# 문서 13 §4 **개정판** 우선순위: 릴리스 포인트 일관성(tm_relstd)은 강등(Wakamiya 2024 — RP 산포는
# BB/9 예측력 0), 승격된 1·2순위는 **무브먼트/구속 일관성**(공이 떠나는 방향의 반복성 대리).
MEAS = {"rel_speed": "velo", "spin_rate": "spin",
        "induced_vert_break": "ivb", "horz_break": "hb"}
GROUPS = ["fastball", "breaking", "offspeed"]
TM_FEATURES = ["tm_movestd", "tm_velostd", "tm_velo_fb",
               "tm_farm_share", "tm_spin_fb", "tm_velosep"]
MARK_TM_BEGIN = "# --- TM_PROFILE:BEGIN ---"
MARK_TM_END = "# --- TM_PROFILE:END ---"
MIN_PREFIX_N = 100          # 커리어 접두 측정 표본 하한 (미만이면 전 피처 NaN)
MIN_SEP_N = 30              # velosep 계산에 필요한 군별 하한


def _measure_cells(tm: pd.DataFrame) -> pd.DataFrame:
    """(tm_id, season, 구종군) 셀별 n·Σz·Σz². z는 **(시즌×구종군) 내 표준화** —
    장비/구장 드리프트와 레퍼토리 교락을 동시에 소거한다(문서 13 §4 서두)."""
    d = tm[tm["pitch_type_group"].isin(GROUPS)].copy()
    key = ["pitcher_trackman_id", rd.SEASON, "pitch_type_group"]
    agg = {}
    for raw, m in MEAS.items():
        g = d.groupby([rd.SEASON, "pitch_type_group"])[raw]
        z = (d[raw] - g.transform("mean")) / g.transform("std")
        d[f"z_{m}"] = z
        d[f"q_{m}"] = z * z
        agg[f"n_{m}"] = (f"z_{m}", "count")
        agg[f"s_{m}"] = (f"z_{m}", "sum")
        agg[f"q_{m}"] = (f"q_{m}", "sum")
    cells = d.groupby(key).agg(**agg).reset_index()
    rows = (tm.assign(_farm=(tm["chan"] == "F").astype(float))
              .groupby(["pitcher_trackman_id", rd.SEASON])
              .agg(n_rows=("chan", "size"), n_farm=("_farm", "sum")).reset_index())
    return cells, rows


def _prefix_features(cells, rows, season):
    """시즌 s 행에 붙일 프로필 = **s−1까지 커리어 전체**(as-of (a) 설계, 서빙 동형)."""
    c = cells[cells[rd.SEASON] < season]
    r = rows[rows[rd.SEASON] < season]
    if not len(c):
        return pd.DataFrame(columns=TM_FEATURES)

    def _pooled(sub, m):
        """셀 내 분산만 풀링(셀=시즌×구종군) — 시즌 드리프트·레퍼토리 차를 분산에 넣지 않는다."""
        n, s_, q = sub[f"n_{m}"], sub[f"s_{m}"], sub[f"q_{m}"]
        ok = n >= 2
        ss = (q - s_ * s_ / n.where(n > 0)).where(ok, 0.0)
        dof = (n - 1).where(ok, 0.0)
        return ss.groupby(sub["pitcher_trackman_id"]).sum(), dof.groupby(sub["pitcher_trackman_id"]).sum()

    out = pd.DataFrame(index=pd.Index(sorted(c["pitcher_trackman_id"].unique()),
                                      name="pitcher_trackman_id"))
    ss_v, dof_v = _pooled(c, "velo")
    out["tm_velostd"] = np.sqrt(ss_v / dof_v.where(dof_v > 0))
    ss_i, dof_i = _pooled(c, "ivb")
    ss_h, dof_h = _pooled(c, "hb")
    out["tm_movestd"] = np.sqrt((ss_i + ss_h) / (dof_i + dof_h).where((dof_i + dof_h) > 0))

    fb = c[c["pitch_type_group"] == "fastball"].groupby("pitcher_trackman_id")
    os_ = c[c["pitch_type_group"] == "offspeed"].groupby("pitcher_trackman_id")
    fb_n, fb_v = fb["n_velo"].sum(), fb["s_velo"].sum()
    out["tm_velo_fb"] = (fb_v / fb_n.where(fb_n >= MIN_SEP_N))
    fb_sn, fb_sp = fb["n_spin"].sum(), fb["s_spin"].sum()
    out["tm_spin_fb"] = (fb_sp / fb_sn.where(fb_sn >= MIN_SEP_N))
    os_n, os_v = os_["n_velo"].sum(), os_["s_velo"].sum()
    out["tm_velosep"] = out["tm_velo_fb"] - (os_v / os_n.where(os_n >= MIN_SEP_N))

    rr = r.groupby("pitcher_trackman_id")[["n_rows", "n_farm"]].sum()
    out["tm_farm_share"] = rr["n_farm"] / rr["n_rows"].where(rr["n_rows"] > 0)

    tot_n = c.groupby("pitcher_trackman_id")["n_velo"].sum()
    out.loc[tot_n.reindex(out.index).fillna(0) < MIN_PREFIX_N, TM_FEATURES] = np.nan
    return out[TM_FEATURES]


def build_pitcher_profile(matches: pd.DataFrame, tm: pd.DataFrame, tiers=(1,),
                          seasons=range(2020, 2026)) -> dict:
    """{pid*10000+season: [6피처]} — Tier 필터 통과 쌍만. Tier2/3은 미등재(=NaN, HGB 결측 분기)."""
    sel = matches[matches["tier"].isin(tiers)]
    pid_of = dict(zip(sel["tm_id"], sel["pitcher_id"]))
    tm_sel = tm[tm["pitcher_trackman_id"].isin(pid_of)]
    cells, rows = _measure_cells(tm_sel)
    prof = {}
    for s in seasons:
        f = _prefix_features(cells, rows, s)
        for tid, vec in zip(f.index, f.to_numpy()):
            if np.all(np.isnan(vec)):
                continue
            prof[int(pid_of[tid]) * 10000 + int(s)] = [float(v) for v in vec]
    return prof


def emit_serve_literal(profile: dict, meta: dict, decimals=4) -> str:
    keys = sorted(profile)
    idx = ", ".join(f"{k}: {i}" for i, k in enumerate(keys))
    # NaN은 `np.nan`으로 적는다 — SERVE 블록은 numpy만 import하므로 이게 유일하게 안전한 표기다.
    mat = ", ".join("[" + ", ".join("np.nan" if np.isnan(v) else str(round(v, decimals))
                                    for v in profile[k]) + "]" for k in keys)
    return "\n".join([
        MARK_TM_BEGIN,
        "# sweep/trackman.py --profile 이 재생성한다 — 손으로 편집 금지.",
        "# (pid*10000+season) → 인덱스, 인덱스 → [tm_movestd, tm_velostd, tm_velo_fb,",
        "#  tm_farm_share, tm_spin_fb, tm_velosep] (as-of: 시즌 s ← ≤s−1 트랙맨 커리어).",
        f"TM_PROFILE_META = {meta!r}",
        f"TM_PROFILE_IDX = {{{idx}}}",
        f"TM_PROFILE_MAT = [{mat}]",
        MARK_TM_END])


def splice(literal, begin, end, path=None):
    import re
    p = Path(path or HERE / "real_data.py")
    src = p.read_text(encoding="utf-8")
    pat = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    if not pat.search(src):
        sys.exit(f"마커({begin}...{end})를 {p}에서 찾지 못했다 — SERVE 블록에 먼저 추가할 것")
    p.write_text(pat.sub(lambda _: literal, src), encoding="utf-8")
    print(f"스플라이스 완료: {p} ({len(literal):,} bytes)")


def run_profile(tiers=(1,), decimals=4, do_splice=True):
    import json
    mf = OUT_DIR / "matches.csv"
    if not mf.exists():
        sys.exit("matches.csv 없음 — 먼저 `python sweep/trackman.py --match`")
    matches = pd.read_csv(mf)
    print(f">> Tier{tiers} 쌍 {int(matches.tier.isin(tiers).sum())}/{len(matches)}")
    print(">> trackman 로드")
    tm = load_trackman()
    prof = build_pitcher_profile(matches, tm, tiers=tiers)
    arr = np.array([v for v in prof.values()], dtype=float)
    print(f">> 프로필 {len(prof):,} 엔트리 (pid×season)")
    print(pd.DataFrame(arr, columns=TM_FEATURES).describe().round(3).to_string())

    ver = f"tm1-t{''.join(map(str, tiers))}-n{len(prof)}"
    meta = {"version": ver, "tiers": list(tiers), "n": len(prof),
            "features": TM_FEATURES, "asof": "season s <- <=s-1 career"}
    lit = emit_serve_literal(prof, meta, decimals=decimals)
    print(f">> 리터럴 {len(lit):,} bytes")
    (OUT_DIR / "tm_profile.json").write_text(
        json.dumps({"meta": meta, "profile": prof}, ensure_ascii=False), encoding="utf-8")
    if do_splice:
        splice(lit, MARK_TM_BEGIN, MARK_TM_END)
        print(f"⚠ real_data.py의 SERVE_VERSION을 '{ver}' 를 포함하도록 갱신할 것 "
              "(캐시 무효화 — 문서 13 §5 함정 3)")
    return prof


# ---------------------------------------------------------------- P1 드라이버 + 4중 검증
def run_match(sig_tr, sig_tm, nperm=2000, seed=0):
    import json
    rng = np.random.default_rng(seed)
    ents_tr = _entity_tables(sig_tr)
    ents_tm = _entity_tables(sig_tm)
    ents = (ents_tr, ents_tm)
    e_tr, e_tm = ents_tr[0], ents_tm[0]
    print(f">> 엔티티 train {len(e_tr)} × tm {len(e_tm)}")

    # --- ① 손 극성 자가검증 (§3.5): 두 극성으로 전체 할당을 풀어 총비용 비교
    print("\n[1] 손 극성 자가검증")
    pol = {}
    for name, hmap in (("2=Right", {2: "Right", 1: "Left"}), ("2=Left", {2: "Left", 1: "Right"})):
        C, n_co = build_cost(sig_tr, sig_tm, hmap, ents=ents)
        r, c = assign(C)
        tot = float(C[r, c].sum())
        n_big = int((C[r, c] >= BIG / 2).sum())
        pol[name] = (tot, n_big, hmap)
        print(f"    {name}: 총비용 {tot:,.1f}  (BIG 강제 {n_big})")
    best_pol = min(pol, key=lambda k: pol[k][0])
    ratio = max(pol[k][0] for k in pol) / max(min(pol[k][0] for k in pol), 1e-9)
    print(f"    → 채택 {best_pol}  (열세/우세 비 {ratio:.2f}× — 1.5× 미만이면 시그니처 붕괴 경고)")
    hand_map = pol[best_pol][2]

    # --- ② 커리어 대 커리어 매칭 (주 결과)
    print("\n[2] 커리어 Hungarian")
    C, n_co = build_cost(sig_tr, sig_tm, hand_map, ents=ents)
    rows, cols = assign(C)
    mr, mc = _margins(C, rows, cols)
    co = n_co[rows, cols]
    m = pd.DataFrame({"pitcher_id": e_tr[rows], "tm_id": e_tm[cols],
                      "cost": C[rows, cols], "n_co": co,
                      "cost_per_season": C[rows, cols] / np.maximum(co, 1),
                      "margin_row": mr, "margin_col": mc})
    m["forced"] = m["cost"] >= BIG / 2
    print(f"    쌍 {len(m)} · 공존시즌 있는 쌍 {int((m.n_co>0).sum())} · BIG 강제 {int(m.forced.sum())}")
    print(f"    cost/season 중앙값 {m.loc[m.n_co>0,'cost_per_season'].median():.3f} · "
          f"margin_row 중앙값 {m.margin_row.median():.3f}")

    # --- ③ 예약 차원(요일) permutation 검정 — 주 판정 (§3.6-1)
    print("\n[3] 요일 permutation 검정 (비용에 쓰지 않은 축)")
    R, R_co = build_cost(sig_tr, sig_tm, hand_map, ents=ents, reserved_only=True)
    hand_tr = np.array([hand_map.get(ents_tr[2].get(e)) for e in e_tr], dtype=object)
    hand_tm = np.array([ents_tm[2].get(e) for e in e_tm], dtype=object)
    ok = (co > 0) & (~m.forced.to_numpy())
    ri, cj = rows[ok], cols[ok]
    stat = lambda a, b: float(np.mean(R[a, b] / np.maximum(R_co[a, b], 1)))
    obs = stat(ri, cj)

    # 블록 정의 2종. 손 블록만으로는 **팀 수준 정합**밖에 증명하지 못한다 —
    # 같은 팀 동료는 일정을 공유해 요일 분포가 서로 비슷하기 때문이다.
    # 팀-손 블록 내 순열이 팀 내 **투수 식별**을 분리 검증하는 진짜 판정이다
    # (선발은 5일 로테이션이라 팀 내에서도 요일 분포가 개인적이다).
    team_of = (sig_tr.reset_index().sort_values("n_all")
               .groupby("pitcher_id")["team_raw"].last())
    tm_team_of = np.array([str(team_of.get(e, -1)) for e in e_tr], dtype=object)
    blocks = {"손": hand_tr[ri],
              "팀×손": np.array([f"{a}|{b}" for a, b in zip(tm_team_of[ri], hand_tr[ri])],
                                dtype=object)}
    res = {}
    for bname, blk in blocks.items():
        null = np.empty(nperm)
        for t in range(nperm):
            perm_c = cj.copy()
            for sd in np.unique(blk):
                idx = np.where(blk == sd)[0]
                if len(idx) > 1:
                    perm_c[idx] = cj[rng.permutation(idx)]
            null[t] = stat(ri, perm_c)
        pv = (1 + int((null <= obs).sum())) / (1 + nperm)
        zv = (np.mean(null) - obs) / (np.std(null) + 1e-12)
        res[bname] = (pv, float(zv), float(null.mean()), float(null.std()))
        print(f"    [{bname} 블록] 관측 {obs:.4f} · 귀무 {null.mean():.4f}±{null.std():.4f} · "
              f"z={zv:.1f} · p={pv:.5f}")
    p, z = res["팀×손"][0], res["팀×손"][1]      # 주 판정 = 엄격한 쪽
    print(f"    주 판정 = 팀×손 블록 (팀 정합이 아닌 투수 식별을 검증)  p={p:.5f}  [kill: ≥ 0.01]")

    # --- ④ 방법 간 합치: 시즌별 독립 Hungarian 6회 (§3.6-2)
    print("\n[4] 방법 간 합치 (커리어 vs 시즌별 독립)")
    per_season = {}
    for s in SEASONS:
        Cs, _ = build_cost(sig_tr, sig_tm, hand_map, ents=ents, seasons=[s])
        A = _season_arrays(sig_tr, e_tr, ents_tr[1], s, ents_tr[2])["present"]
        B = _season_arrays(sig_tm, e_tm, ents_tm[1], s, ents_tm[2])["present"]
        ia, ib = np.where(A)[0], np.where(B)[0]
        if not len(ia) or not len(ib):
            continue
        rr, cc = assign(Cs[np.ix_(ia, ib)])
        for a, b in zip(ia[rr], ib[cc]):
            per_season.setdefault(e_tr[a], []).append(e_tm[b])
    agree, tot_multi = [], 0
    mode_map = {}
    for pid, lst in per_season.items():
        vals, cnts = np.unique(lst, return_counts=True)
        mode_map[pid] = vals[np.argmax(cnts)]
        if len(lst) >= 2:
            tot_multi += 1
            agree.append(mode_map[pid])
    m["season_mode"] = m["pitcher_id"].map(mode_map)
    m["n_season_obs"] = m["pitcher_id"].map({k: len(v) for k, v in per_season.items()}).fillna(0)
    multi = m[m.n_season_obs >= 2]
    rate = float((multi.season_mode == multi.tm_id).mean()) if len(multi) else float("nan")
    print(f"    다시즌 투수 {len(multi)}명 최빈 tm_id 일치율 {rate:.1%}   [kill: < 80%]")
    m["agree"] = (m.season_mode == m.tm_id) | (m.n_season_obs < 2)

    # --- ⑤ 시대 분할 안정성 + 가중치 지터 (§3.6-4)
    print("\n[5] 시대 분할 · 가중치 지터")
    era_pairs = {}
    for tag, ss in (("early", [2019, 2020, 2021]), ("late", [2022, 2023, 2024])):
        Ce, nco_e = build_cost(sig_tr, sig_tm, hand_map, ents=ents, seasons=ss)
        rr, cc = assign(Ce)
        era_pairs[tag] = {e_tr[i]: (e_tm[j], nco_e[i, j]) for i, j in zip(rr, cc)}
    both_era = [pid for pid in e_tr
                if era_pairs["early"].get(pid, (None, 0))[1] > 0
                and era_pairs["late"].get(pid, (None, 0))[1] > 0]
    era_rec = float(np.mean([era_pairs["early"][p][0] == era_pairs["late"][p][0]
                             for p in both_era])) if both_era else float("nan")
    print(f"    양 시대 걸친 투수 {len(both_era)}명 · 시대간 쌍 재현율 {era_rec:.1%}")

    base_pair = dict(zip(m.pitcher_id, m.tm_id))
    surv = np.zeros(len(m))
    NJIT = 20
    for t in range(NJIT):
        w = {k: v * float(np.exp(rng.normal(0, 0.25)))
             for k, v in [(a, b) for a, b in SCALAR_AXES + DIRECT_AXES]
             + [(a, w0) for a, _, w0 in HIST_AXES]}
        Cj, _ = build_cost(sig_tr, sig_tm, hand_map, weights=w, ents=ents)
        rr, cc = assign(Cj)
        pair = dict(zip(e_tr[rr], e_tm[cc]))
        surv += np.array([1.0 if pair.get(p) == q else 0.0
                          for p, q in zip(m.pitcher_id, m.tm_id)])
    m["stability"] = surv / NJIT
    print(f"    지터 {NJIT}회 쌍 생존율: 중앙값 {m.stability.median():.2f} · "
          f"≥0.9 비율 {(m.stability >= 0.9).mean():.1%}")

    # --- ⑥ 팀 대조표 (매칭에서 유도) — 독립 구조 검증
    print("\n[6] 팀 대조표 (팀 정보를 비용에 쓰지 않았다 → 블록 구조가 나오면 매칭이 진짜다)")
    tt = sig_tr.reset_index()[["pitcher_id", "season", "team_raw", "n_R"]]
    mt = sig_tm.reset_index()[["pitcher_trackman_id", "season", "team_raw", "n_R"]].rename(
        columns={"pitcher_trackman_id": "tm_id", "team_raw": "tm_team", "n_R": "n_R_tm"})
    j = (tt[tt.n_R > 0].merge(m[["pitcher_id", "tm_id", "stability"]], on="pitcher_id")
         .merge(mt[mt.n_R_tm > 0], on=["tm_id", "season"]))
    ct = pd.crosstab(j["team_raw"], j["tm_team"])
    purity = float((ct.max(axis=1) / ct.sum(axis=1)).mean())
    print(ct.to_string())
    print(f"    팀 순도(행 최빈 비율 평균) {purity:.1%}   [무작위 기대 ≈ 10%]")
    team_map = ct.idxmax(axis=1).to_dict()

    # --- ⑦ Tier + 커버리지 (§3.7–3.8)
    tau = _gmm_antimode(m.loc[m.n_co > 0, "margin_row"].to_numpy())
    m["tier"] = 3
    m.loc[(m.n_co > 0) & (~m.forced) & (m.stability >= 0.6), "tier"] = 2
    m.loc[(m.n_co > 0) & (~m.forced) & m.agree & (m.stability >= 0.9)
          & (m.margin_row >= tau), "tier"] = 1
    n_rows = sig_tr.groupby(level=0)["n_all"].sum()
    m["train_rows"] = m["pitcher_id"].map(n_rows)
    cov_ent = float((m.tier == 1).mean())
    cov_row = float(m.loc[m.tier == 1, "train_rows"].sum() / m["train_rows"].sum())
    print(f"\n[7] Tier1 {int((m.tier==1).sum())}/{len(m)} = {cov_ent:.1%} "
          f"· 행 가중 커버리지 {cov_row:.1%}   [kill: Tier1 <35% 또는 행가중 <50%]")
    print(m.tier.value_counts().sort_index().to_string())

    # --- 판정
    verdict = {"polarity": best_pol, "polarity_ratio": ratio,
               "perm_hand_p": res["손"][0], "perm_hand_z": res["손"][1],
               "perm_p": p, "perm_z": float(z),
               "method_agree": rate, "era_recall": era_rec,
               "stability_med": float(m.stability.median()),
               "team_purity": purity, "tier1_ent": cov_ent, "tier1_row": cov_row,
               "n_pairs": int(len(m)), "tau_margin": tau}
    kills = []
    if not (p < 0.01):
        kills.append(f"permutation p={p:.4f} ≥ 0.01")
    if not (rate >= 0.80):
        kills.append(f"방법합치 {rate:.1%} < 80%")
    if not (cov_ent >= 0.35 or cov_row >= 0.50):
        kills.append(f"Tier1 {cov_ent:.1%} / 행가중 {cov_row:.1%} 미달")
    if ratio < 1.5:
        kills.append(f"손 극성 비 {ratio:.2f}× < 1.5×")
    verdict["kills"] = kills
    verdict["go"] = not kills
    print("\n" + "=" * 78)
    print("GO/NO-GO: " + ("**GO** — 전 kill criteria 통과" if not kills
                          else "**NO-GO** — " + " / ".join(kills)))
    print("=" * 78)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m.to_csv(OUT_DIR / "matches.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "self_consistency.json").write_text(
        json.dumps({**verdict, "team_map": {str(k): v for k, v in team_map.items()}},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {OUT_DIR/'matches.csv'} · {OUT_DIR/'self_consistency.json'}")
    return m, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sig", action="store_true", help="P0: 시그니처 생성 + 캐시 + 리포트")
    ap.add_argument("--match", action="store_true", help="P1: 매칭 + 4중 self-consistency")
    ap.add_argument("--profile", action="store_true", help="P2: 프로필 6피처 → SERVE 리터럴 스플라이스")
    ap.add_argument("--tiers", default="1", help="프로필에 채택할 Tier (예: 1 또는 1,2)")
    ap.add_argument("--no-splice", action="store_true", help="리터럴 생성만, real_data.py 미수정")
    ap.add_argument("--nperm", type=int, default=2000, help="permutation 검정 반복수")
    ap.add_argument("--force", action="store_true", help="캐시 무시하고 재생성")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    f_tr, f_tm = OUT_DIR / "sig_train.pkl", OUT_DIR / "sig_tm.pkl"

    if args.sig:
        if args.force or not (f_tr.exists() and f_tm.exists()):
            print(">> train.csv 로드")
            tr = rd.load_train()
            print(">> 등판 복원")
            tr = add_appearances(tr)
            n_app = int(tr.groupby(["pitcher_id", rd.SEASON])["app_no"].nunique().sum())
            n_r = int(tr[tr.game_type == "R"].groupby("pitcher_id")["app_no"].nunique().sum())
            print(f"   등판 {n_app:,} (R 채널 {n_r:,} — 물리 기대 38~43k)")
            print(">> train 시그니처")
            sig_tr = season_signatures_train(tr)
            sig_tr.to_pickle(f_tr)
            del tr
            print(">> trackman_history.csv 로드")
            tm = load_trackman()
            print(f"   유효 {len(tm):,}행 (제외 후) · 채널 R {int((tm.chan=='R').sum()):,} "
                  f"F {int((tm.chan=='F').sum()):,}")
            print(">> tm 시그니처")
            sig_tm = season_signatures_tm(tm)
            sig_tm.to_pickle(f_tm)
        else:
            print(f">> 캐시 사용: {f_tr.name}, {f_tm.name} (--force로 재생성)")
            sig_tr, sig_tm = pd.read_pickle(f_tr), pd.read_pickle(f_tm)
        profile_report(sig_tr, sig_tm)

    if args.match:
        if not (f_tr.exists() and f_tm.exists()):
            sys.exit("시그니처 캐시 없음 — 먼저 `python sweep/trackman.py --sig`")
        sig_tr, sig_tm = pd.read_pickle(f_tr), pd.read_pickle(f_tm)
        run_match(sig_tr, sig_tm, nperm=args.nperm)

    if args.profile:
        run_profile(tiers=tuple(int(t) for t in args.tiers.split(",")),
                    do_splice=not args.no_splice)


if __name__ == "__main__":
    main()
