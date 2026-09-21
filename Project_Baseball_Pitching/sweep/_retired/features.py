"""
누수 안전(as-of) 피처 레이어.

원칙 (05 blueprint 1장 / 08 증류계획과 공유):
  - 모든 이력 통계는 '현재 투구 이전' 데이터로만 계산 → groupby.shift(1) 후 expanding/rolling.
  - 예측 대상 투구 자신의 결과/실측은 절대 입력하지 않음.
  - 엔티티(투수/타자)는 as-of 타깃 통계 = 누수안전 타깃 인코딩 + 베이즈 수축.

시간분할 정합 (label_cutoff_season):
  - 실대회는 val 시즌(2025) target 전면 비공개(배치 제출). 따라서 val 시즌 행의 **target-파생**
    as-of 통계는 같은 시즌의 앞선 target을 쓰면 안 됨(누수). cutoff가 주어지면 target-파생 통계를
    **cutoff 미만(train)으로만 계산**하고 각 엔티티의 train 경계 집계를 val 행에 이월(freeze).
  - target 무관 피처(맥락, F18 릴리스 SD, F19 pitch_no)는 val 시즌 입력이라 그대로 사용.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from config import DataSpec


# ---------- cutoff(시간분할) 유틸 ----------
def _train_agg(df, key, target, season_col, cutoff, roll=None):
    """엔티티(key)별 train(season<cutoff) target 집계(합, 수). roll이면 마지막 roll개."""
    tr = df.loc[df[season_col] < cutoff, [key, target]]
    g = tr.groupby(key, sort=False)[target]
    if roll is None:
        return g.sum(), g.count()
    return (g.apply(lambda s: s.tail(roll).sum()),
            g.apply(lambda s: s.tail(roll).count()))


def _freeze_val_rows(series, entity, season, cutoff, frozen_map, fill):
    """season>=cutoff(val) 행을 frozen_map[entity]로 대체(미등장→fill). 세 Series 동일 index."""
    val = (season >= cutoff).to_numpy()
    if val.any():
        series = series.copy()
        series.loc[val] = entity[val].map(frozen_map).fillna(fill).to_numpy()
    return series


# ---------- as-of 통계(target-파생; cutoff로 val 동결) ----------
def _asof_rate(df, key, target, roll=None, shrink_k=50.0, global_mean=0.5,
               season_col=None, cutoff=None):
    """key(투수/타자)별 target의 as-of 평균. roll=None이면 expanding, 아니면 last-N rolling.
    베이즈 수축: (합 + k*prior)/(n + k). cutoff 주어지면 val 행을 train 경계 집계로 동결."""
    g = df.groupby(key, sort=False)[target]
    shifted_sum = g.transform(lambda s: s.shift(1).expanding().sum() if roll is None
                              else s.shift(1).rolling(roll, min_periods=1).sum())
    shifted_cnt = g.transform(lambda s: s.shift(1).expanding().count() if roll is None
                              else s.shift(1).rolling(roll, min_periods=1).count())
    rate = ((shifted_sum + shrink_k * global_mean) / (shifted_cnt + shrink_k)).fillna(global_mean)
    cnt = shifted_cnt.fillna(0.0)
    if cutoff is not None and season_col is not None:
        fs, fn = _train_agg(df, key, target, season_col, cutoff, roll)
        frate = (fs + shrink_k * global_mean) / (fn + shrink_k)
        season, entity = df[season_col], df[key]
        rate = _freeze_val_rows(rate, entity, season, cutoff, frate, global_mean)
        cnt = _freeze_val_rows(cnt, entity, season, cutoff, fn, 0.0)
    return rate, cnt


def _asof_sum_cnt(df, key, target, season_col=None, cutoff=None):
    """key별 as-of(shift(1) expanding) 성공 합·시행 수. F17 사후분산·log5 재료.
    cutoff 주어지면 val 행을 train 전체 합/수로 동결."""
    g = df.groupby(key, sort=False)[target]
    s = g.transform(lambda x: x.shift(1).expanding().sum()).fillna(0.0)
    n = g.transform(lambda x: x.shift(1).expanding().count()).fillna(0.0)
    if cutoff is not None and season_col is not None:
        fs, fn = _train_agg(df, key, target, season_col, cutoff, None)
        season, entity = df[season_col], df[key]
        s = _freeze_val_rows(s, entity, season, cutoff, fs, 0.0)
        n = _freeze_val_rows(n, entity, season, cutoff, fn, 0.0)
    return s, n


def _beta_posterior_var(s, n, shrink_k, gm):
    """F17: 베타-이항 사후 Beta(a,b) 분산 = ab / ((a+b)^2 (a+b+1)).
    시행 적을수록(콜드스타트) 분산↑ = 불확실성 피처. a=합+k*gm, b=(n-합)+k*(1-gm)."""
    a = s + shrink_k * gm
    b = (n - s) + shrink_k * (1.0 - gm)
    tot = a + b
    return (a * b) / (tot * tot * (tot + 1.0))


def _asof_group_rate(df, group_vals, target, shrink_k, gm, season_col=None, cutoff=None):
    """임의 그룹(예: 카운트 버킷)별 as-of 리그 성공률(shift(1) expanding). F15 그리드 재료.
    df는 이미 (season,game,order) 정렬 → expanding은 전역 시간순 = 누수 안전.
    cutoff 주어지면 val 행을 그룹별 train 집계로 동결."""
    gser = pd.Series(np.asarray(group_vals), index=df.index)
    tmp = pd.DataFrame({"g": gser, "y": df[target].to_numpy()}, index=df.index)
    grp = tmp.groupby("g", sort=False)["y"]
    s = grp.transform(lambda x: x.shift(1).expanding().sum()).fillna(0.0)
    n = grp.transform(lambda x: x.shift(1).expanding().count()).fillna(0.0)
    rate = (s + shrink_k * gm) / (n + shrink_k)
    if cutoff is not None and season_col is not None:
        tr = tmp.loc[(df[season_col] < cutoff).to_numpy()]
        ag = tr.groupby("g", sort=False)["y"]
        frate = (ag.sum() + shrink_k * gm) / (ag.count() + shrink_k)
        rate = _freeze_val_rows(rate, gser, df[season_col], cutoff, frate, gm)
    return rate.to_numpy()


def _asof_rate_masked(df, key, target, mask, shrink_k, gm, season_col=None, cutoff=None):
    """key별로 (과거 & mask) 행에 한정한 target의 as-of 수축 평균. F27(압박 민감도) 재료.
    mask는 현재 행 기준 조건(예: 고LI) — shift(1)로 현재 행 자신은 제외. cutoff freeze 지원."""
    m = pd.Series(np.asarray(mask, dtype=float), index=df.index)
    ym = df[target].astype(float) * m
    g_y = ym.groupby(df[key], sort=False)
    g_m = m.groupby(df[key], sort=False)
    s = g_y.transform(lambda x: x.shift(1).expanding().sum()).fillna(0.0)
    n = g_m.transform(lambda x: x.shift(1).expanding().sum()).fillna(0.0)
    rate = (s + shrink_k * gm) / (n + shrink_k)
    if cutoff is not None and season_col is not None:
        tr = (df[season_col] < cutoff).to_numpy()
        fs = ym[tr].groupby(df.loc[tr, key]).sum()
        fn = m[tr].groupby(df.loc[tr, key]).sum()
        frate = (fs + shrink_k * gm) / (fn + shrink_k)
        rate = _freeze_val_rows(rate, df[key], df[season_col], cutoff, frate, gm)
    return rate


def _rolling_std_asof(df, key, col, roll=30):
    """key별 col의 as-of 롤링 표준편차(shift(1) → 현재 투구 실측 제외). F18 미니 Kirby.
    col은 릴리스 실측(target 아님) → cutoff 불필요(val 시즌 입력 사용 허용)."""
    g = df.groupby(key, sort=False)[col]
    sd = g.transform(lambda s: s.shift(1).rolling(roll, min_periods=3).std())
    return sd.fillna(sd.median() if sd.notna().any() else 0.0)


def default_prior_fn(df: pd.DataFrame, spec: DataSpec, dim: int = 4):
    """F13 stub: '큰 LLM이 오프라인에서 뽑아둔 콜드스타트 사전확률·의미 임베딩'을 흉내낸다.

    실제 파이프라인에선 이 함수를 **캐시된 LLM 산출물 조회**로 교체한다
    (엔티티/맥락 → 사전확률·임베딩; 추론 시 LLM 호출 0). 여기서는:
      - 누수 0: target/현재 투구 실측을 절대 참조하지 않음. id + pre-release 맥락만.
      - 결정론적: id 해시 → 안정된 pseudo-skill(투수/타자) → 맥락 보정 → 사전확률.
    합성 데이터에선 예측력이 약할 수 있음(정상 — 배관·콜드스타트 층화 검증용 stub).
    반환: DataFrame[llm_prior, llm_emb_0..dim-1] (df와 같은 행 순서).
    id만 쓰므로 cutoff 불필요(target 미참조).
    """
    def _hash01(vals, salt):
        h = ((vals.astype(np.int64) + 1) * 2654435761 + salt) & 0xFFFFFFFF
        return (h % 100000) / 100000.0  # [0,1) 안정 해시

    p_skill = _hash01(df[spec.pitcher_id].to_numpy(), 12345)     # 투수 pseudo-제구력
    b_skill = _hash01(df[spec.batter_id].to_numpy(), 67890)      # 타자 pseudo-난이도
    # pre-release 맥락 보정(3볼/2스트 압박) — 실측 아님
    ctx = -0.15 * df["balls"].to_numpy() - 0.10 * (df["strikes"].to_numpy() == 2)
    logit = 1.2 * (p_skill - 0.5) - 0.6 * (b_skill - 0.5) + ctx
    prior = 1.0 / (1.0 + np.exp(-logit))
    out = {"llm_prior": prior}
    # '의미 임베딩' 축소본(stub): id 해시 기반 저차원 좌표
    for k in range(dim):
        out[f"llm_emb_{k}"] = _hash01(df[spec.pitcher_id].to_numpy(), 1000 + k) - 0.5
    return pd.DataFrame(out, index=df.index)


def build_features(df: pd.DataFrame, spec: DataSpec,
                   include_llm_prior: bool = False, prior_fn=None,
                   rich: bool = False, ctx: bool = False, include_ids: bool = False,
                   label_cutoff_season=None):
    """반환: (X: DataFrame, y: Series, meta: DataFrame[season, game_id, pitcher_id, batter_id, order])
    X는 pre-release + as-of 피처만 포함.
    include_llm_prior=True면 F13(LLM 콜드스타트 사전확률·임베딩) 컬럼을 추가한다.
    rich=True면 F15(카운트 그리드)·F17(사후분산)·F18(릴리스 SD)·F19(피로) 블록을 추가한다.
    include_ids=True면 원본 투수/타자 ID 정수 컬럼을 붙인다(M24/M25 랜덤효과용; 트리엔 사용 안 함).
    label_cutoff_season(=val 시즌)이 주어지면 **target-파생 as-of 피처를 그 시즌부터 동결**
    (train 경계 집계 이월) → 배치제출 시간분할 정합(누수 0)."""
    df = df.sort_values([spec.season, spec.game_id, spec.order]).reset_index(drop=True)
    y = df[spec.target].astype(int)
    gm = float(y.mean())
    sc, cut = spec.season, label_cutoff_season

    feats = {}
    # 1) 맥락/중요도 (그대로 사용 = pre-release, target 무관)
    for c in spec.context:
        feats[c] = df[c].astype(float)
    feats["ahead"] = (df["balls"] < df["strikes"]).astype(float)
    feats["two_strike"] = (df["strikes"] == 2).astype(float)
    feats["three_ball"] = (df["balls"] == 3).astype(float)
    feats["high_leverage"] = (df["leverage"] > 1.5).astype(float)

    # 2) as-of 이력 (투수) — 지배적 신호. target-파생 → cutoff 동결.
    feats["p_asof_rate"], feats["p_asof_n"] = _asof_rate(df, spec.pitcher_id, spec.target, None, 50, gm, sc, cut)
    feats["p_asof_rate10"], _ = _asof_rate(df, spec.pitcher_id, spec.target, 10, 5, gm, sc, cut)
    feats["p_asof_rate50"], _ = _asof_rate(df, spec.pitcher_id, spec.target, 50, 20, gm, sc, cut)

    # 3) as-of 이력 (타자) — 상대 난이도. target-파생 → cutoff 동결.
    feats["b_asof_rate"], feats["b_asof_n"] = _asof_rate(df, spec.batter_id, spec.target, None, 50, gm, sc, cut)

    # 4) 콜드스타트 플래그(히스토리 얕음) — F13/증류 arm의 층화 평가용
    feats["p_coldstart"] = (feats["p_asof_n"] < 30).astype(float)

    X = pd.DataFrame(feats).astype(float)
    X["p_asof_n"] = np.log1p(X["p_asof_n"])
    X["b_asof_n"] = np.log1p(X["b_asof_n"])

    # 5) F13: LLM 콜드스타트 사전확률·임베딩(오프라인 캐시 흉내). 추론 시 LLM 0. (target 무관)
    if include_llm_prior:
        pf = prior_fn or default_prior_fn
        f13 = pf(df, spec).reset_index(drop=True)
        for c in f13.columns:
            X[c] = f13[c].astype(float).values

    # 6) rich 블록(09 신규 arm): F15/F17/F18/F19. target-파생은 cutoff 동결.
    if rich:
        _add_rich_features(df, spec, X, gm, cut)

    # 6.5) ctx 블록(10 신규 arm): F24~F33 맥락·환경·생체. target-파생은 cutoff 동결.
    if ctx:
        _add_ctx_features(df, spec, X, gm, cut)

    # 7) 원본 ID(M24/M25 랜덤효과·F31 era 가중용). 트리 피처가 아니라 별도 컬럼으로만 전달.
    if include_ids:
        X["_pid"] = df[spec.pitcher_id].to_numpy()
        X["_bid"] = df[spec.batter_id].to_numpy()
        X["_season"] = df[spec.season].to_numpy()

    meta = df[[spec.season, spec.game_id, spec.pitcher_id, spec.batter_id, spec.order]].copy()
    return X, y, meta


def _add_rich_features(df: pd.DataFrame, spec: DataSpec, X: pd.DataFrame, gm: float, cut=None):
    """09 신규 피처 블록을 X에 in-place 추가. target-파생(F15/F17/F19 일부)은 cut로 val 동결,
    target 무관(F18 릴리스 SD, F19 pitch_no)은 그대로."""
    sc = spec.season
    # F15: 카운트(볼×스트라이크 12버킷) 조건부 as-of 리그 성공률 그리드 (Location+/PitchingBot 축소판)
    count_bucket = (df["balls"].to_numpy() * 3 + df["strikes"].to_numpy())
    X["f15_count_grid"] = _asof_group_rate(df, count_bucket, spec.target, 200.0, gm, sc, cut)

    # F17: 베타-이항 사후분산(불확실성). 투수/타자 각각. 시행 적을수록↑ = 콜드스타트 신호.
    ps, pn = _asof_sum_cnt(df, spec.pitcher_id, spec.target, sc, cut)
    bs, bn = _asof_sum_cnt(df, spec.batter_id, spec.target, sc, cut)
    X["f17_p_postvar"] = _beta_posterior_var(ps, pn, 50.0, gm).to_numpy()
    X["f17_b_postvar"] = _beta_posterior_var(bs, bn, 50.0, gm).to_numpy()

    # F18: 미니 Kirby Index — 투수별 릴리스 좌표/각도의 as-of 롤링 SD(일관성 = 제구력 프록시).
    #      릴리스는 실측(target 아님) → val 시즌 값 사용 허용(cutoff 무관). 컬럼 있을 때만 on.
    rel_cols = [c for c in getattr(spec, "release", []) if c in df.columns]
    for c in rel_cols:
        X[f"f18_sd_{c}"] = _rolling_std_asof(df, spec.pitcher_id, c, roll=30).to_numpy()

    # F19: 피로·행동조정. 등판 내 투구번호(=count, target 무관) + 직전 결과·최근 실패율(target-파생 → 동결).
    if getattr(spec, "pitch_no", None) and spec.pitch_no in df.columns:
        pno = df[spec.pitch_no].to_numpy().astype(float)
    else:  # pitch_no 없으면 (game,pitcher) as-of 누적 투구수로 파생(target 무관)
        pno = df.groupby([spec.game_id, spec.pitcher_id], sort=False).cumcount().to_numpy().astype(float)
    X["f19_pitch_no"] = pno
    X["f19_pitch_no_sq"] = pno * pno            # 비단조(TTO) 근사

    # 직전 투구 결과(과거) — target-파생. val 행은 엔티티의 마지막 train 결과로 동결.
    prev = df.groupby(spec.pitcher_id, sort=False)[spec.target].shift(1).fillna(gm)
    fail = (1 - df[spec.target]).astype(float)
    recent_fail = (fail.groupby(df[spec.pitcher_id], sort=False)
                   .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())).fillna(1.0 - gm)
    if cut is not None:
        tr = df.loc[df[sc] < cut]
        last_succ = tr.groupby(spec.pitcher_id, sort=False)[spec.target].last()
        last_fail5 = (1 - tr.groupby(spec.pitcher_id, sort=False)[spec.target]
                      .apply(lambda s: s.tail(5).mean()))
        season, entity = df[sc], df[spec.pitcher_id]
        prev = _freeze_val_rows(prev, entity, season, cut, last_succ, gm)
        recent_fail = _freeze_val_rows(recent_fail, entity, season, cut, last_fail5, 1.0 - gm)
    X["f19_prev_success"] = prev.to_numpy()
    X["f19_recent_fail"] = recent_fail.to_numpy()


def _add_ctx_features(df: pd.DataFrame, spec: DataSpec, X: pd.DataFrame, gm: float, cut=None):
    """10 조사 ctx 블록(F24~F33)을 X에 in-place 추가. ✅ 태그(익명 파생 가능) arm만.
    target-파생(F27 압박 델타, F28 구장률)은 cut로 val 동결. 나머지는 target 무관."""
    sc = spec.season
    season = df[sc]
    min_season = int(season.min())
    g_pit = df.groupby(spec.pitcher_id, sort=False)

    # F24: 경험곡선 프록시 — 첫 등장 후 경과 시즌 + 2019 좌측절단 플래그 (target 무관)
    first_season = g_pit[sc].transform("min")
    X["f24_seasons_exp"] = (season - first_season).astype(float).to_numpy()
    X["f24_censored"] = (first_season == min_season).astype(float).to_numpy()  # 경력 미상(절단)

    # F25: 장기 공백 복귀 프록시 — 직전 등판(game_id) 대비 간격 (target 무관)
    #      실데이터에선 game_id 대신 날짜 간격으로 교체. 시즌 경계 넘김 = 큰 gap.
    game_ord = df[spec.game_id].astype(float)
    prev_game = g_pit[spec.game_id].transform(
        lambda s: s.where(s.ne(s.shift(1))).ffill().shift(1))  # 직전 '다른' 경기 id
    gap = (game_ord - prev_game.astype(float)).clip(lower=0)
    X["f25_gap"] = np.log1p(gap.fillna(0.0).to_numpy())
    X["f25_long_gap"] = (gap > 60).astype(float).fillna(0.0).to_numpy()  # 복귀 플래그

    # F26: 등판 내 릴리스 방향성 드리프트 + 당일 그립 프록시 (릴리스=실측이나 과거 투구만, target 무관)
    rel_z = "release_z" if "release_z" in df.columns else None
    if rel_z:
        go = df.groupby([spec.game_id, spec.pitcher_id], sort=False)[rel_z]
        prior_recent = go.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        prior_start = go.transform(lambda s: s.shift(1).expanding().mean())
        X["f26_outing_drift"] = (prior_recent - prior_start).fillna(0.0).to_numpy()
        season_mean = g_pit[rel_z].transform(lambda s: s.shift(1).expanding().mean())
        X["f26_daily_dev"] = (prior_start - season_mean).fillna(0.0).to_numpy()

    # F27: 투수별 압박 민감도 — as-of (고LI 성공률 − 저LI 성공률). target-파생 → cutoff 동결.
    hi = (df["leverage"] > 1.5).to_numpy()
    r_hi = _asof_rate_masked(df, spec.pitcher_id, spec.target, hi, 20.0, gm, sc, cut)
    r_lo = _asof_rate_masked(df, spec.pitcher_id, spec.target, ~hi, 20.0, gm, sc, cut)
    X["f27_press_delta"] = (r_hi - r_lo).to_numpy()

    # F28: 구장 프록시 — is_home + 홈팀ID(=익명 구장) as-of 성공률. 후자는 target-파생 → 동결.
    if spec.is_home in df.columns:
        X["f28_is_home"] = df[spec.is_home].astype(float).to_numpy()
    if spec.home_team_id in df.columns:
        X["f28_stadium_rate"] = _asof_group_rate(
            df, df[spec.home_team_id].to_numpy(), spec.target, 200.0, gm, sc, cut)

    # F29: ABS-era 플래그 + era×카운트 상호작용 (target 무관 — 캘린더 사실)
    era_abs = (season >= spec.abs_era_start).astype(float)
    X["f29_era_abs"] = era_abs.to_numpy()
    X["f29_era_v2"] = (season >= spec.abs_era_start + 1).astype(float).to_numpy()  # 2025 존 v2+클락
    X["f29_era_x_3ball"] = (era_abs * (df["balls"] == 3)).to_numpy()
    X["f29_era_x_2strike"] = (era_abs * (df["strikes"] == 2)).to_numpy()

    # F30: 시즌 인덱스 + 시즌 진행도 (target 무관). z-score 상대화는 트리가 era 분기로 근사.
    X["f30_season_idx"] = (season - min_season).astype(float).to_numpy()
    X["f30_season_prog"] = df.groupby(sc, sort=False)[spec.game_id].transform(
        lambda s: s.rank(pct=True, method="dense")).to_numpy()

    # F33: 도루위협 — 1루 단독 점유 × 박빙 (target 무관; on_base 비트 가정: 1=1루,2=2루,4=3루)
    ob = df["on_base"].astype(int).to_numpy()
    first_only = ((ob & 1) == 1) & ((ob & 6) == 0)
    X["f33_steal_threat"] = (first_only & (df["score_diff"].abs().to_numpy() <= 2)).astype(float)


def build_sequences(df: pd.DataFrame, spec: DataSpec, K: int = 16, seq_cols=None):
    """S3 시퀀스 student용 as-of 투구열 텐서.

    각 행 i → 그 투수의 **직전 K개 투구**의 pre-release 맥락 벡터(현재 투구 제외 = 누수 0),
    좌측 zero-padding. 반환: np.ndarray[N, K, F] (build_features와 동일한 정렬/행순서).
    맥락(target 무관)만 쓰므로 cutoff 불필요.
    주: 스모크 규모(수만 행)에 맞춘 투수별 파이썬 루프. 실데이터(수백만)에선 벡터화 필요(TODO).
    """
    df = df.sort_values([spec.season, spec.game_id, spec.order]).reset_index(drop=True)
    if seq_cols is None:
        seq_cols = list(spec.context)  # pre-release 맥락만(결과/실측 배제)
    vals = df[seq_cols].to_numpy(dtype=np.float32)
    N, F = len(df), len(seq_cols)
    S = np.zeros((N, K, F), dtype=np.float32)
    for _pid, idx in df.groupby(spec.pitcher_id, sort=False).indices.items():
        idx = np.sort(idx)                     # 전역 정렬 → 투수 내 시간순
        arr = vals[idx]
        for t in range(len(idx)):
            lo = max(0, t - K)
            window = arr[lo:t]                 # 직전 투구들(현재 t 제외)
            L = window.shape[0]
            if L:
                S[idx[t], K - L:, :] = window
    return S
