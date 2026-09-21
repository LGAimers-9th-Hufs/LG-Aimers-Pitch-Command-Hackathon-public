# -*- coding: utf-8 -*-
"""JTT-DSF의 동적 선수 상태 피처.

이 모듈은 학습 이력으로 ``StateArtifact``를 만든 뒤 평가 행에는 frozen artifact와
그 행의 공식 ``asof_*`` 값만 적용한다.  따라서 같은 평가 배치의 다른 행, 행 순서,
배치 크기에 의존하지 않는다.

누적 공식 통계는 시즌말 누적치끼리 차분해 시즌 단위 관측으로 바꾼다. 선수의 다음 시즌
prior는 최근 시즌일수록 큰 가중치를 주는 상태 전이 평균과 과정 분산으로 표현하며, 현재
행의 시즌 누적 관측으로 Beta posterior를 갱신한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


EPS = 1e-8
DEFAULT_RHO = 0.68
DEFAULT_SEASON_K = 80.0
MIN_STRENGTH = 25.0
MAX_STRENGTH = 800.0


@dataclass(frozen=True)
class RateSpec:
    name: str
    entity: str
    n_col: str
    rate_col: str
    default: float


RATE_SPECS = (
    RateSpec("p_succ", "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate", 0.50),
    RateSpec("p_rev", "pitcher_id", "asof_pitcher_n", "asof_pitcher_reverse_rate", 0.14),
    RateSpec("p_mid", "pitcher_id", "asof_pitcher_n", "asof_pitcher_middle_rate", 0.16),
    RateSpec("p_ball", "pitcher_id", "asof_pitcher_n", "asof_pitcher_ball_rate", 0.50),
    RateSpec("p_strike", "pitcher_id", "asof_pitcher_n", "asof_pitcher_strike_rate", 0.50),
    RateSpec("b_succ", "batter_id", "asof_batter_n", "asof_batter_success_rate", 0.50),
    RateSpec("b_mid", "batter_id", "asof_batter_n", "asof_batter_middle_rate", 0.16),
    RateSpec(
        "mix_fb", "pitcher_id", "asof_pitcher_pitchmix_n",
        "asof_pitcher_fastball_rate", 1.0 / 3.0,
    ),
    RateSpec(
        "mix_br", "pitcher_id", "asof_pitcher_pitchmix_n",
        "asof_pitcher_breaking_rate", 1.0 / 3.0,
    ),
    RateSpec(
        "mix_os", "pitcher_id", "asof_pitcher_pitchmix_n",
        "asof_pitcher_offspeed_rate", 1.0 / 3.0,
    ),
)


def _json_number(value: float) -> float:
    value = float(value)
    return value if np.isfinite(value) else 0.0


def _last_by_entity_season(history: pd.DataFrame, spec: RateSpec) -> pd.DataFrame:
    """공식 누적 통계의 투수/타자×시즌 마지막 관측을 반환한다."""
    cols = [spec.entity, "season", spec.n_col, spec.rate_col]
    missing = [c for c in cols if c not in history]
    if missing:
        raise KeyError(f"dynamic state columns missing: {missing}")
    d = history[cols].copy()
    # row_id는 데이터에서 시간순이며, 없을 때도 입력 순서를 보존한다.
    if "row_id" in history:
        d["_row_id"] = history["row_id"].astype(str).to_numpy()
        d = d.sort_values([spec.entity, "season", "_row_id"], kind="mergesort")
    else:
        d["_order"] = np.arange(len(d), dtype=np.int64)
        d = d.sort_values([spec.entity, "season", "_order"], kind="mergesort")
    return d.groupby([spec.entity, "season"], sort=False, as_index=False).tail(1)


def _season_increments(last: pd.DataFrame, spec: RateSpec) -> pd.DataFrame:
    d = last[[spec.entity, "season", spec.n_col, spec.rate_col]].copy()
    d = d.sort_values([spec.entity, "season"], kind="mergesort").reset_index(drop=True)
    n = pd.to_numeric(d[spec.n_col], errors="coerce").fillna(0).to_numpy(dtype=float)
    rate = pd.to_numeric(d[spec.rate_col], errors="coerce").to_numpy(dtype=float)
    rate = np.where(np.isfinite(rate), np.clip(rate, 0.0, 1.0), spec.default)
    count = np.rint(rate * n)
    prev_n = d.groupby(spec.entity, sort=False)[spec.n_col].shift(1).fillna(0).to_numpy(dtype=float)
    count_series = pd.Series(count, index=d.index)
    prev_k = count_series.groupby(d[spec.entity], sort=False).shift(1).fillna(0).to_numpy(dtype=float)
    dn = n - prev_n
    dk = count - prev_k
    # 누적 카운터가 리셋되는 데이터가 있으면 해당 시즌 누적 자체를 관측으로 취급한다.
    reset = dn < 0
    dn = np.where(reset, n, dn)
    dk = np.where(reset, count, dk)
    dn = np.maximum(dn, 0.0)
    dk = np.clip(dk, 0.0, dn)
    d["end_n"] = n
    d["end_k"] = count
    d["season_n"] = dn
    d["season_k"] = dk
    return d


def _weighted_league_rate(inc: pd.DataFrame, default: float) -> float:
    ok = inc["season_n"].to_numpy(dtype=float) > 0
    if not ok.any():
        return float(default)
    n = inc.loc[ok, "season_n"].to_numpy(dtype=float)
    k = inc.loc[ok, "season_k"].to_numpy(dtype=float)
    return float(np.clip(k.sum() / max(n.sum(), 1.0), 1e-4, 1 - 1e-4))


def _player_prior(
    player: pd.DataFrame,
    target_season: int,
    league: float,
    rho: float,
    season_k: float,
) -> tuple[float, float, float, float, float]:
    """(mean, variance, strength, trend, history seasons)."""
    player = player[player["season_n"] > 0].sort_values("season")
    if player.empty:
        variance = league * (1.0 - league) / (MIN_STRENGTH + 1.0)
        return league, variance, MIN_STRENGTH, 0.0, 0.0

    n = player["season_n"].to_numpy(dtype=float)
    k = player["season_k"].to_numpy(dtype=float)
    seasons = player["season"].to_numpy(dtype=int)
    season_mean = (k + season_k * league) / (n + season_k)
    ages = np.maximum(target_season - 1 - seasons, 0)
    weights = np.power(rho, ages) * np.sqrt(np.maximum(n, 1.0) / (n + season_k))
    weights = weights / max(weights.sum(), EPS)
    mean = float(np.sum(weights * season_mean))
    process = float(np.sum(weights * (season_mean - mean) ** 2))
    observation = float(np.sum(weights * season_mean * (1.0 - season_mean) / (n + season_k + 1.0)))
    variance = max(process + observation, 1e-5)
    implied = mean * (1.0 - mean) / variance - 1.0
    strength = float(np.clip(implied, MIN_STRENGTH, MAX_STRENGTH))

    if len(season_mean) >= 2:
        prev_w = weights[:-1] / max(weights[:-1].sum(), EPS)
        trend = float(season_mean[-1] - np.sum(prev_w * season_mean[:-1]))
    else:
        trend = float(season_mean[-1] - league)
    return mean, variance, strength, trend, float(len(season_mean))


def fit_state_artifact(
    history: pd.DataFrame,
    target_season: int,
    specs: Iterable[RateSpec] = RATE_SPECS,
    rho: float = DEFAULT_RHO,
    season_k: float = DEFAULT_SEASON_K,
) -> dict[str, Any]:
    """``season < target_season``만 사용해 frozen 동적 상태 artifact를 만든다."""
    target_season = int(target_season)
    rates: dict[str, Any] = {}
    for spec in specs:
        last = _last_by_entity_season(history, spec)
        last = last[last["season"] < target_season].copy()
        inc = _season_increments(last, spec)
        league = _weighted_league_rate(inc, spec.default)
        players: dict[str, list[float]] = {}
        for entity, g in inc.groupby(spec.entity, sort=False):
            g = g.sort_values("season")
            last_row = g.iloc[-1]
            mean, var, strength, trend, n_hist = _player_prior(
                g, target_season, league, rho, season_k
            )
            players[str(int(entity))] = [
                _json_number(last_row["end_n"]),
                _json_number(last_row["end_k"]),
                _json_number(mean),
                _json_number(var),
                _json_number(strength),
                _json_number(trend),
                _json_number(n_hist),
            ]
        rates[spec.name] = {
            "entity": spec.entity,
            "n_col": spec.n_col,
            "rate_col": spec.rate_col,
            "default": float(spec.default),
            "league": float(league),
            "players": players,
        }
    return {
        "version": "jtt-dsf-state-v1",
        "target_season": target_season,
        "rho": float(rho),
        "season_k": float(season_k),
        "rates": rates,
        "provenance": "official train rows with season < target; row-local transform",
    }


def _lookup_matrix(ids: np.ndarray, players: dict[str, list[float]], league: float) -> np.ndarray:
    default_var = league * (1.0 - league) / (MIN_STRENGTH + 1.0)
    fallback = np.array([0.0, 0.0, league, default_var, MIN_STRENGTH, 0.0, 0.0], dtype=float)
    out = np.empty((len(ids), len(fallback)), dtype=float)
    for i, entity in enumerate(ids):
        if not np.isfinite(entity):
            out[i] = fallback
            continue
        value = players.get(str(int(entity)))
        out[i] = fallback if value is None else np.asarray(value, dtype=float)
    return out


def transform_state(df: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    """평가 행에 frozen artifact를 적용한다. 계산은 각 행에서 독립적이다."""
    if "season" in df and len(df):
        seasons = pd.to_numeric(df["season"], errors="coerce").dropna().unique()
        if len(seasons) and np.any(seasons.astype(int) != int(artifact["target_season"])):
            raise ValueError(
                f"artifact target={artifact['target_season']} but rows contain {seasons.tolist()}"
            )
    out = pd.DataFrame(index=df.index)
    for name, meta in artifact["rates"].items():
        entity = pd.to_numeric(df[meta["entity"]], errors="coerce").to_numpy(dtype=float)
        lookup = _lookup_matrix(entity, meta["players"], float(meta["league"]))
        base_n, base_k = lookup[:, 0], lookup[:, 1]
        prior, prior_var, strength, trend, n_hist = (
            lookup[:, 2], lookup[:, 3], lookup[:, 4], lookup[:, 5], lookup[:, 6]
        )
        n = pd.to_numeric(df[meta["n_col"]], errors="coerce").fillna(0).to_numpy(dtype=float)
        rate = pd.to_numeric(df[meta["rate_col"]], errors="coerce").to_numpy(dtype=float)
        rate = np.where(np.isfinite(rate), np.clip(rate, 0.0, 1.0), prior)
        count = np.rint(rate * n)
        dn = np.maximum(n - base_n, 0.0)
        dk = np.clip(count - base_k, 0.0, dn)
        post = (prior * strength + dk) / np.maximum(strength + dn, EPS)
        post_var = post * (1.0 - post) / (strength + dn + 1.0)
        rel = dn / np.maximum(strength + dn, EPS)
        prefix = f"ds_{name}"
        out[f"{prefix}_prior"] = prior.astype("float32")
        out[f"{prefix}_post"] = post.astype("float32")
        out[f"{prefix}_innov"] = (post - prior).astype("float32")
        out[f"{prefix}_var"] = post_var.astype("float32")
        out[f"{prefix}_rel"] = rel.astype("float32")
        out[f"{prefix}_trend"] = trend.astype("float32")
        out[f"{prefix}_logdn"] = np.log1p(dn).astype("float32")
        out[f"{prefix}_hist"] = n_hist.astype("float32")

    mix_cols = [f"ds_mix_{g}_post" for g in ("fb", "br", "os")]
    if all(c in out for c in mix_cols):
        mix = out[mix_cols].to_numpy(dtype=float)
        mix = np.clip(mix, EPS, None)
        mix = mix / np.maximum(mix.sum(axis=1, keepdims=True), EPS)
        out["ds_mix_entropy"] = (-np.sum(mix * np.log(mix), axis=1)).astype("float32")
        out["ds_mix_max"] = mix.max(axis=1).astype("float32")

    # 불확실성 기반 deterministic gate 재료. 학습된 계수가 없어도 안전하게 해석 가능하다.
    core_vars = [c for c in out if c.endswith("_var") and not c.startswith("ds_mix")]
    core_rels = [c for c in out if c.endswith("_rel") and not c.startswith("ds_mix")]
    if core_vars:
        out["ds_state_uncertainty"] = out[core_vars].mean(axis=1).astype("float32")
    if core_rels:
        out["ds_state_reliability"] = out[core_rels].mean(axis=1).astype("float32")
    return out.reset_index(drop=True)


def build_chronological_state_features(
    history: pd.DataFrame,
    seasons: Iterable[int],
) -> tuple[pd.DataFrame, dict[int, dict[str, Any]]]:
    """여러 학습 시즌을 각각 올바른 cutoff artifact로 변환한다."""
    blocks = []
    artifacts: dict[int, dict[str, Any]] = {}
    for season in [int(s) for s in seasons]:
        artifact = fit_state_artifact(history, season)
        rows = history[history["season"] == season]
        block = transform_state(rows, artifact)
        block.index = rows.index
        blocks.append(block)
        artifacts[season] = artifact
    if not blocks:
        return pd.DataFrame(index=history.index), artifacts
    return pd.concat(blocks).sort_index(), artifacts


def build_model_features(
    base_features: pd.DataFrame,
    state_features: pd.DataFrame,
    base_prediction: np.ndarray,
    disagreement: np.ndarray | None = None,
    trackman_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """DSF 모델의 학습/서빙 공통 feature 계약.

    ``base_features``는 기존 exact 계열이 이미 계산한 행 단위 feature다. 나머지도 frozen
    lookup 또는 그 행의 값뿐이다. 인덱스 정렬에 암묵적으로 의존하지 않도록 모두 reset한다.
    """
    base = base_features.reset_index(drop=True).astype("float32")
    state = state_features.reset_index(drop=True).astype("float32")
    if len(base) != len(state):
        raise ValueError(f"base/state row mismatch: {len(base)} != {len(state)}")
    parts = [base, state]
    if trackman_features is not None:
        tm = trackman_features.reset_index(drop=True).astype("float32")
        if len(tm) != len(base):
            raise ValueError(f"base/trackman row mismatch: {len(base)} != {len(tm)}")
        parts.append(tm)
    out = pd.concat(parts, axis=1)
    q = np.asarray(base_prediction, dtype=float)
    if len(q) != len(out):
        raise ValueError(f"base prediction row mismatch: {len(q)} != {len(out)}")
    out["ds_base_q"] = q.astype("float32")
    out["ds_base_margin"] = np.abs(q - 0.5).astype("float32")
    if disagreement is None:
        disagreement = np.zeros(len(out), dtype=float)
    disagreement = np.asarray(disagreement, dtype=float)
    if len(disagreement) != len(out):
        raise ValueError("disagreement row mismatch")
    out["ds_base_disagreement"] = disagreement.astype("float32")

    # 상태×상황 상호작용. 낮은 차원의 사전 정의 항만 두어 tree가 raw ID 조합을 외우지 않게 한다.
    if "count_code" in out:
        count = out["count_code"].to_numpy(dtype=float)
    else:
        count = np.zeros(len(out), dtype=float)
    p_innov = out.get("ds_p_succ_innov", pd.Series(0.0, index=out.index)).to_numpy(dtype=float)
    p_trend = out.get("ds_p_succ_trend", pd.Series(0.0, index=out.index)).to_numpy(dtype=float)
    matchup = (
        out.get("ds_p_succ_post", pd.Series(0.5, index=out.index)).to_numpy(dtype=float)
        - out.get("ds_b_succ_post", pd.Series(0.5, index=out.index)).to_numpy(dtype=float)
    )
    out["ds_matchup_state"] = matchup.astype("float32")
    out["ds_count_x_innov"] = ((count / 11.0) * p_innov).astype("float32")
    out["ds_count_x_trend"] = ((count / 11.0) * p_trend).astype("float32")
    out["ds_disagree_x_uncert"] = (
        disagreement
        * out.get("ds_state_uncertainty", pd.Series(0.0, index=out.index)).to_numpy(dtype=float)
    ).astype("float32")
    for code in range(12):
        out[f"ds_c{code}_innov"] = ((count == code) * p_innov).astype("float32")
        out[f"ds_c{code}_matchup"] = ((count == code) * matchup).astype("float32")
    return out.astype("float32")


def state_expert_columns(columns: Iterable[str]) -> list[str]:
    """동적 상태 전문가가 사용할 제한된 feature 부분공간."""
    keep_context = {
        "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
        "balls_before", "strikes_before", "outs_before", "base_state",
        "num_runners_on", "score_diff_pitcher_team", "li", "hand_match",
        "count_code", "p_logn", "b_logn", "p_sm500", "b_sm500",
    }
    return [c for c in columns if c.startswith("ds_") or c in keep_context]
