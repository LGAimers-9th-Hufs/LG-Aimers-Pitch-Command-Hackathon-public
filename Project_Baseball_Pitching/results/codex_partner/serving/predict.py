"""Self-contained inference entry point for the Codex partner model."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


_BUNDLE_CACHE = {}
_GAME_MAP = {"F": 0, "R": 1}
_TOP_MAP = {"T": 1.0, "B": 0.0}
_HAND_MAP = {"L": 1.0, "R": 0.0, "S": 0.0}


def _numeric(frame, name, default=0.0):
    values = frame.get(name)
    if values is None:
        return np.full(len(frame), default, dtype=np.float32)
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float32, copy=False)


def _mapped(frame, name, mapping, default, dtype):
    values = frame.get(name)
    if values is None:
        return np.full(len(frame), default, dtype=dtype)
    raw = values.astype(str).to_numpy()
    return np.fromiter((mapping.get(value, default) for value in raw), dtype=dtype, count=len(raw))


def _career(values):
    safe = np.where(np.isfinite(values), values, 0.0)
    return np.where(safe < 250.0, 0, np.where(safe < 1500.0, 1, 2)).astype(np.int8)


def _segments(frame):
    game = _mapped(frame, "game_type", _GAME_MAP, 1, np.int8)
    return (game * 3 + _career(_numeric(frame, "asof_pitcher_n"))).astype(np.int8)


def _features(frame):
    balls = _numeric(frame, "balls_before")
    strikes = _numeric(frame, "strikes_before")
    outs = _numeric(frame, "outs_before")
    inning = _numeric(frame, "inning")
    score = np.clip(_numeric(frame, "score_diff_pitcher_team"), -8, 8)
    run_total = np.log1p(np.clip(_numeric(frame, "run_total_before"), 0, 30))
    r1 = _numeric(frame, "runner_on_1b")
    r2 = _numeric(frame, "runner_on_2b")
    r3 = _numeric(frame, "runner_on_3b")
    li = np.log1p(np.clip(_numeric(frame, "li"), 0, 10))
    month = _numeric(frame, "game_month")
    dow = _numeric(frame, "game_dayofweek")

    pitcher_home = _mapped(frame, "top_bottom", _TOP_MAP, 0.0, np.float32)
    home_we = _numeric(frame, "home_win_expectancy", 50.0) / 100.0
    away_we = _numeric(frame, "away_win_expectancy", 50.0) / 100.0
    pitcher_we = np.where(pitcher_home > 0, home_we, away_we).astype(np.float32)
    pitcher_left = _mapped(frame, "pitcher_hand", _HAND_MAP, 0.0, np.float32)
    batter_left = _mapped(frame, "batter_hand", _HAND_MAP, 0.0, np.float32)
    pitcher_hand_raw = frame.get("pitcher_hand")
    batter_hand_raw = frame.get("batter_hand")
    if pitcher_hand_raw is None:
        pitcher_hand_raw = pd.Series(["R"] * len(frame), index=frame.index)
    if batter_hand_raw is None:
        batter_hand_raw = pd.Series(["R"] * len(frame), index=frame.index)
    same_hand = (
        pitcher_hand_raw.astype(str).to_numpy() == batter_hand_raw.astype(str).to_numpy()
    ).astype(np.float32)

    reverse = _numeric(frame, "asof_pitcher_reverse_rate", np.nan)
    middle = _numeric(frame, "asof_pitcher_middle_rate", np.nan)
    ball_rate = _numeric(frame, "asof_pitcher_ball_rate", np.nan)
    strike_rate = _numeric(frame, "asof_pitcher_strike_rate", np.nan)
    batter_success = _numeric(frame, "asof_batter_success_rate", np.nan)
    batter_middle = _numeric(frame, "asof_batter_middle_rate", np.nan)
    pitcher_success = _numeric(frame, "asof_pitcher_success_rate", np.nan)
    pitcher_n = _numeric(frame, "asof_pitcher_n")
    batter_n = _numeric(frame, "asof_batter_n")
    pitchmix_n = _numeric(frame, "asof_pitcher_pitchmix_n")
    fast = _numeric(frame, "asof_pitcher_fastball_rate", np.nan)
    breaking = _numeric(frame, "asof_pitcher_breaking_rate", np.nan)
    offspeed = _numeric(frame, "asof_pitcher_offspeed_rate", np.nan)
    prev1_success = _numeric(frame, "asof_pitcher_prev1_game_success_rate", np.nan)
    prev3_success = _numeric(frame, "asof_pitcher_prev3_game_success_rate", np.nan)
    prev5_success = _numeric(frame, "asof_pitcher_prev5_game_success_rate", np.nan)
    prev1_middle = _numeric(frame, "asof_pitcher_prev1_game_middle_rate", np.nan)
    prev3_middle = _numeric(frame, "asof_pitcher_prev3_game_middle_rate", np.nan)
    prev5_middle = _numeric(frame, "asof_pitcher_prev5_game_middle_rate", np.nan)
    mix = np.column_stack([fast, breaking, offspeed]).astype(np.float32)
    mix_safe = np.clip(mix, 1e-6, 1.0)
    mix_entropy = -np.nansum(mix_safe * np.log(mix_safe), axis=1).astype(np.float32)
    mix_for_max = np.where(np.isfinite(mix), mix, -np.inf)
    mix_max = np.max(mix_for_max, axis=1)
    mix_max[~np.isfinite(mix_max)] = np.nan

    must_attack = (balls == 3).astype(np.float32)
    waste_option = ((strikes == 2) & (balls <= 1)).astype(np.float32)
    two_strike = (strikes == 2).astype(np.float32)
    full_count = ((balls == 3) & (strikes == 2)).astype(np.float32)
    first_pitch = ((balls == 0) & (strikes == 0)).astype(np.float32)
    late = (inning >= 7).astype(np.float32)
    close_game = (np.abs(score) <= 1).astype(np.float32)
    risp = ((r2 + r3) > 0).astype(np.float32)
    double_play = ((r1 > 0) & (outs < 2)).astype(np.float32)
    sacrifice_fly = ((r3 > 0) & (outs < 2)).astype(np.float32)
    steal_pressure = ((r1 > 0) & (r2 == 0) & (r3 == 0) & (outs < 2)).astype(np.float32)
    open_first = (r1 == 0).astype(np.float32)
    intentional_avoid = (open_first * risp * (outs == 2)).astype(np.float32)
    win_tension = (1.0 - 2.0 * np.abs(pitcher_we - 0.5)).astype(np.float32)
    career = _career(pitcher_n)
    game_is_f = (_mapped(frame, "game_type", _GAME_MAP, 1, np.int8) == 0).astype(np.float32)
    career_progress = np.where(
        career == 0,
        np.clip(pitcher_n / 250.0, 0, 1),
        np.where(
            career == 1,
            np.clip((pitcher_n - 250.0) / 1250.0, 0, 1),
            np.clip((pitcher_n - 1500.0) / 5000.0, 0, 1),
        ),
    ).astype(np.float32)
    non_success = np.clip(1.0 - pitcher_success, 0.02, 1.0)
    unexplained_failure = np.clip(non_success - reverse - middle, -0.10, 1.0)

    data = {
        "month": month,
        "month_phase": np.clip((month - 3.0) / 7.0, 0, 1),
        "dow": dow,
        "inning": np.clip(inning, 1, 12),
        "extra_innings": (inning >= 10).astype(np.float32),
        "balls": balls,
        "strikes": strikes,
        "outs": outs,
        "pa_depth": balls + strikes,
        "count_balance": strikes - balls,
        "first_pitch": first_pitch,
        "must_attack": must_attack,
        "waste_option": waste_option,
        "two_strike": two_strike,
        "full_count": full_count,
        "pitcher_ahead": (strikes > balls).astype(np.float32),
        "hitter_ahead": (balls > strikes).astype(np.float32),
        "run_total_log": run_total,
        "score_diff_clip": score,
        "close_game": close_game,
        "late": late,
        "close_late": close_game * late,
        "protecting_lead": (score > 0).astype(np.float32),
        "r1": r1,
        "r2": r2,
        "r3": r3,
        "num_runners": _numeric(frame, "num_runners_on"),
        "risp": risp,
        "double_play": double_play,
        "bases_loaded": ((r1 + r2 + r3) == 3).astype(np.float32),
        "two_out_risp": ((outs == 2) & (risp > 0)).astype(np.float32),
        "sacrifice_fly_situation": sacrifice_fly,
        "steal_pressure": steal_pressure,
        "lefty_hold_pressure": steal_pressure * pitcher_left,
        "intentional_avoid_proxy": intentional_avoid,
        "li_log": li,
        "pitcher_win_expectancy": pitcher_we,
        "win_tension": win_tension,
        "inning_x_pressure": np.clip(inning, 1, 12) * li,
        "late_deficit_pressure": late * (score < 0).astype(np.float32) * win_tension,
        "pitcher_home": pitcher_home,
        "same_hand": same_hand,
        "pitcher_left": pitcher_left,
        "batter_left": batter_left,
        "reverse_rate": reverse,
        "middle_rate": middle,
        "ball_rate": ball_rate,
        "strike_rate": strike_rate,
        "failure_style_sum": reverse + middle,
        "reverse_minus_middle": reverse - middle,
        "zone_tendency": strike_rate - ball_rate,
        "batter_success_rate": batter_success,
        "batter_middle_rate": batter_middle,
        "pitcher_success_rate": pitcher_success,
        "success_rate_squared": pitcher_success * pitcher_success,
        "unexplained_failure_rate": unexplained_failure,
        "reverse_share_of_failure": reverse / non_success,
        "middle_share_of_failure": middle / non_success,
        "failure_style_entropy_proxy": reverse * middle * unexplained_failure,
        "cumulative_rate_incoherence": np.abs(pitcher_success + reverse + middle - 1.0),
        "fastball_rate": fast,
        "breaking_rate": breaking,
        "offspeed_rate": offspeed,
        "pitchmix_entropy": mix_entropy,
        "pitchmix_max": mix_max,
        "prev1_success": prev1_success,
        "prev3_success": prev3_success,
        "prev5_success": prev5_success,
        "acute_success_change": prev1_success - prev5_success,
        "success_form_curve": prev1_success - 2.0 * prev3_success + prev5_success,
        "success_form_disagreement": np.abs(prev1_success - prev5_success),
        "cumulative_minus_prev1_success": pitcher_success - prev1_success,
        "cumulative_minus_prev5_success": pitcher_success - prev5_success,
        "prev1_middle": prev1_middle,
        "prev3_middle": prev3_middle,
        "prev5_middle": prev5_middle,
        "acute_middle_change": prev1_middle - prev5_middle,
        "middle_form_curve": prev1_middle - 2.0 * prev3_middle + prev5_middle,
        "attack_x_middle": must_attack * middle,
        "attack_x_fastball": must_attack * fast,
        "waste_x_breaking": waste_option * breaking,
        "two_strike_x_offspeed": two_strike * offspeed,
        "platoon_x_breaking": (1.0 - same_hand) * breaking,
        "risp_x_reverse": risp * reverse,
        "close_late_x_middle": close_game * late * middle,
        "pressure_x_ball": li * ball_rate,
        "pressure_x_reverse": li * reverse,
        "traffic_x_attack": (r1 + r2 + r3) * must_attack,
        "sac_fly_x_fastball": sacrifice_fly * fast,
        "avoid_x_offspeed": intentional_avoid * offspeed,
        "career_rookie": (career == 0).astype(np.float32),
        "career_developing": (career == 1).astype(np.float32),
        "career_veteran": (career == 2).astype(np.float32),
        "game_type_f": game_is_f,
        "career_progress": career_progress,
        "batter_cold": (batter_n < 100).astype(np.float32),
        "batter_established": (batter_n >= 1500).astype(np.float32),
        "pitchmix_cold": (pitchmix_n < 100).astype(np.float32),
        "pitchmix_established": (pitchmix_n >= 1500).astype(np.float32),
    }
    for b in range(4):
        for s in range(3):
            flag = ((balls == b) & (strikes == s)).astype(np.float32)
            prefix = f"count_{b}_{s}"
            data[prefix] = flag
            data[f"{prefix}_fast"] = flag * fast
            data[f"{prefix}_breaking"] = flag * breaking
            data[f"{prefix}_reverse"] = flag * reverse
            data[f"{prefix}_middle"] = flag * middle
            data[f"{prefix}_traffic"] = flag * (r1 + r2 + r3)
    inning_stage = np.where(inning <= 3, 0, np.where(inning <= 6, 1, np.where(inning <= 8, 2, 3)))
    for stage in range(4):
        data[f"inning_stage_{stage}"] = (inning_stage == stage).astype(np.float32)
    base_code = (r1 + 2.0 * r2 + 4.0 * r3).astype(np.int8)
    for state in range(8):
        data[f"base_state_{state}"] = (base_code == state).astype(np.float32)
    score_bucket = np.where(
        score <= -4,
        0,
        np.where(score <= -2, 1, np.where(score <= 1, 2, np.where(score <= 3, 3, 4))),
    )
    for bucket in range(5):
        data[f"score_bucket_{bucket}"] = (score_bucket == bucket).astype(np.float32)
    for month_value in range(3, 11):
        data[f"month_{month_value}"] = (month == month_value).astype(np.float32)
    return pd.DataFrame(data, index=frame.index, dtype=np.float32)


def _load_bundle(model_dir):
    root = str(Path(model_dir).resolve())
    cached = _BUNDLE_CACHE.get(root)
    if cached is not None:
        return cached
    base = Path(root)
    with (base / "constants.json").open("r", encoding="utf-8") as handle:
        constants = json.load(handle)
    models = {}
    for key, filenames in constants.get("models", {}).items():
        models[key] = [lgb.Booster(model_file=str(base / name)) for name in filenames]
    bundle = (constants, models)
    _BUNDLE_CACHE[root] = bundle
    return bundle


def _pair_prediction(models, key, values):
    pair = models.get(key, [])
    if len(pair) == 0:
        return np.zeros(len(values), dtype=np.float64)
    total = np.zeros(len(values), dtype=np.float64)
    for booster in pair:
        total += booster.predict(values, num_threads=6)
    return total / float(len(pair))


def predict(test_df, model_dir):
    """Return one independent probability for each input row."""
    constants, models = _load_bundle(model_dir)
    features = _features(test_df)
    feature_names = constants.get("feature_names", list(features.columns))
    features = features.reindex(columns=feature_names, fill_value=np.nan)
    segments = _segments(test_df)
    games = _mapped(test_df, "game_type", _GAME_MAP, 1, np.int8)

    global_prediction = _pair_prediction(models, "global", features)
    parent_prediction = np.empty(len(test_df), dtype=np.float64)
    game_constants = constants.get("games", {})
    for game in range(2):
        mask = games == game
        if not np.any(mask):
            continue
        specific = _pair_prediction(models, f"game_{game}", features.loc[mask])
        config = game_constants.get(str(game), {})
        weight = float(config.get("weight", 0.0))
        parent_prediction[mask] = weight * specific + (1.0 - weight) * global_prediction[mask]

    output = np.empty(len(test_df), dtype=np.float64)
    segment_constants = constants.get("segments", {})
    success = _numeric(test_df, "asof_pitcher_success_rate", np.nan).astype(np.float64)
    residual_scale = float(constants.get("residual_scale", 0.90))
    fallback = constants.get("fallback", {})
    for segment in range(6):
        mask = segments == segment
        if not np.any(mask):
            continue
        cell = _pair_prediction(models, f"segment_{segment}", features.loc[mask])
        config = segment_constants.get(str(segment), fallback)
        cell_weight = float(config.get("cell_weight", 0.0))
        residual = cell_weight * cell + (1.0 - cell_weight) * parent_prediction[mask]
        asof_reference = float(config.get("asof_reference", fallback.get("asof_reference", 0.5)))
        segment_base = float(config.get("segment_base", fallback.get("segment_base", 0.5)))
        base_weight = float(config.get("base_weight", 0.55))
        local_success = success[mask]
        local_success = np.where(np.isfinite(local_success), local_success, asof_reference)
        baseline = segment_base + base_weight * (local_success - asof_reference)
        output[mask] = baseline + residual_scale * residual

    output = np.clip(output, 0.02, 0.98).astype(np.float64, copy=False)
    if output.shape != (len(test_df),) or not np.all(np.isfinite(output)):
        raise RuntimeError("Partner prediction contract failed")
    return output

