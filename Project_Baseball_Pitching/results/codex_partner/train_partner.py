#!/usr/bin/env python3
"""Train the Codex blend-partner model and create strict season-cutoff predictions.

The model is deliberately built around information axes that are not the team's
main recipe: separate game-type/career models, a season-centred pitcher-baseline
residual target, and hand-written baseball situation features.  Every inference
feature is either a transformation of one row or a constant learned from rows
strictly before the prediction season.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


SEED = 260809
TRAIN_SEASONS = 3
SEASON_DECAY = 0.50
PITCHER_BASE_WEIGHTS = np.array([0.25, 0.55, 0.80], dtype=np.float64)
RESIDUAL_SCALE = 0.90
CAREER_CUTS = (250, 1500)
ENSEMBLE_SEEDS = (260809, 260921)
CELL_SHRINK_N = 50000.0
GAME_SHRINK_N = 100000.0

INPUT_COLUMNS = [
    "row_id", "season", "game_month", "game_dayofweek", "inning",
    "top_bottom", "game_type", "balls_before", "strikes_before",
    "outs_before", "run_total_before", "score_diff_pitcher_team",
    "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
    "home_win_expectancy", "away_win_expectancy", "li",
    "pitcher_hand", "batter_hand", "asof_pitcher_n",
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate", "asof_batter_success_rate",
    "asof_batter_middle_rate", "asof_batter_n", "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "control_success",
]


def career_bin(values: pd.Series) -> np.ndarray:
    n = values.to_numpy(dtype=np.float64, copy=False)
    return np.where(n < CAREER_CUTS[0], 0, np.where(n < CAREER_CUTS[1], 1, 2)).astype(np.int8)


def segment_code(frame: pd.DataFrame) -> np.ndarray:
    game = np.where(frame["game_type"].astype(str).to_numpy() == "F", 0, 1)
    return (game * 3 + career_bin(frame["asof_pitcher_n"])).astype(np.int8)


def _f(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float32, copy=False)


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Only deterministic row-wise features; never inspects neighboring rows."""
    balls = _f(frame, "balls_before")
    strikes = _f(frame, "strikes_before")
    outs = _f(frame, "outs_before")
    inning = _f(frame, "inning")
    score = np.clip(_f(frame, "score_diff_pitcher_team"), -8, 8)
    run_total = np.log1p(np.clip(_f(frame, "run_total_before"), 0, 30))
    r1 = _f(frame, "runner_on_1b")
    r2 = _f(frame, "runner_on_2b")
    r3 = _f(frame, "runner_on_3b")
    li = np.log1p(np.clip(_f(frame, "li"), 0, 10))
    month = _f(frame, "game_month")
    dow = _f(frame, "game_dayofweek")

    pitcher_home = (frame["top_bottom"].astype(str).to_numpy() == "T").astype(np.float32)
    home_we = _f(frame, "home_win_expectancy") / 100.0
    away_we = _f(frame, "away_win_expectancy") / 100.0
    pitcher_we = np.where(pitcher_home > 0, home_we, away_we).astype(np.float32)
    ph = frame["pitcher_hand"].astype(str).to_numpy()
    bh = frame["batter_hand"].astype(str).to_numpy()
    same_hand = (ph == bh).astype(np.float32)
    pitcher_left = (ph == "L").astype(np.float32)
    batter_left = (bh == "L").astype(np.float32)

    reverse = _f(frame, "asof_pitcher_reverse_rate")
    middle = _f(frame, "asof_pitcher_middle_rate")
    ball_rate = _f(frame, "asof_pitcher_ball_rate")
    strike_rate = _f(frame, "asof_pitcher_strike_rate")
    batter_success = _f(frame, "asof_batter_success_rate")
    batter_middle = _f(frame, "asof_batter_middle_rate")
    pitcher_success = _f(frame, "asof_pitcher_success_rate")
    pitcher_n = _f(frame, "asof_pitcher_n")
    batter_n = _f(frame, "asof_batter_n")
    pitchmix_n = _f(frame, "asof_pitcher_pitchmix_n")
    fast = _f(frame, "asof_pitcher_fastball_rate")
    breaking = _f(frame, "asof_pitcher_breaking_rate")
    offspeed = _f(frame, "asof_pitcher_offspeed_rate")
    prev1_success = _f(frame, "asof_pitcher_prev1_game_success_rate")
    prev3_success = _f(frame, "asof_pitcher_prev3_game_success_rate")
    prev5_success = _f(frame, "asof_pitcher_prev5_game_success_rate")
    prev1_middle = _f(frame, "asof_pitcher_prev1_game_middle_rate")
    prev3_middle = _f(frame, "asof_pitcher_prev3_game_middle_rate")
    prev5_middle = _f(frame, "asof_pitcher_prev5_game_middle_rate")
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
    career = career_bin(frame["asof_pitcher_n"])
    game_is_f = (frame["game_type"].astype(str).to_numpy() == "F").astype(np.float32)
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
        # State variables are intentionally compact; player/team IDs are omitted.
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
        "num_runners": _f(frame, "num_runners_on"),
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
        # Stable failure-style and intent information.
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
        # Raw missing values are preserved.  Divergences encode acute-vs-stable form;
        # there is no denominator reconstruction or test-derived filling.
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
        # Human baseball hypotheses: intent changes with count, platoon and pressure.
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
        # Parent F/R models use these to share strength without erasing career regimes.
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
    # Exact baseball-state bases let shallow trees express non-monotone intent
    # without increasing tree depth.  These remain strictly row-wise.
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
    score_bucket = np.where(score <= -4, 0, np.where(score <= -2, 1, np.where(score <= 1, 2, np.where(score <= 3, 3, 4))))
    for bucket in range(5):
        data[f"score_bucket_{bucket}"] = (score_bucket == bucket).astype(np.float32)
    for month_value in range(3, 11):
        data[f"month_{month_value}"] = (month == month_value).astype(np.float32)
    return pd.DataFrame(data, index=frame.index, dtype=np.float32)


def forecast_overall_rate(train_frame: pd.DataFrame) -> tuple[float, dict[int, float]]:
    rates = train_frame.groupby("season", sort=True)["control_success"].mean()
    rate_dict = {int(k): float(v) for k, v in rates.items()}
    values = rates.to_numpy(dtype=np.float64)
    if len(values) >= 3:
        d1, d2 = values[-2] - values[-3], values[-1] - values[-2]
        if d1 * d2 > 0:  # extrapolate only after a direction persists twice
            forecast = float(np.polyval(np.polyfit(np.arange(3), values[-3:], 1), 3.0))
        else:
            forecast = float(values[-1])
    else:
        forecast = float(values[-1])
    forecast = float(np.clip(forecast, values[-1] - 0.03, values[-1] + 0.03))
    return forecast, rate_dict


MONOTONE_DIRECTIONS = {
    "pitcher_success_rate": 1,
    "reverse_rate": -1,
    "middle_rate": -1,
    "ball_rate": -1,
    "strike_rate": 1,
    "prev1_success": 1,
    "prev3_success": 1,
    "prev5_success": 1,
    "prev1_middle": -1,
    "prev3_middle": -1,
    "prev5_middle": -1,
    "batter_success_rate": 1,
    "batter_middle_rate": -1,
}


def make_model(n_train: int, seed: int, parent: bool, columns: pd.Index) -> lgb.LGBMRegressor:
    """Deliberately shallow, strongly regularized trees for season transfer."""
    return lgb.LGBMRegressor(
        objective="regression_l2",
        n_estimators=260,
        learning_rate=0.035,
        num_leaves=7,
        max_depth=3,
        min_child_samples=2000 if parent else min(1500, max(500, n_train // 12)),
        min_split_gain=1e-5,
        subsample=0.82,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.50,
        reg_lambda=6.0,
        monotone_constraints=[MONOTONE_DIRECTIONS.get(str(c), 0) for c in columns],
        monotone_constraints_method="advanced",
        random_state=seed,
        bagging_seed=seed + 17,
        feature_fraction_seed=seed + 29,
        data_random_seed=seed + 43,
        n_jobs=6,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def ensemble_predict(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    weights: np.ndarray,
    x_pred: pd.DataFrame,
    parent: bool,
) -> np.ndarray:
    predictions = []
    for seed in ENSEMBLE_SEEDS:
        model = make_model(len(x_train), seed, parent, x_train.columns)
        model.fit(x_train, y_train, sample_weight=weights)
        predictions.append(model.predict(x_pred))
    return np.mean(predictions, axis=0)


def fit_predict_fold(frame: pd.DataFrame, features: pd.DataFrame, target_year: int) -> tuple[np.ndarray, dict]:
    train_end = target_year - 1
    train_start = max(int(frame["season"].min()), train_end - TRAIN_SEASONS + 1)
    train_mask = (frame["season"] >= train_start) & (frame["season"] <= train_end)
    pred_mask = frame["season"] == target_year
    tr = frame.loc[train_mask].copy()
    va = frame.loc[pred_mask]
    if va.empty:
        raise ValueError(f"No rows found for target season {target_year}")

    tr["segment"] = segment_code(tr)
    va_segment = segment_code(va)
    tr_success = pd.to_numeric(tr["asof_pitcher_success_rate"], errors="coerce")
    tr["asof_success_clean"] = tr_success

    # Season/segment centering uses labels only from the permitted training cutoff.
    group = tr.groupby(["season", "segment"], observed=True)
    seg_y = group["control_success"].transform("mean").to_numpy(dtype=np.float64)
    seg_asof = group["asof_success_clean"].transform("mean").to_numpy(dtype=np.float64)
    asof_values = tr_success.to_numpy(dtype=np.float64)
    asof_values = np.where(np.isfinite(asof_values), asof_values, seg_asof)
    train_base_weight = PITCHER_BASE_WEIGHTS[tr["segment"].to_numpy(dtype=np.int8) % 3]
    baseline_train = seg_y + train_base_weight * (asof_values - seg_asof)
    residual_target = tr["control_success"].to_numpy(dtype=np.float64) - baseline_train

    overall_forecast, annual_rates = forecast_overall_rate(tr)
    last = tr[tr["season"] == train_end]
    last_overall = float(last["control_success"].mean())
    last_stats = last.groupby("segment", observed=True).agg(
        y_rate=("control_success", "mean"),
        asof_ref=("asof_success_clean", "mean"),
        n=("control_success", "size"),
    )
    global_asof_ref = float(last["asof_success_clean"].mean())

    y_pred = np.empty(len(va), dtype=np.float64)
    model_rows: dict[str, dict] = {}
    train_season = tr["season"].to_numpy(dtype=np.int16)
    season_weight = np.power(SEASON_DECAY, train_end - train_season).astype(np.float32)
    x_train_all = features.loc[train_mask]
    x_pred_all = features.loc[pred_mask]

    # A global model stabilizes the smaller F parent; F/R and career cells still
    # retain their own separately trained components.
    global_predictions = ensemble_predict(
        x_train_all,
        residual_target,
        season_weight,
        x_pred_all,
        parent=True,
    )

    # Two parent models share signal across career cells within F and R.
    parent_predictions = np.empty(len(va), dtype=np.float64)
    tr_game = np.where(tr["game_type"].astype(str).to_numpy() == "F", 0, 1)
    va_game = np.where(va["game_type"].astype(str).to_numpy() == "F", 0, 1)
    for game in range(2):
        parent_train = tr_game == game
        parent_pred = va_game == game
        game_specific = ensemble_predict(
            x_train_all.loc[parent_train],
            residual_target[parent_train],
            season_weight[parent_train],
            x_pred_all.loc[parent_pred],
            parent=True,
        )
        game_weight = float(parent_train.sum() / (parent_train.sum() + GAME_SHRINK_N))
        parent_predictions[parent_pred] = (
            game_weight * game_specific
            + (1.0 - game_weight) * global_predictions[parent_pred]
        )

    for seg in range(6):
        local_train = tr["segment"].to_numpy() == seg
        local_pred = va_segment == seg
        if not np.any(local_pred):
            continue
        n_train = int(local_train.sum())
        if n_train < 500:
            raise RuntimeError(f"Segment {seg} has only {n_train} training rows")

        cell_residual = ensemble_predict(
            x_train_all.loc[local_train],
            residual_target[local_train],
            season_weight[local_train],
            x_pred_all.loc[local_pred],
            parent=False,
        )
        cell_weight = float(n_train / (n_train + CELL_SHRINK_N))
        residual_pred = (
            cell_weight * cell_residual
            + (1.0 - cell_weight) * parent_predictions[local_pred]
        )

        if seg in last_stats.index:
            stat = last_stats.loc[seg]
            offset_shrink = float(stat["n"] / (stat["n"] + 2000.0))
            segment_base = overall_forecast + offset_shrink * (float(stat["y_rate"]) - last_overall)
            asof_ref = float(stat["asof_ref"])
            last_n = int(stat["n"])
        else:
            segment_base = overall_forecast
            asof_ref = global_asof_ref
            last_n = 0
        va_asof = pd.to_numeric(
            va.loc[va.index[local_pred], "asof_pitcher_success_rate"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        va_asof = np.where(np.isfinite(va_asof), va_asof, asof_ref)
        base_weight = float(PITCHER_BASE_WEIGHTS[seg % 3])
        row_base = segment_base + base_weight * (va_asof - asof_ref)
        y_pred[local_pred] = row_base + RESIDUAL_SCALE * residual_pred
        model_rows[str(seg)] = {
            "train_rows": n_train,
            "predict_rows": int(local_pred.sum()),
            "last_season_rows": last_n,
            "segment_base": float(segment_base),
            "asof_reference": float(asof_ref),
            "pitcher_base_weight": base_weight,
            "cell_prediction_weight": cell_weight,
        }

    y_pred = np.clip(y_pred, 0.02, 0.98)
    truth = va["control_success"].to_numpy(dtype=np.float64)
    r = float(truth.mean())
    brier = float(np.mean((truth - y_pred) ** 2))
    score = max(0.0, 100000.0 * (1.0 - brier / (r * (1.0 - r))))
    corr = float(np.corrcoef(truth, y_pred)[0, 1])
    affine_score = 100000.0 * corr * corr
    diagnostics = {
        "target_year": target_year,
        "training_years": [train_start, train_end],
        "train_rows": int(train_mask.sum()),
        "predict_rows": int(pred_mask.sum()),
        "annual_training_rates": annual_rates,
        "forecast_overall_rate": overall_forecast,
        "actual_rate": r,
        "prediction_mean": float(y_pred.mean()),
        "prediction_std": float(y_pred.std()),
        "brier": brier,
        "score": score,
        "affine_optimal_score": affine_score,
        "prediction_truth_corr": corr,
        "segments": model_rows,
    }
    return y_pred, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/data/train.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/codex_partner"))
    parser.add_argument("--years", nargs="+", type=int, default=[2024, 2023, 2022, 2021])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.train, usecols=INPUT_COLUMNS, low_memory=False)
    features = make_features(frame)
    all_metrics: dict[str, dict] = {}

    for year in args.years:
        pred, diagnostics = fit_predict_fold(frame, features, year)
        mask = frame["season"] == year
        output = pd.DataFrame({
            "row_id": frame.loc[mask, "row_id"].to_numpy(),
            "pred": pred,
        })
        output_path = args.output_dir / f"team_pred_{year}.csv"
        output.to_csv(output_path, index=False, encoding="utf-8", float_format="%.10f")
        all_metrics[str(year)] = diagnostics
        print(json.dumps(diagnostics, ensure_ascii=False, indent=2))

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
