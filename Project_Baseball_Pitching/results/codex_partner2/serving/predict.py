"""Row-independent serving entry point for codex_partner2."""

from __future__ import annotations

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd


def _context(df, include_history=True):
    size = len(df)
    month = df["game_month"].to_numpy(dtype=np.float32)
    dow = df["game_dayofweek"].to_numpy(dtype=np.float32)
    base_map = {"___": 0, "1__": 1, "_2_": 2, "__3": 3,
                "12_": 4, "1_3": 5, "_23": 6, "123": 7}
    values = {
        "month": month,
        "month_sin": np.sin(2 * np.pi * month / 12).astype(np.float32),
        "month_cos": np.cos(2 * np.pi * month / 12).astype(np.float32),
        "dow": dow,
        "dow_sin": np.sin(2 * np.pi * dow / 7).astype(np.float32),
        "dow_cos": np.cos(2 * np.pi * dow / 7).astype(np.float32),
        "inning": np.clip(df["inning"].to_numpy(), 1, 12).astype(np.float32),
        "top_bottom": (df["top_bottom"].to_numpy() == "B").astype(np.float32),
        "game_type": (df["game_type"].to_numpy() == "F").astype(np.float32),
        "balls": np.clip(df["balls_before"].to_numpy(), 0, 3).astype(np.float32),
        "strikes": np.clip(df["strikes_before"].to_numpy(), 0, 2).astype(np.float32),
        "outs": np.clip(df["outs_before"].to_numpy(), 0, 2).astype(np.float32),
        "count_code": (np.clip(df["balls_before"].to_numpy(), 0, 3) * 3
                       + np.clip(df["strikes_before"].to_numpy(), 0, 2)).astype(np.float32),
        "score_diff": np.clip(df["score_diff_pitcher_team"].to_numpy(), -8, 8).astype(np.float32),
        "run_total": np.clip(df["run_total_before"].to_numpy(), 0, 20).astype(np.float32),
        "runner_1b": df["runner_on_1b"].to_numpy(dtype=np.float32),
        "runner_2b": df["runner_on_2b"].to_numpy(dtype=np.float32),
        "runner_3b": df["runner_on_3b"].to_numpy(dtype=np.float32),
        "num_runners": df["num_runners_on"].to_numpy(dtype=np.float32),
        "base_state": np.asarray([base_map.get(str(v), 0) for v in df["base_state"]], dtype=np.float32),
        "li_log": np.log1p(np.clip(df["li"].to_numpy(dtype=np.float32), 0, 20)),
        "home_we": (df["home_win_expectancy"].to_numpy(dtype=np.float32) - 50) / 50,
        "pitcher_hand": df["pitcher_hand"].to_numpy(dtype=np.float32),
        "batter_hand": df["batter_hand"].to_numpy(dtype=np.float32),
        "hand_match": (df["pitcher_hand"].to_numpy() == df["batter_hand"].to_numpy()).astype(np.float32),
        "pitcher_team": df["pitcher_team_id"].to_numpy(dtype=np.float32),
        "batter_team": df["batter_team_id"].to_numpy(dtype=np.float32),
    }
    if include_history:
        history_cols = [
            "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
            "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
            "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
            "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
            "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
            "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
            "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
        ]
        for col in history_cols:
            arr = df[col].to_numpy(dtype=np.float32)
            if col.endswith("_n"):
                arr = np.log1p(np.maximum(arr, 0))
            values[col] = arr
        ps = df["asof_pitcher_success_rate"].to_numpy(dtype=np.float32)
        pm = df["asof_pitcher_middle_rate"].to_numpy(dtype=np.float32)
        values["relative_pitcher_batter_success"] = ps - df["asof_batter_success_rate"].to_numpy(dtype=np.float32)
        values["relative_pitcher_batter_middle"] = pm - df["asof_batter_middle_rate"].to_numpy(dtype=np.float32)
        for window in [1, 3, 5]:
            values[f"recent{window}_success_gap"] = (
                df[f"asof_pitcher_prev{window}_game_success_rate"].to_numpy(dtype=np.float32) - ps
            )
            values[f"recent{window}_middle_gap"] = (
                df[f"asof_pitcher_prev{window}_game_middle_rate"].to_numpy(dtype=np.float32) - pm
            )
    return pd.DataFrame(values, index=np.arange(size))


def predict(test_df, model_dir):
    """Return independent per-row probabilities as a float64 numpy array."""
    model_dir = os.fspath(model_dir)
    with open(os.path.join(model_dir, "model_meta.json"), "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    packed = np.load(os.path.join(model_dir, "trajectory_lookup.npz"), allow_pickle=False)
    ids = packed["ids"]
    matrix = packed["values"]
    fallback = packed["fallback"]
    names = [str(v) for v in packed["feature_names"]]
    lookup = {int(pid): matrix[i] for i, pid in enumerate(ids)}
    pitcher_ids = test_df["pitcher_id"].to_numpy(dtype=np.int64)
    trajectory = np.vstack([lookup.get(int(pid), fallback) for pid in pitcher_ids])
    tm_frame = pd.DataFrame(trajectory, columns=names)
    context = _context(test_df, include_history=True)
    features = pd.concat([context.reset_index(drop=True), tm_frame.reset_index(drop=True)], axis=1)
    tm_model = lgb.Booster(model_file=os.path.join(model_dir, "trajectory_model.txt"))
    hybrid_model = lgb.Booster(model_file=os.path.join(model_dir, "hybrid_model.txt"))
    tm_raw = tm_model.predict(features[meta["context_columns"]])
    hybrid_raw = hybrid_model.predict(features[meta["hybrid_columns"]])
    weight = float(meta["trajectory_weight"])
    residual = ((1.0 - weight) * hybrid_raw + weight * tm_raw
                - float(meta.get("shape_center", 0.0)))
    result = (float(meta["base_rate"]) + float(meta.get("residual_intercept", 0.0))
              + float(meta["residual_scale"]) * residual)
    result = np.clip(result, 0.001, 0.999)
    return np.asarray(result, dtype=np.float64)
