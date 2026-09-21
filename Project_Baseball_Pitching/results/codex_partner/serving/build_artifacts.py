"""Train and save the frozen R2 serving artifacts through season 2024."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SERVING_DIR = Path(__file__).resolve().parent
PARTNER_DIR = SERVING_DIR.parent
sys.path.insert(0, str(PARTNER_DIR))
import train_partner as tp  # noqa: E402


def save_ensemble(x, target, weight, parent, key, model_map):
    filenames = []
    for index, seed in enumerate(tp.ENSEMBLE_SEEDS):
        model = tp.make_model(len(x), seed, parent, x.columns)
        model.fit(x, target, sample_weight=weight)
        filename = f"{key}_seed{index}.txt"
        model.booster_.save_model(str(SERVING_DIR / filename))
        filenames.append(filename)
    model_map[key] = filenames


def main():
    source = Path("data/data/train.csv")
    frame = pd.read_csv(source, usecols=tp.INPUT_COLUMNS, low_memory=False)
    train = frame[frame["season"].between(2022, 2024)].copy().reset_index(drop=True)
    features = tp.make_features(train)
    train["segment"] = tp.segment_code(train)
    train_success = pd.to_numeric(train["asof_pitcher_success_rate"], errors="coerce")
    train["asof_success_clean"] = train_success

    grouped = train.groupby(["season", "segment"], observed=True)
    segment_y = grouped["control_success"].transform("mean").to_numpy(dtype=np.float64)
    segment_asof = grouped["asof_success_clean"].transform("mean").to_numpy(dtype=np.float64)
    asof_values = train_success.to_numpy(dtype=np.float64)
    asof_values = np.where(np.isfinite(asof_values), asof_values, segment_asof)
    segment_values = train["segment"].to_numpy(dtype=np.int8)
    base_weights = tp.PITCHER_BASE_WEIGHTS[segment_values % 3]
    baseline = segment_y + base_weights * (asof_values - segment_asof)
    residual_target = train["control_success"].to_numpy(dtype=np.float64) - baseline
    season_weight = np.power(0.5, 2024 - train["season"].to_numpy(dtype=np.int16)).astype(np.float32)

    model_map = {}
    save_ensemble(features, residual_target, season_weight, True, "global", model_map)
    game_values = np.where(train["game_type"].astype(str).to_numpy() == "F", 0, 1)
    game_config = {}
    for game in range(2):
        mask = game_values == game
        save_ensemble(
            features.loc[mask], residual_target[mask], season_weight[mask], True,
            f"game_{game}", model_map,
        )
        count = int(mask.sum())
        game_config[str(game)] = {
            "train_rows": count,
            "weight": float(count / (count + tp.GAME_SHRINK_N)),
        }

    overall_forecast, annual_rates = tp.forecast_overall_rate(train)
    last = train[train["season"] == 2024]
    last_overall = float(last["control_success"].mean())
    last_stats = last.groupby("segment", observed=True).agg(
        y_rate=("control_success", "mean"),
        asof_reference=("asof_success_clean", "mean"),
        n=("control_success", "size"),
    )
    global_asof_reference = float(last["asof_success_clean"].mean())
    segment_config = {}
    for segment in range(6):
        mask = segment_values == segment
        save_ensemble(
            features.loc[mask], residual_target[mask], season_weight[mask], False,
            f"segment_{segment}", model_map,
        )
        count = int(mask.sum())
        if segment in last_stats.index:
            stat = last_stats.loc[segment]
            offset_weight = float(stat["n"] / (stat["n"] + 2000.0))
            segment_base = overall_forecast + offset_weight * (float(stat["y_rate"]) - last_overall)
            asof_reference = float(stat["asof_reference"])
        else:
            segment_base = overall_forecast
            asof_reference = global_asof_reference
        segment_config[str(segment)] = {
            "train_rows": count,
            "cell_weight": float(count / (count + tp.CELL_SHRINK_N)),
            "segment_base": float(segment_base),
            "asof_reference": asof_reference,
            "base_weight": float(tp.PITCHER_BASE_WEIGHTS[segment % 3]),
        }

    constants = {
        "recipe": "r2_monotone_hierarchical_two_seed",
        "trained_seasons": [2022, 2023, 2024],
        "trained_through_season": 2024,
        "season_weights": {"2022": 0.25, "2023": 0.5, "2024": 1.0},
        "annual_training_rates": annual_rates,
        "overall_forecast_2025": overall_forecast,
        "residual_scale": float(tp.RESIDUAL_SCALE),
        "feature_names": list(features.columns),
        "models": model_map,
        "games": game_config,
        "segments": segment_config,
        "fallback": {
            "segment_base": overall_forecast,
            "asof_reference": global_asof_reference,
            "base_weight": 0.55,
            "cell_weight": 0.0,
        },
    }
    (SERVING_DIR / "constants.json").write_text(
        json.dumps(constants, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "train_rows": len(train),
        "feature_count": len(features.columns),
        "model_count": sum(len(value) for value in model_map.values()),
        "overall_forecast_2025": overall_forecast,
    }, indent=2))


if __name__ == "__main__":
    main()
