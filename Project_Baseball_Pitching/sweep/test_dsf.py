# -*- coding: utf-8 -*-
"""Fast synthetic tests for the JTT-DSF row-independence contract."""
from __future__ import annotations

import copy
import json
import unittest

import numpy as np
import pandas as pd

import dsf_features as DF
from dsf_trackman import PROFILE_FEATURES, transform_soft_profile
from dsf_entity_residual import fit_lookup, key_frame, predict_lookup


def synthetic_history() -> pd.DataFrame:
    rows = []
    order = 0
    for season, endpoint in ((2020, 100), (2021, 220), (2022, 350)):
        for pid, offset in ((11, 0), (22, 20)):
            for n in (endpoint - 1, endpoint):
                order += 1
                succ = 0.50 + 0.02 * (pid == 11) + 0.01 * (season - 2020)
                rows.append({
                    "row_id": f"TRAIN_{order:07d}", "season": season,
                    "pitcher_id": pid, "batter_id": pid + 100,
                    "asof_pitcher_n": n + offset,
                    "asof_pitcher_success_rate": succ,
                    "asof_pitcher_reverse_rate": 0.14,
                    "asof_pitcher_middle_rate": 0.16,
                    "asof_pitcher_ball_rate": 0.31,
                    "asof_pitcher_strike_rate": 0.39,
                    "asof_batter_n": n + offset,
                    "asof_batter_success_rate": 0.48 + 0.01 * (pid == 22),
                    "asof_batter_middle_rate": 0.17,
                    "asof_pitcher_pitchmix_n": n + offset,
                    "asof_pitcher_fastball_rate": 0.50,
                    "asof_pitcher_breaking_rate": 0.30,
                    "asof_pitcher_offspeed_rate": 0.20,
                })
    return pd.DataFrame(rows)


class DynamicStateTests(unittest.TestCase):
    def test_future_rows_do_not_change_artifact(self):
        history = synthetic_history()
        a = DF.fit_state_artifact(history, 2022)
        changed = history.copy()
        future = changed["season"] >= 2022
        changed.loc[future, "asof_pitcher_success_rate"] = 0.99
        changed.loc[future, "asof_batter_success_rate"] = 0.01
        b = DF.fit_state_artifact(changed, 2022)
        self.assertEqual(
            json.dumps(a, sort_keys=True, allow_nan=False),
            json.dumps(b, sort_keys=True, allow_nan=False),
        )

    def test_row_order_batch_and_single_prediction_parity(self):
        history = synthetic_history()
        artifact = DF.fit_state_artifact(history, 2022)
        rows = history[history["season"] == 2022].reset_index(drop=True)
        full = DF.transform_state(rows, artifact)

        perm = np.array([3, 0, 2, 1])
        shuffled = DF.transform_state(rows.iloc[perm].reset_index(drop=True), artifact)
        np.testing.assert_allclose(full.iloc[perm].to_numpy(), shuffled.to_numpy(), rtol=0, atol=0)

        pieces = []
        for i in range(len(rows)):
            pieces.append(DF.transform_state(rows.iloc[[i]].reset_index(drop=True), artifact))
        singles = pd.concat(pieces, ignore_index=True)
        np.testing.assert_allclose(full.to_numpy(), singles.to_numpy(), rtol=0, atol=0)

    def test_artifact_target_mismatch_fails_closed(self):
        history = synthetic_history()
        artifact = DF.fit_state_artifact(history, 2022)
        bad = history[history["season"] == 2021]
        with self.assertRaises(ValueError):
            DF.transform_state(bad, artifact)


class SoftTrackmanServingTests(unittest.TestCase):
    def test_lookup_is_row_local(self):
        n = len(PROFILE_FEATURES)
        vector = [float(i) for i in range(n)] + [0.1] * n + [0.2, 0.8, 1.3, 0.9, 3.0]
        artifact = {
            "target_season": 2025,
            "features": list(PROFILE_FEATURES),
            "profiles": {"11": vector},
        }
        rows = pd.DataFrame({"season": [2025, 2025, 2025], "pitcher_id": [11, 99, 11]})
        full = transform_soft_profile(rows, artifact)
        shuffled = transform_soft_profile(rows.iloc[::-1].reset_index(drop=True), artifact)
        np.testing.assert_allclose(
            full.iloc[::-1].to_numpy(), shuffled.to_numpy(), rtol=0, atol=0, equal_nan=True
        )
        self.assertEqual(float(full.loc[1, "ds_tm_entropy"]), 1.0)
        self.assertTrue(np.isnan(full.loc[1, f"ds_tm_{PROFILE_FEATURES[0]}_mean"]))


class EntityLookupTests(unittest.TestCase):
    def test_entity_lookup_is_batch_independent(self):
        rows = pd.DataFrame({
            "pitcher_id": [1, 1, 2, 2], "batter_id": [8, 9, 8, 9],
            "balls_before": [0, 3, 0, 3], "strikes_before": [0, 1, 2, 1],
            "pitcher_hand": [1, 1, 2, 2], "batter_hand": [2, 1, 2, 1],
        })
        keys = key_frame(rows)
        artifact = fit_lookup(keys, np.array([0.1, -0.1, 0.05, -0.05]), [("p", ["pitcher_id"], 2.0)])
        full = predict_lookup(keys, artifact)
        order = np.array([2, 0, 3, 1])
        shuffled = predict_lookup(key_frame(rows.iloc[order].reset_index(drop=True)), artifact)
        np.testing.assert_allclose(full[order], shuffled, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
