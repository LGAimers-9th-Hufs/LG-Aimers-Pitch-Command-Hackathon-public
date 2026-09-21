"""Run the serving contract checks and finalize MANIFEST.json/SELFTEST.md."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


SERVING_DIR = Path(__file__).resolve().parent
PARTNER_DIR = SERVING_DIR.parent
PROJECT_DIR = PARTNER_DIR.parent.parent
sys.path.insert(0, str(SERVING_DIR))
sys.path.insert(0, str(PARTNER_DIR))

import predict as serving_predict  # noqa: E402
import train_partner as tp  # noqa: E402


PROXY_ROWS = 245789
FORBIDDEN = {
    "mean", "groupby", "value_counts", "transform", "rolling",
    "expanding", "cumsum", "rank",
}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def static_check(path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    ast_hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in FORBIDDEN:
                ast_hits.append({"line": node.lineno, "call": node.func.attr})
    pattern = re.compile(r"\.(mean|groupby|value_values|value_counts|transform|rolling|expanding|cumsum|rank)\s*\(")
    grep_hits = []
    for number, line in enumerate(source.splitlines(), start=1):
        if pattern.search(line):
            grep_hits.append({"line": number, "text": line.strip()})
    return ast_hits, grep_hits


def main():
    train_path = PROJECT_DIR / "data" / "data" / "train.csv"
    validation_path = PARTNER_DIR / "team_pred_2024.csv"
    frame = pd.read_csv(train_path, usecols=tp.INPUT_COLUMNS, low_memory=False)
    season_2024 = frame.loc[frame["season"] == 2024].reset_index(drop=True)
    test_columns = [name for name in season_2024.columns if name != "control_success"]
    proxy = season_2024.loc[: PROXY_ROWS - 1, test_columns].copy()

    serving_predict._BUNDLE_CACHE.clear()
    started = time.perf_counter()
    proxy_prediction = serving_predict.predict(proxy, str(SERVING_DIR))
    proxy_seconds = time.perf_counter() - started

    started = time.perf_counter()
    full_prediction = serving_predict.predict(season_2024[test_columns], str(SERVING_DIR))
    full_seconds_warm = time.perf_counter() - started

    validation = pd.read_csv(validation_path)
    order_match = bool(validation["row_id"].equals(season_2024["row_id"]))
    correlation = float(np.corrcoef(full_prediction, validation["pred"].to_numpy(dtype=np.float64))[0, 1])

    synthetic = proxy.iloc[:4].copy()
    synthetic.loc[synthetic.index[0], "game_type"] = "UNSEEN"
    synthetic.loc[synthetic.index[1], "pitcher_hand"] = 999
    synthetic.loc[synthetic.index[2], "batter_hand"] = 999
    synthetic.loc[synthetic.index[3], "top_bottom"] = "UNSEEN"
    if "pitcher_id" in synthetic.columns:
        synthetic.loc[synthetic.index[0], "pitcher_id"] = 99999999
    synthetic.loc[synthetic.index[0], "asof_pitcher_success_rate"] = np.nan
    synthetic_prediction = serving_predict.predict(synthetic, str(SERVING_DIR))
    unseen_safe = bool(np.all(np.isfinite(synthetic_prediction)))

    ast_hits, grep_hits = static_check(SERVING_DIR / "predict.py")
    result = {
        "proxy_rows": int(len(proxy_prediction)),
        "proxy_seconds_cold": float(proxy_seconds),
        "proxy_dtype": str(proxy_prediction.dtype),
        "proxy_nan_count": int(np.isnan(proxy_prediction).sum()),
        "proxy_min": float(np.min(proxy_prediction)),
        "proxy_max": float(np.max(proxy_prediction)),
        "proxy_mean": float(np.mean(proxy_prediction)),
        "proxy_std": float(np.std(proxy_prediction)),
        "full_2024_rows": int(len(full_prediction)),
        "full_2024_seconds_warm": float(full_seconds_warm),
        "raw_mean_2024": float(np.mean(full_prediction)),
        "raw_std_2024": float(np.std(full_prediction)),
        "validation_row_order_match": order_match,
        "validation_prediction_correlation": correlation,
        "unseen_category_safe": unseen_safe,
        "forbidden_ast_hits": ast_hits,
        "forbidden_grep_hits": grep_hits,
    }
    result["passed"] = bool(
        result["proxy_rows"] == PROXY_ROWS
        and result["proxy_seconds_cold"] < 60.0
        and result["proxy_dtype"] == "float64"
        and result["proxy_nan_count"] == 0
        and 0.0 <= result["proxy_min"] <= result["proxy_max"] <= 1.0
        and result["validation_prediction_correlation"] >= 0.90
        and result["validation_row_order_match"]
        and result["unseen_category_safe"]
        and len(ast_hits) == 0
        and len(grep_hits) == 0
    )

    results_path = SERVING_DIR / "selftest_results.json"
    results_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    status = "PASS" if result["passed"] else "FAIL"
    selftest_text = f"""# Serving self-test

Overall: **{status}**

## 245,789-row cold-start proxy

- elapsed: {result['proxy_seconds_cold']:.3f} seconds (contract: <60 seconds)
- output rows: {result['proxy_rows']:,}
- dtype: `{result['proxy_dtype']}`
- NaN: {result['proxy_nan_count']}
- range: [{result['proxy_min']:.10f}, {result['proxy_max']:.10f}]
- prediction mean: {result['proxy_mean']:.10f}
- prediction SD: {result['proxy_std']:.10f}

The proxy is the first 245,789 rows of train season 2024 in original order, with
`control_success` removed. Timing includes cold artifact loading and feature construction.

## Full train-2024 reference

- rows: {result['full_2024_rows']:,}
- warm elapsed: {result['full_2024_seconds_warm']:.3f} seconds
- `raw_mean_2024`: {result['raw_mean_2024']:.10f}
- `raw_std_2024`: {result['raw_std_2024']:.10f}
- correlation with R2 `team_pred_2024.csv`: {result['validation_prediction_correlation']:.10f}
- row order exact match: {result['validation_row_order_match']}

The serving model is trained through 2024, whereas the validation model is trained through
2023, so exact equality is neither expected nor required.

## Static and fallback checks

- forbidden AST calls in `predict.py`: {len(ast_hits)}
- forbidden grep matches in `predict.py`: {len(grep_hits)}
- unseen game/hand/top-bottom values and missing pitcher history: {result['unseen_category_safe']}
- custom pickle objects: none; all 18 models are LightGBM text boosters

No test-batch aggregation or post-hoc test calibration is performed.

## Test-harness correction

The first self-test attempt completed both large inference calls but failed while constructing
the synthetic unseen-value fixture: pandas 3.0 rejected a string sentinel in the integer-typed
hand column. The fixture now uses an unseen integer sentinel (`999`). No serving model,
artifact, feature, or prediction logic was changed because of this harness-only correction.
"""
    (SERVING_DIR / "SELFTEST.md").write_text(selftest_text, encoding="utf-8")

    files = sorted(
        path for path in SERVING_DIR.iterdir()
        if path.is_file() and path.name != "MANIFEST.json"
    )
    manifest = {
        "files": [path.name for path in files],
        "sha256": {path.name: file_hash(path) for path in files},
        "trained_through_season": 2024,
        "trained_seasons": [2022, 2023, 2024],
        "recipe": "r2_monotone_hierarchical_two_seed",
        "model_count": 18,
        "raw_mean_2024": result["raw_mean_2024"],
        "raw_std_2024": result["raw_std_2024"],
        "selftest_passed": result["passed"],
    }
    (SERVING_DIR / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
