"""Self-test the partner serving contract on a 2024 proxy batch."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, os.fspath(HERE))
from predict import predict


def main():
    train = pd.read_csv(ROOT / "data" / "data" / "train.csv")
    proxy = train[train["season"] == 2024].iloc[:245789].drop(columns=["control_success"])
    started = time.perf_counter()
    pred = predict(proxy, os.fspath(HERE))
    elapsed = time.perf_counter() - started
    avg = float(np.sum(pred) / len(pred))
    sd = float(np.sqrt(np.sum(np.square(pred - avg)) / len(pred)))
    expected = pd.read_csv(HERE.parent / "team_pred_2024.csv")
    merged = proxy[["row_id"]].merge(expected, on="row_id", how="left")
    other = merged["pred"].to_numpy(dtype=np.float64)
    ok = np.isfinite(other)
    a = pred[ok] - float(np.sum(pred[ok]) / int(np.sum(ok)))
    b = other[ok] - float(np.sum(other[ok]) / int(np.sum(ok)))
    corr = float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))
    result = {
        "rows": int(len(pred)), "dtype": str(pred.dtype),
        "nan_count": int(np.sum(~np.isfinite(pred))),
        "minimum": float(np.min(pred)), "maximum": float(np.max(pred)),
        "mean": avg, "std": sd, "elapsed_seconds": elapsed,
        "correlation_with_fold_2024": corr,
    }
    assert len(pred) == len(proxy)
    assert result["nan_count"] == 0
    assert result["minimum"] >= 0.0 and result["maximum"] <= 1.0
    assert pred.dtype == np.float64
    (HERE / "selftest_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
