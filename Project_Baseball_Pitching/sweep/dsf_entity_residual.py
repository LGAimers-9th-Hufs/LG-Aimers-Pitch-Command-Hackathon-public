# -*- coding: utf-8 -*-
"""Hierarchical same-regime player residual expert for JTT-DSF.

The official as-of rates describe average player quality, but they need not absorb stable
player-specific *shape* errors left by the champion.  This module estimates those errors with
ordered month-forward validation and strongly shrunk frozen lookups.  Inference is a sum of
lookups keyed only by the current row.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import lgbm_family as L  # noqa: E402
import real_data as rd  # noqa: E402
from dsf_pipeline import (  # noqa: E402
    apply_projection, bounded_correction, fold_baseline, projection_coeff, remove_target_axes,
)


OUT_DIR = ROOT / "results" / "dsf_entity"
CAPS = (0.010, 0.015, 0.020, 0.025, 0.030)
GAMMAS = (0.25, 0.375, 0.50, 0.625, 0.75, 1.00)
HISTORY_WEIGHTS = (0.15, 0.30, 0.45, 0.60)
CONFIGS = {
    "pitcher": [("p", ["pitcher_id"], 250.0)],
    "pitcher_batter": [
        ("p", ["pitcher_id"], 250.0),
        ("b", ["batter_id"], 350.0),
    ],
    "hier_context": [
        ("p", ["pitcher_id"], 250.0),
        ("b", ["batter_id"], 350.0),
        ("pc", ["pitcher_id", "count_group"], 500.0),
        ("bc", ["batter_id", "count_group"], 700.0),
        ("ph", ["pitcher_id", "same_hand"], 600.0),
    ],
    "hier_attack": [
        ("p", ["pitcher_id"], 150.0),
        ("b", ["batter_id"], 220.0),
        ("pc", ["pitcher_id", "count_group"], 350.0),
        ("bc", ["batter_id", "count_group"], 500.0),
        ("ph", ["pitcher_id", "same_hand"], 400.0),
    ],
    "hier_attack_x": [
        ("p", ["pitcher_id"], 80.0),
        ("b", ["batter_id"], 120.0),
        ("pc", ["pitcher_id", "count_group"], 200.0),
        ("bc", ["batter_id", "count_group"], 300.0),
        ("ph", ["pitcher_id", "same_hand"], 250.0),
    ],
}


def score(y: np.ndarray, p: np.ndarray) -> float:
    return L.score(np.asarray(y, dtype=float), np.clip(np.asarray(p, dtype=float), 0.0, 1.0))


def key_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["pitcher_id"] = pd.to_numeric(df["pitcher_id"], errors="coerce").fillna(-1).astype("int64")
    out["batter_id"] = pd.to_numeric(df["batter_id"], errors="coerce").fillna(-1).astype("int64")
    balls = pd.to_numeric(df["balls_before"], errors="coerce").fillna(0).to_numpy(dtype=int)
    strikes = pd.to_numeric(df["strikes_before"], errors="coerce").fillna(0).to_numpy(dtype=int)
    out["count_group"] = np.select([strikes > balls, balls > strikes], [0, 2], default=1).astype("int8")
    out["same_hand"] = (
        pd.to_numeric(df["pitcher_hand"], errors="coerce").to_numpy()
        == pd.to_numeric(df["batter_hand"], errors="coerce").to_numpy()
    ).astype("int8")
    return out.reset_index(drop=True)


def _encode_key(row: tuple[Any, ...]) -> str:
    return "|".join(str(int(v)) for v in row)


def fit_lookup(
    keys: pd.DataFrame,
    residual: np.ndarray,
    spec: list[tuple[str, list[str], float]],
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    residual = np.asarray(residual, dtype=float).copy()
    if sample_weight is None:
        sample_weight = np.ones(len(residual), dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float)
    if len(sample_weight) != len(residual):
        raise ValueError("sample_weight row mismatch")
    original = residual.copy()
    components = []
    # Two conservative backfitting passes. Each pass learns only the remaining residual.
    for pass_no in range(2):
        for name, cols, shrink in spec:
            work = keys[cols].copy()
            work["_wr"] = residual * sample_weight
            work["_w"] = sample_weight
            stats = work.groupby(cols, sort=False)[["_wr", "_w"]].sum().reset_index()
            stats["effect"] = stats["_wr"] / (stats["_w"] + float(shrink))
            mapping = {
                _encode_key(tuple(row[c] for c in cols)): float(row["effect"])
                for _, row in stats.iterrows()
            }
            effect = apply_component(keys, {"cols": cols, "mapping": mapping})
            residual -= effect
            components.append({
                "name": f"{name}{pass_no + 1}", "cols": cols, "shrink": float(shrink),
                "mapping": mapping,
            })
    fitted = original - residual
    return {
        "version": "jtt-dsf-entity-v1",
        "components": components,
        "train_correction_rms": float(np.sqrt(np.mean(fitted ** 2))),
        "provenance": "training labels only; frozen row-key lookups",
    }


def apply_component(keys: pd.DataFrame, component: dict[str, Any]) -> np.ndarray:
    cols = component["cols"]
    mapping = component["mapping"]
    tuples = keys[cols].itertuples(index=False, name=None)
    return np.fromiter((mapping.get(_encode_key(row), 0.0) for row in tuples), dtype=float, count=len(keys))


def predict_lookup(keys: pd.DataFrame, artifact: dict[str, Any]) -> np.ndarray:
    out = np.zeros(len(keys), dtype=float)
    for component in artifact["components"]:
        out += apply_component(keys, component)
    return out


def evaluate_split(
    df_season: pd.DataFrame,
    y: np.ndarray,
    q: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    split_name: str,
    history: list[tuple[pd.DataFrame, np.ndarray, int]] | None = None,
    history_weight: float = 0.0,
    configs: dict[str, list[tuple[str, list[str], float]]] | None = None,
    caps: tuple[float, ...] = CAPS,
    gammas: tuple[float, ...] = GAMMAS,
) -> list[dict[str, Any]]:
    keys = key_frame(df_season)
    shape, _ = remove_target_axes(y[train_mask], q[train_mask])
    train_keys = [keys.loc[train_mask].reset_index(drop=True)]
    train_residual = [shape]
    train_weight = [np.ones(len(shape), dtype=float)]
    for hist_keys, hist_residual, age in history or []:
        weight = float(history_weight) ** int(age)
        if weight <= 0:
            continue
        train_keys.append(hist_keys.reset_index(drop=True))
        train_residual.append(np.asarray(hist_residual, dtype=float))
        train_weight.append(np.full(len(hist_residual), weight, dtype=float))
    pooled_keys = pd.concat(train_keys, ignore_index=True)
    pooled_residual = np.concatenate(train_residual)
    pooled_weight = np.concatenate(train_weight)
    rows = []
    for config_name, spec in (configs or CONFIGS).items():
        artifact = fit_lookup(pooled_keys, pooled_residual, spec, pooled_weight)
        raw_train = predict_lookup(keys.loc[train_mask].reset_index(drop=True), artifact)
        raw_val = predict_lookup(keys.loc[val_mask].reset_index(drop=True), artifact)
        for cap in caps:
            projection = projection_coeff(bounded_correction(raw_train, cap), q[train_mask])
            corr = apply_projection(bounded_correction(raw_val, cap), q[val_mask], projection)
            corr = np.clip(corr, -cap, cap)
            residual_val = y[val_mask] - q[val_mask]
            correlation = float(np.corrcoef(residual_val, corr)[0, 1]) if np.std(corr) else 0.0
            coverage = float(np.mean(np.abs(raw_val) > 0))
            for gamma in gammas:
                pred = q[val_mask] + gamma * corr
                rows.append({
                    "split": split_name,
                    "config": config_name,
                    "cap": cap,
                    "gamma": gamma,
                    "history_weight": history_weight,
                    "gain": score(y[val_mask], pred) - score(y[val_mask], q[val_mask]),
                    "corr_residual": correlation,
                    "coverage": coverage,
                    "n_train": int(train_mask.sum()),
                    "n_val": int(val_mask.sum()),
                })
    return rows


def validate(df: pd.DataFrame, refine: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    if refine:
        configs = {"hier_context": CONFIGS["hier_context"]}
        caps = (0.050, 0.075, 0.100, 0.150)
        gammas = (0.20, 0.25, 0.30, 0.35, 0.375, 0.40)
        history_weights = (0.45, 0.60)
    else:
        configs = CONFIGS
        caps = CAPS
        gammas = GAMMAS
        history_weights = HISTORY_WEIGHTS
    rows = []
    by_season = {}
    for season in (2022, 2023, 2024):
        d = df[df[rd.SEASON] == season].reset_index(drop=True)
        y, q, _ = fold_baseline(season, d)
        by_season[season] = (d, y, q)
        month = d["game_month"].to_numpy(dtype=int)
        for cutoff in (6, 8):
            train_mask = month <= cutoff
            val_mask = month > cutoff
            if train_mask.sum() < 10000 or val_mask.sum() < 10000:
                continue
            history = []
            for old in sorted(s for s in by_season if s < season):
                od, oy, oq = by_season[old]
                old_shape, _ = remove_target_axes(oy, oq)
                history.append((key_frame(od), old_shape, season - old))
            for history_weight in history_weights:
                rows += evaluate_split(
                    d, y, q, train_mask, val_mask, f"within_{season}_m{cutoff}",
                    history=history, history_weight=history_weight,
                    configs=configs, caps=caps, gammas=gammas,
                )

    # Stable pre-ABS cross-season check. Artifact uses all 2022 labels, prediction uses 2023 rows only.
    d22, y22, q22 = by_season[2022]
    d23, y23, q23 = by_season[2023]
    k22, k23 = key_frame(d22), key_frame(d23)
    shape22, _ = remove_target_axes(y22, q22)
    for history_weight in history_weights:
        for config_name, spec in configs.items():
            artifact = fit_lookup(k22, shape22, spec)
            raw_tr = predict_lookup(k22, artifact)
            raw_val = predict_lookup(k23, artifact)
            for cap in caps:
                projection = projection_coeff(bounded_correction(raw_tr, cap), q22)
                corr = apply_projection(bounded_correction(raw_val, cap), q23, projection)
                corr = np.clip(corr, -cap, cap)
                correlation = float(np.corrcoef(y23 - q23, corr)[0, 1]) if np.std(corr) else 0.0
                for gamma in gammas:
                    rows.append({
                        "split": "cross_2022_2023", "config": config_name,
                        "cap": cap, "gamma": gamma, "history_weight": history_weight,
                        "gain": score(y23, q23 + gamma * corr) - score(y23, q23),
                        "corr_residual": correlation,
                        "coverage": float(np.mean(np.abs(raw_val) > 0)),
                        "n_train": len(d22), "n_val": len(d23),
                    })

    table = pd.DataFrame(rows)
    group_keys = ["config", "cap", "gamma", "history_weight"]
    summary = table.groupby(group_keys, as_index=False).agg(
        gain_mean=("gain", "mean"), gain_min=("gain", "min"),
        positive=("gain", lambda s: int((s > 0).sum())),
        n_splits=("gain", "size"), corr_mean=("corr_residual", "mean"),
    )
    attack_weights = {
        "within_2024_m6": 0.25,
        "within_2024_m8": 0.25,
        "cross_2022_2023": 0.25,
        "within_2023_m6": 0.075,
        "within_2023_m8": 0.075,
        "within_2022_m6": 0.05,
        "within_2022_m8": 0.05,
    }
    attack_objective = []
    latest_mean = []
    cross_gain = []
    for _, cell in summary.iterrows():
        mask = np.ones(len(table), dtype=bool)
        for key in group_keys:
            mask &= table[key].to_numpy() == cell[key]
        gains = table.loc[mask].set_index("split")["gain"]
        attack_objective.append(float(sum(attack_weights[s] * gains.loc[s] for s in attack_weights)))
        latest_mean.append(float(gains[["within_2024_m6", "within_2024_m8"]].mean()))
        cross_gain.append(float(gains.loc["cross_2022_2023"]))
    summary["attack_objective"] = attack_objective
    summary["within_2024_mean"] = latest_mean
    summary["cross_gain"] = cross_gain
    eligible = summary[(summary["within_2024_mean"] > 0) & (summary["cross_gain"] > 0)]
    if eligible.empty:
        eligible = summary
    summary = summary.sort_values(
        ["attack_objective", "within_2024_mean", "gain_mean"], ascending=False
    )
    best = eligible.sort_values(
        ["attack_objective", "within_2024_mean", "gain_mean"], ascending=False
    ).iloc[0]
    selected_rows = table[
        (table["config"] == best["config"])
        & (table["cap"] == best["cap"])
        & (table["gamma"] == best["gamma"])
        & (table["history_weight"] == best["history_weight"])
    ]
    within24 = selected_rows[selected_rows["split"].str.startswith("within_2024")]["gain"]
    cross = selected_rows[selected_rows["split"] == "cross_2022_2023"]["gain"]
    decision = {
        "go": bool(float(best["attack_objective"]) >= 90.0),
        "incremental_go": bool(
            len(within24) == 2 and float(within24.mean()) > 0
            and len(cross) == 1 and float(cross.iloc[0]) > 0
        ),
        "config": str(best["config"]), "cap": float(best["cap"]),
        "gamma": float(best["gamma"]), "gain_mean": float(best["gain_mean"]),
        "history_weight": float(best["history_weight"]),
        "attack_objective": float(best["attack_objective"]),
        "gain_min": float(best["gain_min"]), "positive": int(best["positive"]),
        "n_splits": int(best["n_splits"]),
        "within_2024_mean": float(within24.mean()) if len(within24) else None,
        "cross_2022_2023": float(cross.iloc[0]) if len(cross) else None,
        "promotion_rule": (
            "maximize attack objective: 50% 2024 within-season, 25% stable cross-season, "
            "25% older within-season; require latest and cross positive"
        ),
    }
    print("\n[summary]")
    print(summary.head(16).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print("\n[selected split results]")
    print(selected_rows.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print("\n[decision]")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return table, decision


def fit_final(df: pd.DataFrame, decision: dict[str, Any]) -> Path:
    """2024 OOF champion residual로 2025 frozen entity lookup을 만든다.

    ``incremental_go``가 아니면 artifact 생성을 거부한다. ``go``는 1100 목표용 강한 gate이고,
    incremental gate는 기존 JTT를 보존한 별도 후보를 만들기 위한 조건이다.
    """
    if not decision.get("incremental_go"):
        raise RuntimeError("entity residual incremental gate is STOP")
    season = 2024
    rows = df[df[rd.SEASON] == season].reset_index(drop=True)
    y, q, _ = fold_baseline(season, rows)
    shape, target_axes = remove_target_axes(y, q)
    config = str(decision["config"])
    pooled_keys = [key_frame(rows)]
    pooled_residual = [shape]
    pooled_weight = [np.ones(len(shape), dtype=float)]
    history_weight = float(decision.get("history_weight", 0.0))
    for old in (2023, 2022):
        old_rows = df[df[rd.SEASON] == old].reset_index(drop=True)
        old_y, old_q, _ = fold_baseline(old, old_rows)
        old_shape, _ = remove_target_axes(old_y, old_q)
        weight = history_weight ** (season - old)
        if weight > 0:
            pooled_keys.append(key_frame(old_rows))
            pooled_residual.append(old_shape)
            pooled_weight.append(np.full(len(old_shape), weight, dtype=float))
    artifact = fit_lookup(
        pd.concat(pooled_keys, ignore_index=True), np.concatenate(pooled_residual),
        CONFIGS[config], np.concatenate(pooled_weight),
    )
    raw = predict_lookup(key_frame(rows), artifact)
    cap = float(decision["cap"])
    projection = projection_coeff(bounded_correction(raw, cap), q)
    payload = {
        "version": "jtt-dsf-entity-final-v1",
        "training_season": season,
        "config": config,
        "cap": cap,
        "gamma": float(decision["gamma"]),
        "projection": [float(projection[0]), float(projection[1])],
        "target_axes_removed": [float(target_axes[0]), float(target_axes[1])],
        "artifact": artifact,
        "validation": decision,
        "provenance": (
            "2024 official train labels against chronological OOF exact-family deployment-shape residual; "
            "frozen row-key lookup; no evaluation aggregation"
        ),
    }
    out = OUT_DIR / "entity_final.json"
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"saved {out} ({out.stat().st_size / 1e6:.2f} MB)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--fit-final", action="store_true")
    ap.add_argument("--attack-refine", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(">> official train")
    df = rd.load_train()
    table, decision = validate(df, refine=args.attack_refine)
    table.to_json(OUT_DIR / "forward_gate.json", orient="records", indent=2)
    (OUT_DIR / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.fit_final:
        fit_final(df, decision)


if __name__ == "__main__":
    main()
