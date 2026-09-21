# -*- coding: utf-8 -*-
"""JTT-DSF chronological training, validation and final artifact builder.

Examples
--------
    python sweep/dsf_pipeline.py --validate
    python sweep/dsf_pipeline.py --validate --with-trackman
    python sweep/dsf_pipeline.py --fit-final

The fold baseline is the frozen ENS-9 chronological cache.  No affine parameter is fitted on
the target fold.  DSF learns only the component of ``y - baseline`` orthogonal to level and
slope, so leaderboard-selected calibration axes are not learned again.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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

import dsf_features as DF  # noqa: E402
import lgbm_family as L  # noqa: E402
import real_data as rd  # noqa: E402
from eb_carrier import deploy_q, ens9_pred, get_members9  # noqa: E402


OUT_DIR = ROOT / "results" / "dsf"
CACHE_DIR = OUT_DIR / "cache"
MODEL_DIR = OUT_DIR / "model"
FOLDS = (2021, 2022, 2023, 2024)
VALID_TARGETS = (2022, 2023, 2024)
CAPS = (0.010, 0.015, 0.020, 0.025, 0.030)
GAMMAS = (0.25, 0.50, 0.75, 1.00)

MODEL_SPECS = {
    "state_l7": {
        "num_leaves": 7,
        "min_data_in_leaf": 5000,
        "lambda_l1": 2.0,
        "lambda_l2": 180.0,
        "feature_fraction": 0.85,
        "num_iterations": 400,
        "seed": 1107,
        "space": "state",
    },
    "full_l7": {
        "num_leaves": 7,
        "min_data_in_leaf": 7000,
        "lambda_l1": 3.0,
        "lambda_l2": 240.0,
        "feature_fraction": 0.70,
        "num_iterations": 350,
        "seed": 1117,
        "space": "full",
    },
}


def score(y: np.ndarray, p: np.ndarray) -> float:
    return L.score(np.asarray(y, dtype=float), np.clip(np.asarray(p, dtype=float), 0.0, 1.0))


def projection_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return np.stack([np.ones(len(q)), q - 0.5], axis=1)


def remove_target_axes(y: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    residual = np.asarray(y, dtype=float) - np.asarray(q, dtype=float)
    a = projection_matrix(q)
    coef = np.linalg.lstsq(a, residual, rcond=None)[0]
    return residual - a @ coef, coef


def projection_coeff(correction: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(projection_matrix(q), np.asarray(correction, dtype=float), rcond=None)[0]


def apply_projection(correction: np.ndarray, q: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return np.asarray(correction, dtype=float) - projection_matrix(q) @ np.asarray(coef, dtype=float)


def recalibration_invariant_gain(y: np.ndarray, q: np.ndarray, correction: np.ndarray) -> float:
    def project(cols: list[np.ndarray]) -> float:
        x = np.column_stack([np.ones(len(y))] + [np.asarray(c, dtype=float) for c in cols])
        beta = np.linalg.lstsq(x, np.asarray(y, dtype=float), rcond=None)[0]
        return score(y, x @ beta)

    return project([q, correction]) - project([q])


def fold_baseline(
    val: int,
    rows: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological exact-family cache in its closest deployment form.

    Historical caches do not contain the later regime-transfer leg.  They do contain the exact
    family, so apply its frozen affine/segment deployment transform whenever fold rows are supplied.
    This avoids learning a correction against raw probabilities and serving it after calibrated JTT.
    """
    members, corr, corr_pm, y = get_members9(val)
    raw = ens9_pred(members, corr, corr_pm)
    if rows is None:
        q = np.clip(raw, 1e-6, 1 - 1e-6)
    else:
        if len(rows) != len(raw):
            raise ValueError(f"fold {val}: rows={len(rows)} baseline={len(raw)}")
        q = deploy_q(raw, seg=rows[rd.SEGMENT].to_numpy())
    member_matrix = np.column_stack([members[k] for k in sorted(members)])
    disagreement = np.std(member_matrix, axis=1)
    return np.asarray(y, dtype=float), q, disagreement


def _cache_path(season: int, with_trackman: bool) -> Path:
    suffix = "tm" if with_trackman else "state"
    return CACHE_DIR / f"fold_{season}_{suffix}_v1.pkl"


def prepare_fold_cache(
    df: pd.DataFrame,
    season: int,
    with_trackman: bool,
    tm_artifact: dict[str, Any] | None = None,
    force: bool = False,
) -> Path:
    path = _cache_path(season, with_trackman)
    if path.exists() and not force:
        return path
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows = df[df[rd.SEASON] == season].reset_index(drop=True)
    y, q, disagreement = fold_baseline(season, rows)
    if len(rows) != len(y):
        raise ValueError(f"fold {season}: rows={len(rows)} baseline={len(y)}")
    state_artifact = DF.fit_state_artifact(df, season)
    state = DF.transform_state(rows, state_artifact)
    base = L.build_features(rows).reset_index(drop=True)
    tm_features = None
    if with_trackman:
        if tm_artifact is None:
            raise ValueError("with_trackman=True requires a cutoff-safe artifact")
        from dsf_trackman import transform_soft_profile

        tm_features = transform_soft_profile(rows, tm_artifact)
    x = DF.build_model_features(base, state, q, disagreement, tm_features)
    payload = {
        "version": "jtt-dsf-fold-v1",
        "season": season,
        "X": x,
        "y": y.astype("float32"),
        "q": q.astype("float32"),
        "disagreement": disagreement.astype("float32"),
        "state_artifact": state_artifact,
        "tm_artifact": tm_artifact,
    }
    pd.to_pickle(payload, path)
    return path


def prepare_caches(df: pd.DataFrame, with_trackman: bool, force: bool = False) -> None:
    tm_artifacts = None
    if with_trackman:
        print(">> cutoff-safe soft TrackMan artifacts")
        import trackman as TM
        from dsf_trackman import build_soft_profile_artifacts

        tm = TM.load_trackman()
        tm_artifacts = build_soft_profile_artifacts(df, tm, FOLDS + (2025,))
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for target, artifact in tm_artifacts.items():
            (OUT_DIR / f"soft_tm_{target}.json").write_text(
                json.dumps(artifact, ensure_ascii=False), encoding="utf-8"
            )
    for season in FOLDS:
        print(f">> fold {season} feature cache")
        prepare_fold_cache(
            df, season, with_trackman,
            None if tm_artifacts is None else tm_artifacts[season], force=force,
        )


def load_fold(season: int, with_trackman: bool) -> dict[str, Any]:
    path = _cache_path(season, with_trackman)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run cache preparation first")
    return pd.read_pickle(path)


def model_columns(x: pd.DataFrame, spec: dict[str, Any]) -> list[str]:
    if spec["space"] == "state":
        return DF.state_expert_columns(x.columns)
    return list(x.columns)


def train_model(
    x: pd.DataFrame,
    residual: np.ndarray,
    sample_weight: np.ndarray,
    spec: dict[str, Any],
):
    import lightgbm as lgb

    params = {
        "objective": "regression",
        "metric": "l2",
        "boosting": "gbdt",
        "learning_rate": 0.025,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
        "force_col_wise": True,
        "verbosity": -1,
        "num_threads": 6,
        **{k: v for k, v in spec.items() if k not in {"num_iterations", "space"}},
    }
    ds = lgb.Dataset(
        x, label=np.asarray(residual, dtype=float), weight=np.asarray(sample_weight, dtype=float),
        free_raw_data=False,
    )
    return lgb.train(params, ds, num_boost_round=int(spec["num_iterations"]))


def training_data(sources: list[int], with_trackman: bool) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    xs, rs, qs, ws = [], [], [], []
    newest = max(sources)
    for season in sources:
        fold = load_fold(season, with_trackman)
        shape_residual, _ = remove_target_axes(fold["y"], fold["q"])
        xs.append(fold["X"])
        rs.append(shape_residual)
        qs.append(np.asarray(fold["q"], dtype=float))
        # 최근 전이의 행당 총 가중치가 더 크되, 시즌 표본 수가 결과를 지배하지 않게 정규화한다.
        season_weight = 0.60 ** (newest - season)
        ws.append(np.full(len(shape_residual), season_weight / len(shape_residual), dtype=float))
    weight = np.concatenate(ws)
    weight *= len(weight) / max(weight.sum(), 1e-12)
    return pd.concat(xs, ignore_index=True), np.concatenate(rs), np.concatenate(qs), weight


def bounded_correction(raw: np.ndarray, cap: float) -> np.ndarray:
    return float(cap) * np.tanh(np.asarray(raw, dtype=float) / float(cap))


def validate(with_trackman: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in VALID_TARGETS:
        sources = [s for s in FOLDS if s < target]
        xtr, residual, qtr, weight = training_data(sources, with_trackman)
        target_fold = load_fold(target, with_trackman)
        for name, spec in MODEL_SPECS.items():
            columns = model_columns(xtr, spec)
            t0 = time.time()
            model = train_model(xtr[columns], residual, weight, spec)
            train_raw = model.predict(xtr[columns], num_threads=6)
            val_raw = model.predict(target_fold["X"][columns], num_threads=6)
            for cap in CAPS:
                tr_cap = bounded_correction(train_raw, cap)
                proj = projection_coeff(tr_cap, qtr)
                val_corr = apply_projection(bounded_correction(val_raw, cap), target_fold["q"], proj)
                val_corr = np.clip(val_corr, -cap, cap)
                corr_r = float(np.corrcoef(
                    np.asarray(target_fold["y"], dtype=float) - np.asarray(target_fold["q"], dtype=float),
                    val_corr,
                )[0, 1])
                oracle_gain = recalibration_invariant_gain(
                    target_fold["y"], target_fold["q"], val_corr
                )
                for gamma in GAMMAS:
                    pred = np.clip(target_fold["q"] + gamma * val_corr, 0.0, 1.0)
                    rows.append({
                        "target": target,
                        "sources": sources,
                        "model": name,
                        "cap": cap,
                        "gamma": gamma,
                        "gain": score(target_fold["y"], pred) - score(target_fold["y"], target_fold["q"]),
                        "gain_recal_invariant": oracle_gain,
                        "corr_residual": corr_r,
                        "correction_rms": float(np.sqrt(np.mean(val_corr ** 2))),
                        "seconds": time.time() - t0,
                    })
            best = max(
                (r for r in rows if r["target"] == target and r["model"] == name),
                key=lambda r: r["gain"],
            )
            print(
                f"[V{target} {name}] best={best['gain']:+.2f} cap={best['cap']:.3f} "
                f"gamma={best['gamma']:.2f} corr={best['corr_residual']:+.4f} "
                f"recal-oracle={best['gain_recal_invariant']:+.2f} [{time.time()-t0:.0f}s]"
            )

    table = pd.DataFrame(rows)
    summary = table.groupby(["model", "cap", "gamma"], as_index=False).agg(
        gain_v22=("gain", lambda s: float(s.iloc[0])),
        gain_mean=("gain", "mean"),
        gain_min=("gain", "min"),
        positive=("gain", lambda s: int((s > 0).sum())),
    )
    weighted = []
    for _, cell in summary.iterrows():
        sub = table[
            (table["model"] == cell["model"])
            & (table["cap"] == cell["cap"])
            & (table["gamma"] == cell["gamma"])
        ].set_index("target")
        weighted.append(float(0.2 * sub.loc[2022, "gain"] + 0.3 * sub.loc[2023, "gain"] + 0.5 * sub.loc[2024, "gain"]))
    summary["gain_weighted"] = weighted
    summary = summary.sort_values(["positive", "gain_min", "gain_weighted"], ascending=False)
    stable = summary[(summary["positive"] == 3) & (summary["gain_min"] > 0)]
    best = stable.iloc[0] if len(stable) else summary.iloc[0]
    decision = {
        "go": bool(
            best["positive"] == 3
            and best["gain_weighted"] >= 90.0
            and best["gain_min"] >= 50.0
        ),
        "model": str(best["model"]),
        "cap": float(best["cap"]),
        "gamma": float(best["gamma"]),
        "gain_weighted": float(best["gain_weighted"]),
        "gain_min": float(best["gain_min"]),
        "positive_folds": int(best["positive"]),
        "promotion_rule": "3/3 positive, weighted >= 90, minimum >= 50",
        "with_trackman": with_trackman,
    }
    print("\n[fixed configuration summary]")
    print(summary.head(16).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print("\n[decision]")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return table, decision


def fit_final(df: pd.DataFrame, decision: dict[str, Any], with_trackman: bool) -> None:
    if not decision.get("go"):
        raise RuntimeError("DSF promotion gate is STOP; refusing to modify the champion")
    sources = list(FOLDS)
    xtr, residual, qtr, weight = training_data(sources, with_trackman)
    spec = MODEL_SPECS[decision["model"]]
    columns = model_columns(xtr, spec)
    model = train_model(xtr[columns], residual, weight, spec)
    raw = model.predict(xtr[columns], num_threads=6)
    cap = float(decision["cap"])
    proj = projection_coeff(bounded_correction(raw, cap), qtr)
    state_2025 = DF.fit_state_artifact(df, 2025)
    tm_2025 = None
    tm_path = OUT_DIR / "soft_tm_2025.json"
    if with_trackman and tm_path.exists():
        tm_2025 = json.loads(tm_path.read_text(encoding="utf-8"))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_DIR / "dsf_residual.txt"))
    metadata = {
        "version": "jtt-dsf-v1",
        "feature_names": columns,
        "model": decision["model"],
        "model_spec": spec,
        "cap": cap,
        "gamma": float(decision["gamma"]),
        "projection": [float(proj[0]), float(proj[1])],
        "state_artifact": state_2025,
        "soft_trackman_artifact": tm_2025,
        "validation": decision,
        "provenance": "chronological OOF residuals; official train/trackman only; no test aggregation",
    }
    (MODEL_DIR / "dsf_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )
    print(f"saved {MODEL_DIR / 'dsf_residual.txt'}")
    print(f"saved {MODEL_DIR / 'dsf_metadata.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--fit-final", action="store_true")
    ap.add_argument("--with-trackman", action="store_true")
    ap.add_argument("--force-cache", action="store_true")
    args = ap.parse_args()
    if not (args.validate or args.fit_final):
        args.validate = True

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(">> official train")
    df = rd.load_train()
    prepare_caches(df, args.with_trackman, force=args.force_cache)
    table, decision = validate(args.with_trackman)
    table.to_json(OUT_DIR / "rolling_gate.json", orient="records", indent=2)
    (OUT_DIR / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.fit_final:
        fit_final(df, decision, args.with_trackman)


if __name__ == "__main__":
    main()
