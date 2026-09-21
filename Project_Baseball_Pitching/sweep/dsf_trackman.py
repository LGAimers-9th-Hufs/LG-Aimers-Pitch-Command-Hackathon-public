# -*- coding: utf-8 -*-
"""Cutoff-safe TrackMan soft entity profile for JTT-DSF.

기존 hard match는 확실한 한 후보만 남긴다. 이 모듈은 동일한 train/TrackMan 행동 지문 비용을
이용하되 상위 후보를 posterior로 보존하고, 물리 프로필의 평균·분산·entropy를 frozen lookup으로
만든다. 평가 시에는 pitcher_id 한 개를 조회할 뿐이라 test 행 독립성을 유지한다.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

import trackman as TM


PROFILE_FEATURES = tuple(TM.TM_FEATURES)


def _softmax_cost(cost: np.ndarray, temperature: float) -> np.ndarray:
    z = -(np.asarray(cost, dtype=float) - float(np.nanmin(cost))) / max(float(temperature), 1e-6)
    z = np.clip(z, -40.0, 0.0)
    w = np.exp(z)
    return w / max(w.sum(), 1e-12)


def _temperature(cost: np.ndarray, finite: np.ndarray) -> float:
    gaps = []
    for i in range(len(cost)):
        v = np.sort(cost[i, finite[i]])
        if len(v) >= 2 and np.isfinite(v[1] - v[0]) and v[1] > v[0]:
            gaps.append(float(v[1] - v[0]))
    if not gaps:
        return 0.25
    # 전형적 1·2위 gap에서 top1:top2 odds가 약 4:1이 되도록 한다.
    return float(max(np.median(gaps) / np.log(4.0), 0.05))


def _candidate_indices(
    row_cost: np.ndarray,
    assigned: int | None,
    top_k: int,
) -> np.ndarray:
    finite = np.where(np.isfinite(row_cost) & (row_cost < TM.BIG / 2))[0]
    if not len(finite):
        return finite
    order = finite[np.argsort(row_cost[finite], kind="mergesort")[:top_k]]
    if assigned is not None and assigned in finite and assigned not in order:
        order = np.concatenate([[assigned], order[: max(top_k - 1, 0)]])
    return np.unique(order)


def build_soft_profile_artifacts(
    train: pd.DataFrame,
    tm: pd.DataFrame,
    targets: Iterable[int],
    top_k: int = 5,
    temperature: float | None = None,
) -> dict[int, dict[str, Any]]:
    """각 target 시즌 직전까지만 사용한 소프트 프로필 artifact를 만든다."""
    targets = sorted({int(v) for v in targets})
    tr_app = TM.add_appearances(train)
    sig_tr_all = TM.season_signatures_train(tr_app)
    sig_tm_all = TM.season_signatures_tm(tm)
    cells, rows = TM._measure_cells(tm)
    artifacts: dict[int, dict[str, Any]] = {}

    for target in targets:
        tr_mask = sig_tr_all.index.get_level_values(1) < target
        tm_mask = sig_tm_all.index.get_level_values(1) < target
        sig_tr = sig_tr_all.loc[tr_mask].copy()
        sig_tm = sig_tm_all.loc[tm_mask].copy()
        if sig_tr.empty or sig_tm.empty:
            artifacts[target] = {
                "version": "jtt-dsf-soft-tm-v1",
                "target_season": target,
                "features": list(PROFILE_FEATURES),
                "profiles": {},
                "temperature": None,
            }
            continue

        ents_tr = TM._entity_tables(sig_tr)
        ents_tm = TM._entity_tables(sig_tm)
        ents = (ents_tr, ents_tm)
        e_tr, e_tm = ents_tr[0], ents_tm[0]
        seasons = sorted(
            set(sig_tr.index.get_level_values(1)) & set(sig_tm.index.get_level_values(1))
        )
        hand_map = {2: "Right", 1: "Left"}
        cost, _ = TM.build_cost(sig_tr, sig_tm, hand_map, seasons=seasons, ents=ents)
        rr, cc = TM.assign(cost)
        assigned = {int(i): int(j) for i, j in zip(rr, cc)}
        finite = np.isfinite(cost) & (cost < TM.BIG / 2)
        tau = float(temperature) if temperature is not None else _temperature(cost, finite)
        tm_profile = TM._prefix_features(cells, rows, target)
        profiles: dict[str, list[float]] = {}

        for i, pid in enumerate(e_tr):
            cand = _candidate_indices(cost[i], assigned.get(i), top_k)
            if not len(cand):
                continue
            cids = e_tm[cand]
            available = np.array([cid in tm_profile.index for cid in cids], dtype=bool)
            if not available.any():
                continue
            cand = cand[available]
            cids = e_tm[cand]
            weights = _softmax_cost(cost[i, cand], tau)
            mat = tm_profile.reindex(cids).to_numpy(dtype=float)
            means = np.empty(mat.shape[1], dtype=float)
            variances = np.empty(mat.shape[1], dtype=float)
            coverage = np.empty(mat.shape[1], dtype=float)
            for j in range(mat.shape[1]):
                ok = np.isfinite(mat[:, j])
                coverage[j] = float(weights[ok].sum())
                if not ok.any():
                    means[j] = np.nan
                    variances[j] = np.nan
                    continue
                w = weights[ok] / max(weights[ok].sum(), 1e-12)
                means[j] = float(np.sum(w * mat[ok, j]))
                variances[j] = float(np.sum(w * (mat[ok, j] - means[j]) ** 2))
            entropy = float(-np.sum(weights * np.log(np.maximum(weights, 1e-12))))
            norm_entropy = entropy / max(np.log(max(len(weights), 2)), 1e-12)
            sorted_cost = np.sort(cost[i, cand])
            margin = float(sorted_cost[1] - sorted_cost[0]) if len(sorted_cost) > 1 else 0.0
            vector = np.concatenate(
                [means, variances, [norm_entropy, weights.max(), margin, coverage.mean(), len(weights)]]
            )
            profiles[str(int(pid))] = [float(v) if np.isfinite(v) else None for v in vector]

        artifacts[target] = {
            "version": "jtt-dsf-soft-tm-v1",
            "target_season": target,
            "features": list(PROFILE_FEATURES),
            "top_k": int(top_k),
            "temperature": tau,
            "profiles": profiles,
            "provenance": "train/trackman rows with season < target; posterior lookup is frozen",
        }
    return artifacts


def transform_soft_profile(df: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    if "season" in df and len(df):
        seasons = pd.to_numeric(df["season"], errors="coerce").dropna().unique()
        if len(seasons) and np.any(seasons.astype(int) != int(artifact["target_season"])):
            raise ValueError(
                f"artifact target={artifact['target_season']} but rows contain {seasons.tolist()}"
            )
    n_phys = len(artifact.get("features", PROFILE_FEATURES))
    columns = (
        [f"ds_tm_{c}_mean" for c in artifact.get("features", PROFILE_FEATURES)]
        + [f"ds_tm_{c}_var" for c in artifact.get("features", PROFILE_FEATURES)]
        + ["ds_tm_entropy", "ds_tm_top1", "ds_tm_margin", "ds_tm_coverage", "ds_tm_candidates"]
    )
    out = np.full((len(df), len(columns)), np.nan, dtype=np.float32)
    profiles = artifact.get("profiles", {})
    ids = pd.to_numeric(df["pitcher_id"], errors="coerce").to_numpy(dtype=float)
    for i, pid in enumerate(ids):
        if not np.isfinite(pid):
            continue
        value = profiles.get(str(int(pid)))
        if value is None:
            continue
        out[i] = np.array([np.nan if v is None else v for v in value], dtype=np.float32)
    # 프로필이 없는 행은 불확실성을 명시적으로 최대값으로 둔다. 물리량 자체는 NaN 유지.
    missing = np.isnan(out[:, 2 * n_phys])
    out[missing, 2 * n_phys] = 1.0
    out[missing, 2 * n_phys + 1] = 0.0
    out[missing, 2 * n_phys + 2] = 0.0
    out[missing, 2 * n_phys + 3] = 0.0
    out[missing, 2 * n_phys + 4] = 0.0
    return pd.DataFrame(out, columns=columns, index=df.index).reset_index(drop=True)
