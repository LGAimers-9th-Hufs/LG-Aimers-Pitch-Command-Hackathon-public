"""Train the second, TrackMan-trajectory blend partner.

The validation contract is strictly forward chained.  A row from season S sees
only TrackMan summaries through S-1, and a validation fold V is fitted only on
main-table labels through V-1.  TrackMan identities are resolved independently
at every cutoff, so later activity cannot improve an earlier fold's matching.

All generated files stay below results/codex_partner2/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial.distance import cdist


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "codex_partner2"
CACHE = OUT / "cache"
SERVING = OUT / "serving"
TRAIN_PATH = ROOT / "data" / "data" / "train.csv"
TRACK_PATH = ROOT / "data" / "data" / "trackman_history.csv"
SEASONS = tuple(range(2019, 2025))
SEED = 260809

MAIN_MATCH_COLS = [
    "season", "game_month", "game_dayofweek", "inning", "top_bottom",
    "game_type", "balls_before", "strikes_before", "outs_before",
    "pitcher_id", "pitcher_hand", "batter_hand", "pitcher_team_id",
    "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]
TRACK_MATCH_COLS = [
    "season", "game_month", "game_dayofweek", "inning", "top_bottom",
    "balls_before", "strikes_before", "outs_before", "pitcher_trackman_id",
    "pitcher_hand", "batter_hand", "pitcher_team", "pitch_type_group",
]
PHYS = [
    "rel_speed", "spin_rate", "induced_vert_break", "horz_break",
    "extension", "rel_height", "rel_side", "zone_speed",
]


def _safe_slope(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 3:
        return np.nan
    xx = x[ok].astype(np.float64)
    yy = y[ok].astype(np.float64)
    xx = xx - xx.sum() / len(xx)
    den = float(np.dot(xx, xx))
    if den <= 1e-12:
        return 0.0
    yy = yy - yy.sum() / len(yy)
    return float(np.dot(xx, yy) / den)


def _entropy(values: np.ndarray) -> float:
    z = values[np.isfinite(values) & (values > 0)]
    if len(z) == 0:
        return np.nan
    z = z / z.sum()
    return float(-np.dot(z, np.log(z)))


def _match_block(df: pl.DataFrame, ids: np.ndarray, id_map: dict[int, int],
                 is_main: bool) -> tuple[np.ndarray, np.ndarray]:
    """Create comparable, season-specific behavioral fingerprints."""
    # presence, volume, month, weekday, inning, side, count, batter hand,
    # minor/major fraction, cumulative three-way pitch mix
    block = 1 + 1 + 12 + 7 + 12 + 2 + 36 + 2 + 1 + 3
    x = np.zeros((len(ids), len(SEASONS) * block), dtype=np.float32)
    counts = np.zeros((len(ids), len(SEASONS)), dtype=np.float32)
    id_col = "pitcher_id" if is_main else "pitcher_trackman_id"
    rid = np.asarray([id_map[int(v)] for v in df[id_col].to_numpy()], dtype=np.int32)
    ss = df["season"].to_numpy().astype(np.int16) - SEASONS[0]
    np.add.at(counts, (rid, ss), 1.0)

    month = np.clip(df["game_month"].to_numpy() - 1, 0, 11)
    dow = np.clip(df["game_dayofweek"].to_numpy(), 0, 6)
    inning = np.clip(df["inning"].to_numpy(), 1, 12) - 1
    side = np.where(np.isin(df["top_bottom"].to_numpy(), ["T", "Top"]), 0, 1)
    balls = np.clip(df["balls_before"].to_numpy(), 0, 3)
    strikes = np.clip(df["strikes_before"].to_numpy(), 0, 2)
    outs = np.clip(df["outs_before"].to_numpy(), 0, 2)
    count = balls * 9 + strikes * 3 + outs
    if is_main:
        batter_hand = np.clip(df["batter_hand"].to_numpy() - 1, 0, 1)
        minor = (df["game_type"].to_numpy() == "F").astype(np.float32)
    else:
        batter_hand = np.where(df["batter_hand"].to_numpy() == "Left", 0, 1)
        teams = df["pitcher_team"].to_numpy()
        minor = np.asarray(
            [str(v).startswith("MIN_") or str(v).startswith("KBO_") for v in teams],
            dtype=np.float32,
        )
    value_blocks = [(month, 12), (dow, 7), (inning, 12), (side, 2),
                    (count, 36), (batter_hand, 2)]

    for si, season in enumerate(SEASONS):
        base = si * block
        den = np.maximum(counts[:, si], 1.0)
        mask = ss == si
        x[:, base] = counts[:, si] > 0
        x[:, base + 1] = np.log1p(counts[:, si]) / 8.0
        cur = base + 2
        for vals, width in value_blocks:
            tmp = np.zeros((len(ids), width), dtype=np.float32)
            np.add.at(tmp, (rid[mask], vals[mask]), 1.0)
            tmp /= den[:, None]
            x[:, cur:cur + width] = tmp
            cur += width
        tmp = np.zeros(len(ids), dtype=np.float32)
        np.add.at(tmp, rid[mask], minor[mask])
        x[:, cur] = tmp / den
        cur += 1

        if is_main:
            last = (
                df.filter(pl.col("season") == season)
                .sort("asof_pitcher_pitchmix_n")
                .group_by(id_col, maintain_order=True)
                .tail(1)
            )
            for row in last.iter_rows(named=True):
                vals = np.asarray([
                    row["asof_pitcher_fastball_rate"],
                    row["asof_pitcher_breaking_rate"],
                    row["asof_pitcher_offspeed_rate"],
                ], dtype=np.float64)
                if np.all(np.isfinite(vals)):
                    x[id_map[int(row[id_col])], cur:cur + 3] = vals
        else:
            mix = (
                df.filter(pl.col("season") <= season)
                .group_by([id_col, "pitch_type_group"])
                .len()
            )
            pitch_index = {"fastball": 0, "breaking": 1, "offspeed": 2}
            for row in mix.iter_rows(named=True):
                group = row["pitch_type_group"]
                if group in pitch_index:
                    x[id_map[int(row[id_col])], cur + pitch_index[group]] = row["len"]
            z = x[:, cur:cur + 3]
            dz = z.sum(axis=1)
            ok = dz > 0
            z[ok] /= dz[ok, None]

    weights = np.ones(x.shape[1], dtype=np.float32)
    for si in range(len(SEASONS)):
        base = si * block
        weights[base] = 0.5
        weights[base + 1] = 2.0
        cur = base + 2
        for width, weight in [(12, 1.5), (7, 1.5), (12, 0.7), (2, 0.3),
                              (36, 0.7), (2, 0.3), (1, 1.5), (3, 3.0)]:
            weights[cur:cur + width] = weight
            cur += width
    return x * weights, counts


def build_entity_maps(main: pl.DataFrame, track: pl.DataFrame) -> tuple[dict, pd.DataFrame]:
    main_ids = np.sort(main["pitcher_id"].unique().to_numpy())
    track_ids = np.sort(track["pitcher_trackman_id"].unique().to_numpy())
    main_idx = {int(v): i for i, v in enumerate(main_ids)}
    track_idx = {int(v): i for i, v in enumerate(track_ids)}
    xm, cm = _match_block(main, main_ids, main_idx, True)
    xt, ct = _match_block(track, track_ids, track_idx, False)
    block = xm.shape[1] // len(SEASONS)

    mh_frame = main.group_by("pitcher_id").agg(pl.col("pitcher_hand").mode().first())
    th_frame = track.group_by("pitcher_trackman_id").agg(pl.col("pitcher_hand").mode().first())
    mh = dict(zip(mh_frame["pitcher_id"].to_list(), mh_frame["pitcher_hand"].to_list()))
    th = dict(zip(th_frame["pitcher_trackman_id"].to_list(), th_frame["pitcher_hand"].to_list()))

    all_maps: dict[str, dict[str, dict[str, float | int]]] = {}
    diagnostics: list[dict] = []
    for cutoff in SEASONS:
        width = (cutoff - SEASONS[0] + 1) * block
        active_m = cm[:, :cutoff - SEASONS[0] + 1].sum(axis=1) > 0
        active_t = ct[:, :cutoff - SEASONS[0] + 1].sum(axis=1) > 0
        cutoff_map: dict[str, dict[str, float | int]] = {}
        for main_hand, track_hand in [(1, "Left"), (2, "Right")]:
            aa = np.asarray([i for i, v in enumerate(main_ids)
                             if active_m[i] and mh[int(v)] == main_hand], dtype=np.int32)
            bb = np.asarray([i for i, v in enumerate(track_ids)
                             if active_t[i] and th[int(v)] == track_hand], dtype=np.int32)
            if len(aa) == 0 or len(bb) < 2:
                continue
            dist = cdist(xm[aa, :width], xt[bb, :width], metric="euclidean")
            order = np.argsort(dist, axis=1)[:, :2]
            reverse = np.argmin(dist, axis=0)
            for local_i, (q1, q2) in enumerate(order):
                d1 = float(dist[local_i, q1])
                d2 = float(dist[local_i, q2])
                ratio = d1 / max(d2, 1e-12)
                mutual = bool(reverse[q1] == local_i)
                # Very close fingerprints are accepted even if a tiny-sample alias
                # steals mutual-NN status.  Everything else must be mutual and have
                # a visible margin over the runner-up.
                accepted = bool((d1 < 0.45 and ratio < 0.95) or (mutual and ratio < 0.98))
                pid = int(main_ids[aa[local_i]])
                tid = int(track_ids[bb[q1]])
                confidence = float(max(0.0, min(1.0, 1.0 - ratio)))
                row = {
                    "cutoff": cutoff, "pitcher_id": pid, "pitcher_trackman_id": tid,
                    "distance": d1, "ratio": ratio, "mutual": int(mutual),
                    "accepted": int(accepted), "confidence": confidence,
                    "main_rows": int(cm[main_idx[pid], :cutoff - 2018].sum()),
                    "track_rows": int(ct[track_idx[tid], :cutoff - 2018].sum()),
                }
                diagnostics.append(row)
                if accepted:
                    cutoff_map[str(pid)] = {
                        "trackman_id": tid, "confidence": confidence,
                        "distance": d1, "ratio": ratio,
                    }
        all_maps[str(cutoff)] = cutoff_map
    return all_maps, pd.DataFrame(diagnostics)


def build_trajectory_seasons(track: pl.DataFrame) -> pd.DataFrame:
    """Turn pitch sequences into appearance- and season-trajectory descriptors."""
    phys_cols = PHYS + ["auto_pitch_type", "game_date", "trackman_game_id", "pitch_no"]
    have = [c for c in phys_cols if c not in track.columns]
    if have:
        raise ValueError(f"TrackMan columns missing: {have}")
    t = track.with_columns([
        pl.col("game_date").str.strptime(pl.Date, "%m/%d/%Y", strict=False).alias("date"),
        pl.when(pl.col("pitch_type_group").is_in(["fastball", "breaking", "offspeed"]))
        .then(pl.col("pitch_type_group")).otherwise(pl.lit("other")).alias("pitch_group"),
    ]).sort(["pitcher_trackman_id", "trackman_game_id", "pitch_no"])
    t = t.with_columns(
        pl.col("pitch_group").shift(1)
        .over(["pitcher_trackman_id", "trackman_game_id"])
        .alias("prev_pitch_group")
    )

    seq = t.filter(pl.col("prev_pitch_group").is_not_null()).group_by(
        ["pitcher_trackman_id", "season"]
    ).agg([
        pl.len().alias("transition_n"),
        (pl.col("pitch_group") == pl.col("prev_pitch_group")).mean().alias("repeat_rate"),
        ((pl.col("prev_pitch_group") == "fastball") & (pl.col("pitch_group") == "breaking")).mean().alias("fb_to_br_rate"),
        ((pl.col("prev_pitch_group") == "breaking") & (pl.col("pitch_group") == "fastball")).mean().alias("br_to_fb_rate"),
        ((pl.col("prev_pitch_group") == "offspeed") & (pl.col("pitch_group") == "fastball")).mean().alias("os_to_fb_rate"),
    ])

    keys = ["pitcher_trackman_id", "season", "game_date", "trackman_game_id"]
    agg_expr = [pl.len().alias("game_pitches")]
    for group, short in [("fastball", "fb"), ("breaking", "br"),
                         ("offspeed", "os"), ("other", "ot")]:
        agg_expr.append((pl.col("pitch_group") == group).mean().alias(f"{short}_share"))
    for col in PHYS:
        agg_expr.extend([
            pl.col(col).mean().alias(f"{col}_mean"),
            pl.col(col).std().alias(f"{col}_within_sd"),
        ])
    # Fastball-only speed and spin isolate fatigue from repertoire changes.
    for col in ["rel_speed", "spin_rate"]:
        filt = pl.col("pitch_group") == "fastball"
        agg_expr.extend([
            pl.col("pitch_no").filter(filt).count().alias(f"{col}_fb_n"),
            pl.col("pitch_no").filter(filt).sum().alias(f"{col}_fb_x"),
            (pl.col("pitch_no") ** 2).filter(filt).sum().alias(f"{col}_fb_x2"),
            pl.col(col).filter(filt).sum().alias(f"{col}_fb_y"),
            (pl.col("pitch_no") * pl.col(col)).filter(filt).sum().alias(f"{col}_fb_xy"),
        ])
    game = t.group_by(keys).agg(agg_expr).sort(["pitcher_trackman_id", "season", "game_date"])
    gp = game.to_pandas()
    gp["date_ord"] = pd.to_datetime(gp["game_date"], errors="coerce").map(
        lambda v: v.toordinal() if pd.notna(v) else np.nan
    )
    for col in ["rel_speed", "spin_rate"]:
        n = gp[f"{col}_fb_n"].to_numpy(dtype=float)
        sx = gp[f"{col}_fb_x"].to_numpy(dtype=float)
        sx2 = gp[f"{col}_fb_x2"].to_numpy(dtype=float)
        sy = gp[f"{col}_fb_y"].to_numpy(dtype=float)
        sxy = gp[f"{col}_fb_xy"].to_numpy(dtype=float)
        den = sx2 - sx * sx / np.maximum(n, 1.0)
        num = sxy - sx * sy / np.maximum(n, 1.0)
        gp[f"{col}_fatigue_slope"] = np.where((n >= 5) & (den > 0), num / den, np.nan)

    raw = t.group_by(["pitcher_trackman_id", "season"]).agg([
        pl.len().alias("season_pitches"),
        pl.col("trackman_game_id").n_unique().alias("season_games"),
        pl.col("auto_pitch_type").n_unique().alias("auto_types_n"),
        *[(pl.col("pitch_group") == group).mean().alias(f"{short}_share_raw")
          for group, short in [("fastball", "fb"), ("breaking", "br"),
                               ("offspeed", "os"), ("other", "ot")]],
        *[pl.col(col).mean().alias(f"{col}_season_mean") for col in PHYS],
        *[pl.col(col).std().alias(f"{col}_season_sd") for col in PHYS],
    ]).to_pandas()
    seq_pd = seq.to_pandas()

    records: list[dict] = []
    trend_cols = [f"{c}_mean" for c in PHYS] + ["fb_share", "br_share", "os_share"]
    for (pid, season), g in gp.groupby(["pitcher_trackman_id", "season"], sort=False):
        rec: dict[str, float | int] = {
            "pitcher_trackman_id": int(pid), "season": int(season),
            "workload_mean": float(g["game_pitches"].mean()),
            "workload_sd": float(g["game_pitches"].std(ddof=0)),
            "workload_max": float(g["game_pitches"].max()),
        }
        date = g["date_ord"].to_numpy(dtype=float)
        if np.isfinite(date).any():
            lo = np.nanmin(date)
            hi = np.nanmax(date)
            date = (date - lo) / max(hi - lo, 1.0)
        for col in PHYS:
            rec[f"{col}_between_sd"] = float(g[f"{col}_mean"].std(ddof=0))
            rec[f"{col}_within_sd_mean"] = float(g[f"{col}_within_sd"].mean())
        for col in trend_cols:
            rec[f"{col}_date_slope"] = _safe_slope(date, g[col].to_numpy(dtype=float))
        for col in ["fb_share", "br_share", "os_share"]:
            rec[f"{col}_game_sd"] = float(g[col].std(ddof=0))
        for col in ["rel_speed", "spin_rate"]:
            rec[f"{col}_fatigue_mean"] = float(g[f"{col}_fatigue_slope"].mean())
            rec[f"{col}_fatigue_sd"] = float(g[f"{col}_fatigue_slope"].std(ddof=0))
        records.append(rec)
    season = pd.DataFrame(records).merge(raw, on=["pitcher_trackman_id", "season"], how="left")
    season = season.merge(seq_pd, on=["pitcher_trackman_id", "season"], how="left")
    mix_cols = ["fb_share_raw", "br_share_raw", "os_share_raw", "ot_share_raw"]
    season["repertoire_entropy"] = season[mix_cols].apply(
        lambda row: _entropy(row.to_numpy(dtype=float)), axis=1
    )
    season["release_area"] = season["rel_height_season_sd"] * season["rel_side_season_sd"]
    season["speed_loss_mean"] = season["rel_speed_season_mean"] - season["zone_speed_season_mean"]
    return season.sort_values(["pitcher_trackman_id", "season"]).reset_index(drop=True)


def build_snapshots(season: pd.DataFrame) -> tuple[dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]], list[str]]:
    excluded_direct = {
        "pitcher_trackman_id", "season",
        *[f"{c}_season_mean" for c in PHYS],
        "fb_share_raw", "br_share_raw", "os_share_raw", "ot_share_raw",
    }
    direct = [c for c in season.columns if c not in excluded_direct]
    delta_source = [f"{c}_season_mean" for c in PHYS] + [
        "fb_share_raw", "br_share_raw", "os_share_raw", "ot_share_raw"
    ]
    feature_names = [f"tm_{c}" for c in direct] + [f"tm_delta_{c}" for c in delta_source] + [
        "tm_years_since", "tm_history_seasons"
    ]
    snapshots: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for cutoff in SEASONS:
        rows: list[np.ndarray] = []
        tids: list[int] = []
        for tid, hist in season[season["season"] <= cutoff].groupby("pitcher_trackman_id", sort=False):
            hist = hist.sort_values("season")
            last = hist.iloc[-1]
            prev = hist.iloc[-2] if len(hist) >= 2 else None
            vals = [float(last[c]) if pd.notna(last[c]) else np.nan for c in direct]
            for col in delta_source:
                if prev is None or pd.isna(last[col]) or pd.isna(prev[col]):
                    vals.append(np.nan)
                else:
                    vals.append(float(last[col] - prev[col]))
            vals.extend([float(cutoff - int(last["season"])), float(len(hist))])
            tids.append(int(tid))
            rows.append(np.asarray(vals, dtype=np.float32))
        matrix = np.vstack(rows).astype(np.float32)
        default = np.nanmedian(matrix, axis=0).astype(np.float32)
        bad = ~np.isfinite(default)
        default[bad] = 0.0
        matrix = np.where(np.isfinite(matrix), matrix, default[None, :]).astype(np.float32)
        snapshots[cutoff] = (np.asarray(tids, dtype=np.int64), matrix, default)
    return snapshots, feature_names


def prepare() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    print("reading matching columns", flush=True)
    main = pl.read_csv(TRAIN_PATH, columns=MAIN_MATCH_COLS)
    track_match = pl.read_csv(TRACK_PATH, columns=TRACK_MATCH_COLS)
    maps, diagnostics = build_entity_maps(main, track_match)
    (CACHE / "entity_maps.json").write_text(json.dumps(maps, ensure_ascii=False), encoding="utf-8")
    diagnostics.to_csv(CACHE / "entity_resolution.csv", index=False)

    print("reading TrackMan trajectory columns", flush=True)
    track_cols = list(dict.fromkeys(TRACK_MATCH_COLS + [
        "game_date", "trackman_game_id", "pitch_no", "auto_pitch_type", *PHYS
    ]))
    track = pl.read_csv(TRACK_PATH, columns=track_cols)
    trajectory = build_trajectory_seasons(track)
    trajectory.to_csv(CACHE / "trajectory_seasons.csv", index=False)
    snapshots, names = build_snapshots(trajectory)
    arrays: dict[str, np.ndarray] = {"feature_names": np.asarray(names)}
    for cutoff, (ids, matrix, default) in snapshots.items():
        arrays[f"ids_{cutoff}"] = ids
        arrays[f"values_{cutoff}"] = matrix
        arrays[f"default_{cutoff}"] = default
    np.savez_compressed(CACHE / "track_snapshots.npz", **arrays)

    val_counts = main.filter(pl.col("season").is_in([2021, 2022, 2023, 2024])).group_by(
        ["season", "pitcher_id"]
    ).len().to_pandas()
    coverage = []
    for season in [2021, 2022, 2023, 2024]:
        cutoff = season - 1
        accepted = {int(k) for k in maps[str(cutoff)]}
        sub = val_counts[val_counts["season"] == season]
        hit = sub[sub["pitcher_id"].isin(accepted)]["len"].sum()
        coverage.append({
            "validation_season": season, "cutoff": cutoff,
            "matched_pitchers": len(accepted), "row_coverage": float(hit / sub["len"].sum()),
        })
    pd.DataFrame(coverage).to_csv(CACHE / "coverage.csv", index=False)
    print(pd.DataFrame(coverage).to_string(index=False), flush=True)


def load_cache() -> tuple[dict, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]], list[str]]:
    maps = json.loads((CACHE / "entity_maps.json").read_text(encoding="utf-8"))
    z = np.load(CACHE / "track_snapshots.npz", allow_pickle=False)
    names = [str(v) for v in z["feature_names"]]
    snapshots = {
        cutoff: (z[f"ids_{cutoff}"], z[f"values_{cutoff}"], z[f"default_{cutoff}"])
        for cutoff in SEASONS
    }
    return maps, snapshots, names


def tm_lookup_for_cutoff(cutoff: int, maps: dict,
                         snapshots: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[dict[int, np.ndarray], np.ndarray]:
    tids, values, default = snapshots[cutoff]
    by_tid = {int(tid): values[i] for i, tid in enumerate(tids)}
    lookup: dict[int, np.ndarray] = {}
    for pid, info in maps[str(cutoff)].items():
        tid = int(info["trackman_id"])
        if tid in by_tid:
            confidence = float(info["confidence"])
            lookup[int(pid)] = np.concatenate([
                by_tid[tid], np.asarray([confidence, 1.0], dtype=np.float32)
            ])
    fallback = np.concatenate([default, np.asarray([0.0, 0.0], dtype=np.float32)])
    return lookup, fallback


def attach_tm(df: pd.DataFrame, maps: dict,
              snapshots: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
              tm_names: list[str]) -> pd.DataFrame:
    out = np.empty((len(df), len(tm_names) + 2), dtype=np.float32)
    seasons = df["season"].to_numpy(dtype=np.int16)
    pids = df["pitcher_id"].to_numpy(dtype=np.int64)
    for season in np.unique(seasons):
        mask = seasons == season
        cutoff = int(season) - 1
        if cutoff < SEASONS[0]:
            # The 2019 rows intentionally carry no TrackMan history.
            fallback = np.zeros(len(tm_names) + 2, dtype=np.float32)
            out[mask] = fallback
            continue
        lookup, fallback = tm_lookup_for_cutoff(cutoff, maps, snapshots)
        out[mask] = np.vstack([lookup.get(int(pid), fallback) for pid in pids[mask]])
    return pd.DataFrame(out, columns=tm_names + ["tm_match_confidence", "tm_matched"])


def make_context(df: pd.DataFrame, include_history: bool) -> pd.DataFrame:
    n = len(df)
    month = df["game_month"].to_numpy(dtype=np.float32)
    dow = df["game_dayofweek"].to_numpy(dtype=np.float32)
    base_map = {"___": 0, "1__": 1, "_2_": 2, "__3": 3,
                "12_": 4, "1_3": 5, "_23": 6, "123": 7}
    data: dict[str, np.ndarray] = {
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
            vals = df[col].to_numpy(dtype=np.float32)
            if col.endswith("_n"):
                vals = np.log1p(np.maximum(vals, 0))
            data[col] = vals
        ps = df["asof_pitcher_success_rate"].to_numpy(dtype=np.float32)
        pm = df["asof_pitcher_middle_rate"].to_numpy(dtype=np.float32)
        data["relative_pitcher_batter_success"] = ps - df["asof_batter_success_rate"].to_numpy(dtype=np.float32)
        data["relative_pitcher_batter_middle"] = pm - df["asof_batter_middle_rate"].to_numpy(dtype=np.float32)
        for window in [1, 3, 5]:
            data[f"recent{window}_success_gap"] = (
                df[f"asof_pitcher_prev{window}_game_success_rate"].to_numpy(dtype=np.float32) - ps
            )
            data[f"recent{window}_middle_gap"] = (
                df[f"asof_pitcher_prev{window}_game_middle_rate"].to_numpy(dtype=np.float32) - pm
            )
    return pd.DataFrame(data, index=np.arange(n))


def make_features(df: pd.DataFrame, maps: dict, snapshots: dict,
                  tm_names: list[str], include_history: bool) -> pd.DataFrame:
    context = make_context(df, include_history).reset_index(drop=True)
    trajectory = attach_tm(df, maps, snapshots, tm_names).reset_index(drop=True)
    return pd.concat([context, trajectory], axis=1)


def season_rates(df: pd.DataFrame, through: int) -> dict[int, float]:
    sub = df[df["season"] <= through]
    return {int(s): float(g["control_success"].sum() / len(g))
            for s, g in sub.groupby("season", sort=True)}


def forecast_rate(rates: dict[int, float], target: int) -> float:
    years = sorted(y for y in rates if y < target)
    if len(years) <= 2:
        return float(rates[years[-1]])
    use = years[-3:]
    x = np.asarray(use, dtype=np.float64)
    y = np.asarray([rates[v] for v in use], dtype=np.float64)
    slope = float(np.polyfit(x, y, 1)[0])
    intercept = float((y.sum() - slope * x.sum()) / len(x))
    return float(np.clip(intercept + slope * target, 0.40, 0.60))


def centered_target(df: pd.DataFrame, rates: dict[int, float]) -> np.ndarray:
    y = df["control_success"].to_numpy(dtype=np.float32)
    centers = np.asarray([rates[int(s)] for s in df["season"]], dtype=np.float32)
    return y - centers


def model_params(objective: str) -> dict:
    params = {
        "objective": objective, "learning_rate": 0.025, "num_leaves": 31,
        "max_depth": 7, "min_data_in_leaf": 500, "feature_fraction": 0.82,
        "bagging_fraction": 0.82, "bagging_freq": 1, "lambda_l1": 0.4,
        "lambda_l2": 7.0, "max_bin": 127, "verbosity": -1,
        "seed": SEED, "feature_fraction_seed": SEED + 1,
        "bagging_seed": SEED + 2, "data_random_seed": SEED + 3,
        "deterministic": True, "force_col_wise": True, "num_threads": 6,
    }
    if objective == "huber":
        params["alpha"] = 0.85
    return params


def fit_pair(train_df: pd.DataFrame, x: pd.DataFrame, cutoff: int,
             final_fit: bool = False) -> tuple[lgb.Booster, lgb.Booster, dict]:
    rates = season_rates(train_df, cutoff)
    y_res = centered_target(train_df, rates)
    seasons = train_df["season"].to_numpy(dtype=np.int16)
    inner = cutoff
    fit_mask = seasons < inner
    val_mask = seasons == inner
    if fit_mask.sum() < 10000 or val_mask.sum() < 10000:
        fit_mask = seasons <= cutoff
        val_mask = seasons == cutoff

    tm_cols = [c for c in x.columns if c.startswith("tm_")]
    context_cols = [c for c in x.columns if not c.startswith("asof_")]
    hybrid_cols = list(x.columns)
    categorical = [c for c in ["month", "dow", "inning", "top_bottom", "game_type",
                                      "balls", "strikes", "outs", "count_code", "base_state",
                                      "pitcher_hand", "batter_hand", "pitcher_team", "batter_team"]
                   if c in x.columns]

    def early(cols: list[str], objective: str) -> tuple[int, np.ndarray]:
        dtrain = lgb.Dataset(x.loc[fit_mask, cols], label=y_res[fit_mask],
                             categorical_feature=[c for c in categorical if c in cols],
                             free_raw_data=False)
        dvalid = lgb.Dataset(x.loc[val_mask, cols], label=y_res[val_mask],
                             categorical_feature=[c for c in categorical if c in cols],
                             reference=dtrain, free_raw_data=False)
        booster = lgb.train(
            model_params(objective), dtrain, num_boost_round=700,
            valid_sets=[dvalid], callbacks=[lgb.early_stopping(60, verbose=False)]
        )
        it = max(90, int(booster.best_iteration or 220))
        pred = booster.predict(x.loc[val_mask, cols], num_iteration=it)
        return it, pred

    tm_it, tm_val = early(context_cols, "huber")
    hy_it, hy_val = early(hybrid_cols, "regression")
    actual = y_res[val_mask].astype(np.float64)
    best_alpha = 0.0
    best_mse = float("inf")
    best_shape = hy_val
    for alpha in np.linspace(0.0, 1.0, 21):
        shape = (1 - alpha) * hy_val + alpha * tm_val
        den = float(np.dot(shape, shape))
        scale = float(np.dot(shape, actual) / max(den, 1e-12))
        scale = float(np.clip(scale, 0.0, 1.5))
        mse = float(np.square(scale * shape - actual).sum() / len(actual))
        if mse < best_mse:
            best_mse, best_alpha, best_shape = mse, float(alpha), shape
    scale = float(np.dot(best_shape, actual) / max(float(np.dot(best_shape, best_shape)), 1e-12))
    scale = float(np.clip(scale, 0.0, 1.5))
    # Fixed, training-only correction for a systematic residual offset.  This is
    # estimated on the last available labelled season, never on the prediction
    # batch or target season.
    residual_intercept = float((actual - scale * best_shape).sum() / len(actual))

    all_mask = seasons <= cutoff
    def refit(cols: list[str], objective: str, iterations: int) -> lgb.Booster:
        ds = lgb.Dataset(x.loc[all_mask, cols], label=y_res[all_mask],
                         categorical_feature=[c for c in categorical if c in cols],
                         free_raw_data=False)
        return lgb.train(model_params(objective), ds, num_boost_round=iterations)

    tm_model = refit(context_cols, "huber", tm_it)
    hybrid_model = refit(hybrid_cols, "regression", hy_it)
    center_mask = seasons == cutoff
    final_tm_center = tm_model.predict(x.loc[center_mask, context_cols])
    final_hy_center = hybrid_model.predict(x.loc[center_mask, hybrid_cols])
    final_shape = (1.0 - best_alpha) * final_hy_center + best_alpha * final_tm_center
    final_shape_center = float(final_shape.sum() / len(final_shape))
    meta = {
        "cutoff": cutoff, "tm_iterations": tm_it, "hybrid_iterations": hy_it,
        "trajectory_weight": best_alpha, "residual_scale": scale,
        "residual_intercept": 0.0, "inner_residual_intercept": residual_intercept,
        "shape_center": final_shape_center,
        "inner_residual_mse": best_mse, "context_columns": context_cols,
        "hybrid_columns": hybrid_cols, "tm_columns_count": len(tm_cols),
        "rates": {str(k): v for k, v in rates.items()},
    }
    return tm_model, hybrid_model, meta


def predict_pair(tm_model: lgb.Booster, hybrid_model: lgb.Booster, meta: dict,
                 x: pd.DataFrame, target_season: int) -> np.ndarray:
    a = float(meta["trajectory_weight"])
    scale = float(meta["residual_scale"])
    tm_pred = tm_model.predict(x[meta["context_columns"]])
    hy_pred = hybrid_model.predict(x[meta["hybrid_columns"]])
    shape = (1 - a) * hy_pred + a * tm_pred - float(meta.get("shape_center", 0.0))
    rates = {int(k): float(v) for k, v in meta["rates"].items()}
    base = forecast_rate(rates, target_season)
    intercept = float(meta.get("residual_intercept", 0.0))
    return np.clip(base + intercept + scale * shape, 0.001, 0.999).astype(np.float64)


def score(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    r = float(y.sum() / len(y))
    brier = float(np.square(pred - y).sum() / len(y))
    skill = 100000.0 * (1.0 - brier / (r * (1.0 - r)))
    return {
        "n": int(len(y)), "rate": r, "pred_mean": float(pred.sum() / len(pred)),
        "pred_std": float(np.sqrt(np.square(pred - pred.sum() / len(pred)).sum() / len(pred))),
        "brier": brier, "score": skill,
    }


def train_validation_fold(val_season: int) -> dict:
    maps, snapshots, tm_names = load_cache()
    train = pd.read_csv(TRAIN_PATH)
    cutoff = val_season - 1
    used = train[train["season"] <= val_season].reset_index(drop=True)
    x = make_features(used, maps, snapshots, tm_names, include_history=True)
    tm_model, hybrid_model, meta = fit_pair(used[used["season"] <= cutoff].reset_index(drop=True),
                                             x.loc[used["season"] <= cutoff].reset_index(drop=True),
                                             cutoff)
    val_mask = used["season"].to_numpy() == val_season
    pred = predict_pair(tm_model, hybrid_model, meta, x.loc[val_mask].reset_index(drop=True), val_season)
    out = pd.DataFrame({"row_id": used.loc[val_mask, "row_id"].to_numpy(), "pred": pred})
    out.to_csv(OUT / f"team_pred_{val_season}.csv", index=False)
    metrics = score(used.loc[val_mask, "control_success"].to_numpy(dtype=np.float64), pred)
    meta["validation"] = metrics
    (OUT / f"fold_{val_season}_metrics.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"fold": val_season, **metrics,
                      "trajectory_weight": meta["trajectory_weight"],
                      "residual_scale": meta["residual_scale"]}, indent=2), flush=True)
    return meta


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def train_serving() -> None:
    maps, snapshots, tm_names = load_cache()
    train = pd.read_csv(TRAIN_PATH)
    x = make_features(train, maps, snapshots, tm_names, include_history=True)
    tm_model, hybrid_model, meta = fit_pair(train, x, 2024, final_fit=True)
    SERVING.mkdir(parents=True, exist_ok=True)
    tm_model.save_model(str(SERVING / "trajectory_model.txt"))
    hybrid_model.save_model(str(SERVING / "hybrid_model.txt"))

    lookup, fallback = tm_lookup_for_cutoff(2024, maps, snapshots)
    ids = np.asarray(sorted(lookup), dtype=np.int64)
    values = np.vstack([lookup[int(pid)] for pid in ids]).astype(np.float32)
    np.savez_compressed(SERVING / "trajectory_lookup.npz", ids=ids, values=values,
                        fallback=fallback.astype(np.float32),
                        feature_names=np.asarray(tm_names + ["tm_match_confidence", "tm_matched"]))
    serving_meta = dict(meta)
    serving_meta["target_season"] = 2025
    serving_meta["base_rate"] = forecast_rate({int(k): float(v) for k, v in meta["rates"].items()}, 2025)
    (SERVING / "model_meta.json").write_text(
        json.dumps(serving_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    source = OUT / "serving_predict_template.py"
    shutil.copyfile(source, SERVING / "predict.py")
    source_test = OUT / "serving_selftest_template.py"
    shutil.copyfile(source_test, SERVING / "selftest.py")

    # Contract normalization constants: final serving model applied to 2024 rows.
    import importlib.util
    spec = importlib.util.spec_from_file_location("partner2_predict", SERVING / "predict.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    proxy = train[train["season"] == 2024].drop(columns=["control_success"])
    raw = module.predict(proxy, str(SERVING))
    raw_mean = float(raw.sum() / len(raw))
    raw_std = float(np.sqrt(np.square(raw - raw_mean).sum() / len(raw)))
    files = ["predict.py", "trajectory_model.txt", "hybrid_model.txt",
             "trajectory_lookup.npz", "model_meta.json", "selftest.py"]
    manifest = {
        "files": files, "sha256": {f: sha256(SERVING / f) for f in files},
        "trained_through_season": 2024, "raw_mean_2024": raw_mean,
        "raw_std_2024": raw_std, "seed": SEED,
    }
    (SERVING / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--fold", type=int, choices=[2021, 2022, 2023, 2024])
    parser.add_argument("--serving", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        prepare()
    if args.fold:
        train_validation_fold(args.fold)
    if args.serving:
        train_serving()
    if not (args.prepare or args.fold or args.serving):
        parser.error("select --prepare, --fold YEAR, or --serving")


if __name__ == "__main__":
    main()
