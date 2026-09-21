# -*- coding: utf-8 -*-
"""Build a conservative JTT + hierarchical entity-residual candidate package.

The source JTT directory is copied byte-for-byte, then one frozen JSON lookup and a row-local
post-processing function are added.  The original champion is never modified.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORKSPACE = ROOT.parent.parent
DEFAULT_BASE = WORKSPACE / "open" / "experiments_backup" / "external" / "FINAL_submit_JTT"
DEFAULT_ENTITY = ROOT / "results" / "dsf_entity" / "entity_final.json"
DEFAULT_OUTPUT = HERE / "dist" / "FINAL_submit_JTT_DSF_ENTITY.zip"

FUNCTION_MARKER = "# --- JTT_DSF_ENTITY:BEGIN ---"
CALL_OLD = """        if regime_p is not None:
            p = np.clip(0.7412437855967673 * p + 0.25875621440323276 * regime_p, 0.0, 1.0)"""
CALL_NEW = CALL_OLD + """
        p = np.clip(p + dsf_entity_correction(test, p), 0.0, 1.0)"""

SERVE_CODE = r'''
# --- JTT_DSF_ENTITY:BEGIN ---
def dsf_entity_correction(test, base_p):
    """Frozen hierarchical residual lookup; every output uses only the current row."""
    try:
        with open(os.path.join(MODEL_DIR, "dsf_entity.json"), "r", encoding="utf-8") as f:
            dsf = json.load(f)
        artifact = dsf["artifact"]
        n = len(test)
        raw = np.zeros(n, dtype=np.float64)
        pitcher = pd.to_numeric(test["pitcher_id"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
        batter = pd.to_numeric(test["batter_id"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
        balls = pd.to_numeric(test["balls_before"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        strikes = pd.to_numeric(test["strikes_before"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        count_group = np.select([strikes > balls, balls > strikes], [0, 2], default=1).astype(np.int64)
        ph = pd.to_numeric(test["pitcher_hand"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
        bh = pd.to_numeric(test["batter_hand"], errors="coerce").fillna(-2).to_numpy(dtype=np.int64)
        same_hand = (ph == bh).astype(np.int64)
        values = {"pitcher_id": pitcher, "batter_id": batter,
                  "count_group": count_group, "same_hand": same_hand}
        key_cache = {}
        for component in artifact["components"]:
            cols = tuple(component["cols"])
            keys = key_cache.get(cols)
            if keys is None:
                keys = []
                for i in range(n):
                    keys.append("|".join(str(int(values[c][i])) for c in cols))
                key_cache[cols] = keys
            mapping = component["mapping"]
            for i in range(n):
                raw[i] += float(mapping.get(keys[i], 0.0))
        cap = float(dsf["cap"])
        bounded = cap * np.tanh(raw / cap)
        projection = dsf["projection"]
        correction = bounded - (float(projection[0])
                                + float(projection[1]) * (np.asarray(base_p, dtype=float) - 0.5))
        correction = np.clip(correction, -cap, cap)
        return float(dsf["gamma"]) * correction
    except Exception:
        traceback.print_exc()
        print("!! DSF entity correction failed - JTT prediction retained")
        return np.zeros(len(test), dtype=np.float64)
# --- JTT_DSF_ENTITY:END ---
'''


def patch_script(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    if FUNCTION_MARKER in source:
        raise ValueError("base script already contains DSF entity patch")
    sentinel = "\n\ndef main():"
    pos = source.find(sentinel)
    if pos < 0:
        raise ValueError("could not locate main() in base script")
    source = source[:pos] + "\n" + SERVE_CODE + source[pos:]
    if source.count(CALL_OLD) != 1:
        raise ValueError(f"expected exactly one JTT blend call, found {source.count(CALL_OLD)}")
    source = source.replace(CALL_OLD, CALL_NEW)
    path.write_text(source, encoding="utf-8")


def build(base: Path, entity: Path, output: Path) -> Path:
    if not base.is_dir():
        raise FileNotFoundError(base)
    if not entity.is_file():
        raise FileNotFoundError(entity)
    payload = json.loads(entity.read_text(encoding="utf-8"))
    if not payload.get("validation", {}).get("incremental_go"):
        raise RuntimeError("entity artifact did not pass its incremental gate")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="jtt_dsf_entity_") as td:
        work = Path(td) / "package"
        shutil.copytree(base, work)
        shutil.copy2(entity, work / "model" / "dsf_entity.json")
        patch_script(work / "script.py")
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for path in sorted(work.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(work).as_posix())
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--entity", type=Path, default=DEFAULT_ENTITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    out = build(args.base, args.entity, args.output)
    print(f"saved {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
