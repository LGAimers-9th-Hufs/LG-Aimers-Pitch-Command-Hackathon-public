"""
run 기록 레이어 — 스윕 결과를 논문 근거 파일로 남긴다 (하네스 로직 불변).

산출물 (Project_Baseball_Pitching/results/):
  runs/<run_id>/run.json    단일 진실 소스(메타+진단+리더보드 전체). figure는 이것만으로 재생성.
  runs/<run_id>/leaderboard.csv   finalist 표 (엑셀/판다스 바로 열람용)
  runs.jsonl                run당 1줄 요약 (append-only)
  leaderboard_history.csv   전 run × 후보 롱포맷 (매번 재생성) = "arm X가 언제부터 앵커를
                            이겼나" 추적 → 논문 ablation 원자료

원칙: 기록 실패가 스윕을 막으면 안 됨 → git 조회 등은 실패 시 None으로 강등.
"""
from __future__ import annotations
import csv
import dataclasses
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HARNESS_VERSION = "0.1"  # 하네스 호환성 표식(스키마 바뀌면 올림)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"  # cwd 무관


class _NpEncoder(json.JSONEncoder):
    """np 스칼라/배열 → 파이썬 기본형. NaN/±inf → None (allow_nan 비의존)."""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            f = float(o)
            return f if np.isfinite(f) else None
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)

    def iterencode(self, o, _one_shot=False):
        return super().iterencode(_sanitize(o), _one_shot)


def _sanitize(o):
    """중첩 구조의 float NaN/inf를 None으로 (json.dump는 default를 float에 안 태움)."""
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def git_info():
    """{commit, branch, dirty} 또는 None (git 없음/실패해도 기록은 계속)."""
    try:
        root = Path(__file__).resolve().parent

        def _run(*args):
            return subprocess.run(["git", *args], cwd=root, capture_output=True,
                                  text=True, timeout=10, check=True).stdout.strip()

        return {
            "commit": _run("rev-parse", "HEAD"),
            "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_run("status", "--porcelain")),
        }
    except Exception:
        return None


def _lib_versions():
    out = {}
    for lib in ("numpy", "pandas", "sklearn", "scipy", "torch", "matplotlib"):
        try:
            out[lib] = __import__(lib).__version__
        except Exception:
            out[lib] = None
    return out


def collect_meta(spec, *, seed, data_source, tag=None):
    """run 메타데이터. run_id는 Windows 파일명 안전(':' 없음)."""
    now = datetime.now(timezone.utc)
    return {
        "run_id": now.strftime("%Y%m%d-%H%M%S") + f"_{data_source}",
        "timestamp_utc": now.isoformat(),
        "git": git_info(),
        "harness_version": HARNESS_VERSION,
        "seed": seed,
        "data_source": data_source,  # "synthetic" | "real"
        "tag": tag,
        "spec": dataclasses.asdict(spec),
        "lib_versions": _lib_versions(),
    }


_LB_COLS = ["name", "role", "tier", "refine_v24", "refine_min", "refine_r2019", "d_r2019",
            "score", "refine", "refine_R", "refine_F", "bss", "brier", "logloss", "auc",
            "pred_mean", "d", "score_cold", "score_m1", "se_vs_anchor", "n_folds",
            "delta_vs_anchor", "beats_anchor",
            # 합성 run(구버전) 키 — 하위호환용
            "brier_cal", "logloss_cal", "logloss_cold"]
# score = 대회 공식 산식 100000×(1−Brier/(r(1−r))). 실데이터 run의 1순위 정렬 키(클수록 좋음).
# ⚠ 합성 run은 logloss 기반(작을수록 좋음)이라 delta_vs_anchor의 부호 규약이 반대다.
#   두 소스를 한 표에서 비교하지 말 것 — 합성 run은 results/archive_synthetic/으로 격리한다.


def _write_leaderboard_csv(path, leaderboard):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_LB_COLS, extrasaction="ignore")
        w.writeheader()
        for r in leaderboard:
            w.writerow({k: r.get(k) for k in _LB_COLS})


def record_run(report, diagnostics, meta):
    """run.json + leaderboard.csv 저장, runs.jsonl append, history 재생성. 반환 run_dir."""
    run_dir = RESULTS_DIR / "runs" / meta["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)

    run = {"meta": meta, "diagnostics": diagnostics, **report}
    with open(run_dir / "run.json", "w", encoding="utf-8") as f:
        json.dump(run, f, cls=_NpEncoder, ensure_ascii=False, indent=1)

    _write_leaderboard_csv(run_dir / "leaderboard.csv", report["leaderboard"])

    # best = **채택 가능한** 최상위 항목. oracle/reference(상수·val 라벨 사용)는 제외한다 —
    # 그러지 않으면 runs.jsonl과 history에 상한선 arm이 run의 최고 성적으로 기록된다.
    _lb = report["leaderboard"]
    _cand = [r for r in _lb if r.get("role", "candidate") == "candidate"]
    best = (_cand or _lb or [{}])[0]
    summary = {
        "run_id": meta["run_id"],
        "timestamp_utc": meta["timestamp_utc"],
        "data_source": meta["data_source"],
        "commit": (meta.get("git") or {}).get("commit"),
        "anchor_logloss": report["harness"].get("anchor_logloss"),
        "anchor_score": report["harness"].get("anchor_score"),
        "best": best.get("name"),
        "best_logloss": best.get("logloss_cal", best.get("logloss")),
        "best_score": best.get("score"),
        "n_candidates": len(report.get("candidates", [])),
        "n_beats_anchor": sum(1 for r in report["leaderboard"] if r.get("beats_anchor")),
    }
    with open(RESULTS_DIR / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, cls=_NpEncoder, ensure_ascii=False) + "\n")

    _rebuild_history()
    return run_dir


def _rebuild_history():
    """runs/*/run.json 전체 → leaderboard_history.csv (run_id × candidate 롱포맷)."""
    rows = []
    for rj in sorted((RESULTS_DIR / "runs").glob("*/run.json")):
        try:
            run = json.loads(rj.read_text(encoding="utf-8"))
        except Exception:
            continue  # 손상 run은 history에서만 제외
        meta = run.get("meta", {})
        for r in run.get("leaderboard", []):
            rows.append({"run_id": meta.get("run_id"),
                         "timestamp_utc": meta.get("timestamp_utc"),
                         "data_source": meta.get("data_source"),
                         **{k: r.get(k) for k in _LB_COLS}})
    with open(RESULTS_DIR / "leaderboard_history.csv", "w", encoding="utf-8", newline="") as f:
        cols = ["run_id", "timestamp_utc", "data_source"] + _LB_COLS
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def latest_run():
    """runs.jsonl 마지막 줄의 run_id로 (run dict, run_dir) 로드. 없으면 (None, None)."""
    jl = RESULTS_DIR / "runs.jsonl"
    if not jl.exists():
        return None, None
    lines = [ln for ln in jl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return None, None
    run_id = json.loads(lines[-1])["run_id"]
    run_dir = RESULTS_DIR / "runs" / run_id
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    return run, run_dir
