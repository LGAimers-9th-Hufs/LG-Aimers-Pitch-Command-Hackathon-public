"""
논문 재료 내보내기 (run_smoke와 분리 실행):
  1) 최신 run(results/runs.jsonl 마지막 줄) 로드, figures 없으면 재렌더
  2) paper/assets/ 로 figure 8파일 복사 + leaderboard.md(마크다운 표) 생성
  3) paper/outline.md 의 AUTO 마커(leaderboard / run_meta) 사이를 최신 내용으로 치환(멱등)

실행:  python sweep/make_paper_assets.py
"""
import os
import shutil
import sys
from pathlib import Path

try:  # Windows 콘솔 한글 깨짐 방지
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reporting import latest_run

PAPER_DIR = Path(__file__).resolve().parent.parent / "paper"


def _fmt(v, nd=4):
    return "-" if v is None else f"{v:.{nd}f}"


def leaderboard_md(run):
    lines = ["| # | candidate | tier | LogLoss | Brier | AUC | Cold | Δ anchor | gate |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(run["leaderboard"], 1):
        gate = ("**beats anchor**" if r.get("beats_anchor")
                else ("anchor" if r["name"] == run["harness"]["anchor"] else "-"))
        lines.append(
            f"| {i} | {r['name']} | {r['tier']} | {_fmt(r['logloss_cal'])} | "
            f"{_fmt(r['brier_cal'])} | {_fmt(r.get('auc'), 3)} | {_fmt(r.get('logloss_cold'))} | "
            f"{r['delta_vs_anchor']:+.4f} | {gate} |")
    return "\n".join(lines)


def run_meta_md(run):
    m = run["meta"]
    git = m.get("git") or {}
    av = run.get("diagnostics", {}).get("adversarial", {})
    return "\n".join([
        f"- run_id: `{m['run_id']}`  (data_source={m['data_source']}, seed={m['seed']})",
        f"- git: `{git.get('commit', 'n/a')}` (branch={git.get('branch', 'n/a')}, "
        f"dirty={git.get('dirty', 'n/a')})",
        f"- harness v{m['harness_version']}, timestamp {m['timestamp_utc']}",
        f"- adversarial validation AUC: full={_fmt(av.get('auc'), 3)}, "
        f"context-only={_fmt(av.get('context_auc'), 3)}",
    ])


def replace_marker(text, key, payload):
    begin, end = f"<!-- AUTO:{key}:begin -->", f"<!-- AUTO:{key}:end -->"
    i, j = text.find(begin), text.find(end)
    if i < 0 or j < 0:
        raise SystemExit(f"outline.md에 {begin}/{end} 마커가 없음")
    return text[: i + len(begin)] + "\n" + payload + "\n" + text[j:]


def main():
    run, run_dir = latest_run()
    if run is None:
        raise SystemExit("기록된 run이 없음 — 먼저 python sweep/run_smoke.py 실행")

    fig_dir = run_dir / "figures"
    if not fig_dir.exists() or not any(fig_dir.iterdir()):
        from figures import render_all
        render_all(run, run_dir)

    assets = PAPER_DIR / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(fig_dir.iterdir()):
        shutil.copy2(p, assets / p.name)
        n += 1
    (assets / "leaderboard.md").write_text(
        f"<!-- {run['meta']['run_id']} 리더보드 (자동 생성: make_paper_assets.py) -->\n"
        + leaderboard_md(run) + "\n", encoding="utf-8")

    outline = PAPER_DIR / "outline.md"
    text = outline.read_text(encoding="utf-8")
    text = replace_marker(text, "leaderboard", leaderboard_md(run))
    text = replace_marker(text, "run_meta", run_meta_md(run))
    outline.write_text(text, encoding="utf-8")

    print(f"[paper] {assets} 에 figure {n}파일 + leaderboard.md 갱신")
    print(f"[paper] outline.md AUTO 마커 치환 완료 (run={run['meta']['run_id']})")


if __name__ == "__main__":
    main()
