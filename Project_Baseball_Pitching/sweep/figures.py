"""
논문 figure 렌더 — run.json(디스크 기록)만을 입력으로 4종 × (PNG 300dpi + PDF 벡터) = 8파일.

입력을 run.json으로 한정하는 이유: "기록만으로 figure 완전 재생성 가능"을 매 실행 보증.
라벨은 전부 영문(논문 직행). 팔레트는 Okabe-Ito 기반 CVD-safe 6색(검증 통과).
LogLoss 차이는 소수 셋째 자리라 0-기준 bar는 정보가 없음 → 리더보드/캘리 A/B는
줌 가능한 dot(+오차막대) 형태를 쓴다(잘린 bar는 논문 심사 단골 지적).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
ACCENT = "#0072B2"       # beats_anchor 강조
NEUTRAL = "#B8B8B8"      # 미달 후보
ANCHOR_BLACK = "#000000"  # 앵커 라인
ANCHOR_REF = "#D55E00"    # 앵커 기준선(수직선)

RC = {
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "figure.figsize": (3.5, 2.6),
    "savefig.bbox": "tight", "savefig.dpi": 300,
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "lines.linewidth": 1.6,
}


def _save(fig, out_dir, name):
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)


def _fold_vals(entry, key="logloss_cal"):
    return [f[key] for f in entry["per_fold"]]


def fig_leaderboard(run, out_dir):
    """finalist CV LogLoss (mean ± per-fold std), 앵커 수직 기준선, beats_anchor 강조."""
    lb = run["leaderboard"]
    anchor = run["harness"]["anchor"]
    names = [r["name"] for r in lb][::-1]          # 위=1위가 되도록 역순
    means = [r["logloss_cal"] for r in lb][::-1]
    stds = [float(np.std(_fold_vals(r))) for r in lb][::-1]
    colors = [ACCENT if r.get("beats_anchor") else
              (ANCHOR_BLACK if r["name"] == anchor else NEUTRAL) for r in lb][::-1]

    fig, ax = plt.subplots(figsize=(3.5, 0.32 * len(lb) + 0.9))
    ypos = np.arange(len(lb))
    ax.axvline(run["harness"]["anchor_logloss"], color=ANCHOR_REF, ls="--", lw=1.0,
               label="anchor (GBDT)", zorder=1)
    for yy, m, s, c in zip(ypos, means, stds, colors):
        ax.errorbar(m, yy, xerr=s, fmt="o", ms=5, color=c,
                    ecolor=c, elinewidth=1.0, capsize=2, zorder=3)
    ax.set_yticks(ypos, names)
    ax.set_xlabel("CV LogLoss (calibrated, mean ± fold s.d.)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right", frameon=False)
    _save(fig, out_dir, "leaderboard_logloss")


def fig_perfold(run, out_dir, max_colored=6):
    """폴드(=val 시즌)별 LogLoss. 앵커=굵은 검정 파선, 상위 6개 색+마커, 나머지 회색 'others'."""
    lb = run["leaderboard"]
    anchor = run["harness"]["anchor"]
    others = [r for r in lb if r["name"] != anchor]
    colored, rest = others[:max_colored], others[max_colored:]
    anchor_row = next(r for r in lb if r["name"] == anchor)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    seasons = [f["val_season"] for f in anchor_row["per_fold"]]
    for r in rest:  # de-emphasized 컨텍스트 라인(단일 범례 항목)
        ax.plot([f["val_season"] for f in r["per_fold"]], _fold_vals(r),
                color=NEUTRAL, lw=0.9, zorder=1)
    if rest:
        ax.plot([], [], color=NEUTRAL, lw=0.9, label="others")
    for i, r in enumerate(colored):
        ax.plot([f["val_season"] for f in r["per_fold"]], _fold_vals(r),
                color=PALETTE[i % len(PALETTE)], marker=MARKERS[i % len(MARKERS)],
                ms=4, lw=1.4, label=r["name"], zorder=2)
    ax.plot(seasons, _fold_vals(anchor_row), color=ANCHOR_BLACK, ls="--", lw=2.2,
            label="anchor (GBDT)", zorder=3)
    ax.set_xticks(seasons)
    ax.set_xlabel("Validation season")
    ax.set_ylabel("LogLoss (calibrated)")
    ax.legend(frameon=False, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.28))  # 데이터 가림 방지: 플롯 아래 배치
    _save(fig, out_dir, "perfold_logloss")


def _agg_reliability(entry):
    """전 폴드 reliability bins를 count 가중으로 합산 → (mean_pred, frac_pos, count, edges, ece)."""
    acc = {}
    edges = None
    for f in entry["per_fold"]:
        rel = f.get("reliability")
        if not rel:
            continue
        edges = rel["edges"]
        for b, mp, fp, c in zip(rel["bin"], rel["mean_pred"], rel["frac_pos"], rel["count"]):
            s = acc.setdefault(b, [0.0, 0.0, 0])
            s[0] += mp * c
            s[1] += fp * c
            s[2] += c
    bins = sorted(acc)
    mp = np.array([acc[b][0] / acc[b][2] for b in bins])
    fp = np.array([acc[b][1] / acc[b][2] for b in bins])
    n = np.array([acc[b][2] for b in bins])
    ece = float(np.sum(n / n.sum() * np.abs(fp - mp)))
    return mp, fp, n, edges, ece


def fig_reliability(run, out_dir):
    """앵커 + 리더보드 1위(앵커면 2위)의 calibration diagram + 예측분포 히스토그램."""
    lb = run["leaderboard"]
    anchor = run["harness"]["anchor"]
    anchor_row = next(r for r in lb if r["name"] == anchor)
    best = next((r for r in lb if r["name"] != anchor), anchor_row)

    fig, (ax, axh) = plt.subplots(2, 1, figsize=(3.5, 3.4), sharex=True,
                                  height_ratios=[3, 1],
                                  gridspec_kw={"hspace": 0.12})
    ax.plot([0, 1], [0, 1], color="#999999", ls=":", lw=1.0, zorder=1)  # 완전 캘리 기준
    hist = None
    for row, color, marker in ((anchor_row, ANCHOR_BLACK, "s"), (best, ACCENT, "o")):
        mp, fp, n, edges, ece = _agg_reliability(row)
        ax.plot(mp, fp, color=color, marker=marker, ms=4, lw=1.4,
                label=f"{row['name']} (ECE={ece:.3f})", zorder=2)
        if row is best:
            hist = (mp, n)
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left")

    mp, n = hist
    width = 0.9 / len(run["leaderboard"][0]["per_fold"][0]["reliability"]["edges"])
    axh.bar(mp, n, width=width, color=ACCENT, alpha=0.85)
    axh.set_xlabel("Predicted probability (calibrated)")
    axh.set_ylabel("Count")
    axh.grid(axis="x", visible=False)
    _save(fig, out_dir, "reliability")


def fig_calib_ab(run, out_dir):
    """C5 캘리 3모드 A/B: LogLoss·Brier 2패널 dot (차이가 미세해 0-기준 bar는 부적합)."""
    ab = run["calib_ab"]
    modes = list(ab)
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.0))
    for ax, key, title in zip(axes, ("logloss_cal", "brier_cal"), ("LogLoss", "Brier")):
        vals = [ab[m][key] for m in modes]
        ypos = np.arange(len(modes))
        for yy, v, c in zip(ypos, vals, PALETTE):
            ax.plot(v, yy, "o", ms=6, color=c)
            ax.annotate(f"{v:.4f}", (v, yy), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=6.5)
        ax.set_yticks(ypos, modes if ax is axes[0] else [""] * len(modes))
        ax.set_title(title, fontsize=8.5)
        ax.grid(axis="y", visible=False)
        lo, hi = min(vals), max(vals)
        pad = max((hi - lo) * 0.5, 1e-4)
        ax.set_xlim(lo - pad, hi + pad)
        ax.tick_params(axis="x", labelsize=6.5)
    _save(fig, out_dir, "calib_ab")


def render_all(run, run_dir):
    """run dict(=run.json 로드본) → run_dir/figures/ 에 4종 × png+pdf."""
    out_dir = Path(run_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RC):
        fig_leaderboard(run, out_dir)
        fig_perfold(run, out_dir)
        fig_reliability(run, out_dir)
        fig_calib_ab(run, out_dir)
    print(f"[figures] {out_dir} 에 8파일(png+pdf×4) 저장")
    return out_dir
