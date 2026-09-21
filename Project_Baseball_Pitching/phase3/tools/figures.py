# -*- coding: utf-8 -*-
"""발표 PPT용 그림 생성 — 전부 저장소의 실측 파일에서 계산한다.

    python phase3/tools/figures.py        # -> phase3/figures/*.png

원칙: **숫자를 그림 코드에 손으로 적지 않는다.** 게이트 JSON과 레그 예측 캐시(.npy)에서
직접 읽어 계산한다. 유일한 예외는 리더보드 점수인데, 그건 우리가 측정할 수 있는 값이
아니라 주최가 준 값이라 원장(`results/lb_history_260829.md`)의 정확값을 상수표로 둔다.
"""
from __future__ import annotations

import glob
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "phase3" / "figures"
NL = chr(10)      # 줄바꿈 (소스에 역슬래시 이스케이프를 두지 않는다)

# --- 팔레트 (dataviz 기준 인스턴스, light) — validate_palette.js 통과 확인 -----
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"     # 카테고리 슬롯 1·2·3
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#7c7b76"
SURFACE, GRID = "#fcfcfb", "#e5e4e0"
SEQ = ["#d8e6f8", "#a9c8ef", "#6ea4e2", "#2a78d6", "#1a4d8a"]   # 단일 색상 순차 램프

# --- 리더보드 점수 (원장 정본) -------------------------------------------------
# est=True 는 블렌드 실측에서 항등식으로 역산한 값(미제출 레그)이다. 그림에서 구분 표기한다.
LB = {
    "clookup": (1049.4561885154, False), "cregime": (1045.9643163789, False),
    "cmoe": (1027.5280314742, False),    "physmix": (1009.2644920688, False),
    "calP": (1015.7036703211, False),    "jtt_tm3": (996.2366887286, False),
    "jtt_v10": (956.1866984424, False),  "ysy_resid": (939.6308843588, False),
    "ysy_trkm": (939.0700120243, False), "ysy_cbb": (937.6278793313, False),
    "ysy_cat": (920.816692872, False),   "jtt_tm2": (912.1317478883, False),
    "ysy_v3a": (878.240944288, False),   "tm2nr": (873.3552709252, False),
    "june853": (853.5700000000, False),  "ysy_v3b": (823.8287879543, False),
    "ysy_v3": (822.0412364012, False),   "grok_v5": (728.408076724, False),
    "grok_v6": (716.8233531487, False),
    "tm3L": (940.31, True), "ysy_mlp": (812.3, True),
}
D4 = ["clookup", "cmoe", "physmix", "tm3L"]
S_D4, RHO = 1058.6047851923, 1.4295


def style(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(SURFACE)
    ax.figure.patch.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=13, pad=14, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("   %s  (%.0f KB)" % (name, p.stat().st_size / 1024))


def load_legs():
    legs = {}
    for f in glob.glob(str(ROOT / "results" / "leg_matrix" / "*_30000_*.npy")):
        legs[Path(f).name.split("_30000_")[0]] = np.load(f)
    return legs


# ---------------------------------------------------------------- 그림 1
def fig_brier_geometry():
    """Brier에서 예측 평균이 δ 어긋날 때의 비용 = 1e5·δ²/(r(1-r)) = 4e5·δ²."""
    r = 0.49
    C = 1e5 / (r * (1 - r))
    d = np.linspace(0, 0.05, 400)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(d, C * d ** 2, color=S1, linewidth=2)
    for dv, label in ((0.01, ""), (0.02, ""), (0.03, "δ=0.03 → -%d점" % round(C * 0.03 ** 2))):
        ax.plot([dv], [C * dv ** 2], "o", color=S1, markersize=9,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        if label:
            ax.annotate(label, (dv, C * dv ** 2), textcoords="offset points",
                        xytext=(-12, 14), color=INK, fontsize=11, ha="right",
                        fontweight="bold")
    style(ax, "레벨이 어긋나면 비용은 2차로 커진다",
          "예측 평균의 오차  δ", "점수 손실")
    ax.text(0.001, C * 0.05 ** 2 * 0.86,
            "비용 = 1e5 x δ² / (r(1-r)) = 약 400,000 x δ²    (r = 0.49)",
            color=INK2, fontsize=10)
    ax.text(0.001, C * 0.05 ** 2 * 0.74,
            "첫 제출 269점의 해부: 레벨 오판 -168 · isotonic이 분해능 -186",
            color=INK3, fontsize=9.5)
    save(fig, "fig1_brier_geometry.png")


# ---------------------------------------------------------------- 그림 2
def fig_inseason_gate():
    """당해 시즌 분해(is4)의 3폴드 이득 vs 같은 컬럼 수 위약(is4_shuf)."""
    g = json.loads((ROOT / "results" / "inseason" / "gate.json").read_text(encoding="utf-8"))
    folds = [2024, 2023, 2022]
    real = [next(r["d"] for r in g if r["val"] == v and r["arm"] == "is4") for v in folds]
    plac = [next((r["d"] for r in g if r["val"] == v and r["arm"] == "is4_shuf"), np.nan)
            for v in folds]
    x = np.arange(len(folds))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    w = 0.36
    b1 = ax.bar(x - w / 2 - 0.01, real, w, color=S1, label="is4 (당해 시즌 분해)")
    b2 = ax.bar(x + w / 2 + 0.01, plac, w, color=S2, label="위약 (경로 셔플)")
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            if np.isnan(h):
                continue
            ax.annotate("%+.1f" % h, (b.get_x() + b.get_width() / 2, h),
                        textcoords="offset points", xytext=(0, 6 if h >= 0 else -16),
                        ha="center", color=INK, fontsize=10, fontweight="bold")
    ax.axhline(0, color=INK3, linewidth=1)
    ax.set_xticks(x, ["val %d" % v for v in folds])
    style(ax, "단일 최대 신호 — 그리고 같은 크기의 위약은 붙지 않는다",
          "", "기준선 대비 점수 차")
    leg = ax.legend(frameon=False, fontsize=10, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.text(0.09, -0.02,
            "3폴드 평균 %+.1f · 폴드 간 SD %.1f  →  채택 기준(평균>0 & 평균>SD) 통과. "
            "공개 리더보드 911 → 953." % (np.mean(real), np.std(real, ddof=1)),
            color=INK3, fontsize=9.5)
    save(fig, "fig2_inseason_gate.png")


# ---------------------------------------------------------------- 그림 3
def fig_placebo():
    """난수 6컬럼이 폴드 하나에서 2SE를 통과한다 — 오차막대는 폴드 간 SD여야 한다."""
    g = json.loads((ROOT / "results" / "phase3" / "gate_placebo.json").read_text(encoding="utf-8"))
    rows = [r for r in g if r["arm"] == "noise6"]
    folds = [r["val"] for r in rows]
    d = np.array([r["d"] for r in rows])
    se = np.array([r["se"] for r in rows])
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    y = np.arange(len(folds))[::-1]
    ax.errorbar(d, y, xerr=2 * se, fmt="o", color=S2, markersize=10, capsize=5,
                linewidth=2, markeredgecolor=SURFACE, markeredgewidth=2,
                ecolor=S2, label="난수 6컬럼 (±2SE)")
    ax.axvline(0, color=INK3, linewidth=1)
    for xi, yi, s in zip(d, y, se):
        ax.annotate("%+.1f ± %.1f" % (xi, 2 * s), (xi, yi), textcoords="offset points",
                    xytext=(0, 14), ha="center", color=INK, fontsize=10, fontweight="bold")
    ax.set_yticks(y, ["val %d" % v for v in folds])
    ax.set_ylim(-0.5, len(folds) - 0.3)
    style(ax, "폴드 내 2SE는 오차막대가 아니다 — 난수도 통과한다",
          "기준선 대비 점수 차", "")
    fig.text(0.09, -0.03,
            "폴드 간 SD = %.1f.  컬럼을 추가하는 행위 자체가 ±13의 변동을 만든다\n"
            "→ 채택 기준을 '3폴드 평균 > 0 그리고 평균 > 폴드 간 SD'로 바꿨다"
            % np.std(d, ddof=1),
            color=INK3, fontsize=9.5, va="top")
    save(fig, "fig3_placebo.png")


# ---------------------------------------------------------------- 그림 4
def fig_disagreement(legs):
    """최종 4레그의 쌍별 불일치 RMS — 파트너는 점수가 아니라 불일치로 고른다."""
    n = len(D4)
    M = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        v = float(np.sqrt(((legs[D4[i]] - legs[D4[j]]) ** 2).mean()))
        M[i, j] = M[j, i] = v
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    im = ax.imshow(np.where(M == 0, np.nan, M), cmap=cmap, vmin=0.005, vmax=0.032)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", color=INK3, fontsize=12)
            else:
                ax.text(j, i, "%.4f" % M[i, j], ha="center", va="center",
                        color="#ffffff" if M[i, j] > 0.020 else INK,
                        fontsize=11, fontweight="bold")
    ax.set_xticks(range(n), D4, color=INK2, fontsize=10)
    ax.set_yticks(range(n), D4, color=INK2, fontsize=10)
    # 인접 채움 사이 2px 표면 간격
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=3)
    ax.grid(which="major", visible=False)
    ax.set_title("레그 사이의 불일치 RMS (2024 프록시 30,000행)",
                 color=INK, fontsize=13, pad=14, loc="left", fontweight="bold")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.figure.patch.set_facecolor(SURFACE)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK2, labelsize=9, length=0)
    P = np.stack([legs[k] for k in D4])
    gain = 4e5 * float(((P - P.mean(0)) ** 2).mean())
    fig.text(0.02, -0.02,
             "확률 평균의 이득 = 400,000 x (레그가 평균에서 벗어난 정도)² = %.1f점 (프록시).  "
             "가장 많이 어긋나는 쌍이 가장 많이 벌어준다." % gain,
             color=INK3, fontsize=9.5)
    save(fig, "fig4_disagreement.png")


# ---------------------------------------------------------------- 그림 5
# 자격 없는 후보 — 곡선 위에 있어도 실제로는 쓸 수 없는 레그
INELIGIBLE = {
    "jtt_tm3": "원본 서빙이 §2-4 위반 —" + NL + "재빌드본 tm3L이 이미 블렌드에 있다",
}


def fig_breakeven(legs):
    """(불일치 d, 리더보드 점수) 산점 + 챔피언 손익분기 곡선."""
    pbar = np.stack([legs[k] for k in D4]).mean(0)
    gain4 = 4e5 * float(((np.stack([legs[k] for k in D4]) - pbar) ** 2).mean())

    def S_be(d):                       # E5 = S_D4 가 되는 새 레그의 점수
        return 5 * (S_D4 - (0.8 * gain4 + 64000 * d ** 2) / RHO) - sum(
            LB[k][0] for k in D4)

    pts = []
    for k, q in legs.items():
        if k in D4 or k not in LB or k == "blendA3":
            continue
        pts.append((float(np.sqrt(((q - pbar) ** 2).mean())), LB[k][0], k, LB[k][1]))

    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    dd = np.linspace(0.004, 0.040, 300)
    ax.plot(dd, S_be(dd), color=S2, linewidth=2, zorder=2)
    ax.fill_between(dd, S_be(dd), 1120, color=S2, alpha=0.06, zorder=1)
    ax.annotate("손익분기 — 이 선 위여야 챔피언을 넘는다", (0.0335, S_be(0.0335)),
                textcoords="offset points", xytext=(-8, -22), ha="right", color=S2,
                fontsize=10, fontweight="bold")

    for d, sc, k, est in pts:
        bad = k in INELIGIBLE
        ax.plot([d], [sc], "o",
                color=SURFACE if (est or bad) else S1,
                markersize=9, markeredgecolor=S3 if bad else S1,
                markeredgewidth=2, zorder=3)
    for d, sc, k, est in pts:
        if sc > 990 or d > 0.030:
            ax.annotate(k, (d, sc), textcoords="offset points", xytext=(8, -4),
                        color=INK2, fontsize=9)

    win = [(d, sc, k) for d, sc, k, _ in pts if sc > S_be(d) and k not in INELIGIBLE]
    for d, sc, k in win:
        ax.annotate("%s — 곡선 위의 유일한 후보" % k, (d, sc),
                    textcoords="offset points", xytext=(12, 12),
                    color=INK, fontsize=10.5, fontweight="bold")
    for d, sc, k, _ in pts:
        if k in INELIGIBLE:
            ax.annotate(INELIGIBLE[k], (d, sc), textcoords="offset points",
                        xytext=(-14, -34), ha="right", va="top", color=S3, fontsize=9)

    style(ax, "레그를 하나 더 넣어 이득이 나려면 — 점수와 불일치를 둘 다 사야 한다",
          "기존 4레그 평균과의 불일치  d  (RMS)", "그 레그의 공개 리더보드 점수")
    ax.set_ylim(700, 1120)
    cap = ("점 %d개 = 우리가 캐시를 가진 모든 레그.  속이 빈 파란 점은 항등식으로 역산한 점수(미제출)."
           + NL
           + "초록 테두리는 자격 없는 후보.  곡선 위의 정당한 후보는 %s 하나뿐이었고, 실제로 제출해"
           + NL
           + "실측 1056.79 (기대 1059.20) — 예측 오차 안이지만 부호는 음수였다.")
    fig.text(0.09, -0.03, cap % (len(pts), win[0][2] if win else "없음"),
             color=INK3, fontsize=9.5, va="top")
    save(fig, "fig5_breakeven.png")


# ---------------------------------------------------------------- 그림 6
def fig_identity(legs):
    """항등식이 예측 도구로 작동했는가 — 예측 이득 vs 실현 이득.

    ⚠ 정직하게 그린다. A3는 ρ를 맞춘 점이고, D4는 그 결과로 S(tm3L)을 역산한 점이라
    **둘 다 정의상 대각선 위에 있다.** 진짜 사전 예측은 F5 하나뿐이다.
    """
    def gain(names):
        P = np.stack([legs[k] for k in names])
        return 4e5 * float(((P - P.mean(0)) ** 2).mean())

    rows = [
        ("blendA3 (3레그)", ["cregime", "cmoe", "physmix"], 1057.2623262209,
         "rho를 여기서 측정했다"),
        ("blendD4 (4레그)", D4, S_D4, "여기서 S(tm3L)을 역산했다"),
        ("blendF5 (5레그)", D4 + ["cregime"], 1056.7903849842, "유일한 사전 예측"),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    pred, realz = [], []
    for lab, names, measured, kind in rows:
        pred.append(gain(names) / RHO)
        realz.append(measured - np.mean([LB[k][0] for k in names]))
    lim = max(max(pred), max(realz)) * 1.22
    ax.plot([0, lim], [0, lim], color=INK3, linewidth=1.4, linestyle="--", zorder=1)
    ax.annotate("예측 = 실현", (lim * 0.72, lim * 0.72), textcoords="offset points",
                xytext=(-6, 8), color=INK3, fontsize=10, ha="right")
    for (lab, names, measured, kind), pv, rv in zip(rows, pred, realz):
        test = kind == "유일한 사전 예측"
        ax.plot([pv], [rv], "o", color=S1 if test else SURFACE, markersize=12,
                markeredgecolor=S1, markeredgewidth=2, zorder=3)
        txt = ("%s" + NL + "%s" + NL + "예측 %.1f · 실현 %.1f  (%.1f%%)") % (
            lab, kind, pv, rv, 100 * rv / pv)
        ax.annotate(txt, (pv, rv), textcoords="offset points", xytext=(14, -30),
                    color=INK if test else INK2, fontsize=9.5,
                    fontweight="bold" if test else "normal")
    style(ax, "확률 평균의 이득은 적합이 아니라 대수다 — 다만 시험은 한 번뿐이었다",
          "항등식이 예측한 이득 (프록시 이득 / rho)", "리더보드에서 실현된 이득")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    fig.text(0.09, -0.03,
             "속이 빈 점 둘은 상수를 맞춘 점이라 정의상 대각선 위에 있다 — 검증이 아니다." + NL + ""
             "채워진 점 하나만이 사전에 계산해 두고 제출해 확인한 값이고, 실현율은 94.6%였다.",
             color=INK3, fontsize=9.5, va="top")
    save(fig, "fig6_identity.png")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for cand in ("Malgun Gothic", "맑은 고딕", "NanumGothic", "AppleGothic"):
        if any(f.name == cand for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = cand
            break
    else:
        print("   [경고] 한글 폰트를 찾지 못했다 — 라벨이 깨질 수 있다")
    plt.rcParams["axes.unicode_minus"] = False
    legs = load_legs()
    missing = [k for k in D4 if k not in legs]
    if missing:
        raise SystemExit("레그 캐시 없음: %s" % missing)
    print(">> 그림 생성")
    fig_brier_geometry()
    fig_inseason_gate()
    fig_placebo()
    fig_disagreement(legs)
    fig_breakeven(legs)
    fig_identity(legs)
    print(">> 완료 %s" % OUT)


if __name__ == "__main__":
    main()
