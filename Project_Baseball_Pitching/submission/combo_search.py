# -*- coding: utf-8 -*-
"""전 조합 블렌드 기대점수 탐색 — 컷(1100) 국면의 도착일 도구 (docs/log/37 §5).

    python submission/combo_search.py
    python submission/combo_search.py --leg cat=1062.3679303525 --leg f24=1058.8749660623

새 zip이 오면: ① scan_probe_provenance (BLOCK이면 폐기) ② leg_matrix --leg name=zip 으로
캐시 생성(p23 규약) ③ 이 스크립트에 --leg name=LB점수 추가 → 균등가중 전 조합의
E = 평균LB + gain/ρ 를 정확 항등식으로 계산해 컷 통과 후보를 출력한다.

수학: gain_M = 4e5 · (1/M²) · Σ_{i<j} E_rows[(p_i−p_j)²] (쌍별 d² 행렬로 정확, 행 재접근 불필요).
ρ = 1.4295 (A3 앵커 42.423→29.677). 캐시는 results/leg_matrix/{name}_30000_*.npy (최신 파일).

기본 레그 = 제출 가능·채점 완료·규정 준수만. jtt_tm2/3/4(§5 위반 as-is)·BLOCK 계보·
calP(provenance 문장 부정확 — 팀 결정으로 physmix 사용)는 제외한다. era는 미채점이라
전이 추정(758±)으로만 넣고 [est] 표시 — est 포함 조합은 참고용.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / "leg_matrix"
RHO = 1.4295
CUT = 1100.0

# name: (LB, est?)  — 원장 results/lb_history_260829.md 정본
DEFAULT_LEGS = {
    "clookup":  (1049.4561885154, False),
    "cregime":  (1045.9643163789, False),
    "cmoe":     (1027.5280314742, False),
    "physmix":  (1009.2644920688, False),
    "bis_tm":   (1003.5142600363, False),
    "is_k100":  (984.0806860421, False),
    "tm3L":     (940.31, True),            # blendD4 역산 (±10)
    "ysy_resid": (939.6308843588, False),
    "ysy_trkm": (939.0700120243, False),
    "ysy_cbb":  (937.6278793313, False),
    "june853":  (853.5697653812, False),
    "ysy_mlp":  (812.27, True),            # blendE5 역산 (±20)
    "era":      (757.9, True),             # 전이 추정 0.7541·579.47+320.9
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", action="append", default=[], metavar="NAME=LB",
                    help="추가 레그 (캐시 results/leg_matrix/NAME_30000_*.npy 필요)")
    ap.add_argument("--maxlegs", type=int, default=6)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--rho", type=float, default=RHO)
    args = ap.parse_args()

    legs = dict(DEFAULT_LEGS)
    for spec in args.leg:
        nm, _, sc = spec.partition("=")
        legs[nm] = (float(sc), False)

    preds, scores, est = {}, {}, {}
    for nm, (sc, e) in legs.items():
        fs = sorted(glob.glob(str(CACHE / f"{nm}_30000_*.npy")))
        if not fs:
            print(f"[skip] 캐시 없음: {nm}")
            continue
        preds[nm] = np.load(fs[-1])
        scores[nm], est[nm] = sc, e
    names = list(preds)
    n = len(names)
    print(f">> 레그 {n}개 · ρ={args.rho} · 컷 {CUT}")

    D2 = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D2[i, j] = D2[j, i] = float(np.mean((preds[names[i]] - preds[names[j]]) ** 2))

    rows = []
    idx = range(n)
    for M in range(2, args.maxlegs + 1):
        for combo in itertools.combinations(idx, M):
            s = sum(scores[names[i]] for i in combo) / M
            g = 4e5 * sum(D2[i, j] for i, j in itertools.combinations(combo, 2)) / M ** 2
            rows.append((s + g / args.rho, s, g, combo))
    rows.sort(reverse=True)

    print(f"\n{'E':>9} {'평균LB':>9} {'gain':>7}  구성")
    shown = 0
    for E, s, g, combo in rows:
        if shown >= args.top:
            break
        tag = "🎯" if E >= CUT else ("△" if E >= CUT - 5 else "  ")
        nm_s = "+".join(names[i] + ("[est]" if est[names[i]] else "") for i in combo)
        print(f"{tag}{E:9.2f} {s:9.2f} {g:7.2f}  {nm_s}")
        shown += 1
    n_cut = sum(1 for E, *_ in rows if E >= CUT)
    print(f"\n컷(≥{CUT:.0f}) 통과 조합: {n_cut}개 / 전체 {len(rows):,}개")


if __name__ == "__main__":
    main()
