# -*- coding: utf-8 -*-
"""LB 실측 점수 + 예측 불일치로 블렌드를 푸는 도구 — 손계산 금지용.

    python sweep/lb_solver.py --predict "0.4:new3,0.6:baseline"     # 기대점수·하한
    python sweep/lb_solver.py --optimize new3,baseline              # 최적 가중
    python sweep/lb_solver.py --measure 583.4 --of X --with "0.6:baseline"   # 미지 멤버 역산

세그먼트 절편 축(중심화 지시자)의 프로브·역산은 `sweep/seg_probe.py`에 있다 — 곡률 해석값
`b = C·w(1−w)` 검산이 붙어 있어 레그 비동일(멤버 무음 스킵)을 잡는다. 대수를 두 곳에 두지 않는다.

근거가 되는 항등식 (정확한 대수, 근사 아님):

    p̄ = Σ wᵢ pᵢ  (Σwᵢ = 1)  ⇒  Brier(p̄) = Σ wᵢ·Brierᵢ − ½ Σᵢⱼ wᵢwⱼ·mean((pᵢ−pⱼ)²)

    Score = 1e5·(1 − Brier/V),  V = r(1−r)
    ⇒  Score(p̄) = Σ wᵢ·Scoreᵢ + (1e5/2V)·Σᵢⱼ wᵢwⱼ·Dᵢⱼ,   Dᵢⱼ = mean((pᵢ−pⱼ)²)

두 번째 항이 **다양성 이득**이고 항상 ≥ 0이다 → 확률 가중평균은 멤버 점수의 가중평균 아래로
내려갈 수 없다. **실측이 그 하한을 깨면 산식이 아니라 "레그가 참조 제출물과 같다"는 전제가 틀린 것**이다
(BLEND-1 483.97이 정확히 그 경우였고, 원인은 레그에 섞인 z_asof였다 — D-19).

⚠ 점수는 반드시 **제출기록에서 직접 확인한 값**만 쓴다. 기억·문서 인용으로 채우면 안 된다.
불일치 D는 서버 핀 환경에서 프록시 입력(2024 전량)으로 잰 값을 쓴다 — 2025 실제값의 대리이며,
홀드아웃 폴드에서 0.0223~0.0274로 관측돼 프록시(0.0236)와 같은 범위다.
"""
from __future__ import annotations
import argparse
import itertools
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

R_2025 = 0.499                      # β쌍 연립 추정(문서 14 §6). V=r(1−r)는 결과에 거의 영향 없다.

# 제출기록 실측 LB 점수 (memory/lb-verified-scores.md와 동기)
SCORES = {
    "baseline":   549.5119345223,   # 주최 RF (raw47) — 3회 제출 동일
    "new3":       500.6103395903,   # glm_offset_hgb 82피처 β=−0.02
    "new1":       457.2942506222,   # 〃 β=+0.04
    "blend1":     483.9680177379,   # 0.5×(83피처 GLM) + 0.5×baseline
    "s1_iso":     269.0486102216,
}

# 멤버 쌍의 예측 불일치 mean(Δ²) — 서버 핀 venv, 2024 전량(245k~253k행) 실측
DISAGREE = {
    ("new3", "baseline"): 0.02358 ** 2,
}


def _d2(a, b):
    if a == b:
        return 0.0
    for k in ((a, b), (b, a)):
        if k in DISAGREE:
            return DISAGREE[k]
    raise KeyError(f"불일치 미측정: {a} vs {b} — 서버 핀 환경에서 프록시 입력으로 먼저 재라")


def blend_score(members, weights, scores=None, r=R_2025):
    """Σ wᵢSᵢ + (1e5/2V)·Σ wᵢwⱼDᵢⱼ. 반환 = (기대점수, 하한=가중평균, 이득)."""
    sc = scores or SCORES
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    V = r * (1 - r)
    base = float(sum(wi * sc[m] for wi, m in zip(w, members)))
    gain = 0.0
    for i, j in itertools.product(range(len(members)), repeat=2):
        if i != j:
            gain += w[i] * w[j] * _d2(members[i], members[j])
    gain *= 1e5 / (2 * V)
    return base + gain, base, gain


def optimize(members, step=0.005, r=R_2025):
    """단체(simplex) 격자 탐색 — 멤버 수가 적어 닫힌 해를 쓸 필요가 없다."""
    n = len(members)
    best = None
    grid = np.arange(0.0, 1.0 + 1e-9, step)
    for combo in itertools.product(grid, repeat=n - 1):
        s = sum(combo)
        if s > 1.0 + 1e-9:
            continue
        w = list(combo) + [1.0 - s]
        tot, base, gain = blend_score(members, w, r=r)
        if best is None or tot > best[0]:
            best = (tot, w, base, gain)
    return best


def measure(s_blend, unknown, known, r=R_2025):
    """블렌드 실측 점수에서 미지 멤버의 2025 점수를 역산.

    known = [(멤버, 가중), ...] (미지 멤버 가중 = 1 − Σ). 미지 멤버와의 불일치는 DISAGREE 필요.
    """
    V = r * (1 - r)
    w_known = np.array([w for _, w in known], dtype=float)
    w_u = 1.0 - w_known.sum()
    if w_u <= 0:
        sys.exit("미지 멤버 가중이 0 이하다")
    members = [m for m, _ in known] + [unknown]
    w = list(w_known) + [w_u]
    gain = 0.0
    for i, j in itertools.product(range(len(members)), repeat=2):
        if i != j:
            gain += w[i] * w[j] * _d2(members[i], members[j])
    gain *= 1e5 / (2 * V)
    known_part = float(sum(wi * SCORES[m] for (m, _), wi in zip(known, w_known)))
    return (s_blend - gain - known_part) / w_u, gain


def _parse(spec):
    ms, ws = [], []
    for tok in spec.split(","):
        w, _, m = tok.strip().rpartition(":")
        ms.append(m)
        ws.append(float(w or 1.0))
    return ms, ws


def shift_probe(s0, s_plus, s_minus, delta=0.005):
    """절편 대칭 프로브 역산 (D-57, Codex 처방).

    같은 예측에 shift만 ±δ 바꾼 두 제출의 점수로 2025 최적 절편을 정확히 식별한다:
        S(x) = S* − C·(x − δ*)²  (Brier가 shift의 2차식이므로 정확)
        C  = −(S₊ + S₋ − 2S₀) / (2δ²)      [검산: ≈ 1e5/V ≈ 4e5]
        δ* = (S₊ − S₋) / (4Cδ)              [현 shift 대비 이동량]
        G* = C·δ*²                           [최적 절편의 기대 이득]
    ⚠ 클리핑 행 비율이 크면 2차식이 휘어진다 — C 검산이 그 가드다.
    """
    C = -(s_plus + s_minus - 2.0 * s0) / (2.0 * delta ** 2)
    print(f"곡률 C = {C:,.0f}   [검산: 4e5 근방이어야 함 — 벗어나면 레그 비동일/클리핑/오류]")
    if s_plus + s_minus > 2.0 * s0:
        print("🚫 S₊+S₋ > 2S₀ — 볼록성 위반. 레그 동일성/구현을 의심하고 계산 중단.")
        return None
    dstar = (s_plus - s_minus) / (4.0 * C * delta)
    gstar = C * dstar ** 2
    print(f"δ* = {dstar:+.5f}  →  최적 shift = {-0.01 + dstar:+.5f} (현 −0.01)")
    print(f"G* = {gstar:+.2f}  (ENS-9 대비 최적 절편의 기대 이득)")
    # ⚠ shift는 서빙식에서 slope **뒤에** 더해진다(`0.5 + 1.04(raw−0.5) + shift`) → 1.04가 안 붙는다.
    #   최종 확률 척도의 평균오차 = −δ*, raw 척도로 환산하면 −δ*/1.04. (구 코드의 −1.04·δ*는 오류)
    print(f"암시된 2025 예측평균 오차 ≈ {-dstar:+.5f} (최종 확률 척도) "
          f"/ {-dstar / 1.04:+.5f} (raw 척도)")
    if gstar >= 10:
        print("→ **G* ≥ +10: shift* 재빌드·제출 권고**")
    else:
        print("→ G* < +10: ENS-9 캘리가 이미 근사 최적 — 레벨 채널 종결 확인")
    return dict(C=C, dstar=dstar, shift_star=-0.01 + dstar, gstar=gstar)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", help="'w:멤버,w:멤버' 기대점수·하한")
    ap.add_argument("--optimize", help="'멤버,멤버[,멤버]' 최적 가중")
    ap.add_argument("--measure", type=float, help="블렌드 실측 점수")
    ap.add_argument("--of", help="--measure와 함께: 역산할 미지 멤버 이름")
    ap.add_argument("--with", dest="known", help="--measure와 함께: 'w:멤버,...' 기지 멤버")
    ap.add_argument("--shift-probe", nargs=3, type=float, metavar=("S0", "S_PLUS", "S_MINUS"),
                    help="절편 프로브 역산: ENS-9 점수, shift −0.005 점수, shift −0.015 점수")
    args = ap.parse_args()

    if args.shift_probe:
        shift_probe(*args.shift_probe)

    if args.predict:
        ms, ws = _parse(args.predict)
        tot, base, gain = blend_score(ms, ws)
        w = np.array(ws) / sum(ws)
        print("구성: " + " + ".join(f"{wi:.3f}×{m}({SCORES[m]:.2f})" for wi, m in zip(w, ms)))
        print(f"  하한(가중평균, 볼록성 보장) = {base:.2f}")
        print(f"  다양성 이득                = {gain:.2f}")
        print(f"  **기대 점수**              = {tot:.2f}")
        print(f"  현 최고(549.51) 대비        = {tot - SCORES['baseline']:+.2f}")

    if args.optimize:
        ms = [m.strip() for m in args.optimize.split(",")]
        tot, w, base, gain = optimize(ms)
        print("최적 가중: " + ", ".join(f"{m}={wi:.3f}" for m, wi in zip(ms, w)))
        print(f"  기대 {tot:.2f} (하한 {base:.2f} + 이득 {gain:.2f})")
        for probe in (0.3, 0.35, 0.4, 0.45, 0.5):
            if len(ms) == 2:
                t, _, _ = blend_score(ms, [probe, 1 - probe])
                print(f"    w({ms[0]})={probe:.2f} → {t:.2f}")

    if args.measure is not None:
        known = list(zip(*_parse(args.known)))
        known = [(m, w) for m, w in zip(*_parse(args.known))]
        s, gain = measure(args.measure, args.of, known)
        print(f"역산: {args.of} 의 2025 점수 = {s:.2f}  (이득 {gain:.2f} 가정)")


if __name__ == "__main__":
    main()
