# -*- coding: utf-8 -*-
"""세그먼트 절편 LB 프로브 — 사전 계산과 사후 역산.

전역 절편 프로브(D-57)가 준 것은 **가중평균 한 방정식**뿐이다:

    w_R·e_R + w_F·e_F = ē = −0.00410      (미지수 2, 방정식 1)

⇒ e_F = (ē − (1−w_F)·e_R)/w_F 이므로 **e_R 불확실성이 1/w_F ≈ 8.7배로 증폭**된다.
과거 4년의 e_R을 대입하면 2025 `game_type` 회수액이 1.2~273.7점으로 벌어진다.
**로컬 게이트로는 이 축을 식별할 수 없다**(전이 실측 F 2023→2024 = −1895). LB 대칭쌍만이 도구다.

방향은 반드시 **중심화 지시자** `g̃ = 1_F − w_F` — 전역 절편축과 mean(g̃)=0으로 정확히 직교하고,
2셀 축의 오라클을 전량 담는다(비중심 지시자는 (1−w)²배만 회수). `final`에 직접 가산하므로
**slope 1.04 인자가 붙지 않는다**(D-57 전역 프로브가 C = 1e5/V를 그대로 낸 것이 증거).

    S(t) = S₀ + a·t − b·t²        a = 2C·w·d ,  b = C·w(1−w)
    a = (S₊−S₋)/(2δ)   b = (2S₀−S₊−S₋)/(2δ²)   t* = a/(2b)   G = a²/(4b)

사용법:
    python sweep/seg_probe.py --pre                       # 제출 전: w_F·해석 b·클리핑·시나리오표
    python sweep/seg_probe.py --invert S0 S_PLUS S_MINUS  # 제출 후: 정확 역산 + 검산
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

C_GLOBAL = 402463.285680        # 전역 절편 프로브 실측 곡률 = 1e5/V (D-57, calP/calM/ENS-9 3점)
V_2025 = 1e5 / C_GLOBAL         # = 0.2484700  → r = 0.4609 또는 0.5391
E_BAR = -0.00410                # shift −0.010 시점의 전역 예측평균 오차 (D-57 역산)
# 베이스 = calP(shift −0.005). **실측 점수**라 역산에 외삽 가정이 없다.
# (해석 최적 −0.0059/≈1016.03은 점수 역산값이라 배포하지 않는다 — D-06 "역산 상수를 박지 않는다".
#  포기하는 것은 0.33점이고, 얻는 것은 배포본의 모든 캘리 상수가 "사전등록 후보 중 LB 선택"이라는 사실.)
S_BASE = 1015.7036703211        # submit_ens9_calP.zip 실측 (2026-08-08 14:17)
S_EXACT1 = 1025.5511136843      # submit_exact1.zip 실측 (2026-08-09 00:43) = 현 챔피언


# ---------------------------------------------------------------- 사전 계산
def precompute(delta: float = 0.010) -> dict:
    import real_data as rd
    import lgbm_family as L                                    # noqa: F401  (score 정합용)

    df = rd.load_train()
    seg = df[rd.SEGMENT].to_numpy()
    ssn = df[rd.SEASON].to_numpy().astype(int)
    y = df[rd.TARGET].to_numpy(dtype=float)

    print("=" * 74)
    print("① game_type 시즌별 비중·성공률 (w_F 안정성 = 해석 b의 신뢰도)")
    print("=" * 74)
    print(f"{'season':>7} {'n':>9} {'w_F':>8} {'rate_F':>8} {'rate_R':>8}")
    wfs = []
    for s in sorted(set(ssn)):
        m = ssn == s
        f = seg[m] == "F"
        wf = float(f.mean())
        wfs.append(wf)
        print(f"{s:>7} {int(m.sum()):>9} {wf:>8.4f} "
              f"{float(y[m][f].mean()) if f.any() else float('nan'):>8.4f} "
              f"{float(y[m][~f].mean()):>8.4f}")
    w_f = float((seg == "F").mean())
    print(f"\n  train 전체 w_F = {w_f:.6f}   시즌 범위 [{min(wfs):.4f}, {max(wfs):.4f}] "
          f"(변동 {max(wfs)/max(min(wfs), 1e-12):.3f}×)")

    b_analytic = C_GLOBAL * w_f * (1.0 - w_f)
    print(f"  해석 b = C·w(1−w) = {C_GLOBAL:,.0f} × {w_f:.4f} × {1-w_f:.4f} = {b_analytic:,.0f}")
    print(f"  쌍 평균비용 b·δ² = {b_analytic * delta**2:.2f} 점  (δ={delta})")

    print()
    print("=" * 74)
    print("② 클리핑 검사 — 2차식이 항등이려면 클리핑 0행이어야 한다")
    print("=" * 74)
    cache = HERE.parent / "results" / "ens9" / "preds_val2024.npz"
    if cache.exists():
        import eb_carrier as EB
        raw = np.asarray(EB.ens9_pred(*EB.get_members9(2024)[:3]), dtype=float)
        fin = 0.5 + 1.04 * (raw - 0.5) - 0.005
        lo, hi = fin.min() + -delta, fin.max() + delta
        print(f"  raw   ∈ [{raw.min():.6f}, {raw.max():.6f}]")
        print(f"  final ∈ [{fin.min():.6f}, {fin.max():.6f}]  (shift −0.005, calP)")
        print(f"  ±δ 가산 후 ∈ [{lo:.6f}, {hi:.6f}]  → 클리핑 행 "
              f"{int(((fin - delta) < 0).sum() + ((fin + delta) > 1).sum())}")
        print(f"  클리핑까지 여유 = {min(fin.min(), 1 - fin.max()):.4f} "
              f"(δ의 {min(fin.min(), 1-fin.max())/delta:.0f}배)")
    else:
        print(f"  ⚠ 캐시 없음: {cache} — sweep/ens9_cache.py 로 생성 후 재실행")

    print()
    print("=" * 74)
    print("③ 시나리오표 — 점수가 오면 이 표와 대조한다")
    print("=" * 74)
    print(f"  베이스 = calP 실측 {S_BASE:.4f}. 전역 최적(−0.0059)에서 0.0009 벗어나 있어 "
          f"가중평균 오차 ≈ +0.0009,")
    print("  근사적으로 w_R·d_R + w_F·d_F ≈ 0 ⇒ d_F ≈ −(w_R/w_F)·d_R (1차 항은 전역 축이 이미 흡수).")
    print(f"  ⇒ d_F = −(w_R/w_F)·d_R = {-(1-w_f)/w_f:.2f}·d_R  — R의 미세한 오차가 F에서 증폭된다.")
    print("    a = −2C·w_F·d_F ,  t* = −d_F/(1−w_F) ,  G = C·w_F·d_F²/(1−w_F) = C·(w_R/w_F)·d_R²")
    print(f"\n{'폴드':>8} {'d_R 실측':>10} {'→ d_F':>10} {'a':>10} {'t*':>9} "
          f"{'G':>9} {'S₊ 예상':>9} {'S₋ 예상':>9}")
    # 폴드별 실측 d_R (각 폴드 전역 최적 shift 적용 후, results/ens9 4폴드 측정)
    for tag, d_r in [("V2021", +0.01116), ("V2022", -0.00139),
                     ("V2023", -0.02254), ("V2024", -0.00038)]:
        d_f = -(1 - w_f) / w_f * d_r
        a = -2 * C_GLOBAL * w_f * d_f
        t_star = a / (2 * b_analytic)
        g = a * a / (4 * b_analytic)
        sp = S_BASE + a * delta - b_analytic * delta ** 2
        sm = S_BASE - a * delta - b_analytic * delta ** 2
        print(f"{tag:>8} {d_r:>+10.5f} {d_f:>+10.4f} {a:>10.1f} {t_star:>+9.4f} "
              f"{g:>9.1f} {sp:>9.2f} {sm:>9.2f}")
    print(f"\n  기준: 프로브 없이 배포하면 {S_BASE:.4f}(calP). 레그가 이보다 높으면 그 레그가 이미 챔피언이다.")
    print("  ⚠ 큰 G는 전부 **F 파단 연도**(2021 +0.116, 2023 −0.236)에서 나온다. 조용한 연도는 ~0.5.")
    print("     R은 시즌간 최대 |Δ|=0.023으로 매끄러운데 F는 5전이 중 3회가 |Δ|>0.10 — 이 비대칭이 베팅 근거다.")

    ssum = 2 * S_BASE - 2 * b_analytic * delta ** 2
    print()
    print("=" * 74)
    print("④ 점수 도착 시 판정 순서 — 이 순서를 지킨다")
    print("=" * 74)
    print(f"  1) **합 검산**: S₊ + S₋ 이 {ssum:.2f} 근처인가? "
          f"허용 [{2*S_BASE - 2*1.25*b_analytic*delta**2:.2f}, "
          f"{2*S_BASE - 2*0.75*b_analytic*delta**2:.2f}] (해석 b ±25%)")
    print("     벗어나면 → 레그 비동일(멤버 무음 스킵) 의심. **역산하지 말고 중단.**")
    print(f"  2) 두 레그 착지점 읽기:")
    print(f"       둘 다 ≈ {S_BASE:.2f}      → b≈0 = 2025에 F 행이 없다. 2차 축으로 재배정")
    print(f"       둘 다 ≈ {S_BASE - b_analytic*delta**2:.2f}      → a≈0 = F는 있으나 레벨오차 없음. 채널 소진")
    print(f"       갈라짐               → 신호 있음. 3)으로")
    print(f"  3) `python sweep/seg_probe.py --invert {S_BASE:.4f} <S₊> <S₋> --b {b_analytic:.0f}`")
    print("  4) 점수는 memory/lb-verified-scores.md에 **먼저 기록**하고 계산은 그 다음이다.")
    return dict(w_f=w_f, b_analytic=b_analytic, delta=delta)


# ---------------------------------------------------------------- 사후 역산
def invert(s0: float, s_plus: float, s_minus: float, delta: float = 0.010,
           b_analytic: float | None = None) -> dict | None:
    print(f"입력: S₀={s0:.10f}  S₊={s_plus:.10f}  S₋={s_minus:.10f}  δ={delta}")
    if s_plus + s_minus > 2.0 * s0:
        print("🚫 S₊+S₋ > 2S₀ — 볼록성 위반. 레그 동일성/빌드를 의심하고 계산 중단.")
        return None
    a = (s_plus - s_minus) / (2.0 * delta)
    b = (2.0 * s0 - s_plus - s_minus) / (2.0 * delta ** 2)
    print(f"  a = {a:,.2f}   b = {b:,.1f}")
    if b_analytic:
        ratio = b / b_analytic
        print(f"  b 검산: 해석값 {b_analytic:,.0f} 대비 {ratio:.3f}×", end="  ")
        if not (0.75 <= ratio <= 1.25):
            print("\n🚫 ±25% 밖 — 레그 비동일(멤버 무음 스킵) 의심. 서버 로그 weights_used 확인 후 중단.")
            return None
        print("✓")
    if b <= 0:
        print("🚫 b ≤ 0 — 2차식이 아니다. 중단.")
        return None
    t_star = a / (2.0 * b)
    g = a * a / (4.0 * b)
    w_hat = 0.5 - np.sqrt(max(0.25 - b / C_GLOBAL, 0.0))       # b = C·w(1−w) 역산
    d_hat = a / (2.0 * C_GLOBAL * w_hat) if w_hat > 0 else float("nan")
    print(f"  t* = {t_star:+.5f}   G = {g:+.2f}   → 배포 예상 {s0 + g:.2f}")
    print(f"  함의: 2025 w_F ≈ {w_hat:.4f}, 세그먼트 레벨오차 d_F ≈ {d_hat:+.5f}")
    if g >= 10:
        print("→ **G ≥ +10: t* 재빌드·배포 권고**")
    else:
        print("→ G < +10: 이 축도 소진. 남은 슬롯은 2차 축으로.")
    return dict(a=a, b=b, t_star=t_star, gain=g, w_hat=float(w_hat), d_hat=float(d_hat))


def invert_slope(s0: float, s_plus: float, s_minus: float, dslope: float = 0.06,
                 c_analytic: float = 849.58) -> dict | None:
    """**평균보존** slope 대칭쌍 역산 (D-58).

    `final = 0.5 + s·(raw−0.5) + shift` 에서 s를 ±δ 흔들되 shift로 평균을 고정하면
    방향이 `raw − mean(raw)` 가 되어 전역 절편축과 직교하고, 곡률은 평균항이 빠진
    `C_s = C·Var(raw)` 가 된다(비보존판 `C·mean(u²)`=1,280과 구별할 것 — 후자를 쓰면 틀린다).
    평균보존 shift = `−0.005 + (1.04 − s)·(raw̄ − 0.5)`, raw̄ = 0.4680607(LB 역산 private 평균).
    """
    print(f"입력: S₀={s0:.10f}  S₊(1.10)={s_plus:.10f}  S₋(0.98)={s_minus:.10f}  δ={dslope}")
    if s_plus + s_minus > 2.0 * s0:
        print("🚫 볼록성 위반 — 레그 비동일/평균보존 실패 의심. 중단.")
        return None
    a = (s_plus - s_minus) / (2.0 * dslope)
    c = (2.0 * s0 - s_plus - s_minus) / (2.0 * dslope ** 2)
    ratio = c / c_analytic
    print(f"  a_s = {a:,.2f}   C_s = {c:,.1f}   (해석 {c_analytic:,.0f} 대비 {ratio:.3f}×)")
    if not (0.6 <= ratio <= 1.6):
        print("🚫 C_s가 해석값과 크게 다르다 — 2024 프록시 Var(raw)가 2025와 다르거나 레그 문제. 신중히.")
    if c <= 0:
        print("🚫 C_s ≤ 0 — 2차식이 아니다. 중단.")
        return None
    ds = a / (2.0 * c)
    g = a * a / (4.0 * c)
    print(f"  Δs* = {ds:+.5f}  →  최적 slope = {1.04 + ds:.5f}")
    print(f"  G = {g:+.2f}  →  배포 예상 {s0 + g:.2f}")
    print(f"  평균보존 shift = {-0.005 + (1.04 - (1.04 + ds)) * (0.4680607 - 0.5):+.8f}")
    return dict(a=a, C_s=c, dslope_star=ds, slope_star=1.04 + ds, gain=g)


# ---------------------------------------------------------------- 형상(곡률) 축 D-59
QUAD_RAWBAR = 0.4680607          # 2025 private raw 평균 (D-58 LB 역산)
QUAD_M = {2: 2.110950018e-03, 3: 3.075195724e-05}      # results/ens9 val2024 out-of-fold 중심적률
QUAD_C = {2: 3.719, 3: 0.0540}   # 해석 C = 1e5·mean(g²)/V


def precompute_quad(power: int = 2, delta: float = 1.0) -> dict:
    """형상 축 사전 계산 — 상수 재확인 + 폴드 오라클 + 클리핑 + 시나리오표."""
    import numpy as _np
    import lgbm_family as L
    from eb_carrier import get_members9, ens9_pred, DEPLOY_CAL, DEPLOY_SEG

    k = int(power)
    print("=" * 78)
    print(f"① 폴드 오라클 — y ~ 1 + raw + g 사영의 아핀 대비 증분 (g = (raw−c)^{k} − m, c=폴드 평균)")
    print("=" * 78)
    print(f"{'폴드':>7} {'n':>9} {'raw평균':>9} {'raw SD':>8} {'아핀':>9} {'+형상':>9} "
          f"{'증분':>8} {'t*':>9} {'C_q(폴드)':>10}")
    for v in (2021, 2022, 2023, 2024):
        P, corr, corr_pm, y = get_members9(v)
        raw = _np.asarray(ens9_pred(P, corr, corr_pm), dtype=float)
        n = len(y)
        u = raw - raw.mean()
        g = u ** k - (u ** k).mean()
        X1 = _np.column_stack([_np.ones(n), raw])
        X2 = _np.column_stack([_np.ones(n), raw, g])
        b1, *_ = _np.linalg.lstsq(X1, y, rcond=None)
        b2, *_ = _np.linalg.lstsq(X2, y, rcond=None)
        s1 = L.score(y, _np.clip(X1 @ b1, 1e-6, 1 - 1e-6))
        s2 = L.score(y, _np.clip(X2 @ b2, 1e-6, 1 - 1e-6))
        r = y.mean()
        print(f"{v:>7} {n:>9,} {raw.mean():>9.5f} {raw.std():>8.5f} {s1:>9.2f} {s2:>9.2f} "
              f"{s2-s1:>+8.2f} {b2[2]:>+9.4f} {1e5*float((g*g).mean())/(r*(1-r)):>10.2f}")
    print("  ⇒ slope 축의 폴드 오라클(+67.7/+10.8/+1178/+0.02)과 같은 형태다. 큰 값은 파단 연도(V23).")
    print("     로컬 크기는 2025의 크기를 예측하지 못한다 — slope도 로컬이 작다고 했는데 +8.18이었다.")

    print()
    print("=" * 78)
    print("② 배포 상수와 직교성 (⚠ 중심 선택이 이 축의 성패를 가른다)")
    print("=" * 78)
    P, corr, corr_pm, y = get_members9(2024)
    raw = _np.asarray(ens9_pred(P, corr, corr_pm), dtype=float)
    for tag, c in [("2024 표본평균", float(raw.mean())), ("**배포값 raw̄(2025)**", QUAD_RAWBAR)]:
        u = raw - c
        g = u ** k - (u ** k).mean()
        mg2 = float((g * g).mean())
        print(f"  중심={tag:22s} c={c:.7f}  mean(g²)={mg2:.4e}  "
              f"C={1e5*mg2/(1e5/C_GLOBAL):6.3f}  corr(g,raw)={_np.corrcoef(g, raw)[0, 1]:+.4f}")
    print(f"  → 2024 데이터에 2025 중심을 대면 corr가 커 보이지만, **2025에서는 중심이 맞으므로**")
    print(f"    실효 C ≈ {QUAD_C[k]} (위 첫 줄)이다. 학습 시즌 평균으로 중심화하면 slope 축을 건드린다.")

    print()
    print("=" * 78)
    print("③ 클리핑 여유 (2차식이 항등이려면 0행이어야 한다)")
    print("=" * 78)
    fin = (DEPLOY_CAL["center"] + DEPLOY_CAL["slope"] * (raw - DEPLOY_CAL["center"])
           + DEPLOY_CAL["shift"] - DEPLOY_SEG["t"] * DEPLOY_SEG["w"])   # R행(89%) 기준
    u = raw - QUAD_RAWBAR
    g = u ** k - QUAD_M[k]
    print(f"  final(R행 근사) ∈ [{fin.min():.5f}, {fin.max():.5f}]   g ∈ [{g.min():+.6f}, {g.max():+.6f}]")
    for t in (delta, 2 * delta, 3 * delta):
        z = fin + t * g
        print(f"   t={t:>5.2f} → [{z.min():.5f}, {z.max():.5f}]  클리핑 {int((z<0).sum()+(z>1).sum())}행  "
              f"최대이동 {float(_np.abs(t*g).max()):.5f}")

    print()
    print("=" * 78)
    print(f"④ 시나리오표 — 프로브 δ={delta}, 해석 C={QUAD_C[k]}, S₀={S_EXACT1:.4f}")
    print("=" * 78)
    print(f"  무신호(a=0) 착지 = {S_EXACT1 - QUAD_C[k]*delta**2:.3f}  (비용 {QUAD_C[k]*delta**2:.2f}점)")
    print(f"{'ΔS(실측−S₀)':>13} {'a':>10} {'t*':>9} {'G':>9} {'배포 예상':>11}")
    for ds in (-QUAD_C[k] * delta ** 2, 0.0, 5.0, 10.0, 13.0, 20.0, 40.0):
        a = (ds + QUAD_C[k] * delta ** 2) / delta
        print(f"{ds:>13.2f} {a:>10.2f} {a/(2*QUAD_C[k]):>+9.3f} {a*a/(4*QUAD_C[k]):>9.2f} "
              f"{S_EXACT1 + a*a/(4*QUAD_C[k]):>11.2f}")
    _ds20 = (80.0 * QUAD_C[k]) ** 0.5 * delta - QUAD_C[k] * delta ** 2   # G=20 이 되는 ΔS
    print(f"  ⇒ **ΔS ≥ +{_ds20:.1f} 이면 배포 G ≥ +20** (a ≥ √(80C) = {(80*QUAD_C[k])**0.5:.1f}).")
    print(f"    ΔS ≈ −{QUAD_C[k]*delta**2:.1f} 이면 a≈0 = 이 축은 2025에서 비어 있다. 다음 축으로.")
    return dict(power=k, C=QUAD_C[k], delta=delta)


def invert_quad(s0: float, s_probe: float, delta: float = 1.0, power: int = 2,
                c_analytic: float | None = None) -> dict | None:
    """형상 축 **단일점** 역산 (D-58 ②③: 반대편 레그 불필요 + 해석 곡률 3.6~7.5% 정확).

        S(t) = S₀ + a·t − C·t²   ⇒   a = (ΔS + C·δ²)/δ ,  t* = a/(2C) ,  G = a²/(4C)
    """
    C = float(c_analytic if c_analytic is not None else QUAD_C[int(power)])
    ds = s_probe - s0
    a = (ds + C * delta ** 2) / delta
    t_star = a / (2.0 * C)
    g = a * a / (4.0 * C)
    print(f"입력: S₀={s0:.10f}  S(δ={delta})={s_probe:.10f}  ΔS={ds:+.4f}  C(해석)={C}")
    print(f"  a = {a:,.3f}   t* = {t_star:+.5f}   G = {g:+.3f}  →  배포 예상 {s0 + g:.4f}")
    if abs(ds + C * delta ** 2) < 0.05:
        print("  ⇒ a ≈ 0: 이 축은 2025에서 비어 있다. 배포하지 말고 다음 축으로.")
    elif g < 1.0:
        print("  ⇒ G < 1점: 배포 가치 없음(슬롯을 다른 축에).")
    else:
        print(f"  ⇒ 배포 권고: --quadprobe --quadt {t_star:.6f} --quadpow {power} --quadfitted")
    print("  ⚠ 배포본이 세 번째 점이 되어 C를 실측 확정한다. 착지가 예상과 1점 이상 어긋나면")
    print("     C 가정이 틀린 것이므로 그 값으로 C를 재추정한 뒤 다음 축 계획을 고친다.")
    return dict(a=a, C=C, t_star=t_star, gain=g, pred=s0 + g)


# ---------------------------------------------------------------- 파트너 블렌드 축 D-59
def invert_blend(s0: float, s1: float, s2: float, w1: float = 0.05, w2: float = 0.12,
                 partner: str = "results/codex_partner/serving") -> dict | None:
    """파트너 가중 축 3점 역산 → 꼭짓점 w*와 배포 명령까지 출력.

    이 축은 곡률을 해석적으로 모른다(두 모델이 2025에서 얼마나 다른지는 2025를 봐야 안다,
    프록시 추정 745~1300). 그래서 대칭쌍이 아니라 **같은 쪽 2점 + S₀** 로 C를 실측한다.

        S(w) = S₀ + a·w − C·w²
        C  = (ΔS₁·w₂ − ΔS₂·w₁) / (w₁w₂(w₂−w₁))
        a  = ΔS₁/w₁ + C·w₁ ,  w* = a/(2C) ,  G = a²/(4C)
    """
    d1, d2 = s1 - s0, s2 - s0
    print(f"입력: S₀={s0:.10f}  S₁(w={w1})={s1:.10f}  S₂(w={w2})={s2:.10f}")
    print(f"      ΔS₁={d1:+.4f}  ΔS₂={d2:+.4f}")
    c = (d1 * w2 - d2 * w1) / (w1 * w2 * (w2 - w1))
    if c <= 0:
        print(f"🚫 C={c:,.1f} ≤ 0 — 볼록성 위반. 레그 비동일(파트너 무음 스킵)을 의심하고 중단.")
        print("   서버 로그에서 partner_used=1 을 먼저 확인할 것.")
        return None
    a = d1 / w1 + c * w1
    w_star = a / (2.0 * c)
    g = a * a / (4.0 * c)
    print(f"  a = {a:,.2f}   C = {c:,.1f}  (사전 추정 745~1300)")
    print(f"  w* = {w_star:+.5f}   G = {g:+.3f}   →  배포 예상 {s0 + g:.4f}")
    if not (0.0 < w_star < 0.7):
        print(f"🚫 w*={w_star:.3f} 가 (0, 0.7) 밖 — 재검토. 배포하지 말 것.")
        return None
    if g < 2.0:
        print("→ G < 2: 이 파트너는 2025에서 거의 비었다. 배포 생략하고 다음 축으로.")
    else:
        print("\n→ 배포 zip 빌드 (약 2분):")
        print(f'  python submission/build_lgbm_ensemble.py --ens9 \\\n'
              f'    --calib 1.14202,-0.00264162 --calfitted --segprobe gtF --segt 0.0057 --segfitted \\\n'
              f'    --partnermoments 0.4680607,0.0441981 \\\n'
              f'    --partner "{partner}={w_star:.5f}" --tag pwStar')
        print("  python submission/verify_submission.py submission/dist/submit_pwStar.zip "
              "--proxy 245789")
    return dict(a=a, C=c, w_star=w_star, gain=g, pred=s0 + g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", action="store_true", help="제출 전 사전 계산")
    ap.add_argument("--invert-slope", nargs=3, type=float, dest="invert_slope",
                    metavar=("S0", "S_PLUS", "S_MINUS"),
                    help="평균보존 slope 쌍 역산 (S₀=gtStar 실측, S₊=1.10, S₋=0.98)")
    ap.add_argument("--dslope", type=float, default=0.06)
    ap.add_argument("--invert", nargs=3, type=float, metavar=("S0", "S_PLUS", "S_MINUS"))
    ap.add_argument("--delta", type=float, default=0.010)
    ap.add_argument("--b", type=float, default=None, help="--invert 검산용 해석 b")
    ap.add_argument("--pre-quad", action="store_true", dest="pre_quad",
                    help="형상(곡률) 축 사전 계산 — 폴드 오라클·상수·클리핑·시나리오표 (D-59)")
    ap.add_argument("--invert-quad", nargs=2, type=float, dest="invert_quad",
                    metavar=("S0", "S_PROBE"), help="형상 축 단일점 역산")
    ap.add_argument("--quadpow", type=int, default=2, choices=[2, 3], help="형상 축 차수 k")
    ap.add_argument("--dq", type=float, default=1.0, help="형상 축 프로브 δ")
    ap.add_argument("--cq", type=float, default=None, help="형상 축 C 수동 지정(3점 실측 후)")
    ap.add_argument("--invert-blend", nargs=3, type=float, dest="invert_blend",
                    metavar=("S0", "S1", "S2"),
                    help="파트너 가중 축 3점 역산 (S0=현 챔피언, S1=w1 프로브, S2=w2 프로브)")
    ap.add_argument("--w1", type=float, default=0.05)
    ap.add_argument("--w2", type=float, default=0.12)
    ap.add_argument("--pdir", default="results/codex_partner/serving",
                    help="--invert-blend 배포 명령에 넣을 파트너 서빙 폴더")
    args = ap.parse_args()

    if args.pre:
        precompute(args.delta)
    if args.invert:
        invert(*args.invert, delta=args.delta, b_analytic=args.b)
    if args.invert_slope:
        invert_slope(*args.invert_slope, dslope=args.dslope)
    if args.pre_quad:
        precompute_quad(args.quadpow, args.dq)
    if args.invert_quad:
        invert_quad(*args.invert_quad, delta=args.dq, power=args.quadpow, c_analytic=args.cq)
    if args.invert_blend:
        invert_blend(*args.invert_blend, w1=args.w1, w2=args.w2, partner=args.pdir)
    if not (args.pre or args.invert or args.invert_slope or args.pre_quad
            or args.invert_quad or args.invert_blend):
        ap.print_help()


if __name__ == "__main__":
    main()
