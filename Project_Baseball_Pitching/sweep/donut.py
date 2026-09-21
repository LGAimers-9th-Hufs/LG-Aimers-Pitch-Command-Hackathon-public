# -*- coding: utf-8 -*-
"""N15 donut sign-flip — 조준-산포 분해 피처 (docs/research/11 §2 N14/N15의 실용 축약판).

    python sweep/donut.py --build     # (σ,a) 역산 + δa(카운트) 적합 + 리터럴 emit → real_data.py 스플라이스

모델: 착점 ~ N2(조준점, σ²I). 성공영역 = 도넛(내측 r0 = 한복판, 외측 R1 = 크게 벗어남 경계)
− 반대방향 반평면(경계 m). 폐형식(Grubbs 1964 계보):
    p_mid(σ,a) = 1 − Q1(a/σ, r0/σ)      (내측 원반, Q1 = Marcum Q = ncx2.sf)
    p_out(σ,a) = Q1(a/σ, R1/σ)          (외측 이탈)
    p_rev(σ)   = Φ(−m/σ)                 (방향 실패 — 조준의 직교축이라 a와 독립 근사)
    p_succ = 1 − p_mid − p_out − p_rev

사전 관측검정(2026-08-06, 사전등록): 3볼 페널티 Δ가 σ프록시와 기울기 +0.290(z=3.48) —
정밀한 투수가 한복판 강제 상황에서 더 크게 무너진다. **∂P/∂σ 부호가 조준점 조건부로 반전**하는
구조가 실재하므로, 트리가 근사하기 어려운 이 매끄러운 물리 함수형을 피처 1개로 주입한다.

산출물(전부 train 고정, SERVE 리터럴):
  SF_GRID   : P_succ를 (log σ × a) 균일 격자로 표본화 — 서빙은 순수 numpy 쌍선형 보간
  SF_SIGA   : (pid*10000+season) → (σ, a) — as-of(시즌 s ← s−1까지 커리어 asof 율) 역산
  SF_DA     : 카운트 버킷(3b+s, 12종) → 조준 이동 δa — 리그 카운트별 성공률에서 1D 해 찾기
기하 상수(r0, R1, m)는 리그 평균율에서 σ_league=1 규약으로 캘리브레이션.

주의: 3률(mid/resid/rev) 중 (mid, resid)로 (σ,a)를 정확 식별(2식 2미지수), rev는 기하 검증에만.
역산은 오프라인(scipy 사용 가능 — 서버 비반입). 리터럴만 서빙으로 나간다.
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ncx2, norm

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import real_data as rd  # noqa: E402

MARK_BEGIN = "# --- SIGNFLIP:BEGIN ---"
MARK_END = "# --- SIGNFLIP:END ---"

# 리그 평균율 (train 2019-24, 2026-08-06 산출 — z_asof/S1.5와 동일 출처)
M_SUCC, M_MID, M_REV, M_RESID = 0.5352, 0.1419, 0.2152, 0.1077
SHRINK_K = 500.0                       # 율 수축 강도 (z_asof 스캔의 동결값과 일관)

# ---------------- 기하 캘리브레이션 (σ_league = 1, a_league = 0 규약) ----------------
R0 = float(np.sqrt(-2.0 * np.log(1.0 - M_MID)))      # P(R<r0 | d=0) = 1−exp(−r0²/2) = M_MID
R1 = float(np.sqrt(-2.0 * np.log(M_RESID)))          # P(R>R1 | d=0) = exp(−R1²/2)   = M_RESID
MREV = float(-norm.ppf(M_REV))                        # Φ(−m) = M_REV


def q1(a, b):
    """Marcum Q1(a,b) = P[ncx2(df=2, nc=a²) > b²] — 벡터화."""
    return ncx2.sf(np.asarray(b, float) ** 2, 2, np.asarray(a, float) ** 2)


def p_parts(sigma, a):
    """(σ, |a|) → (p_mid, p_out, p_rev, p_succ). a는 반경 방향 조준 오프셋(중심 기준)."""
    sigma = np.asarray(sigma, float)
    a = np.abs(np.asarray(a, float))
    p_mid = 1.0 - q1(a / sigma, R0 / sigma)
    p_out = q1(a / sigma, R1 / sigma)
    p_rev = norm.cdf(-MREV / sigma)
    return p_mid, p_out, p_rev, 1.0 - p_mid - p_out - p_rev


# ---------------- (mid, resid) → (σ, a) 역산 ----------------
def invert_rates(mid, res, sig_grid=None, a_grid=None):
    """정밀 격자에서 (p_mid, p_out) 최근접으로 (σ, a) 역산 + rev 예측치(misfit 진단용)."""
    sig_grid = sig_grid if sig_grid is not None else np.exp(np.linspace(np.log(0.45), np.log(2.2), 160))
    a_grid = a_grid if a_grid is not None else np.linspace(0.0, 1.6, 160)
    S, A = np.meshgrid(sig_grid, a_grid, indexing="ij")
    pm, po, pr, _ = p_parts(S, A)
    pm_f, po_f = pm.ravel(), po.ravel()
    mid = np.asarray(mid, float)[:, None]
    res = np.asarray(res, float)[:, None]
    # 정규화 거리(리그율 스케일)로 최근접 — 5천 투수-시즌 × 25.6k 격자 = 1.3e8 floats, 청크로
    out_s = np.empty(len(mid)); out_a = np.empty(len(mid)); out_rev = np.empty(len(mid))
    step = 500
    for i in range(0, len(mid), step):
        d = ((pm_f[None, :] - mid[i:i+step]) / M_MID) ** 2 \
            + ((po_f[None, :] - res[i:i+step]) / M_RESID) ** 2
        j = np.argmin(d, axis=1)
        out_s[i:i+step] = S.ravel()[j]
        out_a[i:i+step] = A.ravel()[j]
        out_rev[i:i+step] = pr.ravel()[j]
    return out_s, out_a, out_rev


def build_sigma_a(df):
    """(pid, season) → (σ, a) as-of 룩업. 시즌 s 값 = s−1 시즌말(최대 n 행)의 커리어 asof 율."""
    idx = df.groupby(["pitcher_id", rd.SEASON])["asof_pitcher_n"].idxmax()
    eos = df.loc[idx, ["pitcher_id", rd.SEASON, "asof_pitcher_n",
                       "asof_pitcher_success_rate", "asof_pitcher_middle_rate",
                       "asof_pitcher_reverse_rate"]].copy()
    n = eos["asof_pitcher_n"].to_numpy(float)
    succ = eos["asof_pitcher_success_rate"].to_numpy(float)
    mid = eos["asof_pitcher_middle_rate"].to_numpy(float)
    rev = eos["asof_pitcher_reverse_rate"].to_numpy(float)
    res = 1.0 - succ - mid - rev
    mid_sh = (mid * n + M_MID * SHRINK_K) / (n + SHRINK_K)
    res_sh = (np.clip(res, 0, 1) * n + M_RESID * SHRINK_K) / (n + SHRINK_K)
    ok = np.isfinite(mid_sh) & np.isfinite(res_sh) & (n > 0)
    sig, a, rev_hat = invert_rates(mid_sh[ok], res_sh[ok])
    keys = (eos["pitcher_id"].astype("int64") * 10000
            + (eos[rd.SEASON].astype("int64") + 1)).to_numpy()   # s−1 말 값 → 시즌 s 키
    table = {}
    for k, s_, a_ in zip(keys[ok], sig, a):
        table[int(k)] = (round(float(s_), 4), round(float(a_), 4))
    # rev misfit 진단(전체): 관측 rev(수축) vs 기하 예측 rev
    rev_sh = (rev * n + M_REV * SHRINK_K) / (n + SHRINK_K)
    mis = rev_sh[ok] - rev_hat
    print(f"   (σ,a) 역산 {ok.sum():,}건 | σ 분포 {np.percentile(sig,[5,50,95]).round(3)} "
          f"| a 분포 {np.percentile(a,[5,50,95]).round(3)}")
    print(f"   rev misfit(관측−예측): mean {mis.mean():+.4f}, |mis| p90 {np.percentile(np.abs(mis),90):.4f}"
          f"  (기하 1축 근사의 한계 — 진단용)")
    return table


# ---------------- δa(카운트) 적합 ----------------
def fit_da(df):
    """카운트 버킷 c → δa: 리그 수준에서 P_succ(σ=1, δa_c) = 리그 성공률(c)이 되도록 1D 해."""
    cnt = (df["balls_before"].astype(int) * 3 + df["strikes_before"].astype(int))
    league = df.groupby(cnt)[rd.TARGET].mean()
    a_line = np.linspace(0.0, 1.6, 2001)
    _, _, _, ps = p_parts(np.ones_like(a_line), a_line)   # σ=1 단면 (a에 단조 감소 아님 — 도넛!)
    j0 = int(np.argmin(np.abs(ps - M_SUCC)))
    a_league = float(a_line[j0])                          # 리그 전체 성공률에 대응하는 기준 조준
    da = {}
    for c, r in league.items():
        # P_succ(1, a)는 a 작을 때 낮고(한복판 위험) 중간에서 최대 후 감소 — 목표율과 가장 가까운
        # a 중 "공격적(작은 a)" 해를 취한다: 3볼(c=9)일수록 목표율이 낮고 a도 작아야 물리적으로 정합.
        j = int(np.argmin(np.abs(ps - float(r))))
        da[int(c)] = round(float(a_line[j]) - a_league, 4)   # 리그 기준 중심화 — a_p에 더하는 이동량
    print(f"   a_league={a_league:.4f}, δa(중심화):", da)
    print("   리그 성공률(카운트):", {int(k): round(float(v), 4) for k, v in league.items()})
    return da


# ---------------- 그리드 + emit ----------------
def build_grid(n_sig=48, n_a=64):
    lsig = np.linspace(np.log(0.45), np.log(2.2), n_sig)
    a = np.linspace(0.0, 2.4, n_a)                        # a_p + δa 범위까지 커버
    S, A = np.meshgrid(np.exp(lsig), a, indexing="ij")
    _, _, _, ps = p_parts(S, A)
    return lsig, a, np.clip(ps, 0.0, 1.0)


def emit(table, da, lsig, a_grid, grid):
    rows = ", ".join("[" + ", ".join(f"{v:.4f}" for v in row) + "]" for row in grid)
    siga = ", ".join(f"{k}: ({v[0]}, {v[1]})" for k, v in sorted(table.items()))
    das = ", ".join(f"{k}: {v}" for k, v in sorted(da.items()))
    return "\n".join([
        MARK_BEGIN,
        "# sweep/donut.py가 재생성한다 — 손으로 편집 금지. N15 donut sign-flip 리터럴.",
        f"SF_LSIG0, SF_LSIG1, SF_NSIG = {lsig[0]!r}, {lsig[-1]!r}, {len(lsig)}",
        f"SF_A0, SF_A1, SF_NA = {a_grid[0]!r}, {a_grid[-1]!r}, {len(a_grid)}",
        f"SF_GRID = np.array([{rows}], dtype='float32')",
        f"SF_SIGA = {{{siga}}}",
        f"SF_DA = {{{das}}}",
        MARK_END])


def splice(literal, path=None):
    p = Path(path or HERE / "real_data.py")
    src = p.read_text(encoding="utf-8")
    pat = re.compile(re.escape(MARK_BEGIN) + r".*?" + re.escape(MARK_END), re.DOTALL)
    if not pat.search(src):
        sys.exit(f"마커({MARK_BEGIN})가 real_data.py에 없다 — SERVE 블록에 먼저 추가할 것")
    p.write_text(pat.sub(lambda _: literal, src), encoding="utf-8")
    print(f"스플라이스 완료: {p} (+{len(literal):,} bytes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    print(f">> 기하: r0={R0:.4f} R1={R1:.4f} m_rev={MREV:.4f} (σ_league=1 규약)")
    if not args.build:
        return
    df = rd.load_train()
    table = build_sigma_a(df)
    da = fit_da(df)
    lsig, a_grid, grid = build_grid()
    splice(emit(table, da, lsig, a_grid, grid))


if __name__ == "__main__":
    main()
