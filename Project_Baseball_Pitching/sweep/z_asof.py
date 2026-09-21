# -*- coding: utf-8 -*-
"""(pid, season) as-of 투수 신뢰도 로짓 테이블 — N11(z→GLM) + N22(k 예측우도 스캔) + N23(시즌경계 ρ).

    python sweep/z_asof.py --scan                 # (K, ρ) 그리드를 2022/2023 평가로 스캔 (2024는 게이트용으로 봉인)
    python sweep/z_asof.py --emit --k 250 --rho 0.75   # 리터럴 생성 → real_data.py SERVE 블록에 스플라이스

설계 (docs/research/11 §3 N22/N23, 13 §6 N11):
  - 시즌 s 행의 z = 시즌 < s 데이터만으로 계산한 수축 로짓 → **구성상 전 폴드 누수 0 + 서빙 동형**
    (2025 키 = ≤2024 전체 = 실제 배포 값).
  - 수축: shrunk = m + n/(n+K)·(p−m). K는 모멘트법(기존 build_credibility) 또는 고정 그리드 —
    선택은 **홀드아웃 refinement**(레벨 제거 지표)로 한다. 모멘트법 τ² 과소추정이 스프레드 20%
    과수축의 유력 원인이므로(문서 11 §1.3) K를 직접 고르는 것이 이 모듈의 존재 이유다.
  - 시즌 경계 지속 ρ: z' = logit(m) + ρ·(logit(shrunk) − logit(m)). 근거 prior = 시즌간 신호
    유지율 52~74%(Böttger&Vischer 2026), xCTRL r=0.65.
  - 키 = pid*10000 + season (int). 2019는 선행 시즌이 없어 미등재(NaN → HGB 분기/GLM 중앙값 대치).
  - ⚠ (K, ρ) 선택은 2022/2023 평가만 쓴다 — 2024를 쓰면 주 게이트(refV24)가 선택에 오염된다.

역방향 폴드(R2019)용: build_z_asof(exclude=(2019,)) — 2019 라벨이 fit 행의 as-of 집계에 새는
것을 차단한 대체 테이블(run_real.py가 reverse 폴드에서 교체 사용).
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import real_data as rd  # noqa: E402

MARK_BEGIN = "# --- Z_ASOF:BEGIN ---"
MARK_END = "# --- Z_ASOF:END ---"
K_GRID = ["moment", 60, 120, 250, 500, 1000]
RHO_GRID = [1.0, 0.85, 0.75, 0.6, 0.5]
EPS = 1e-6


def _refinement(y, p):
    """레벨(평균 오차) 성분을 제거한 skill 점수 — run_real.refinement와 동일 수식(클립 없음)."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    r = y.mean()
    b = float(np.mean((p - y) ** 2))
    d = float(p.mean() - r)
    return 1e5 * (1.0 - b / (r * (1 - r)) + d * d / (r * (1 - r)))


def _prefix_stats(df, exclude=()):
    """시즌 s → (그 이전 시즌 전체의) 투수별 n·성공수 + 리그 m + 모멘트 K.

    한 번 계산해 두면 (K, ρ) 그리드는 벡터 연산만으로 순회할 수 있다.
    """
    seasons = sorted(int(s) for s in df[rd.SEASON].unique())
    out = {}
    for s in seasons[1:] + [seasons[-1] + 1]:            # 2020..2025
        mask = (df[rd.SEASON] < s) & ~df[rd.SEASON].isin(list(exclude))
        sub = df.loc[mask]
        if not len(sub):
            continue
        g = sub.groupby("pitcher_id")[rd.TARGET]
        n = g.size().astype(float)
        k = g.sum().astype(float)
        p = k / n
        N = float(n.sum())
        m = float(k.sum() / N)
        I = len(n)
        epv = float((n * p * (1 - p)).sum() / max((n - 1).sum(), 1.0))
        c_star = N - float((n ** 2).sum()) / N
        vhm = (float((n * (p - m) ** 2).sum()) - (I - 1) * epv) / max(c_star, 1.0)
        k_moment = epv / vhm if vhm > 1e-10 else 1e9
        out[s] = {"n": n, "p": p, "m": m, "k_moment": float(k_moment)}
    return out


def _table_from_stats(stats, K="moment", rho=1.0):
    """prefix 통계 → {pid*10000+season: z_logit} (ρ는 리그 로짓 기준으로 감쇠)."""
    table = {}
    for s, st in stats.items():
        n, p, m = st["n"], st["p"], st["m"]
        Kv = st["k_moment"] if K == "moment" else float(K)
        z = n / (n + Kv)
        shrunk = np.clip(m + z * (p - m), 1e-3, 1 - 1e-3)
        lg = np.log(shrunk / (1 - shrunk))
        lg_m = float(np.log(m / (1 - m)))
        lg = lg_m + rho * (lg - lg_m)
        for pid, v in zip(n.index.astype(int), lg.astype(float)):
            table[int(pid) * 10000 + int(s)] = v
    return table


def build_z_asof(df, K="moment", rho=1.0, exclude=()):
    return _table_from_stats(_prefix_stats(df, exclude=exclude), K=K, rho=rho)


def _eval_table(df, table, eval_season):
    """평가 시즌 행에서 σ(z)의 refinement — 커버된 행만(그리드 간 커버리지 동일해 공정)."""
    sub = df[df[rd.SEASON] == eval_season]
    key = sub["pitcher_id"].astype("int64") * 10000 + int(eval_season)
    z = key.map(table)
    ok = z.notna().to_numpy()
    y = sub[rd.TARGET].to_numpy()[ok]
    p = 1.0 / (1.0 + np.exp(-z.to_numpy(dtype=float)[ok]))
    return _refinement(y, p), float(ok.mean())


def scan(df, eval_seasons=(2022, 2023)):
    print(f">> (K, ρ) 그리드 스캔 — 평가 {eval_seasons} (2024 봉인), 지표 = σ(z) 단독 refinement")
    stats = _prefix_stats(df)
    print("   prefix 모멘트 K:", {s: round(st["k_moment"]) for s, st in stats.items()})
    rows = []
    for K in K_GRID:
        for rho in RHO_GRID:
            table = _table_from_stats(stats, K=K, rho=rho)
            refs, covs = [], []
            for es in eval_seasons:
                ref, cov = _eval_table(df, table, es)
                refs.append(ref), covs.append(cov)
            rows.append({"K": K, "rho": rho, "ref_mean": float(np.mean(refs)),
                         **{f"ref{es}": r for es, r in zip(eval_seasons, refs)},
                         "coverage": float(np.mean(covs))})
    out = pd.DataFrame(rows).sort_values("ref_mean", ascending=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    best = out.iloc[0]
    print(f"\n권고: K={best['K']} ρ={best['rho']}  (ref_mean {best['ref_mean']:.1f}, "
          f"평가시즌 간 순위 안정성은 표에서 육안 확인)")
    return out


def emit(table, K, rho, decimals=4):
    items = ", ".join(f"{k}: {round(v, decimals)}" for k, v in sorted(table.items()))
    lines = [MARK_BEGIN,
             "# sweep/z_asof.py가 재생성한다 — 손으로 편집 금지. (pid*10000+season) → as-of 수축 로짓.",
             f'Z_ASOF_META = {{"K": {K!r}, "rho": {rho}, "n": {len(table)}}}',
             f"Z_ASOF = {{{items}}}",
             MARK_END]
    return "\n".join(lines)


def splice(literal, path=None):
    p = Path(path or HERE / "real_data.py")
    src = p.read_text(encoding="utf-8")
    pat = re.compile(re.escape(MARK_BEGIN) + r".*?" + re.escape(MARK_END), re.DOTALL)
    if not pat.search(src):
        sys.exit(f"마커({MARK_BEGIN}...{MARK_END})를 {p}에서 찾지 못했다 — SERVE 블록에 먼저 추가할 것")
    p.write_text(pat.sub(lambda _: literal, src), encoding="utf-8")
    print(f"스플라이스 완료: {p} ({len(literal):,} bytes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--k", default="moment")
    ap.add_argument("--rho", type=float, default=1.0)
    args = ap.parse_args()
    K = args.k if args.k == "moment" else int(args.k)

    print(">> train.csv 로드")
    df = rd.load_train()
    if args.scan:
        scan(df)
    if args.emit:
        table = build_z_asof(df, K=K, rho=args.rho)
        print(f"   테이블 {len(table):,} 엔트리 (K={K}, ρ={args.rho})")
        splice(emit(table, K, args.rho))


if __name__ == "__main__":
    main()
