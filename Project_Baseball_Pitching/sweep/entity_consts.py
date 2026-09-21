# -*- coding: utf-8 -*-
"""Program E — **트리가 만들 수 없는 per-entity 상수** 계열. `is4` 원리의 확장.

    python sweep/entity_consts.py --vals 2024,2023,2022        # june 2종 기준 3폴드 + 위약
    python sweep/entity_consts.py --ens --vals 2024,2023        # ENS-4 기준선 (채택 판단용)

## 가설

`is4`(당해 시즌 성적 분리)가 로컬 +88.6 / LB +41.9를 낸 이유는 새 모델도 새 데이터도 아니라
**"주최가 준 정보인데 트리가 원리적으로 만들 수 없는 값"**이었기 때문이다 —
`asof_*`는 커리어 누적이고, 당해분을 분리하려면 **투수별 상수**(직전 시즌 말 누적)를 빼야 하는데
june 레시피가 `pitcher_id`를 피처에서 제거했으므로 트리는 그걸 할 수 없다.

**반증 근거**: `prevdelta`(prev5 − 커리어, **이미 컬럼으로 있는 값들의 차분**)는 3폴드 −5.1로 0이었다.
⇒ **차분이 힘이 아니라 "접근 불가능한 baseline"이 힘이다.**

⇒ `pitcher_id`·`batter_id`·`season`이 전부 제거돼 있으므로 **train에서 산출되는 모든 per-entity 상수**가
같은 성격의 미개척 채널이다. 이 모듈이 그걸 소진한다.

## E1 — 직전 **완결** 시즌 rate (1순위)

```
prev_n = n_end(s−1) − n_end(s−2)      prev_k = k_end(s−1) − k_end(s−2)
ps_sm    = (prev_k + K/2)/(prev_n + K)     직전 시즌 단독 성적
ps_delta = ps_sm − p_sm500                  직전 시즌 − 커리어
ps_logn  = log1p(prev_n)                    그 추정의 신뢰도
```

**왜 이게 클 수 있나**: 이 대회 최대 레버가 **"학습은 최신 1시즌만"(+530)** 이었다 —
**타깃에서 recency가 지배적**이라는 뜻이다. 그렇다면 피처에서도 그래야 하는데, 트리가 보는
`asof_pitcher_success_rate`는 **6시즌이 뒤섞인 커리어 평균**이다. 직전 시즌만 떼어내려면
`n_end(s−2)`가 필요하고 그건 per-pitcher 상수다.

**실측 커버리지**(2026-08-07): 2024 행의 **70%**가 직전 2시즌 보유 · 직전시즌 rate의 투수 간
**sd 0.058**(모델 전체 예측 산포 0.048보다 크다).

## E2 — 시즌 간 추세 · E3 — 장기 변동성 · E4 — 타자측

- **E2** `ps_trend = ps_sm(s−1) − ps_sm(s−2)`. 노쇠/성장 방향. per-entity 상수 3개가 필요하다.
- **E3** 투수별 **시즌 단위 성적의 장기 산포**. `prev1/3/5`는 최근 3점만 주고 장기 산포는 못 준다.
- **E4** 타자측 당해시즌 블록(`is4`와 같은 형태, `batter_id` 키).

## 게이트

3폴드(2024/2023/2022) + **위약**(투수↔이력 대응 셔플) + D-25 규칙(평균>0 **및** 평균>폴드간 SD).
`--ens`로 **ENS 기준선에서 재측정**한다 — june 2종 기준 수치는 정제형에서 4배까지 과대평가한다
(K=100: june +13.0 → ENS +3.2).
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import real_data as rd            # noqa: E402
import lgbm_family as L           # noqa: E402
import inseason as IS             # noqa: E402
from season_centering import best_cal, paired_se, fit_pair, predict_pair   # noqa: E402

OUT_DIR = HERE.parent / "results" / "entity"
K_PS = 200.0          # 직전 시즌 수축 (is4와 같은 값에서 출발)
K_BAT = 100.0

E1 = ["ps_logn", "ps_sm", "ps_delta"]
E2 = ["ps_trend"]
E3 = ["ps_vol"]
E4 = ["bis_logn", "bis_share", "bis_sm", "bis_delta"]


def season_history(df, ent_col, ncol, rate_col):
    """{entity: {season: (n_end, k_end)}} — 시즌 말 누적. `inseason.career_end_lookup`과 같은 산식."""
    n = df[ncol].to_numpy(dtype=float)
    rate = df[rate_col].fillna(0.0).to_numpy(dtype=float)
    d = pd.DataFrame({"e": df[ent_col].to_numpy(), "s": df[rd.SEASON].to_numpy(), "n": n,
                      "k": np.rint(rate * n), "y": df[rd.TARGET].to_numpy(dtype=float)})
    last = d.loc[d.groupby(["e", "s"])["n"].idxmax()]
    out = {}
    for r in last.itertuples(index=False):
        out.setdefault(int(r.e), {})[int(r.s)] = (float(r.n) + 1.0, float(r.k) + float(r.y))
    return out


def build_entity_cols(df, hist, shuffle_seed=None):
    """시즌 s 행에 붙일 per-entity 상수. **s 미만 시즌만** 쓴다(as-of 준수)."""
    if shuffle_seed is not None:                     # 위약: 투수↔이력 대응만 파괴
        rng = np.random.default_rng(shuffle_seed)
        keys = list(hist)
        vals = [hist[k] for k in keys]
        hist = {k: vals[j] for k, j in zip(keys, rng.permutation(len(vals)))}

    ent = df["pitcher_id"].to_numpy().astype(int)
    ssn = df[rd.SEASON].to_numpy().astype(int)
    n = df["asof_pitcher_n"].to_numpy(dtype=float)
    rate = df["asof_pitcher_success_rate"].fillna(0.0).to_numpy(dtype=float)
    car_sm = (rate * n + 250.0) / (n + 500.0)

    cache, m = {}, len(df)
    ps_sm = np.full(m, np.nan)
    ps_logn = np.full(m, np.nan)
    ps_trend = np.full(m, np.nan)
    ps_vol = np.full(m, np.nan)
    for i in range(m):
        key = (ent[i], ssn[i])
        v = cache.get(key)
        if v is None:
            h = hist.get(ent[i], {})
            prev = sorted(q for q in h if q < ssn[i])
            a = b = c = d_ = np.nan
            if len(prev) >= 2:                        # 직전 완결 시즌 = prev[-1] − prev[-2]
                n1, k1 = h[prev[-1]]
                n0, k0 = h[prev[-2]]
                dn, dk = n1 - n0, k1 - k0
                if dn >= 1:
                    a = (dk + K_PS * 0.5) / (dn + K_PS)
                    b = float(np.log1p(dn))
                if len(prev) >= 3:                    # 추세 = 직전 − 그 전
                    n2, k2 = h[prev[-3]]
                    dn2, dk2 = n0 - n2, k0 - k2
                    if dn >= 1 and dn2 >= 1:
                        c = a - (dk2 + K_PS * 0.5) / (dn2 + K_PS)
                # 장기 변동성 = 시즌 단독 rate들의 표준편차
                rs = []
                for j in range(1, len(prev)):
                    na, ka = h[prev[j]]
                    nb_, kb_ = h[prev[j - 1]]
                    if na - nb_ >= 50:
                        rs.append((ka - kb_) / (na - nb_))
                if len(rs) >= 2:
                    d_ = float(np.std(rs, ddof=1))
            v = (a, b, c, d_)
            cache[key] = v
        ps_sm[i], ps_logn[i], ps_trend[i], ps_vol[i] = v

    return pd.DataFrame({
        "ps_logn": ps_logn.astype("float32"),
        "ps_sm": ps_sm.astype("float32"),
        "ps_delta": (ps_sm - car_sm).astype("float32"),
        "ps_trend": ps_trend.astype("float32"),
        "ps_vol": ps_vol.astype("float32")})


def build_batter_inseason(df, bhist):
    """E4 — 타자측 당해 시즌 분리(`is4`와 동일 구조, batter_id 키)."""
    ent = df["batter_id"].to_numpy().astype(int)
    ssn = df[rd.SEASON].to_numpy().astype(int)
    n = df["asof_batter_n"].to_numpy(dtype=float)
    rate = df["asof_batter_success_rate"].fillna(0.0).to_numpy(dtype=float)
    k = np.rint(rate * n)
    nb = np.zeros(len(df))
    kb = np.zeros(len(df))
    cache = {}
    for i in range(len(df)):
        key = (ent[i], ssn[i])
        v = cache.get(key)
        if v is None:
            h = bhist.get(ent[i], {})
            prev = [q for q in h if q < ssn[i]]
            v = h[max(prev)] if prev else (0.0, 0.0)
            cache[key] = v
        nb[i], kb[i] = v
    is_n = np.maximum(n - nb, 0.0)
    is_k = np.clip(k - kb, 0.0, is_n)
    is_sm = (is_k + K_BAT * 0.5) / (is_n + K_BAT)
    car_sm = (rate * n + K_BAT * 2.5) / (n + K_BAT * 5)
    return pd.DataFrame({
        "bis_logn": np.log1p(is_n).astype("float32"),
        "bis_share": (is_n / np.maximum(n, 1.0)).astype("float32"),
        "bis_sm": is_sm.astype("float32"),
        "bis_delta": (is_sm - car_sm).astype("float32")})


ARMS = [("base", []), ("E1", E1), ("E1+E2", E1 + E2), ("E1+E3", E1 + E3),
        ("E1+E2+E3", E1 + E2 + E3), ("E4(타자)", E4), ("E1+E4", E1 + E4),
        ("E1_shuf(위약)", ["sh_" + c for c in E1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024,2023,2022")
    ap.add_argument("--k", type=float, default=100.0, help="is4의 K (ENS-4는 100)")
    ap.add_argument("--ens", action="store_true", help="ENS-4 기준선에서 재측정")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f">> train.csv 로드 (is4 K={args.k:g})")
    L.K_IS = args.k
    df = rd.load_train()
    lut = IS.career_end_lookup(df)
    nb0, kb0 = IS.base_for(lut, df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy())
    X = L.build_features(df, base=(nb0, kb0)).reset_index(drop=True)   # 57피처 = ENS-4 기준선
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()

    t = time.time()
    ph = season_history(df, "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate")
    bh = season_history(df, "batter_id", "asof_batter_n", "asof_batter_success_rate")
    E = build_entity_cols(df, ph)
    SH = build_entity_cols(df, ph, shuffle_seed=13)[E1].add_prefix("sh_")
    B = build_batter_inseason(df, bh)
    ALL = pd.concat([E, SH, B], axis=1)
    print(f"   per-entity 상수 생성 [{time.time()-t:.0f}s] · 투수 {len(ph)} · 타자 {len(bh)}")
    for c in E1 + E2 + E3 + E4:
        v = ALL[c].to_numpy()
        ok = ~np.isnan(v)
        print(f"     {c:10s} 결측 {1-ok.mean():5.1%} · 평균 {np.nanmean(v):+.4f} "
              f"sd {np.nanstd(v):.4f} · q05 {np.nanquantile(v,.05):+.3f} q95 {np.nanquantile(v,.95):+.3f}")

    res = []
    for v in [int(s) for s in args.vals.split(",")]:
        fit, val = (season == v - 1), (season == v)
        yv = y[val]
        cov = float((~np.isnan(ALL["ps_sm"].to_numpy()[val])).mean())
        print(f"\n[val {v}]  fit {fit.sum():,} → val {val.sum():,}  ps 커버 {cov:.1%}")
        base_p = None
        for name, cols in ARMS:
            t = time.time()
            XA = X if not cols else pd.concat([X, ALL[cols]], axis=1)
            if args.ens:
                import phase2_ens_check as EC
                P = EC.train_members(XA, y, fit, val)
                p = np.clip(sum(P[n_] * w for n_, _, w, _ in EC.MEMBERS), 0, 1)
            else:
                p = np.clip(predict_pair(fit_pair(XA[fit], y[fit], None), XA[val]), 0, 1)
            if base_p is None:
                base_p = p
            pc, bpc = L.calibrate(p), L.calibrate(base_p)
            rec = dict(val=v, arm=name, ncol=XA.shape[1], fixed=L.score(yv, pc),
                       best=best_cal(yv, p)[0], d=L.score(yv, pc) - L.score(yv, bpc),
                       se=paired_se(yv, pc, bpc), sec=round(time.time() - t, 1))
            res.append(rec)
            print(f"  {name:14s} ({XA.shape[1]:2d}col) 고정 {rec['fixed']:8.2f} "
                  f"재적합 {rec['best']:8.2f}  Δ {rec['d']:+7.2f}±{rec['se']:.1f} [{rec['sec']}s]")

    t = pd.DataFrame(res)
    print("\n" + "=" * 94)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\n[폴드 요약 — D-25 규칙: 평균 > 0 **및** 평균 > 폴드 간 SD]")
    g = t[t.arm != "base"].groupby("arm").agg(
        ncol=("ncol", "max"), d_mean=("d", "mean"), d_sd=("d", "std"), d_min=("d", "min"),
        pos=("d", lambda s: int((s > 0).sum())), n=("d", "size"))
    g["통과"] = (g.d_mean > 0) & (g.d_mean > g.d_sd)
    print(g.sort_values("d_mean", ascending=False).to_string(float_format=lambda v: f"{v:.2f}"))
    print("\n  ⚠ 위약(E1_shuf)이 E1과 구별되지 않으면 채널이 없는 것이다.")

    tag = "_ens" if args.ens else ""
    (OUT_DIR / f"gate{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
