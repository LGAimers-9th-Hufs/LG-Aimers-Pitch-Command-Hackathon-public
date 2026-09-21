# -*- coding: utf-8 -*-
"""당해 시즌 분해를 **모든 누적 통계**로 확장 — 투수 5률 + 타자 2률 + 구종믹스 3률.

    python sweep/inseason_full.py --vals 2024,2023,2022

`inseason.py`가 `asof_pitcher_success_rate` 하나에서 3폴드 +123(위약 −4.9)을 냈다.
주최 `asof_*` 19컬럼 중 **커리어 누적인 것은 전부 같은 분해가 가능하다**:

| 엔티티 | 카운터 | 비율 |
|---|---|---|
| 투수 | `asof_pitcher_n` | success · reverse · middle · ball · strike |
| 타자 | `asof_batter_n` | success · middle |
| 구종믹스 | `asof_pitcher_pitchmix_n` | fastball · breaking · offspeed |

각 비율 r에 대해 커리어 성공수 `k = r·n`을 복원하고, train으로 만든 **직전 시즌 말 누적**을 빼면
당해 시즌분이 남는다. 핵심 피처는 항상 **`당해 수축률 − 커리어 수축률`(= 당해 폼의 이탈)**이다.

⚠ 컬럼명은 전부 `is` 접두 — june 피처셋에 이미 `p_logn`/`b_logn`이 있어 충돌한다.
⚠ 컬럼 비용(D-25: 6컬럼 추가만으로 폴드 SD 13)을 감안해 **계단식**으로 잰다:
`is4`(현재, 4) → `p7`(투수 전 5률) → `pb`(+타자) → `all`(+구종믹스).
컬럼이 늘어 이득이 줄면 거기서 멈춘다.
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
from season_centering import best_cal, paired_se, fit_pair, predict_pair   # noqa: E402

OUT_DIR = HERE.parent / "results" / "inseason"
K_P, K_B, K_M = 200.0, 100.0, 200.0     # 엔티티별 수축 상수 (타자는 표본이 작아 더 세게)

# (블록, 엔티티 키, 카운터, [(비율컬럼, 짧은 이름)], 수축상수)
BLOCKS = [
    ("p", "pitcher_id", "asof_pitcher_n", [
        ("asof_pitcher_success_rate", "succ"), ("asof_pitcher_reverse_rate", "rev"),
        ("asof_pitcher_middle_rate", "mid"), ("asof_pitcher_ball_rate", "ball"),
        ("asof_pitcher_strike_rate", "strk")], K_P),
    ("b", "batter_id", "asof_batter_n", [
        ("asof_batter_success_rate", "succ"), ("asof_batter_middle_rate", "mid")], K_B),
    ("m", "pitcher_id", "asof_pitcher_pitchmix_n", [
        ("asof_pitcher_fastball_rate", "fb"), ("asof_pitcher_breaking_rate", "br"),
        ("asof_pitcher_offspeed_rate", "os")], K_M),
]


def end_lookup(df, ent, ncol, rate_cols):
    """{(entity, season): (n_end, [k_end...])} — 시즌 s 말 누적.

    시즌 내 카운터 최대 행이 마지막 투구 직전이므로, 거기에 그 투구 자신을 더한다.
    ⚠ 라벨(y)은 **success 계열에만** 더할 수 있다(다른 실패유형은 라벨에 없다).
    나머지 비율은 마지막 한 구의 기여(1/n)를 무시한다 — n이 수백~수천이라 오차는 무시 가능.
    """
    n = df[ncol].to_numpy(dtype=float)
    d = pd.DataFrame({"e": df[ent].to_numpy(), "s": df[rd.SEASON].to_numpy(), "n": n,
                      "y": df[rd.TARGET].to_numpy(dtype=float)})
    for c, nm in rate_cols:
        d["k_" + nm] = np.rint(df[c].fillna(0.0).to_numpy(dtype=float) * n)
    last = d.loc[d.groupby(["e", "s"])["n"].idxmax()]
    out = {}
    for row in last.itertuples(index=False):
        ks = []
        for c, nm in rate_cols:
            k = getattr(row, "k_" + nm)
            ks.append(k + (row.y if nm == "succ" else 0.0))
        out[(int(row.e), int(row.s))] = (float(row.n) + 1.0, ks)
    return out


def base_arrays(lut, ent_vals, season, nrates):
    by = {}
    for (e, s), v in lut.items():
        by.setdefault(e, {})[s] = v
    nb = np.zeros(len(season))
    kb = np.zeros((len(season), nrates))
    cache = {}
    for i, (e, s) in enumerate(zip(ent_vals.astype(int), season.astype(int))):
        v = cache.get((e, s))
        if v is None:
            hist = by.get(e)
            v = (0.0, [0.0] * nrates)
            if hist:
                prev = [q for q in hist if q < s]
                if prev:
                    v = hist[max(prev)]
            cache[(e, s)] = v
        nb[i] = v[0]
        kb[i] = v[1]
    return nb, kb


def build_block(df, tag, ent, ncol, rate_cols, K, lut):
    n = df[ncol].to_numpy(dtype=float)
    nb, kb = base_arrays(lut, df[ent].to_numpy(), df[rd.SEASON].to_numpy(), len(rate_cols))
    is_n = np.maximum(n - nb, 0.0)
    cols = {f"is{tag}_logn": np.log1p(is_n).astype("float32"),
            f"is{tag}_share": (is_n / np.maximum(n, 1.0)).astype("float32")}
    for j, (c, nm) in enumerate(rate_cols):
        r = df[c].fillna(0.0).to_numpy(dtype=float)
        k = np.rint(r * n)
        is_k = np.clip(k - kb[:, j], 0.0, is_n)
        is_sm = (is_k + K * 0.5) / (is_n + K)
        car_sm = (k + K * 0.5) / (n + K)
        # ⚠ 레벨(_sm)과 이탈(_d)을 **둘 다** 낸다. 초판은 _d만 냈고, 그래서 4컬럼 검증본(+92.5)이
        #   3컬럼(+43.3)으로 반토막 나 확장 비교가 잘못된 기준 위에서 이뤄졌다.
        cols[f"is{tag}_{nm}_sm"] = is_sm.astype("float32")
        cols[f"is{tag}_{nm}_d"] = (is_sm - car_sm).astype("float32")
    return pd.DataFrame(cols)


ARMS = [
    ("base",  []),
    ("is4",   ["isp_logn", "isp_share", "isp_succ_d"]),     # inseason.py 대응(수축상수만 통일)
    ("p7",    None),                                        # 투수 블록 전체
    ("pb",    None),                                        # + 타자
    ("all",   None),                                        # + 구종믹스
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024,2023,2022")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(">> train.csv 로드")
    df = rd.load_train()
    blocks = {}
    for tag, ent, ncol, rc, K in BLOCKS:
        t = time.time()
        lut = end_lookup(df, ent, ncol, rc)
        blocks[tag] = build_block(df, tag, ent, ncol, rc, K, lut)
        print(f"   블록 {tag}: 룩업 {len(lut):,} · 컬럼 {blocks[tag].shape[1]} [{time.time()-t:.0f}s]")

    ALL = pd.concat([blocks["p"], blocks["b"], blocks["m"]], axis=1)
    # 위약: 타자 블록만 타자 사이에서 뒤섞는다(주변분포·컬럼 수 동일, 대응만 파괴)
    rng = np.random.default_rng(11)
    SH = blocks["b"].iloc[rng.permutation(len(blocks["b"]))].reset_index(drop=True)
    SH.columns = [c + "_shuf" for c in SH.columns]
    ALL = pd.concat([ALL, SH], axis=1)

    IS4 = ["isp_logn", "isp_share", "isp_succ_sm", "isp_succ_d"]      # ← LB 953.06의 그 구성
    BAT = ["isb_logn", "isb_share", "isb_succ_sm", "isb_succ_d", "isb_mid_sm", "isb_mid_d"]
    PRT = [c for c in blocks["p"].columns if c not in IS4]            # 나머지 실패유형률
    MIX = list(blocks["m"].columns)
    arms = [("base", []), ("is4", IS4),
            ("is4+bat", IS4 + BAT),
            ("is4+prate", IS4 + PRT),
            ("is4+bat+prate", IS4 + BAT + PRT),
            ("is4+mix", IS4 + MIX),
            ("is4+bat_shuf", IS4 + [c + "_shuf" for c in BAT])]        # 위약

    X = L.build_features(df).reset_index(drop=True)
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()

    res = []
    for v in [int(s) for s in args.vals.split(",")]:
        fit, val = (season == v - 1), (season == v)
        yv = y[val]
        print(f"\n[val {v}]  fit {fit.sum():,} → val {val.sum():,}  r={yv.mean():.4f}")
        base_p = None
        for name, cols in arms:
            t = time.time()
            XA = X if not cols else pd.concat([X, ALL[cols]], axis=1)
            p = np.clip(predict_pair(fit_pair(XA[fit], y[fit], None), XA[val]), 0, 1)
            if base_p is None:
                base_p = p
            pc, bpc = L.calibrate(p), L.calibrate(base_p)
            rec = dict(val=v, arm=name, ncol=XA.shape[1], fixed=L.score(yv, pc),
                       best=best_cal(yv, p)[0], d=L.score(yv, pc) - L.score(yv, bpc),
                       se=paired_se(yv, pc, bpc), sec=round(time.time() - t, 1))
            res.append(rec)
            print(f"  {name:6s} ({XA.shape[1]:2d}col)  고정 {rec['fixed']:8.2f}  "
                  f"재적합 {rec['best']:8.2f}  Δ {rec['d']:+7.2f}±{rec['se']:.1f}  [{rec['sec']}s]")

    t = pd.DataFrame(res)
    print("\n" + "=" * 92)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\n[폴드 요약 — D-25 규칙]")
    g = t[t.arm != "base"].groupby("arm").agg(
        ncol=("ncol", "max"), d_mean=("d", "mean"), d_sd=("d", "std"), d_min=("d", "min"),
        pos=("d", lambda s: int((s > 0).sum())), n=("d", "size"))
    g["통과"] = (g.d_mean > 0) & (g.d_mean > g.d_sd)
    print(g.sort_values("d_mean", ascending=False).to_string(float_format=lambda v: f"{v:.2f}"))

    (OUT_DIR / "full_gate.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR/'full_gate.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
