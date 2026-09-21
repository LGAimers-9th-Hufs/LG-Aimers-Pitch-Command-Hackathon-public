# -*- coding: utf-8 -*-
"""당해 시즌 성적 분리 — Program C가 특정한 신호를 **합법적으로** 취하는 시도.

    python sweep/inseason.py --vals 2024,2023,2022

## 어디서 나온 발상인가

`ceiling_protocol.py`(Program C) 결과:

| 용량 | P-A 랜덤KFold | P-B 투수그룹 | P-D 전이 |
|---|---|---|---|
| lv7/min1500 | 935.2 | **691.4** | **697.5** |
| lv31/min500 | **1185.9** | 523.8 | 579.8 |

**같은 시즌 데이터를 80% 가져도(P-B) 직전 시즌만 쓰는 것(P-D)보다 낫지 않다.**
⇒ P-A의 +240~600은 전부 **"그 투수의 같은 시즌 다른 투구 라벨"**이었다.
"남은 412점 = 시즌 전이 비용"은 존재하지 않는다.

**하지만 그 신호가 무엇인지는 특정됐다 — 투수 자신의 당해 시즌 성적.**
그리고 2025 test에서 그것의 **일부는 합법적으로 얻을 수 있다.**

## 구성 (전부 §5 적법)

`asof_pitcher_n`은 **커리어 누적**이고(A-2 검증: 투수 내 100% 단조 +1), `asof_pitcher_success_rate`는
그 시점까지의 커리어 성공률이다. 따라서 한 행에서 커리어 성공 수 `k = rate·n`을 정확히 복원할 수 있다.
여기서 **train으로 만든 "직전 시즌 말 누적" 룩업 `(n_base, k_base)`를 빼면 당해 시즌분만 남는다**:

    is_n = n − n_base          (그 투수의 이번 시즌 투구 수)
    is_k = k − k_base          (이번 시즌 성공 수)
    is_sm = (is_k + K/2)/(is_n + K)      ← 행 단위 수축 (june의 p_sm500과 같은 형태)
    is_delta = is_sm − p_sm500           ← **당해 폼 − 커리어**, 이게 본체다

- **행 단위 + 동봉 룩업**이므로 test 행 간 참조가 없다(§5 준수).
- **`pitcher_id`는 피처에서 빠져 있으므로**(june 레시피의 out-of-support 회피) 트리는 이 값을
  원리적으로 유도할 수 없다. 커리어 누적 rate는 이력에 희석돼 당해 신호를 못 드러낸다.
- 룩업에 없는 투수(신인) → base 0 ⇒ `is_n = 커리어 n`. **정의상 옳다**(그 투수에겐 커리어가 곧 당해).

## z_asof(−192)와 무엇이 다른가

z_asof는 train에서 만든 **레벨 룩업**이라 시즌이 갈수록 누적 분포가 이동해 out-of-support가 됐다.
여기서는 **학습과 서빙에서 같은 방식으로 구성**한다 — 시즌 s 행의 base는 항상 s−1 말이다.
따라서 `is_n`은 매 시즌 0에서 시작해 같은 범위를 돈다. **분포 이동이 구조적으로 없다.**

## 게이트

3폴드(2024/2023/2022) + **위약 대조**(base 룩업을 투수 사이에 셔플) + 폴드 간 SD 규칙(D-25).
컬럼 비용을 재기 위해 4컬럼본(`is4`)과 1컬럼본(`is1` = is_delta만)을 함께 잰다.
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
K_SHRINK = 200.0
COLS4 = ["is_logn", "is_sm", "is_delta", "is_share"]


def career_end_lookup(df: pd.DataFrame) -> dict:
    """{(pitcher_id, season): (n_end, k_end)} — 그 투수의 **시즌 s 말** 커리어 누적.

    시즌 s 안에서 `asof_pitcher_n`이 최대인 행을 잡으면 그게 마지막 투구 직전의 누적이므로,
    거기에 그 투구 자신(라벨 포함)을 더하면 시즌 말 누적이 된다.
    """
    n = df["asof_pitcher_n"].to_numpy(dtype=float)
    rate = df["asof_pitcher_success_rate"].fillna(0.0).to_numpy(dtype=float)
    d = pd.DataFrame({"pid": df["pitcher_id"].to_numpy(), "s": df[rd.SEASON].to_numpy(),
                      "n": n, "k": np.rint(rate * n), "y": df[rd.TARGET].to_numpy(dtype=float)})
    last = d.loc[d.groupby(["pid", "s"])["n"].idxmax()]
    return {(int(p), int(s)): (float(nn) + 1.0, float(kk) + float(yy))
            for p, s, nn, kk, yy in zip(last.pid, last.s, last.n, last.k, last.y)}


def base_for(lut: dict, pid: np.ndarray, season: np.ndarray, shuffle_seed=None):
    """시즌 s 행의 base = **s 미만에서 가장 최근 시즌 말** 누적. 없으면 (0, 0)."""
    # (pid, s) → 그 투수가 기록을 가진 시즌들
    by_pid = {}
    for (p, s), v in lut.items():
        by_pid.setdefault(p, {})[s] = v
    if shuffle_seed is not None:                      # 위약: 투수↔이력 대응만 파괴
        rng = np.random.default_rng(shuffle_seed)
        keys = list(by_pid)
        vals = [by_pid[k] for k in keys]
        by_pid = {k: vals[j] for k, j in zip(keys, rng.permutation(len(vals)))}

    nb = np.zeros(len(pid))
    kb = np.zeros(len(pid))
    cache = {}
    for i, (p, s) in enumerate(zip(pid.astype(int), season.astype(int))):
        key = (p, s)
        v = cache.get(key)
        if v is None:
            hist = by_pid.get(p)
            v = (0.0, 0.0)
            if hist:
                prev = [q for q in hist if q < s]
                if prev:
                    v = hist[max(prev)]
            cache[key] = v
        nb[i], kb[i] = v
    return nb, kb


def build_inseason(df: pd.DataFrame, nb, kb) -> pd.DataFrame:
    n = df["asof_pitcher_n"].to_numpy(dtype=float)
    rate = df["asof_pitcher_success_rate"].fillna(0.0).to_numpy(dtype=float)
    k = np.rint(rate * n)
    is_n = np.maximum(n - nb, 0.0)
    is_k = np.clip(k - kb, 0.0, is_n)
    is_sm = (is_k + K_SHRINK / 2.0) / (is_n + K_SHRINK)
    p_sm500 = (rate * n + 250.0) / (n + 500.0)
    return pd.DataFrame({
        "is_logn": np.log1p(is_n).astype("float32"),
        "is_sm": is_sm.astype("float32"),
        "is_delta": (is_sm - p_sm500).astype("float32"),
        "is_share": (is_n / np.maximum(n, 1.0)).astype("float32"),
    })


def gate(df, val_season, arms, lut):
    X = L.build_features(df).reset_index(drop=True)
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()
    pid = df["pitcher_id"].to_numpy()
    fit = (season == val_season - 1)
    val = (season == val_season)
    yv = y[val]
    print(f"\n[val {val_season}]  fit {fit.sum():,} → val {val.sum():,}  r={yv.mean():.4f}")

    cache = {}
    res, base_p = [], None
    for name, cols, seed in arms:
        t = time.time()
        if cols is None:
            XA = X
        else:
            if seed not in cache:
                nb, kb = base_for(lut, pid, season, shuffle_seed=seed)
                cache[seed] = build_inseason(df, nb, kb)
                a = cache[seed]
                m = val
                print(f"    [base seed={seed}] val 커버(is_n<커리어) "
                      f"{float((a['is_share'].to_numpy()[m] < 0.999).mean()):.3f} · "
                      f"is_delta 평균 {a['is_delta'].to_numpy()[m].mean():+.4f} "
                      f"sd {a['is_delta'].to_numpy()[m].std():.4f}")
            XA = pd.concat([X, cache[seed][cols]], axis=1)
        bs = fit_pair(XA[fit], y[fit], None)
        p = np.clip(predict_pair(bs, XA[val]), 0, 1)
        if base_p is None:
            base_p = p
        pc, bpc = L.calibrate(p), L.calibrate(base_p)
        rec = dict(val=val_season, arm=name, ncol=XA.shape[1], fixed=L.score(yv, pc),
                   best=best_cal(yv, p)[0], d=L.score(yv, pc) - L.score(yv, bpc),
                   se=paired_se(yv, pc, bpc), std=float(p.std()),
                   rms=float(np.sqrt(((p - base_p) ** 2).mean())), sec=round(time.time() - t, 1))
        res.append(rec)
        print(f"  {name:12s} ({XA.shape[1]:2d}col)  고정 {rec['fixed']:8.2f}  "
              f"재적합 {rec['best']:8.2f}  Δ {rec['d']:+7.2f}±{rec['se']:.1f}  "
              f"rms {rec['rms']:.4f}  [{rec['sec']}s]")
    return res


ARMS = [
    ("base",     None,               None),
    ("is4",      COLS4,              None),
    ("is1",      ["is_delta"],       None),
    ("is2",      ["is_delta", "is_logn"], None),
    ("is4_shuf", COLS4,              7),      # 위약: 투수↔이력 대응 파괴
]


def ens_gate(df, lut, vals):
    """**결정적 시험** — june 2종이 아니라 챔피언 ENS-2 위에서도 더해지는가.

    ENS-2 멤버 5종(june_l15 / nn_lin / et_l100_d28 / et_l200_d20 / june_l7)을 53피처와 57피처로
    각각 학습해 같은 가중·같은 캘리로 블렌드한 뒤 직접 비교한다. 멤버를 바꾸지 않고 **입력만** 바꾼다.
    """
    import phase2_ens_check as EC
    X = L.build_features(df).reset_index(drop=True)
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()
    pid = df["pitcher_id"].to_numpy()
    nb, kb = base_for(lut, pid, season)
    A = build_inseason(df, nb, kb)
    XI = pd.concat([X, A[COLS4]], axis=1)

    out = []
    for v in vals:
        fit, val = (season == v - 1), (season == v)
        yv = y[val]
        print(f"\n[val {v}] ENS-2 대조  fit {fit.sum():,} → val {val.sum():,}")
        row = {"val": v}
        for tag, XX in (("53피처(현 ENS-2)", X), ("57피처(+is4)", XI)):
            P = EC.train_members(XX, y, fit, val)
            ens = sum(P[n] * w for n, _, w, _ in EC.MEMBERS)
            fx = L.score(yv, L.calibrate(ens, **EC.ENS2_CAL))
            bc, arg = best_cal(yv, ens)
            print(f"    {tag:16s} 고정캘리 {fx:8.2f} · 캘리재적합 {bc:8.2f} {arg}")
            row[tag] = dict(fixed=fx, best=bc, cal=str(arg))
        d = row["57피처(+is4)"]["best"] - row["53피처(현 ENS-2)"]["best"]
        row["delta_best"] = d
        print(f"    ⇒ Δ(캘리재적합) {d:+.2f}")
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024,2023,2022")
    ap.add_argument("--ens", action="store_true",
                    help="ENS-2 5멤버를 53 vs 57피처로 학습해 직접 대조 (결정적 시험)")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(">> train.csv 로드")
    df = rd.load_train()
    print(">> 시즌 말 커리어 누적 룩업 생성")
    lut = career_end_lookup(df)
    print(f"   {len(lut):,} (pid, season) 엔트리")

    if args.ens:
        res = ens_gate(df, lut, [int(s) for s in args.vals.split(",")])
        (OUT_DIR / "ens_gate.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
        print(f"\n>> 저장 {OUT_DIR/'ens_gate.json'}  ({time.time()-t0:.0f}s)")
        return

    allres = []
    for v in [int(s) for s in args.vals.split(",")]:
        allres += gate(df, v, ARMS, lut)

    t = pd.DataFrame(allres)
    print("\n" + "=" * 100)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n[3폴드 요약 — D-25 규칙: 평균 > 0 **및** 평균 > 폴드 간 SD]")
    g = t[t.arm != "base"].groupby("arm").agg(
        d_mean=("d", "mean"), d_sd=("d", "std"), d_min=("d", "min"),
        pos=("d", lambda s: int((s > 0).sum())), n=("d", "size"))
    g["통과"] = (g.d_mean > 0) & (g.d_mean > g.d_sd)
    print(g.sort_values("d_mean", ascending=False).to_string(float_format=lambda v: f"{v:.2f}"))
    print("\n  ⚠ 위약(is4_shuf)이 실arm과 구별되지 않으면 채널이 없는 것이다.")

    (OUT_DIR / "gate.json").write_text(json.dumps(allres, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
