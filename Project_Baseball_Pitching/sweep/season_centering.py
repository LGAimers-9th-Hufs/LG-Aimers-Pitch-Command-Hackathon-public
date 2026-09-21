# -*- coding: utf-8 -*-
"""Phase 2 — 시즌별 타깃 센터링. "남은 412점 = 시즌 전이 비용"에 직격하는 유일한 값싼 수단.

    python sweep/season_centering.py --val 2024          # 본 게이트
    python sweep/season_centering.py --val 2023          # Phase 5 교차검증(이득의 부호만 본다)

## 무엇을 재는가

june853의 최대 레버는 **학습을 최신 1시즌으로 자른 것**이었다(707.9 vs 전 시즌 319.0, +530).
그런데 그 비교는 **원시 풀링**만 본 것이다. 시즌 기저율이 단조 하락(2019 .5647 → 2024 .4861)하므로
전 시즌을 그냥 합치면 **레벨이 오염**된다. 레벨만 제거하면 253k → 1.2M(5배)를 쓸 수 있다.

  - `center` : y_c = y − r_s + r_ref  (r_s = 그 시즌 기저율, r_ref = fit 최신 시즌)
  - `scale`  : y_c = r_ref + (y − r_s)·√(V_ref/V_s), V = r(1−r)  — 레벨 + 산포 정규화
  - `decay`  : 시즌 recency 가중 (센터링과 직교, 조합 가능)
  - `ft`     : 전 시즌 센터링으로 pretrain → **최신 1시즌으로 continue boosting**(init_model)

**가설 분기**: 센터링이 이기면 "낡는 것은 레벨"이고 데이터가 5배로 늘어난다. 지면 "낡는 것은
조건부 구조"가 확정돼 Phase 3(트랙맨 = 새 정보원)의 근거가 된다. 어느 쪽이든 결론이 남는다.

## 판정

`sweep/lgbm_family.py`의 june l7+l15를 고정 레시피로 쓰고 fit 집합만 바꾼다(단일 변인).
- 절대 점수(고정 캘리 0.91/−0.01)와 **캘리 재적합 최고점**(레벨 운 제거)을 함께 본다.
- **paired SE**: 같은 val 행에서 제곱오차 차이의 표준오차 — 독립 2SE보다 훨씬 검정력이 높다.
  (D-17이 +11.4 < 2SE 18.6로 저검정력 판정났던 실패를 반복하지 않기 위한 장치.)
- 단독 성능이 져도 **기준과의 불일치 RMS**가 크면 앙상블 멤버 후보다([[june853-recipe]] 채택 기준).
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
import real_data as rd        # noqa: E402
import lgbm_family as L       # noqa: E402

OUT_DIR = HERE.parent / "results" / "phase2"
V = lambda r: r * (1.0 - r)


# ---------------------------------------------------------------- 채점 도구
def best_cal(y, p):
    """캘리 재적합 후 최고점 — 채널 비교를 레벨 운에서 분리한다(oracle_ceiling과 동일 격자)."""
    best, arg = -1e18, None
    for s in np.arange(0.60, 1.61, 0.01):
        q = 0.5 + s * (np.asarray(p) - 0.5)
        for h in np.arange(-0.04, 0.041, 0.0025):
            v = L.score(y, np.clip(q + h, 1e-6, 1 - 1e-6))
            if v > best:
                best, arg = v, (round(float(s), 3), round(float(h), 4))
    return best, arg


def paired_se(y, pa, pb):
    """score(a) − score(b)의 **짝지은** 표준오차. 같은 val 행이므로 r·V는 공통 상수다."""
    y = np.asarray(y, dtype=float)
    d = (np.asarray(pb) - y) ** 2 - (np.asarray(pa) - y) ** 2      # b의 오차 − a의 오차
    return 1e5 / V(y.mean()) * float(np.std(d, ddof=1)) / np.sqrt(len(d))


# ---------------------------------------------------------------- 타깃 변환
def make_target(y, season, fit_mask, mode, ref_season):
    """fit 행에만 적용되는 타깃 변환. 통계는 **fit 집합에서만** 계산한다(val 라벨 미사용)."""
    yf = y[fit_mask]
    if mode == "raw":
        return yf
    sf = season[fit_mask]
    means = {int(s): float(yf[sf == s].mean()) for s in np.unique(sf)}
    r_ref = means[int(ref_season)]
    r_s = np.array([means[int(s)] for s in sf], dtype=float)
    if mode == "center":
        return yf - r_s + r_ref
    if mode == "scale":
        sc = np.sqrt(V(r_ref) / np.array([V(m) for m in r_s]))
        return r_ref + (yf - r_s) * sc
    raise ValueError(mode)


def make_weight(season, fit_mask, decay, ref_season):
    if decay is None:
        return None
    sf = season[fit_mask]
    return np.power(float(decay), (int(ref_season) - sf).astype(float))


# ---------------------------------------------------------------- 학습
def fit_pair(X, yt, w, specs=None, init=None, rounds=None):
    """june 원본 2종(l7·l15)을 같은 타깃으로 학습해 booster dict 반환."""
    import lightgbm as lgb
    specs = specs or L.ORIGINAL
    out = {}
    for name, sp in specs.items():
        params = {**L.BASE_PARAMS, **{k: v for k, v in sp.items() if k != "num_iterations"}}
        ds = lgb.Dataset(X, label=np.asarray(yt, dtype=float), weight=w, free_raw_data=False)
        out[name] = lgb.train(params, ds, num_boost_round=rounds or sp["num_iterations"],
                              init_model=None if init is None else init[name])
    return out


def predict_pair(boosters, Xv):
    return np.mean([b.predict(Xv, num_threads=4) for b in boosters.values()], axis=0)


# ---------------------------------------------------------------- 아
def arms_for(val_season, seasons):
    """fit 후보는 전부 val 미만 시즌. 'lastk'는 최근 k시즌."""
    prev = [s for s in seasons if s < val_season]
    ref = max(prev)
    A = []

    def add(name, k, mode, decay=None, ft=None):
        fs = prev if k is None else prev[-k:]
        A.append(dict(name=name, fit=fs, mode=mode, decay=decay, ft=ft, ref=ref))

    add("base_1s", 1, "raw")                       # ← june853 레시피 (기준)
    add("all_raw", None, "raw")                    # ← 알려진 붕괴 (로컬 319)
    add("all_center", None, "center")              # ★ 본 가설
    add("all_scale", None, "scale")
    add("all_center_d85", None, "center", 0.85)
    add("all_center_d70", None, "center", 0.70)
    add("all_center_d50", None, "center", 0.50)
    add("all_raw_d70", None, "raw", 0.70)          # 센터링 없이 감쇠만 (분해용)
    add("s2_center", 2, "center")
    add("s3_center", 3, "center")
    add("s4_center", 4, "center")
    add("s2_raw", 2, "raw")                        # 2시즌에서 센터링의 순효과
    # pretrain(전 시즌 센터링) → 최신 1시즌으로 continue boosting
    add("ft_all2last", None, "center", None, ft=dict(pre=300, post=200, k=1))
    add("ft_s3_2last", 3, "center", None, ft=dict(pre=300, post=200, k=1))
    return A


def run_arm(a, X, y, season, val, ref_pred=None):
    fit = np.isin(season, a["fit"])
    yt = make_target(y, season, fit, a["mode"], a["ref"])
    w = make_weight(season, fit, a["decay"], a["ref"])
    Xf = X[fit]
    if a["ft"] is None:
        bs = fit_pair(Xf, yt, w, rounds=None)
    else:
        pre = fit_pair(Xf, yt, w, rounds=a["ft"]["pre"])
        f2 = np.isin(season, a["fit"][-a["ft"]["k"]:])
        y2 = make_target(y, season, f2, "raw", a["ref"])
        bs = fit_pair(X[f2], y2, None, init=pre, rounds=a["ft"]["post"])
    p = np.clip(predict_pair(bs, X[val]), 0, 1)
    return p, int(fit.sum())


def two_fold_blend(vals=(2024, 2023), ws=(0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)):
    """저장된 예측으로 **두 폴드 동시** 블렌드 이득을 잰다 — 재학습 없음.

    폴드마다 최적 w를 따로 고르면 그건 선택 낙관이다. 여기서는 **w를 고정**하고 두 폴드에서 동시에
    재서, 최악 폴드의 이득이 가장 큰 arm·w(maximin)를 고른다. 2024와 2023은 레짐이 정반대이므로
    (1시즌 학습이 2024에선 +389, 2023에선 −749) 둘 다에서 양수인 것만 진짜 로버스트다.
    """
    Z = {v: np.load(OUT_DIR / f"preds_val{v}.npz") for v in vals}
    base = {v: best_cal(Z[v]["y"], Z[v]["base_1s"])[0] for v in vals}
    print("[기준] " + "  ".join(f"val{v} base_1s = {base[v]:.2f}" for v in vals))
    rows = []
    for arm in Z[vals[0]].files:
        if arm in ("y", "base_1s") or any(arm not in Z[v].files for v in vals):
            continue
        for w in ws:
            g = {v: best_cal(Z[v]["y"], (1 - w) * Z[v]["base_1s"] + w * Z[v][arm])[0] - base[v]
                 for v in vals}
            rows.append(dict(arm=arm, w=w, **{f"g{v}": g[v] for v in vals},
                             worst=min(g.values()), mean=float(np.mean(list(g.values())))))
    t = pd.DataFrame(rows)
    print("\n[고정 w 2폴드 동시 이득] worst = 최악 폴드 이득 (maximin 기준)")
    print(t.sort_values("worst", ascending=False).head(20)
          .to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\n[arm별 최선의 고정 w]")
    best = t.loc[t.groupby("arm")["worst"].idxmax()].sort_values("worst", ascending=False)
    print(best.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    (OUT_DIR / "twofold_blend.json").write_text(
        t.to_json(orient="records"), encoding="utf-8")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=int, default=2024)
    ap.add_argument("--tag", default="")
    ap.add_argument("--twofold", action="store_true",
                    help="저장된 예측으로 두 폴드 고정-w 블렌드 분석 (재학습 없음)")
    args = ap.parse_args()
    if args.twofold:
        two_fold_blend()
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(">> train.csv 로드")
    df = rd.load_train()
    X = L.build_features(df)
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()
    seasons = sorted(int(s) for s in np.unique(season))
    val = (season == args.val)
    yv = y[val]
    Xv = X[val]
    print(f"   val={args.val} {val.sum():,}행  r={yv.mean():.4f}  "
          f"시즌 기저율 {dict((int(s), round(float(y[season==s].mean()), 4)) for s in seasons)}")

    rows, preds = [], {}
    base_p = None
    for a in arms_for(args.val, seasons):
        t = time.time()
        p, nfit = run_arm(a, X, y, season, val)
        preds[a["name"]] = p
        if base_p is None:
            base_p = p
        bc, arg = best_cal(yv, p)
        rec = dict(arm=a["name"], fit=f"{min(a['fit'])}-{max(a['fit'])}", n_fit=nfit,
                   fixed_cal=L.score(yv, L.calibrate(p)), best_cal=bc, cal_arg=str(arg),
                   std=float(p.std()), mean=float(p.mean()),
                   rms_vs_base=float(np.sqrt(((p - base_p) ** 2).mean())),
                   d_paired=L.score(yv, L.calibrate(p)) - L.score(yv, L.calibrate(base_p)),
                   se_paired=paired_se(yv, L.calibrate(p), L.calibrate(base_p)),
                   sec=round(time.time() - t, 1))
        rows.append(rec)
        print(f"  {rec['arm']:16s} n={nfit:>9,}  고정캘리 {rec['fixed_cal']:8.2f}  "
              f"재적합 {bc:8.2f}  Δ(paired) {rec['d_paired']:+8.2f} ±{rec['se_paired']:.1f}  "
              f"std {rec['std']:.4f}  rms_vs_base {rec['rms_vs_base']:.4f}  [{rec['sec']}s]")

    t = pd.DataFrame(rows).sort_values("best_cal", ascending=False)
    print("\n" + "=" * 110)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- 기준과의 2원 블렌드: 단독으로 져도 멤버로는 값을 할 수 있다
    print("\n[블렌드] base_1s + arm (캘리 재적합 후 최고점)")
    bl = []
    for nm, p in preds.items():
        if nm == "base_1s":
            continue
        best = max((best_cal(yv, (1 - w) * base_p + w * p)[0], w) for w in (0.2, 0.3, 0.4, 0.5))
        bl.append(dict(arm=nm, blend_best=best[0], w=best[1],
                       gain=best[0] - best_cal(yv, base_p)[0]))
    b = pd.DataFrame(bl).sort_values("gain", ascending=False)
    print(b.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    out = OUT_DIR / f"centering_val{args.val}{args.tag}.json"
    out.write_text(json.dumps({"val": args.val, "arms": rows, "blend": bl},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez_compressed(OUT_DIR / f"preds_val{args.val}{args.tag}.npz", y=yv, **preds)
    print(f"\n>> 저장 {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
