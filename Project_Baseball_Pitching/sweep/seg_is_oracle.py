# -*- coding: utf-8 -*-
"""세그먼트×당해시즌 라벨 오라클 — "1200~1320점대가 합법 채널인가"의 최종 판정 실험.

    python sweep/seg_is_oracle.py --val 2024            # Stage A (~15-20분)
    python sweep/seg_is_oracle.py --vals 2024,2023,2022 # Stage B: Δ≥+30 arm만 폴드 확장

⚠ 이름이 비슷한 sweep/seg_probe.py(캘리/세그먼트 LB 프로빙 대수)와 **무관**하다.
이 스크립트는 로컬 라벨 오라클 측정 전용이며, 여기서 만든 피처는 **배포 불가**다
(당해시즌 세그먼트 라벨 통계 — 공식 asof 컬럼이 제공하지 않으므로 2025에서 계산 불가).

## 설계 (계획: ~/.claude/plans/jolly-bouncing-falcon.md · 배경: docs/research/14 §2)

1위 1319.9의 잔여 합법 가설 = "당해시즌(as-of) 분해의 세그먼트·상대별 심화".
train 라벨로 (투수, 시즌, 세그먼트)별 **as-of(현재 라벨 제외) 성공률**을 만들어
게이트 눈금(inseason.gate 미러: fit v−1 → val v, june 2종 pair)에서 상한을 잰다.

- 순서: `asof_pitcher_n`이 투수 내 단조 +1 커리어 시계(A-2 검증)라 (pid,season,seg)
  그룹을 이 시계로 정렬해 cumcount/cumsum−y 로 정확한 as-of가 나온다. 위반율 >0.1%면
  LOO-only 모드로 강등(폐쇄 판정력은 유지).
- as-of판(주) + LOO 시즌전체판(부, oracle_ceiling.loo_rate — 시즌 미래분 포함 = 더
  관대한 상한). LOO≫as-of 차이 = "시즌 미래분, 합법 접근 불가" 자체가 판정 정보.
- 앵커: base53≈707.9 · base57≈800~801(is4 실측 +92.5의 재현) — 미재현 시 측정 중단.
  캐너리: iid 재추첨 라벨로 만든 블록은 Δ≈0이어야 한다(자기 라벨 누수 검출).
- 판정(Δ* = V24 전 변형 최대 증분 vs base57): ≤+30 폐쇄(확정) · +30~100 회색(해석 절차)
  · ≥+100 합법 실재 필요조건 · **≥+350이어야만 "1200대=합법" 가설이 산술 생존**.
- Δ 합산 금지 — 총 여력은 all 조합 arm으로만 읽는다.
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
import real_data as rd                                      # noqa: E402
import lgbm_family as L                                     # noqa: E402
import inseason as IS                                       # noqa: E402
from season_centering import best_cal, paired_se, fit_pair, predict_pair   # noqa: E402
from oracle_ceiling import loo_rate, best_cal as best_cal_wide             # noqa: E402

OUT = HERE.parent / "results" / "seg_oracle"
K_GRID_DENSE = (50.0, 100.0, 200.0)
K_GRID_SPARSE = (10.0, 25.0, 50.0)


# ---------------------------------------------------------------- 시계·블록
def pitcher_clock(df):
    """투수 내 시간순 정렬(asof_pitcher_n 시계) + A-2 재검증(단조 +1 위반율)."""
    pid = df["pitcher_id"].to_numpy()
    n = df["asof_pitcher_n"].to_numpy(dtype=np.int64)
    order = np.lexsort((n, pid))
    ps, ns = pid[order], n[order]
    same = ps[1:] == ps[:-1]
    viol = float((np.diff(ns)[same] != 1).mean()) if same.any() else 1.0
    return order, viol


def _restore(vals, order, n_rows):
    out = np.empty(n_rows, dtype=np.float64)
    out[order] = vals
    return out


def asof_seg_block(df, order, seg, K, y=None, tag="seg"):
    """(pid, season, seg)별 as-of(현재 라벨 제외) 수축 성공률 3열 [원 행 순서로 반환].

    시계 정렬 후 그룹 내 cumcount/cumsum−y — shift(1) 규율과 동치(현재 행 라벨 미포함).
    """
    y = df[rd.TARGET].to_numpy(dtype=np.float64) if y is None else np.asarray(y, float)
    d = pd.DataFrame({
        "pid": df["pitcher_id"].to_numpy()[order],
        "s": df[rd.SEASON].to_numpy()[order],
        "seg": np.asarray(seg)[order],
        "y": y[order],
    })
    g = d.groupby(["pid", "s", "seg"], sort=False)["y"]
    n_past = g.cumcount().to_numpy(dtype=np.float64)
    k_past = g.cumsum().to_numpy(dtype=np.float64) - d["y"].to_numpy()
    sm = (k_past + K / 2.0) / (n_past + K)
    nr = len(df)
    return (_restore(sm, order, nr), _restore(n_past, order, nr))


def block_frame(df, order, seg, K, ov_sm, y=None, tag="seg"):
    sm, n_past = asof_seg_block(df, order, seg, K, y=y, tag=tag)
    return pd.DataFrame({
        f"{tag}_sm": sm.astype("float32"),
        f"{tag}_d": (sm - ov_sm).astype("float32"),
        f"{tag}_logn": np.log1p(n_past).astype("float32"),
    })


def loo_seg_block(df, seg, K, ov_sm, tag="seg"):
    """LOO 시즌 전체판(미래분 포함) — oracle_ceiling.loo_rate 재사용. 관대한 상한."""
    y = df[rd.TARGET].to_numpy(dtype=np.float64)
    key = pd.MultiIndex.from_arrays([
        df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy(), np.asarray(seg)]).factorize()[0]
    sm = loo_rate(key, y, K, float(y.mean()))
    cnt = pd.Series(key).map(pd.Series(key).value_counts()).to_numpy(dtype=np.float64) - 1.0
    return pd.DataFrame({
        f"{tag}_sm": sm.astype("float32"),
        f"{tag}_d": (sm - ov_sm).astype("float32"),
        f"{tag}_logn": np.log1p(np.maximum(cnt, 0.0)).astype("float32"),
    })


def legal_prior_block(df, seg, K, ov_prior, tag="seg"):
    """**합법 실현판** — 각 행이 자기 시즌 **미만**의 train (pid, seg) 누적만 조회.

    배포 대응(2025 행 = ≤2024 룩업)이고 test 행 간 참조가 없다(§5 준수). 오라클과 달리
    당해시즌 라벨을 보지 않는다 — "train 룩업으로 실현 가능한 상한". 매치업(pid×batter)이
    핵심: batter_id가 시즌 지속되면 이 룩업은 그대로 동봉 가능한 합법 피처다.
    ov_prior = 같은 방식의 투수-전체(seg 무시) 시즌미만 사전율 (중심화용).
    """
    pid = df["pitcher_id"].to_numpy()
    s = df[rd.SEASON].to_numpy()
    y = df[rd.TARGET].to_numpy(dtype=np.float64)
    seg = np.asarray(seg)
    d = pd.DataFrame({"pid": pid, "seg": seg, "s": s, "y": y})
    agg = d.groupby(["pid", "seg", "s"], sort=True)["y"].agg(["size", "sum"]).reset_index()
    agg = agg.sort_values("s")
    gp = agg.groupby(["pid", "seg"], sort=False)
    agg["cn"] = gp["size"].cumsum() - agg["size"]           # s 미만 누적 표본
    agg["ck"] = gp["sum"].cumsum() - agg["sum"]             # s 미만 누적 성공
    key = pd.MultiIndex.from_arrays([d.pid, d.seg, d.s])
    ai = pd.MultiIndex.from_arrays([agg.pid, agg.seg, agg.s])
    cn = pd.Series(agg["cn"].to_numpy(), index=ai).reindex(key).to_numpy(dtype=np.float64)
    ck = pd.Series(agg["ck"].to_numpy(), index=ai).reindex(key).to_numpy(dtype=np.float64)
    m = float(y.mean())
    sm = (ck + K * m) / (cn + K)
    return pd.DataFrame({
        f"{tag}_sm": sm.astype("float32"),
        f"{tag}_d": (sm - ov_prior).astype("float32"),
        f"{tag}_logn": np.log1p(np.maximum(cn, 0.0)).astype("float32"),
    }), float((cn[s == s.max()] > 0).mean())


def unit_assert():
    """합성 미니그룹 결정적 검증: y=[1,0,1] → n_past=[0,1,2], k_past=[0,1,1]."""
    mini = pd.DataFrame({
        "pitcher_id": [7, 7, 7], rd.SEASON: [2024] * 3, rd.TARGET: [1.0, 0.0, 1.0],
        "asof_pitcher_n": [10, 11, 12]})
    order, _ = pitcher_clock(mini)
    sm, n_past = asof_seg_block(mini, order, np.zeros(3), K=0.0 + 1e-9)
    k_past = sm * (n_past + 1e-9)
    assert np.allclose(n_past, [0, 1, 2]), n_past
    assert np.allclose(np.rint(k_past), [0, 1, 1]), k_past
    print("   [unit] as-of cumsum 규율 OK (n_past=[0,1,2], k_past=[0,1,1])")


# ---------------------------------------------------------------- 게이트
def run_gate(name, XA, fit, val, y, yv, base_p, results, note="", base_best=None):
    t = time.time()
    bs = fit_pair(XA[fit], y[fit], None)
    p = np.clip(predict_pair(bs, XA[val]), 0, 1)
    pc, bpc = L.calibrate(p), L.calibrate(base_p)
    fixed = L.score(yv, pc)
    best, arg = best_cal(yv, p)
    d_fixed = fixed - L.score(yv, bpc)
    if base_best is None:
        base_best = best_cal(yv, base_p)[0]
    d_best = best - base_best
    rec = dict(arm=name, ncol=int(XA.shape[1]), fixed=round(fixed, 2), best=round(best, 2),
               d_fixed=round(d_fixed, 2), d_best=round(d_best, 2),
               se=round(paired_se(yv, pc, bpc), 2), cal=str(arg), note=note,
               sec=round(time.time() - t, 1))
    results.append(rec)
    print(f"  {name:16s} ({rec['ncol']:2d}col) 고정 {fixed:8.2f} 재적합 {best:8.2f} "
          f"Δfix {d_fixed:+7.2f}±{rec['se']:.1f} Δbest {d_best:+7.2f} [{rec['sec']}s] {note}")
    return p, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024")
    ap.add_argument("--val", type=int, default=None)
    ap.add_argument("--legal", action="store_true",
                    help="합법 실현판만 측정 — 각 행이 시즌 미만 train (pid,seg) 룩업만 조회")
    args = ap.parse_args()
    vals = [args.val] if args.val else [int(s) for s in args.vals.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    unit_assert()
    print(">> train.csv 로드")
    df = rd.load_train()
    y = df[rd.TARGET].to_numpy(dtype=np.float64)
    season = df[rd.SEASON].to_numpy()
    pid = df["pitcher_id"].to_numpy()

    print(">> 투수 시계 검증 (A-2 재검증)")
    order, viol = pitcher_clock(df)
    asof_ok = viol <= 0.001
    print(f"   단조 +1 위반율 = {viol:.6f} → {'as-of판 사용' if asof_ok else '⚠ LOO-only 강등'}")

    print(">> 기준선 조립 (june53 + is4 = base57, inseason.COLS4 순서)")
    X53 = L.build_features(df).reset_index(drop=True)
    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, pid, season)
    A = IS.build_inseason(df, nb, kb)
    X57 = pd.concat([X53, A[IS.COLS4]], axis=1)

    # 전 축 정의 (이름, 세그 값, K격자)
    cgrp3 = np.sign(df["balls_before"].to_numpy(int) - df["strikes_before"].to_numpy(int))
    axes = {
        "bhand": (df["batter_hand"].to_numpy(), K_GRID_DENSE),
        "cgrp3": (cgrp3, K_GRID_DENSE),
        "month": (df["game_month"].to_numpy(), K_GRID_DENSE),
        "gtype": (df[rd.SEGMENT].astype(str).to_numpy(), K_GRID_DENSE),
        "mu":    (df["batter_id"].to_numpy(), K_GRID_SPARSE),
    }
    const_seg = np.zeros(len(df))

    # ---- 합법 실현판 (train 시즌미만 룩업, 배포 대응) ---------------------------------
    if args.legal:
        # 투수-전체 시즌미만 사전율(중심화 기준)
        ov_prior_blk, _ = legal_prior_block(df, const_seg, 200.0, np.full(len(df), y.mean()),
                                            tag="ovp")
        ov_prior = ov_prior_blk["ovp_sm"].to_numpy(dtype=np.float64)
        for v in vals:
            fit, val = season == v - 1, season == v
            yv = y[val]
            print(f"\n{'='*88}\n[val {v} · 합법 실현판] fit {fit.sum():,} → val {val.sum():,}")
            results = []
            base_p, base_best = run_gate("base57", X57, fit, val, y, yv,
                                         np.full(val.sum(), yv.mean()),
                                         results, note="기준선", base_best=0.0)
            blocks = []
            for nm, (seg, grid) in axes.items():
                blk, cov = legal_prior_block(df, seg, grid[1], ov_prior, tag=f"L{nm}")
                run_gate(f"legal_{nm}", pd.concat([X57, blk], axis=1), fit, val, y, yv,
                         base_p, results, note=f"val커버 {cov:.3f}", base_best=base_best)
                blocks.append(blk)
            run_gate("legal_all5", pd.concat([X57] + blocks, axis=1), fit, val, y, yv,
                     base_p, results, note="합법 총 여력", base_best=base_best)
            (OUT / f"legal_val{v}.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n>> 저장 {OUT}  [{time.time()-t0:.0f}s]")
        return

    all_results = {}
    for v in vals:
        fit, val = season == v - 1, season == v
        yv = y[val]
        print(f"\n{'='*88}\n[val {v}] fit {fit.sum():,} → val {val.sum():,}  r={yv.mean():.4f}")
        results = []

        # ---- 앵커
        base53_p, b53_best = run_gate("base53", X53, fit, val, y, yv,
                                      np.full(val.sum(), yv.mean()), results,
                                      note="앵커 ≈707.9", base_best=0.0)
        # 투수×당해 as-of ①(라벨판 overall) — is4 몫의 실측
        ov_sm200, ov_n = asof_seg_block(df, order, const_seg, 200.0)
        p_sm500 = ((df["asof_pitcher_success_rate"].fillna(0.0).to_numpy(float)
                    * df["asof_pitcher_n"].to_numpy(float) + 250.0)
                   / (df["asof_pitcher_n"].to_numpy(float) + 500.0))
        ov_blk = pd.DataFrame({"ov_sm": ov_sm200.astype("float32"),
                               "ov_d": (ov_sm200 - p_sm500).astype("float32"),
                               "ov_logn": np.log1p(ov_n).astype("float32")})
        run_gate("b53+asof_ov", pd.concat([X53, ov_blk], axis=1), fit, val, y, yv,
                 base53_p, results, note="① is4 등가 정보 (기대 +85~100 vs base53)",
                 base_best=b53_best)

        base_p, base_best = run_gate("base57", X57, fit, val, y, yv, base53_p, results,
                                     note="주 기준선 ≈800~801 (Δ는 base53 대비)",
                                     base_best=b53_best)

        # 앵커 검증
        b53 = next(r for r in results if r["arm"] == "base53")["best"]
        b57 = next(r for r in results if r["arm"] == "base57")["best"]
        if not (690 <= b53 <= 730 and 780 <= b57 <= 830) and v == 2024:
            print(f"   🚨 앵커 미재현(base53 {b53} / base57 {b57}) — 측정 중단")
            sys.exit(1)

        # ---- 캐너리 (iid 재추첨 라벨 블록 → Δ≈0 요구)
        rng = np.random.default_rng(20260830)
        r_season = pd.Series(y).groupby(pd.Series(season)).transform("mean").to_numpy()
        y_fake = rng.binomial(1, r_season).astype(np.float64)
        if asof_ok:
            cblk = block_frame(df, order, axes["bhand"][0], 100.0, ov_sm200,
                               y=y_fake, tag="cnry")
            run_gate("canary", pd.concat([X57, cblk], axis=1), fit, val, y, yv,
                     base_p, results, note="iid 라벨 — Δ≈0이어야 함", base_best=base_best)

        # ---- 단일 축 × K격자 (as-of판)
        best_k = {}
        if asof_ok:
            for nm, (seg, grid) in axes.items():
                for K in grid:
                    blk = block_frame(df, order, seg, K, ov_sm200, tag=nm)
                    run_gate(f"{nm}_K{K:g}", pd.concat([X57, blk], axis=1),
                             fit, val, y, yv, base_p, results, base_best=base_best)
                arm_rows = [r for r in results if r["arm"].startswith(nm + "_K")]
                best_k[nm] = float(max(arm_rows, key=lambda r: r["d_best"])["arm"].split("K")[1])

            # ---- 조합 arm (총 여력은 이것으로 읽는다)
            blocks4 = [block_frame(df, order, axes[nm][0], best_k[nm], ov_sm200, tag=nm)
                       for nm in ("bhand", "cgrp3", "month", "gtype")]
            run_gate("all4", pd.concat([X57] + blocks4, axis=1), fit, val, y, yv,
                     base_p, results, note="총 여력(as-of판)", base_best=base_best)
            blocks5 = blocks4 + [block_frame(df, order, axes["mu"][0], best_k["mu"],
                                             ov_sm200, tag="mu")]
            run_gate("all5", pd.concat([X57] + blocks5, axis=1), fit, val, y, yv,
                     base_p, results, note="총 여력+매치업", base_best=base_best)

        # ---- LOO판 (관대한 상한: 시즌 미래분 포함)
        loo4 = [loo_seg_block(df, axes[nm][0], best_k.get(nm, 100.0), ov_sm200, tag=nm)
                for nm in ("bhand", "cgrp3", "month", "gtype")]
        run_gate("all4_LOO", pd.concat([X57] + loo4, axis=1), fit, val, y, yv,
                 base_p, results, note="관대 상한(시즌 전체 LOO)", base_best=base_best)
        loo5 = loo4 + [loo_seg_block(df, axes["mu"][0], best_k.get("mu", 25.0),
                                     ov_sm200, tag="mu")]
        run_gate("all5_LOO", pd.concat([X57] + loo5, axis=1), fit, val, y, yv,
                 base_p, results, note="관대 상한+매치업", base_best=base_best)

        # ---- 직접예측 arm (학습 없음, wide 캘리)
        d1 = best_cal_wide(yv, np.clip(ov_sm200[val], 1e-6, 1 - 1e-6), coarse=True)
        key_ps = pd.MultiIndex.from_arrays([pid, season]).factorize()[0]
        loo_ov = loo_rate(key_ps, y, 50.0, float(y.mean()))
        d2 = best_cal_wide(yv, np.clip(loo_ov[val], 1e-6, 1 - 1e-6), coarse=True)
        print(f"  [직접] D1 as-of overall(K200) = {d1:.1f} · "
              f"D2 LOO 투수시즌(K50) = {d2:.1f} (O1 앵커 723.9 대조)")
        results.append(dict(arm="direct_D1", best=round(d1, 2)))
        results.append(dict(arm="direct_D2", best=round(d2, 2)))

        all_results[v] = results
        (OUT / f"gate_val{v}.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------------------------------------------------------------- 판정
    v0 = vals[0]
    rows = [r for r in all_results[v0]
            if r["arm"] not in ("base53", "b53+asof_ov", "base57", "canary")
            and "d_best" in r]
    dstar = max((r["d_best"] for r in rows), default=float("nan"))
    dstar_arm = max(rows, key=lambda r: r["d_best"])["arm"] if rows else "-"
    cnry = next((r["d_best"] for r in all_results[v0] if r["arm"] == "canary"), None)
    print(f"\n{'='*88}\n판정 (V{v0}, Δ* = 전 변형 최대 d_best vs base57)")
    print(f"  Δ* = {dstar:+.2f}  (arm: {dstar_arm})   캐너리 Δ = "
          f"{cnry if cnry is not None else 'n/a'}")
    if dstar <= 30:
        verdict = ("폐쇄(확정): 합법 세그먼트 채널 상한 ≤ +30 → LB 환산 ≤ +14~23. "
                   "1200~1320대는 이 채널로 설명 불가 — 잔여 설명 = test 배치 재구성(§5).")
    elif dstar < 100:
        verdict = ("회색(+30~100): 해석 절차 적용 — LOO≫as-of 여부·month 주도 여부·"
                   "폴드 부호 안정성 확인 후 재판정 (계획 §판정표).")
    elif dstar < 350:
        verdict = ("합법 실재 필요조건 충족(≥+100): 실현 설계 착수 대상. 단 1200대 설명에는 "
                   "여전히 미달(≥+350 필요) — 상위권 가설과는 별개로 우리 점수 개선 후보.")
    else:
        verdict = "≥+350: '1200~1320 = 합법 채널' 가설이 산술 생존 — 전면 재평가."
    print(f"  ⇒ {verdict}")
    summary = dict(vals=vals, viol_rate=viol, asof_mode=asof_ok, dstar=dstar,
                   dstar_arm=dstar_arm, canary=cnry, verdict=verdict, best_k=best_k)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"\n>> 저장 {OUT}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
