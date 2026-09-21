"""
explore-exploit 스윕 오케스트레이터 (successive halving).

- 원칙(06 탐색원칙): 하드 제외 없음. 모든 후보를 동일 시간분할 CV LogLoss 게이트로.
- prior(tier)는 예산 배분에만: 1차 rung은 저비용(1 폴드)로 전 후보 스크리닝 →
  상위 생존자만 전체 폴드로 승격(exploit). 앵커는 항상 전체 폴드 평가(기준선).
- 결과: LogLoss 리더보드 + 앵커(GBDT) 대비 승패 플래그.

fold-aware 피처: 피처는 **폴드(val 시즌)별로** 생성한다(cutoff=val 시즌 → target-파생 통계를
train 경계에서 동결 = 배치제출 시간분할 정합, 누수 0). target 무관 seq는 한 번만 만든다.
"""
from __future__ import annotations
import math
from features import build_features, build_sequences
from validation import make_time_folds, evaluate_candidate
from candidates import get_candidates, ANCHOR


def _build_fold_reps(df, spec, val_season, seq_cache):
    """val_season을 cutoff로 하는 폴드 전용 입력 표현. seq(target무관)는 캐시 재사용."""
    X, _, _ = build_features(df, spec, label_cutoff_season=val_season)
    Xf13, _, _ = build_features(df, spec, include_llm_prior=True, label_cutoff_season=val_season)
    Xrich, _, _ = build_features(df, spec, rich=True, label_cutoff_season=val_season)
    Xctx, _, _ = build_features(df, spec, rich=True, ctx=True, label_cutoff_season=val_season)
    Xid, _, _ = build_features(df, spec, rich=True, ctx=True, include_ids=True,
                               label_cutoff_season=val_season)
    return {"flat": X, "flat_f13": Xf13, "flat_plus": Xrich, "flat_ctx": Xctx, "flat_id": Xid,
            "seq": seq_cache, "seq_kd": (seq_cache, X)}


def run_sweep(df, spec, keep_frac=0.6):
    # y/meta는 cutoff와 무관(정렬 동일) → 한 번만. seq도 target 무관 → 한 번만.
    _, y, meta = build_features(df, spec)
    seq_cache = build_sequences(df, spec, K=16)
    folds = make_time_folds(meta, spec, min_train_seasons=3)
    cands = get_candidates()

    # 폴드별(=val 시즌) 표현 캐시. 모든 후보·rung이 재사용.
    fold_reps = {f[3]: _build_fold_reps(df, spec, f[3], seq_cache) for f in folds}

    def get_rep(cand, val_season):
        reps = fold_reps[val_season]
        return reps.get(cand.needs, reps["flat"])

    f0 = fold_reps[folds[0][3]]
    print(f"[data] rows={len(meta)} feats(flat)={f0['flat'].shape[1]} "
          f"feats(f13)={f0['flat_f13'].shape[1]} feats(plus)={f0['flat_plus'].shape[1]} "
          f"feats(ctx)={f0['flat_ctx'].shape[1]} "
          f"seq={seq_cache.shape} folds={len(folds)} "
          f"seasons={sorted(meta[spec.season].unique())}")
    print(f"[cands] {len(cands)}개: {[(c.name, c.needs) for c in cands]}")
    print("[fold-aware] target-파생 피처는 val 시즌부터 동결(cutoff) = 배치제출 누수 0\n")

    # 앵커 = 전체 폴드 기준선
    anchor_res = next(evaluate_candidate(c, get_rep, y, meta, spec, folds)
                      for c in cands if c.name == ANCHOR)
    anchor_ll = anchor_res["logloss_cal"]

    # rung 0: 최신 폴드(val=마지막 시즌) 1개로 전 후보 스크리닝(explore).
    # 첫 폴드(val 2022)를 쓰면 era-의존 arm(F31 등)이 가중 조건(season>=2024) 공허로
    # 앵커와 동일해져 특색을 못 보이고 탈락 — test(2025) 체제와 같은 최신 폴드로 예선.
    rung0 = []
    for c in cands:
        r = evaluate_candidate(c, get_rep, y, meta, spec, folds[-1:])
        rung0.append(r)
        print(f"  [rung0] {c.name:<20} logloss(1fold)={r['logloss_cal']:.4f}")
    rung0.sort(key=lambda r: r["logloss_cal"])
    k = max(1, math.ceil(keep_frac * len(rung0)))
    survivors = {r["name"] for r in rung0[:k]} | {ANCHOR}
    print(f"\n[halving] rung0 상위 {k}/{len(rung0)} 생존 → 전체 폴드 승격: {sorted(survivors)}\n")

    # rung 1: 생존자 전체 폴드(exploit)
    final = []
    for c in cands:
        if c.name not in survivors:
            continue
        r = anchor_res if c.name == ANCHOR else evaluate_candidate(c, get_rep, y, meta, spec, folds)
        r["beats_anchor"] = r["logloss_cal"] < anchor_ll - 1e-4
        r["delta_vs_anchor"] = r["logloss_cal"] - anchor_ll
        final.append(r)
    final.sort(key=lambda r: r["logloss_cal"])

    print("=" * 86)
    print(f"{'순위':<4}{'후보':<22}{'LogLoss':<10}{'Brier':<9}{'AUC':<7}{'Cold':<9}{'vs앵커':<9}{'판정'}")
    print("-" * 86)
    for i, r in enumerate(final, 1):
        flag = "★채택가능" if r.get("beats_anchor") else ("(앵커)" if r["name"] == ANCHOR else "미달")
        cold = r.get("logloss_cold", float("nan"))
        cold_s = f"{cold:<9.4f}" if cold == cold else f"{'-':<9}"  # NaN 처리
        print(f"{i:<4}{r['name']:<22}{r['logloss_cal']:<10.4f}{r['brier_cal']:<9.4f}"
              f"{r['auc']:<7.3f}{cold_s}{r['delta_vs_anchor']:+.4f}   {flag}")
    print("=" * 86)
    print(f"앵커(GBDT) LogLoss={anchor_ll:.4f}. '★채택가능'만 앵커를 이긴 후보(게이트 통과).")

    # C5 A/B: 앵커를 캘리 모드별로 비교(게이트는 기본 모드로만 판정, 이건 참고).
    print("\n[C5 캘리 A/B — 앵커, 참고용]")
    calib_ab = {}
    for mode in ("isotonic_latest", "recent_window", "prior_shift"):
        r = evaluate_candidate(next(c for c in cands if c.name == ANCHOR),
                               get_rep, y, meta, spec, folds, calib_mode=mode)
        calib_ab[mode] = {"logloss_cal": r["logloss_cal"], "brier_cal": r["brier_cal"]}
        print(f"  calib={mode:<16} LogLoss={r['logloss_cal']:.4f} Brier={r['brier_cal']:.4f}")

    print("\n주의: LogLoss로만 판정(AUC 참고). 실데이터·다시드 재현 확인 후에만 최종 채택.")
    # 기록 레이어(reporting.record_run)로 넘어가는 report dict. 게이트·리더보드 로직 불변.
    return {
        "harness": {"anchor": ANCHOR, "anchor_logloss": anchor_ll, "keep_frac": keep_frac},
        "data": {"rows": len(meta), "seasons": sorted(int(s) for s in meta[spec.season].unique()),
                 "n_features_flat": int(f0["flat"].shape[1]), "n_folds": len(folds)},
        "candidates": [{"name": c.name, "tier": c.tier, "kind": c.kind, "needs": c.needs}
                       for c in cands],
        "rung0": [{"name": r["name"], "logloss_cal": r["logloss_cal"]} for r in rung0],
        "survivors": sorted(survivors),
        "leaderboard": final,
        "calib_ab": calib_ab,
    }
