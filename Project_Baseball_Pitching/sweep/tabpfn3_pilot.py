# -*- coding: utf-8 -*-
"""TabPFN-3 파일럿 — 라운드5 T1 후보의 게이트/d/속도 실측.

    python sweep/tabpfn3_pilot.py --context c23 --rows 30000 --n-est 1

측정 3종 (docs/research/14 §4의 사전 등록 킬 기준에 대응):
  1) 쌍대 게이트: 같은 컨텍스트(시즌 슬라이스)·같은 82피처로 학습한 LGBM(june풍)과
     같은 쿼리 행에서 Brier 쌍대 비교. (절대점수는 30k 표본이라 ±수백점 노이즈 —
     **쌍대 delta만 유효**. SE_delta로 판단한다.)
  2) 정직 d: 쿼리 = leg_matrix와 동일한 프록시 행(season==2024 head N) →
     캐시된 레그 예측 npy들과 RMS 직접 비교.
  3) 속도: 2k 쿼리를 먼저 재서 245,789행(서버 전량) 예상 시간을 외삽.
     로컬 예상 > --abort-min 이면 예측을 중단하고 타이밍만 보고.

주의: 이 v0은 is4(당해시즌 분해) 피처가 없다 — 참조 LGBM도 없으므로 쌍대는 공정.
절대 레벨 해석 시 is4 핸디캡(로컬 +90급)을 감안할 것.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sweep"))
OUT = ROOT / "results" / "tabpfn3"

CTX = {
    "c23": (2023, 2023),
    "c2223": (2022, 2023),
    "c1923": (2019, 2023),
}


def score(p, y):
    return 1e5 * (1 - float(np.mean((p - y) ** 2)) / 0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", default="c23", choices=list(CTX))
    ap.add_argument("--ctx-cap", type=int, default=250_000)
    ap.add_argument("--rows", type=int, default=30_000, help="proxy query rows")
    ap.add_argument("--n-est", type=int, default=1)
    ap.add_argument("--abort-min", type=float, default=8.0,
                    help="local full-serve projection abort threshold (min)")
    ap.add_argument("--fit-mode", default="fit_with_cache",
                    choices=["low_memory", "fit_preprocessors", "fit_with_cache", "batched"])
    ap.add_argument("--kv", default="auto", help="kv_cache_precision: auto|int8|fp8|none")
    ap.add_argument("--model-path", default=None,
                    help="e.g. tabpfn-v3-classifier-v3_20260417_binary.ckpt")
    ap.add_argument("--skip-ref", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"{args.context}_e{args.n_est}"
    OUT.mkdir(parents=True, exist_ok=True)

    import real_data as rd
    t0 = time.time()
    print(">> load train")
    df = rd.load_train()
    X = rd.build_features(df)
    y = df[rd.TARGET].to_numpy(dtype=np.float32)
    season = df[rd.SEASON].to_numpy()
    print(f"   {X.shape} in {time.time()-t0:.0f}s")

    lo, hi = CTX[args.context]
    ctx_idx = np.where((season >= lo) & (season <= hi))[0]
    if len(ctx_idx) > args.ctx_cap:
        rng = np.random.default_rng(0)
        ctx_idx = np.sort(rng.choice(ctx_idx, args.ctx_cap, replace=False))
    # 쿼리 = leg_matrix 프록시와 동일: season==2024 첫 rows행 (train 순서)
    q_idx = np.where(season == 2024)[0][: args.rows]
    Xc, yc = X.iloc[ctx_idx].to_numpy(dtype=np.float32), y[ctx_idx]
    Xq, yq = X.iloc[q_idx].to_numpy(dtype=np.float32), y[q_idx]
    cat_idx = list(range(len(rd.CAT_COLS)))
    print(f"   context {args.context}: {len(ctx_idx):,} rows | query: {len(q_idx):,} rows"
          f" | cat cols {cat_idx}")

    report = {"tag": tag, "context": args.context, "n_ctx": int(len(ctx_idx)),
              "n_query": int(len(q_idx)), "n_est": args.n_est}

    # ---------------------------------------------------------------- LGBM 참조
    p_ref = None
    if not args.skip_ref:
        import lightgbm as lgb
        t = time.time()
        m = lgb.train(
            dict(objective="binary", metric="binary_logloss", learning_rate=0.05,
                 num_leaves=15, min_data_in_leaf=1000, feature_fraction=0.85,
                 bagging_fraction=0.8, bagging_freq=1, verbosity=-1, seed=0),
            lgb.Dataset(Xc, label=yc), num_boost_round=600)
        p_ref = m.predict(Xq)
        report["ref_lgbm"] = {"score": score(p_ref, yq), "train_s": round(time.time()-t, 1)}
        print(f"[ref ] lgbm june-ish: score={report['ref_lgbm']['score']:.2f}"
              f" ({report['ref_lgbm']['train_s']}s)")

    # ---------------------------------------------------------------- TabPFN-3
    import torch
    from tabpfn import TabPFNClassifier
    kv = None if args.kv == "none" else args.kv
    extra = {"model_path": args.model_path} if args.model_path else {}
    clf = TabPFNClassifier(device="cuda", n_estimators=args.n_est,
                           categorical_features_indices=cat_idx,
                           fit_mode=args.fit_mode, kv_cache_precision=kv,
                           ignore_pretraining_limits=True, **extra)
    t = time.time()
    clf.fit(Xc, yc)
    fit_s = time.time() - t
    report["fit_mode"] = args.fit_mode
    report["kv"] = str(kv)
    print(f"[tpfn] fit({args.fit_mode}, kv={kv}) {len(ctx_idx):,} ctx rows: {fit_s:.1f}s"
          f" | VRAM {torch.cuda.max_memory_allocated()//2**20} MB")

    # 속도 측정: 2k 쿼리 → 서버 전량(245,789) 외삽
    t = time.time()
    warm = clf.predict_proba(Xq[:2000])[:, 1]
    per1k = (time.time() - t) / 2.0
    proj_full_min = per1k * 245.789 / 60
    proj_query_min = per1k * len(q_idx) / 1000 / 60
    report["timing"] = {"fit_s": round(fit_s, 1), "per_1k_s": round(per1k, 2),
                        "proj_245k_min_local": round(proj_full_min, 1)}
    print(f"[tpfn] {per1k:.2f}s/1k query | proj 245.8k rows: {proj_full_min:.1f} min (local)"
          f" | this query set: {proj_query_min:.1f} min")
    print(f"[tpfn] VRAM peak {torch.cuda.max_memory_allocated()//2**20} MB")

    if proj_query_min > args.abort_min:
        print(f"[abort] projected query time {proj_query_min:.1f}min > {args.abort_min}min"
              " — timing-only report saved")
        (OUT / f"pilot_{tag}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        return

    t = time.time()
    chunks = [warm]
    B = 20000
    for s in range(2000, len(q_idx), B):
        chunks.append(clf.predict_proba(Xq[s:s+B])[:, 1])
        print(f"   ... {min(s+B, len(q_idx)):,}/{len(q_idx):,} ({time.time()-t:.0f}s)")
    p = np.concatenate(chunks)
    report["tabpfn"] = {"score": score(p, yq), "pred_s": round(time.time()-t, 1),
                        "mean": float(p.mean()), "sd": float(p.std())}
    print(f"[tpfn] score={report['tabpfn']['score']:.2f} mean={p.mean():.4f} sd={p.std():.4f}")

    if p_ref is not None:
        d_row = (p - yq) ** 2 - (p_ref - yq) ** 2
        se = 4e5 * float(np.std(d_row)) / np.sqrt(len(d_row))
        delta = report["tabpfn"]["score"] - report["ref_lgbm"]["score"]
        report["paired"] = {"delta": round(delta, 2), "se": round(se, 2)}
        print(f"[gate] paired delta (tpfn - lgbm) = {delta:+.2f}  (SE {se:.2f})")

    # ---------------------------------------------------------------- d vs 캐시 레그
    import glob
    dvs = {}
    for name in ("cregime", "clookup", "cmoe", "physmix", "tm3L", "ysy_mlp", "calP"):
        fs = sorted(glob.glob(str(ROOT / "results/leg_matrix" / f"{name}_{args.rows}_*.npy")))
        if fs:
            q = np.load(fs[-1])
            if len(q) == len(p):
                dvs[name] = round(float(np.sqrt(np.mean((p - q) ** 2))), 5)
    if dvs:
        legs4 = [n for n in ("clookup", "cmoe", "physmix", "tm3L") if n in dvs]
        if len(legs4) == 4:
            P4 = np.mean([np.load(sorted(glob.glob(str(ROOT / "results/leg_matrix" / f"{n}_{args.rows}_*.npy")))[-1]) for n in legs4], axis=0)
            dvs["D4_mean"] = round(float(np.sqrt(np.mean((p - P4) ** 2))), 5)
        report["rms_vs_legs"] = dvs
        print("[d   ] " + "  ".join(f"{k}={v}" for k, v in dvs.items()))

    np.save(OUT / f"pilot_{tag}_pred.npy", p)
    (OUT / f"pilot_{tag}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f">> saved results/tabpfn3/pilot_{tag}.json")


if __name__ == "__main__":
    main()
