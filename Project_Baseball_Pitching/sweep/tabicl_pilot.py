# -*- coding: utf-8 -*-
"""TabICLv2 파일럿 — "외부의 힘" 최후 미실측 후보(BSD-3·비게이트·합성 사전학습)의 실측.

    python sweep/tabicl_pilot.py --context c23 --ctx-cap 100000 --n-est 4

tabpfn3_pilot.py(라운드5 T1, 3구성 전패 −433/−2120/−933)와 같은 프로토콜:
  1) 쌍대 게이트 G1: 같은 컨텍스트·같은 82피처 LGBM(june풍)과 같은 쿼리 행에서
     쌍대 delta ≥ −1·SE 여야 생존. (절대점수는 30k 표본 노이즈 — 쌍대만 유효)
  2) 정직 d: 쿼리 = leg_matrix 프록시 행(season==2024 head N) → 캐시 레그 npy와 RMS.
     (컨텍스트 빈티지가 캐시와 다르면 d는 부풀려짐 — 참고 브래킷으로만 해석)
  3) 속도 G0: 2k 쿼리 실측 → 245,789행 외삽. 서버 L4는 3080보다 빠르지 않다고 가정.
  4) (신규) §5 행 독립 프로브: 같은 행을 1행 단독 / 50행 배치로 예측해 최대 편차 보고.
     TabICL의 column-then-row attention이 쿼리 행끼리 섞으면 점수 무관 즉사.

컨텍스트:
  c23  = 2023 시즌 (v3 파일럿 pilot_c23k100과 직접 비교용)
  c24x = 2024 시즌에서 프록시 쿼리 블록(head rows)을 제외한 나머지 — ICL 메모리제이션이
         점수/d를 왜곡하는 것을 차단한 최신 레짐 컨텍스트 (train-측 선택이므로 합법)
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
OUT = ROOT / "results" / "tabicl"
CKPT = ROOT / "data" / "tabicl" / "tabicl-classifier-v2-20260212.ckpt"

CTX = {
    "c23": (2023, 2023),
    "c2223": (2022, 2023),
    "c24x": None,  # 특수 처리: 2024 minus 프록시 블록
}


def score(p, y):
    return 1e5 * (1 - float(np.mean((p - y) ** 2)) / 0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", default="c23", choices=list(CTX))
    ap.add_argument("--ctx-cap", type=int, default=100_000)
    ap.add_argument("--rows", type=int, default=30_000, help="proxy query rows")
    ap.add_argument("--n-est", type=int, default=4)
    ap.add_argument("--abort-min", type=float, default=8.0,
                    help="local query-set projection abort threshold (min)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--kv-cache", action="store_true")
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

    # 쿼리 = leg_matrix 프록시와 동일: season==2024 첫 rows행 (train 순서)
    q_idx = np.where(season == 2024)[0][: args.rows]
    if args.context == "c24x":
        ctx_idx = np.where(season == 2024)[0][args.rows:]
    else:
        lo, hi = CTX[args.context]
        ctx_idx = np.where((season >= lo) & (season <= hi))[0]
    if len(ctx_idx) > args.ctx_cap:
        rng = np.random.default_rng(0)
        ctx_idx = np.sort(rng.choice(ctx_idx, args.ctx_cap, replace=False))
    Xc, yc = X.iloc[ctx_idx].to_numpy(dtype=np.float32), y[ctx_idx]
    Xq, yq = X.iloc[q_idx].to_numpy(dtype=np.float32), y[q_idx]
    print(f"   context {args.context}: {len(ctx_idx):,} rows | query: {len(q_idx):,} rows")

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

    # ---------------------------------------------------------------- TabICLv2
    import torch
    from tabicl import TabICLClassifier
    clf = TabICLClassifier(n_estimators=args.n_est, device="cuda",
                           model_path=str(CKPT), allow_auto_download=False,
                           batch_size=args.batch_size, kv_cache=args.kv_cache,
                           random_state=42, verbose=False)
    t = time.time()
    clf.fit(Xc, yc)
    fit_s = time.time() - t
    report["kv_cache"] = bool(args.kv_cache)
    print(f"[ticl] fit {len(ctx_idx):,} ctx rows: {fit_s:.1f}s"
          f" | VRAM {torch.cuda.max_memory_allocated()//2**20} MB")

    # 속도 측정: 2k 쿼리 → 서버 전량(245,789) 외삽
    t = time.time()
    warm = clf.predict_proba(Xq[:2000])[:, 1]
    per1k = (time.time() - t) / 2.0
    proj_full_min = per1k * 245.789 / 60
    proj_query_min = per1k * len(q_idx) / 1000 / 60
    report["timing"] = {"fit_s": round(fit_s, 1), "per_1k_s": round(per1k, 2),
                        "proj_245k_min_local": round(proj_full_min, 1)}
    print(f"[ticl] {per1k:.2f}s/1k query | proj 245.8k rows: {proj_full_min:.1f} min (local)"
          f" | this query set: {proj_query_min:.1f} min")
    print(f"[ticl] VRAM peak {torch.cuda.max_memory_allocated()//2**20} MB")

    # §5 행 독립 프로브: 같은 행을 단독/배치로 — 편차가 0이 아니면 서빙 불가 소지
    p_solo = clf.predict_proba(Xq[0:1])[:, 1]
    p_b50 = clf.predict_proba(Xq[0:50])[:, 1]
    p_b50r = clf.predict_proba(Xq[0:50])[:, 1]  # 같은 배치 반복 — 비결정성 대조군
    indep = {"solo_vs_batch50": float(abs(p_solo[0] - p_b50[0])),
             "warm_vs_batch50_max": float(np.max(np.abs(warm[:50] - p_b50))),
             "batch50_repeat_max": float(np.max(np.abs(p_b50 - p_b50r)))}
    report["row_independence"] = indep
    print(f"[§5  ] row0 solo-vs-b50 diff={indep['solo_vs_batch50']:.2e}"
          f" | warm-vs-b50 max={indep['warm_vs_batch50_max']:.2e}"
          f" | b50-repeat max={indep['batch50_repeat_max']:.2e}")

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
    report["tabicl"] = {"score": score(p, yq), "pred_s": round(time.time()-t, 1),
                        "mean": float(p.mean()), "sd": float(p.std())}
    print(f"[ticl] score={report['tabicl']['score']:.2f} mean={p.mean():.4f} sd={p.std():.4f}")

    if p_ref is not None:
        d_row = (p - yq) ** 2 - (p_ref - yq) ** 2
        se = 4e5 * float(np.std(d_row)) / np.sqrt(len(d_row))
        delta = report["tabicl"]["score"] - report["ref_lgbm"]["score"]
        report["paired"] = {"delta": round(delta, 2), "se": round(se, 2)}
        print(f"[gate] paired delta (ticl - lgbm) = {delta:+.2f}  (SE {se:.2f})")

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
    print(f">> saved results/tabicl/pilot_{tag}.json")


if __name__ == "__main__":
    main()
