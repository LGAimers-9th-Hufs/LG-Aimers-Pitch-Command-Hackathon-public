# -*- coding: utf-8 -*-
"""Program J — is4 이후 전면 HPO. 모든 멤버 파라미터가 53피처 시절 유산이라는 빈틈(J1~J3)을 메운다.

    python sweep/hpo.py --target june   --trials 60      # 2024 학습 계열
    python sweep/hpo.py --target allraw --trials 30      # 전 시즌 1.47M행 계열 (J3: 용량 상향 허용)
    python sweep/hpo.py --target june --verify           # 상위 3구성을 val2022(미사용 폴드)로 확인

## 왜 지금인가

- Phase 0의 용량 격자("+2.1 여지")는 **is4 이전 53피처**에서 측정했다. is4가 정보 구조를 바꿨다
  (june_l15 단독 697→791). 최적점이 이동했을 수 있는데 재탐색한 적이 없다.
- lr 0.03/iter 400~500은 june에게 물려받은 값. **저 lr × 다 iter**(대회 정석)를 한 번도 안 봤다.
- `bagging_fraction`은 아예 미사용이었다.
- allraw 멤버는 1.47M행(6배)인데 253k용 파라미터 그대로 → 과소적합 가능성.

## 프로토콜 (선택 낙관 방지 — F에서 검증된 구조)

- 목적함수 = **maximin**: `min(Δ_{fit2023→val2024}, Δ_{fit2022→val2023})`, Δ는 기준 스펙 대비
  affine 재적합 점수 차. **val2022는 탐색에 절대 쓰지 않고** `--verify`에서만 본다.
- 트라이얼 0 = 현 스펙 (정합성 앵커 겸 기준).
- `num_iterations`는 lr에 종속(`iters ≈ mult × 12/lr`, mult ∈ {0.7..1.6})으로 샘플 — lr·iter 를
  독립으로 뽑으면 공간 낭비가 크다.

⚠ 함정 등록: 전이 체제에서 용량↑는 붕괴 방향이었다(docs/log/20 — 정직 프로토콜 전부 leaves 7 최적).
탐색이 그걸 재확인하면 받아들인다. 목적은 "용량 키우기"가 아니라 **is4 이후 최적점 재측정**이다.
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
from season_centering import best_cal   # noqa: E402

OUT_DIR = HERE.parent / "results" / "hpo"
FOLDS = ((2023, 2024), (2022, 2023))          # (fit막시즌, val) — june은 fit=단일, allraw는 ≤fit
VERIFY_FOLD = (2021, 2022)                     # 탐색 미사용


def sample_space(rng, target):
    lr = float(np.exp(rng.uniform(np.log(0.008), np.log(0.05))))
    mult = float(rng.uniform(0.7, 1.6))
    iters = int(np.clip(mult * 12.0 / lr, 300, 3500))
    p = dict(
        learning_rate=round(lr, 4),
        num_iterations=iters,
        num_leaves=int(rng.choice([7, 11, 15, 23, 31, 47, 63])),
        min_data_in_leaf=int(rng.choice([300, 500, 800, 1200, 1800, 2500])),
        lambda_l1=float(np.exp(rng.uniform(np.log(0.05), np.log(10)))),
        lambda_l2=float(np.exp(rng.uniform(np.log(0.5), np.log(50)))),
        feature_fraction=round(float(rng.uniform(0.6, 1.0)), 2),
        bagging_fraction=round(float(rng.uniform(0.6, 1.0)), 2),
        bagging_freq=1,
        seed=57,
    )
    if target == "allraw":                     # J3: 6배 데이터 — 용량·표본 하한 상향 허용
        p["min_data_in_leaf"] = int(rng.choice([800, 1500, 3000, 6000, 10000]))
        p["num_leaves"] = int(rng.choice([15, 31, 63, 127]))
    k_is = int(rng.choice([50, 100, 150, 200, 300]))
    return p, k_is


BASELINE = {
    # 현 스펙: june_l15 파라미터 + K=100 (트라이얼 0 = 정합성 앵커)
    "june": (dict(num_leaves=15, min_data_in_leaf=1000, num_iterations=400,
                  learning_rate=0.03, lambda_l1=0.5, lambda_l2=8.0,
                  feature_fraction=0.85, seed=57), 100),
    "allraw": (dict(num_leaves=15, min_data_in_leaf=1000, num_iterations=400,
                    learning_rate=0.03, lambda_l1=0.5, lambda_l2=8.0,
                    feature_fraction=0.85, seed=57), 100),
}


class Data:
    """K_IS별 is_sm/is_delta 재계산을 싸게 하기 위한 준비물."""

    def __init__(self):
        print(">> train.csv 로드")
        self.df = rd.load_train()
        lut = IS.career_end_lookup(self.df)
        self.nb, self.kb = IS.base_for(lut, self.df["pitcher_id"].to_numpy(),
                                       self.df[rd.SEASON].to_numpy())
        L.K_IS = 100.0
        self.X = L.build_features(self.df, base=(self.nb, self.kb))
        self.y = self.df[rd.TARGET].to_numpy().astype(float)
        self.season = self.df[rd.SEASON].to_numpy()
        n = self.df["asof_pitcher_n"].astype("float64").to_numpy()
        rate = self.df["asof_pitcher_success_rate"].fillna(0.0).astype("float64").to_numpy()
        k = np.rint(rate * n)
        self.is_n = np.maximum(n - self.nb, 0.0)
        self.is_k = np.clip(k - self.kb, 0.0, self.is_n)
        self.car_sm = (rate * n + 250.0) / (n + 500.0)
        self._cur_k = 100

    def set_k(self, k_is):
        if k_is == self._cur_k:
            return
        is_sm = (self.is_k + k_is * 0.5) / (self.is_n + k_is)
        self.X["is_sm"] = is_sm.astype("float32")
        self.X["is_delta"] = (is_sm - self.car_sm).astype("float32")
        self._cur_k = k_is


def run_trial(D, params, k_is, target, folds=FOLDS):
    import lightgbm as lgb
    D.set_k(k_is)
    scores = []
    for fit_last, val_s in folds:
        fit = (D.season == fit_last) if target == "june" else (D.season <= fit_last)
        val = D.season == val_s
        pp = {**L.BASE_PARAMS, **params, "num_threads": 6}
        pp.pop("num_iterations", None)
        ds = lgb.Dataset(D.X[fit], label=D.y[fit], free_raw_data=False)
        m = lgb.train(pp, ds, num_boost_round=params["num_iterations"])
        p = np.clip(m.predict(D.X[val], num_threads=6), 0, 1)
        scores.append(best_cal(D.y[val], p)[0])
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["june", "allraw"], default="june")
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", action="store_true",
                    help="저장된 상위 3구성을 val2022(탐색 미사용)로 확인")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = OUT_DIR / f"trials_{args.target}.jsonl"
    t0 = time.time()
    D = Data()

    if args.verify:
        rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
        base = rows[0]
        top = sorted(rows[1:], key=lambda r: -r["maximin"])[:3]
        print(f"[verify] 기준 + 상위 3구성을 val{VERIFY_FOLD[1]}에서 확인 (탐색 미사용 폴드)")
        b22 = run_trial(D, base["params"], base["k_is"], args.target, folds=(VERIFY_FOLD,))[0]
        print(f"  기준     val2022 {b22:8.2f}   ({base['params']})")
        for r in top:
            s22 = run_trial(D, r["params"], r["k_is"], args.target, folds=(VERIFY_FOLD,))[0]
            mark = "✅" if s22 > b22 else "❌ 미사용 폴드에서 역전 — 채택 금지"
            print(f"  trial{r['trial']:3d} val2022 {s22:8.2f}  Δ {s22-b22:+7.2f} {mark}"
                  f"  maximin {r['maximin']:+.2f}  K={r['k_is']}  {r['params']}")
        return

    rng = np.random.default_rng(args.seed)
    base_params, base_k = BASELINE[args.target]
    print(f"[{args.target}] 트라이얼 0 = 현 스펙 (앵커)")
    base_scores = run_trial(D, base_params, base_k, args.target)
    print(f"   기준: " + " ".join(f"val{v}={s:.2f}" for (_, v), s in zip(FOLDS, base_scores)))
    with open(log, "w", encoding="utf-8") as f:
        f.write(json.dumps({"trial": 0, "params": base_params, "k_is": base_k,
                            "scores": base_scores, "maximin": 0.0}) + "\n")

    best = (0.0, 0)
    for t in range(1, args.trials + 1):
        params, k_is = sample_space(rng, args.target)
        tt = time.time()
        try:
            scores = run_trial(D, params, k_is, args.target)
        except Exception as e:
            print(f"  t{t:3d} FAILED {type(e).__name__}: {e}")
            continue
        deltas = [s - b for s, b in zip(scores, base_scores)]
        mm = min(deltas)
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"trial": t, "params": params, "k_is": k_is,
                                "scores": scores, "maximin": mm}) + "\n")
        star = ""
        if mm > best[0]:
            best = (mm, t)
            star = " ★"
        print(f"  t{t:3d} maximin {mm:+7.2f}  (Δ24 {deltas[0]:+7.2f} Δ23 {deltas[1]:+7.2f})  "
              f"lr={params['learning_rate']} it={params['num_iterations']} "
              f"lv={params['num_leaves']} min={params['min_data_in_leaf']} "
              f"K={k_is}  [{time.time()-tt:.0f}s]{star}")

    print(f"\n최고: trial {best[1]} maximin {best[0]:+.2f}  "
          f"({(time.time()-t0)/60:.0f}분, {log})")
    print("다음: --verify 로 val2022 확인 후 채택 판단")


if __name__ == "__main__":
    main()
