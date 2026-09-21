# -*- coding: utf-8 -*-
"""B1 — 깊은 NN × **다중 시드 평균**. D-23 기각 사유를 직접 제거한다.

    python sweep/nn_deep.py --seeds 12 --val 2024

## 가설

D-23은 깊은 MLP를 기각했는데 이유가 **성능이 아니라 시드 노이즈**였다:

| 후보 | 단독 | ENS와 불일치 | **시드 노이즈** | 블렌드 이득 |
|---|---|---|---|---|
| nn_lin(선형) | 643 | 0.0256 | **0.0004** | **+26.4** |
| 깊은 MLP 3종 | −194~380 | **0.042~0.056** | **0.019~0.028** | −16~+9 |

즉 깊은 NN은 **다양성의 절반이 학습 노이즈**였다. 그런데 노이즈는 **평균으로 지울 수 있고**
(시드 k개 평균 → 노이즈 1/√k) **불일치는 남는다**. 10시드면 0.025 → 0.008이다.
⇒ 기각 사유만 제거되고 s ≈ 0.045가 살아남는다면, 이건 우리가 가진 **가장 큰 s의 원천**이다.

**이 검증이 가설의 생사를 가른다**: 시드 노이즈가 실제로 1/√k로 줄어드는가.
줄지 않으면(= 노이즈가 아니라 다중해(multi-modality)라면) 평균은 s까지 같이 지워버린다.

## 측정 방법

시드 K개를 학습해 **서로 겹치지 않는 두 그룹**(각 k개)으로 나눠 평균한다.
두 그룹 평균의 RMS = √2 × (k-평균의 시드 노이즈) ⇒ `noise(k) = RMS/√2`.
동시에 k-평균의 **단독 점수**와 **ENS-2와의 불일치 s**, 저가중 블렌드 이득을 잰다.

⚠ 서빙 호환: `train_mlp`은 Linear/GELU/Dropout만 쓴다 — **BatchNorm 없음**(§5 배치통계 금지 준수).
Dropout은 eval 모드에서 비활성이므로 추론은 순수 행 단위 사상이다.
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
import season_centering as SC     # noqa: E402
import phase2_ens_check as EC     # noqa: E402
import nn_member as NM            # noqa: E402

OUT_DIR = HERE.parent / "results" / "programB"
WS = (0.05, 0.10, 0.15, 0.20, 0.30)

CONFIGS = {
    # D-23에서 불일치가 가장 컸던 계열. dropout/폭을 키워 s를 유지하되 시드평균으로 노이즈를 지운다.
    "deep_256_128": dict(hidden=(256, 128), dropout=0.15, epochs=60, lr=3e-4, wd=1e-4),
    "deep_512_256_128": dict(hidden=(512, 256, 128), dropout=0.20, epochs=60, lr=3e-4, wd=1e-4),
}


def train_seeds(Z, yf, Zv, spec, n_seeds):
    ps = []
    for s in range(n_seeds):
        t = time.time()
        model, dev = NM.train_mlp(Z, yf, seed=s, **spec)
        ps.append(np.clip(NM.predict_mlp(model, dev, Zv), 0, 1))
        del model
        print(f"      seed {s:2d}  {time.time()-t:5.1f}s")
    return np.array(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=int, default=2024)
    ap.add_argument("--seeds", type=int, default=12, help="총 학습 시드 수(짝수)")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(">> train.csv 로드")
    df = rd.load_train()
    X = L.build_features(df)
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()
    fit = (season == args.val - 1)
    val = (season == args.val)
    yv = y[val]
    print(f"   fit {fit.sum():,} → val {val.sum():,}  r={yv.mean():.4f}")

    print(">> 기준 ENS-2 재구성 (정합성 앵커: val2024 = 773.48)")
    P5 = EC.train_members(X, y, fit, val)
    ens = sum(P5[n] * w for n, _, w, _ in EC.MEMBERS)
    ens_best = SC.best_cal(yv, ens)[0]
    print(f"   ENS-2 캘리재적합 {ens_best:.2f}")

    Xf = X[fit]
    st, oh = NM.prep_fit(Xf), NM.onehot_fit(Xf)
    Z = np.concatenate([NM.prep_apply(Xf, st), NM.onehot_apply(Xf, oh)], axis=1)
    Zv = np.concatenate([NM.prep_apply(X[val], st), NM.onehot_apply(X[val], oh)], axis=1)
    print(f"   NN 입력 {Z.shape[1]}차원 (원핫 포함)")

    rows, keep = [], {"_ens2": ens}
    for cname, spec in CONFIGS.items():
        print(f"\n[{cname}] 시드 {args.seeds}개 학습")
        A = train_seeds(Z, y[fit], Zv, spec, args.seeds)
        keep[cname] = A.mean(axis=0)

        # k-평균의 시드 노이즈 = 서로 겹치지 않는 두 그룹 평균의 RMS / √2
        print(f"  {'k':>3s} {'단독':>9s} {'s(vs ENS2)':>11s} {'시드노이즈':>11s} "
              f"{'1/√k 예측':>10s} {'최적w':>6s} {'이득':>8s}")
        base_noise = None
        for k in [k for k in (1, 2, 3, 6) if 2 * k <= args.seeds]:
            g1, g2 = A[:k].mean(axis=0), A[k:2 * k].mean(axis=0)
            noise = float(np.sqrt(((g1 - g2) ** 2).mean()) / np.sqrt(2))
            if base_noise is None:
                base_noise = noise
            p = A[:2 * k].mean(axis=0)
            s = float(np.sqrt(((p - ens) ** 2).mean()))
            gains = {w: SC.best_cal(yv, (1 - w) * ens + w * p)[0] - ens_best for w in WS}
            bw = max(gains, key=gains.get)
            rows.append(dict(config=cname, k=k, solo=L.score(yv, L.calibrate(p)), s=s,
                             noise=noise, pred=base_noise / np.sqrt(k),
                             best_w=bw, gain=gains[bw],
                             **{f"g{int(w*100):02d}": gains[w] for w in WS}))
            r = rows[-1]
            print(f"  {k:3d} {r['solo']:9.2f} {s:11.4f} {noise:11.4f} "
                  f"{r['pred']:10.4f} {bw:6.2f} {r['gain']:+8.2f}")

    t = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n기준 ENS-2 = {ens_best:.2f}")
    print("\n[판정] '시드노이즈' 열이 '1/√k 예측' 열을 따라가면 가설이 맞다.")
    print("       따라가면서 s가 유지되면 → 깊은 NN은 우리가 가진 최대 s 원천이다.")
    print("       s까지 같이 줄면 → 다양성이 노이즈였다는 뜻이고 D-23 기각이 옳았다.")

    np.savez_compressed(OUT_DIR / f"nn_deep_val{args.val}.npz", y=yv, **keep)
    (OUT_DIR / f"nn_deep_val{args.val}.json").write_text(
        json.dumps({"ens_best": ens_best, "rows": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
