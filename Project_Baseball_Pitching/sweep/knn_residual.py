# -*- coding: utf-8 -*-
"""H1 — kNN 잔차 뱅크 (N32). 트리의 해상도 한계 **아래** 국소 구조를 검색으로 회수한다.

    python sweep/knn_residual.py --vals 2024,2023,2022

## 왜 이것이 마지막 카드인가 (2026-08-07 저녁 기준)

오라클이 덮은 채널은 전부 포화다: 투수 723.9 · 타자 78.2 · 상황 24.6 · 투수×카운트 510.8 —
**모두 현재 점수(877.8) 미만**. E1(직전 시즌 rate = per-투수 상수)이 위약과 구별되지 않은 것,
트랙맨 프로필(=per-투수-시즌 상수)이 죽은 것 모두 같은 이유다.
**is4가 통한 진짜 이유는 (투수,시즌) 안에서 행마다 변하는 값이었기 때문**이다.

kNN 잔차는 오라클이 덮지 않는 층을 노린다: 우리 트리는 `min_data_in_leaf` 1000~1500이라
**1,000행 미만의 국소 조건부 구조를 원리적으로 표현하지 못한다.** 잎 안에서의 체계적 편향(잔차)이
피처 공간에서 매끄럽다면, 이웃 검색이 그걸 회수한다. 새 알고리즘이 아니라 **같은 데이터에 대한
다른 접근**이다 — 알고리즘 다양성(XGB·CatBoost·NN·GAM 전멸)과 성격이 다르다.

## 설계 (문서 12 §2의 kNN v2 교정 반영)

- **뱅크** = fit 시즌 행들 + 그 시즌에 학습한 베이스 모델의 잔차 `r = y − p`.
  (in-sample 잔차지만 강정칙 트리는 잎 수준 편향을 그대로 남긴다 — 그게 회수 대상이다.)
- **leave-own-pitcher-out**: 질의 행과 같은 `pitcher_id`인 뱅크 행은 제외한다.
  이게 없으면 kNN이 포화된 투수 식별 채널을 재학습할 뿐이다(문서 12 §2 교정 ⑧).
- **하드 오프셋 금지**(269 사태 경로): 보정은 `p + w·(Σr_nn)/(k + K_reg)` — 수축 평균의 저가중 가산.
- **위약**: 뱅크의 잔차를 행 사이에서 섞는다(피처↔잔차 대응만 파괴). 보정 이득이 위약과 같으면 잡음이다.
- 거리 = 57피처 표준화(중앙값 대치) 유클리드, GPU 청크 top-k.

## 게이트

3폴드(2024/2023/2022), 베이스 = june l7+l15(57피처, K=100). (k, w) 격자는 **폴드 간 일관성**으로
판정한다 — 한 폴드에서만 좋은 (k,w)는 선택 낙관이다. 통과 시 ENS 기준선에서 재측정.
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

OUT_DIR = HERE.parent / "results" / "knn"
KS = (100, 400)                    # 이웃 수 격자
WS = (0.25, 0.5, 1.0)              # 보정 가중 격자
K_REG = 100.0                      # 잔차 평균 수축
CHUNK = 1024


def knn_mean_residual(Xb, rb, pid_b, Xq, pid_q, k, device):
    """질의별 k-이웃 잔차의 수축 평균. 같은 투수 뱅크 행은 제외(leave-own-pitcher-out)."""
    import torch
    tb = torch.tensor(Xb, device=device)                     # (B, F)
    tr = torch.tensor(rb, device=device)
    tpb = torch.tensor(pid_b, device=device)
    nb2 = (tb * tb).sum(1)                                   # (B,)
    out = np.empty(len(Xq), dtype=np.float64)
    for i in range(0, len(Xq), CHUNK):
        q = torch.tensor(Xq[i:i + CHUNK], device=device)     # (C, F)
        d2 = (q * q).sum(1, keepdim=True) + nb2[None, :] - 2.0 * (q @ tb.T)
        mask = tpb[None, :] == torch.tensor(pid_q[i:i + CHUNK], device=device)[:, None]
        d2 = d2.masked_fill(mask, float("inf"))
        idx = torch.topk(d2, k, dim=1, largest=False).indices
        s = tr[idx].sum(1)
        out[i:i + CHUNK] = (s / (k + K_REG)).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024,2023,2022")
    ap.add_argument("--k_is", type=float, default=100.0)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">> device={device} · train.csv 로드 (is4 K={args.k_is:g})")
    L.K_IS = args.k_is
    df = rd.load_train()
    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy())
    X = L.build_features(df, base=(nb, kb))
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()
    pid = df["pitcher_id"].to_numpy().astype(np.int64)

    rows = []
    rng = np.random.default_rng(7)
    for v in [int(s) for s in args.vals.split(",")]:
        fit, val = (season == v - 1), (season == v)
        yv = y[val]
        print(f"\n[val {v}]  fit {fit.sum():,} → val {val.sum():,}")

        # 베이스 = june 2종 (57피처)
        t = time.time()
        ps = []
        pf = []
        for name, sp in L.ORIGINAL.items():
            m = L.train_lgbm(X[fit], y[fit], {**sp, "num_threads": 6})
            ps.append(m.predict(X[val], num_threads=6))
            pf.append(m.predict(X[fit], num_threads=6))
        p_val = np.clip(np.mean(ps, axis=0), 0, 1)
        p_fit = np.clip(np.mean(pf, axis=0), 0, 1)
        resid = y[fit] - p_fit
        base_best = best_cal(yv, p_val)[0]
        print(f"    베이스 재적합 {base_best:8.2f}  · fit 잔차 sd {resid.std():.4f}  [{time.time()-t:.0f}s]")

        # 거리 공간: 표준화 57피처 (fit 통계로)
        med = X[fit].median()
        mu = X[fit].fillna(med).mean().to_numpy(dtype=np.float32)
        sd = X[fit].fillna(med).std().replace(0, 1).to_numpy(dtype=np.float32)
        Xb = ((X[fit].fillna(med).to_numpy(dtype=np.float32) - mu) / sd)
        Xq = ((X[val].fillna(med).to_numpy(dtype=np.float32) - mu) / sd)

        r_sh = rng.permutation(resid)                        # 위약: 대응 파괴
        for k in KS:
            t = time.time()
            m_real = knn_mean_residual(Xb, resid.astype(np.float32), pid[fit], Xq, pid[val], k, device)
            m_shuf = knn_mean_residual(Xb, r_sh.astype(np.float32), pid[fit], Xq, pid[val], k, device)
            dt = time.time() - t
            for w in WS:
                s_real = best_cal(yv, np.clip(p_val + w * m_real, 0, 1))[0]
                s_shuf = best_cal(yv, np.clip(p_val + w * m_shuf, 0, 1))[0]
                rows.append(dict(val=v, k=k, w=w, base=base_best,
                                 d_real=s_real - base_best, d_shuf=s_shuf - base_best,
                                 edge=(s_real - s_shuf)))
                print(f"    k={k:4d} w={w:.2f}  Δ실제 {s_real-base_best:+8.2f} · "
                      f"Δ위약 {s_shuf-base_best:+8.2f} · 순수 {s_real-s_shuf:+8.2f}  [{dt:.0f}s]")

    t = pd.DataFrame(rows)
    print("\n" + "=" * 86)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\n[(k,w)별 3폴드 요약 — 순수효과(실제−위약) 기준]")
    g = t.groupby(["k", "w"]).agg(d_mean=("d_real", "mean"), d_sd=("d_real", "std"),
                                  d_min=("d_real", "min"),
                                  edge_mean=("edge", "mean"), edge_min=("edge", "min"))
    g["통과"] = (g.d_mean > 0) & (g.d_mean > g.d_sd) & (g.edge_min > 0)
    print(g.to_string(float_format=lambda v: f"{v:.2f}"))
    (OUT_DIR / "gate.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR/'gate.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
