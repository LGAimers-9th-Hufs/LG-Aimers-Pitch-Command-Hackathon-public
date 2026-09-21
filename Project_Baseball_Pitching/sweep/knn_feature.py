# -*- coding: utf-8 -*-
"""N32 kNN 잔차 뱅크 **피처형** — 라운드6 미탐색 카드 (보정기형은 D-34 −4.2 킬).

    python sweep/knn_feature.py --vals 2024,2023,2022

## 보정기형과 무엇이 다른가 (knn_residual.py 대비)
보정기형은 `p + w·(Σr_nn)`로 예측에 직접 가산했다 — 트리가 그 신호를 재조정하지 못하고
전 격자 평균 −4.2로 죽었다. **피처형은 kNN 통계를 트리의 입력 컬럼으로 넣어**, 트리가 다른
57피처와의 상호작용 속에서 국소 신호를 쓰도록 한다(Home Credit/Otto 전례, `research/12:34`).
같은 정보의 다른 운반이라 d 벽(0.024~0.029)에 흡수될 공산이 크나 — 저비용 확인.

## kNN 피처 4종 (leave-own-pitcher-out 검색, 거리=57피처 표준화 유클리드)
- knn_r  = 이웃 잔차(y−p_fit) 수축평균 — 트리가 못 잡은 잎 하위 편향
- knn_y  = 이웃 라벨 수축평균 — leave-own-pitcher-out 타깃 인코딩(투수 식별 재학습 차단)
- knn_d  = k-이웃 평균거리 log — 국소 밀도(희소 영역 게이팅)
- knn_rs = 이웃 잔차 std — 국소 불확실성

## 게이트
3폴드(2024/2023/2022) · base=june 2종 57피처 · Δbest(재적합) vs base57 · **위약**(뱅크 잔차·
라벨을 행 사이 셔플 = 피처↔정보 대응만 파괴) · d(예측 RMS vs base) — 블렌드 값은 d≥0.030 필요.
채택 = 3폴드 평균>0 & 평균>SD & 위약분리(edge_min>0). 오라클 아님 — 배포 가능 피처다
(kNN 뱅크 = fit 시즌 고정, 서빙 시 train 동봉 조회 = §5 합법, train-only frozen).
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
from season_centering import best_cal, paired_se, fit_pair, predict_pair   # noqa: E402

OUT_DIR = HERE.parent / "results" / "knn"
K_NN = 200                        # 이웃 수
K_REG = 100.0                     # 수축
CHUNK = 1024
COLS = ["knn_r", "knn_y", "knn_d", "knn_rs"]


def knn_features(Xb, rb, yb, pid_b, Xq, pid_q, k, device):
    """질의별 k-이웃 4통계 [원 순서]. 같은 투수 뱅크 행 제외(leave-own-pitcher-out)."""
    import torch
    tb = torch.tensor(Xb, device=device)
    tr = torch.tensor(rb, device=device)
    ty = torch.tensor(yb, device=device)
    tpb = torch.tensor(pid_b, device=device)
    nb2 = (tb * tb).sum(1)
    out = np.empty((len(Xq), 4), dtype=np.float64)
    for i in range(0, len(Xq), CHUNK):
        q = torch.tensor(Xq[i:i + CHUNK], device=device)
        d2 = (q * q).sum(1, keepdim=True) + nb2[None, :] - 2.0 * (q @ tb.T)
        mask = tpb[None, :] == torch.tensor(pid_q[i:i + CHUNK], device=device)[:, None]
        d2 = d2.masked_fill(mask, float("inf"))
        vals, idx = torch.topk(d2, k, dim=1, largest=False)
        rr = tr[idx]                                          # (C, k)
        yy = ty[idx]
        knn_r = rr.sum(1) / (k + K_REG)
        knn_y = (yy.sum(1) + K_REG * 0.5) / (k + K_REG)
        knn_d = torch.log1p(vals.clamp_min(0).sqrt().mean(1))
        knn_rs = rr.std(1)
        blk = torch.stack([knn_r, knn_y, knn_d, knn_rs], dim=1)
        out[i:i + CHUNK] = blk.cpu().numpy()
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
    X = L.build_features(df, base=(nb, kb)).reset_index(drop=True)
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
        bs = fit_pair(X[fit], y[fit], None)
        p_val = np.clip(predict_pair(bs, X[val]), 0, 1)
        p_fit = np.clip(predict_pair(bs, X[fit]), 0, 1)
        resid = (y[fit] - p_fit).astype(np.float32)
        base_best = best_cal(yv, p_val)[0]
        print(f"    base57 재적합 {base_best:8.2f} · fit 잔차 sd {resid.std():.4f} [{time.time()-t:.0f}s]")

        # 거리 공간: 표준화 57 (fit 통계)
        med = X[fit].median()
        mu = X[fit].fillna(med).mean().to_numpy(dtype=np.float32)
        sd = X[fit].fillna(med).std().replace(0, 1).to_numpy(dtype=np.float32)
        Xb = (X[fit].fillna(med).to_numpy(dtype=np.float32) - mu) / sd
        Xq = (X[val].fillna(med).to_numpy(dtype=np.float32) - mu) / sd
        Xqf = (X[fit].fillna(med).to_numpy(dtype=np.float32) - mu) / sd

        t = time.time()
        # 실제: val·fit 각각 kNN 피처 (fit은 자기 뱅크 조회 = 약한 누수, val은 누수 0 — 판정은 val)
        f_val = knn_features(Xb, resid, y[fit].astype(np.float32), pid[fit], Xq, pid[val], K_NN, device)
        f_fit = knn_features(Xb, resid, y[fit].astype(np.float32), pid[fit], Xqf, pid[fit], K_NN, device)
        # 위약: 뱅크 잔차·라벨을 같은 순열로 섞음(피처↔정보 대응 파괴)
        perm = rng.permutation(fit.sum())
        fs_val = knn_features(Xb, resid[perm], y[fit].astype(np.float32)[perm], pid[fit], Xq, pid[val], K_NN, device)
        fs_fit = knn_features(Xb, resid[perm], y[fit].astype(np.float32)[perm], pid[fit], Xqf, pid[fit], K_NN, device)
        print(f"    kNN 검색 k={K_NN} [{time.time()-t:.0f}s]")

        for tag, Ff, Fv in (("real", f_fit, f_val), ("shuf", fs_fit, fs_val)):
            Xtr = pd.concat([X[fit].reset_index(drop=True),
                             pd.DataFrame(Ff, columns=COLS)], axis=1)
            Xvl = pd.concat([X[val].reset_index(drop=True),
                             pd.DataFrame(Fv, columns=COLS)], axis=1)
            bsk = fit_pair(Xtr, y[fit], None)
            pk = np.clip(predict_pair(bsk, Xvl), 0, 1)
            sk = best_cal(yv, pk)[0]
            dd = float(np.sqrt(((pk - p_val) ** 2).mean()))
            rows.append(dict(val=v, arm=tag, best=round(sk, 2), d_best=round(sk - base_best, 2),
                             d_rms=round(dd, 5), se=round(paired_se(yv, L.calibrate(pk),
                                                                    L.calibrate(p_val)), 2)))
            print(f"    +knn4 [{tag}] 재적합 {sk:8.2f} · Δbest {sk-base_best:+7.2f} · "
                  f"d(base) {dd:.5f}")

    t = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    real = t[t.arm == "real"]
    shuf = t[t.arm == "shuf"]
    dm, dsd, dmin = real.d_best.mean(), real.d_best.std(), real.d_best.min()
    edge = real.d_best.to_numpy() - shuf.d_best.to_numpy()
    dmean_rms = real.d_rms.mean()
    print(f"\n[3폴드 요약] Δbest 평균 {dm:+.2f} · SD {dsd:.2f} · 최소 {dmin:+.2f} · "
          f"위약분리(edge) 평균 {edge.mean():+.2f}/최소 {edge.min():+.2f}")
    print(f"           d(base평균) {dmean_rms:.5f}  (블렌드 값 문턱 0.030)")
    passed = (dm > 0) and (dm > dsd) and (edge.min() > 0)
    print(f"  ⇒ 채택 게이트: {'통과' if passed else '미달'} "
          f"{'· d≥0.030 블렌드 가능' if dmean_rms >= 0.030 else '· d<0.030 블렌드 무익'}")
    (OUT_DIR / "feature_gate.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR/'feature_gate.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
