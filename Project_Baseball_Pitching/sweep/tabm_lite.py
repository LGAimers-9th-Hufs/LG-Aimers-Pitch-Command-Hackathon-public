# -*- coding: utf-8 -*-
"""H2 — TabM-lite (N34). BatchEnsemble식 rank-1 어댑터 k개 = 파라미터 효율적 암묵 앙상블.

    python sweep/tabm_lite.py --vals 2024,2023

## 왜 지금까지 미뤘고 왜 지금 하나

MLP 계열의 단독 품질이 이 데이터에서 143~644로 실측돼(D-30 B1) 사전 기각을 권고했으나,
사용자 원칙("어떤 후보도 하드 제외 금지 — 예상 밖의 결과가 있을 수 있다")에 따라 실행한다.
TabM은 단순 깊은 MLP와 두 가지가 다르다: ①k개 헤드가 가중치를 공유해 정칙화가 세다
②rank-1 곱셈 어댑터가 헤드 다양성을 만든다 — TabReD(시간 시프트 정형 벤치)에서 실증된 구성.

## 구현 (TabM 논문의 핵심만)

- 각 Linear를 BatchEnsemble화: 공유 W + 헤드별 (r_i ∈ R^d_in, s_i ∈ R^d_out, b_i).
  forward: y_i = ((x ⊙ r_i) W^T) ⊙ s_i + b_i. r,s는 ±1 무작위 부호 초기화(논문 방식).
- 헤드 k=8 · 은닉 (256,128) GELU · MSE(각 헤드) · 예측 = 헤드 평균.
- 입력 = nn_member의 전처리(표준화 + 원핫) 재사용. **BatchNorm 없음**(§5).

## 게이트

멤버 후보 기준: **ENS-4 캐시 블렌드 대비** 불일치 s + 저가중 블렌드 이득 (2폴드).
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
import ens5_pool as EP            # noqa: E402
import nn_member as NM            # noqa: E402
from season_centering import best_cal   # noqa: E402

CACHE = HERE.parent / "results" / "ens5"
OUT_DIR = HERE.parent / "results" / "tabm"
WS = (0.05, 0.10, 0.15, 0.20)
K_HEADS = 8


def train_tabm(Z, yv_, Zv, hidden=(256, 128), k=K_HEADS, epochs=60, lr=3e-4,
               wd=1e-4, bs=4096, dropout=0.15, seed=0):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    class BELinear(nn.Module):
        def __init__(self, din, dout):
            super().__init__()
            self.W = nn.Linear(din, dout)                       # 공유
            sign = lambda *sh: (torch.randint(0, 2, sh).float() * 2 - 1)
            self.r = nn.Parameter(sign(k, din))                 # ±1 초기화 (TabM 방식)
            self.s = nn.Parameter(sign(k, dout))
            self.b = nn.Parameter(torch.zeros(k, dout))

        def forward(self, x):                                   # x: (k, B, din)
            return self.W(x * self.r[:, None, :]) * self.s[:, None, :] + self.b[:, None, :]

    class TabM(nn.Module):
        def __init__(self, din):
            super().__init__()
            dims = [din] + list(hidden)
            self.layers = nn.ModuleList([BELinear(dims[i], dims[i + 1])
                                         for i in range(len(hidden))])
            self.act = nn.GELU()
            self.drop = nn.Dropout(dropout)
            self.head = BELinear(dims[-1], 1)
            p0 = float(np.mean(yv_))
            with torch.no_grad():
                self.head.b.fill_(float(np.log(p0 / (1 - p0))))
                self.head.W.weight.mul_(0.1)

        def forward(self, x):                                   # (B, din) → (k, B)
            h = x[None].expand(k, -1, -1)
            for lyr in self.layers:
                h = self.drop(self.act(lyr(h)))
            return torch.sigmoid(self.head(h)).squeeze(-1)

    model = TabM(Z.shape[1]).to(dev)
    Xt = torch.tensor(Z, device=dev)
    Yt = torch.tensor(np.asarray(yv_, dtype=np.float32), device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = int(np.ceil(len(Xt) / bs)) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    n = len(Xt)
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            b = idx[i:i + bs]
            opt.zero_grad(set_to_none=True)
            pred = model(Xt[b])                                 # (k, B)
            loss = ((pred - Yt[b][None, :]) ** 2).mean()        # 각 헤드 MSE
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Zv), 200_000):
            p = model(torch.tensor(Zv[i:i + 200_000], device=dev))
            out.append(p.mean(0).cpu().numpy())                 # 헤드 평균
    return np.clip(np.concatenate(out), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024,2023")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(">> train.csv 로드 (K_IS=100)")
    L.K_IS = 100.0
    df = rd.load_train()
    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy())
    X = L.build_features(df, base=(nb, kb))
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()

    rows = []
    for v in [int(s) for s in args.vals.split(",")]:
        fit, val = (season == v - 1), (season == v)
        yv = y[val]
        z = np.load(CACHE / f"preds_val{v}.npz")
        ens = sum(z[m] * w for m, w in EP.ENS4_W.items())
        base = best_cal(yv, ens)[0]
        print(f"\n[val {v}]  fit {fit.sum():,} → val {val.sum():,}  ENS-4 기준 {base:.2f}")

        st, oh = NM.prep_fit(X[fit]), NM.onehot_fit(X[fit])
        Z = np.concatenate([NM.prep_apply(X[fit], st), NM.onehot_apply(X[fit], oh)], axis=1)
        Zv = np.concatenate([NM.prep_apply(X[val], st), NM.onehot_apply(X[val], oh)], axis=1)

        for tag, kw in (("tabm_256_128", {}),
                        ("tabm_128_64", dict(hidden=(128, 64), dropout=0.1))):
            t = time.time()
            p = train_tabm(Z, y[fit], Zv, **kw)
            s = float(np.sqrt(((p - ens) ** 2).mean()))
            gains = {w: best_cal(yv, (1 - w) * ens + w * p)[0] - base for w in WS}
            bw = max(gains, key=gains.get)
            rows.append(dict(val=v, arm=tag, solo=L.score(yv, L.calibrate(p)), s=s,
                             best_w=bw, gain=gains[bw], sec=round(time.time() - t)))
            print(f"    {tag:14s} 단독 {rows[-1]['solo']:8.2f}  s={s:.4f}  "
                  f"최적w {bw:.2f} → {gains[bw]:+7.2f}  [{rows[-1]['sec']}s]")

    t = pd.DataFrame(rows)
    print("\n" + "=" * 84)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    g = t.groupby("arm").agg(solo=("solo", "mean"), s=("s", "mean"),
                             gain_mean=("gain", "mean"), gain_min=("gain", "min"))
    print("\n[요약 — ENS-4 대비 저가중 이득]")
    print(g.to_string(float_format=lambda v: f"{v:.3f}"))
    (OUT_DIR / "gate.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR/'gate.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
