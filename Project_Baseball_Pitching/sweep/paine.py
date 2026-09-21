# -*- coding: utf-8 -*-
"""PAINE — 자체 설계 신모델 (PLE-Additive-INteraction-Ensemble).

    python sweep/paine.py --stage additive --vals 2024,2023,2022     # P1 가법 백본
    python sweep/paine.py --stage full --pairs 30 --vals 2024,2023,2022   # P2 +상호작용

## 설계 = 이틀간의 실측을 아키텍처로 번역한 것

| 실측 | 결정 |
|---|---|
| 신호 거의 가법적 (선형 805 vs 깊은 MLP 143) | **가법 백본**: 피처별 형상함수 Σf_i(x_i)가 본체 |
| 트리(828) − 선형(805) = +23뿐 | 상호작용은 **선별된 쌍만** — t54 부스터의 분할 경로 동시출현에서 채굴 |
| rank-1 어댑터가 143→750 실증 (tabm_lite) | 검증된 그 부품(BELinear)을 재사용 |
| PLE = 시간시프트 벤치 SOTA 공통 부품 | 수치 입력 전부 PLE (분위수 경계 = fit 상수 → 서빙 동봉, §5 적법) |
| 강정칙화가 전부 | 얕음 + AdamW wd + 가법 경로 지배 |

형상함수 구현이 곧 가법성 보장이다: **PLE 블록별 선형 가중 = 조각별 선형 GAM** —
컨캣된 PLE 위의 단일 선형층이 정확히 Σf_i다(블록 간 곱셈 항이 없으므로).

## 게이트 (변경 없음)

ENS-4 캐시 대비 단독·s·블렌드. P1 기대선 = 선형(805) 초과. 미달 시 그 자리에서 중단·기록.
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
OUT_DIR = HERE.parent / "results" / "paine"
WS = (0.05, 0.10, 0.15, 0.20, 0.30)
T54 = dict(learning_rate=0.0175, num_iterations=726, num_leaves=7, min_data_in_leaf=2500,
           lambda_l1=7.196, lambda_l2=0.774, feature_fraction=0.6,
           bagging_fraction=0.73, bagging_freq=1, seed=57)


# ---------------------------------------------------------------- PLE (fit 상수 → 서빙 동봉 가능)
def ple_fit(Xf: pd.DataFrame, T=16):
    """피처별 분위수 경계 (fit에서만). 반환 {col: edges(np.array, 길이 T+1)}."""
    edges = {}
    qs = np.linspace(0, 1, T + 1)
    for c in Xf.columns:
        v = Xf[c].to_numpy(dtype=np.float64)
        v = v[~np.isnan(v)]
        e = np.quantile(v, qs) if len(v) else np.zeros(T + 1)
        e[0], e[-1] = e[0] - 1e-9, e[-1] + 1e-9
        edges[c] = e.astype(np.float32)
    return edges


def ple_apply(X: pd.DataFrame, edges: dict) -> np.ndarray:
    """표준 PLE(Gorishniy 2022): PLE(x)_t = clip((x−b_{t−1})/(b_t−b_{t−1}), 0, 1).
    NaN → 전부 0 (결측 지시자는 별도 컬럼). 반환 (N, F×T) float32 — 전부 행 단위 산술."""
    outs = []
    for c in X.columns:
        e = edges[c]
        d = np.maximum(e[1:] - e[:-1], 1e-9)                    # (T,)
        x = X[c].to_numpy(dtype=np.float32)[:, None]            # (N,1)
        r = np.clip((x - e[:-1][None, :]) / d[None, :], 0.0, 1.0)
        r[np.isnan(x[:, 0])] = 0.0
        outs.append(r.astype(np.float32))
    return np.concatenate(outs, axis=1)


def build_inputs(Xf, Xv, T=16):
    """PLE(전 57피처) ⊕ 원핫(범주) ⊕ 결측 지시자. 상수는 전부 fit 산출."""
    edges = ple_fit(Xf, T=T)
    oh = NM.onehot_fit(Xf)
    miss_cols = [c for c in Xf.columns if Xf[c].isna().any()]

    def enc(X):
        parts = [ple_apply(X, edges), NM.onehot_apply(X, oh).astype(np.float32)]
        if miss_cols:
            parts.append(X[miss_cols].isna().to_numpy(dtype=np.float32))
        return np.concatenate(parts, axis=1)

    return enc(Xf), enc(Xv), edges


# ---------------------------------------------------------------- 상호작용 쌍 채굴 (t54 부스터)
def mine_pairs(Xf, yf, top_p=30):
    """t54 LGBM의 트리에서 **같은 경로에 함께 등장하는 피처쌍**을 세어 상위 P쌍을 고른다.
    데이터(부스터)가 고른 쌍 — 사람이 고르지 않는다. fit에서만 채굴(폴드별 재채굴)."""
    import lightgbm as lgb
    from collections import Counter
    pp = {**L.BASE_PARAMS, **{k: v for k, v in T54.items() if k != "num_iterations"},
          "num_threads": 6}
    ds = lgb.Dataset(Xf, label=yf, free_raw_data=False)
    m = lgb.train(pp, ds, num_boost_round=T54["num_iterations"])
    cnt = Counter()

    def walk(node, path):
        if "split_feature" not in node:
            return
        f = node["split_feature"]
        for g in path:
            if g != f:
                cnt[tuple(sorted((g, f)))] += 1
        for k in ("left_child", "right_child"):
            if node.get(k):
                walk(node[k], path + [f])

    for tree in m.dump_model()["tree_info"]:
        walk(tree["tree_structure"], [])
    return [p for p, _ in cnt.most_common(top_p)]


# ---------------------------------------------------------------- 모델
def train_paine(Zf, yf, Zv, ple_f, ple_v, pairs_idx, T, k=8, epochs=60, lr=2e-3,
                wd=3e-3, dropout=0.05, bs=8192, seed=0, stage="full"):
    """Zf/Zv = 가법 입력(PLE⊕원핫⊕결측). ple_f/ple_v = (N, 57, T) 텐서용 원본 PLE.
    pairs_idx = [(i,j)] 피처 인덱스 쌍."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    P = len(pairs_idx) if stage == "full" else 0
    H = 8

    class BELinear(nn.Module):
        def __init__(self, din, dout):
            super().__init__()
            self.W = nn.Linear(din, dout)
            sign = lambda *sh: (torch.randint(0, 2, sh).float() * 2 - 1)
            self.r = nn.Parameter(sign(k, din))
            self.s = nn.Parameter(sign(k, dout))
            self.b = nn.Parameter(torch.zeros(k, dout))

        def forward(self, x):                                   # (k,B,din)
            return self.W(x * self.r[:, None, :]) * self.s[:, None, :] + self.b[:, None, :]

    class PAINE(nn.Module):
        def __init__(self, din):
            super().__init__()
            self.add = BELinear(din, 1)                         # Σf_i — 가법 백본(조각별 선형 GAM)
            p0 = float(np.mean(yf))
            with torch.no_grad():
                self.add.b.fill_(float(np.log(p0 / (1 - p0))))
                self.add.W.weight.mul_(0.05)
            self.drop = nn.Dropout(dropout)
            if P:
                self.W1 = nn.Parameter(torch.randn(P, 2 * T, H) * 0.05)
                self.b1 = nn.Parameter(torch.zeros(P, H))
                self.W2 = nn.Parameter(torch.randn(P, H) * 0.05)

        def forward(self, z, xp):                               # z:(B,din) xp:(B,P,2T)
            h = z[None].expand(k, -1, -1)
            logit = self.add(self.drop(h)).squeeze(-1)          # (k,B)
            if P:
                g = torch.einsum("bpt,pth->bph", xp, self.W1) + self.b1[None]
                g = torch.nn.functional.gelu(g)
                inter = torch.einsum("bph,ph->b", g, self.W2)   # (B,) — 헤드 공유
                logit = logit + inter[None, :]
            return torch.sigmoid(logit)

    model = PAINE(Zf.shape[1]).to(dev)
    Xt = torch.tensor(Zf, device=dev)
    Yt = torch.tensor(np.asarray(yf, dtype=np.float32), device=dev)
    if P:
        idx = np.array(pairs_idx)                               # (P,2)
        XPf = np.concatenate([ple_f[:, idx[:, 0], :], ple_f[:, idx[:, 1], :]], axis=2)
        XPv = np.concatenate([ple_v[:, idx[:, 0], :], ple_v[:, idx[:, 1], :]], axis=2)
        XPt = torch.tensor(XPf, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = int(np.ceil(len(Xt) / bs)) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    n = len(Xt)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            pred = model(Xt[b], XPt[b] if P else None)
            loss = ((pred - Yt[b][None, :]) ** 2).mean()
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Zv), 100_000):
            zb = torch.tensor(Zv[i:i + 100_000], device=dev)
            xb = torch.tensor(XPv[i:i + 100_000], device=dev) if P else None
            out.append(model(zb, xb).mean(0).cpu().numpy())
    return np.clip(np.concatenate(out), 1e-6, 1 - 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["additive", "full"], default="additive")
    ap.add_argument("--vals", default="2024,2023,2022")
    ap.add_argument("--pairs", type=int, default=30)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--wd", type=float, default=3e-3)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f">> PAINE stage={args.stage} T={args.T} pairs={args.pairs if args.stage=='full' else 0} "
          f"wd={args.wd} seeds={args.seeds}")
    L.K_IS = 100.0
    df = rd.load_train()
    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy())
    X = L.build_features(df, base=(nb, kb)).reset_index(drop=True)
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()
    F = X.shape[1]

    rows, save = [], {}
    for v in [int(s) for s in args.vals.split(",")]:
        fit, val = (season == v - 1), (season == v)
        yv = y[val]
        z = np.load(CACHE / f"preds_val{v}.npz")
        A = sum(z[m] * w for m, w in EP.ENS4_W.items())
        sA = best_cal(yv, A)[0]
        print(f"\n[val {v}]  fit {fit.sum():,} → val {val.sum():,}  ENS-4 = {sA:.2f}")

        t = time.time()
        Zf, Zv, edges = build_inputs(X[fit], X[val], T=args.T)
        ple_f = ple_apply(X[fit], edges).reshape(fit.sum(), F, args.T)
        ple_v = ple_apply(X[val], edges).reshape(val.sum(), F, args.T)
        print(f"    입력 {Zf.shape[1]}차원 (PLE {F}×{args.T} + 원핫 + 결측)  [{time.time()-t:.0f}s]")

        pairs = []
        if args.stage == "full":
            t = time.time()
            pairs = mine_pairs(X[fit], y[fit], top_p=args.pairs)
            names = list(X.columns)
            print(f"    쌍 채굴 {len(pairs)}개 [{time.time()-t:.0f}s] 상위 5: "
                  + ", ".join(f"({names[i]}×{names[j]})" for i, j in pairs[:5]))

        ps = []
        for sd in range(args.seeds):
            t = time.time()
            p = train_paine(Zf, y[fit], Zv, ple_f, ple_v, pairs, args.T,
                            wd=args.wd, seed=sd, stage=args.stage)
            ps.append(p)
            print(f"    seed {sd}  단독 {best_cal(yv, p)[0]:8.2f}  [{time.time()-t:.0f}s]")
        p = np.mean(ps, axis=0)
        solo = best_cal(yv, p)[0]
        s = float(np.sqrt(((p - A) ** 2).mean()))
        noise = float(np.sqrt(((ps[0] - ps[-1]) ** 2).mean()) / np.sqrt(2)) if len(ps) > 1 else 0.0
        gains = {w: best_cal(yv, (1 - w) * A + w * p)[0] - sA for w in WS}
        bw = max(gains, key=gains.get)
        rows.append(dict(val=v, solo=solo, s=s, noise=noise, best_w=bw, gain=gains[bw],
                         **{f"g{int(w*100):02d}": gains[w] for w in WS}))
        save[v] = dict(p=p, y=yv)
        print(f"    ▶ {args.seeds}시드 평균: 단독 {solo:8.2f} · s(A)={s:.4f} · "
              f"시드노이즈 {noise:.4f} · 블렌드 최적 w{bw:.2f} → {gains[bw]:+.2f}")

    t_ = pd.DataFrame(rows)
    print("\n" + "=" * 96)
    print(t_.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n[대조선] 선형(nn_lin) 단독 ≈ 805(V24) · t54 828 · b_lgb82_a 830 · ENS-4 877.8")
    tag = f"{args.stage}_T{args.T}_p{args.pairs if args.stage=='full' else 0}_wd{args.wd:g}"
    (OUT_DIR / f"gate_{tag}.json").write_text(t_.to_json(orient="records"), encoding="utf-8")
    for v, d in save.items():
        np.savez_compressed(OUT_DIR / f"preds_{tag}_val{v}.npz", **d)
    print(f">> 저장 {OUT_DIR} ({tag})  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
