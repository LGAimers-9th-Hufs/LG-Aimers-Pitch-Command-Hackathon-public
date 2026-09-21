# -*- coding: utf-8 -*-
"""NN 다양성 멤버 탐색 — 트리 일색인 앙상블에 다른 편향 구조를 넣을 수 있는가.

    python sweep/nn_member.py --scan            # NN 구성 스캔 (fit 2023 → val 2024)
    python sweep/nn_member.py --greedy          # 기존 5멤버 + NN 후보로 그리디 재구성

## 왜 NN인가 (그리고 왜 보장은 없는가)

지금 앙상블 멤버는 **전부 트리**(LGBM 2 + ExtraTrees 3)다. 블렌드 이득은
`Σ wᵢwⱼ·mean((pᵢ−pⱼ)²)`로 결정되므로 **다른 방식으로 틀리는** 멤버가 필요한데,
문서 16 §3의 실측이 그 창이 좁다는 걸 보여준다:

| 후보 | 단독 점수 | 기준과의 불일치 | 블렌드 이득 |
|---|---|---|---|
| ExtraTrees | 685.5 (동급) | 0.0219 | **+30.8** |
| sk_hgb | 692.1 (최고) | **0.0089** (너무 닮음) | +1.1 |
| lg_dart | −196.4 (붕괴) | **0.0522** (너무 다름) | −36.2 |

NN은 **구간상수(트리)가 아니라 매끄러운 함수**라 구조적으로 다르게 틀릴 여지가 크지만,
신호가 약한 판(BSS ~0.9%)에서 단독 성능이 무너지면 `lg_dart`처럼 해롭다.
**그래서 추측하지 않고 같은 프로토콜에서 잰다 — 제출 슬롯이 들지 않는다.**
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import real_data as rd        # noqa: E402
import lgbm_family as L       # noqa: E402


# 저카디널리티 범주형 — 정수 그대로 MLP에 넣으면 **순서형으로 오해**한다(v1 실패의 주원인).
ONEHOT = ["top_bottom", "game_type", "base_state", "count_code", "game_month",
          "game_dayofweek", "inning", "outs_before", "balls_before", "strikes_before",
          "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id",
          "num_runners_on"]


def onehot_fit(Xtr: pd.DataFrame, cols=ONEHOT):
    """학습 구간에서 본 범주만 고정 — 미지 값은 전부 0 벡터(서빙에서 상수로 동봉 가능)."""
    return {c: np.array(sorted(pd.unique(Xtr[c].dropna()))) for c in cols if c in Xtr}


def onehot_apply(X: pd.DataFrame, maps: dict) -> np.ndarray:
    blocks = []
    for c, vals in maps.items():
        v = X[c].to_numpy()
        blocks.append((v[:, None] == vals[None, :]).astype(np.float32))
    return np.concatenate(blocks, axis=1) if blocks else np.zeros((len(X), 0), dtype=np.float32)


def prep_fit(Xtr: pd.DataFrame):
    """표준화 + 결측 지시자. 통계는 **학습 구간에서만** 산출해 얼린다(서빙 시 상수로 동봉)."""
    A = Xtr.to_numpy(dtype=np.float64)
    miss_cols = np.where(np.isnan(A).any(axis=0))[0]
    med = np.nanmedian(A, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    B = np.where(np.isnan(A), med, A)
    mu, sd = B.mean(axis=0), B.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return {"median": med, "mean": mu, "scale": sd, "miss_cols": miss_cols}


def prep_apply(X: pd.DataFrame, st: dict) -> np.ndarray:
    A = X.to_numpy(dtype=np.float64)
    nanmask = np.isnan(A)
    B = np.where(nanmask, st["median"], A)
    Z = (B - st["mean"]) / st["scale"]
    if len(st["miss_cols"]):
        Z = np.concatenate([Z, nanmask[:, st["miss_cols"]].astype(np.float64)], axis=1)
    return Z.astype(np.float32)


def train_mlp(Ztr, ytr, hidden=(256, 128), dropout=0.15, epochs=25, lr=1e-3,
              bs=4096, wd=1e-4, seed=0, device=None, verbose=False):
    """확률 출력 + **MSE 손실** — 대회 지표가 곧 제곱오차이므로 Brier를 직접 최소화한다."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    layers, d = [], Ztr.shape[1]
    for h in hidden:
        layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]
        d = h
    head = nn.Linear(d, 1)
    p0 = float(np.mean(ytr))
    with torch.no_grad():                      # 출력 바이어스를 기저율 로짓으로 초기화
        head.bias.fill_(float(np.log(p0 / (1 - p0))))
        head.weight.mul_(0.1)
    model = nn.Sequential(*layers, head, nn.Sigmoid()).to(dev)

    X = torch.tensor(Ztr, device=dev)
    Y = torch.tensor(np.asarray(ytr, dtype=np.float32).reshape(-1, 1), device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = int(np.ceil(len(X) / bs)) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    lossf = nn.MSELoss()
    n = len(X)
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, bs):
            b = idx[i:i + bs]
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(X[b]), Y[b])
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(b)
        if verbose and (ep + 1) % 5 == 0:
            print(f"      ep{ep+1:>3d} train MSE {tot/n:.6f}")
    model.eval()
    return model, dev


def predict_mlp(model, dev, Z, bs=200_000):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(Z), bs):
            out.append(model(torch.tensor(Z[i:i + bs], device=dev)).cpu().numpy().ravel())
    return np.concatenate(out)


CONFIGS = [
    # v2: 원핫 + 더 긴 학습 + 낮은 lr. v1(정수 범주형·짧은 학습)은 전부 붕괴했다.
    ("nn_128_64_e60",   dict(hidden=(128, 64), dropout=0.10, epochs=60, lr=3e-4)),
    ("nn_256_128_e60",  dict(hidden=(256, 128), dropout=0.15, epochs=60, lr=3e-4)),
    ("nn_64_32_e80",    dict(hidden=(64, 32), dropout=0.05, epochs=80, lr=3e-4, wd=1e-3)),
    ("nn_lin",          dict(hidden=(), dropout=0.0, epochs=60, lr=1e-3, wd=1e-4)),
    ("nn_256_128_e25",  dict(hidden=(256, 128), dropout=0.15, epochs=25)),
]


def run(df, fit_seasons=(2023,), val_season=2024, seeds=(0, 1, 2), greedy=False):
    X = L.build_features(df)
    y = df[rd.TARGET].to_numpy().astype(float)
    fit = df[rd.SEASON].isin(list(fit_seasons)).to_numpy()
    val = (df[rd.SEASON] == val_season).to_numpy()
    yv = y[val]

    st = prep_fit(X[fit])
    oh = onehot_fit(X[fit])
    Ztr = np.concatenate([prep_apply(X[fit], st), onehot_apply(X[fit], oh)], axis=1)
    Zv = np.concatenate([prep_apply(X[val], st), onehot_apply(X[val], oh)], axis=1)
    print(f"입력 {Ztr.shape[1]}차원 (수치+결측지시자 {len(st['miss_cols'])} + "
          f"원핫 {sum(len(v) for v in oh.values())}) · 학습 {fit.sum():,} → val {val.sum():,}")

    # 기준: june l7+l15
    ref_ps = [L.train_lgbm(X[fit], y[fit], sp).predict(X[val], num_threads=4)
              for sp in L.ORIGINAL.values()]
    ref = np.mean(ref_ps, axis=0)
    print(f"[기준] june l7+l15 cal = {L.score(yv, L.calibrate(ref)):.2f}\n")

    preds = {}
    for name, kw in CONFIGS:
        ps = []
        for s in seeds:
            m, dev = train_mlp(Ztr, y[fit], seed=s, **kw)
            ps.append(predict_mlp(m, dev, Zv))
        p = np.clip(np.mean(ps, axis=0), 0, 1)
        preds[name] = p
        seed_rms = float(np.sqrt(((ps[0] - ps[1]) ** 2).mean())) if len(ps) > 1 else float("nan")
        print(f"  {name:16s} cal={L.score(yv, L.calibrate(p)):8.2f}  raw={L.score(yv, p):8.2f}  "
              f"rms_vs_ref={np.sqrt(((p-ref)**2).mean()):.4f}  (시드간 rms {seed_rms:.4f})")

    print("\n[기준 + NN 2원 블렌드]")
    base = L.score(yv, L.calibrate(ref))
    for name, p in preds.items():
        row = [f"w{w}:{L.score(yv, L.calibrate((1-w)*ref + w*p)):7.1f}" for w in (0.2, 0.3, 0.4, 0.5)]
        best = max(L.score(yv, L.calibrate((1-w)*ref + w*p)) for w in (0.2, 0.3, 0.4, 0.5))
        print(f"  {name:16s} " + "  ".join(row) + f"   최대이득 {best-base:+7.1f}")

    if greedy:
        print("\n[그리디] 현 5멤버 + NN 후보 전체")
        from sklearn.ensemble import ExtraTreesRegressor
        Xn = np.nan_to_num(X[fit].to_numpy(dtype=np.float32), nan=-999.0)
        Xvn = np.nan_to_num(X[val].to_numpy(dtype=np.float32), nan=-999.0)
        pool = {"june_l7": np.clip(ref_ps[0], 0, 1), "june_l15": np.clip(ref_ps[1], 0, 1)}
        for nm, kw in [("et_l100_d28", dict(n_estimators=200, min_samples_leaf=100,
                                            max_depth=28, max_features=0.7)),
                       ("et_l200_d20", dict(n_estimators=200, min_samples_leaf=200,
                                            max_depth=20, max_features=1.0)),
                       ("et_l400_f05", dict(n_estimators=200, min_samples_leaf=400,
                                            max_depth=20, max_features=0.5))]:
            m = ExtraTreesRegressor(n_jobs=-1, random_state=9, **kw).fit(Xn, y[fit])
            pool[nm] = np.clip(m.predict(Xvn), 0, 1)
        base_pool = dict(pool)
        w0, s0, _, f0 = L.greedy_ensemble(base_pool, yv, init=["june_l7", "june_l15"])
        pool.update(preds)
        w1, s1, _, f1 = L.greedy_ensemble(pool, yv, init=["june_l7", "june_l15"])

        def best_cal(p):
            return max(L.score(yv, L.calibrate(p, slope=sl, shift=sh))
                       for sl in np.arange(0.85, 1.15, 0.01) for sh in np.arange(-0.02, 0.011, 0.005))
        print(f"  NN 제외: {s0:.2f} → 캘리 재적합 {best_cal(f0):.2f}")
        print("     " + " ".join(f"{k}={v:.2f}" for k, v in sorted(w0.items(), key=lambda kv: -kv[1])))
        print(f"  NN 포함: {s1:.2f} → 캘리 재적합 {best_cal(f1):.2f}")
        print("     " + " ".join(f"{k}={v:.2f}" for k, v in sorted(w1.items(), key=lambda kv: -kv[1])))
        print(f"\n  ⇒ NN 추가 이득 = {best_cal(f1) - best_cal(f0):+.2f}")


def finalize(df, fit_seasons=(2023,), val_season=2024):
    """배포할 구성을 확정한다 — 6멤버(선형까지) vs 7멤버(+MLP)의 실제 차이를 재고,
    **배포와 동일한 형태**(선형은 단일 시드)로 측정한다. 측정한 것과 배포한 것이 달랐던 게
    BLEND-1 사고의 본질이므로 여기서부터 어긋나지 않게 한다."""
    from sklearn.ensemble import ExtraTreesRegressor
    X = L.build_features(df)
    y = df[rd.TARGET].to_numpy().astype(float)
    fit = df[rd.SEASON].isin(list(fit_seasons)).to_numpy()
    val = (df[rd.SEASON] == val_season).to_numpy()
    yv = y[val]

    pool = {}
    for nm, sp in L.ORIGINAL.items():
        m = L.train_lgbm(X[fit], y[fit], sp)
        pool[f"june_{nm}"] = np.clip(m.predict(X[val], num_threads=4), 0, 1)
    Xn = np.nan_to_num(X[fit].to_numpy(dtype=np.float32), nan=-999.0)
    Xvn = np.nan_to_num(X[val].to_numpy(dtype=np.float32), nan=-999.0)
    for nm, kw in [("et_l100_d28", dict(n_estimators=200, min_samples_leaf=100,
                                        max_depth=28, max_features=0.7)),
                   ("et_l200_d20", dict(n_estimators=200, min_samples_leaf=200,
                                        max_depth=20, max_features=1.0)),
                   ("et_l400_f05", dict(n_estimators=200, min_samples_leaf=400,
                                        max_depth=20, max_features=0.5))]:
        pool[nm] = np.clip(ExtraTreesRegressor(n_jobs=-1, random_state=9, **kw)
                           .fit(Xn, y[fit]).predict(Xvn), 0, 1)

    st, oh = prep_fit(X[fit]), onehot_fit(X[fit])
    Ztr = np.concatenate([prep_apply(X[fit], st), onehot_apply(X[fit], oh)], axis=1)
    Zv = np.concatenate([prep_apply(X[val], st), onehot_apply(X[val], oh)], axis=1)
    m, dev = train_mlp(Ztr, y[fit], hidden=(), dropout=0.0, epochs=60, lr=1e-3, wd=1e-4, seed=0)
    pool["nn_lin"] = np.clip(predict_mlp(m, dev, Zv), 0, 1)          # 배포와 동일: 단일 시드
    m2, dev2 = train_mlp(Ztr, y[fit], hidden=(128, 64), dropout=0.10,
                         epochs=60, lr=3e-4, seed=0)
    pool["nn_mlp"] = np.clip(predict_mlp(m2, dev2, Zv), 0, 1)

    def best_cal(p):
        b = max(((L.score(yv, L.calibrate(p, slope=sl, shift=sh)), round(sl, 3), round(sh, 4))
                 for sl in np.arange(0.85, 1.16, 0.01)
                 for sh in np.arange(-0.02, 0.011, 0.0025)))
        return b

    print(f"\n{'구성':<10s} {'그리디':>9s} {'캘리재적합':>11s}   가중")
    out = {}
    for tag, names in (("5멤버", ["june_l7", "june_l15", "et_l100_d28", "et_l200_d20", "et_l400_f05"]),
                       ("6멤버", ["june_l7", "june_l15", "et_l100_d28", "et_l200_d20",
                                  "et_l400_f05", "nn_lin"]),
                       ("7멤버", list(pool))):
        sub = {n: pool[n] for n in names}
        w, s, _, final = L.greedy_ensemble(sub, yv, init=["june_l7", "june_l15"])
        sc, sl, sh = best_cal(final)
        out[tag] = (w, sc, sl, sh)
        ws = " ".join(f"{k}={v:.3f}" for k, v in sorted(w.items(), key=lambda kv: -kv[1]))
        print(f"{tag:<10s} {s:9.2f} {sc:11.2f}   (slope {sl}, shift {sh})\n           {ws}")

    d = out["7멤버"][1] - out["6멤버"][1]
    print(f"\n7멤버 − 6멤버 = {d:+.2f}  →  " +
          ("MLP 유지 권장" if d >= 2.0 else "**6멤버 채택**(MLP 기여 < 2점, 서빙 복잡도만 늘린다)"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finalize", action="store_true",
                    help="배포 구성 확정: 5/6/7멤버 비교 + 가중·캘리 출력")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--fit", default="2023")
    ap.add_argument("--val", type=int, default=2024)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    print(">> train.csv 로드")
    df = rd.load_train()
    fits = [int(s) for s in args.fit.split(",")]
    if args.scan or args.greedy:
        run(df, fit_seasons=fits, val_season=args.val,
            seeds=tuple(range(args.seeds)), greedy=args.greedy)
    if args.finalize:
        finalize(df, fit_seasons=fits, val_season=args.val)


if __name__ == "__main__":
    main()
