# -*- coding: utf-8 -*-
"""L1 — **표준형** 시즌 내 OOF 스태킹. combiner의 실패와 다른 형태임을 명확히 하고 재심한다.

    python sweep/oof_stack.py

## combiner(기각)와 무엇이 다른가

combiner는 **교차 인스턴스**였다: 2023 예측(멤버 = 2022 학습 인스턴스)으로 메타를 배워
2024 예측(2023 인스턴스)에 적용 → 메타가 폴드 레짐을 학습해 폭주(−16~−931).

표준형(Kaggle 스태킹의 정의)은 **같은 인스턴스**를 본다:
1. 학습 시즌 S 안에서 **월 블록 5-fold**로 멤버 OOF 예측 생성 (시간 블록 = 경기 누수 차단)
2. 메타(ridge)를 S의 OOF 예측 + S 라벨로 학습
3. 멤버를 S 전량으로 재적합 → S+1 예측 → **메타(멤버 예측)** vs 고정 가중 블렌드 비교

`allraw` 멤버는 S 미포함(<S 학습)이므로 그 예측이 이미 OOF다 — 재폴드 불필요.
서빙 레그(S 전량 재적합 멤버의 S+1 예측)는 `results/ens5/preds_val*.npz` 캐시를 재사용한다.

## 게이트

S=2023→2024와 S=2022→2023 두 실험. Δ = 같은 멤버의 ENS-4 고정 가중 + 아핀 재적합 대비.
combiner식 징후(한 방향만 개선)가 보이면 즉시 기각. λ는 {1,100,10000} 전부 보고(선택 낙관 방지).
"""
from __future__ import annotations
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
OUT_DIR = HERE.parent / "results" / "oof_stack"
MEMBERS = list(EP.ENS4_W)          # nn_lin, et_l100_d28, allraw_l15, june_l15, june_l7
MONTH_BLOCKS = [(3, 4), (5, 5), (6, 6), (7, 8), (9, 10)]


def train_member_preds(name, X, y, fit_mask, pred_mask):
    """단일 멤버를 fit_mask로 학습해 pred_mask 행을 예측."""
    from sklearn.ensemble import ExtraTreesRegressor
    if name in ("june_l15", "june_l7"):
        sp = L.ORIGINAL["l15" if name.endswith("l15") else "l7"]
        m = L.train_lgbm(X[fit_mask], y[fit_mask], {**sp, "num_threads": 6})
        return np.clip(m.predict(X[pred_mask], num_threads=6), 0, 1)
    if name == "et_l100_d28":
        Xn = np.nan_to_num(X[fit_mask].to_numpy(dtype=np.float32), nan=-999.0)
        Xv = np.nan_to_num(X[pred_mask].to_numpy(dtype=np.float32), nan=-999.0)
        m = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=100, max_depth=28,
                                max_features=0.7, n_jobs=6, random_state=9).fit(Xn, y[fit_mask])
        return np.clip(m.predict(Xv), 0, 1)
    if name == "nn_lin":
        st, oh = NM.prep_fit(X[fit_mask]), NM.onehot_fit(X[fit_mask])
        Z = np.concatenate([NM.prep_apply(X[fit_mask], st), NM.onehot_apply(X[fit_mask], oh)], axis=1)
        model, dev = NM.train_mlp(Z, y[fit_mask], hidden=(), dropout=0.0, epochs=60,
                                  lr=1e-3, wd=1e-4, seed=0)
        Zv = np.concatenate([NM.prep_apply(X[pred_mask], st), NM.onehot_apply(X[pred_mask], oh)], axis=1)
        return np.clip(NM.predict_mlp(model, dev, Zv), 0, 1)
    raise ValueError(name)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(">> train.csv 로드 (K_IS=100)")
    L.K_IS = 100.0
    df = rd.load_train()
    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy())
    X = L.build_features(df, base=(nb, kb)).reset_index(drop=True)
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()
    month = df["game_month"].to_numpy()

    results = []
    for S, T in ((2023, 2024), (2022, 2023)):
        print(f"\n[S={S} → T={T}]")
        inS = season == S
        # --- 1) S 안 월 블록 OOF (시즌 학습 멤버 4종) ---------------------------
        OOF = {m: np.full(int(inS.sum()), np.nan) for m in MEMBERS}
        idxS = np.where(inS)[0]
        pos = {g: i for i, g in enumerate(idxS)}
        # allraw는 <S 학습이라 S 전체가 이미 OOF
        fit_all = season < S
        sp = L.ORIGINAL["l15"]
        m_all = L.train_lgbm(X[fit_all], y[fit_all], {**sp, "num_threads": 6})
        OOF["allraw_l15"] = np.clip(m_all.predict(X[inS], num_threads=6), 0, 1)
        for lo, hi in MONTH_BLOCKS:
            held = inS & (month >= lo) & (month <= hi)
            tr = inS & ~held
            hpos = np.array([pos[g] for g in np.where(held)[0]])
            if len(hpos) == 0:
                continue
            for nm in ("june_l15", "june_l7", "et_l100_d28", "nn_lin"):
                OOF[nm][hpos] = train_member_preds(nm, X, y, tr, held)
            print(f"    OOF 블록 {lo}-{hi}월: {held.sum():,}행 완료 [{time.time()-t0:.0f}s]")
        A_oof = np.stack([OOF[m] for m in MEMBERS], axis=1)
        yS = y[inS]

        # --- 2) 메타 학습 (ridge, λ 전부 보고) --------------------------------
        D = np.concatenate([A_oof, np.ones((len(A_oof), 1))], axis=1)
        metas = {}
        for lam in (1.0, 100.0, 10000.0):
            G = D.T @ D + lam * np.eye(D.shape[1])
            metas[lam] = np.linalg.solve(G, D.T @ yS)

        # --- 3) 서빙 레그 = 캐시 (S 전량 학습 멤버의 T 예측) --------------------
        z = np.load(CACHE / f"preds_val{T}.npz")
        A_srv = np.stack([z[m] for m in MEMBERS], axis=1)
        yT = z["y"]
        ens_fixed = A_srv @ np.array([EP.ENS4_W[m] for m in MEMBERS])
        base = best_cal(yT, ens_fixed)[0]
        print(f"    기준(고정 가중+아핀) {base:8.2f}")
        for lam, c in metas.items():
            p = np.clip(np.concatenate([A_srv, np.ones((len(A_srv), 1))], axis=1) @ c,
                        1e-6, 1 - 1e-6)
            s = best_cal(yT, p)[0]
            results.append(dict(S=S, T=T, lam=lam, d=s - base,
                                w=[round(float(v), 3) for v in c]))
            print(f"    메타 λ={lam:<7g} Δ {s-base:+8.2f}   가중 {results[-1]['w']}")

    t = pd.DataFrame(results)
    print("\n" + "=" * 80)
    piv = t.pivot_table(index="lam", columns="T", values="d")
    piv["worst"] = piv.min(axis=1)
    print(piv.to_string(float_format=lambda v: f"{v:+.2f}"))
    print("\n  판정: 두 방향 모두 양수여야 채택 후보. 한 방향만 개선 = combiner식 레짐 학습 → 기각.")
    (OUT_DIR / "gate.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    print(f">> 저장 {OUT_DIR/'gate.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
