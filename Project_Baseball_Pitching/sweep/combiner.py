# -*- coding: utf-8 -*-
"""결합기(combiner) 개선 — 멤버·피처가 소진된 뒤 유일하게 덜 판 층.

    python sweep/combiner.py

## 왜 여기인가

우리 결합기는 가장 단순한 형태다: **고정 볼록 가중 + 아핀 캘리**. 그런데,

1. 멤버들의 강점이 **행마다 다르다**는 실측이 있다 — `all_raw`(전 시즌 학습)는 저경험 투수·파단
   레짐에서 강하고 june 계열은 그 반대다. 고정 가중은 이를 못 쓴다. 행별 가중(mixture-of-experts)은
   **트리가 구조적으로 못 하는 일**이다(멤버 예측을 입력으로 못 본다) — is4와 같은 종류의 접근 불가층.
2. `score = 1e5·Var(p)/V`인데 산포 조절 수단이 slope 하나뿐이었다. 신뢰도 곡선 기울기가 구간마다
   다르면 조각별 단조 맵이 산포를 더 뽑는다. isotonic(−274)은 과적합 극단이고 저자유도 버전은 미시험.

## 방법 — 전부 캐시(`results/ens5/preds_val*.npz`)에서, 재학습 없음

**교차 폴드 전이 검증이 핵심이다**: 결합기를 폴드 A(멤버는 A−1 학습 인스턴스)에서 학습해
폴드 B(다른 인스턴스)에 적용한다. 서빙이 정확히 이 상황이다(결합기는 2024 예측으로 학습되지만
서빙 멤버는 2024 학습 인스턴스).

- **S1 선형 스태킹**: ridge(멤버 예측 → y). 고정 가중의 일반화(음수·비볼록 허용).
- **S2 게이트 스태킹(MoE)**: 멤버 예측 × 게이트(is_logn 3버킷·game_type·is_share) 상호작용항.
  행별 유효 가중 = 게이트의 선형 함수. 서빙은 동봉 계수의 행 단위 내적 — §5 적법.
- **C1 조각별 단조 캘리**: ENS-4 블렌드에 2~3 매듭 조각별 선형(단조 제약) 맵을 폴드 A에서 적합,
  폴드 B에서 아핀 재적합 대비 이득을 잰다.

## 판정

적용 폴드에서 **ENS-4 고정 가중 + 아핀 재적합** 대비 Δ. 두 방향(2023→2024, 2022+2023→2024)과
역방향(→2023)을 모두 본다. 제출 규칙(D-32): 합쳐서 로컬 +10 미만이면 채택하지 않는다.
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
from season_centering import best_cal   # noqa: E402

CACHE = HERE.parent / "results" / "ens5"
OUT_DIR = HERE.parent / "results" / "combiner"
MEMBERS = list(EP.ENS4_W)                       # ENS-4의 5멤버만 쓴다(풀 확대는 F에서 기각됨)


def load_fold(v):
    z = np.load(CACHE / f"preds_val{v}.npz")
    P = np.stack([z[m] for m in MEMBERS], axis=1)          # (N, 5)
    return P, z["y"]


def gates_for(v, df, X):
    """게이트 피처(행 단위): is_logn 버킷 2 + is_share + game_type(F 여부). 전부 §5 적법."""
    season = df[rd.SEASON].to_numpy()
    m = season == v
    g_logn = X["is_logn"].to_numpy()[m]
    g_share = X["is_share"].to_numpy()[m]
    gt = X["game_type"].to_numpy()[m]
    return np.stack([
        (g_logn < 4.0).astype(float),                       # 저경험 (당해 <55구)
        (g_logn >= 6.5).astype(float),                      # 고경험
        g_share,
        (gt != 1).astype(float),                            # game_type != R
    ], axis=1)


def ridge_fit(A, y, lam):
    G = A.T @ A + lam * np.eye(A.shape[1])
    return np.linalg.solve(G, A.T @ y)


def design_s1(P):
    return np.concatenate([P, np.ones((len(P), 1))], axis=1)


def design_s2(P, G):
    inter = np.concatenate([P * G[:, [j]] for j in range(G.shape[1])], axis=1)
    return np.concatenate([P, inter, G, np.ones((len(P), 1))], axis=1)


def pwl_fit(p, y, knots):
    """조각별 선형 단조 캘리: 기저 = [1, p, relu(p−k1), relu(p−k2), ...]. 최소제곱 후 단조 검사."""
    B = [np.ones_like(p), p] + [np.maximum(p - k, 0.0) for k in knots]
    A = np.stack(B, axis=1)
    c = np.linalg.lstsq(A, y, rcond=None)[0]
    slopes = np.cumsum(c[1:])                                # 구간별 기울기
    return c, bool(np.all(slopes > 0))


def pwl_apply(p, c, knots):
    out = c[0] + c[1] * p
    for j, k in enumerate(knots):
        out = out + c[2 + j] * np.maximum(p - k, 0.0)
    return np.clip(out, 1e-6, 1 - 1e-6)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(">> 캐시 로드 + 게이트 피처 생성")
    L.K_IS = 100.0
    df = rd.load_train()
    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy())
    X = L.build_features(df, base=(nb, kb))

    D = {}
    for v in (2022, 2023, 2024):
        P, y = load_fold(v)
        D[v] = dict(P=P, y=y, G=gates_for(v, df, X))
        w = np.array([EP.ENS4_W[m] for m in MEMBERS])
        ens = P @ w
        D[v]["ens"] = ens
        D[v]["base"] = best_cal(y, ens)[0]
        print(f"   val {v}: {len(y):,}행 · ENS-4 고정가중+아핀 = {D[v]['base']:.2f}")

    results = []
    pairs = [((2023,), 2024), ((2022, 2023), 2024), ((2022, 2024), 2023), ((2024,), 2023)]
    for train_vs, test_v in pairs:
        Ptr = np.concatenate([D[v]["P"] for v in train_vs])
        ytr = np.concatenate([D[v]["y"] for v in train_vs])
        Gtr = np.concatenate([D[v]["G"] for v in train_vs])
        te = D[test_v]
        tag = f"{'+'.join(map(str, train_vs))}→{test_v}"
        base = te["base"]

        for name, dfn in (("S1 선형스택", design_s1), ("S2 게이트스택", None)):
            for lam in (1.0, 100.0, 10000.0):
                if name.startswith("S1"):
                    Atr, Ate = design_s1(Ptr), design_s1(te["P"])
                else:
                    Atr, Ate = design_s2(Ptr, Gtr), design_s2(te["P"], te["G"])
                c = ridge_fit(Atr, ytr, lam)
                s = best_cal(te["y"], np.clip(Ate @ c, 1e-6, 1 - 1e-6))[0]
                results.append(dict(cfg=f"{name} λ={lam:g}", pair=tag, d=s - base))

        for knots in ((0.45, 0.55), (0.40, 0.50, 0.60)):
            c, mono = pwl_fit(np.concatenate([D[v]["ens"] for v in train_vs]), ytr, knots)
            if not mono:
                results.append(dict(cfg=f"C1 pwl{len(knots)}", pair=tag, d=np.nan))
                continue
            s = L.score(te["y"], pwl_apply(te["ens"], c, knots))
            results.append(dict(cfg=f"C1 pwl{len(knots)}", pair=tag, d=s - base))

    t = pd.DataFrame(results)
    piv = t.pivot_table(index="cfg", columns="pair", values="d")
    piv["worst"] = piv.min(axis=1)
    piv["mean"] = piv.mean(axis=1)
    print("\n" + "=" * 96)
    print("[기준 대비 Δ — 결합기를 다른 폴드에서 학습해 적용(모델 인스턴스 전이 포함)]")
    print(piv.sort_values("worst", ascending=False).to_string(float_format=lambda v: f"{v:+.2f}"))
    print(f"\n  판정: worst > 0 이고 →2024 이득 ≥ +10 이어야 슬롯 후보 (D-32/D-34 규칙)")
    (OUT_DIR / "gate.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    print(f">> 저장 {OUT_DIR/'gate.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
