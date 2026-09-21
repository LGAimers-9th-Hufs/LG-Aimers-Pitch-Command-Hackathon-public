# -*- coding: utf-8 -*-
"""C3 — 트랙맨 T-b **실제 실행**: 프로필을 주 트리에 넣지 않고, TM×맥락 상호작용의
별도 잔차 보정기로 운반한다. (Codex 지적 ② — 이 운반안은 명시돼 있었으나 실행된 적이 없다.)

    python sweep/tm_member.py --vals 2024,2023,2022

## 설계

- 프로필: `results/phase3/profiles.json`의 **career_rank**(셔플 대비 +35.2로 신호가 가장 컸던 형태),
  시즌 s 행 ← (pid, s−1). 커버율 ~0.68.
- 보정기: 커버된 fit 행에서 june쌍 in-sample 잔차 `r = y − p_june`를
  **ridge([TM6 ⊗ (1, count, hand_match, is_sm, fb_rate)] = 30항)**으로 적합.
  → TM 상수가 맥락과 곱해져 **행마다 변하는 값**이 된다(판별 기준 통과).
- 적용: `p_new = ens + w·corr` (커버 행만, 비커버는 ens 그대로 → ΔENS에 희석 없음 명시).
- **위약 = 시즌 내 pid↔프로필 셔플 매칭**을 같은 운반기로 — Δinfo(진짜−셔플)가 판정.
- 종료 조건: ENS 증분 +10 미만이면 트랙맨을 **실행 후** 종료(이번엔 진짜로).
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
from season_centering import best_cal   # noqa: E402

CACHE = HERE.parent / "results" / "ens5"
PROF = HERE.parent / "results" / "phase3" / "profiles.json"
OUT_DIR = HERE.parent / "results" / "tm_member"
TM6 = ["tm_movestd", "tm_velostd", "tm_velo_fb", "tm_farm_share", "tm_spin_fb", "tm_velosep"]
WGRID = (0.25, 0.5, 1.0)


def load_profile(form="career_rank", shuffle_seed=None):
    obj = json.loads(PROF.read_text(encoding="utf-8"))[form]
    prof = {(int(k.split("|")[0]), int(k.split("|")[1])): v for k, v in obj.items()}
    if shuffle_seed is not None:                      # 시즌 내 pid↔프로필 대응 파괴
        rng = np.random.default_rng(shuffle_seed)
        by_season = {}
        for (p, s), v in prof.items():
            by_season.setdefault(s, []).append((p, v))
        out = {}
        for s, items in by_season.items():
            pids = [p for p, _ in items]
            vals = [v for _, v in items]
            for p, j in zip(pids, rng.permutation(len(vals))):
                out[(p, s)] = vals[j]
        return out
    return prof


def design(df, X, prof, rows):
    """커버 행의 [TM6 ⊗ (1, count, hand, is_sm, fb)] 30항 + 절편. 반환 (D, cov마스크)."""
    pid = df["pitcher_id"].to_numpy().astype(int)
    ssn = df[rd.SEASON].to_numpy().astype(int)
    M = np.full((len(df), 6), np.nan)
    for i in np.where(rows)[0]:
        v = prof.get((pid[i], ssn[i] - 1))
        if v is not None:
            M[i] = v
    cov = rows & ~np.isnan(M).any(axis=1)
    ctx = np.stack([
        np.ones(len(df)),
        (df["balls_before"].to_numpy() * 3 + df["strikes_before"].to_numpy()) / 11.0,
        (df["pitcher_hand"].to_numpy() == df["batter_hand"].to_numpy()).astype(float),
        np.nan_to_num(X["is_sm"].to_numpy(dtype=float), nan=0.5) - 0.5,
        np.nan_to_num(df["asof_pitcher_fastball_rate"].to_numpy(dtype=float), nan=0.5) - 0.5,
    ], axis=1)                                        # (N, 5)
    Mc = np.nan_to_num(M, nan=0.0) - 0.5              # rank 프로필은 0~1 → 중심화
    D = np.einsum("nf,nc->nfc", Mc, ctx).reshape(len(df), -1)   # (N, 30)
    D = np.concatenate([D, np.ones((len(df), 1))], axis=1)
    return D, cov


def june_fit_resid(X, y, fit):
    ps = []
    for name, sp in L.ORIGINAL.items():
        m = L.train_lgbm(X[fit], y[fit], {**sp, "num_threads": 6})
        ps.append(m.predict(X[fit], num_threads=6))
    return y[fit] - np.clip(np.mean(ps, axis=0), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024,2023,2022")
    ap.add_argument("--lam", type=float, default=1000.0)
    args = ap.parse_args()
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

    profiles = {"real": load_profile(), "shuf0": load_profile(shuffle_seed=0),
                "shuf1": load_profile(shuffle_seed=1)}

    res = []
    for v in [int(s) for s in args.vals.split(",")]:
        fit, val = (season == v - 1), (season == v)
        yv = y[val]
        z = np.load(CACHE / f"preds_val{v}.npz")
        ens = sum(z[m] * w for m, w in EP.ENS4_W.items())
        s_ref = best_cal(yv, ens)[0]
        t = time.time()
        resid = june_fit_resid(X, y, fit)
        print(f"\n[val {v}]  ENS-4 기준 {s_ref:.2f} · fit 잔차 sd {resid.std():.4f} [{time.time()-t:.0f}s]")

        for tag, prof in profiles.items():
            Df, covf = design(df, X, prof, fit)
            Dv, covv = design(df, X, prof, val)
            # 잔차(resid)는 fit 행 순서로 정렬돼 있다 → 커버 부분만: resid[covf[fit]]
            A = Df[fit & covf]
            r = resid[covf[fit]]
            G = A.T @ A + args.lam * np.eye(A.shape[1])
            c = np.linalg.solve(G, A.T @ r)
            corr = Dv @ c
            corr[~covv] = 0.0
            corr_v = corr[val]
            line = {"val": v, "tag": tag, "cover": float(covv[val].mean())}
            for w in WGRID:
                p = np.clip(ens + w * corr_v, 1e-6, 1 - 1e-6)
                line[f"w{w:g}"] = best_cal(yv, p)[0] - s_ref
            res.append(line)
            print(f"  {tag:6s} cover {line['cover']:.2f}  " +
                  "  ".join(f"w{w:g}: {line[f'w{w:g}']:+.2f}" for w in WGRID))

    t = pd.DataFrame(res)
    print("\n" + t.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\n[Δinfo = real − shuffle 평균]")
    for w in WGRID:
        k = f"w{w:g}"
        piv = t.pivot_table(index="val", columns="tag", values=k)
        piv["info"] = piv["real"] - (piv["shuf0"] + piv["shuf1"]) / 2
        print(f"  w={w:g}: " + "  ".join(f"V{v} {piv.loc[v,'info']:+.2f}" for v in piv.index)
              + f"   real평균 {piv['real'].mean():+.2f}")
    (OUT_DIR / "gate.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    print(f"\n({time.time()-t0:.0f}s)  종료 조건: ENS 증분 +10 미만이면 트랙맨 최종 종료")


if __name__ == "__main__":
    main()
