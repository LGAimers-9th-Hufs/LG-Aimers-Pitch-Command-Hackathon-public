# -*- coding: utf-8 -*-
"""이중조사 스크린 2 — TM 프로필 **form 교체**를 배포 동형 보정기(31계수·λ1000·w0.5)로.

    python sweep/tm_form_carrier.py --vals 2024            # V24 스크린
    python sweep/tm_form_carrier.py --vals 2024,2023,2022,2021

## 근거 (이중조사 병합 후보 — Codex #2 처방 2)

- 현 ENS-7 보정기는 career_rank **하나만** 사용. season_raw/season_rank/career_raw/
  t12_season_rank는 **트리 직삽에서만** 기각됐고(D-25) 보정기 운반으로는 미측정.
- 실험 = **교체**: 캐시된 corr(career_rank) 성분을 form별 재구축 corr로 갈아끼운 증분.
  career_rank 재구축 arm이 하네스 파리티 검사를 겸한다(캐시와 RMS 대조).

## 게이트 (D-42 개정판)

ENS-7 가중·캘리·멤버 동결 · 주 판정 = 동결 배포 캘리 · 잔차 타깃 = june_fit_resid(배포 동형)
· 위약 = **커버 보존 경로 교환**(tm_clean 방식 — 매칭된 투수 집합 안에서 프로필 경로 통째 교환) 2시드.
"""
from __future__ import annotations
import argparse
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
import tm_member as TMM           # noqa: E402
from eb_carrier import get_members, deploy_score, ENS7_W, TM_W  # noqa: E402

OUT_DIR = HERE.parent / "results" / "dual"
FORMS = ["career_rank", "season_rank", "career_raw", "season_raw", "t12_season_rank"]
WGRID = (0.25, 0.5, 1.0)


def cover_preserving_placebo(prof, seed):
    """매칭된 투수 집합 안에서 프로필 경로를 통째 교환 — 커버 마스크 불변 (tm_clean 방식)."""
    pids = sorted({p for (p, _s) in prof})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pids))
    pmap = {pids[i]: pids[perm[i]] for i in range(len(pids))}
    return {(p, s): prof.get((pmap[p], s), v) for (p, s), v in prof.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024")
    ap.add_argument("--lam", type=float, default=1000.0)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(">> 준비 (K_IS=100)")
    L.K_IS = 100.0
    df = rd.load_train()
    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), df[rd.SEASON].to_numpy())
    X = L.build_features(df, base=(nb, kb)).reset_index(drop=True)
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()

    profiles = {f: TMM.load_profile(form=f) for f in FORMS}
    best_form = "season_rank"                       # 위약은 최우선 대안 form에만 (스크린)
    placebos = {f"{best_form}_pl{sd}": cover_preserving_placebo(profiles[best_form], sd)
                for sd in range(2)}

    res = []
    for v in [int(s) for s in args.vals.split(",")]:
        fit, val = (season == v - 1), (season == v)
        P, corr_cached, yv = get_members(df, X, y, season, v)
        ens_wo = sum(P[m] * w for m, w in ENS7_W.items())
        s_ref = deploy_score(yv, ens_wo + TM_W * corr_cached)
        s_nocorr = deploy_score(yv, ens_wo)
        t1 = time.time()
        resid = TMM.june_fit_resid(X, y, fit)
        print(f"\n[val {v}] ENS-7 기준 {s_ref:.2f} (corr 제거 시 {s_nocorr - s_ref:+.2f}) "
              f"· fit 잔차 sd {resid.std():.4f} [{time.time()-t1:.0f}s]")

        for tag, prof in {**profiles, **placebos}.items():
            Df, covf = TMM.design(df, X, prof, fit)
            Dv, covv = TMM.design(df, X, prof, val)
            A = Df[fit & covf]
            G = A.T @ A + args.lam * np.eye(A.shape[1])
            c = np.linalg.solve(G, A.T @ resid[covf[fit]])
            corr = Dv @ c
            corr[~covv] = 0.0
            corr_v = corr[val]
            line = {"val": v, "tag": tag, "cover": float(covv[val].mean())}
            if tag == "career_rank":
                line["rms_vs_cache"] = float(np.sqrt(np.mean((corr_v - corr_cached) ** 2)))
            for w in WGRID:
                line[f"w{w:g}"] = deploy_score(yv, ens_wo + w * corr_v) - s_ref
            res.append(line)
            extra = f"  [캐시 RMS {line['rms_vs_cache']:.2e}]" if "rms_vs_cache" in line else ""
            print(f"  {tag:20s} cover {line['cover']:.3f}  " +
                  "  ".join(f"w{w:g}: {line[f'w{w:g}']:+.2f}" for w in WGRID) + extra)

    t = pd.DataFrame(res)
    print("\n" + t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    core = t[(~t.tag.str.contains("_pl")) & t.val.isin([2024, 2023, 2022])]
    if len(core):
        g = core.groupby("tag").agg(d_mean=("w0.5", "mean"), d_sd=("w0.5", "std"),
                                    d_min=("w0.5", "min"),
                                    pos=("w0.5", lambda s: int((s > 0).sum())), n=("w0.5", "size"))
        if g["n"].max() >= 3:
            g["통과"] = (g.d_mean > 0) & (g.d_mean > g.d_sd)
        print("\n[요약 w0.5 — V24/23/22 (V21 감사 제외)]")
        print(g.sort_values("d_mean", ascending=False).to_string(float_format=lambda x: f"{x:.2f}"))
    (OUT_DIR / "tm_form_gate.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR/'tm_form_gate.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
