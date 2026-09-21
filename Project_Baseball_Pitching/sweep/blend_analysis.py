# -*- coding: utf-8 -*-
"""멤버별 사전 캘리브레이션이 다양성을 되살리는지 — 재학습 없이 저장된 예측으로 판정.

    python sweep/blend_analysis.py

## 발견한 오류

Program A/B의 모든 후보가 이득 ≈ 0이었다(A1 +0.7 · A2 +2.2 · A7 −2.8 · B1 +0.3).
그런데 블렌드 항등식은 다른 답을 준다. 정확한 항등식:

    score_mix = (1−w)·S_a + w·S_b + 1e5·w(1−w)·E[(a−b)²] / V

깊은 NN(s=0.046)을 넣으면 최적 w≈0.13에서 **+13.9**가 나와야 하는데 실측은 −4.6이었다.

**원인: `S_b`의 정의 불일치.** 리포트의 "단독" 점수는 **캘리 적용 후** 값인데,
`ENS-2`는 멤버들의 **원시 예측**을 가중평균한 뒤 마지막에 한 번만 캘리한다.
스케일이 어긋난 멤버(깊은 NN·GAM은 평균·분산이 크게 다르다)는 원시로 섞이는 순간
그 어긋남을 블렌드에 주입한다 — **다양성이 아니라 편향이 먼저 들어간다.**

⇒ 가설: **멤버를 각자 캘리한 뒤 섞으면** 항등식이 약속한 다양성 이득이 실제로 나온다.
(june853/ENS-2가 "마지막에 한 번만 캘리"한 것은 멤버들이 전부 같은 계열이라 스케일이 비슷했기 때문이고,
계열을 넓히는 순간 그 전제가 깨진다.)

## 방법

`results/programA/*.npz`, `results/programB/*.npz`에 저장된 val 2024 예측을 그대로 쓴다(재학습 0).
멤버마다 affine (slope, shift)를 val에서 적합한 뒤 블렌드해 **원시 블렌드와 직접 대조**한다.

⚠ 멤버별 캘리를 val에서 고르므로 선택 낙관이 붙는다(멤버당 2파라미터).
여기서 살아남는 것만 `--val 2023` 재현으로 확인한다.
"""
from __future__ import annotations
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
import lgbm_family as L           # noqa: E402
from season_centering import best_cal   # noqa: E402

ROOT = HERE.parent / "results"
WS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)


def affine_fit(y, p):
    """멤버 개별 affine 캘리: p* = 0.5 + a(p − 0.5) + b. 최소제곱 해를 격자 없이 닫힌형으로."""
    p = np.asarray(p, dtype=float)
    x = p - 0.5
    # min_a,b  E[(0.5 + a·x + b − y)²]  → 1차 회귀
    A = np.stack([x, np.ones_like(x)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y - 0.5, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    return a, b


def apply_affine(p, a, b):
    return np.clip(0.5 + a * (np.asarray(p) - 0.5) + b, 1e-6, 1 - 1e-6)


def load_all(val=2024):
    files = sorted(ROOT.glob(f"program*/**/*val{val}*.npz")) + \
            sorted(ROOT.glob(f"program*/*val{val}*.npz"))
    seen, members, ens, y = set(), {}, None, None
    for f in dict.fromkeys(files):
        z = np.load(f)
        if y is None:
            y = z["y"]
        if "_ens2" in z.files:
            e = z["_ens2"]
            if ens is None:
                ens = e
            else:
                r = float(np.sqrt(((e - ens) ** 2).mean()))
                if r > 1e-9:
                    print(f"   ⚠ {f.name}의 _ens2가 다르다 (RMS {r:.2e}) — 배관 확인 필요")
        for k in z.files:
            if k in ("y", "_ens2") or k in seen:
                continue
            seen.add(k)
            members[k] = z[k]
        print(f"   {f.parent.name}/{f.name}: {len(z.files)-2}멤버")
    return y, ens, members


def main():
    val = 2024
    print(f">> 저장된 val {val} 예측 로드")
    y, ens, M = load_all(val)
    if ens is None:
        sys.exit("_ens2 예측이 없다")
    ens_best = best_cal(y, ens)[0]
    print(f"\n기준 ENS-2 = {ens_best:.2f}  ·  멤버 {len(M)}개  ·  val {len(y):,}행")

    V = float(y.mean() * (1 - y.mean()))
    rows = []
    for nm, p in M.items():
        a, b = affine_fit(y, p)
        pc = apply_affine(p, a, b)
        s_raw = float(np.sqrt(((p - ens) ** 2).mean()))
        s_cal = float(np.sqrt(((pc - ens) ** 2).mean()))
        g_raw = max(best_cal(y, (1 - w) * ens + w * p)[0] - ens_best for w in WS)
        gains = {w: best_cal(y, (1 - w) * ens + w * pc)[0] - ens_best for w in WS}
        bw = max(gains, key=gains.get)
        # 항등식 예측(원시 스케일 기준): 최적 w에서의 이론 이득
        S_b = L.score(y, np.clip(p, 1e-6, 1 - 1e-6))
        S_bc = L.score(y, pc)
        rows.append(dict(name=nm, solo_raw=S_b, solo_cal=S_bc, a=a, b=b,
                         s_raw=s_raw, s_cal=s_cal, gain_raw=g_raw,
                         gain_cal=gains[bw], w=bw,
                         ident=max(w * ((S_bc - ens_best) + (1 - w) * 1e5 * s_cal ** 2 / V)
                                   for w in WS)))

    t = pd.DataFrame(rows).sort_values("gain_cal", ascending=False)
    print("\n" + "=" * 118)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n[핵심 대조] 원시 블렌드 이득 vs 멤버별 캘리 후 이득")
    c = t[["name", "solo_raw", "solo_cal", "s_cal", "gain_raw", "gain_cal", "w", "ident"]]
    print(c.head(12).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\n  원시 최대 {t.gain_raw.max():+.2f}  →  캘리 후 최대 {t.gain_cal.max():+.2f}")

    # ---- 그리디: 캘리된 멤버들로 ENS-2 위에 쌓는다
    print("\n[그리디] ENS-2 + 캘리된 멤버 누적 (val 2024에서 선택 — 낙관 포함)")
    cal = {r["name"]: apply_affine(M[r["name"]], r["a"], r["b"]) for _, r in t.iterrows()}
    cur, wsum, picks = ens.copy(), 1.0, []
    best = ens_best
    for it in range(12):
        cand = None
        for nm, p in cal.items():
            for w in (0.05, 0.10, 0.15):
                q = (cur * wsum + p * w) / (wsum + w)
                sc = best_cal(y, q)[0]
                if cand is None or sc > cand[0]:
                    cand = (sc, nm, w, q)
        if cand[0] <= best + 1e-6:
            break
        best, nm, w, cur = cand[0], cand[1], cand[2], cand[3]
        wsum += w
        picks.append((nm, w, best))
        print(f"   +{nm:16s} w={w:.2f}  →  {best:8.2f}  ({best-ens_best:+.2f})")
    print(f"\n  최종 {best:.2f}  (ENS-2 {ens_best:.2f}, 이득 {best-ens_best:+.2f})")
    print("  ⚠ 이 수치는 val 2024에서 멤버·가중·캘리를 전부 고른 값이다 — 낙관 포함,")
    print("    val 2023 절차 재현으로 낙관 크기를 재기 전에는 제출 판단에 쓰지 않는다.")


if __name__ == "__main__":
    main()
