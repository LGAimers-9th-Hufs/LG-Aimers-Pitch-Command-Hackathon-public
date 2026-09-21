# -*- coding: utf-8 -*-
"""month 분할축 프로브 6종의 배관·비용 검증 (제출 전 필수).

    python submission/check_month_probe.py --rows 40000

검사 3종
  ① 방향이 실제로 실렸는가 — δ_j = p(d_j) − p(null) 이 0이 아니고, **월별로 정확히 상수**인가
     (`map` 항은 game_month에만 의존하므로 같은 월의 모든 행에서 δ가 동일해야 한다)
  ② δ_j 가 동봉 map 값과 **비트 수준으로 일치**하는가 (구현 오독·스케일 사고 차단)
  ③ 무신호 비용 = `C·mean(δ²)` 이 **1점**인가 — 역산식 `a_j = ΔS_j + 1`의 전제
     ⚠ mean(δ²)는 평가셋 월 분포에 의존한다. 2025를 볼 수 없으므로 train 각 시즌으로 감도를 본다.
"""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "submission" / "dist"
sys.path.insert(0, str(ROOT / "sweep"))
C_SCORE = 4e5                    # 1e5/(r(1−r)), r≈0.494
NDIR = 6


def design_maps():
    """설계 원본(`_deferred`, t=1.0)의 월별 오프셋. **최종 출력 δ가 이 값과 같아야 한다** —
    빌더가 t에 1/w_champ 을 곱해두고 서빙에서 w_champ 배로 희석되므로 정확히 상쇄된다."""
    out = {}
    for j in range(1, NDIR + 1):
        with zipfile.ZipFile(DIST / "_deferred" / ("submit_mo_d%d.zip" % j)) as z:
            m = json.loads(z.read("model/metadata.json").decode("utf-8"))
        t = [x for x in m["seg_probe"] if x.get("col") == "game_month"][0]
        tt = float(t.get("t", 1.0))
        out[j] = {str(k): float(v) * tt for k, v in t["map"].items()}
    return out


def design_weights():
    """partition_design 의 셀 가중(무신호 비용 1.0의 기준 분포). m4 = 3월+4월."""
    p = ROOT / "results" / "codex_research" / "partition_design_month_m34.json"
    w = json.loads(p.read_text(encoding="utf-8"))["weights"]
    out = {}
    for cell, v in w.items():
        m = cell[1:]
        out["3" if m == "4" else m] = float(v)      # m4 셀은 3·4월 합산 가중
    return out


def run_zip(tag: str, zp: Path, sub: pd.DataFrame, tmp: Path, id_col: str, target: str):
    work = tmp / tag
    work.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp) as z:
        z.extractall(work)
    (work / "data").mkdir(exist_ok=True)
    test = sub.drop(columns=[target])
    test.to_csv(work / "data" / "test.csv", index=False, encoding="utf-8")
    pd.DataFrame({id_col: test[id_col], target: 0.5}).to_csv(
        work / "data" / "sample_submission.csv", index=False, encoding="utf-8")
    r = subprocess.run([sys.executable, "script.py"], cwd=work, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=3600)
    if r.returncode != 0 or not (work / "output" / "submission.csv").exists():
        print("   !! 실패\n   " + (r.stderr or "")[-600:].replace("\n", "\n   "))
        return None
    for ln in r.stdout.splitlines():
        if "!!" in ln:
            print("   ⚠ " + ln)
    return pd.read_csv(work / "output" / "submission.csv")[target].to_numpy(dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=40000)
    args = ap.parse_args()

    import real_data as rd
    df = rd.load_train()
    # ⚠ head()는 시간순이라 3~4월만 뽑힌다 — 월 분할축 검증에 치명적이다. 무작위 표본을 쓴다.
    pool = df[df[rd.SEASON] == 2024]
    sub = pool.sample(n=min(args.rows, len(pool)), random_state=42).reset_index(drop=True)
    month = sub["game_month"].astype(str).to_numpy()
    sh = pd.Series(month).value_counts(normalize=True).sort_index(key=lambda s: s.astype(int))
    print(">> 프록시 2024 %s행 (무작위)" % f"{len(sub):,}")
    print(">> 월 분포: %s\n" % " ".join("%s=%.3f" % (m, v) for m, v in sh.items()))

    MAPS = design_maps()
    tmp = Path(tempfile.mkdtemp(prefix="mochk_"))
    try:
        print("[run ] null")
        p0 = run_zip("null", DIST / "submit_moJ_null.zip", sub, tmp, rd.ID, rd.TARGET)
        if p0 is None:
            return
        print()
        print("=" * 78)
        print("① 방향 적재 · ② map 일치")
        print("=" * 78)
        print("%-5s %12s %12s %14s %10s" % ("방향", "δ RMS", "δ 최대", "월별 상수?", "map 일치?"))
        print("-" * 78)
        deltas = {}
        for j in range(1, NDIR + 1):
            pj = run_zip("d%d" % j, DIST / ("submit_moJ_d%d.zip" % j), sub, tmp, rd.ID, rd.TARGET)
            if pj is None:
                continue
            d = pj - p0
            deltas[j] = d
            mp = MAPS[j]
            const_err = max(float(np.ptp(d[month == m])) for m in set(month))
            exp = np.array([mp.get(m, 0.0) for m in month])
            map_err = float(np.abs(d - exp).max())
            print("%-5s %12.3e %12.3e %14.2e %10.2e  %s"
                  % ("d%d" % j, float(np.sqrt((d ** 2).mean())), float(np.abs(d).max()),
                     const_err, map_err,
                     "✓" if (const_err < 1e-12 and map_err < 1e-12) else "🚫"))

        print()
        print("=" * 78)
        print("③ 무신호 비용 = C·mean(δ²) — 역산식 `a_j = ΔS_j + 1`의 전제")
        print("=" * 78)
        seasons = [2021, 2022, 2023, 2024]
        dw = design_weights()
        cols = ["설계가중"] + ["%d" % s for s in seasons]
        print("%-5s %s" % ("방향", "".join("%12s" % c for c in cols)))
        print("-" * 78)
        shares = {"설계가중": dw}
        for s in seasons:
            shares["%d" % s] = (df[df[rd.SEASON] == s]["game_month"].astype(str)
                                .value_counts(normalize=True).to_dict())
        for j in range(1, NDIR + 1):
            mp = MAPS[j]
            row = []
            for c in cols:
                sh = shares[c]
                m2 = sum(sh.get(k, 0.0) * v * v for k, v in mp.items())
                row.append("%12.4f" % (C_SCORE * m2))
            print("%-5s %s" % ("d%d" % j, "".join(row)))
        print("\n  '설계가중' 열이 1.0이면 설계가 의도대로다. 나머지 열은 평가셋 월 분포가")
        print("  설계와 다를 때 비용이 얼마나 흔들리는지를 보여준다 — 역산식")
        print("  `a_j = ΔS_j + 비용`의 '비용' 불확실성이 그만큼이다.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
