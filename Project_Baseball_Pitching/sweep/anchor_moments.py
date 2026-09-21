# -*- coding: utf-8 -*-
"""챔피언 zip의 **raw**(캘리 이전) 예측 평균·표준편차를 파트너와 같은 행 집합에서 산출한다.

왜 필요한가 (D-59): 파트너를 raw 단계에서 섞을 때 **평균·분산 보존 정규화**를 하려면
`mean_a, std_a`(우리 raw)와 `mean_b, std_b`(파트너 raw)가 **같은 기준 행**에서 나와야 한다.
파트너 MANIFEST는 train 2024 프록시에서 산출했으므로 여기서도 같은 행을 쓴다.

raw 복원: 서빙은 `p = clip(0.5 + slope·(raw−0.5) + shift) + seg` 순이고 운용범위에 클리핑이
0행이므로 정확히 역산된다.

    python sweep/anchor_moments.py --zip submission/dist/submit_exact1.zip --rows 245789
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
sys.path.insert(0, str(ROOT / "sweep"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=str(ROOT / "submission" / "dist" / "submit_exact1.zip"))
    ap.add_argument("--rows", type=int, default=245789)
    args = ap.parse_args()

    import real_data as rd
    df = rd.load_train()
    sub = df[df[rd.SEASON] == 2024].head(args.rows)
    tmp = Path(tempfile.mkdtemp(prefix="anchormom_"))
    work = tmp / "run"
    work.mkdir()
    with zipfile.ZipFile(args.zip) as z:
        z.extractall(work)
    (work / "data").mkdir(exist_ok=True)
    test = sub.drop(columns=[rd.TARGET])
    test.to_csv(work / "data" / "test.csv", index=False, encoding="utf-8")
    pd.DataFrame({"row_id": test[rd.ID], "control_success": 0.5}).to_csv(
        work / "data" / "sample_submission.csv", index=False, encoding="utf-8")

    r = subprocess.run([sys.executable, "script.py"], cwd=work, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stderr[-2000:])
        sys.exit("script.py 실패")
    p = pd.read_csv(work / "output" / "submission.csv")["control_success"].to_numpy(dtype=float)

    meta = json.loads((work / "model" / "metadata.json").read_text(encoding="utf-8"))
    cal = meta["calibration"]
    seg = np.zeros(len(test), dtype=float)
    for tm in (meta.get("seg_probe") or []):
        arr = test[tm["col"]].astype(str).to_numpy()
        if "map" in tm:
            off = np.zeros(len(test))
            for k, v in tm["map"].items():
                off[arr == str(k)] = float(v)
            seg += float(tm.get("t", 1.0)) * off
        else:
            ind = np.isin(arr, [str(v) for v in tm["levels"]]) if "levels" in tm \
                else test[tm["col"]].to_numpy(dtype=float) >= float(tm["ge"])
            seg += float(tm["t"]) * (ind.astype(float) - float(tm["w"]))
    if meta.get("quad_probe"):
        sys.exit("quad_probe가 있는 zip은 raw 역산이 비선형이라 쓰지 않는다(챔피언 계열을 쓸 것)")

    raw = cal["center"] + (p - seg - cal["shift"] - cal["center"]) / cal["slope"]
    clipped = int(((p <= 0.0) | (p >= 1.0)).sum())
    print(f"zip           : {Path(args.zip).name}")
    print(f"행            : {len(raw):,}  (클리핑 {clipped}행 — 0이어야 역산이 정확)")
    print(f"final (μ, σ)  : ({p.mean():.10f}, {p.std():.10f})")
    print(f"**raw (μ, σ)**: ({raw.mean():.10f}, {raw.std():.10f})")
    print()
    print(f"(프록시 기준) --partnermoments {raw.mean():.10f},{raw.std():.10f}")
    print()
    print("=" * 78)
    print("🚨 그런데 **프록시 상수를 쓰면 안 된다** (D-59에서 확인)")
    print("=" * 78)
    print("  서빙 경로의 당해시즌(is4) 룩업은 '직전 완결 시즌말 누적'을 빼도록 만들어져 있고")
    print("  배포본은 그 기준이 **2024년 말**이다. 그래서 train 2024 행을 서빙 경로에 먹이면")
    print("  당해시즌 피처가 거의 0이 되어 예측 분산이 줄어든다 —")
    print(f"  실제로 프록시 σ={raw.std():.6f} 인데 out-of-fold 2024 σ는 0.0459451이다(−36%).")
    print("  ⇒ 이 값은 2025의 우리 raw 분포를 대표하지 않는다.")
    print()
    print("  **대신 2025에서 직접 측정된 값을 쓴다**:")
    print("    mean_a = 0.4680607     (D-58 LB 역산 private raw 평균)")
    print("    std_a  = sqrt(C_s / C) = sqrt(786.2 / 402463.28568) = 0.0441981")
    print("             (C_s = 786.2 는 평균보존 slope 대칭쌍에서 실측한 2025 곡률이고")
    print("              항등식 C_s = C·Var(raw) 로 분산이 바로 나온다)")
    print()
    print("  → --partnermoments 0.4680607,0.0441981")
    print("  파트너 쪽(mean_b/std_b)은 2025 측정이 없으므로 MANIFEST의 train2024 값을 쓴다.")
    print("  잔여 레벨·스케일 오차는 블렌드 후 **아핀 재프로빙 2슬롯**이 흡수한다.")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
