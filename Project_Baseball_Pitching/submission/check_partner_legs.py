# -*- coding: utf-8 -*-
"""파트너 블렌드 프로브 쌍의 **레그 동일성** 검사 (D-59).

프로브 역산은 "두 제출물이 파트너 가중 `w` 하나만 다르다"는 전제 위에 서 있다.
서빙 템플릿은 멤버가 실패하면 조용히 건너뛰고 재정규화하며(`script_lgbm_template.py`),
파트너 레그도 예외 시 무음으로 빠진다. 그러면 전제가 깨져 역산 전체가 거짓이 된다.

검사 원리 — 클리핑이 없으면 예측차는 **w에 정확히 선형**이다:

    pred(w) = 0.5 + slope·((1−w)·raw_A + w·raw_B' − 0.5) + shift + seg
            = pred(0) + slope·w·(raw_B' − raw_A)

    ⇒ (pred(w₂) − pred(0)) / (pred(w₁) − pred(0)) = w₂/w₁   **원소마다 정확히**

이 비율이 어긋나면 어느 한 레그에서 무언가 다르게 돌아간 것이다.

    python submission/check_partner_legs.py --base submit_exact1 --legs pw12:0.12,pw30:0.30 --rows 60000
"""
from __future__ import annotations
import argparse
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


def run_zip(tag: str, sub: pd.DataFrame, tmp: Path, id_col: str, target: str):
    work = tmp / tag
    work.mkdir()
    with zipfile.ZipFile(DIST / f"submit_{tag}.zip") as z:
        z.extractall(work)
    (work / "data").mkdir(exist_ok=True)
    test = sub.drop(columns=[target])
    test.to_csv(work / "data" / "test.csv", index=False, encoding="utf-8")
    pd.DataFrame({"row_id": test[id_col], "control_success": 0.5}).to_csv(
        work / "data" / "sample_submission.csv", index=False, encoding="utf-8")
    r = subprocess.run([sys.executable, "script.py"], cwd=work, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stderr[-1500:])
        sys.exit(f"{tag}: script.py 실패")
    flag = [ln for ln in r.stdout.splitlines() if "partner_used" in ln or "weights_used" in ln]
    print(f"  [{tag}] " + " | ".join(flag))
    return pd.read_csv(work / "output" / "submission.csv")["control_success"].to_numpy(dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="exact1", help="w=0 기준 zip 태그(submit_ 제외)")
    ap.add_argument("--legs", required=True, help="'tag:w,tag:w' — 파트너 가중이 다른 제출물들")
    ap.add_argument("--rows", type=int, default=60000)
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "sweep"))
    import real_data as rd
    df = rd.load_train()
    sub = df[df[rd.SEASON] == 2024].head(args.rows)

    tmp = Path(tempfile.mkdtemp(prefix="pleg_"))
    try:
        print(f">> 기준 {args.base} 실행 ({len(sub):,}행)")
        p0 = run_zip(args.base, sub, tmp, rd.ID, rd.TARGET)
        legs = []
        for item in args.legs.split(","):
            tag, w = item.split(":")
            print(f">> 레그 {tag} (w={w}) 실행")
            legs.append((tag, float(w), run_zip(tag, sub, tmp, rd.ID, rd.TARGET)))

        print()
        ok = True
        ref_tag, ref_w, ref_p = legs[0]
        d_ref = ref_p - p0
        print(f"기준 방향: {ref_tag} − {args.base}, RMS {np.sqrt((d_ref**2).mean()):.6f}, "
              f"최대 {np.abs(d_ref).max():.6f}")
        if np.abs(d_ref).max() == 0.0:
            print("🚫 방향이 0이다 — 파트너 레그가 반영되지 않았다.")
            sys.exit(1)
        for tag, w, p in legs[1:]:
            exp = (w / ref_w) * d_ref
            err = float(np.abs((p - p0) - exp).max())
            print(f"  {tag}: 예측차 / 기준차 = {w/ref_w:.4f} 배여야 함 → 최대오차 {err:.3e}", end="  ")
            if err < 1e-12:
                print("✓")
            else:
                print("🚫 선형성 위반 — 레그 비동일. 역산하지 말 것")
                ok = False
        print()
        print("모든 검사 통과" if ok else "실패")
        sys.exit(0 if ok else 1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
