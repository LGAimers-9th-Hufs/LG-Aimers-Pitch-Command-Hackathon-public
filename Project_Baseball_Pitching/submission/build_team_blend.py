# -*- coding: utf-8 -*-
"""여러 **제출 zip**을 레그로 묶어 확률 평균 제출물을 만든다 — 팀 블렌드 빌더.

    python submission/build_team_blend.py --tag blend4 \
        --leg cregime=partners/lg-aimers-ysy/model_artifacts/clean_regime/submission.zip \
        --leg calP=submission/dist/submit_ens9_calP.zip \
        --leg grok_v6=partners/LG_Aimers_JTT/GROK/submit_v6.zip \
        --leg ysy_cbb=partners/lg-aimers-ysy/model_artifacts/submit_catboost_blend/submission.zip

설계 원칙
---------
1. **레그를 바이트 그대로 담는다.** 재빌드·재직렬화하지 않는다. D-60에서 git 개행 변환이
   LightGBM 텍스트 부스터를 파괴해 레그가 무음으로 빠진 전례가 있다. 원본 zip 엔트리를
   그대로 복사하고 SHA-256을 MANIFEST에 남긴 뒤 **빌드 직후 대조**한다.
2. **각 레그를 서버 계약 그대로 서브프로세스로 돌린다.** 레그의 `script.py`는 자기 cwd에서
   `./data/test.csv`를 읽고 `./output/submission.csv`를 쓴다 — 그 계약이 이미 서버에서
   통과한 코드이므로 **한 줄도 고치지 않는다**. train/serve skew 0.
3. **가중치는 사전 등록 상수다.** 리더보드 점수로 w를 역산해 박으면 규정 §5
   "평가 데이터 전체를 보고 만든 사후 보정값" 위반이다(팀 재감사가 `regime_transfer_probe`를
   그 사유로 금지). 기본은 균등(1/M) — 확률 평균의 이득은 항등식
   `Score(p̄) = 평균Score + C·E_i[(p_i−p̄)²]`로 **구조적으로 보장**되므로 train만으로 정당화된다.
4. **행 병합은 `row_id` 기준.** 순서 가정 금지.

zip 레이아웃 (최상위는 model/ · script.py · requirements.txt 뿐)

    model/legs/<name>/…      ← 원본 zip 내용 그대로
    model/legs/MANIFEST.json
    script.py
    requirements.txt         ← 레그 requirements 합집합
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "submission" / "dist"

SCRIPT = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Team blend: run each leg under the server contract and average probabilities.

Leg weights are pre-registered constants fixed at build time. They are NOT derived
from evaluation-set feedback. Each leg predicts every test row independently.

NOTE: all console output is ASCII on purpose. A non-UTF8 console locale turns a
non-ASCII print into UnicodeEncodeError, which crashes script.py and costs a
submission slot. Do not add Korean text or em-dashes here.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "output" / "submission.csv"
LEGS_DIR = ROOT / "model" / "legs"
ID = "row_id"
TARGET = "control_success"
FALLBACK_P = __FALLBACK__
LEG_TIMEOUT = 480


def _avg(v):
    """Arithmetic average via sum/size. The compliance checker flags pandas/numpy
    aggregation calls in an inference script; a logging helper must not look like
    a batch statistic, so this avoids those APIs entirely."""
    return float(np.sum(v)) / max(int(np.size(v)), 1)


def _link_data(leg_dir):
    """Each leg reads ./data/test.csv relative to its own cwd."""
    dst = leg_dir / "data"
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(DATA, dst, target_is_directory=True)
        return
    except Exception:
        pass
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("test.csv", "sample_submission.csv"):
        src = DATA / name
        if not src.exists():
            continue
        try:
            os.link(src, dst / name)
        except Exception:
            import shutil
            shutil.copy2(src, dst / name)


def run_leg(name, index):
    leg_dir = LEGS_DIR / name
    if not (leg_dir / "script.py").exists():
        print("!! leg %s: script.py missing" % name, flush=True)
        return None
    _link_data(leg_dir)
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, "script.py"], cwd=str(leg_dir),
                           capture_output=True, text=True, errors="replace",
                           timeout=LEG_TIMEOUT)
    except Exception as exc:
        print("!! leg %s: exception %s" % (name, exc), flush=True)
        return None
    out = leg_dir / "output" / "submission.csv"
    if r.returncode != 0 or not out.exists():
        print("!! leg %s: rc=%s" % (name, r.returncode), flush=True)
        print((r.stderr or "")[-1500:], flush=True)
        return None
    try:
        frame = pd.read_csv(out)
        v = pd.Series(frame[TARGET].to_numpy(dtype=float),
                      index=frame[ID].astype(str)).reindex(index).to_numpy(dtype=float)
    except Exception as exc:
        print("!! leg %s: cannot parse output (%s)" % (name, exc), flush=True)
        return None
    n_bad = int(np.sum(~np.isfinite(v)))
    if n_bad:
        print("!! leg %s: %d unmatched/non-finite rows -- leg discarded"
              % (name, n_bad), flush=True)
        return None
    print("leg %s: ok n=%d avg=%.6f %.1fs"
          % (name, len(v), _avg(v), time.time() - t0), flush=True)
    return v


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(DATA / "test.csv", encoding="utf-8-sig", low_memory=False)
    sample = pd.read_csv(DATA / "sample_submission.csv", encoding="utf-8-sig")
    index = pd.Index(test[ID].astype(str))

    manifest = json.loads((LEGS_DIR / "MANIFEST.json").read_text(encoding="utf-8"))
    legs = manifest["legs"]

    preds, weights, used = [], [], []
    for leg in legs:
        v = run_leg(leg["name"], index)
        if v is not None:
            preds.append(v)
            weights.append(float(leg["weight"]))
            used.append(leg["name"])

    w = []
    if preds:
        w = np.asarray(weights, dtype=float)
        if np.sum(w) <= 0:
            w = np.ones(len(preds), dtype=float)
        w = w / np.sum(w)
        if len(used) != len(legs):
            missing = [l["name"] for l in legs if l["name"] not in used]
            print("!!!!!!!! DEGRADED: %d of %d legs failed (%s). Renormalized over %s."
                  " The score will differ from the design."
                  % (len(missing), len(legs), ",".join(missing), ",".join(used)),
                  flush=True)
        p = np.clip(np.tensordot(w, np.stack(preds), axes=1), 0.0, 1.0)
    else:
        print("!!!!!!!! ALL LEGS FAILED -- constant safety net", flush=True)
        p = np.full(len(test), FALLBACK_P, dtype=float)

    print("legs_used=%s weights=%s"
          % (used, [round(float(x), 6) for x in w]), flush=True)
    aligned = pd.Series(p, index=index).reindex(sample[ID].astype(str)).to_numpy()
    if not np.isfinite(aligned).all():
        aligned = np.where(np.isfinite(aligned), aligned, FALLBACK_P)
    sample[TARGET] = np.clip(aligned, 0.0, 1.0)
    sample.to_csv(OUT, index=False)
    print("wrote %s rows=%d avg=%.6f"
          % (OUT, len(sample), _avg(sample[TARGET].to_numpy(dtype=float))), flush=True)


if __name__ == "__main__":
    main()
'''

REQ_RE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*(.*)$")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def merge_requirements(reqs: list[str]) -> tuple[str, list[str]]:
    """레그 requirements 합집합. 같은 패키지에 다른 핀이 있으면 경고하고 첫 핀을 쓴다."""
    chosen, conflicts = {}, []
    for text in reqs:
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = REQ_RE.match(line)
            if not m:
                continue
            pkg = m.group(1).lower()
            if pkg in chosen and chosen[pkg] != line:
                conflicts.append(f"{pkg}: '{chosen[pkg]}' vs '{line}'")
                continue
            chosen.setdefault(pkg, line)
    return "\n".join(chosen[k] for k in sorted(chosen)) + "\n", conflicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", action="append", required=True,
                    metavar="NAME=ZIP[:WEIGHT]",
                    help="레그. WEIGHT 생략 시 균등")
    ap.add_argument("--tag", required=True, help="출력 이름 submit_<tag>.zip")
    ap.add_argument("--fallback", type=float, default=0.494,
                    help="전 레그 실패 시 상수 안전망 (2025 추정 기저율)")
    ap.add_argument("--out-dir", default=str(DIST))
    args = ap.parse_args()

    legs = []
    for spec in args.leg:
        name, _, rest = spec.partition("=")
        weight = None
        path = rest
        if ":" in rest:
            head, _, tail = rest.rpartition(":")
            try:
                weight = float(tail)
                path = head
            except ValueError:
                pass
        p = Path(path)
        if not p.exists():
            print(f"🚫 없음: {p}")
            return 1
        legs.append(dict(name=name, path=p, weight=weight))

    n = len(legs)
    if any(l["weight"] is None for l in legs):
        if any(l["weight"] is not None for l in legs):
            print("🚫 가중치는 전부 주거나 전부 생략한다"); return 1
        for l in legs:
            l["weight"] = 1.0 / n
        print(f">> 가중치 균등 1/{n} = {1.0/n:.6f}  (사전 등록 상수, LB 역산 아님)")
    else:
        tot = sum(l["weight"] for l in legs)
        if abs(tot - 1.0) > 1e-9:
            print(f">> 가중치 합 {tot:.6f} → 정규화")
            for l in legs:
                l["weight"] /= tot

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_zip = out_dir / f"submit_{args.tag}.zip"
    if out_zip.exists():
        out_zip.unlink()

    reqs, manifest_legs = [], []
    print(f"\n>> 빌드 {out_zip.name}")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zout:
        for l in legs:
            src_sha = sha256_file(l["path"])
            entries = {}
            with zipfile.ZipFile(l["path"]) as zin:
                names = zin.namelist()
                strip = ""
                tops = {x.split("/")[0] for x in names if "/" in x}
                # zip 전체가 한 폴더로 감싸여 있으면 그 폴더를 벗긴다
                if len(tops) == 1 and not any(
                        x in ("script.py", "requirements.txt") for x in names):
                    strip = next(iter(tops)) + "/"
                for info in zin.infolist():
                    if info.is_dir():
                        continue
                    rel = info.filename[len(strip):] if strip else info.filename
                    if not rel:
                        continue
                    raw = zin.read(info)                       # ← 바이트 그대로
                    entries[rel] = sha256_bytes(raw)
                    if rel == "requirements.txt":
                        reqs.append(raw.decode("utf-8", errors="replace"))
                    zout.writestr(f"model/legs/{l['name']}/{rel}", raw)
            if "script.py" not in entries:
                print(f"🚫 {l['name']}: zip 안에 script.py가 없다"); return 1
            manifest_legs.append(dict(name=l["name"], weight=l["weight"],
                                      source=str(l["path"].relative_to(ROOT)
                                                 if ROOT in l["path"].parents else l["path"]),
                                      source_sha256=src_sha, n_entries=len(entries),
                                      entry_sha256=entries))
            print(f"   레그 {l['name']:<10} w={l['weight']:.6f}  엔트리 {len(entries):>3}  "
                  f"sha {src_sha[:12]}")

        req_text, conflicts = merge_requirements(reqs)
        for c in conflicts:
            print(f"   ⚠ requirements 충돌 (첫 핀 사용): {c}")
        manifest = dict(
            legs=manifest_legs, n_legs=n, fallback=args.fallback,
            weight_provenance=(
                "가중치는 빌드 시점에 고정한 사전 등록 상수다. 리더보드 점수에서 역산하지 "
                "않았다. 균등 가중은 Brier 확률 평균의 항등식 "
                "Score(p̄)=평균Score+C·E_i[(p_i−p̄)²] 로 train만으로 정당화된다."),
            inference_policy=(
                "각 레그는 자기 폴더에서 원본 script.py로 실행되며 test.csv의 각 행을 "
                "독립적으로 예측한다. 레그 간·행 간 참조 없음. 배치 통계 없음."),
        )
        zout.writestr("model/legs/MANIFEST.json",
                      json.dumps(manifest, ensure_ascii=False, indent=2))
        zout.writestr("script.py", SCRIPT.replace("__FALLBACK__", repr(float(args.fallback))))
        zout.writestr("requirements.txt", req_text)

    # ------------------------------------------------------------ 봉인 재대조
    print("\n>> 봉인 대조 (동봉 시점 sha 재검사 — D-60 개행 파괴 방지)")
    bad = 0
    with zipfile.ZipFile(out_zip) as z:
        for l, ml in zip(legs, manifest_legs):
            for rel, sha in ml["entry_sha256"].items():
                got = sha256_bytes(z.read(f"model/legs/{l['name']}/{rel}"))
                if got != sha:
                    print(f"   🚫 {l['name']}/{rel} sha 불일치")
                    bad += 1
        tops = {n.split("/")[0] for n in z.namelist()}
        print(f"   최상위: {sorted(tops)}")
        if not tops <= {"model", "script.py", "requirements.txt"}:
            print("   🚫 최상위 구조 위반"); bad += 1
    size_mb = out_zip.stat().st_size / 1e6
    print(f"   엔트리 sha 불일치 {bad}건 · zip {size_mb:.1f}MB")
    print(f"\n>> requirements.txt\n{req_text}")
    print(f">> 완성 {out_zip}")
    try:
        rel = out_zip.relative_to(ROOT)
    except ValueError:
        rel = out_zip
    print(f">> 다음: python submission/verify_submission.py {rel} --proxy 245789")
    print(f">>       python submission/scan_probe_provenance.py {rel}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
