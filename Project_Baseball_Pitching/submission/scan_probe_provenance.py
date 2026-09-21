# -*- coding: utf-8 -*-
"""제출 아티팩트에서 **리더보드 역산 성분**을 기계 검사한다 — 규정 §5 위반 탐지기.

    python submission/scan_probe_provenance.py submission/dist
    python submission/scan_probe_provenance.py --json partners/ysy/model_artifacts

왜 필요한가 (2026-08-29)
------------------------
`data/data_description.md` §5 금지 목록의 마지막 항목이 **"평가 데이터 전체를 보고 만든 사후
보정값"**이다. 리더보드 점수는 평가셋 전체 라벨의 함수이므로, 그 점수에서 최적 상수를 **역산해
박아 넣으면** 그 상수는 평가 데이터를 보고 만든 사후 보정값이 된다.

`docs/SUBMISSION_COMPLIANCE.md`가 이미 같은 선을 그어놨다:

    사전 등록된 후보 중 **점수로 고르는 것**(선택) = 허용 — 리더보드의 존재 이유다
    점수에서 **최적값을 풀어 상수로 박는 것**(역산) = 위반

D-58/D-59에서 우리가 이 선을 넘었고(seg t*, slope 1.14202, 블렌드 w*), 팀원 재감사
(`lg-aimers-ysy/pipelines/dsf_upgrade/COMPLIANCE_REAUDIT.md`, 2026-08-28)가 그것을 발견해
1035~1075점 라인 전체를 제출 금지시켰다. 이 스크립트는 같은 검사를 모든 후보에 기계적으로
돌려 위반 성분이 다시 제출물에 섞이지 않게 한다.

판정
----
  BLOCK   위반 서명 발견 — 제출 금지
  REVIEW  회색지대 — 사람이 provenance를 읽고 판단
  CLEAN   서명 없음
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import zipfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- 서명
# 1) 역산 전용 metadata 키 — 존재만으로 BLOCK
BLOCK_KEYS = {
    "seg_probe": "세그먼트 절편 t*를 LB 대칭쌍 제출로 역산 (D-58)",
    "regime_transfer_probe": "블렌드 가중 w*를 LB 3점 정확해로 역산 (D-61/JTT)",
    "quad_probe": "곡률 축 계수를 LB 프로브로 역산 (D-60)",
    "partner_probe": "파트너 가중을 LB 프로브로 역산",
}

# 2) 알려진 역산 상수 — 값이 나오면 BLOCK
BLOCK_VALUES = {
    1.14202: "exact1 slope — slopeP/slopeStar 3점 역산 (로컬 최적은 1.035270)",
    -0.00264162: "exact1 shift — 평균보존식 + LB 역산",
    0.0057: "gtF 세그먼트 t* — gtP/gtStar 역산",
    0.0057024: "gtF 세그먼트 t* (정밀값)",
    0.7412437855967673: "JTT 챔피언 가중 — LB 3점 곡률 K=106.35 역산",
    0.25875621440323276: "JTT regime 가중 — 위와 동일",
    0.4680607: "raw_bar — LB 절편 프로브로 역산한 2025 예측 평균",
    0.0441981: "sigma_A — slope 곡률 C_s=786.2에서 역산",
}

# 3) provenance 문자열 서명 (BLOCK)
BLOCK_PATTERNS = [
    (r"public\s+(?:leader\s*board\s+|lb\s+)?scores?", "provenance가 공개 점수 사용을 명시"),
    (r"[+]\s*delta\s*/\s*-\s*delta", "대칭쌍(+delta/-delta) 프로브 언급"),
    (r"one[- ]point\s+exact\s+quadratic", "단일점 정확 2차 역산"),
    (r"inver(?:t|sion|ted)[^.\n]{0,40}scores?", "점수로부터 역산"),
    (r"리더보드[^.\n]{0,30}(?:역산|프로브|probe)", "리더보드 역산/프로브"),
    (r"(?:역산|프로브)[^.\n]{0,30}리더보드", "리더보드 역산/프로브"),
    (r"vert(?:ex|ices)[^.\n]{0,40}public", "공개 점수 포물선 꼭짓점"),
]

# 4) 회색지대 (REVIEW)
REVIEW_PATTERNS = [
    (r"probe", "'probe' 문자열 — 맥락 확인 필요"),
    (r"trackman[^.\n]{0,40}pitcher_id|pitcher_id[^.\n]{0,40}trackman",
     "트랙맨-train ID 매핑 — 재현 코드/provenance 동봉 필요"),
    (r"leaderboard|리더보드", "리더보드 언급 — 선택인지 역산인지 확인"),
]

# 부정 표현 — 근처에 있으면 BLOCK을 REVIEW로 낮춘다.
# "리더보드 역산을 하지 않았다" 같은 문장은 준수 선언이지 위반이 아니다.
# 팀원 패키지도 "excluded: DSF champion leaderboard probes"처럼 부정으로 쓴다.
NEGATIONS = ("않", "아니", "없", "제외", "미사용", "배제",
             " no ", " not ", "without", "excluded", "exclude", "never",
             "free of", "does not", "did not", "n't ")

TEXT_SUFFIX = {".json", ".py", ".md", ".txt", ".cfg", ".ini", ""}
SKIP_SUFFIX = {".joblib", ".pkl", ".npz", ".npy", ".so", ".pyd", ".bin", ".png", ".pdf"}


# --------------------------------------------------------------------------- 검사
def _walk_json(obj, path=""):
    """중첩 dict/list를 (경로, 키, 값)으로 평탄화."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            yield p, k, v
            yield from _walk_json(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{path}[{i}]"
            yield p, None, v
            yield from _walk_json(v, p)


def _num_hits(v):
    """알려진 역산 상수와 일치하는가 (float 왕복 오차 허용)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return []
    out = []
    for known, why in BLOCK_VALUES.items():
        if abs(float(v) - known) <= max(1e-12, abs(known) * 1e-9):
            out.append((known, why))
    return out


def _negated(window: str) -> bool:
    """부정 표현이 근처에 있으면 준수 선언으로 본다.

    "리더보드 점수에서 역산하지 않았다" · "excluded: DSF champion leaderboard probes" 처럼
    **위반을 배제했다고 밝히는 문장**이 서명과 같은 단어를 쓴다. 이걸 BLOCK으로 두면
    정직하게 적은 패키지일수록 더 많이 걸리는 역선택이 된다. 단 CLEAN으로 내리지는
    않는다 — REVIEW로 남겨 사람이 읽게 한다.
    """
    low = window.lower()
    return any(neg in low for neg in NEGATIONS)


def scan_text(name, text, findings):
    for pat, why in BLOCK_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            s = max(0, m.start() - 70)
            window = text[s:m.end() + 100].replace("\n", " ")
            if _negated(window):
                findings.append(("REVIEW", name,
                                 why + " (부정문 — 준수 선언으로 보임)", window))
            else:
                findings.append(("BLOCK", name, why, window))
    for pat, why in REVIEW_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            s = max(0, m.start() - 70)
            findings.append(("REVIEW", name, why,
                             text[s:m.end() + 100].replace("\n", " ")))


def scan_json(name, raw, findings):
    try:
        obj = json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        return False
    for path, key, val in _walk_json(obj):
        if key in BLOCK_KEYS:
            findings.append(("BLOCK", name, BLOCK_KEYS[key],
                             f"{path} = {json.dumps(val, ensure_ascii=False)[:200]}"))
        for known, why in _num_hits(val):
            findings.append(("BLOCK", name, why,
                             f"{path} = {val!r}  (알려진 역산 상수 {known})"))
        if isinstance(val, str) and len(val) > 20:
            scan_text(f"{name}:{path}", val, findings)
    return True


def scan_member(name, raw, findings):
    suf = Path(name).suffix.lower()
    if suf in SKIP_SUFFIX:
        return
    if suf == ".json" and scan_json(name, raw, findings):
        return
    if suf not in TEXT_SUFFIX:
        return
    text = raw[:400_000].decode("utf-8", errors="replace")
    if text.startswith("tree\nversion=") or text.startswith("tree\r\nversion="):
        return                                    # LightGBM 텍스트 부스터
    scan_text(name, text, findings)


def scan_zip(zpath, findings, depth=0):
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            inner = f"{zpath.name}!{info.filename}"
            raw = z.read(info)
            if info.filename.lower().endswith(".zip") and depth < 2:
                try:
                    with zipfile.ZipFile(io.BytesIO(raw)) as z2:
                        for i2 in z2.infolist():
                            if not i2.is_dir():
                                scan_member(f"{inner}!{i2.filename}",
                                            z2.read(i2), findings)
                    continue
                except Exception:
                    pass
            scan_member(inner, raw, findings)


def scan_dir(d, findings):
    for p in sorted(d.rglob("*")):
        if not p.is_file() or p.suffix.lower() in SKIP_SUFFIX or p.suffix.lower() == ".zip":
            continue
        if ".git" in p.parts:
            continue
        try:
            scan_member(str(p.relative_to(d)), p.read_bytes(), findings)
        except Exception:
            pass


def verdict_of(findings):
    if any(f[0] == "BLOCK" for f in findings):
        return "BLOCK"
    if any(f[0] == "REVIEW" for f in findings):
        return "REVIEW"
    return "CLEAN"


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+", help="zip 파일 또는 디렉터리")
    ap.add_argument("--json", action="store_true", help="판정표를 JSON으로 저장")
    ap.add_argument("--quiet", action="store_true", help="증거 줄 생략")
    args = ap.parse_args()

    targets = []
    for t in args.targets:
        p = Path(t)
        if p.is_dir():
            zs = sorted(x for x in p.rglob("*.zip") if ".git" not in x.parts)
            targets.extend(zs if zs else [p])
        elif p.exists():
            targets.append(p)
        else:
            print(f"[skip] 없음: {t}")

    rows = []
    for p in targets:
        findings = []
        try:
            if p.suffix.lower() == ".zip":
                scan_zip(p, findings)
            else:
                scan_dir(p, findings)
        except Exception as exc:                                  # noqa: BLE001
            rows.append(dict(target=str(p), verdict="ERROR", n_block=0,
                             n_review=0, error=str(exc)))
            print(f"\n💥 ERROR  {p}\n      {exc}")
            continue
        v = verdict_of(findings)
        nb = sum(1 for f in findings if f[0] == "BLOCK")
        nr = sum(1 for f in findings if f[0] == "REVIEW")
        rows.append(dict(target=str(p), verdict=v, n_block=nb, n_review=nr,
                         findings=[dict(level=a, where=b, why=c, evidence=d)
                                   for a, b, c, d in findings]))
        icon = {"BLOCK": "🚫", "REVIEW": "⚠", "CLEAN": "✅"}[v]
        print(f"\n{icon} {v:<6} {p}")
        if not args.quiet:
            seen = set()
            for lvl, where, why, ev in findings:
                if (lvl, why) in seen:
                    continue
                seen.add((lvl, why))
                print(f"      [{lvl}] {why}")
                print(f"             {where}")
                print(f"             ...{ev[:170].strip()}...")

    print("\n" + "=" * 82)
    print("%-56s %-7s %6s %6s" % ("대상", "판정", "BLOCK", "REVIEW"))
    print("-" * 82)
    for r in rows:
        print("%-56s %-7s %6d %6d"
              % (Path(r["target"]).name[:56], r["verdict"],
                 r.get("n_block", 0), r.get("n_review", 0)))
    n_block = sum(1 for r in rows if r["verdict"] == "BLOCK")
    print("=" * 82)
    print(f"BLOCK {n_block} / {len(rows)} — BLOCK 아티팩트는 제출하지 않는다")

    if args.json:
        out = ROOT / "results" / "compliance_scan.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f">> 저장 {out}")
    return 1 if n_block else 0


if __name__ == "__main__":
    sys.exit(main())
