# -*- coding: utf-8 -*-
"""Phase 3 제출용 재현 학습 코드 패키지 수집기.

    python phase3/tools/collect_package.py            # phase3/package/ 트리 + phase3/dist/*.zip
    python phase3/tools/collect_package.py --no-zip   # 트리만

대회 규칙 §3 "[제출 파일 목록] o (필수) Private Score 재현용 학습 코드"에 대응한다.
설계 원칙 3가지:

1. **저장소 상대경로를 그대로 미러링한다.** `sweep/`는 평면 sibling import 패키지이고
   `real_data.py`는 `parent.parent/"data"/"data"`로, `tm_physmix.py`는
   `parent.parent/"results"/"trackman"/matches.csv`로 자원을 찾는다. 폴더를 재배치하면 전부 깨진다.
2. **전체 복사가 아니라 실측 폐포만 담는다.** sweep 75개 중 재현에 실제로 로드되는 모듈만
   넣는다(근거 = 실제 빌드 1회에서 채집한 `sys.modules` 스냅샷,
   `results/phase3_repro_modules.json`).
3. **재현에 필요한 입력 아티팩트를 커밋 해시로 고정해 동봉한다.** 특히
   `results/trackman/matches.csv`는 2026-08-30 연구 라운드(GMM Tier 재컷, 커밋 3917b5f)가
   같은 경로를 덮어썼다. 제출본이 쓴 것은 그 이전 판(커밋 17fc655)이며 이 스크립트가
   git에서 그 판을 꺼내 담는다. 경위는 package README의 재현 결과 절.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import shutil
import stat
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent      # Project_Baseball_Pitching
GITROOT = ROOT.parent                                      # 저장소 git 루트
PKG = ROOT / "phase3" / "package"
DIST = ROOT / "phase3" / "dist"
PKG_NAME = "ABS_kkangtongzone_phase3_code"                 # zip 이름은 ASCII (메일 첨부 안전)

# --- 1. 본 저장소 sweep 모듈 --------------------------------------------------
# physmix 빌드 실행 중 실제 로드된 17개 + 검증기가 끌어오는 6개 + 매칭 provenance 1개.
SWEEP = [
    # physmix 학습 경로 (실측 폐포)
    "real_data.py", "lgbm_family.py", "inseason.py", "inseason_full.py",
    "nn_member.py", "nn_carrier.py", "eb_carrier.py", "interaction_carrier.py",
    "is_corrector.py", "tm_member.py", "tm_physmix.py", "trackman.py",
    "prev_denom.py", "season_centering.py", "phase2_ens_check.py",
    "ens4_weights.py", "ens5_pool.py",
    # 규정 검증 경로
    "test_regulation.py", "validation.py", "recal.py", "reporting.py",
    "run_real.py", "z_asof.py",
    # 엔티티 매칭 provenance (COMPLIANCE_DISCLOSURE 5절)
    "trackman_v2.py",
]

# --- 2. 본 저장소 submission 도구 ---------------------------------------------
SUBMISSION = [
    "build_lgbm_ensemble.py",      # physmix 레그 학습·패키징
    "script_lgbm_template.py",     # physmix 서빙 템플릿 (빌더가 zip에 심는다)
    "build_submission.py",         # SERVE 블록 바이트 복사 = 파리티 0의 근거
    "script_template.py",
    "build_tm3_rowwise.py",        # tm3L 레그 행독립 재빌드
    "build_team_blend.py",         # 4레그 균등 블렌드 조립
    "verify_submission.py",        # 서버 계약 검증
    "scan_probe_provenance.py",    # 역산 상수 기계 검사
    "leg_matrix.py",               # 레그 불일치 RMS 행렬
    "proxy_rms.py",
    "combo_search.py",             # 블렌드 조합 기대점수 전수 열거
]

# --- 3. 재현에 필요한 입력 아티팩트 -------------------------------------------
# (패키지 내 경로, 저장소 상대경로, 고정 커밋 또는 None=작업트리 현재본)
ARTIFACTS = [
    ("results/trackman/matches.csv", "results/trackman/matches.csv", "17fc655"),
    ("results/phase3/profiles.json", "results/phase3/profiles.json", None),
]

# --- 4. 문서 -------------------------------------------------------------------
DOCS = [
    ("README.md", "phase3/package_README.md"),          # 패키지 첫 문서
    ("COMPLIANCE_DISCLOSURE.md", "phase3/COMPLIANCE_DISCLOSURE.md"),
    ("docs/RULES_PHASE3.md", "phase3/RULES_PHASE3.md"),
    ("docs/SOLUTION_OUTLINE.md", "phase3/SOLUTION_OUTLINE.md"),
    ("docs/SUBMISSION_COMPLIANCE.md", "docs/SUBMISSION_COMPLIANCE.md"),
    ("docs/HACKATHON_TASK.md", "docs/HACKATHON_TASK.md"),
]

# --- 5. 팀원 저장소 -------------------------------------------------------------
# 정적 import 폐포로는 동적 로드를 놓치므로 팀원 코드는 학습 파이프라인 전체를 담는다
# (가중치 폴더와 바이너리는 제외 — 코드만 각각 666KB / 96KB 로 작다).
#
# extra_glob = 제외 규칙을 뚫고 반드시 포함할 것. 블렌드 레그의 **학습 산출물**이다.
# 이것이 있어야 `build_team_blend.py`로 최종 제출물을 그 자리에서 재조립·대조할 수 있다
# (레그는 바이트 그대로 담기므로 학습 산출물 없이는 조립 단계가 재현되지 않는다).
PARTNERS = [
    dict(name="lg-aimers-ysy", commit="e76e6f1", include_all=True,
         include_glob=(),
         extra_glob=("model_artifacts/clean_lookup/README.md",
                     "model_artifacts/clean_lookup/submission.zip",
                     "model_artifacts/clean_moe/README.md",
                     "model_artifacts/clean_moe/submission.zip"),
         exclude_prefix=("model_artifacts/",),
         exclude_suffix=(".zip", ".pkl", ".joblib", ".pt", ".npz", ".bin")),
    dict(name="LG_Aimers_JTT", commit="d7ecbbc", include_all=False,
         include_glob=("trackman_pipeline/*.py", "trackman_pipeline/README.md",
                       "trackman_pipeline/data/v3/*"),
         extra_glob=(),
         exclude_prefix=(), exclude_suffix=()),
]

CODE_EXT = {".py", ".ipynb"}
TEXT_EXT = CODE_EXT | {".md", ".txt", ".json", ".csv", ".cfg", ".toml", ".yml", ".yaml"}

DATA_NOTE = """# 주최 배포 데이터를 여기에 둔다

`open.zip`을 풀어 아래 4개를 이 폴더에 배치한다.

    data/data/train.csv
    data/data/test.csv
    data/data/sample_submission.csv
    data/data/trackman_history.csv

`sweep/real_data.py`가 자기 위치 기준 `parent.parent/"data"/"data"`를 보므로 경로가 고정이다.
데이터는 주최가 배포한 것이므로 패키지에 포함하지 않는다.
"""


def git_show(repo: Path, rev: str, rel: str) -> bytes:
    r = subprocess.run(["git", "show", f"{rev}:{rel}"], cwd=repo, capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"git show 실패: {repo}:{rev}:{rel}\n"
                         f"{r.stderr.decode('utf-8', 'replace')}")
    if r.stdout.startswith(b"version https://git-lfs"):
        raise SystemExit(f"git show가 LFS 포인터를 반환했다(실제 내용 아님): {rel}")
    return r.stdout


def git_files(repo: Path, rev: str) -> list:
    r = subprocess.run(["git", "ls-tree", "-r", "--name-only", rev], cwd=repo,
                       capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise SystemExit(f"git ls-tree 실패: {repo}:{rev}\n{r.stderr}")
    return [x for x in r.stdout.splitlines() if x.strip()]


def assert_worktree_at(repo: Path, rev: str) -> None:
    """작업트리가 고정 커밋과 정확히 같은지 확인한다.

    내용은 `git show`가 아니라 **작업트리에서** 읽는다. `git show`는
    (a) Git LFS로 저장된 파일에 대해 실제 내용 대신 132바이트 포인터를 뱉고,
    (b) 텍스트로 오판한 이진 파일의 개행을 정규화해 조용히 파괴한다.
    실제로 JTT의 `submit_v3.zip`이 (a)에 걸려 1.94MB -> 132B 로 나왔다.
    그래서 "커밋 고정"은 이 검사로 보장하고, 바이트는 디스크에서 가져온다.
    """
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo,
                          capture_output=True, text=True, encoding="utf-8").stdout.strip()
    want = subprocess.run(["git", "rev-parse", "--short", rev], cwd=repo,
                          capture_output=True, text=True, encoding="utf-8").stdout.strip()
    if head != want:
        raise SystemExit(f"{repo.name}: 작업트리 HEAD {head} != 고정 커밋 {want} ({rev}). "
                         f"`git -C partners/{repo.name} checkout {rev}` 후 다시 실행한다.")
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo,
                           capture_output=True, text=True, encoding="utf-8").stdout.strip()
    if dirty:
        raise SystemExit(f"{repo.name}: 작업트리에 변경이 있다 — 고정 커밋 내용을 보장할 수 없다.\n{dirty}")


def safe_rmtree(root: Path) -> None:
    """정션/심링크를 먼저 링크만 끊고 지운다.

    이 트리에는 재현 실행용으로 `data/data` 정션을 걸어두는 경우가 있다. 그대로
    `shutil.rmtree`를 부르면 Windows에서 링크를 타고 들어가 **원본 데이터(train.csv 등)를
    지울 수 있다.** 실제로 한 번 걸릴 뻔했다 — 링크를 먼저 끊는다.
    """
    if not root.exists():
        return

    def is_reparse(p: Path) -> bool:
        try:
            st = os.lstat(p)
        except OSError:
            return False
        return bool(getattr(st, "st_file_attributes", 0)
                    & stat.FILE_ATTRIBUTE_REPARSE_POINT) or os.path.islink(p)

    # 깊은 것부터 끊어야 부모를 지울 때 링크가 남지 않는다
    for p in sorted(root.rglob("*"), key=lambda q: len(q.parts), reverse=True):
        if not is_reparse(p):
            continue
        try:
            os.rmdir(p) if p.is_dir() else os.unlink(p)   # 링크만 끊긴다
            print("   [링크 해제] %s" % p)
        except OSError as e:
            raise SystemExit("링크 해제 실패(원본 삭제 위험) — 수동 확인 필요: %s (%s)" % (p, e))

    remaining = [p for p in root.rglob("*") if is_reparse(p)]
    if remaining:
        raise SystemExit("링크가 남아 있어 삭제를 중단한다: %s" % remaining)
    shutil.rmtree(root)


def put(dest_rel: str, data: bytes, written: dict) -> None:
    p = PKG / dest_rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    written[dest_rel] = hashlib.sha256(data).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    safe_rmtree(PKG)
    PKG.mkdir(parents=True)
    written = {}

    print(">> 본 저장소 sweep 모듈")
    for f in SWEEP:
        src = ROOT / "sweep" / f
        if not src.exists():
            raise SystemExit(f"없음: {src}")
        put("sweep/" + f, src.read_bytes(), written)
    print("   %d개" % len(SWEEP))

    print(">> 본 저장소 submission 도구")
    for f in SUBMISSION:
        src = ROOT / "submission" / f
        if not src.exists():
            raise SystemExit(f"없음: {src}")
        put("submission/" + f, src.read_bytes(), written)
    print("   %d개" % len(SUBMISSION))

    print(">> 재현 입력 아티팩트")
    for dest, rel, rev in ARTIFACTS:
        if rev:
            data = git_show(GITROOT, rev, ROOT.name + "/" + rel)
            print("   %s  <- git %s (고정)" % (dest, rev))
        else:
            data = (ROOT / rel).read_bytes()
            print("   %s  <- 작업트리" % dest)
        put(dest, data, written)

    print(">> 문서")
    for dest, rel in DOCS:
        src = ROOT / rel
        if not src.exists():
            print("   [건너뜀] %s" % rel)
            continue
        put(dest, src.read_bytes(), written)

    print(">> 팀원 저장소")
    for spec in PARTNERS:
        repo = ROOT / "partners" / spec["name"]
        rev = spec["commit"]
        assert_worktree_at(repo, rev)
        keep = []
        for f in git_files(repo, rev):
            if any(fnmatch.fnmatch(f, g) for g in spec["extra_glob"]):
                keep.append(f)                      # 제외 규칙보다 우선
                continue
            if any(f.startswith(p) for p in spec["exclude_prefix"]):
                continue
            if any(f.endswith(s) for s in spec["exclude_suffix"]):
                continue
            if not spec["include_all"] and not any(
                    fnmatch.fnmatch(f, g) for g in spec["include_glob"]):
                continue
            keep.append(f)
        for f in keep:
            src = repo / f
            if not src.exists():
                raise SystemExit(f"작업트리에 없음: {src}")
            put("partners/%s/%s" % (spec["name"], f), src.read_bytes(), written)
        print("   %s @ %s: %d개 (작업트리, 커밋 일치 확인됨)" % (spec["name"], rev, len(keep)))

    put("data/data/PUT_OFFICIAL_CSV_HERE.md", DATA_NOTE.encode("utf-8"), written)

    # 학습 환경 핀 — 실제 로컬에 설치된 버전을 그대로 읽어 적는다(손으로 쓰지 않는다)
    print(">> requirements_train.txt (설치본 실측)")
    pins = []
    for mod, pkg in (("numpy", "numpy"), ("pandas", "pandas"), ("sklearn", "scikit-learn"),
                     ("scipy", "scipy"), ("lightgbm", "lightgbm"), ("joblib", "joblib"),
                     ("torch", "torch")):
        try:
            v = __import__(mod).__version__
        except Exception:
            print("   [경고] %s 미설치 — 핀에서 제외" % pkg)
            continue
        pins.append("%s==%s" % (pkg, v.split("+")[0]))
        print("   %s==%s" % (pkg, v))
    header = ("# 학습(재현) 환경 핀 — collect_package.py가 설치본에서 실측해 기록한다.\n"
              "# torch는 팀원 레그 cmoe(FactorTabM) 학습에만 쓰인다. physmix·tm3L은 CPU만으로 재현된다.\n")
    put("requirements_train.txt", (header + "\n".join(pins) + "\n").encode("utf-8"), written)
    put("requirements_serve.txt",
        ("# 제출 zip에 들어가는 추론 환경 핀 (평가 서버). torch는 서버 사전설치본을 쓴다.\n"
         "joblib==1.5.3\nlightgbm==4.6.0\npandas==2.3.3\nscikit-learn==1.8.0\n").encode("utf-8"),
        written)

    # --- 검사: 확장자·인코딩 (규칙 3절: .py/.ipynb, UTF-8) --------------------
    print(">> 검사")
    bad_enc, n_code = [], 0
    for rel in sorted(written):
        p = PKG / rel
        ext = p.suffix.lower()
        if ext in CODE_EXT:
            n_code += 1
        if ext in TEXT_EXT:
            try:
                p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                bad_enc.append(rel)
    print("   파일 %d개 (코드 .py/.ipynb %d개)" % (len(written), n_code))
    if bad_enc:
        print("   [실패] UTF-8 디코딩 불가:")
        for x in bad_enc:
            print("      ", x)
        raise SystemExit(1)
    print("   UTF-8 디코딩: 전부 통과")
    total = sum((PKG / r).stat().st_size for r in written)
    print("   합계 %.2f MB" % (total / 1e6))

    lines = ["%s  %s" % (written[r], r) for r in sorted(written)]
    (PKG / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("   SHA256SUMS.txt %d행" % len(lines))

    if not args.no_zip:
        DIST.mkdir(parents=True, exist_ok=True)
        zpath = DIST / (PKG_NAME + ".zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(PKG.rglob("*")):
                if p.is_file():
                    z.write(p, PKG_NAME + "/" + p.relative_to(PKG).as_posix())
        h = hashlib.sha256(zpath.read_bytes()).hexdigest()
        print(">> 완성 %s  (%.2f MB)" % (zpath, zpath.stat().st_size / 1e6))
        print("   sha256 %s" % h)


if __name__ == "__main__":
    main()
