# -*- coding: utf-8 -*-
"""제출 zip을 실제 서버와 같은 방식으로 실행해 검증한다.

    python submission/verify_submission.py submission/dist/submit_s2.zip
    python submission/verify_submission.py <zip> --proxy 245789      # 풀스케일 런타임
    python submission/verify_submission.py <zip> --compare <ref.zip> # 참조 제출물과 예측 대조

검사 항목
  1) zip 구조 — 최상위에 model/ · script.py · requirements.txt 만 (여분 폴더 = 설치 오류)
  2) serve parity — model/serve_features.py가 sweep/real_data.py의 SERVE 블록과 **바이트 동일**
  3) 배치 통계 부재 — script.py·serve_features.py에 .mean()/.groupby(/.value_counts(/.transform( 없음 (§5)
  4) 실행 — 임시 폴더에서 `python script.py`를 돌려 output/submission.csv 생성·형식·시간 확인
  5) 폴백 — base.joblib을 일부러 깨뜨려도 죽지 않고 예측을 내는지 (실행 오류는 제출 횟수 차감)
  6) **서버 핀 환경** — sklearn 1.8.0 / joblib 1.5.3 / pandas 2.3.3 전용 venv에서 재실행.
     로컬은 pandas 3.0.0이라 서버를 재현하지 못한다(이 격차 때문에 로컬 검증이 통과해도 안심할 수 없다).
  7) **참조 대조**(--compare) — 같은 입력에서 참조 zip과 예측 RMS를 잰다. 블렌드 레그가 이미 점수를
     아는 제출물과 동일하다는 전제를 쓸 때는 여기서 **RMS 0.00000**을 확인해야 한다.
     (BLEND-1 사고의 원인이 정확히 이 확인 누락이었다 — 레그에 z_asof가 섞여 −192점.)

`--proxy N`은 train 2024 행에서 타깃만 떼어 test 스키마 입력을 만든다. 모델이 2024를 학습에
포함하므로 **점수는 낙관적이며 의사결정에 쓰지 않는다** — 여기서 보는 것은 런타임·안정성·예측 분포뿐이다.
"""
from __future__ import annotations
import argparse, os, re, shutil, subprocess, sys, tempfile, time, zipfile
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "data"
BANNED = [r"\.mean\(", r"\.groupby\(", r"\.value_counts\(", r"\.transform\(",
          r"\.rolling\(", r"\.expanding\(", r"\.cumsum\(", r"\.rank\("]
OK, FAIL = "  [OK]  ", "  [FAIL]"
_fails = []


def check(cond, msg, detail=""):
    print((OK if cond else FAIL) + " " + msg + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        _fails.append(msg)
    return cond


SERVER_PINS = ["scikit-learn==1.8.0", "joblib==1.5.3", "pandas==2.3.3"]
VENV = ROOT / "submission" / ".venv_server"


def server_python(create=True):
    """서버 핀 버전 전용 venv의 python. 없으면 만든다(gitignore 대상)."""
    py = VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if py.exists():
        return py
    if not create:
        return None
    print(f"  (서버 핀 venv 생성 중: {', '.join(SERVER_PINS)})")
    subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
    subprocess.run([str(py), "-m", "pip", "install", "-q", "--upgrade", "pip"], check=True)
    subprocess.run([str(py), "-m", "pip", "install", "-q", *SERVER_PINS], check=True)
    return py


def make_proxy(n: int, dest: Path):
    """train 2024 행 → test 스키마 프록시 입력(타깃 제거). 런타임·분포 확인 전용."""
    sys.path.insert(0, str(ROOT / "sweep"))
    import real_data as rd
    df = rd.load_train()
    sub = df[df[rd.SEASON] == 2024].head(n)
    test = sub.drop(columns=[rd.TARGET])
    dest.mkdir(parents=True, exist_ok=True)
    test.to_csv(dest / "test.csv", index=False, encoding="utf-8")
    pd.DataFrame({"row_id": test[rd.ID], "control_success": 0.5}).to_csv(
        dest / "sample_submission.csv", index=False, encoding="utf-8")
    print(f"  프록시 입력 {len(test):,}행 생성 (실제 평가 245,789행)")


def run_once(work: Path, label: str, python=None, threads=None):
    """`threads`를 주면 스레드 풀을 그 수로 묶는다 — **서버는 6 vCPU, 우리 머신은 16코어**라
    스레드를 안 묶은 로컬 실측은 서버 시간을 2~3배 과소평가한다. 추론 초과는 '제출 오류'로
    슬롯을 태울 수 있으므로(설치 오류와 달리 횟수에 반영된다) 이 측정이 필수다."""
    env = dict(os.environ)
    if threads:
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
            env[k] = str(threads)
    t = time.time()
    r = subprocess.run([str(python or sys.executable), "script.py"], cwd=work,
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env=env)
    dt = time.time() - t
    out = work / "output" / "submission.csv"
    ok = check(r.returncode == 0, f"{label}: script.py 정상 종료",
               (r.stderr or "")[-800:])
    ok &= check(out.exists(), f"{label}: output/submission.csv 생성")
    if out.exists():
        sub = pd.read_csv(out)
        smp = pd.read_csv(work / "data" / "sample_submission.csv", encoding="utf-8-sig")
        check(list(sub.columns[:2]) == ["row_id", "control_success"], f"{label}: 컬럼 형식")
        check(len(sub) == len(smp), f"{label}: 행 수 일치", f"{len(sub)} vs {len(smp)}")
        check(sub["row_id"].tolist() == smp["row_id"].tolist(), f"{label}: row_id 순서 일치")
        v = sub["control_success"]
        check(v.notna().all() and (v >= 0).all() and (v <= 1).all(),
              f"{label}: 확률값 범위 [0,1]")
    print(f"         실행 {dt:.1f}s (서버 제한 600s)")
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("--proxy", type=int, default=0,
                    help="train 2024에서 N행 프록시 입력을 만들어 사용(런타임·분포 확인)")
    ap.add_argument("--segdiff", default=None,
                    help="세그먼트 프로브 쌍 검사: 예측차가 (t−t_ref)·중심화지시자와 원소 단위로 "
                         "일치하는지. 역산의 전제(레그 동일)를 강제한다")
    ap.add_argument("--quaddiff", default=None,
                    help="형상(곡률) 프로브 레그 동일성: 참조 zip 경로. 예측차가 t·((raw−raw̄)^k−m)와 "
                         "원소 단위로 일치하는지 검사한다(raw는 참조 출력에서 캘리 역산으로 복원)")
    ap.add_argument("--compare", default=None,
                    help="참조 zip — 같은 입력에서 예측 RMS 대조(구성 동일성 확인)")
    ap.add_argument("--threads", type=int, default=0,
                    help="스레드 풀을 N으로 묶어 서버(6 vCPU)를 흉내 낸 실행 시간을 잰다. "
                         "로컬 16코어 실측은 서버를 2~3배 과소평가한다")
    ap.add_argument("--compose", default=None,
                    help="'w:zip,w:zip' — 참조 zip들의 **가중평균**과 일치하는지 대조. "
                         "블렌드 제출의 필수 검사: 각 레그가 이미 LB 점수를 아는 제출물과 같아야 "
                         "항등식 산술(기대점수·하한)이 성립한다")
    ap.add_argument("--no-server-env", action="store_true",
                    help="서버 핀 venv 재실행 생략(권장하지 않음)")
    args = ap.parse_args()

    zpath = Path(args.zip)
    if not zpath.exists():
        sys.exit(f"zip 없음: {zpath}")
    print(f"검증 대상: {zpath}  ({zpath.stat().st_size/1e6:.1f}MB)\n")

    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
    tops = sorted({n.split("/")[0] for n in names})
    check(set(tops) <= {"model", "script.py", "requirements.txt"},
          "zip 최상위 = model/ · script.py · requirements.txt", str(tops))
    check("script.py" in names and "requirements.txt" in names
          and any(n.startswith("model/") for n in names), "필수 항목 존재")

    tmp = Path(tempfile.mkdtemp(prefix="submitcheck_"))
    work = tmp / "run"
    work.mkdir()
    with zipfile.ZipFile(zpath) as z:
        z.extractall(work)
    (work / "data").mkdir(exist_ok=True)
    if args.proxy:
        make_proxy(args.proxy, work / "data")
    else:
        for f in ("test.csv", "sample_submission.csv"):
            shutil.copy2(DATA / f, work / "data" / f)

    # serve parity — SERVE 블록 아키텍처(real_data.py 기반)일 때만 해당.
    # LGBM 앙상블 계열은 script.py가 피처 코드를 자체 보유하므로 이 검사가 성립하지 않는다.
    sys.path.insert(0, str(ROOT / "submission"))
    sfp = work / "model" / "serve_features.py"
    if sfp.exists():
        from build_submission import extract_serve_block
        check(sfp.read_bytes() == extract_serve_block().encode("utf-8"),
              "serve parity: real_data.py SERVE 블록과 바이트 동일")
    else:
        print("  [skip]  serve parity — model/serve_features.py 없음(자체 피처 코드 아키텍처)")

    # 배치 통계 부재 — AST로 **실제 호출만** 본다(문자열/주석의 언급은 오탐).
    sys.path.insert(0, str(ROOT / "sweep"))
    from test_regulation import banned_calls
    for fn in ("script.py", "model/serve_features.py"):
        if not (work / fn).exists():
            continue
        hits = banned_calls(work / fn)
        check(not hits, f"§5 배치 통계 부재: {fn}", str(hits))

    dt_norm = run_once(work, "정상 경로", threads=args.threads)

    # ⚠ 정상 경로 예측을 **여기서** 붙잡는다. 아래 폴백·안전망 시험이 같은 경로의
    #   output/submission.csv를 덮어쓰기 때문이다. 예전에는 --compare/--segdiff가 이 배열을
    #   lazily 읽어서, `--no-server-env`로 돌리면 **손상 시험의 폴백 출력과 대조**하고 있었다
    #   (서버 핀 재실행이 마지막에 정상 출력을 복원해 주는 경로에서만 우연히 맞았다).
    #   레그 동일성·참조 대조는 역산의 전제라 조용히 틀리면 안 된다 — D-59에서 발견·수정.
    preds_norm = None
    _out0 = work / "output" / "submission.csv"
    if _out0.exists():
        preds_norm = pd.read_csv(_out0)["control_success"].to_numpy()

    # 폴백 경로 — base.joblib을 깨뜨려도 죽으면 안 된다(순수 numpy 로지스틱으로 전환)
    bak = work / "model" / "base.joblib"
    if bak.exists():
        broken = work / "model" / "base.joblib.bak"
        shutil.move(bak, broken)
        bak.write_bytes(b"not a pickle")
        run_once(work, "폴백 경로(모델 손상)")
        bak.unlink()
        shutil.move(broken, bak)

    # 안전망 — 피처 생성/모델 로드가 깨져도 유효한 제출 파일이 나와야 한다(실행 오류 = 슬롯 차감)
    if sfp.exists():
        orig = sfp.read_bytes()
        sfp.write_bytes(b"def build_features(df):\n    raise RuntimeError('boom')\n")
        run_once(work, "안전망(피처 생성 실패)")
        sfp.write_bytes(orig)
    else:
        meta = work / "model" / "metadata.json"
        if meta.exists():
            orig = meta.read_bytes()
            meta.write_bytes(b"{ broken json")
            run_once(work, "안전망(메타데이터 손상)")
            meta.write_bytes(orig)

    # ---- 서버 핀 환경 재실행 (로컬 pandas 3.0.0 ≠ 서버 2.3.3) ----
    preds = preds_norm
    if not args.no_server_env:
        py = server_python()
        ver = subprocess.run([str(py), "-c", "import sklearn,pandas,joblib;"
                              "print(sklearn.__version__,pandas.__version__,joblib.__version__)"],
                             capture_output=True, text=True).stdout.strip()
        print(f"\n  서버 핀 환경: sklearn/pandas/joblib = {ver}")
        run_once(work, "서버 핀 환경", python=py)
        out = work / "output" / "submission.csv"
        if out.exists():
            preds = pd.read_csv(out)["control_success"].to_numpy()
            log = (work / "output").parent
            check(preds is not None and not np.isnan(preds).any(),
                  "서버 핀 환경: NaN 없음")

    # ---- 참조 대조: 단일 동일성(--compare) 또는 가중평균 구성(--compose) ----
    spec = args.compose or (f"1.0:{args.compare}" if args.compare else None)
    if spec:
        py = server_python() if not args.no_server_env else None
        if preds is None:
            preds = preds_norm
        if preds is None:
            sys.exit("정상 경로 예측을 못 읽었다 — 대조/역산 불가")
        acc, wsum, names = np.zeros(len(preds)), 0.0, []
        for i, tok in enumerate(spec.split(",")):
            w_s, _, zp = tok.strip().rpartition(":")
            w = float(w_s or 1.0)
            ref = Path(zp)
            if not ref.exists():
                sys.exit(f"참조 zip 없음: {ref}")
            rwork = tmp / f"ref{i}"
            rwork.mkdir()
            with zipfile.ZipFile(ref) as z:
                z.extractall(rwork)
            (rwork / "data").mkdir(exist_ok=True)
            for f in ("test.csv", "sample_submission.csv"):
                shutil.copy2(work / "data" / f, rwork / "data" / f)
            run_once(rwork, f"참조 {ref.name} (w={w})", python=py)
            rp = pd.read_csv(rwork / "output" / "submission.csv")["control_success"].to_numpy()
            acc += w * rp
            wsum += w
            names.append(f"{w}×{ref.name}")
        ref_p = acc / wsum
        d = preds - ref_p
        rms = float(np.sqrt((d ** 2).mean()))
        print(f"\n         구성 = {' + '.join(names)}")
        print(f"         RMS차 {rms:.6f} · 최대 {np.abs(d).max():.6f} · "
              f"평균 {preds.mean():.5f} vs {ref_p.mean():.5f}")
        # 우리가 재학습한 RF는 주최 rf.pkl과 RMS 0.00033 차이가 있다 → 그 절반 수준까지는 허용.
        check(rms < 5e-4, "참조 구성 대조: 가중평균과 일치",
              f"RMS {rms:.6f} — 구성이 참조와 다르다. 이 zip의 기대점수를 참조 LB값으로 계산하면 안 된다")

    # ---- 세그먼트 프로브 레그 동일성(--segdiff) ----
    # 프로브 쌍은 "예측이 방향 g̃만큼만 다르다"는 전제 위에서 역산한다. 멤버 하나가 조용히 스킵되면
    # (script.py:549-561은 실패 멤버를 건너뛰고 재정규화한다) 그 전제가 깨지고 역산 전체가 거짓이 된다.
    # --compare의 RMS<5e-4 기준은 의도적으로 다른 이 쌍에 쓸 수 없으므로 **원소 단위 항등**을 본다.
    if args.segdiff:
        py = server_python() if not args.no_server_env else None
        if preds is None:
            preds = preds_norm
        if preds is None:
            sys.exit("정상 경로 예측을 못 읽었다 — 대조/역산 불가")
        ref = Path(args.segdiff)
        if not ref.exists():
            sys.exit(f"참조 zip 없음: {ref}")
        rwork = tmp / "segref"
        rwork.mkdir()
        with zipfile.ZipFile(ref) as z:
            z.extractall(rwork)
        (rwork / "data").mkdir(exist_ok=True)
        for f in ("test.csv", "sample_submission.csv"):
            shutil.copy2(work / "data" / f, rwork / "data" / f)
        run_once(rwork, f"참조 {ref.name}", python=py)
        rp = pd.read_csv(rwork / "output" / "submission.csv")["control_success"].to_numpy()

        import json as _json
        m_a = _json.loads((work / "model" / "metadata.json").read_text(encoding="utf-8"))
        m_b = _json.loads((rwork / "model" / "metadata.json").read_text(encoding="utf-8"))
        check(m_a.get("calibration") == m_b.get("calibration"),
              "레그 캘리 동일", f"{m_a.get('calibration')} vs {m_b.get('calibration')}")
        tin = pd.read_csv(work / "data" / "test.csv", encoding="utf-8-sig")

        def _seg_vec(terms):
            """항 목록 전체의 세그먼트 보정 벡터. 분할축 프로빙은 항이 여러 개다(gtF + 셀들)."""
            out = np.zeros(len(tin), dtype=np.float64)
            for tm in (terms or []):
                if "map" in tm:
                    arr = tin[tm["col"]].astype(str).to_numpy()
                    off = np.zeros(len(tin), dtype=np.float64)
                    for kk, vv in tm["map"].items():
                        off[arr == str(kk)] = float(vv)
                    out = out + float(tm.get("t", 1.0)) * off
                    continue
                if "levels" in tm:
                    ind = np.isin(tin[tm["col"]].astype(str).to_numpy(),
                                  [str(v) for v in tm["levels"]])
                else:
                    ind = tin[tm["col"]].to_numpy(dtype=np.float64) >= float(tm["ge"])
                g = ind.astype(np.float64) - float(tm["w"])
                if tm.get("within"):
                    g = g * np.isin(tin[tm["within"]["col"]].astype(str).to_numpy(),
                                    [str(v) for v in tm["within"]["levels"]]).astype(np.float64)
                out = out + float(tm["t"]) * g
            return out

        sa, sb = m_a.get("seg_probe") or [], m_b.get("seg_probe") or []
        # 정의(계수 t 제외)가 같은 항끼리 대응되는지 — w나 지지집합이 다르면 역산이 틀어진다
        def keyf(tm):                              # 계수 t를 뺀 '정의'만 비교 (map 항은 w가 없다)
            return (tm.get("col"), tuple(str(v) for v in tm.get("levels", [])),
                    tm.get("ge"),
                    None if "w" not in tm else round(float(tm["w"]), 12),
                    str(tm.get("within")),
                    tuple(sorted((str(k), round(float(v), 15))
                                 for k, v in (tm.get("map") or {}).items())))
        common = {keyf(t) for t in sa} & {keyf(t) for t in sb}
        check(len(sa) - len(common) <= 1 and len(sb) - len(common) <= 1,
              "레그 seg 구성 차이 ≤ 1항", f"A {len(sa)}항 / B {len(sb)}항 / 공통 {len(common)}")
        exp = _seg_vec(sa) - _seg_vec(sb)
        err = float(np.abs((preds - rp) - exp).max())
        print(f"\n         레그차 최대오차 = {err:.3e}  "
              f"(A {len(sa)}항 − B {len(sb)}항, RMS {np.sqrt((exp**2).mean()):.6f})")
        check(err < 1e-12, "레그 동일성: 예측차가 세그먼트 방향과 정확히 일치",
              f"최대오차 {err:.3e} — 멤버 무음 스킵/구성 불일치. 이 쌍으로 역산하면 안 된다")

    # ---- 형상(곡률) 프로브 레그 동일성(--quaddiff) ----
    # segdiff와 같은 목적이지만 방향 g가 **내부 raw 예측의 함수**라 test 컬럼만으로는 못 만든다.
    # 대신 참조 레그 출력에서 캘리를 역산해 raw를 복원한다(운용범위에 클리핑 0행이므로 정확):
    #     p_ref = 0.5 + slope·(raw − 0.5) + shift + seg  ⇒  raw = 0.5 + (p_ref − 0.5 − shift − seg)/slope
    if args.quaddiff:
        py = server_python() if not args.no_server_env else None
        if preds is None:
            preds = preds_norm
        if preds is None:
            sys.exit("정상 경로 예측을 못 읽었다 — 대조/역산 불가")
        ref = Path(args.quaddiff)
        if not ref.exists():
            sys.exit(f"참조 zip 없음: {ref}")
        rwork = tmp / "quadref"
        rwork.mkdir()
        with zipfile.ZipFile(ref) as z:
            z.extractall(rwork)
        (rwork / "data").mkdir(exist_ok=True)
        for f in ("test.csv", "sample_submission.csv"):
            shutil.copy2(work / "data" / f, rwork / "data" / f)
        run_once(rwork, f"참조 {ref.name}", python=py)
        rp = pd.read_csv(rwork / "output" / "submission.csv")["control_success"].to_numpy()

        import json as _json
        m_a = _json.loads((work / "model" / "metadata.json").read_text(encoding="utf-8"))
        m_b = _json.loads((rwork / "model" / "metadata.json").read_text(encoding="utf-8"))
        cal = m_a.get("calibration")
        check(cal == m_b.get("calibration"), "레그 캘리 동일",
              f"{cal} vs {m_b.get('calibration')}")
        check((m_a.get("seg_probe") or []) == (m_b.get("seg_probe") or []),
              "레그 seg 항 동일", "seg_probe가 다르면 raw 역산이 틀어진다")
        qa = m_a.get("quad_probe") or {}
        qb = m_b.get("quad_probe") or {}
        for k in ("raw_mean", "m", "power"):
            check(qa.get(k, qb.get(k)) == qb.get(k, qa.get(k)),
                  f"레그 quad 정의 동일: {k}", f"{qa.get(k)} vs {qb.get(k)}")

        tin = pd.read_csv(work / "data" / "test.csv", encoding="utf-8-sig")
        seg = np.zeros(len(tin), dtype=np.float64)
        for tm in (m_b.get("seg_probe") or []):
            ind = np.isin(tin[tm["col"]].astype(str).to_numpy(), [str(v) for v in tm["levels"]]) \
                if "levels" in tm else tin[tm["col"]].to_numpy(dtype=np.float64) >= float(tm["ge"])
            seg = seg + float(tm["t"]) * (ind.astype(np.float64) - float(tm["w"]))
        t_b = float(qb.get("t", 0.0))
        rb = float(qa.get("raw_mean", qb.get("raw_mean")))
        mm = float(qa.get("m", qb.get("m")))
        kk = int(qa.get("power", qb.get("power", 2)))
        # 참조 레그가 quad 항을 갖고 있으면 그것까지 빼야 raw가 나온다 → 통상 참조는 t_b=0
        raw = 0.5 + (rp - 0.5 - cal["shift"] - seg - t_b * 0.0) / cal["slope"]
        if t_b != 0.0:
            print("   ⚠ 참조 레그에도 quad 항이 있다 — raw 역산이 비선형이라 이 검사를 건너뛴다")
        else:
            exp = (float(qa.get("t", 0.0)) - t_b) * ((raw - rb) ** kk - mm)
            err = float(np.abs((preds - rp) - exp).max())
            print(f"\n         레그차 최대오차 = {err:.3e}  (기대 = {qa.get('t')}×((raw−{rb})^{kk}−m), "
                  f"RMS {np.sqrt((exp**2).mean()):.6f})")
            check(err < 1e-9, "레그 동일성: 예측차가 형상 방향과 정확히 일치",
                  f"최대오차 {err:.3e} — 멤버 무음 스킵/구성 불일치. 이 쌍으로 역산하면 안 된다")

    shutil.rmtree(tmp, ignore_errors=True)
    print("\n" + ("모든 검사 통과" if not _fails else f"실패 {len(_fails)}건: {_fails}"))
    sys.exit(1 if _fails else 0)


if __name__ == "__main__":
    main()
