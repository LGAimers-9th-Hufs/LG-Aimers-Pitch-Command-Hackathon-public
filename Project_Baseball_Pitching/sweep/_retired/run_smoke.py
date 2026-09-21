"""
스모크 테스트 엔트리포인트: 합성 데이터로 하네스를 end-to-end 실행.
실행:  python sweep/run_smoke.py
실데이터 도착 시:  make_synthetic 대신 pd.read_parquet(...)로 교체하고 DataSpec 컬럼명만 맞추면 됨.

구성: (1) 플라시보 누수 테스트 → (2) S0 진단(P1 AV·P5) → (3) explore-exploit 스윕.
"""
import os, sys
try:  # Windows 콘솔 한글 깨짐 방지
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json

from config import DataSpec, make_synthetic
from sweep import run_sweep
from diagnostics import run_adversarial, cv_protocol_summary
from reporting import collect_meta, record_run
import test_leakage


def main():
    print("=" * 60)
    print(">> [0] 플라시보 누수 테스트 (fold-aware 피처 정합성)")
    print("=" * 60)
    test_leakage.main()

    print("\n>> 합성 pitch-level 데이터 생성 중...")
    df = make_synthetic(pitches_per_season=4000, seed=7)
    spec = DataSpec()

    print("\n" + "=" * 60)
    print(">> [S0] 사전 진단 (실데이터에선 sweep 전에 실행)")
    print("=" * 60)
    av = run_adversarial(df, spec)
    print()
    cv_folds = cv_protocol_summary(df, spec)

    print("\n" + "=" * 60)
    print(">> [sweep] explore-exploit, 시간분할 CV, LogLoss 게이트")
    print("=" * 60 + "\n")
    report = run_sweep(df, spec)

    # 기록: run.json/csv 저장 → 디스크에서 재로드해 figure 렌더(= run.json만으로 재생성 검증).
    meta = collect_meta(spec, seed=7, data_source="synthetic")
    run_dir = record_run(report, {"adversarial": av, "cv_folds": cv_folds}, meta)
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    try:
        from figures import render_all
        render_all(run, run_dir)
    except Exception as e:  # figure 실패(예: matplotlib 미설치)가 기록을 막으면 안 됨
        print(f"[record] figure 렌더 skip: {e}")
    print(f"[record] run 저장: {run_dir}")


if __name__ == "__main__":
    main()
