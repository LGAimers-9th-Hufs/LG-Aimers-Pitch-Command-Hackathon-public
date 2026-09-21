"""
플라시보(future-scramble) 누수 테스트 — 실데이터 없이 as-of 누수 수정의 정확성을 증명.

원리: val 시즌 target을 무작위로 뒤섞어도, 올바른 fold-aware 피처(cutoff=val 시즌)라면
**target-파생 피처가 전혀 변하지 않아야** 한다(미래 정답을 안 봤다는 뜻).
  - 수정판(cutoff 지정): 통과(불변) 기대.
  - 현행 누수판(cutoff=None): val 피처가 변함 → 누수 존재를 시연.

실행:  python sweep/test_leakage.py
"""
import os, sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from config import DataSpec, make_synthetic
from features import build_features

# target에서 파생된 피처(누수 대상). 이들이 val target 셔플에 불변이어야 함.
TARGET_DERIVED = [
    "p_asof_rate", "p_asof_rate10", "p_asof_rate50", "b_asof_rate",
    "p_asof_n", "b_asof_n", "p_coldstart",
    "f15_count_grid", "f17_p_postvar", "f17_b_postvar",
    "f19_prev_success", "f19_recent_fail",
    "f27_press_delta", "f28_stadium_rate",   # 10 ctx 블록의 target-파생분
]


def _scramble_val_targets(df, spec, val_season, seed=123):
    df2 = df.copy()
    m = (df2[spec.season] == val_season).to_numpy()
    rng = np.random.default_rng(seed)
    vals = df2.loc[m, spec.target].to_numpy()
    df2.loc[m, spec.target] = rng.permutation(vals)
    return df2


def _val_block(df, spec, val_season, cutoff):
    X, _, meta = build_features(df, spec, rich=True, ctx=True, label_cutoff_season=cutoff)
    m = (meta[spec.season] == val_season).to_numpy()
    cols = [c for c in TARGET_DERIVED if c in X.columns]
    return X.loc[m, cols].to_numpy(), cols


def main():
    spec = DataSpec()
    df = make_synthetic(pitches_per_season=3000, seed=7)
    val_season = sorted(df[spec.season].unique())[-1]      # 2025 = hidden
    df_scr = _scramble_val_targets(df, spec, val_season)
    print(f">> 플라시보 테스트: val 시즌={val_season} target 셔플 → target-파생 val 피처 불변 검사\n")

    # 1) 수정판: cutoff=val_season → 불변이어야 통과
    a, cols = _val_block(df, spec, val_season, cutoff=val_season)
    b, _ = _val_block(df_scr, spec, val_season, cutoff=val_season)
    max_abs = float(np.max(np.abs(a - b)))
    ok_fixed = max_abs < 1e-9
    print(f"[수정판 cutoff={val_season}] val target-파생 피처 최대 변화 = {max_abs:.2e} "
          f"→ {'PASS(불변)' if ok_fixed else 'FAIL(누수)'}  (검사 컬럼 {len(cols)}개)")

    # 2) 현행 누수판: cutoff=None → 변해야(누수 시연)
    c, _ = _val_block(df, spec, val_season, cutoff=None)
    d, _ = _val_block(df_scr, spec, val_season, cutoff=None)
    max_abs_leak = float(np.max(np.abs(c - d)))
    leak_shown = max_abs_leak > 1e-6
    print(f"[누수판  cutoff=None ] val target-파생 피처 최대 변화 = {max_abs_leak:.2e} "
          f"→ {'누수 확인(변함)' if leak_shown else '변화 없음'}")

    # 3) 컬럼별 기여(누수판에서 어떤 피처가 새는지)
    if leak_shown:
        per = np.max(np.abs(c - d), axis=0)
        worst = sorted(zip(cols, per), key=lambda t: -t[1])[:5]
        print("   누수 상위 피처:", ", ".join(f"{n}({v:.3f})" for n, v in worst))

    print()
    assert ok_fixed, "수정판이 val target 셔플에 불변이 아님 → 누수 수정 실패"
    assert leak_shown, "누수판이 불변 → 합성 데이터에 누수 유발 구조가 없음(테스트 무의미)"
    print("✅ 플라시보 통과: fold-aware 피처는 누수 0, 현행 전역계산은 누수 있음(수정 정당성 증명).")


if __name__ == "__main__":
    main()
