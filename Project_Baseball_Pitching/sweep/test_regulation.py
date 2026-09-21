# -*- coding: utf-8 -*-
"""대회 규정(§5·§6) 준수와 train/serve 파리티를 **코드로 강제**하는 테스트.

    python sweep/test_regulation.py

기존 `test_leakage.py`(합성 데이터용 플라시보)를 대체한다. 실경로에서 증명해야 할 명제가
"미래 라벨이 새지 않았다"에서 **"어떤 행의 피처도 다른 행에 의존하지 않는다"**로 바뀌었기 때문이다
(주최측 `asof_*`는 이미 leave-current-out으로 계산돼 제공되므로 우리가 만들 이력 피처는 없다).
"""
from __future__ import annotations
import re, sys, io
from pathlib import Path

import numpy as np
import pandas as pd

try:  # Windows 콘솔 cp949 → 한글 출력 시 UnicodeEncodeError.
    # reconfigure를 쓴다: TextIOWrapper로 감싸면 이 모듈을 **import한** 쪽의 stdout이
    # 나중에 닫혀 "I/O operation on closed file"이 난다.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import real_data as rd  # noqa: E402

BANNED_CALLS = {"mean", "groupby", "value_counts", "transform", "rolling",
                "expanding", "cumsum", "rank", "shift", "nanmean", "median"}
_fails = []


def banned_calls(path: Path) -> list:
    """AST로 **실제 메서드 호출만** 검사한다(문자열/주석의 언급은 오탐이므로 제외).
    배치 전체를 보는 연산이 서빙 경로에 있으면 §5 위반이다."""
    import ast
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in BANNED_CALLS:
                hits.append(f"{node.func.attr}() @L{node.lineno}")
    return hits


def check(cond, name, detail=""):
    print(("  [OK]   " if cond else "  [FAIL] ") + name + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        _fails.append(name)
    return cond


def _sample(n=60_000):
    """시즌이 섞이도록 균등 샘플. nrows는 2019만 읽히므로 쓰지 않는다."""
    df = rd.load_train()
    per = max(2000, n // df[rd.SEASON].nunique())
    idx = np.concatenate([g.sample(min(len(g), per), random_state=7).index.to_numpy()
                          for _, g in df.groupby(rd.SEASON)])
    return df.loc[np.sort(idx)].reset_index(drop=True)


def test_row_independence(df):
    """§5의 기계적 증명: 한 행만 넣어 만든 피처가 전체 프레임에서 만든 그 행과 비트 동일해야 한다."""
    full = rd.build_features(df)
    rng = np.random.default_rng(0)
    picks = rng.choice(len(df), size=200, replace=False)
    bad = 0
    for i in picks:
        one = rd.build_features(df.iloc[[i]])
        a, b = one.to_numpy()[0], full.to_numpy()[i]
        if not np.array_equal(a, b, equal_nan=True):
            bad += 1
    check(bad == 0, "row independence: 단일 행 == 전체 프레임의 해당 행 (200행 표본)",
          f"불일치 {bad}건")

    # 부분집합(10%)으로 만들어도 동일해야 한다 — 배치 크기 의존성 탐지
    sub_idx = np.sort(rng.choice(len(df), size=len(df) // 10, replace=False))
    part = rd.build_features(df.iloc[sub_idx].reset_index(drop=True)).to_numpy()
    check(np.array_equal(part, full.to_numpy()[sub_idx], equal_nan=True),
          "row independence: 10% 부분집합 결과가 전체와 동일 (배치 크기 무관)")


def test_derived_not_allnan(df):
    """reindex 무음 NaN 가드 — 컬럼 생성이 누락돼도 reindex가 조용히 NaN 컬럼을 만들어 통과시킨다
    (문서 13 §5 함정 4). DERIVED 각 컬럼은 2020+ 표본에서 최소한 일부 값이 있어야 한다."""
    X = rd.build_features(df)
    sub = X[df[rd.SEASON].to_numpy() >= 2020]
    dead = [c for c in rd.DERIVED if c in sub and sub[c].notna().sum() == 0]
    check(not dead, "DERIVED 전 컬럼이 2020+ 표본에서 all-NaN 아님", f"죽은 컬럼 {dead}")
    tm_cols = [c for c in rd.DERIVED if c.startswith("tm_")]
    if tm_cols and rd.TM_PROFILE_IDX:
        cov = float(sub[tm_cols[0]].notna().mean())
        check(cov > 0.30, f"트랙맨 프로필 2020+ 행 커버리지 > 30%", f"{cov:.1%}")
        print(f"         (tm 프로필 커버리지 {cov:.1%}, 리터럴 {len(rd.TM_PROFILE_IDX):,} 엔트리)")


def test_target_permutation(df):
    """라벨을 섞어도 피처가 불변 — build_features가 target을 보지 않음."""
    a = rd.build_features(df).to_numpy()
    d2 = df.copy()
    d2[rd.TARGET] = np.random.default_rng(1).permutation(d2[rd.TARGET].to_numpy())
    check(np.array_equal(a, rd.build_features(d2).to_numpy(), equal_nan=True),
          "target permutation: 라벨 셔플에도 피처 불변")


def test_feature_order(df):
    """학습·추론이 공유하는 유일한 컬럼 계약."""
    tr = rd.build_features(df)
    te = rd.build_features(rd.load_test())
    check(list(tr.columns) == list(rd.FEATURE_ORDER), "FEATURE_ORDER: train 산출 순서 일치")
    check(list(te.columns) == list(rd.FEATURE_ORDER), "FEATURE_ORDER: test 산출 순서 일치")
    check(len(rd.FEATURE_ORDER) == len(set(rd.FEATURE_ORDER)), "FEATURE_ORDER: 중복 없음")
    check(not np.isinf(tr.to_numpy()[~np.isnan(tr.to_numpy())]).any(),
          "피처에 inf 없음 (sklearn은 NaN은 받아도 inf는 거부)")


def test_no_batch_statistics(tmp_path=None):
    """§5 재발 방지 가드: 서빙 경로에 배치 전체를 보는 연산이 없어야 한다."""
    import tempfile
    sys.path.insert(0, str(ROOT / "submission"))
    from build_submission import extract_serve_block
    tmp = Path(tempfile.mkdtemp()) / "serve_features.py"
    tmp.write_text(extract_serve_block(), encoding="utf-8")
    check(not banned_calls(tmp), "SERVE 블록에 배치 통계 연산 없음", str(banned_calls(tmp)))

    script = ROOT / "submission" / "script_template.py"
    if script.exists():
        hits = banned_calls(script)
        check(not hits, "script_template.py에 배치 통계 연산 없음", str(hits))


def test_serve_block_standalone():
    """SERVE 블록이 numpy/pandas만으로 단독 실행되는지(제출 zip에서 그대로 돈다)."""
    sys.path.insert(0, str(ROOT / "submission"))
    from build_submission import extract_serve_block
    ns = {}
    try:
        exec(compile(extract_serve_block(), "<serve>", "exec"), ns)
        ok = callable(ns.get("build_features")) and "FEATURE_ORDER" in ns
    except Exception as e:                                   # noqa: BLE001
        ok = False
        print("        ", repr(e))
    check(ok, "SERVE 블록 단독 실행 가능 (build_features·FEATURE_ORDER 정의)")


def test_asof_sanity(df):
    """주최 asof_* 의 구조 확인 — 실패 3유형이 사실상 분할인지(N2 근거)."""
    s = df["asof_pitcher_success_rate"]
    m = df["asof_pitcher_middle_rate"]
    r = df["asof_pitcher_reverse_rate"]
    resid = 1.0 - s - m - r
    ok_frac = float((resid[resid.notna()] >= -1e-6).mean())
    check(ok_frac > 0.99, "asof 3유형 분할: 1−succ−mid−rev ≥ 0 비율 > 99%",
          f"{ok_frac:.4%}")
    print(f"         (잔차 = '크게 벗어남' 비율 추정치, 중앙값 {float(resid.median()):.4f})")


def test_fold_hygiene(df):
    for f in rd.make_folds(df, val_seasons=(2023, 2024), mode="refit"):
        vs = f["val_season"]
        sets = {k: set(f[k].tolist()) for k in ("fit1", "cal", "val")}
        check(not (sets["fit1"] & sets["val"]), f"fold {vs}: fit1 ∩ val = ∅")
        check(not (sets["cal"] & sets["val"]), f"fold {vs}: cal ∩ val = ∅")
        smax = df[rd.SEASON].to_numpy()
        check(smax[f["fit1"]].max() < smax[f["cal"]].max() < smax[f["val"]].max(),
              f"fold {vs}: 시즌 순서 fit1 < cal < val")
        check(set(smax[f["fit2"]]) == set(x for x in np.unique(smax) if x < vs),
              f"fold {vs}: fit2 = val 이전 전체 (제출 동형 refit)")


def main():
    print("== 규정 준수 / 파리티 테스트 ==")
    print(">> 표본 로드")
    df = _sample()
    print(f"   {len(df):,}행\n")
    test_row_independence(df)
    test_derived_not_allnan(df)
    test_target_permutation(df)
    test_feature_order(df)
    test_no_batch_statistics()
    test_serve_block_standalone()
    test_asof_sanity(df)
    test_fold_hygiene(df)
    print("\n" + ("전부 통과" if not _fails else f"실패 {len(_fails)}건: {_fails}"))
    sys.exit(1 if _fails else 0)


if __name__ == "__main__":
    main()
