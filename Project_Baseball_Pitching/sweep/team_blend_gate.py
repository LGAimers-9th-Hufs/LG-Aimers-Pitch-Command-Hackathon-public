# -*- coding: utf-8 -*-
"""팀 블렌드 게이트 — 팀원 예측 CSV 도착 즉시 s 실측·w*·이득 곡선·항등식 검산.

    python sweep/team_blend_gate.py --csv results/dual/team_pred_2024.csv --val 2024
    python sweep/team_blend_gate.py --csv a.csv,b.csv --val 2024,2023

CSV 계약: 컬럼 (row_id, pred) 또는 (row_id, control_success). pred = 팀원 **최종 배포 형태**
(자체 캘리 적용 후) 확률. row_id는 train.csv 값 그대로.

우리 레그 q_A = **ENS-9**(챔피언 1009.26) 배포 확률(동결 캘리 적용 후). 블렌드는 확률 평균 —
추가 캘리 없음. 캐시가 없으면 먼저 `python sweep/ens9_cache.py`.

판정 참고치:
- 수렴 법칙: s ≤ 0.0134 (독립 파이프라인 B 실측)이면 "다른 정보" 주장 기각.
- 항등식: Score(p̄) 이득 = C_v·w(1−w)·s²,  C_v = 1e5/(r(1−r)).  w=0.5 검산 필수.
- 점수는 max(0,·) 미적용 raw_brier_points (음수 폴드 비교용 — V23).
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import real_data as rd            # noqa: E402
import lgbm_family as L           # noqa: E402
from eb_carrier import (get_members9, ens9_pred,   # noqa: E402
                        deploy_q, DEPLOY_CAL, DEPLOY_SEG)
OUT_DIR = HERE.parent / "results" / "dual"
WSHOW = (0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5)


def affine_oracle(y, p):
    """이 폴드에서 **아핀(레벨+기울기) 재캘리의 정확 최적**과 그 점수.

    왜 이게 필요한가 (D-59): 팀 블렌드의 최대 리스크로 알려져 있던 것은 "파트너가 전 시즌으로
    학습했으면 mean(p_B) − r ≈ 0.06이라 w=0.5에서 −362점인데 로컬 게이트가 못 본다"였다.
    그런데 블렌드 뒤 레벨·기울기를 **LB로 다시 프로빙하면**(D-57/D-58이 그 축에서 +14.6을
    회수했고 꼭짓점 예측이 실측과 3e-7점 일치했다) 그 레벨 오차는 아핀에 그대로 흡수된다.
    ⇒ 파트너 심사에서 실제로 중요한 것은 레벨이 아니라 **형상 불일치**다.

    Brier는 MSE라 아핀 최적이 y~p OLS의 닫힌해다(클리핑 무시 — 운용범위에서 클리핑 0행).
    이 열은 "프로브 2슬롯을 쓰면 도달하는 점수"의 정직한 추정치이지 배포값이 아니다.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    vp = float(p.var())
    if vp <= 0:
        return float(y.mean()), 0.0, L.score(y, np.full_like(p, y.mean()))
    b = float(((p - p.mean()) * (y - y.mean())).mean() / vp)
    a = float(y.mean() - b * p.mean())
    q = np.clip(a + b * p, 1e-6, 1 - 1e-6)
    return a, b, L.score(y, q)


def proj_score(y, cols):
    """y를 span{1, *cols}에 OLS 사영한 예측의 점수 + 계수.

    **이 게이트의 판정 수치다.** 이유: 블렌드 뒤 우리가 실제로 하는 일은
    `q = a + b·((1−w)p_A + w·q_B)` 의 (a, b, w)를 **LB 프로브로 다시 잡는 것**이고,
    그 도달 가능 집합이 정확히 span{1, p_A, q_B}이다. 따라서

        gain_re = score(proj{1, p_A, q_B}) − score(proj{1, p_A})

    는 **현재 캘리 상수가 무엇이든 값이 변하지 않는** 불변량이고, "파트너에게 슬롯을
    쓸 가치가 있는가"의 직답이다. 고정 캘리 기준 이득표는 파트너의 레벨 오차에 오염되지만
    이 수치는 오염되지 않는다(레벨은 a가, 스케일은 b가 흡수한다).

    ⚠ 이것은 그 폴드의 라벨로 3계수를 맞춘 **오라클**이다. 2025에서는 LB 프로브가 같은
    3계수를 2025 라벨로 맞추므로 도달 가능하지만, 전제는 "q_B가 2025에도 신호를 나른다"는
    것이다. 그 전제의 증거는 **폴드 간 gain_re의 안정성**이지 한 폴드의 크기가 아니다.
    """
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, dtype=float) for c in cols])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    q = np.clip(X @ beta, 1e-6, 1 - 1e-6)
    return L.score(y, q), beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="쉼표 구분 팀원 예측 CSV 목록")
    ap.add_argument("--val", required=True, help="쉼표 구분, csv와 같은 순서의 val 시즌")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    csvs = [c.strip() for c in args.csv.split(",")]
    vals = [int(v) for v in args.val.split(",")]
    assert len(csvs) == len(vals), "--csv와 --val 개수 불일치"

    print(">> train.csv 로드 (정렬 기준)")
    df = rd.load_train()
    season = df[rd.SEASON].to_numpy()

    res = []
    for csv_path, v in zip(csvs, vals):
        val = season == v
        sub = df.loc[val, ["row_id", rd.TARGET, rd.SEGMENT]].reset_index(drop=True)
        y = sub[rd.TARGET].to_numpy(dtype=float)
        seg_v = sub[rd.SEGMENT].to_numpy()

        P, corr, corr_pm, y9 = get_members9(v)
        assert np.array_equal(y9, y), f"V{v}: npz y ≠ train.csv y — 행 순서 가정 붕괴"
        p_A_raw = ens9_pred(P, corr, corr_pm)
        q_A = deploy_q(p_A_raw, seg=seg_v)

        tb = pd.read_csv(csv_path)
        pred_col = "pred" if "pred" in tb.columns else rd.TARGET
        assert "row_id" in tb.columns and pred_col in tb.columns, f"{csv_path}: 컬럼 계약 위반"
        tb = tb.drop_duplicates("row_id")
        merged = sub.merge(tb[["row_id", pred_col]], on="row_id", how="left")
        n_miss = int(merged[pred_col].isna().sum())
        q_B = merged[pred_col].to_numpy(dtype=float)
        print(f"\n[val {v}] {csv_path} — 행 {len(tb):,} / 매칭 {len(merged)-n_miss:,} / 결측 {n_miss}")
        if n_miss:
            print(f"  ⚠ 결측 {n_miss}행 — 해당 행 q_A로 대체(보수적)")
            q_B = np.where(np.isnan(q_B), q_A, q_B)
        bad = int(((q_B < 0) | (q_B > 1)).sum())
        if bad:
            print(f"  ⚠ [0,1] 범위 밖 {bad}행 — clip")
            q_B = np.clip(q_B, 1e-6, 1 - 1e-6)

        r = y.mean()
        C_v = 1e5 / (r * (1 - r))
        S_A, S_B = L.score(y, q_A), L.score(y, q_B)
        d = q_B - q_A
        s_rms = float(np.sqrt(np.mean(d ** 2)))
        w_closed = float(np.clip(0.5 + (S_B - S_A) / (2 * C_v * s_rms ** 2), 0, 1)) \
            if s_rms > 0 else 0.0
        w_direct = float(np.clip(-np.mean((q_A - y) * d) / np.mean(d ** 2), 0, 1)) \
            if s_rms > 0 else 0.0

        line = dict(val=v, r=round(r, 4), S_A=round(S_A, 2), S_B=round(S_B, 2),
                    s_rms=round(s_rms, 5), w_closed=round(w_closed, 3),
                    w_direct=round(w_direct, 3))
        print(f"  S_A {S_A:.2f} · S_B {S_B:.2f} · s(RMS) {s_rms:.4f}"
              f"  → w*(닫힌형) {w_closed:.3f} / w*(직접) {w_direct:.3f}")
        if s_rms <= 0.0134:
            print("  🚫 수렴 법칙: s ≤ 0.0134 — '다른 정보' 주장 기각, 블렌드 무가치 영역")

        # 아핀 재캘리 후 = "블렌드 뒤 레벨·기울기를 LB 프로브 2슬롯으로 다시 잡으면" (D-59)
        _, _, S_A_re = affine_oracle(y, q_A)
        print(f"  [재캘리 기준] 앵커 단독 {S_A:.2f} → 아핀 최적 {S_A_re:.2f} "
              f"({S_A_re - S_A:+.2f} = 이 폴드에서 로컬이 못 보는 캘리 여유)")
        print(f"    {'w':>5} {'S(고정캘리)':>12} {'ΔA':>8} │ {'S(아핀재캘리)':>14} {'Δ재캘리':>9}")
        for w in list(WSHOW) + [w_direct]:
            q_w = (1 - w) * q_A + w * q_B
            Sw = L.score(y, q_w)
            _, _, Sw_re = affine_oracle(y, q_w)
            line[f"S_w{w:.2f}"] = round(Sw, 2)
            line[f"Sre_w{w:.2f}"] = round(Sw_re, 2)
            tagw = " ← w*" if abs(w - w_direct) < 1e-9 and w not in WSHOW else ""
            print(f"    {w:5.2f} {Sw:12.2f} {Sw-S_A:+8.2f} │ {Sw_re:14.2f} "
                  f"{Sw_re - S_A_re:+9.2f}{tagw}")
        # ★ 판정 수치 — 캘리 상수에 불변인 3계수 사영 (docstring proj_score 참조)
        S_proj_A, _ = proj_score(y, [p_A_raw])
        S_proj_AB, beta = proj_score(y, [p_A_raw, q_B])
        gain_re = S_proj_AB - S_proj_A
        w_impl = float(beta[2] / (beta[1] + beta[2])) if abs(beta[1] + beta[2]) > 1e-12 else float("nan")
        line["S_projA"] = round(float(S_proj_A), 2)
        line["S_projAB"] = round(float(S_proj_AB), 2)
        line["gain_re"] = round(float(gain_re), 2)
        line["w_implied"] = round(w_impl, 3)
        print(f"  ★ 사영 판정 (캘리 불변): 앵커만 {S_proj_A:.2f} → +파트너 {S_proj_AB:.2f}"
              f"  ⇒ **gain_re = {gain_re:+.2f}**  (함의 w ≈ {w_impl:.3f})")
        print(f"     고정캘리 기준 최고 이득 {max(line[f'S_w{w:.2f}'] for w in WSHOW)-S_A:+.2f} "
              f"— 이 값은 파트너 레벨오차에 오염되므로 판정에 쓰지 않는다")
        S_half_pred = 0.5 * (S_A + S_B) + C_v * s_rms ** 2 / 4
        gap = line["S_w0.50"] - S_half_pred
        print(f"  항등식 검산 w=0.5: 실측 {line['S_w0.50']:.2f} vs 예측 {S_half_pred:.2f}"
              f" (차 {gap:+.3f}) {'✅' if abs(gap) < 0.5 else '🚫 레그 정렬 의심'}")
        line["identity_gap"] = round(float(gap), 3)
        res.append(line)

    t = pd.DataFrame(res)
    print("\n" + t.to_string(index=False))
    (OUT_DIR / "team_blend_gate.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR/'team_blend_gate.json'}  ({time.time()-t0:.0f}s)")
    print(">> 다음 단계: 폴드 ≥2개면 LOFO w 선택 → 결합 zip 빌드(레그 파리티 --compose) → 슬롯 합의")


if __name__ == "__main__":
    main()
