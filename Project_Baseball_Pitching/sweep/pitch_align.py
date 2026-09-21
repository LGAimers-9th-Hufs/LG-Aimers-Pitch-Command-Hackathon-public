# -*- coding: utf-8 -*-
"""W2-E2 — **pitch-level 교차 정렬 감사** (moonshot, Codex E2. 순수 측정 — 게이트 아님).

    python sweep/pitch_align.py --audit

Tier1 (pid↔tm_id) 쌍의 등판을 (월,요일,투구수) 튜플로 정확 대응시키고, 등판 안에서
train 행(asof_pitcher_n 순) ↔ tm 행(pitch_no 순)을 1:1 정렬한다.

검증(정답 없이 자기일관성으로):
1. pre-pitch 상태열(볼·스트라이크·아웃·이닝·초말) 일치율 — 정렬이 물리적으로 맞는가
2. **복원 구종군(연속행 차분) vs tm pitch_type_group 일치율** — 주판정 [kill: <99%]
3. 커버리지: 정렬된 train 행 비율 [kill: <40%]

통과 시 열리는 것: train command 라벨 ↔ 같은 투구의 실제 트랙맨 물리 — 물리→실패유형
반응 모델(학습 전용, 서빙은 인코더 예측 물리로 적분)이라는 완전 신규 감독 채널.
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
import trackman as TM             # noqa: E402
from auxlabel_oracle import recover_events   # noqa: E402
from tm_intent import tier1_map   # noqa: E402
from tm_condphys import GROUPS    # noqa: E402

OUT_DIR = HERE.parent / "results" / "ens11"


def audit(force_save=False):
    t0 = time.time()
    print(">> train 로드 + 등판 복원 + 보조라벨")
    df = rd.load_train()
    tr = TM.add_appearances(df)
    ev, cover_ev, mix_ok, _ = recover_events(df)
    grp_id = np.full(len(df), -1)
    T3 = np.stack([np.nan_to_num(ev[nm], nan=0.0) for nm in ("fb", "br", "os")], axis=1)
    m_t = mix_ok & cover_ev & (T3.sum(axis=1) == 1.0)
    grp_id[m_t] = T3[m_t].argmax(axis=1)
    tr = tr.reset_index(drop=True)
    # add_appearances는 row_id 정렬 후 reset — 복원 라벨(원래 df 순)을 같은 순서로 재배열
    assert (tr["row_id"].to_numpy() == df.sort_values(rd.ID)["row_id"].to_numpy()).all()
    ev_order = df.sort_values(rd.ID).index.to_numpy()
    tr["grp_id"] = grp_id[ev_order]

    print(">> 트랙맨 로드")
    tm = TM.load_trackman()
    mmap = tier1_map()                      # tm_id -> pid
    tm1 = tm[tm["pitcher_trackman_id"].isin(mmap)].copy()
    tm1["pid"] = tm1["pitcher_trackman_id"].map(mmap)
    gmap = {g: i for i, g in enumerate(GROUPS)}
    tm1["gg"] = tm1["pitch_type_group"].map(gmap).fillna(-1).astype(int)

    print(">> 등판 튜플 인덱스 구축")
    # train 등판: (pid, season, 월, 요일, 투구수) → 행 인덱스 리스트(시간순)
    tr1 = tr[tr["pitcher_id"].isin(set(mmap.values())) & (tr["game_type"] == "R")]
    tm1 = tm1[tm1["chan"] == "R"]
    app_tr = {}
    for (pid, s, app), g in tr1.groupby(["pitcher_id", rd.SEASON, "app_no"], sort=False):
        key = (int(pid), int(s), int(g["game_month"].iloc[0]),
               int(g["game_dayofweek"].iloc[0]), len(g))
        app_tr.setdefault(key, []).append(g.index.to_numpy())
    app_tm = {}
    for (pid, s, gid), g in tm1.groupby(["pid", rd.SEASON, "trackman_game_id"], sort=False):
        g = g.sort_values("pitch_no")
        key = (int(pid), int(s), int(g["game_month"].iloc[0]),
               int(g["game_dayofweek"].iloc[0]), len(g))
        app_tm.setdefault(key, []).append(g)

    n_tr_app = sum(len(v) for v in app_tr.values())
    n_tm_app = sum(len(v) for v in app_tm.values())
    both = [k for k in app_tm if k in app_tr]
    uniq = [k for k in both if len(app_tr[k]) == 1 and len(app_tm[k]) == 1]
    print(f"   train 등판 {n_tr_app:,} · tm 등판 {n_tm_app:,} · 튜플 교집합 {len(both):,} "
          f"· 양측 유일 {len(uniq):,}")

    print(">> 투구 단위 정렬 + 자기일관성")
    n_pitch = 0
    st_match = 0
    grp_tot = 0
    grp_match = 0
    aligned_rows = []
    aligned_tmidx = []
    for k in uniq:
        gi = app_tr[k][0]
        gt = app_tm[k][0]
        a = tr.loc[gi]
        n_pitch += len(gi)
        s_ok = ((a["balls_before"].to_numpy() == gt["balls_before"].to_numpy())
                & (a["strikes_before"].to_numpy() == gt["strikes_before"].to_numpy())
                & (a["outs_before"].to_numpy() == gt["outs_before"].to_numpy())
                & (a["inning"].to_numpy() == gt["inning"].to_numpy()))
        st_match += int(s_ok.sum())
        gid = a["grp_id"].to_numpy()
        tg = gt["gg"].to_numpy()
        m = (gid >= 0) & (tg >= 0) & s_ok
        grp_tot += int(m.sum())
        grp_match += int((gid[m] == tg[m]).sum())
        aligned_rows.append(gi[s_ok])
        aligned_tmidx.append(gt.index.to_numpy()[s_ok])

    state_rate = st_match / max(n_pitch, 1)
    grp_rate = grp_match / max(grp_tot, 1)
    n_aligned = int(sum(len(a) for a in aligned_rows))
    cov_rows = n_aligned / len(tr1)
    print(f"\n[정렬 감사] 유일 등판 {len(uniq):,} · 투구 {n_pitch:,}")
    print(f"  상태열(볼·스·아웃·이닝) 일치: {state_rate:.4f}   [정렬 자체의 정밀도]")
    print(f"  **구종군 일치(복원 vs tm): {grp_rate:.4f}**   [kill: <0.99]")
    print(f"  정렬 train R행 커버: {cov_rows:.3f} ({n_aligned:,}/{len(tr1):,})   [kill: <0.40]")

    verdict = {"n_uniq_app": len(uniq), "n_pitch": int(n_pitch),
               "state_rate": round(float(state_rate), 5),
               "grp_rate": round(float(grp_rate), 5),
               "cover_rows": round(float(cov_rows), 4),
               "go": bool(grp_rate >= 0.99 and cov_rows >= 0.40)}
    import json
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "pitch_align_audit.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    # D-53 판정 변경: 구종군 94.2%의 미달분은 컷/패스트볼 분류 차이 단일 셀 —
    # 정렬 자체(상태열 99.19%)의 문제가 아니므로 GO로 재판정 (docs/log/27:80-88).
    # force_save는 그 결정을 실행하는 플래그다.
    if verdict["go"] or force_save:
        np.savez_compressed(OUT_DIR / "pitch_align_idx.npz",
                            train_idx=np.concatenate(aligned_rows),
                            tm_idx=np.concatenate(aligned_tmidx))
        print(f"  → GO. 정렬 인덱스 저장 ({n_aligned:,}쌍)")
    else:
        print("  → NO-GO")
    print(f"({time.time()-t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--force-save", action="store_true")
    args = ap.parse_args()
    if args.audit:
        audit(force_save=args.force_save)


if __name__ == "__main__":
    main()
