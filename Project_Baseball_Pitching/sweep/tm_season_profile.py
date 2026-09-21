# -*- coding: utf-8 -*-
"""Phase 3 — 트랙맨 **시즌 단위 프로필 + 시즌 내 분위수 정규화** 재심 (D-17 재도전).

    python sweep/tm_season_profile.py --build            # 프로필만 생성 (results/phase3/)
    python sweep/tm_season_profile.py --gate             # 3폴드 게이트 (val 2024/2023/2022)

## D-17은 왜 기각됐고 무엇을 바꾸는가

기각 당시 프로필은 **커리어 누적**(시즌 s ← ≤s−1 전체)이었다. 이건 z_asof(−192)와 **같은 실패 형태**다:
누적량이 시즌마다 커져 분포가 이동하고, 학습 시즌의 support 밖 값이 예측 시즌에 나온다.
게다가 게이트가 refV24 단독(+11.4 < 2SE 18.6)이라 검정력이 없었다.

바꾸는 것 두 가지:

1. **시즌 단위 프로필** — 시즌 s 행에는 **s−1 한 시즌**의 트랙맨만 쓴다. 누적이 없다.
2. **시즌 내 분위수 정규화** — 값을 그 시즌 매칭 투수 집단 내 백분위(0~1)로 바꾼다.
   매 시즌 분포가 균일분포로 고정되므로 **out-of-support가 구조적으로 불가능**하다.
   (측정치 z는 이미 (시즌×구종군) 내 표준화라 장비 드리프트는 아래층에서 한 번 더 제거돼 있다.)

## 분해 설계 (무엇이 효과의 원인인지 분리한다)

| arm | 형태 | 답하는 질문 |
|---|---|---|
| `base` | 53피처 | 기준 |
| `t1_career_raw` | 커리어·원값 | **D-17 원형 재현** |
| `t1_career_rank` | 커리어·분위수 | 분위수화만의 효과 |
| `t1_season_raw` | 시즌·원값 | 시즌단위화만의 효과 |
| `t1_season_rank` | 시즌·분위수 | ★ 본안 |
| `t1_season_rank_d` | +전시즌 대비 변화량 | 추세(새 축) |
| `t12_season_rank` | Tier1+2 | 커버리지 확대 민감도(D-17에서 부호가 뒤집혔던 지점) |
| `cov_only` | 매칭 여부 지시자만 | **교락 대조** — 이게 이기면 이득은 물리량이 아니라 "매칭된 투수 = 1군 주력"이다 |

## 판정

- 3폴드(val 2024/2023/2022) × **paired SE**(같은 val 행의 제곱오차 차이 → 독립 2SE보다 검정력 高).
- 전체 val과 **커버 부분집합**을 분리 보고한다. 피처가 작동할 수 있는 층은 커버 층뿐이므로
  전체 델타는 커버율만큼 희석된다 — 두 수치가 같은 부호여야 진짜다.
"""
from __future__ import annotations
import argparse
import json
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
import real_data as rd          # noqa: E402
import lgbm_family as L         # noqa: E402
import trackman as TM           # noqa: E402
from season_centering import best_cal, paired_se, fit_pair, predict_pair   # noqa: E402

OUT_DIR = HERE.parent / "results" / "phase3"
FEATS = TM.TM_FEATURES                      # 6종
DELTAS = ["tm_movestd", "tm_velostd", "tm_velo_fb"]
MIN_SEASON_N = 100                          # 시즌 프로필 최소 투구수 (미만 → NaN)


# ---------------------------------------------------------------- 프로필 생성
def _pool(sub, m):
    """셀(시즌×구종군) 내 분산만 풀링 — 레퍼토리 구성 차이를 산포에 넣지 않는다."""
    n, s_, q = sub[f"n_{m}"], sub[f"s_{m}"], sub[f"q_{m}"]
    ok = n >= 2
    ss = (q - s_ * s_ / n.where(n > 0)).where(ok, 0.0)
    dof = (n - 1).where(ok, 0.0)
    g = sub["pitcher_trackman_id"]
    return ss.groupby(g).sum(), dof.groupby(g).sum()


def _features_from(c, r):
    """셀 집합 c(이미 기간 필터됨) → 투수별 6피처 원값."""
    out = pd.DataFrame(index=pd.Index(sorted(c["pitcher_trackman_id"].unique()),
                                      name="pitcher_trackman_id"))
    ss_v, dof_v = _pool(c, "velo")
    out["tm_velostd"] = np.sqrt(ss_v / dof_v.where(dof_v > 0))
    ss_i, dof_i = _pool(c, "ivb")
    ss_h, dof_h = _pool(c, "hb")
    out["tm_movestd"] = np.sqrt((ss_i + ss_h) / (dof_i + dof_h).where((dof_i + dof_h) > 0))

    fb = c[c["pitch_type_group"] == "fastball"].groupby("pitcher_trackman_id")
    os_ = c[c["pitch_type_group"] == "offspeed"].groupby("pitcher_trackman_id")
    fb_n, fb_v = fb["n_velo"].sum(), fb["s_velo"].sum()
    out["tm_velo_fb"] = fb_v / fb_n.where(fb_n >= TM.MIN_SEP_N)
    fb_sn, fb_sp = fb["n_spin"].sum(), fb["s_spin"].sum()
    out["tm_spin_fb"] = fb_sp / fb_sn.where(fb_sn >= TM.MIN_SEP_N)
    os_n, os_v = os_["n_velo"].sum(), os_["s_velo"].sum()
    out["tm_velosep"] = out["tm_velo_fb"] - (os_v / os_n.where(os_n >= TM.MIN_SEP_N))

    rr = r.groupby("pitcher_trackman_id")[["n_rows", "n_farm"]].sum()
    out["tm_farm_share"] = rr["n_farm"] / rr["n_rows"].where(rr["n_rows"] > 0)

    tot = c.groupby("pitcher_trackman_id")["n_velo"].sum().reindex(out.index).fillna(0)
    out.loc[tot < MIN_SEASON_N, FEATS] = np.nan
    out["tm_n"] = tot
    return out[FEATS + ["tm_n"]]


def build_profiles(tm_all, tiers=(1,), seasons=range(2019, 2025)):
    """{form: {(pid, src_season): [6값]}} — form ∈ {season_raw, season_rank, career_raw, career_rank}.

    `src_season`은 **프로필이 계산된 시즌**이다. 학습/서빙에서는 시즌 s 행에 src_season = s−1을 붙인다.
    분위수 정규화는 **그 시점에 사용 가능한 투수 집단 내부**에서만 계산한다 — 미래 정보가 없다.
    """
    mf = OUT_DIR.parent / "trackman" / "matches.csv"
    matches = pd.read_csv(mf)
    sel = matches[matches["tier"].isin(tiers)]
    pid_of = dict(zip(sel["tm_id"].astype(int), sel["pitcher_id"].astype(int)))
    print(f">> Tier{tuple(tiers)} 쌍 {len(sel)}/{len(matches)}")

    tm = tm_all[tm_all["pitcher_trackman_id"].isin(pid_of)]
    cells, rows = TM._measure_cells(tm)

    def _emit(f, src):
        f = f.dropna(how="all", subset=FEATS)
        raw, rank = {}, {}
        rk = f[FEATS].rank(pct=True, method="average")       # 시즌 내 백분위 (NaN 유지)
        for tid in f.index:
            pid = pid_of[int(tid)]
            raw[(pid, src)] = [float(v) for v in f.loc[tid, FEATS]]
            rank[(pid, src)] = [float(v) for v in rk.loc[tid]]
        return raw, rank

    out = {k: {} for k in ("season_raw", "season_rank", "career_raw", "career_rank")}
    for s in seasons:
        cs, rs = cells[cells[rd.SEASON] == s], rows[rows[rd.SEASON] == s]
        if len(cs):
            a, b = _emit(_features_from(cs, rs), s)
            out["season_raw"].update(a)
            out["season_rank"].update(b)
        cc, rc = cells[cells[rd.SEASON] <= s], rows[rows[rd.SEASON] <= s]
        if len(cc):
            a, b = _emit(_features_from(cc, rc), s)
            out["career_raw"].update(a)
            out["career_rank"].update(b)
        print(f"   {s}: season {sum(1 for k in out['season_raw'] if k[1]==s):4d}명 · "
              f"career {sum(1 for k in out['career_raw'] if k[1]==s):4d}명")
    return out


def attach(prof: dict, pid: np.ndarray, season: np.ndarray, cols=None):
    """시즌 s 행 ← 프로필(s−1). 반환 (DataFrame[6], 커버 마스크)."""
    cols = cols or FEATS
    n = len(pid)
    M = np.full((n, len(cols)), np.nan)
    keys = list(zip(pid.astype(int), (season - 1).astype(int)))
    for i, k in enumerate(keys):
        v = prof.get(k)
        if v is not None:
            M[i] = v
    return pd.DataFrame(M, columns=cols), ~np.isnan(M).all(axis=1)


def attach_delta(prof: dict, pid: np.ndarray, season: np.ndarray):
    """(s−1) − (s−2) 변화량. 두 시즌 다 있어야 값이 난다."""
    idx = [FEATS.index(c) for c in DELTAS]
    M = np.full((len(pid), len(DELTAS)), np.nan)
    for i, (p, s) in enumerate(zip(pid.astype(int), season.astype(int))):
        a, b = prof.get((p, s - 1)), prof.get((p, s - 2))
        if a is not None and b is not None:
            M[i] = [a[j] - b[j] for j in idx]
    return pd.DataFrame(M, columns=[f"d_{c}" for c in DELTAS])


# ---------------------------------------------------------------- 게이트
def gate(df, profs, val_season, arms):
    X = L.build_features(df)
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()
    pid = df["pitcher_id"].to_numpy()
    fit = (season == val_season - 1)
    val = (season == val_season)
    yv = y[val]
    print(f"\n[val {val_season}]  fit {fit.sum():,} → val {val.sum():,}  r={yv.mean():.4f}")

    res, preds = [], {}
    base_p = None
    # ⚠ 커버 마스크는 **arm과 무관하게** 프로필에서 한 번 정한다.
    #   (초판은 base arm(전부 True)에서 가져와 cover=1.00, Δ|cov ≡ Δ가 됐다 — 희석 분해가 죽었다.)
    cov_ref = attach(profs["season_rank"], pid, season)[1][val]
    print(f"  커버(season_rank, s−1 프로필 보유 행) = {cov_ref.mean():.3f}")
    for name, form, extra in arms:
        t = time.time()
        if form is None:
            XA = X
            cov = np.ones(len(X), dtype=bool)
        else:
            A, cov = attach(profs[form], pid, season)
            parts = [X.reset_index(drop=True), A]
            if extra == "delta":
                parts.append(attach_delta(profs[form], pid, season))
            if extra == "cov":
                parts = [X.reset_index(drop=True),
                         pd.DataFrame({"tm_cov": cov.astype(float)})]
            if extra == "noise":
                # 커버 패턴이 같은 **무의미 6컬럼** — "컬럼을 6개 더한 것"만의 효과를 잰다.
                rng = np.random.default_rng(1234)
                N = rng.random((len(X), len(FEATS)))
                N[~cov] = np.nan
                parts = [X.reset_index(drop=True),
                         pd.DataFrame(N, columns=[f"noise{i}" for i in range(len(FEATS))])]
            XA = pd.concat(parts, axis=1)
        bs = fit_pair(XA[fit], y[fit], None)
        p = np.clip(predict_pair(bs, XA[val]), 0, 1)
        preds[name] = p
        if base_p is None:
            base_p = p
        pc, bpc = L.calibrate(p), L.calibrate(base_p)
        bc, arg = best_cal(yv, p)
        # 커버 층만 따로 (희석 제거) — 같은 행 집합에서의 짝 비교
        m = cov_ref
        rec = dict(val=val_season, arm=name, ncol=XA.shape[1],
                   fixed_cal=L.score(yv, pc), best_cal=bc, cal_arg=str(arg),
                   d=L.score(yv, pc) - L.score(yv, bpc), se=paired_se(yv, pc, bpc),
                   d_cov=(L.score(yv[m], pc[m]) - L.score(yv[m], bpc[m])) if m.any() else np.nan,
                   se_cov=paired_se(yv[m], pc[m], bpc[m]) if m.any() else np.nan,
                   cover=float(m.mean()), std=float(p.std()),
                   rms=float(np.sqrt(((p - base_p) ** 2).mean())), sec=round(time.time() - t, 1))
        res.append(rec)
        print(f"  {name:20s} ({XA.shape[1]:2d}col)  고정 {rec['fixed_cal']:8.2f}  "
              f"재적합 {bc:8.2f}  Δ {rec['d']:+7.2f}±{rec['se']:.1f}  "
              f"Δ|cov {rec['d_cov']:+7.2f}±{rec['se_cov']:.1f}  "
              f"cover {rec['cover']:.2f}  rms {rec['rms']:.4f}  [{rec['sec']}s]")
    return res, preds, yv


def shuffle_profile(prof: dict, seed=0):
    """**위약 프로필** — 시즌 내에서 값 벡터를 투수 사이에 무작위 재배치한다.

    주변분포·NaN 패턴·커버리지가 전부 동일하고 **투수↔프로필 대응만 파괴**된다.
    따라서 진짜 프로필과 위약의 차이가 곧 "매칭이 실어 나른 정보"다.
    둘이 같은 크기로 움직이면 우리가 재던 것은 정보가 아니라 컬럼 추가 섭동이었다는 뜻이다.
    """
    rng = np.random.default_rng(seed)
    out = {}
    src = {}
    for (pid, s), v in prof.items():
        src.setdefault(s, []).append((pid, v))
    for s, items in src.items():
        pids = [p for p, _ in items]
        vals = [v for _, v in items]
        for p, j in zip(pids, rng.permutation(len(vals))):
            out[(p, s)] = vals[j]
    return out


ARMS = [
    ("base",              None,          None),
    ("cov_only",          "season_rank", "cov"),
    ("t1_career_raw",     "career_raw",  None),
    ("t1_career_rank",    "career_rank", None),
    ("t1_season_raw",     "season_raw",  None),
    ("t1_season_rank",    "season_rank", None),
    ("t1_season_rank_d",  "season_rank", "delta"),
]


def recheck(df, profs, vals):
    """저장된 예측(preds_val*.npz)으로 **커버 층 분해**만 다시 계산한다 — 재학습 없음.

    피처는 커버된 행에서만 값을 가지므로, 전체 val의 Δ는 커버율만큼 희석된다.
    커버 층에서 크게 양수인데 전체가 0 근처면 "희석된 진짜"이고,
    커버 층에서도 0이면 채널 자체가 없는 것이다. 비커버 층의 Δ는 **부작용**(0이어야 정상)이다.
    """
    season = df[rd.SEASON].to_numpy()
    pid = df["pitcher_id"].to_numpy()
    masks = {k: attach(profs[k], pid, season)[1] for k in ("season_rank", "career_rank")}
    if "t12_season_rank" in profs:
        masks["t12_season_rank"] = attach(profs["t12_season_rank"], pid, season)[1]
    out = []
    for v in vals:
        f = OUT_DIR / f"preds_val{v}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        val = (season == v)
        yv = z["y"]
        bpc = L.calibrate(z["base"])
        print(f"\n[val {v}] 커버율  " + "  ".join(
            f"{k}={m[val].mean():.3f}" for k, m in masks.items()))
        for arm in z.files:
            if arm in ("y", "base"):
                continue
            key = ("career_rank" if "career" in arm else
                   "t12_season_rank" if arm.startswith("t12") else "season_rank")
            m = masks[key][val]
            pc = L.calibrate(z[arm])
            rec = dict(val=v, arm=arm, cover=float(m.mean()),
                       d_all=L.score(yv, pc) - L.score(yv, bpc),
                       d_cov=L.score(yv[m], pc[m]) - L.score(yv[m], bpc[m]),
                       se_cov=paired_se(yv[m], pc[m], bpc[m]),
                       d_unc=L.score(yv[~m], pc[~m]) - L.score(yv[~m], bpc[~m]),
                       se_unc=paired_se(yv[~m], pc[~m], bpc[~m]))
            out.append(rec)
            print(f"  {arm:20s} cover {rec['cover']:.3f}  Δ전체 {rec['d_all']:+7.2f}  "
                  f"Δ커버 {rec['d_cov']:+7.2f}±{rec['se_cov']:.1f}  "
                  f"Δ비커버 {rec['d_unc']:+7.2f}±{rec['se_unc']:.1f}")
    t = pd.DataFrame(out)
    print("\n[커버 층 3폴드 요약]")
    print(t.groupby("arm").agg(cover=("cover", "mean"), d_cov_mean=("d_cov", "mean"),
                               d_cov_min=("d_cov", "min"), pos=("d_cov", lambda s: int((s > 0).sum())),
                               d_unc_mean=("d_unc", "mean"))
          .sort_values("d_cov_mean", ascending=False).to_string(float_format=lambda v: f"{v:.2f}"))
    (OUT_DIR / "recheck.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--recheck", action="store_true",
                    help="저장된 예측으로 커버 층 분해만 재계산 (재학습 없음)")
    ap.add_argument("--vals", default="2024,2023,2022")
    ap.add_argument("--t12", action="store_true", help="Tier1+2 커버리지 확대본도 함께")
    ap.add_argument("--placebo", action="store_true",
                    help="위약 대조만 실행 (무의미 6컬럼 + 시즌내 셔플 프로필)")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    pf = OUT_DIR / "profiles.json"
    if args.build or not pf.exists():
        print(">> trackman_history.csv 로드 (1회)")
        tm_all = TM.load_trackman()
        P = build_profiles(tm_all, tiers=(1,))
        obj = {k: {f"{a}|{b}": v for (a, b), v in d.items()} for k, d in P.items()}
        P2 = build_profiles(tm_all, tiers=(1, 2))          # 커버리지 확대 민감도용
        obj["t12_season_rank"] = {f"{a}|{b}": v for (a, b), v in P2["season_rank"].items()}
        del tm_all, P, P2
        pf.write_text(json.dumps(obj), encoding="utf-8")
        print(f">> 프로필 저장 {pf} ({pf.stat().st_size/1e6:.1f}MB, {time.time()-t0:.0f}s)")
    if not (args.gate or args.recheck):
        return

    obj = json.loads(pf.read_text(encoding="utf-8"))
    profs = {k: {(int(a.split("|")[0]), int(a.split("|")[1])): v for a, v in d.items()}
             for k, d in obj.items()}
    arms = list(ARMS) + ([("t12_season_rank", "t12_season_rank", None)]
                         if "t12_season_rank" in profs else [])
    if args.placebo:
        # 위약 3종만 돌린다(base 포함). 진짜 arm의 폴드 변동이 정보인지 섭동인지 가르는 대조.
        for k, seed in (("season_rank", 0), ("career_rank", 1)):
            profs[f"{k}_shuf"] = shuffle_profile(profs[k], seed)
        arms = [("base", None, None),
                ("noise6", "season_rank", "noise"),
                ("t1_season_rank_shuf", "season_rank_shuf", None),
                ("t1_career_rank_shuf", "career_rank_shuf", None)]

    print(">> train.csv 로드")
    if args.recheck and not args.gate:
        recheck(rd.load_train(), profs, [int(s) for s in args.vals.split(",")])
        return
    df = rd.load_train()
    allres = []
    sfx = "_placebo" if args.placebo else ""
    for v in [int(s) for s in args.vals.split(",")]:
        r, preds, yv = gate(df, profs, v, arms)
        allres += r
        np.savez_compressed(OUT_DIR / f"preds_val{v}{sfx}.npz", y=yv, **preds)

    t = pd.DataFrame(allres)
    print("\n" + "=" * 118)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n[3폴드 요약] arm별 Δ 평균 / 부호 일치")
    g = t[t.arm != "base"].groupby("arm").agg(
        d_mean=("d", "mean"), d_min=("d", "min"), d_cov_mean=("d_cov", "mean"),
        pos=("d", lambda s: int((s > 0).sum())), n=("d", "size"))
    print(g.sort_values("d_mean", ascending=False).to_string(float_format=lambda v: f"{v:.2f}"))
    (OUT_DIR / ("gate_placebo.json" if args.placebo else "gate.json")).write_text(json.dumps(allres, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR/'gate.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
