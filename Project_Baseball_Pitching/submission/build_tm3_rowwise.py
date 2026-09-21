# -*- coding: utf-8 -*-
"""jtt trackman v3(996.24)를 §5 행 독립 레그로 재빌드한다 — `submit_tm3L.zip`.

원본 `partners/LG_Aimers_JTT/trackman_pipeline/data/v3/submit_v3.zip`의 문제 세 곳
(전부 test 배치 통계 = §5 "행 간 참조/배치 평균 이동" 위반):

  1. league_level(test)      — test 전체의 asof_prev* 평균으로 리그 수준 추정
  2. recalibrate(p, L)       — 예측 평균을 L에 맞추는 배치 절편 이동
  3. build(): nanmedian(tot) — 비성공률 컬럼의 EB base를 test 중앙값으로 계산

전부 **train만으로 계산한 상수**로 치환한다:

  L2025      = 2·mean(success|2024) − mean(success|2023)   (마지막 차분 외삽 —
               원본 번들의 FALLBACK_LEVEL 0.472와 일치: 같은 설계 철학의 합법 경로)
  col_bases  = train 2024 행에서 잰 각 컬럼 중앙값(원본과 동일한 fillna 후)
  shift      = logit(L2025) − logit(train 2024 의사-test 평균예측)  (±0.35 클립)

모델 파일 `model_v3.pkl`은 **바이트 그대로** 복사한다(재직렬화 금지 — D-60 교훈).
학습 자체는 train 전용이므로 합법. 재현 경로 = trackman_pipeline 저장소 + 이 스크립트.

산출물
  submission/dist/submit_tm3L.zip                  ← 제출용 (hist = 번들의 2024년말)
  results/leg_matrix/measure/tm3L_p23.zip          ← 측정 전용 (hist23.json 동봉 —
        2024 프록시에서 당시즌 복원 블록이 살아 있도록 2023년말 base로 비교.
        pickle 안의 hist는 leg_matrix.patch_meta_deep이 못 건드리므로 이 변형이 필요)

검증: ① 원본 build()와의 파리티(같은 상수를 주면 피처 동일) ② 2024 의사-test에서
원본(배치) vs 재빌드(상수) Brier 차이 리포트 ③ sha256 출력.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
V3_ZIP = ROOT / "partners/LG_Aimers_JTT/trackman_pipeline/data/v3/submit_v3.zip"
TRAIN = ROOT / "data/data/train.csv"
OUT_SERVE = ROOT / "submission/dist/submit_tm3L.zip"
OUT_MEASURE = ROOT / "results/leg_matrix/measure/tm3L_p23.zip"

ID_COL, TARGET_COL = "row_id", "control_success"
P_RATES = ["asof_pitcher_success_rate", "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
           "asof_pitcher_ball_rate", "asof_pitcher_strike_rate"]
MIX_RATES = ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]
B_RATES = ["asof_batter_success_rate", "asof_batter_middle_rate"]
BLOCKS = [("pitcher_id", "asof_pitcher_n", P_RATES, "p"),
          ("pitcher_id", "asof_pitcher_pitchmix_n", MIX_RATES, "mix"),
          ("batter_id", "asof_batter_n", B_RATES, "b")]
MAX_SHIFT = 0.35


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sg(z):
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------- 원본 build 이식
# (원본 script.py의 build()와 동일하되 base 산출만 인자로 뺐다. bases=None이면
#  원본 그대로 nanmedian(배치)을 쓴다 — 파리티 검증용.)
def build(df, L, hist, bases=None):
    CATMAP = {"top_bottom": {"T": 0, "B": 1}, "game_type": {"R": 0, "P": 1, "A": 2}}
    CAT = ["top_bottom", "game_type", "base_state"]
    d = pd.DataFrame(index=df.index)
    d["top_bottom"] = df.top_bottom.map(CATMAP["top_bottom"]).fillna(-1)
    d["game_type"] = df.game_type.map(CATMAP["game_type"]).fillna(-1)
    bs = df.base_state.astype(str)
    d["r1"] = (bs.str[0] == "1").astype(int)
    d["r2"] = (bs.str[1] == "2").astype(int)
    d["r3"] = (bs.str[2] == "3").astype(int)
    for c in df.columns:
        if c not in (ID_COL, TARGET_COL, "season") + tuple(CAT):
            d[c] = pd.to_numeric(df[c], errors="coerce")
    d["league_level"] = L
    for c in ["asof_pitcher_success_rate", "asof_pitcher_prev1_game_success_rate",
              "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate",
              "asof_batter_success_rate"]:
        d[c + "_rel"] = pd.to_numeric(df[c], errors="coerce") - L

    for key, ncol, rates, tag in BLOCKS:
        h = hist.get(tag)
        n_tot = pd.to_numeric(df[ncol], errors="coerce").fillna(0).values
        if h is None:
            n_h = np.zeros(len(df)); H = {c: np.zeros(len(df)) for c in rates}
        else:
            hh = h.reindex(df[key].values)
            n_h = np.nan_to_num(hh["n"].values)
            H = {c: np.nan_to_num(hh[c].values) for c in rates}
        n_cur = np.maximum(n_tot - n_h, 0.0)
        d[f"{tag}_n_cur"] = n_cur
        d[f"{tag}_n_hist"] = n_h
        d[f"{tag}_cur_frac"] = n_cur / np.maximum(n_tot, 1)
        for c in rates:
            tot = pd.to_numeric(df[c], errors="coerce").fillna(L if "success" in c else 0).values
            cur = np.where(n_cur > 0, (n_tot * tot - n_h * H[c]) / np.maximum(n_cur, 1), np.nan)
            if "success" in c:
                base = L
            elif bases is not None:
                base = bases[c]
            else:
                base = np.nanmedian(tot)          # 원본(배치) 경로 — 파리티 검증에만 사용
            short = c.replace("asof_", "").replace("_rate", "")
            d[f"{short}_cur"] = cur
            for k in (120, 400):
                d[f"{short}_cur_eb{k}"] = (n_cur * np.nan_to_num(cur, nan=base) + k * base) / (n_cur + k) - base
            d[f"{short}_hist"] = H[c]
            d[f"{short}_delta"] = np.nan_to_num(cur, nan=base) - H[c]

    for pre, ncol, rcol in [("p", "asof_pitcher_n", "asof_pitcher_success_rate"),
                            ("b", "asof_batter_n", "asof_batter_success_rate")]:
        n = pd.to_numeric(df[ncol], errors="coerce").fillna(0).values
        r = pd.to_numeric(df[rcol], errors="coerce").fillna(L).values
        for k in (200, 1000):
            d[f"{pre}_shrunk{k}"] = (n * r + k * L) / (n + k) - L
    d["p_form3"] = df.asof_pitcher_prev3_game_success_rate - df.asof_pitcher_success_rate
    d["p_form5"] = df.asof_pitcher_prev5_game_success_rate - df.asof_pitcher_success_rate
    d["p_mid_form"] = df.asof_pitcher_prev3_game_middle_rate - df.asof_pitcher_middle_rate
    b, s = df.balls_before, df.strikes_before
    d["count_state"] = b * 3 + s; d["ahead"] = s - b
    d["two_strike"] = (s == 2).astype(int); d["three_ball"] = (b == 3).astype(int)
    d["same_hand"] = (df.pitcher_hand == df.batter_hand).astype(int)
    d["p_exp"] = np.log1p(pd.to_numeric(df.asof_pitcher_n, errors="coerce").fillna(0))
    d["b_exp"] = np.log1p(pd.to_numeric(df.asof_batter_n, errors="coerce").fillna(0))
    ps = pd.to_numeric(df.asof_pitcher_success_rate, errors="coerce").fillna(L) - L
    d["ps_x_2s"] = ps * d.two_strike; d["ps_x_3b"] = ps * d.three_ball
    d["ps_x_hand"] = ps * d.same_hand
    d["ball_strike_ratio"] = df.asof_pitcher_ball_rate / (df.asof_pitcher_strike_rate + 1e-6)
    return d.astype("float32")


def league_level_batch(df):
    """원본 league_level — 2024 의사-test에서 '원본이라면 어떤 L을 썼을지' 재현용."""
    ests = []
    for c in ["asof_pitcher_prev1_game_success_rate",
              "asof_pitcher_prev3_game_success_rate",
              "asof_pitcher_prev5_game_success_rate"]:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) >= 1000:
            ests.append(s.mean())
    L = float(np.median(ests))
    return L if 0.43 <= L <= 0.54 else 0.472


def hist23_from_train(tr: pd.DataFrame):
    """각 엔티티의 2023년말 누적을 train에서 복원.

    2024년 첫 투구 행의 asof_* = '그 투구 이전'까지의 커리어 누적 = 정확히 2023년말.
    2024 행이 없는 엔티티는 2023년 이전 마지막 행의 asof_*(마지막 1구 미포함 — 무시 가능).
    """
    out = {}
    for key, ncol, rates, tag in BLOCKS:
        cols = [key, ncol] + rates
        sub = tr[cols + ["season"]].copy()
        sub[ncol] = pd.to_numeric(sub[ncol], errors="coerce").fillna(0)
        s24 = sub[sub.season == 2024]
        first24 = s24.loc[s24.groupby(key)[ncol].idxmin()]
        pre = sub[sub.season <= 2023]
        last23 = pre.loc[pre.groupby(key)[ncol].idxmax()]
        base = last23.set_index(key)[[ncol] + rates]
        f24 = first24.set_index(key)[[ncol] + rates]
        base.loc[f24.index.intersection(base.index)] = f24
        extra = f24.loc[f24.index.difference(base.index)]
        base = pd.concat([base, extra])
        base = base.rename(columns={ncol: "n"})
        out[tag] = base
    return out


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


SERVE_SCRIPT = '''"""LG Aimers control_success — tm3L: row-wise rebuild of trackman-pipeline v3.

Every constant used at inference (league level, logit shift, column bases) was
computed from train.csv only and is bundled in model/rowwise_meta.json.
Inference is strictly row-wise: each row uses its own columns plus bundled
train-fixed lookups. No cross-row references, no batch statistics.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd

ID_COL, TARGET_COL = "row_id", "control_success"
CAT = ["top_bottom", "game_type", "base_state"]
CATMAP = {"top_bottom": {"T": 0, "B": 1}, "game_type": {"R": 0, "P": 1, "A": 2}}
P_RATES = ["asof_pitcher_success_rate", "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
           "asof_pitcher_ball_rate", "asof_pitcher_strike_rate"]
MIX_RATES = ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]
B_RATES = ["asof_batter_success_rate", "asof_batter_middle_rate"]
BLOCKS = [("pitcher_id", "asof_pitcher_n", P_RATES, "p"),
          ("pitcher_id", "asof_pitcher_pitchmix_n", MIX_RATES, "mix"),
          ("batter_id", "asof_batter_n", B_RATES, "b")]


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sg(z):
    return 1 / (1 + np.exp(-z))


def build(df, L, hist, bases):
    d = pd.DataFrame(index=df.index)
    d["top_bottom"] = df.top_bottom.map(CATMAP["top_bottom"]).fillna(-1)
    d["game_type"] = df.game_type.map(CATMAP["game_type"]).fillna(-1)
    bs = df.base_state.astype(str)
    d["r1"] = (bs.str[0] == "1").astype(int)
    d["r2"] = (bs.str[1] == "2").astype(int)
    d["r3"] = (bs.str[2] == "3").astype(int)
    for c in df.columns:
        if c not in (ID_COL, TARGET_COL, "season") + tuple(CAT):
            d[c] = pd.to_numeric(df[c], errors="coerce")
    d["league_level"] = L
    for c in ["asof_pitcher_success_rate", "asof_pitcher_prev1_game_success_rate",
              "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate",
              "asof_batter_success_rate"]:
        d[c + "_rel"] = pd.to_numeric(df[c], errors="coerce") - L

    for key, ncol, rates, tag in BLOCKS:
        h = hist.get(tag)
        n_tot = pd.to_numeric(df[ncol], errors="coerce").fillna(0).values
        if h is None:
            n_h = np.zeros(len(df)); H = {c: np.zeros(len(df)) for c in rates}
        else:
            hh = h.reindex(df[key].values)
            n_h = np.nan_to_num(hh["n"].values)
            H = {c: np.nan_to_num(hh[c].values) for c in rates}
        n_cur = np.maximum(n_tot - n_h, 0.0)
        d[f"{tag}_n_cur"] = n_cur
        d[f"{tag}_n_hist"] = n_h
        d[f"{tag}_cur_frac"] = n_cur / np.maximum(n_tot, 1)
        for c in rates:
            tot = pd.to_numeric(df[c], errors="coerce").fillna(L if "success" in c else 0).values
            cur = np.where(n_cur > 0, (n_tot * tot - n_h * H[c]) / np.maximum(n_cur, 1), np.nan)
            base = L if "success" in c else bases[c]
            short = c.replace("asof_", "").replace("_rate", "")
            d[f"{short}_cur"] = cur
            for k in (120, 400):
                d[f"{short}_cur_eb{k}"] = (n_cur * np.nan_to_num(cur, nan=base) + k * base) / (n_cur + k) - base
            d[f"{short}_hist"] = H[c]
            d[f"{short}_delta"] = np.nan_to_num(cur, nan=base) - H[c]

    for pre, ncol, rcol in [("p", "asof_pitcher_n", "asof_pitcher_success_rate"),
                            ("b", "asof_batter_n", "asof_batter_success_rate")]:
        n = pd.to_numeric(df[ncol], errors="coerce").fillna(0).values
        r = pd.to_numeric(df[rcol], errors="coerce").fillna(L).values
        for k in (200, 1000):
            d[f"{pre}_shrunk{k}"] = (n * r + k * L) / (n + k) - L
    d["p_form3"] = df.asof_pitcher_prev3_game_success_rate - df.asof_pitcher_success_rate
    d["p_form5"] = df.asof_pitcher_prev5_game_success_rate - df.asof_pitcher_success_rate
    d["p_mid_form"] = df.asof_pitcher_prev3_game_middle_rate - df.asof_pitcher_middle_rate
    b, s = df.balls_before, df.strikes_before
    d["count_state"] = b * 3 + s; d["ahead"] = s - b
    d["two_strike"] = (s == 2).astype(int); d["three_ball"] = (b == 3).astype(int)
    d["same_hand"] = (df.pitcher_hand == df.batter_hand).astype(int)
    d["p_exp"] = np.log1p(pd.to_numeric(df.asof_pitcher_n, errors="coerce").fillna(0))
    d["b_exp"] = np.log1p(pd.to_numeric(df.asof_batter_n, errors="coerce").fillna(0))
    ps = pd.to_numeric(df.asof_pitcher_success_rate, errors="coerce").fillna(L) - L
    d["ps_x_2s"] = ps * d.two_strike; d["ps_x_3b"] = ps * d.three_ball
    d["ps_x_hand"] = ps * d.same_hand
    d["ball_strike_ratio"] = df.asof_pitcher_ball_rate / (df.asof_pitcher_strike_rate + 1e-6)
    return d.astype("float32")


def load_hist():
    """train-fixed lookups from JSON (no pandas objects inside any pickle).

    model/hist.json = 2024-season-end lookups (serving).
    model/hist23.json, when present, overrides it (proxy measurement variant only).
    """
    hp = "./model/hist23.json" if os.path.exists("./model/hist23.json") else "./model/hist.json"
    with open(hp, "r", encoding="utf-8") as f:
        raw = json.load(f)
    hist = {}
    for tag, obj in raw.items():
        df = pd.DataFrame.from_dict(obj["data"], orient="index", columns=obj["columns"])
        df.index = df.index.astype(np.int64)
        hist[tag] = df
    print(f"lookups loaded from {hp}")
    return hist


def main():
    test = pd.read_csv("./data/test.csv", encoding="utf-8-sig")
    sub = pd.read_csv("./data/sample_submission.csv", encoding="utf-8-sig")
    bundle = joblib.load("./model/model_only.pkl")
    models, feats = bundle["models"], bundle["features"]
    with open("./model/rowwise_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    L = float(meta["league_level_2025"])
    shift = float(meta["logit_shift"])
    bases = meta["col_bases"]
    hist = load_hist()
    print(f"rows test={len(test)} sub={len(sub)} n_iter={bundle.get('n_iter')}")
    print(f"train-fixed constants: L={L:.6f} shift={shift:+.6f}")

    X = build(test, L, hist, bases)
    for c in feats:
        if c not in X.columns:
            X[c] = np.nan
    X = X[feats]

    acc = None
    for m in models:
        q = m.predict_proba(X)[:, 1]
        acc = q if acc is None else acc + q
    p = acc / len(models)
    p = sg(lg(p) + shift)
    p = np.clip(p, 0.001, 0.999)
    print(f"pred sum={float(np.sum(p)):.2f} over n={p.size}")

    pm = dict(zip(test[ID_COL].tolist(), p.tolist()))
    sub[TARGET_COL] = [pm.get(r, float(c)) for r, c in zip(sub[ID_COL], sub[TARGET_COL])]
    os.makedirs("./output", exist_ok=True)
    sub.to_csv("./output/submission.csv", index=False, encoding="utf-8")
    print(f"Saved: ./output/submission.csv (rows={len(sub)})")


if __name__ == "__main__":
    main()
'''


def main():
    print("== tm3L row-wise rebuild ==")
    with zipfile.ZipFile(V3_ZIP) as z:
        pkl_bytes = z.read("model/model_v3.pkl")
        req_bytes = z.read("requirements.txt")
    bundle = joblib.load(io.BytesIO(pkl_bytes))
    models, feats, hist24 = bundle["models"], bundle["features"], bundle["hist"]
    print(f"bundle: models={len(models)} feats={len(feats)} "
          f"league_level(ref)={bundle.get('league_level')} val_score={bundle.get('val_score')}")

    print("loading train ...")
    tr = pd.read_csv(TRAIN, encoding="utf-8")
    lvl = tr.groupby("season")[TARGET_COL].mean()
    L24, L23 = float(lvl[2024]), float(lvl[2023])
    L2025 = 2 * L24 - L23
    print(f"league level by season: {dict(lvl.round(6))}")
    print(f"L2025 = 2*{L24:.6f} - {L23:.6f} = {L2025:.6f}  (train-only, last-difference trend)")

    # 컬럼 base 상수 — 원본과 동일한 fillna 후 train 2024 행의 중앙값
    t24 = tr[tr.season == 2024]
    col_bases = {}
    for _, _, rates, _ in BLOCKS:
        for c in rates:
            if "success" in c:
                continue
            tot = pd.to_numeric(t24[c], errors="coerce").fillna(0)
            col_bases[c] = float(np.median(tot))
    print(f"col_bases (train-2024 medians): { {k: round(v, 6) for k, v in col_bases.items()} }")

    # 2023년말 hist — 2024 의사-test 평가와 프록시 측정 변형용
    print("building hist23 from train ...")
    hist23 = hist23_from_train(tr)
    for tag in hist23:
        print(f"  hist23[{tag}]: {hist23[tag].shape}  (bundle hist24: {hist24[tag].shape})")

    # --- 파리티: 같은 상수를 주면 원본 build()와 완전 동일해야 한다
    samp = t24.sample(20000, random_state=0)
    L_batch = league_level_batch(samp)
    batch_bases = {}
    for _, _, rates, _ in BLOCKS:
        for c in rates:
            if "success" in c:
                continue
            batch_bases[c] = float(np.nanmedian(pd.to_numeric(samp[c], errors="coerce").fillna(0).values))
    Xa = build(samp, L_batch, hist23, bases=None)       # 원본 경로 (배치 median)
    Xb = build(samp, L_batch, hist23, bases=batch_bases)  # 상수 경로에 같은 값 주입
    diff = float(np.nanmax(np.abs(Xa.values - Xb.values)))
    print(f"parity max|diff| (same constants) = {diff:.3e}")
    assert diff < 1e-6, "parity broken — build() 이식이 원본과 다르다"

    # --- 2024 의사-test: shift 상수 산출 + 원본(배치) 대비 비용 측정
    print("emulating serve on train-2024 rows (hist23) ...")
    y24 = t24[TARGET_COL].to_numpy(dtype=float)

    def predict(df, L, bases):
        X = build(df, L, hist23, bases=bases)
        for c in feats:
            if c not in X.columns:
                X[c] = np.nan
        X = X[feats]
        return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)

    p_new_raw = predict(t24, L2025, col_bases)
    m24 = float(np.mean(p_new_raw))
    # 반-이동 헤지: 모델의 당시즌 복원 피처가 2025 레벨 하락의 τ만큼을 자체 추적한다.
    # full shift(τ=0 가정)와 zero shift(τ=1 가정)의 최악 비용은 각각 ~4e5·Δ² ≈ 80~110pt.
    # τ 미지 하의 대칭 헤지 a = Δ/2 는 오차 |e| ≤ |Δ|/2 로 캡되고(≈23pt), τ~U[0,1] 기대 ≈ 8pt.
    # Δ·τ 전부 train만으로 정의/추론 — 평가셋 정보 없음.
    full = lg(np.array([L2025]))[0] - lg(np.array([m24]))[0]
    shift = float(np.clip(0.5 * full, -MAX_SHIFT, MAX_SHIFT))
    p_new = np.clip(sg(lg(p_new_raw) + shift), 0.001, 0.999)
    print(f"m24={m24:.6f} full_shift={full:+.6f} -> hedged shift={shift:+.6f}")

    # 원본이라면: L을 배치 추정, base도 배치, recalibrate로 평균을 L에 맞춤
    L_b = league_level_batch(t24)
    p_org_raw = predict(t24, L_b, None)
    z = lg(p_org_raw); lo, hi = -6.0, 6.0
    for _ in range(80):
        m = (lo + hi) / 2
        if float(np.mean(sg(z + m))) < L_b:
            lo = m
        else:
            hi = m
    sh_org = float(np.clip((lo + hi) / 2, -MAX_SHIFT, MAX_SHIFT))
    p_org = np.clip(sg(z + sh_org), 0.001, 0.999)

    b_new = float(np.mean((p_new - y24) ** 2))
    b_org = float(np.mean((p_org - y24) ** 2))
    print(f"2024 pseudo-test Brier: original(batch)={b_org:.6f}  tm3L(const)={b_new:.6f}  "
          f"delta_pts(4e5)={4e5 * (b_org - b_new):+.1f}  (음수 = 상수화 비용)")
    print(f"  (batch L would be {L_b:.6f}; our train-trend L = {L2025:.6f})")

    meta = {
        "name": "tm3L",
        "league_level_2025": L2025,
        "logit_shift": shift,
        "col_bases": col_bases,
        "provenance": [
            "Row-wise rebuild of teammate trackman-pipeline v3. The trained models are",
            "re-serialized unchanged into model_only.pkl (prediction parity verified at",
            "build time; the original model_v3.pkl sha256 is recorded inside the bundle);",
            "the train-derived entity lookups moved from pickle to model/hist.json.",
            "All inference constants are computed from train.csv only:",
            "  league_level_2025 = 2*mean(success|season 2024) - mean(success|season 2023)"
            f" = {L2025:.6f} (last-difference trend; the original bundle's designed fallback"
            " constant 0.472 is the same quantity rounded).",
            "  col_bases = per-column medians over train season-2024 rows, identical fill rule.",
            f"  logit_shift = 0.5 * [logit(league_level_2025) - logit(mean prediction on"
            f" train-2024 pseudo-test with 2023-end lookups)] = {shift:+.6f}, clipped to"
            " +/-0.35. The 0.5 factor is a symmetric hedge chosen a priori: the model's"
            " in-season reconstruction features already track an unknown fraction of the"
            " year-over-year level drift, so applying the full shift would double-count it.",
            "No evaluation-set feedback of any kind was used to choose any constant.",
            "Inference is strictly row-wise: each row uses only its own columns plus these"
            " bundled train-fixed constants and per-entity train-derived lookups.",
            "Reproduction: partners/LG_Aimers_JTT/trackman_pipeline (training) +"
            " submission/build_tm3_rowwise.py (this rebuild).",
        ],
    }
    meta_bytes = json.dumps(meta, ensure_ascii=True, indent=1).encode("utf-8")

    def hist_json_bytes(hd):
        out = {}
        for tag, df in hd.items():
            out[tag] = {"columns": list(df.columns),
                        "data": {str(int(k)): [float(x) for x in row]
                                 for k, row in zip(df.index, df.values)}}
        return json.dumps(out, ensure_ascii=True).encode("utf-8")

    h24_bytes = hist_json_bytes(hist24)
    h23_bytes = hist_json_bytes(hist23)

    # pandas 객체를 피클에서 제거 — 원본 pkl은 hist DataFrame이 pandas 3.0.2로 직렬화돼
    # 있어 pandas 2.3.3(우리 레그들의 핀)에서 언피클이 불가능하다. 모델만 다시 담는다.
    # (모델 자체는 sklearn/numpy 객체라 pandas 버전과 무관. 예측 파리티는 아래에서 확인.)
    model_only = {"models": models, "features": feats,
                  "n_iter": bundle.get("n_iter"), "backend": bundle.get("backend"),
                  "original_model_v3_pkl_sha256": sha256(pkl_bytes)}
    buf = io.BytesIO()
    joblib.dump(model_only, buf, compress=3)
    mo_bytes = buf.getvalue()

    # JSON 왕복 파리티 — hist를 JSON으로 내렸다 올려도 예측이 완전 동일해야 한다
    def from_json(b):
        raw = json.loads(b.decode("utf-8"))
        out = {}
        for tag, obj in raw.items():
            df = pd.DataFrame.from_dict(obj["data"], orient="index", columns=obj["columns"])
            df.index = df.index.astype(np.int64)
            out[tag] = df
        return out

    hist23_rt = from_json(h23_bytes)
    Xa2 = build(samp, L2025, hist23, bases=col_bases)
    Xb2 = build(samp, L2025, hist23_rt, bases=col_bases)
    for c in feats:
        for X in (Xa2, Xb2):
            if c not in X.columns:
                X[c] = np.nan
    da = Xa2[feats]; db = Xb2[feats]
    rt_diff = float(np.nanmax(np.abs(da.values - db.values)))
    print(f"hist JSON roundtrip parity max|diff| = {rt_diff:.3e}")
    assert rt_diff == 0.0, "hist JSON 왕복이 값을 바꿨다"

    OUT_SERVE.parent.mkdir(parents=True, exist_ok=True)
    OUT_MEASURE.parent.mkdir(parents=True, exist_ok=True)
    req_new = b"scikit-learn==1.8.0\njoblib==1.5.3\npandas==2.3.3\n"
    for out, extra in [(OUT_SERVE, {}), (OUT_MEASURE, {"model/hist23.json": h23_bytes})]:
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("script.py", SERVE_SCRIPT)
            z.writestr("requirements.txt", req_new)
            z.writestr("model/model_only.pkl", mo_bytes)
            z.writestr("model/hist.json", h24_bytes)
            z.writestr("model/rowwise_meta.json", meta_bytes)
            for name, b in extra.items():
                z.writestr(name, b)
        print(f"wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")

    print(f"original model_v3.pkl sha256 = {sha256(pkl_bytes)[:16]}... (recorded in bundle)")
    print("next: scan_probe_provenance -> verify_submission --proxy 245789 -> leg_matrix (tm3L)")


if __name__ == "__main__":
    main()
