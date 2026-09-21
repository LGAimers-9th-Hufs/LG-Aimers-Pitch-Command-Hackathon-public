# -*- coding: utf-8 -*-
"""실데이터(대회 배포본) 로더 + 규정 안전 피처 레이어 + 시간분할 폴드.

데이터: `data/data/{train,test,sample_submission,trackman_history}.csv` (open.zip 해제본)
  - train 1,475,092행 × 49컬럼 (= test 48컬럼 + control_success), 2019~2024
  - test는 배포본에 5행 샘플만. 실제 245,789행(2025)은 평가 서버에만 존재.

⚠ 대회 규정(data_description.md §5·§6)이 피처 설계를 강하게 제약한다:
  - **test.csv의 다른 행을 이용한 어떤 피처도 금지** (누적·빈도·분포·rolling·target encoding·
    전체를 보고 만든 사후 보정). "평가 데이터의 각 행은 독립적으로 예측해야 한다"고 명시.
    → 우리가 직접 계산하는 as-of 통계는 2025에서 **계산 자체가 불가능**하다.
       과거 이력은 주최측이 사전계산해 준 `asof_*` 컬럼(공식 입력)만 쓴다.
  - 허용: **행 단위(row-wise) 변환**, train/trackman(2019~24)으로 만든 **고정 룩업/상수**
    (추론 시 zip에 동봉해 조회 — test 행을 참조하지 않으므로 규정 위반이 아니다).
  - 현재 투구의 위치·판정·구종·트랙맨 측정값, 2025 트랙맨은 입력 금지.

`<<<SERVE:BEGIN>>>` ~ `<<<SERVE:END>>>` 블록은 `submission/build_submission.py`가 **바이트 그대로**
잘라 `model/serve_features.py`로 동봉한다. 학습과 추론이 문자 그대로 같은 코드를 쓰므로
train/serve skew가 구조적으로 0이며, `test_regulation.py::test_serve_parity`가 이를 강제한다.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "data"

ID = "row_id"
TARGET = "control_success"


# ================================ <<<SERVE:BEGIN>>> ================================
# 이 블록은 제출 zip에 그대로 복사된다.
# **자기완결이어야 한다** — 외부 의존은 numpy/pandas뿐이고, 여기서 쓰는 상수는 전부 여기서 정의한다.
# (블록 밖 상수를 참조하면 동봉된 serve_features.py가 NameError로 죽는다.)
import numpy as np
import pandas as pd

SEASON = "season"
SEGMENT = "game_type"          # R(정규) / F(퓨처스 추정) — 세그먼트 앵커의 키
EPS = 1e-9

# 리터럴/피처 로직이 바뀔 때마다 갱신 — run_real.py가 캐시 키에 섞어 낡은 npz 재사용(거짓 게이트
# 판독)을 차단한다. 값만 바뀌고 컬럼명이 같은 변경이 정확히 이 함정이다(docs/research/13 §5-3).
SERVE_VERSION = "base82-noz"

# --- Z_ASOF:BEGIN ---
# sweep/z_asof.py가 재생성한다 — 손으로 편집 금지. (pid*10000+season) → as-of 수축 로짓.
# D-19로 피처에서 제거되어 **비활성 자리채움**으로 복귀(테이블 생성 코드는 z_asof.py에 보존).
Z_ASOF_META = {'K': 500, 'rho': 0.85, 'n': 0, 'status': 'disabled by D-19'}
Z_ASOF = {}
# --- Z_ASOF:END ---

# --- SIGNFLIP:BEGIN ---
# sweep/donut.py가 재생성한다. N15는 D-16으로 기각 — 리터럴은 비활성 자리채움으로 복귀(코드는 donut.py 보존).
SF_LSIG0, SF_LSIG1, SF_NSIG = -0.5, 0.5, 2
SF_A0, SF_A1, SF_NA = 0.0, 1.0, 2
SF_GRID = np.array([[0.5, 0.5], [0.5, 0.5]], dtype='float32')
SF_SIGA = {}
SF_DA = {}
# --- SIGNFLIP:END ---

# --- TM_PROFILE:BEGIN ---
# sweep/trackman.py --profile 이 재생성한다 — 손으로 편집 금지.
# N13은 D-17로 미채택 → 비활성 자리채움으로 복귀(매칭·프로필 코드는 trackman.py에 보존,
#  Tier1 프로필 실물은 results/trackman/tm_profile_tier1.json).
TM_PROFILE_META = {}
TM_PROFILE_IDX = {}
TM_PROFILE_MAT = []
# --- TM_PROFILE:END ---

# 리터럴에서 파생하는 조회 dict (마커 밖 정적 코드 — splice에 안전)
SF_SIG = {k: v[0] for k, v in SF_SIGA.items()}
SF_AIM = {k: v[1] for k, v in SF_SIGA.items()}

# N13 트랙맨 프로필: 마지막 행을 전부-NaN 폴백으로 두면 미등재 키를 인덱스 하나로 흡수한다
# (조회가 순수 행 단위 유지 — 다른 행을 보지 않는다). Tier2/3·미매칭·2019행은 여기로 떨어진다.
TM_COLS = ["tm_movestd", "tm_velostd", "tm_velo_fb", "tm_farm_share", "tm_spin_fb", "tm_velosep"]
TM_MAT = np.array((TM_PROFILE_MAT if TM_PROFILE_MAT else []) + [[np.nan] * len(TM_COLS)],
                  dtype="float32")
TM_NA = len(TM_MAT) - 1

# 범주형 3종 — 추론 파리티를 위해 **고정 매핑**(OrdinalEncoder의 학습 의존성 제거).
CAT_MAPS = {
    "top_bottom": {"T": 0, "B": 1},
    "game_type": {"R": 0, "F": 1},
    "base_state": {"___": 0, "1__": 1, "_2_": 2, "__3": 3,
                   "12_": 4, "1_3": 5, "_23": 6, "123": 7},
}
CAT_COLS = list(CAT_MAPS)

# 공식 수치 44컬럼 — **순서를 여기서 고정**한다(df.columns 순회 금지: train/serve 순서 어긋남 방지).
OFFICIAL_NUM = [
    "season", "game_month", "game_dayofweek", "inning",
    "balls_before", "strikes_before", "outs_before",
    "run_top_before", "run_bot_before", "run_total_before",
    "score_diff_home", "score_diff_pitcher_team",
    "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
    "home_win_expectancy", "away_win_expectancy", "li",
    "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
    "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]

# 파생 35컬럼 — build_features가 만드는 순서와 정확히 일치해야 한다.
DERIVED = [
    "count_bucket", "count_diff", "ahead", "behind", "two_strike", "three_ball",
    "full_count", "first_pitch",
    "same_hand", "li_log", "high_li", "we_margin", "score_diff_abs", "close_game",
    "late_inning", "risp",
    "p_n_log", "b_n_log", "p_coldstart", "p_rookie",
    "p_mid_share", "p_rev_share", "p_mid_minus_rev", "p_ball_minus_strike",
    "p_form1", "p_midform1", "p_form3", "p_midform3", "p_form5", "p_midform5",
    "p_form_trend",
    "p_mix_entropy", "p_mix_max",
    "b_succ_minus_p",
    "era_abs",
    # (z_asof는 D-19로 제거 — 로컬 +2.5(“무해”)였으나 2025 LB 역산에서 **최소 −82, 현실 −180**.
    #  BLEND-1 483.97이 볼록성 하한 525를 깬 것이 증거. 리터럴은 자리채움, 생성 코드는 z_asof.py 보존.)
    # (N13 트랙맨 프로필 6피처는 D-17로 미채택·제거 — Tier1에서 refV24 +11.4 < 2SE 18.6,
    #  Tier1+2로 커버리지를 넓히면 −12.0으로 부호 반전. 복원은 git 이력 + trackman.py --profile.)
]

# 학습·추론이 공유하는 **유일한 컬럼 계약**.
FEATURE_ORDER = CAT_COLS + OFFICIAL_NUM + DERIVED

# 트리에 넣으면 위험한 컬럼(익명 ID는 시간순 배정이라 은닉 시간지표이고,
# 2025에는 train 최대값을 넘는 ID가 실재한다 → out-of-support 외삽).
ID_LIKE = ["pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"]
TIME_LIKE = ["season", "era_abs"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """공식 47컬럼 + 행 단위 파생 → 수치 행렬(float32), 컬럼 순서 = FEATURE_ORDER.

    규정 준수: 모든 항이 **그 행의 값만으로** 계산된다. 다른 행/미래 정보 참조 없음.
    결측(asof_* cold-start)은 채우지 않고 NaN으로 남긴다 —
    HistGradientBoosting은 NaN을 분기로 직접 처리하고, '이력 없음' 자체가 신호이기 때문.
    """
    X = pd.DataFrame(index=df.index)

    # 1) 범주형 — 고정 매핑(미지 카테고리 → -1)
    for c in CAT_COLS:
        X[c] = df[c].map(CAT_MAPS[c]).fillna(-1).astype("float32")

    # 2) 공식 수치 컬럼(고정 순서)
    for c in OFFICIAL_NUM:
        X[c] = df[c].astype("float32")

    b = df["balls_before"].astype("float32")
    s = df["strikes_before"].astype("float32")

    # 3) 카운트 구조 — 라벨이 위치 기준이라 카운트가 곧 '의도(조준점)' 변수다.
    #    3-0은 한복판을 강요하고(→ middle 실패), 0-2는 유인구를 부른다(→ 크게 벗어남 실패).
    X["count_bucket"] = b * 3 + s
    X["count_diff"] = b - s
    X["ahead"] = (s > b).astype("float32")
    X["behind"] = (b > s).astype("float32")
    X["two_strike"] = (s == 2).astype("float32")
    X["three_ball"] = (b == 3).astype("float32")
    X["full_count"] = ((b == 3) & (s == 2)).astype("float32")
    X["first_pitch"] = ((b == 0) & (s == 0)).astype("float32")

    # 4) 매치업·상황
    X["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype("float32")
    X["li_log"] = np.log1p(df["li"].astype("float32"))
    X["high_li"] = (df["li"] > 1.5).astype("float32")
    X["we_margin"] = (df["home_win_expectancy"].astype("float32") - 50.0).abs()
    X["score_diff_abs"] = df["score_diff_pitcher_team"].abs().astype("float32")
    X["close_game"] = (X["score_diff_abs"] <= 2).astype("float32")
    X["late_inning"] = (df["inning"] >= 7).astype("float32")
    X["risp"] = ((df["runner_on_2b"] == 1) | (df["runner_on_3b"] == 1)).astype("float32")

    # 5) 이력 피처의 행 단위 재조합 (asof_* 는 공식 입력)
    pn = df["asof_pitcher_n"].astype("float32")
    X["p_n_log"] = np.log1p(pn)
    X["b_n_log"] = np.log1p(df["asof_batter_n"].astype("float32"))
    X["p_coldstart"] = (pn < 200).astype("float32")
    X["p_rookie"] = (pn < 30).astype("float32")

    succ = df["asof_pitcher_success_rate"].astype("float32")
    mid = df["asof_pitcher_middle_rate"].astype("float32")
    rev = df["asof_pitcher_reverse_rate"].astype("float32")
    # 실패 3유형(한복판 / 크게 벗어남 / 역방향) 중 2유형이 공식 피처로 주어진다 →
    # 유형 구성비 = '어떤 방식으로 실패하는 투수인가'. 성공률과 직교하는 스타일 축.
    # 분모에 1e-3을 더해 succ→1일 때 inf가 나오지 않게 한다(sklearn은 NaN은 받아도 inf는 거부).
    fail = (1.0 - succ).clip(lower=0.0) + 1e-3
    X["p_mid_share"] = mid / fail
    X["p_rev_share"] = rev / fail
    X["p_mid_minus_rev"] = mid - rev
    X["p_ball_minus_strike"] = (df["asof_pitcher_ball_rate"].astype("float32")
                                - df["asof_pitcher_strike_rate"].astype("float32"))

    # 6) 최근 폼 델타 (직전 N경기 − 커리어) : 컨디션 변동. 두 값 모두 공식 asof.
    for k in (1, 3, 5):
        X[f"p_form{k}"] = df[f"asof_pitcher_prev{k}_game_success_rate"].astype("float32") - succ
        X[f"p_midform{k}"] = df[f"asof_pitcher_prev{k}_game_middle_rate"].astype("float32") - mid
    X["p_form_trend"] = X["p_form1"] - X["p_form5"]   # 단기 대 중기(핫핸드 방향)

    # 7) 구종 믹스 엔트로피 (레퍼토리 다양성). 이력 0인 신인은 NaN 유지.
    mix = df[["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
              "asof_pitcher_offspeed_rate"]].astype("float32").to_numpy()
    allnan = np.isnan(mix).all(axis=1)
    safe = np.where(np.isnan(mix), 0.0, mix)
    X["p_mix_entropy"] = np.where(
        allnan, np.nan, -np.sum(np.where(safe > 0, safe * np.log(safe + EPS), 0.0), axis=1))
    X["p_mix_max"] = np.where(allnan, np.nan, safe.max(axis=1))

    # 8) 상대 난이도
    X["b_succ_minus_p"] = df["asof_batter_success_rate"].astype("float32") - succ

    # 9) 체제 플래그 (2024~ ABS 1군 전면). test는 season이 2025 상수라 트리엔 out-of-support.
    X["era_abs"] = (df[SEASON] >= 2024).astype("float32")

    # (S1.5 cpk/spc 9피처는 D-15로 기각·제거 — probit/min 등 asof 율의 재조합은 refV24 −15로 유해.
    #  트리는 단조 변환에 불변이고 asof 율 공간은 포화 상태다. 복원은 git 이력 참조.)

    # (N15 p_signflip은 D-16으로 기각·제거 — 관측 현상(3볼 페널티~σ, z=3.48)은 실재하나
    #  트리가 원시 컬럼에서 이미 학습하는 상호작용이라 피처로는 −7.8 유해. 코드는 donut.py 보존.)

    # (N13 트랙맨 프로필 6피처는 D-17로 미채택·제거 — 매칭 자체는 검증 통과(문서 15)했으나
    #  Tier1 refV24 +11.4 < 2SE 18.6, Tier1+2 확대 시 −12.0 부호 반전. TM_PROFILE 리터럴을
    #  자리채움으로 되돌렸고 재생성은 `python sweep/trackman.py --profile`.)

    return X.reindex(columns=FEATURE_ORDER).astype("float32")
# ================================= <<<SERVE:END>>> =================================


# ---------------------------------------------------------------- 로딩
def load_train(path=None, downcast=True) -> pd.DataFrame:
    """train.csv 로드. downcast로 float64→float32 (1.47M행 메모리 절반)."""
    df = pd.read_csv(Path(path or DATA_DIR / "train.csv"), encoding="utf-8-sig")
    if downcast:
        for c in df.select_dtypes("float64").columns:
            df[c] = df[c].astype("float32")
        for c in df.select_dtypes("int64").columns:
            if c != TARGET:
                df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


def load_test(path=None) -> pd.DataFrame:
    return pd.read_csv(Path(path or DATA_DIR / "test.csv"), encoding="utf-8-sig")


def official_columns(df: pd.DataFrame) -> list:
    """주최 베이스라인이 쓰는 원본 47컬럼(= row_id·target 제외). 앵커 재현용."""
    return [c for c in df.columns if c not in (ID, TARGET)]


# ---------------------------------------------------------------- 목표 성공률 추정
def season_means(df: pd.DataFrame, seg: str | None = None) -> pd.Series:
    """시즌별 성공률. seg가 주어지면 그 세그먼트(game_type) 행만."""
    d = df if seg is None else df[df[SEGMENT] == seg]
    return d.groupby(SEASON)[TARGET].mean()


def _extrapolate(means: pd.Series, target_season: int, k: int = 3,
                 clip=(0.30, 0.70)) -> float:
    """직전 k시즌 성공률의 선형 추세를 target_season으로 외삽."""
    idx = [target_season - i for i in range(k, 0, -1)]
    hist = means.reindex(idx).to_numpy(dtype=float)
    if np.isnan(hist).any():
        return float(np.clip(means.dropna().iloc[-1], *clip))
    slope, intercept = np.polyfit(np.arange(k), hist, 1)
    return float(np.clip(slope * k + intercept, *clip))


def extrapolate_rate(means: pd.Series, target_season: int, k: int = 3,
                     clip=(0.30, 0.70)) -> float:
    """전역(세그먼트 무시) 외삽. 하위호환용 — 세그먼트 혼합비에 오염될 수 있다.

    ⚡ 이 값이 대회 점수를 지배한다: 지표가 Brier라 예측 평균이 실제 성공률에서
    d만큼 어긋나면 손실 ≈ d²/(r(1−r)) — d=0.03이면 약 −360점(베이스라인 총점 549).
    """
    return _extrapolate(means, target_season, k, clip)


def segment_rates(df: pd.DataFrame, target_season: int, k: int = 3,
                  shrink_w: float = 0.5) -> dict:
    """`game_type` 세그먼트별 목표 성공률 r̂.

    전역 앵커는 test의 세그먼트 구성비에 **암묵적으로 베팅**한다(2025의 R:F 비율은 알 수 없고,
    로컬 test 5행으로는 추정 불가). 세그먼트별 앵커는 그 구성비에 **불변**이다 — 이것이
    K4를 쓰는 진짜 이유다.

    R(주 세그먼트): 직전 k시즌 선형 외삽.
    F(소수 세그먼트): 표본이 적고 체제 파단 이력이 있어 선형 외삽이 위험 →
        `r̂_F = r̂_R + w·(F_last − R_last)` 로 R 방향으로 수축(w=0이면 완전히 R 앵커, 1이면 F 최근값).
    반환: {"R": float, "F": float, "_global": float, "_w": w}
    """
    out = {}
    r_hat_by_seg = {}
    for seg in sorted(df[SEGMENT].dropna().unique()):
        r_hat_by_seg[seg] = season_means(df, seg)
    base_seg = "R" if "R" in r_hat_by_seg else sorted(r_hat_by_seg)[0]
    r_base = _extrapolate(r_hat_by_seg[base_seg], target_season, k)
    out[base_seg] = r_base
    for seg, means in r_hat_by_seg.items():
        if seg == base_seg:
            continue
        last = float(means.reindex([target_season - 1]).iloc[0])
        base_last = float(r_hat_by_seg[base_seg].reindex([target_season - 1]).iloc[0])
        if np.isnan(last) or np.isnan(base_last):
            out[seg] = r_base
        else:
            out[seg] = float(np.clip(r_base + shrink_w * (last - base_last), 0.30, 0.70))
    out["_global"] = _extrapolate(season_means(df), target_season, k)
    out["_w"] = shrink_w
    return out


# ---------------------------------------------------------------- 시간분할
def make_folds(df: pd.DataFrame, val_seasons=(2022, 2023, 2024), mode: str = "refit"):
    """전진 검증 폴드. 각 폴드 = dict(fit1 / cal / fit2 / val 인덱스 + 메타).

    mode="refit" (기본, **제출과 동형**):
        fit1 = ≤(val−2)  캘리브레이션 상수를 산출할 모델
        cal  = (val−1)   캘리 홀드아웃
        fit2 = ≤(val−1)  실제 예측에 쓸 재적합 모델 (100% refit)
        val  = val
      실제 제출 대응: fit1 ≤2023 · cal 2024 · fit2 ≤2024 · predict 2025.
      ⚠ 상수는 fit1 모델의 확률 분포에 맞춰 적합되고 fit2 모델에 적용된다 →
        **전이 손실**이 존재한다. run_real.py가 이를 측정해 리포트한다.

    mode="gap": fit1=fit2=≤(val−2). 2시즌 갭이라 제출과 비동형(진단용으로만).
    """
    assert df.index.equals(pd.RangeIndex(len(df))), "make_folds는 RangeIndex를 가정한다"
    seasons = np.sort(df[SEASON].unique())
    season_arr = df[SEASON].to_numpy()
    folds = []
    for vs in val_seasons:
        if vs not in seasons or vs - 2 < seasons.min():
            continue
        fit1 = np.where(season_arr < vs - 1)[0]
        cal = np.where(season_arr == vs - 1)[0]
        fit2 = np.where(season_arr < vs)[0] if mode == "refit" else fit1
        val = np.where(season_arr == vs)[0]
        folds.append({
            "val_season": int(vs), "mode": mode,
            "fit1": fit1, "cal": cal, "fit2": fit2, "val": val,
            "fit1_seasons": [int(x) for x in seasons if x < vs - 1],
            "fit2_seasons": [int(x) for x in seasons if x < vs] if mode == "refit"
                            else [int(x) for x in seasons if x < vs - 1],
        })
    return folds


def make_reverse_fold(df: pd.DataFrame, val_season: int = 2019):
    """역방향 폴드 — **상승 레짐 리허설**.

    2025는 base rate가 하락 추세를 깨고 반등했다(LB 역산 r≈0.513). 전진 폴드는 전부 하락
    구간이라 이 상황을 리허설할 수 없다. 유일한 로컬 대안: 2020~2024로 학습하고 **2019**
    (r=.5647, 학습기간 최고치보다 위)를 검증한다.
    구조는 전진 폴드의 거울: fit1 = 2021~2024, cal = 2020(경계 시즌), fit2 = 2020~2024, val = 2019.
    주의: 2019는 데이터 첫해라 asof_*가 초기 커리어(콜드스타트) → 적응 채널이 과소평가된다(보수적).
    """
    assert df.index.equals(pd.RangeIndex(len(df)))
    season_arr = df[SEASON].to_numpy()
    seasons = np.sort(df[SEASON].unique())
    vs = int(val_season)
    return {
        "val_season": vs, "mode": "reverse",
        "fit1": np.where(season_arr > vs + 1)[0],
        "cal": np.where(season_arr == vs + 1)[0],
        "fit2": np.where(season_arr > vs)[0],
        "val": np.where(season_arr == vs)[0],
        "fit1_seasons": [int(x) for x in seasons if x > vs + 1],
        "fit2_seasons": [int(x) for x in seasons if x > vs],
    }


# ---------------------------------------------------------------- 투수 신뢰도 수축 (B1)
def build_credibility(df_fit: pd.DataFrame, entity: str = "pitcher_id"):
    """Bühlmann–Straub 신뢰도 수축 투수 성공률 → 로짓 룩업.

    진단(측정): 투수 이력 십분위별 실제 스프레드 0.1095 vs 모델 예측 0.0881 — 트리가 투수
    효과를 ~20% 과수축한다. 해법은 보험 credibility 이론의 표준형:
        Z_i = n_i / (n_i + K),  K = EPV / VHM  (모멘트법 추정)
        shrunk_i = m + Z_i (p_i − m)
    K가 실제 투수 간 분산(VHM)에 맞춰 추정되므로, 손튜닝 상수와 달리 수축 강도가
    **구성상 데이터의 분산과 일치**한다 (= 야구 Marcel/안정화 상수와 동일한 수학).

    train 기간 데이터로만 계산 → zip에 룩업으로 동봉(규정 §5 허용: 고정 룩업). 반환:
    (table: dict[int, float] 투수→shrunk 로짓, default: float 전역 로짓, K: float, m: float)
    """
    g = df_fit.groupby(entity)[TARGET]
    n = g.size().astype(float)
    k = g.sum().astype(float)
    p = k / n
    N = float(n.sum())
    m = float(k.sum() / N)
    I = len(n)
    # 베르누이: 투수 내 제곱합 = n_i·p_i(1−p_i) → EPV = Σ n_i p_i(1−p_i) / Σ(n_i − 1)
    epv = float((n * p * (1 - p)).sum() / max((n - 1).sum(), 1.0))
    # VHM 모멘트 추정 (Bühlmann–Straub): [Σ n_i(p_i−m)² − (I−1)·EPV] / (N − Σn_i²/N)
    c_star = N - float((n ** 2).sum()) / N
    vhm = (float((n * (p - m) ** 2).sum()) - (I - 1) * epv) / max(c_star, 1.0)
    K = epv / vhm if vhm > 1e-10 else 1e9        # 분산 0이면 완전 수축
    z = n / (n + K)
    shrunk = np.clip(m + z * (p - m), 1e-3, 1 - 1e-3)
    logits = np.log(shrunk / (1 - shrunk))
    default = float(np.log(m / (1 - m)))
    return dict(zip(g.size().index.astype(int), logits.astype(float))), default, float(K), m
