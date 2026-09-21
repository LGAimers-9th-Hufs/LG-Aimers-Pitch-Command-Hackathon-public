# 서빙 계약 — 파트너 모델을 실제 제출 zip에 싣기 위한 인터페이스

> 지금 산출물(검증 CSV)로는 **측정만 되고 제출이 안 된다.** 2025 평가셋 예측을 낼 경로가 필요하다.
> 아래 계약을 지키면 우리 빌더(`--partner`)가 그대로 집어넣는다.

## 1. 디렉터리

```
results/codex_partner/serving/
  predict.py          # 유일한 진입점
  <artifacts>         # LightGBM booster .txt · 룩업 .json/.npz 등 자유
  MANIFEST.json       # {"files": [...], "sha256": {...}, "trained_through_season": 2024}
```

## 2. `predict.py` 계약

```python
def predict(test_df, model_dir):
    """test_df = 대회 test.csv를 그대로 읽은 DataFrame (49컬럼 중 control_success 없음, 245,789행)
       model_dir = 이 파일이 들어 있는 폴더의 절대경로 (str)
       반환 = np.ndarray, dtype float64, shape (len(test_df),), 값 ∈ [0,1], NaN 없음"""
```

**반드시 지킬 것**
- **§5 행 독립**: 각 행은 자기 컬럼 값 + `model_dir` 안의 고정 상수만 사용.
  `.mean()` `.groupby(` `.value_counts(` `.transform(` `.rolling(` `.expanding(` `.cumsum(` `.rank(`
  **호출 금지** — 우리 검증기가 AST로 기계 검사하고, 걸리면 제출 자체를 못 한다.
  (전처리에서 배치 통계가 필요하면 **학습 때 계산해 상수로 저장**하고 서빙에서는 조회만 해라.)
- **자기완결**: import는 `numpy` / `pandas` / `lightgbm` / 표준 라이브러리만. 인터넷 차단 환경이다.
- **커스텀 클래스를 pickle에 넣지 마라** — 서버에 그 모듈이 없어 언피클이 실패한다.
  LightGBM은 `booster.save_model()` / `lgb.Booster(model_file=...)`, 나머지는 JSON/NPZ.
- **미등장 값 안전**: 2025에만 있는 ID·범주는 룩업에 없다. `.get(key, default)` 형태로 폴백해라.
  예외를 던지면 그 제출은 실행 오류 = **슬롯 차감**이다.
- **속도**: 245,789행 전체가 **60초 이내**(전체 zip 예산 600초, 우리 스택이 이미 10초를 쓴다).
- 학습 데이터: **`season ≤ 2024` 전량 사용 가능**(검증 CSV와 달리 컷오프 없음 — 2025를 예측하니까).
  단 `train_partner.py`와 같은 레시피여야 한다. 시즌 가중 0.25/0.5/1.0이면 2022/2023/2024가 된다.

## 3. 자기 검증 (제출해라, 이걸 통과 못 하면 우리가 못 싣는다)

```python
# results/codex_partner/serving/selftest.py 로 만들고 결과를 SELFTEST.md에 적어라
import pandas as pd, numpy as np, time, sys
sys.path.insert(0, "results/codex_partner/serving")
from predict import predict
# 프록시: train 2024에서 245,789행을 뽑아 control_success를 떼고 test 스키마로
```
확인 항목: 행수 일치 · NaN 0 · 범위 [0,1] · 소요시간 · **예측 평균과 SD** ·
`train_partner.py`가 만든 2024 검증 예측과의 상관(같은 레시피인지 확인).

## 4. 우리가 어떻게 쓰는가 (네가 알아야 맞출 수 있다)

우리 서빙은 `raw_A`(우리 블렌드) → 캘리 → 세그먼트 → 형상 순이다. 파트너는 **캘리 이전**에 섞는다:

```
raw_B' = raw̄_A + (σ_A/σ_B)·(raw_B − raw̄_B)        ← 평균·분산 보존 정규화 (동봉 상수)
raw    = (1−w)·raw_A + w·raw_B'
```

**왜 정규화하나**: 우리 캘리 상수(slope 1.14202 · shift −0.00264162)는 리더보드 프로브로 얻은 값이라
`raw_A`의 레벨·스케일에 묶여 있다. 파트너를 날것으로 섞으면 레벨이 흔들려 **블렌드 축과 캘리 축이
뒤섞이고 프로브 역산이 무의미**해진다. 정규화하면 `w`만 순수하게 프로빙할 수 있다.

⇒ **네가 할 일**: `MANIFEST.json`에 네 서빙 예측의 **train 2024 기준 평균·표준편차**를 적어라
(`raw_mean_2024`, `raw_std_2024`). 우리가 σ_B/raw̄_B 상수로 동봉한다.

## 5. 산출물

- `serving/predict.py` · 아티팩트 · `MANIFEST.json` · `selftest.py` · `SELFTEST.md`
- **라운드 2의 모델 개선이 끝난 뒤**에 만들어라(개선본 기준이어야 한다). 시간이 부족하면
  **라운드 2 개선을 줄이고 이 서빙 패키지를 먼저 확보해라** — 제출 못 하면 개선은 0점이다.
