# 제출물 규정 준수 근거

> Private 리더보드 상위 ~100명은 **코드 검증**을 받는다. 제출 zip에 들어가는 아티팩트마다
> `data_description.md` §5(행 독립성)·§6(사용 금지 정보)의 어느 조항으로 허용되는지 한 줄씩 적어둔다.
> 새 아티팩트를 동봉할 때는 반드시 이 표에 행을 추가한다.

## 규정 원문 요지

- **§5**: 평가 데이터의 각 행은 **독립적으로** 예측해야 한다. 금지 — test 내부 행을 이용한 누적·
  빈도·분포·rolling/expanding·target encoding, **평가 데이터 전체를 보고 만든 사후 보정값**.
  허용 — 주최가 제공한 `asof_*`(공식 입력).
- **§6**: 현재 투구 이후 확정되는 정보, 현재 투구의 위치·판정·결과·구종·트랙맨 측정값,
  2025 트랙맨 사용 금지. 사용 가능 = train.csv · 평가환경 test.csv · 2019~24 trackman_history.csv ·
  규칙상 허용되는 외부 데이터.

## 동봉 아티팩트별 근거

| 아티팩트 | 무엇 | 어떻게 만들어졌나 | 허용 근거 |
|---|---|---|---|
| `model/base.joblib` | 학습된 분류기 | `train.csv` 2019~2024 전량으로 적합 | §6 "사용 가능 데이터 = train.csv". test 미참조 |
| `model/serve_features.py` | 피처 생성 코드 | `sweep/real_data.py`의 SERVE 블록을 **바이트 그대로** 복사 | 모든 항이 **그 행의 값만** 사용. `test_regulation.py::test_row_independence`가 단일 행 결과 == 전체 프레임 결과임을 200행 표본 + 10% 부분집합으로 증명 |
| `calib.json: iso` | isotonic 절점 (x, y) | fit ≤2023 모델의 **2024 시즌** 예측과 그 시즌 라벨로 적합 | 전부 train 구간. 추론은 `np.interp` — 행 단위 사상 |
| `calib.json: beta_by_segment` | `game_type`별 로짓 오프셋 | 2024 예측에서 **1회** 산출해 상수로 고정 | 각 행은 **자기 행의 `game_type`** 하나만 참조. test 배치 통계 미사용 |
| `calib.json: r_hat` | 세그먼트별 2025 목표 성공률 | train 시즌 성공률의 선형 외삽(공식은 `provenance`에 문자열로 기록) | train 유래 상수 |
| `calib.json: drift_hat` | cal→예측시즌 예측평균 이동 | **과거 폴드**(2022→2023, 2023→2024)에서 측정한 평균 | train 유래 상수, 미래 정보 없음 |
| `calib.json: fallback` | 순수 numpy 로지스틱 계수 | train 표본으로 적합 | §6 허용 데이터 |
| `calib.json: z_table` (v2) | 투수별 신뢰도 수축 성공률 로짓 (Bühlmann–Straub, 792명) | **train 전체**에서 Z=n/(n+K)로 계산, K는 모멘트법 | §5가 명시 허용하는 "train 유래 고정 룩업". 각 행은 자기 행의 `pitcher_id` 하나로만 조회. 미등장 ID → `z_default` |
| `calib.json: recal` (v2) | 재캘리 상수 (a,b,c,d) + 센터 2개 | cal 시즌들의 **OOF 예측**(각 시즌을 그 이전 데이터로만 학습한 모델이 예측)에서 Brier 직접 최소화로 적합, 계수 경계 b∈[0.2,3] | 전부 train 유래 상수, 행 단위 적용. `d·(prev5−t̄)` 항은 주최 공식 입력을 제 기울기로 소비하는 것 — test 통계 미사용 |
| `calib.json: beta` (v2) | 레벨 절편 스칼라 | **사전 등록 격자**(D-11: +0.04/+0.12)에서 선택. train 유도식 provenance 기록 | 공개 점수로 자기 후보 중 선택 = 리더보드 형식의 정상 관행. 점수 역산값을 직접 박지 않음 |
| `script.py` | 추론 | 배치 통계 연산 없음 | AST 검사로 `mean/groupby/value_counts/transform/rolling/expanding/cumsum/rank/shift` 호출 부재를 강제 |

## 기계적 보증

`python sweep/test_regulation.py`

| 테스트 | 무엇을 증명하나 |
|---|---|
| `test_row_independence` | 한 행만 넣어 만든 피처 == 전체 프레임에서 만든 그 행 (비트 동일). 10% 부분집합도 동일 → **배치 크기 무관** = §5 준수의 기계적 증명 |
| `test_target_permutation` | 라벨을 섞어도 피처 불변 → 피처가 target을 보지 않음 |
| `test_feature_order` | train·test·`FEATURE_ORDER` 컬럼 순서 동일 → train/serve skew 0 |
| `test_no_batch_statistics` | 서빙 경로에 배치 전체를 보는 호출 부재 (AST) |
| `test_serve_block_standalone` | SERVE 블록이 numpy/pandas만으로 단독 실행 |
| `test_asof_sanity` | 주최 asof 3유형이 사실상 분할임을 확인 |
| `test_fold_hygiene` | fit/cal/val 교집합 ∅, 시즌 순서 단조, fit2 = val 이전 전체 |

`python submission/verify_submission.py <zip>`

- zip 최상위 구조 · serve 코드 **바이트 동일성** · §5 배치 통계 부재
- 서버와 같은 방식으로 `script.py` 실행 → 행 수·`row_id` 순서·확률 범위·실행 시간
- **모델 파일을 손상시킨 폴백 경로**까지 실행 (실행 오류는 제출 횟수를 차감하므로)

## 명시적으로 하지 않은 것

- test 예측의 평균을 계산해 확률을 이동시키지 않는다(구 `match_mean` — §5 위반이라 제거).
- test 행 간 통계·pseudo-label·target encoding을 만들지 않는다.
- 트랙맨을 현재 투구에 조인하지 않는다(애초에 ID 체계가 분리돼 있고 2025 데이터도 없다).
- 리더보드 점수를 역산해 얻은 값을 **상수로 박아 넣지 않는다.** 앵커 후보는 전부 train에서
  유도되며 유도식이 `calib.json.provenance`에 남는다. 공개 점수는 **사전 등록된 후보 중 선택**에만 쓴다.
