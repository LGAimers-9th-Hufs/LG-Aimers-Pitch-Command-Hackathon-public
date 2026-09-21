# sweep/ — 실험 하네스 (v2)

실데이터(대회 배포본) 전용 경로. 합성 데이터 시절 모듈은 `_retired/`에 있고 **실행 경로에서 쓰지 않는다**
(사유: `_retired/README.md`, 등록부: `../docs/00_KILL_LIST.md`).

> **v2 (2026-08-05, 첫 제출 269점 부검 후)**: 게이트가 score → **refinement**로 바뀌었다.
> refinement = score + 1e5·d²/(r(1−r)) — 앵커(평균 오차) 운을 제거한 순수 조건부 성능.
> 레벨(절편)은 β 스칼라 하나로 분리해 LB로 결정한다. isotonic은 제출 스택에서 폐기(비교 arm만).
> 배경: `../docs/log/13_autopsy_and_v2.md`.

```
real_data.py         로더 + 규정 안전 피처 82개 + 폴드(전진/역방향) + build_credibility(투수 수축 룩업)
recal.py             재캘리 레이어 (a,b,c,d) — Brier 직접 최소화 · 풀링 적합 · Brier 분해 · 분산 진단
run_real.py          시뮬레이션 러너 (refinement 게이트, RC/RC2 arm, run.json 기록)
test_regulation.py   §5 준수·파리티 검증 8종
validation.py        competition_score + reliability bins
reporting.py         run.json / leaderboard.csv / runs.jsonl / leaderboard_history.csv
figures.py           논문 figure (⚠ 아직 LogLoss 키 기준 — 실데이터 run에 맞춰 재작성 필요)
```

## 핵심 계약 3가지

**1. 피처는 전부 행 단위다.** `real_data.py`의 `<<<SERVE:BEGIN>>>`~`<<<SERVE:END>>>` 블록이
피처 코드 전부이며, 이 블록은 제출 zip에 **바이트 그대로** 복사된다. 그래서 학습과 추론이
문자 그대로 같은 코드를 쓴다(train/serve skew 0). 블록은 **자기완결**이어야 한다 —
블록 밖 상수를 참조하면 동봉본이 `NameError`로 죽는다(실제로 한 번 겪었다).

**2. 컬럼 순서는 `FEATURE_ORDER` 하나로 고정된다.** `df.columns` 순회 금지.

**3. 보정 상수는 참조 시즌에서 1회 산출해 얼린다.** 예측 배치의 평균을 보고 확률을 이동시키면
§5("평가 데이터 전체를 보고 만든 사후 보정값") 위반이다. 라벨을 쓰지 않아도 위반이다.

## 폴드 (제출과 동형)

| | fit1 | cal | fit2 | val |
|---|---|---|---|---|
| V24 (주 게이트) | ≤2022 | 2023 | ≤2023 | 2024 |
| V23 (스트레스 — F 세그먼트 파단 연도) | ≤2021 | 2022 | ≤2022 | 2023 |
| V22 | ≤2020 | 2021 | ≤2021 | 2022 |
| **제출** | ≤2023 | 2024 | ≤2024 | 2025 |

`fit1`에서 캘리 상수를 만들어 `fit2`로 재적합한 모델에 적용한다 → **전이 손실**이 존재하며
리더보드의 `fit1적용` 열이 그것을 진단한다.

## 캘리 구조 arm

| K | 내용 |
|---|---|
| K0 | 무보정 |
| K1 | isotonic (cal 시즌 적합 → 절점 export, 추론은 `np.interp`로 버전 의존성 제거) |
| K2 | 드리프트 보정 앵커: 목표를 `r̂ − Δ̂`로. Δ̂ = **이전 폴드들에서** 측정한 예측평균 이동 |
| K3 | 전역 앵커: 로짓 오프셋으로 목표 성공률에 맞춤 |
| **K4** | **`game_type` 세그먼트 앵커** — 전역 앵커는 test의 세그먼트 구성비에 암묵적으로 베팅하지만 이건 그 미지의 비율에 불변 |

`role`: candidate(채택 가능) / reference(상수 arm) / oracle(val 라벨 사용, 상한 표시).
정렬·채택·`best` 기록에서 candidate 외에는 제외된다.

## 게이트

**주 게이트 = V24**(제출과 동형). 폴드 평균은 V22(+2077)가 지배하므로 순위 근거로 쓰지 않는다.
채택은 `V24 ≥ 현재 최고 + 2×paired SE` **그리고** 최악 폴드가 무너지지 않을 것.
고정 임계(±15)는 폐기했다 — 같은 모델의 상수 비교는 SE≈8점이지만 다른 base model 비교는 30~50점이다.
리더보드의 `±2SE`·`유의` 열이 비교마다 직접 계산된 값이다.

## 실행

```bash
python sweep/test_regulation.py                         # 먼저 이것부터 (~1분)
python sweep/run_real.py --folds 2022,2023,2024         # 전량 3폴드 (~13분, 캐시 후 수십 초)
python sweep/run_real.py --sample 300000 --folds 2024   # 리허설(의사결정 금지 — base rate 왜곡)
python sweep/run_real.py --models rf_official,hgb82,hgb82_noid
```

예측은 `results/cache/`에 캐시된다(모델·폴드·피처해시 키). 캘리 그리드만 바꿔 재실행하면
학습을 건너뛴다. 피처를 바꾸면 해시가 달라져 자동으로 재학습한다.

## 새 후보 추가하는 법

- **base model**: `run_real.py`의 `BASE_MODELS`에 `(tier, factory)` 추가.
  factory는 `(rep, estimator)`를 반환하고 rep은 `"raw47"`(주최 47컬럼) 또는 `"feat82"`/`"feat82_noid"`.
- **피처**: `real_data.py`의 SERVE 블록 안에서 만들고 `DERIVED` 리스트에 **같은 순서로** 이름을 추가.
  그다음 `python sweep/test_regulation.py`로 행 독립성이 유지되는지 반드시 확인한다.
- **캘리 arm**: `run_fold`의 `variants`에 추가. 모든 보정은 상수 → 행 단위여야 한다.
