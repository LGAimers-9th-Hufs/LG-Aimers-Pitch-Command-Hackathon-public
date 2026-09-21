# JTT-1100 모델 설계안

> 구현 결과(2026-08-13): dynamic-state와 soft-TrackMan residual은 시간순 gate에서 기각됐다.
> 대신 시즌 간 계층 prior를 둔 선수×상황 residual lookup이 7/7 forward split 양수,
> 공격 설정에서 평균 +61.65, 최악 +17.88, V2024 forward 평균 +27.74를 기록해 별도 후보로
> 패키징됐다. 1100 승격 기준(+90)은 통과하지 못했다. 상세: `docs/log/34_dsf_implementation.md`.

## 1. 목표와 냉정한 전제

- 현재 안전한 기준 모델: `FINAL_submit_blend = 1032.6717450165`
- 목표 점수: `1100`
- 필요한 추가 점수: `+67.3282549835`
- 현재 평가식에서 필요한 평균 Brier loss 감소량: 약 `0.00016729`
- 최적 스케일의 잔차 방향으로 환산하면, 기존 모델의 잔차와 약 `2.6%` 이상의 상관을 갖는 새로운 정보가 필요하다.

이 차이는 단순 하이퍼파라미터 튜닝으로 얻기 어렵지만, 완전히 비현실적인 크기는 아니다. 다만 아래의 시간 순서 검증 기준을 통과하기 전에는 1100점을 보장하거나 예상 점수라고 표현하지 않는다.

## 2. 현재 모델에서 유지할 것과 버릴 것

### 유지

- JTT의 exact1과 regime 예측
- 현재 시즌 투수/타자 누적 상태, EB shrinkage, count/context interaction
- 이미 확인된 `K=100` current-season shrinkage
- 독립 행 추론과 frozen lookup 원칙

### 주력 방향에서 제외

- affine/curvature 등 전역 calibration 재탐색
- 기존 모델군의 단순 재튜닝 또는 같은 feature를 넣은 XGBoost/CatBoost 교체
- 고정 가중치 2-way/3-way blend
- 단순 failure-state feature 추가
- test batch 통계, test row 간 rolling, pseudo-label, leaderboard 역추론

위 방향들은 이미 닫혔거나 기대 이득이 목표 폭보다 매우 작다.

## 3. 제안 모델: JTT-DSF

`DSF = Dynamic State + Soft Fingerprint + row-wise Fusion`

최종 예측은 다음과 같다.

```text
p0 = 기존 JTT 예측
d1 = 동적 선수 상태 전문가의 보정값
d2 = TrackMan 소프트 프로필 전문가의 보정값
d3 = 투수×타자×상황 저차원 상호작용 전문가의 보정값
g  = 현재 행의 표본 수와 불확실성으로 결정되는 행별 gate

p = clip(p0 + 0.025 * tanh(g1*d1 + g2*d2 + g3*d3), 0.01, 0.99)
```

보정 폭을 제한해 큰 분포 이동에서 JTT가 안전장치가 되도록 한다.

### 3.1 동적 선수 상태 전문가

선수의 실력을 고정 ID 효과가 아니라 시즌 사이에 움직이는 잠재 상태로 모델링한다.

```text
z(player, season) = A*z(player, season-1)
                  + B*TrackMan_change
                  + C*age/role/usage_change
                  + process_noise
```

각 평가 행에서는 그 행에 이미 제공된 현재 시즌 누적치만 이용해 prior를 posterior로 갱신한다.

- 투수와 타자 각각의 posterior mean, variance, effective sample size
- 성공뿐 아니라 복원 가능한 reverse/middle/ball/strike 상태를 보조 목표로 학습
- 직전 시즌, 최근 3시즌, 커리어 기준으로부터의 변화량과 변화 불확실성
- 신규 선수는 유사 프로필 집단 prior로 cold start
- Futures/정규경기, 초반/후반 시즌에 서로 다른 transition variance 사용

핵심은 현재 누적 비율 자체를 다시 넣는 것이 아니라, **그 비율이 장기 prior에서 얼마나 믿을 만하게 이동했는지**를 확률 상태로 표현하는 것이다.

### 3.2 TrackMan 소프트 지문 전문가

기존 hard entity match는 고신뢰 매칭 선수만 활용하고 나머지는 global fallback으로 손실된다. 이를 후보 분포로 바꾼다.

1. 투수마다 가능한 TrackMan ID 후보를 만든다.
2. 활동 시기, 요일/월, 이닝·카운트 분포, 투타 손, 팀/역할, 구종 구성 지문의 거리를 계산한다.
3. 후보별 posterior probability를 만든다.
4. 물리/구종 feature를 한 명의 값이 아니라 posterior 평균과 분산으로 전달한다.

```text
TM_mean = Σ P(candidate | fingerprint) * profile(candidate)
TM_var  = Σ P(candidate | fingerprint) * (profile - TM_mean)^2
```

모델에는 평균뿐 아니라 entropy와 variance도 넣는다. 매칭이 불확실할수록 gate가 자동으로 TrackMan 전문가의 비중을 줄인다. 후보 생성, 거리 가중치, temperature는 모두 해당 검증 시즌 이전 데이터에서만 정한다.

### 3.3 투수×타자×상황 저차원 전문가

고차원 raw ID 조합을 직접 외우지 않고 다음과 같이 분해한다.

```text
interaction = <u_pitcher, v_batter>
            + <u_pitcher, c_count/inning/runners>
            + <v_batter, c_count/inning/runners>
```

- 8~16차원의 작은 embedding
- 선수 embedding은 이전 시즌까지만 학습한 prior와 현재 행 posterior를 결합
- auxiliary event target을 함께 학습해 outcome 한 개에 대한 과적합을 완화
- 신인/희소 선수는 hierarchical shrinkage로 팀·역할·손 방향 집단에 수축

### 3.4 행별 gate

모든 행에 같은 blend weight를 쓰지 않는다. gate 입력은 아래의 낮은 차원 변수만 사용한다.

- 현재 시즌 관측 수와 effective sample size
- 동적 상태 posterior variance
- TrackMan match entropy/variance
- exact1과 regime, 세 신규 전문가 사이의 disagreement
- game type, month, 투수/타자의 cold-start 여부

gate는 작은 ridge logistic 또는 깊이 2 이하의 constrained tree로 제한한다. 출력은 simplex를 통과시켜 음수 가중치와 과도한 extrapolation을 막는다. 기존 exact1/regime 사이의 행별 gate는 예비 실험에서 약 `+4~+27`의 신호만 보여, 단독 해법이 아니라 안전한 보조 축으로 사용한다.

## 4. 학습 목표

기존 JTT가 이미 설명한 부분을 다시 학습하지 않고 fold별 JTT의 잔차를 직접 학습한다.

```text
L = Brier(y, p)
  + lambda_aux * auxiliary_event_loss
  + lambda_state * transition_stability
  + lambda_gate * distance_from_safe_JTT
  + lambda_orth * orthogonality_penalty
```

orthogonality penalty는 신규 보정이 이미 소진된 전역 상수, slope, month, game-type-F 축을 재사용하지 못하게 한다.

모델 후보는 역할별로 제한한다.

- 동적 상태: hierarchical GLM/GAM + 작은 LightGBM residual
- TrackMan: posterior-profile LightGBM
- 상호작용: 작은 factorization machine 또는 2-layer MLP
- 최종 gate: ridge logistic/simplex stacking

모델 종류를 늘리는 것이 아니라 서로 다른 정보원을 가진 보정 방향을 만드는 것이 목적이다.

## 5. 평가 데이터 독립성 보장

최종 추론 함수는 행 하나에 대해 완결되어야 한다.

```text
predict(row):
    prior = frozen_player_state[row.player_id]
    tm    = frozen_soft_trackman_profile[row.player_id]
    state = update(prior, row.current_season_asof_fields)
    return experts_and_gate(row, state, tm)
```

허용:

- train/TrackMan으로 미리 만든 frozen lookup
- 현재 평가 행 자체의 공식 누적/as-of 변수
- 행 단위 lookup과 element-wise 계산

금지:

- 평가 행끼리 집계, 정렬, rolling 또는 이전 평가 행 결과 사용
- 평가 batch의 평균/분산/빈도 사용
- 평가 target 추정치로 재학습 또는 pseudo-label
- leaderboard 결과를 이용한 상수·가중치 역산

배치 추론은 속도를 위한 vectorization만 허용하며, 한 행의 결과는 다른 평가 행의 존재·순서·분할 방식에 따라 달라지면 안 된다. 이를 자동 테스트로 검증한다.

## 6. 올바른 검증 프로토콜

기존 2025용 artifact를 과거 시즌에 그대로 적용한 점수는 모델 선택 근거로 사용하지 않는다. 각 fold의 모든 lookup과 baseline을 cutoff 이전 데이터로 다시 만든다.

| Fold | 학습/lookup cutoff | 검증 |
|---|---:|---:|
| V2022 | 2021 시즌까지 | 2022 |
| V2023 | 2022 시즌까지 | 2023 |
| V2024 | 2023 시즌까지 | 2024 |

각 fold에서 exact1, regime, JTT 자체도 동일 cutoff로 재학습한다. feature, 매칭 후보, 선수 prior, calibration 선택까지 모두 fold 내부에서 끝낸다.

### 1100 도전 통과 기준

최종 2025 모델을 패키징하는 조건은 다음과 같다.

1. 세 fold 모두 기존 fold-JTT보다 점수가 높다.
2. 가중 평균 개선폭 `0.2*V2022 + 0.3*V2023 + 0.5*V2024 >= +90`.
3. 최악 fold 개선폭이 `+50` 이상이고 V2024가 `+80` 이상이다.
4. player/row bootstrap으로 계산한 가중 개선폭의 보수적 하한이 `+67.3`을 넘는다.
5. 행 순서 변경, batch 분할, 단일 행 추론의 예측이 허용 오차 내에서 완전히 같다.

`+90`을 요구하는 이유는 2025 전이 과정에서 약 25%의 성능 감소가 생겨도 `+67.5`가 남도록 하기 위해서다. 이 기준을 못 넘으면 1100점 가능성이 검증되지 않은 것이므로 현재 JTT를 유지한다.

## 7. 실험 순서와 중단 조건

### Stage A — 정직한 baseline 재구축

- V2022~V2024 fold별 exact1/regime/JTT prediction cache 생성
- 현재 1032 모델과 feature parity 검증
- 약한 과거 적용 proxy를 모델 선택에서 제외

중단 조건: fold baseline이 현재 문서의 재현 범위를 벗어나면 신규 모델보다 baseline 문제를 먼저 해결한다.

### Stage B — 정보원 단독 검증

- B1: dynamic state only
- B2: soft TrackMan only
- B3: low-rank interaction only
- B4: row-wise gate only

각 정보원은 실제 모델 점수 외에 residual oracle gain과 residual correlation을 기록한다. 세 fold 중 두 곳 이상에서 residual correlation이 `1%` 미만이면 해당 정보원을 폐기한다.

### Stage C — 결합과 절제

- B1~B3의 correction 간 correlation matrix 확인
- 상관이 높은 두 correction은 하나만 유지
- fold 내부 constrained stacking
- correction cap `0.015/0.020/0.025/0.030`만 사전 정의 grid로 비교

### Stage D — 최종 봉인

- 통과 기준을 만족한 단일 설정만 2019~2024 전체로 재학습
- 2025 lookup/TrackMan profile 생성
- batch-independence 테스트, 시간/메모리 테스트, ZIP 재현성 검증
- 예상 runtime은 제한의 80% 이하, 메모리는 12GB 이하를 목표로 한다.

## 8. 기대 이득의 해석

현재 증거로 확실하게 말할 수 있는 것은 다음이다.

- 행별 exact1/regime gate만으로는 목표에 부족하다.
- 단순 current-season failure-state 추가도 주력 해법이 아니다.
- 1100에 필요한 것은 기존 JTT 잔차와 약 2.6% 이상 상관된 새로운 축이다.
- 가장 큰 미사용 후보는 선수 상태의 시간 전이와 hard match 밖 TrackMan 정보의 불확실성 보존이다.

따라서 이 설계의 목표 개발 구간은 다음처럼 잡는다.

| 구성 | 목표 검증 개선폭 |
|---|---:|
| Dynamic state expert | +35 ~ +70 |
| Soft TrackMan expert | +15 ~ +40 |
| Low-rank interaction | +10 ~ +30 |
| Row-wise fusion/보수적 calibration | +10 ~ +25 |
| 중복 제거 후 전체 | +90 이상 |

위 숫자는 합산 보장이 아니라 실험 중단을 위한 개발 목표다. 정직한 temporal fold가 전체 `+90`과 보수적 하한 `+67.3`을 확인할 때만 “1100점이 나올 만한 모델”로 승인한다.
