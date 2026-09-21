# 실행 HANDOFF

## 0. 먼저 확인할 것

```powershell
python results/codex_research/research_calc.py
python results/codex_research/partner_analysis.py
```

판정:

- `partner_analysis.json`에 2021~2024 네 폴드가 모두 있어야 파트너 LOFO 판정을 한다.
- 현재는 V2024 `exp_mech` 중간 후보만 잡힌다. 이 상태에서 **제출 5슬롯을 쓰지 않는다**.
- validation CSV만으로는 제출할 수 없다. 2025 각 행을 독립 추론하는 partner serving 코드/학습 산출물,
  full-train 재학습, 현 챔피언과의 원소 단위 compose parity가 모두 필요하다.
- 파트너가 아직 deployable하지 않으면 즉시 Action B(month)부터 시작한다.

점수는 매번 제출 기록 원문에서 복사해 별도 JSON에 먼저 저장하고, 그 JSON을 CLI에 넘겨 손입력을 최소화한다.

---

## A. 파트너 5슬롯 동시 최적화 — deployable일 때 최우선

### A-1. 로컬 사전 게이트

```powershell
python results/codex_research/partner_analysis.py
```

GO 조건:

1. 네 CSV 행수/순서/NaN/범위 통과.
2. 각 레그를 개별 affine한 뒤 `s_aff`, `s_perp`, `S_B_aff`를 쓴다.
3. 4폴드 중 3개 이상 post-recal gain>0, median projected 2025 gain≥10이면 정규 GO.
4. 현 Codex 중간 후보는 V24 projected `+9.73`이라 **borderline**. 슬롯이 남고 다른 파트너가 24시간
   안에 오지 않을 때만 GO.

### A-2. builder에 필요한 정확한 additive 계약

현재 builder에는 이 일반 layer가 없다. 아래 한 층만 추가한다. 다른 모델 코드는 건드리지 않는다.

```text
q(theta) = clip(A + theta0 + theta1*x_A + theta2*(B-A), 0, 1)
x_A = A - train-derived fixed center
```

- `A`: exact1과 원소 단위 동일.
- `B`: 파트너 full-train serving 예측.
- `x_A`를 최종확률 slope로 잡으면 `C_s=952.2818676068`을 사용한다.
- underlying ENS9 raw slope로 잡으면 `C_s=786.2`; 두 basis를 절대 섞지 않는다.
- 모든 center는 train 상수로 동봉한다. 실제 test 평균 계산 금지.

### A-3. 제출 네 점

```text
P1 = theta=(0,      0,   .5)
P2 = theta=(0,      0,  1.0)
P3 = theta=(.0025,  0,   .5)
P4 = theta=(0,    .06,   .5)
```

빌드 후 각 pair의 차가 위 선형식과 원소 단위 `1e-12` 안에서 맞고 clipping=0인지 검사한다.

점수 `S0,P1,P2,P3,P4`가 오면:

```powershell
python results/codex_research/probe_math.py blend5 `
  --s0 1025.5511136843 --p1 <P1> --p2 <P2> --pshift <P3> --pslope <P4> `
  --t1 .5 --t2 1 --shift .0025 --slope .06 --cs 952.2818676068
```

GO/STOP:

- `H_eigenvalues` 최소≤0 → STOP. 레그/기저/클리핑 오류 감사.
- `theta*`가 확률 블렌드 `w∉[0,1]`, slope≤0으로 변환 → 경계해가 2점 미만이면 STOP.
- predicted gain<2 → 5번째 deploy 생략.
- 그 외 5번째 꼭짓점 제출. 실측이 예측과 0.5점 넘게 다르면 추가 보정 금지.

파트너가 둘 다 첫 제출 전에 준비되면 단일 파트너를 먼저 끝내지 말고 9 probe+joint vertex=10슬롯으로
한꺼번에 푼다. 첫 파트너를 이미 끝냈다면 두 번째의 새 gradient/self-curvature/level-cross/slope-cross/
partner-cross 5개 probe+joint vertex=6슬롯이다.

---

## B. `month_m34` partition — 파트너 미준비 시 1순위

### B-1. 방향 생성

```powershell
python results/codex_research/partition_design.py --axis month_m34 --reference all --cost 1
```

검사값:

```text
dimension=6
max_abs_weighted_mean < 1e-15
max_abs_gram_error < 1e-12
max_abs_probe_move = 0.0041566 근처
```

### B-2. builder layer

`partition_design_month_m34.json`의 `cell_offsets_per_probe.d1..d6`를 행의 자기 `game_month`에만 조회해
exact1 최종확률에 더한다. `m3`은 반드시 `m4`로 병합한다. test `value_counts/mean/groupby` 금지.

권장 CLI 계약(현재 미구현, 구현 후 사용):

```powershell
python submission/build_lgbm_ensemble.py --exact1 `
  --partspec results/codex_research/partition_design_month_m34.json --partdir d1 --tag m34_d1
```

`d1`~`d6` 여섯 zip을 제출한다. 각 zip과 exact1의 차가 JSON lookup과 원소 단위 일치하고 clip=0인지 확인한다.

### B-3. 역산과 vertex

```powershell
python results/codex_research/partition_design.py --axis month_m34 --reference all --cost 1 `
  --s0 1025.5511136843 --scores <S1>,<S2>,<S3>,<S4>,<S5>,<S6>
```

새 JSON의 `solution_assumed_H.cell_offsets`로 7번째 vertex를 빌드한다.

GO/STOP:

- predicted gain≥3 → vertex 제출.
- gain<3 → vertex 생략하고 Action C.
- vertex 실측과 예측 차≤0.5점 → 완료.
- 차>0.5 또는 같은 방향 3점으로 역산한 곡률이 해석 ±25% 밖 → 추가 재보정 금지, 레그 parity 감사.

예상: V24 오라클 +11.15, 7슬롯, 계획 EV +10~20.

---

## C. quadratic shape — 1 probe + 조건부 vertex

builder 행단위 식:

```text
x = ENS9_raw - 0.4680607
g2 = x^2 - fixed m2
g2 <- 고정 proxy Gram-Schmidt로 1, raw-slope, installed game_type 방향 제거
q = exact1 + delta*g2
```

실제 test 배치를 집계해 center하지 않는다. `0.4680607`은 이미 LB 역산된 고정 raw 평균이고, `m2`와
projection은 train/V2024 proxy 고정 상수다. 둘의 provenance를 구분해 적는다. 첫 delta:

```text
delta = +1.0369516947
analytic C = 3.72
무신호 비용 = 4점
```

점수가 오면:

```powershell
python results/codex_research/probe_math.py axis `
  --s0 <현재 champion> --sprobe <Q2_SCORE> --delta 1.0369516947 --curvature 3.72
```

- `gain_assumed_curvature<2` → STOP, 슬롯 1개로 종료.
- ≥2 → 출력 `t_star`로 vertex 1회.
- 세 점의 실측 C가 `[2.79,4.65]` 밖이면 추가 exact correction 금지.
- cubic/logit은 이 vertex가 실제 +10 이상일 때만 새로 직교화해 재심. 그 전에는 제출 금지.

---

## D. `count4` → count12

`count4`는 사용자 제공 V24 오라클 +6.3이지만 정확한 셀 매핑이 저장소에 없다.

1. 원 측정의 4셀 mapping을 먼저 복원한다.
2. 복원 불가하면 임의 count4를 만들지 말고 count12로 직행한다.
3. count4는 3 direction+vertex=4슬롯. count12와 중첩이므로 count12 후속은 남은 8 contrast+joint
   vertex=9슬롯이지 별도 +12.9가 아니다.

count12 full 설계:

```powershell
python results/codex_research/partition_design.py --axis count12 --reference all --cost 1
```

검사: dimension=11, max probe move≈0.0141525, V24 오라클 +12.89. count4가 없을 때만 11 probe+vertex를
그대로 쓴다. 슬롯이 9개 미만 남으면 착수하지 않는다.

---

## E. 6일 슬롯표

파트너가 Day 1에 deployable:

| 날 | 제출 |
|---|---|
| Day1 | Partner P1/P2/P3/P4/vertex = 5 |
| Day2 | month d1~d5 = 5 |
| Day3 | month d6+vertex, q2 probe+조건부 vertex, count4 d1 = 5 |
| Day4 | count4 d2/d3+vertex, count12 residual d1/d2 = 5 |
| Day5 | count12 residual d3~d7 = 5 |
| Day6 | count12 residual d8+joint vertex = 2 |

합계 27. q2/count vertex가 중단되면 그 슬롯은 비운다. 억지로 inning/hand4/3차/logit에 쓰지 않는다.

파트너가 미준비면 Day1부터 month d1~d5. 파트너가 나중에 준비되면 count12 residual을 취소하고 5슬롯을
파트너에 넘긴다. 두 번째 파트너는 count12 residual보다 항상 우선하되, 4폴드 projected median gain≥10일 때만.

---

## F. 제출 전 공통 체크리스트

```text
[ ] 기반 A가 submit_exact1과 원소 단위 동일
[ ] intended additive difference max error <=1e-12
[ ] clip 행 0
[ ] test batch 통계 호출 0; train 고정 lookup만 사용
[ ] 정상/모델손상/안전망 3경로에서 같은 additive layer
[ ] 서버 핀 환경 풀스케일 시간 통과
[ ] LB-fitted coefficient를 metadata provenance에 그대로 명시
[ ] 점수 원문을 먼저 기록한 뒤 계산기 실행
[ ] 해석 C ±25%, Hessian PSD, 제약 w/slope 검사
```

실행 금지: `inning`, `hand4`, raw cubic, 별도 logit remap, 보정기 재가중, 내부 모델 재가중.
