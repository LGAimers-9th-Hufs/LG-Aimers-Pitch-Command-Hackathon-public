# 35 — 채점 완료 레그 스크리닝 + tm3L 행 독립 재빌드 + blendD4 (2026-08-29)

> 계획: `~/.claude/plans/happy-greeting-llama.md`. 목표 >1100, 출발점 blendA3 1057.2623262209.
> 이 세션의 산출물: 제출 원장 `results/lb_history_260829.md` · 새 레그 `submission/dist/submit_tm3L.zip` ·
> 오늘의 제출 후보 `submission/dist/submit_blendD4.zip` · 빌더 `submission/build_tm3_rowwise.py`.

## 0. 결론 먼저

1. **유저 제공 제출 원장이 전제를 바꿨다** — grok_v5/v6은 이미 728.41/716.82로 사망("로컬 1224" =
   2024 암기의 전이 실패 실증). 미채점 고레버리지 레그는 jtt_tm4 하나뿐이었고, 그 전신 v3는 996.24.
2. **1050+ 채점 후보군(SEQ_DIST 1055.18 포함)은 전부 챔피언 오염** — SEQ_DIST zip 안에
   `base/champion/`으로 exact1 metadata가 통째로 들어 있었다(scan BLOCK 10건). DSF_V2·JTT_V9_1 동일.
   재감사의 "1035~1075 라인 전량" 판정이 제3 저장소까지 미친다는 실증.
3. **trackman v2/v3/v4는 역산 상수는 CLEAN이지만 §5 행 독립성 위반 3종을 담고 있다**:
   `league_level(test)` 배치 추정 · `recalibrate()` 배치 평균 이동 · `nanmedian(test 컬럼)` EB base.
   (v4는 추가로 `add_seq()` = test 행 정렬로 투구 순서 복원 — test 자체 as-of.)
4. 그래서 **tm3L을 만들었다**: v3의 학습된 모델은 그대로, 서빙만 train 유도 상수로 재작성한
   행 독립 레그. scan CLEAN · verify 전 관문 통과 · 서버핀 venv 7.8s.
5. **정직한 d 실측이 계획의 낙관을 깎았다**: tm3L d(A3평균) = **0.02377**. jtt as-is의 0.0354는
   hist가 pickle 안이라 `patch_meta_deep`이 못 패치해 당시즌 복원 블록이 죽은 채 잰 **부풀림**이었다
   (D-61과 같은 경로의 세 번째 실증). ⇒ **오늘 1100 도달 조합은 없다.**
6. 오늘의 argmax = **blendD4 = clookup+cmoe+physmix+tm3L 균등 1/4**. 기대 ~1061→1072
   (S(tm3L) 950→996 구간), 손익분기 S(tm3L) ≥ 938.7. 결과에서 S(tm3L)를 정확 역산한다.

## 1. 정확 점수 재기준화 (원장 = `results/lb_history_260829.md`)

- cregime **1045.9643163789** · cmoe **1027.5280314742** · clookup **1049.4561885154** (기존 기록은 반올림)
- A3 레그평균 1027.5856133073 → 실현 분산이득 +29.6767 → **ρ = 42.423/29.6767 = 1.4295** (실현율 70.0%)
- 정정된 오기: ysy_cbb 937.63(937.6278793313), ysy_v3 822.04 추정(구 939.0은 trkm/resid 점수와 혼동)
- 기대식: `E = 평균LB + 프록시이득/1.4295` · 역산식(blendD4): `S(tm3L) = 4·S₄ − 3293.87`
  (= 4·(S₄ − 74.20/1.4295) − 3086.2487, 3086.2487 = clookup+cmoe+physmix 합)

## 2. 컴플라이언스 스크리닝 결과 (0슬롯)

| zip | scan | 비고 |
|---|---|---|
| SEQ_DIST_260822 (1055.18) | **BLOCK 10** | `base/champion/model/metadata.json`에 slope 1.14202·shift −0.00264162·seg_probe·w*=0.7412 전부 |
| DSF_V2_260813 (1049.36) | **BLOCK 10** | 동일 — KBO repo 라인 전체가 챔피언 기반이라는 실증 |
| 팀원 제출본 V9_1 (1052.46) | **BLOCK 10** | 팀원 라인 동일 + 트랙맨 ID 매핑 REVIEW |
| trackman v2/v3/v4 | CLEAN | 단, §5 행간참조 3종(위) — 그대로는 레그 불가 |
| ysy_mlp (미채점) | CLEAN | 추론 경로 행 독립으로 보임(고정 전처리 상수 + row-wise MLP). `enrich()`만 미검 |
| submit_tm3L (신규) | CLEAN | 아래 §3 |

⚠ `results/compliance_scan.json`은 9종 검증셋만 담고 있다(119개 판정표가 아님) — 후보는 매번 새로 스캔.

## 3. tm3L — trackman v3의 행 독립 재빌드 (`submission/build_tm3_rowwise.py`)

원본 배치 통계 3종을 **train만으로 계산한 상수**로 치환:

| 원본 (test 배치) | tm3L (train 상수) |
|---|---|
| `league_level(test)` = asof_prev* 평균 | `L2025 = 2·mean(succ\|2024) − mean(succ\|2023) = 0.472253` (마지막 차분 외삽 — 원본 fallback 0.472와 일치) |
| `recalibrate()` 평균→L 강제 이동 | 고정 `logit_shift = 0.5·[logit(L2025) − logit(m24)] = −0.033110` |
| `nanmedian(test 컬럼)` EB base | train 2024 행 중앙값 8종 동봉 |

- **half-shift 헤지**: 모델의 당시즌 복원 피처가 레벨 하락의 τ(미지)만큼 자체 추적하므로 full shift는
  이중계산. a=Δ/2는 오차 ≤|Δ|/2로 캡(최악 ~23pt, τ~U[0,1] 기대 ~8pt) vs full/zero의 최악 ~80~110pt.
- **피클 분리**: 원본 pkl의 hist DataFrame이 pandas 3.0.2 직렬화라 2.3.3(우리 핀)에서 언피클 불가
  → 모델만 `model_only.pkl` 재직렬화(원본 sha 동봉 기록) + hist는 `model/hist.json`.
  requirements를 우리 레그와 같은 핀(pandas 2.3.3)으로 정렬 — 블렌드 requirements 충돌 제거.
- 검증: 원본 build() 파리티 **0.000e+00** · hist JSON 왕복 파리티 0 · §5 AST OK ·
  정상/서버핀 양 경로 전 관문 통과(7.3s/7.8s) · scan CLEAN.
- 2024 의사-test 참고치: 원본(배치) 대비 −50.1pt. 단 이는 2025용 상수를 2024에 평가한 불공정
  수치 — 2025에서는 L이 표적 연도에 맞으므로 실제 비용은 훨씬 작을 것. S(tm3L) 추정 950~990.
- **측정 전용 변형 `results/leg_matrix/measure/tm3L_p23.zip`**: hist를 2023년말(`hist23.json`)로 바꿔
  프록시에서 당시즌 복원이 살아 있는 상태로 d를 잰다. hist가 pickle 안인 레그는 patch_meta_deep이
  못 건드려 **as-is RMS가 부풀려진다**(jtt_tm3: as-is 0.03515 → 정직 0.02494). 앞으로 파트너 레그의
  d는 반드시 이 방식으로 잴 것.

## 4. 실측과 조합 선택

tm3L의 정직한 거리: cregime 0.02494 · cmoe 0.03055 · physmix 0.02140 · calP 0.02122 ·
**A3평균 0.02377**. physmix/calP와 가장 가깝다 = 당시즌 복원(is4 계열)이 같은 신호축이라는 뜻.

조합별 기대(캐시 예측으로 정확 계산, ρ=1.4295):

| 조합 (균등) | 프록시이득 | S=950 | 970 | 990 | 996 |
|---|---:|---:|---:|---:|---:|
| A3+tm3L | 74.2 | 1060.1 | 1065.1 | 1070.1 | 1071.6 |
| **clookup+cmoe+physmix+tm3L = blendD4** | 74.3 | 1061.0 | 1066.0 | 1071.0 | 1072.5 |
| clookup+cregime+cmoe+physmix+tm3L (5) | 63.9 | 1061.1 | 1065.1 | 1069.1 | 1070.3 |
| A3+tm3L+ysy_mlp (5, mlp=939 가정) | 102.5 | 1066.1 | 1070.1 | 1074.1 | 1075.3 |

blendD4 선택 근거: ① 미지수 1개(tm3L)라 결과에서 정확 역산 → 내일 mlp 5레그의 근거
② clookup 스왑으로 +0.9(둘의 RMS 0.0019 = 사실상 동일 예측이라 점수만 오름)
③ mlp 동시 투입은 미지수 2개 얽힘 + mlp 점수 리스크(자매 939대이나 함수족이 달라 분산 큼).

빌드: `build_team_blend --tag blendD4` — sha 재대조 불일치 0 · 30.6MB ·
requirements 합집합 = joblib/lightgbm/pandas 2.3.3/sklearn (충돌 없음).

## 4b. 실측 (같은 날 밤 추가)

**blendD4 = 1058.6047851923 — 규정 준수 팀 최고 경신 (+1.34).**
정확 gain₄=74.2828로 역산: **S(tm3L) = 940.31** (ρ=1.4295 가정, ρ±5%→±10; S₄ 하나로는
ρ와 S를 분리할 수 없어 ρ의 제3 측정점은 아니다). v3 996.24와의 차 ≈56 = 배치 통계 제거 +
시즌 전이 비용. 예측표의 S=950 행(1061.0)보다 2.4 아래 착지 — 표의 구조는 유지됐다.

이어서 **blendE5 = D4 + ysy_mlp 균등 1/5**를 빌드했다. ysy_mlp §5 전수 확인: `enrich()`와
`build_features()`(catboost_features.py, 61줄) 모두 행 독립 — groupby/rolling/cum·배치 통계 0건,
전처리 상수는 train 동결(preprocess.json), MLP forward는 행 단위. gain₅=102.5849(실현예상 71.76),
기대 S(mlp)=939 가정 **1064.9(+6.3)**, 손익분기 **S(mlp) ≥ 907.6**(자매 937.6~939.6).
역산식: `S(mlp) = 5·(S₅ − 71.76) − 4026.56`.

**1100 스펙 갱신** (D4에 5번째 레그 L 추가 기준, gain₅ = 0.8·74.28 + 64000·d²):
S_L=1000 → 정직 d ≥ **0.0344** · 1025 → 0.0328 · 1045 → **0.0314**. ysy_mlp는 d=0.0260이라
E5 상한 ~1065-1067에 그친다.

## 5. 남은 경로 (1100은 아직 열리지 않았다)

- 내일: S(tm3L) 역산 → `blendE5 = clookup+cmoe+physmix+tm3L+ysy_mlp` (mlp `enrich()` §5 확인 후).
  상한 ~1075 부근. **1100은 여전히 "LB≥1000 & 정직한 d≥0.030" 레그가 필요하다** — 정직한 d 기준이
  생겼으므로 문턱이 더 높아졌다. 후보: 팀원의 클린 신작 / B2(자체 NN — 로컬 상한 834라 비관) /
  tm4의 seq 없는 재빌드(모델이 seq 피처를 기대해 NaN 강제 — 저확률).
- 팀원 문의(슬롯 0): KBO 신작 zip 6종(CAT_NUMPY_FIX 1062 등 — 단 DSF_V2가 BLOCK이므로 champion
  비상속 여부부터), trackman v4 W의 출처, 최종 Private 선택 자동/수동, 잔여 일정, ysy_mlp 등 미채점 점수.
- 제출 시 확인: 서버 로그 DEGRADED 여부(로컬 venv엔 torch가 없어 clookup/cmoe가 죽은 것처럼 보이는
  것은 알려진 무해 현상 — 실서버는 torch 사전설치).
