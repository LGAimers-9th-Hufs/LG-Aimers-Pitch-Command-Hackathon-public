# 09 — 추가 후보 발굴 + 인접 연구 조사 (5방향 병렬 조사 종합)

> 목적: `06_candidate_pipelines_catalog.md`(M0~M23, F1~F14) 이후 **카탈로그에 없던 새 후보**를 발굴하고,
> 동시에 프로젝트에 활용 가능한 인접 연구(야구 커맨드 지표, 시간분할 대회 기법, 캘리브레이션, 크로스도메인)를 조사.
> 조사 5축: ① 신규 tabular 모델 ② 야구 제구 도메인 ③ 시간분할 대회 우승 기법 ④ 캘리·손실·타깃 심화 ⑤ 크로스도메인·콜드스타트.
> 신규 arm의 카탈로그 반영(M24~M36, F15~F23, T-e/T-f, C4~C6, E3~E4, 신설 축 E=P1~P8)은 `06` 참조.

---

## 0. 종합 — 이번 조사의 5대 발견

1. **시간분할 증거가 "새 모델 추가"보다 "프로토콜"을 가리킨다.** TabReD(ICLR'25, 시간분할 산업 벤치마크)에서 GBDT+MLP-PLR만 최상위, 랜덤분할에서 좋던 신형 구조는 이득 소멸. ICML'25(arXiv:2502.20260)는 아키텍처보다 **시차 최소화 refit·검증셋 설계**가 지배적이라고 보고. → 신설 축 E(P1~P8 프로토콜 arm)가 모델 arm보다 기대 ROI 높음.
2. **익명 ID의 정석은 random effect.** 의료(병원 ID)·보험(직군 ID)에서 반복 검증된 GLMM 구조 — GLMMNet/LMMNN(NN), GPBoost/MERF(GBT 하이브리드)는 신인 투수를 "prior로 자동 fallback"시키는 콜드스타트 내장 구조. M15(베이지안 log5)의 형제 후보로 M24/M25 추가.
3. **릴리스가 곧 로케이션 (Kirby Index).** 수직/수평 릴리스 각도·포인트의 표준편차만으로 위치 분산의 R²=0.85~0.92 설명, 1~2경기 소표본에서도 신뢰 가능 — 2025 신인/콜드스타트에 가장 실용적인 신규 피처(F18). KBO에 위치 기반 제구 지표 공개 사례가 사실상 없음 → **"MLB Location+/CSAA/xCTRL 계보의 KBO 최초 이식" 내러티브가 가산점 어필**.
4. **지표 불확실(LogLoss/AUC/Brier)의 정답은 "판별모델 + 사후 캘리 + 극단 억제".** 캘리는 단조변환이라 AUC 불변, Brier/LogLoss 동시 개선. 시간분할에선 캘리 자체가 깨지므로 **최근 윈도우 재캘리 + Saerens prior-shift 로짓 보정**(C5)이 이 대회 고유의 최대 레버리지.
5. **앙상블 운영 원칙 확정.** LogLoss에서 rank averaging 금지(캘리 파괴), prob-mean/logit-mean/hill-climb만. 스윕에서 나온 top-k 이질 설정을 버리지 않고 앙상블 멤버로 재활용(hyper-deep ensembles, NeurIPS'20)하면 사실상 공짜 성능.

---

## 1. 신규 모델 후보 (조사축 ①: 2024~26 tabular + 검증된 고전)

### 1.1 채택 후보 (카탈로그 추가)

| 신규 ID | 모델 | 요지 | prior 근거 | 티어 |
|---|---|---|---|---|
| **M24** | **GLMMNet / LMMNN** | 익명 투수/타자 ID를 **random effect**로 두는 NN — GLMM NLL을 손실로 통합, 변분추론. GLMMNet은 Bernoulli 직접 지원 | NeurIPS'21, JMLR'23(LMMNN), arXiv:2301.12710(GLMMNet, 보험 실검증). 신인=prior 자동 fallback | T3 보완 |
| **M25** | **GPBoost / MERF** (GBT+랜덤효과 하이브리드) | 상황효과는 트리, 투수효과는 가우시안 랜덤효과 — GBT 본체에 직접 결합 | 의료 벤치마크 PMC11505330, medRxiv 42병원 22.7만명. M15의 GBT 결합판 | T3 보완 |
| **M26** | **상태공간 제구 스킬 추적** (칼만/입자필터) | 투수별 '제구 로짓'을 local-level 랜덤워크 잠재상태로 필터링. Glicko식 비활동 시 분산 증가, TrueSkill2식 보조관측(구속·존이탈) 통합 | arXiv:2308.02414(Elo/Glicko/TrueSkill=상태공간 통일 관점), arXiv:1706.03442(hot-hand 상태공간), TrueSkill2(52%→68%) | T3 보완 |
| **M27** | **GRANDE** (end-to-end 그래디언트 트리 앙상블) | hard axis-aligned 트리를 straight-through로 학습 — 트리 귀납편향+그래디언트 유연성 → GBDT와 오차 비상관 다양성원 | ICLR'24(arXiv:2309.17130), 19개 이진분류에서 XGB/CatBoost 상회 주장. 시간분할 미검증 | T4 보완 |
| **M28** | **xRFM** (커널 특징학습 + 트리 분할) | AGOP 커널 머신 스케일업 — 트리/NN 어느 쪽과도 오차 상관 낮은 제3계열 | arXiv:2508.10053 (200개 분류 GBDT 상회 주장, 랜덤분할·자체평가) | T4 보완·미검증 |
| **M29** | **EBM** (Explainable Boosting Machine) | GA2M 가산모델 — 완전 해석 가능 + GBDT 근접 정확도. **해석성 가산점 각도** + 매끈한 형상함수라 저비용 블렌딩 멤버 | arXiv:1909.09223, interpretml. "새로운 제구력 평가법" 발표 소재로도 활용 | T3 보완(해석성) |
| **M30** | **FFM** (field-aware FM, libffm) | 필드별 잠재벡터 FM — Criteo/Avazu CTR 대회 우승 모델. LogLoss+희소 ID 상호작용이라는 검증 조건이 과제와 동형. CPU 수 분 | Juan et al. RecSys'16 | T4 보완·저비용 |
| **M31** | **ModernNCA** (딥 검색 베이스라인) | TALENT 300셋 DL 최상위. 단 **TabReD(시간분할)에서 이득 전이 실패** — 검색 계열의 분포이동 취약성 명시 | ICLR'25(arXiv:2407.03257) | T4 (시간분할 리스크) |
| **M32** | **TFM 슬롯 다양성원/버전업**: TabDPT · TabPFN-3 · Mitra | TabDPT는 실데이터 사전학습이라 TabPFN(합성 prior)과 오차 비상관 기대. 실데이터 sweep 시점에 최신 체크포인트 채택 | arXiv:2410.18164(TabDPT), 2605.13986(TabPFN-3), 2510.21204(Mitra) | T4 (M18/M19 슬롯 업데이트) |
| **M33** | **Supervised AE+MLP** (Jane Street 1위식) | denoising AE+지도 분기를 **fold 내부에서 공동학습**(전체 데이터 사전학습=누수). Purged time-series CV와 세트 | Jane Street 1위 writeup, Numerai 포럼 해설 | T4 |
| **M34** | **이변량 정규 실행오차 생성모델** (다트 모델) | 착점=의도점+2D 정규 산포. 투수(×구종) 공분산을 **empirical Bayes로 집단 prior에 shrink**, 성공확률=성공영역 적분. 화이트박스 | Tibshirani darts(JRSS-A 2011), arXiv:2302.10750. 좌표+위치기반 라벨이면 T3 승격 | T4 |
| **M35** | 확률적 부스팅 계열: NGBoost / XGBoostLSS / PGBM / RuleFit | 이진 분류는 GBDT가 이미 Bernoulli NLL을 직접 최적화 — 분포 부스팅의 부가가치가 수학적으로 없음(회귀 전용 설계) | arXiv:1910.03225 등 | T5 짐 |
| **M36** | 기타 신규 딥: ExcelFormer / Trompt / BiSHop / TabKANet / MambaTab / CARTE | 랜덤분할 주장만 있거나(ExcelFormer/BiSHop), TabReD 일반화 실패 명시(Trompt), 강점이 익명화로 무력화(CARTE 문자열 의미) | 각 arXiv, TabReD 결과 | T5 미검증/짐 |

### 1.2 "GBDT를 실제로 이긴" 조건 (벤치마크 증거)

- **TabArena** (living benchmark, 51셋, arXiv:2506.16791): 관례 튜닝에선 CatBoost 1위. post-hoc 앙상블 포함 대예산에선 TabM·RealMLP 최상위. TabPFN류는 **<~1만 행에서만** 우세. 결론: *"GBDT를 이기는 모델보다 앙상블에 잘 기여하는 모델을 찾아라"* → 신규 후보 선별 기준=앙상블 기여도.
- **TabReD** (ICLR'25, 시간분할, arXiv:2406.19380): GBDT+MLP-PLR만 최상위, 방법 간 격차도 축소. 이 해커톤과 가장 유사한 조건의 음성 증거.
- **ICML'25 temporal shift** (arXiv:2502.20260): 학습-테스트 **시차 최소화(최신 데이터 포함 최종 refit)** + 검증셋 편향 축소가 전 방법 공통으로 최대 개선 → P3 arm.

---

## 2. 야구 제구 도메인 (조사축 ②: 커맨드 지표·의도추정·릴리스·피로)

### 2.1 공개 커맨드 지표 방법론 → 피처 전환

| 지표 | 계산법 요지 | → 활용 (신규 ID) |
|---|---|---|
| **Location+ / PitchingBot Command** (FanGraphs) | 구위 변수 없이 **구종×카운트(×좌우)별 실제 위치만**으로 run value 예측 (PitchingBot은 XGBoost+최소 피처셋) | **F15**: 카운트×구종×좌우 조건부 위치가치 그리드 피처. PitchingBot 구조(XGB+카운트·좌우·구종군·위치)는 우리 P1 baseline과 동형 — 그대로 복제 가능 |
| **CSAA** (Baseball Prospectus) | 스윙 없는 투구에 **GLMM** — 투수·포수·타자·심판을 random effect로 분리, 투수 BLUP=커맨드 프록시 | M15/M24와 동일 구조의 도메인 선례. "제구 성공"을 GLMM으로 분해 → 투수 random effect를 피처로 |
| **Command+ / Pitch Intent** (STATS) | 포수 셋업=의도 타깃, 인치 단위 miss distance | 포수 위치 없어도 GMM 추정 타깃으로 대체(=F7 xCTRL 정당화). SIS 발견: **수직 미스가 수평보다 치명적** → **F23** miss-direction 비대칭 피처 |
| **Edge%** (Petti & Zimmerman) | 존을 heart/수평 edge/상하 edge 버킷으로 나눠 edge 투구 비율 집계. 연도 간 r=0.59 | F15에 포함: 투수별 롤링 Edge%/Heart% (구종·카운트 조건부) — 구현 비용 최저 |
| **Command Score** (Ben-Porat, THT) | 타자가 약한 위치에 던져 잉여가치를 만드는 능력 (MARS) | 보조 타깃/피처 아이디어 — 위치 정확도가 아닌 **가치 기준** 제구 재정의 |

### 2.2 릴리스 ↔ 커맨드 (콜드스타트 킬러 피처)

- **Kirby Index** (FanGraphs 2024): 수직/수평 **릴리스 각도(VRA/HRA) + 릴리스 포인트** 4변수의 SD → 위치 분산 R²=0.92(수직)/0.85(수평). 연도 간 안정성 R²=0.5로 Location+(0.39)보다 높고 **1~2경기 표본에서 신뢰 가능**.
- Whiteside et al.: 구속 변동성·릴리스 일관성이 FIP 예측. Sensors 2022: 릴리스 타이밍 변수 유의.
- → **F18**: 투수×구종별 릴리스 포인트·각도 롤링 SD(=미니 Kirby Index) + **당일 릴리스 드리프트**(경기 초반 대비 이동량=실시간 피로 프록시). F4의 정밀화·상위호환.

### 2.3 다음 투구 예측 선행연구 (S3/M12 뒷받침)

- MIT Sloan 2012 (투수별 SVM, 다음 구종), IEEE 2022 attention-LSTM(76.7%), MIT Sloan Transformer 투구 결과 예측, JSA DNN 앙상블(구종+존 동시 예측), KCI 국내 논문(KBO 구종 예측+SHAP).
- → "직전 N구가 강한 신호"라는 시퀀스 조건부 구조가 반복 검증됨 — S3(시퀀스 student)·M12(타깃어텐션)의 prior 상향 근거.

### 2.4 피로·TTO (F3 확장 → F19)

- 체계적 리뷰(PMC6673423): 투구수·이닝 증가, 경기/시즌 후반 성적 저하. **Times-through-order는 비단조**(1바퀴 최저→2바퀴 개선→3바퀴 하락, Prospects365) — 선형 피로 피처로는 놓침.
- → **F19**: TTO 구간더미/스플라인 + 등판 간격 + 시즌 누적 이닝 + 직전 이닝 투구수 + gap-time(최근 실패 후 경과 투구수).

---

## 3. 시간분할 대회 우승 기법 (조사축 ③: 신설 축 E의 근거)

### 3.1 유사 구조 대회에서 배운 것

| 대회 | 핵심 교훈 | → arm |
|---|---|---|
| **AMEX Default** (2022, private=학습+18개월) | 1위: LGB+NN 고정가중 앙상블, **시퀀스에 GBDT를 돌려 뽑은 'series 메타피처'(OOF 중간예측을 피처로)**. 상위권 공통: train-vs-private **adversarial validation으로 흔들리는 피처 삭제**. 15위(Deotte): test pseudo-label+distillation | **P1**(AV), **F21**(series 메타피처), **P4**(pseudo-label) |
| **Jane Street** (2021) | 1위: Supervised AE+MLP를 **fold 내 공동학습**(전체 사전학습=누수) + Purged Group Time-Series CV + 시드 평균 | **M33**, P5 |
| **Ubiquant** (2022) | 1위 "Betting Strategy": CV가 미래 레짐을 보장 못함을 인정, **최종 2제출을 서로 다른 가정(전체학습 vs 최근 레짐)에 분산**. 2위: "Robust CV" 자체가 등수 | **P8**(제출 페어링), **P5**(CV 메타실험) |
| **Optiver** (2023-24) | 1위: live 기간 재학습이 최대 단일 개선 → 배치 번역: "test와 가장 가까운 데이터를 마지막에 흡수" | **P3**(best-iter 고정 후 100% refit) |
| **MLB PDE** (2021, 야구) | 2위(H2O): 최강 피처=**타깃의 롤링 lag aggregate**(12개월 평균/분산 등) | F1 재확인 — 롤링 target-lag가 야구에서 최상위 피처라는 직접 증거 |
| **Home Credit 2024** | 지표 자체가 시간 안정성(주별 gini 기울기·분산 페널티). 지표 게이밍보다 안정 피처+정공법이 승리 | **P6**(시즌/월별 LogLoss 분해를 게이트 부가지표로) |

### 3.2 분포 시프트 대응 (2025 시즌 drift)

- **Adversarial validation**: train(2019-24) vs test(2025) 이진분류기 → AUC≥0.6이면 시프트. 대응 3단: 상위 피처 drop(예측력 좋으면 상대화·버킷화로 완화 우선) / test-유사 행 선택 / AV 확률을 sample weight로. **2025 입력을 보유하므로 즉시 실행 가능 — 최우선.**
- **Recency weighting**: decay^(2024−year) 스윕. 야구는 금융보다 drift 완만 → 과한 감쇠는 표본 손실이 더 클 수 있음, 스윕으로 결정.
- **Feature neutralization** (Numerai): logit에서 시프트 피처 선형성분 부분 제거 — 평균 성능을 깎을 수 있는 보험 성격, 저우선.
- **Test-time adaptation의 배치 번역**: 인코딩·스케일러 통계를 train+test 합산으로 계산하는 arm vs train-only arm.

### 3.3 CV 설계·앙상블 세부

- **이원화**: 시즌 GroupKFold=스크리닝(분산 안정), expanding window(2019-22→23, 2019-23→24)=최종 게이트(배포 상황 모사). 롤링 피처가 fold 경계를 넘으면 사실상 누수 → 피처 생성은 엄격 과거-only shift. 같은 경기/타석이 train-valid에 갈리지 않게 경기 단위 그룹화.
- **2024 최종 게이트를 반복 참조하지 말 것** (2024 과적합). 스크리닝 CV와 최종 holdout 분리.
- **LogLoss에서 rank averaging 금지** — 확률 산술평균(과신 완화) 또는 logit 평균. 멤버 간 캘리 수준이 다르면 산술평균이 안전. hill-climb 가중(OOF LogLoss 직접 최소화)이 고정가중보다 우수 (NVIDIA GM 플레이북).
- **스태킹 시간분할 특칙**: base/meta 동일 fold 공유, OOF는 시간질서 fold로 생성, early-stopping 세트와 OOF 세트 분리, 타깃 인코딩은 fold 내부 적합.
- **GBDT 대규모 튜닝 우선순위**: lr+early stop → num_leaves(63~255) → **min_data_in_leaf(500~3000, 시간 안정성의 핵심 손잡이)** → feature/bagging fraction → L2. 스크리닝 lr 0.1 → 최종만 0.03+refit. 미세튜닝보다 피처 블록·시프트 대응·앙상블이 기대이득 큼.

---

## 4. 캘리브레이션·손실·타깃 심화 (조사축 ④)

### 4.1 캘리브레이션 (C4~C6의 근거)

- **최근 윈도우 재캘리 + prior-shift 보정 (C5, T1)**: 시간분할에서 base rate·분포 이동으로 캘리 map이 깨짐. 최근 구간(last-k 20~50%)만으로 재적합 + **Saerens EM/BBSE로 test 이벤트율 추정 → 로짓 오프셋 보정**(King & Zeng 절편 보정과 동계). 순서 주의: prior 보정과 캘리를 함께 설계해야 재보정기가 이벤트율 변화를 잘못 흡수하지 않음.
- **Venn-Abers (C4, T2)**: isotonic 2개(g0/g1)로 [p0,p1] 구간 → p=p1/(1−p0+p1). 소규모 캘리셋에서 isotonic overfit 방어, validity 보장. cross-VAP fold=5.
- **극단 억제 (C6, T1)**: LogLoss는 자신만만한 오답에 무한대 페널티 → prob clip ε∈[1e-3,1e-2]. **지표 불확실 대응**: 캘리=단조변환이라 AUC 불변·Brier/LogLoss 동시 개선 → "판별모델+사후캘리+클립"이면 3지표 모두 견고. 스윕 리포트는 3지표 동시 기록.

### 4.2 손실 변형 (딥 student 한정 주의)

- **Focal loss**: strictly proper 아님 → 구조적 underconfident. 딥 student에 한해 "focal 학습→**반드시 재캘리**" 조합만 (T3). GBDT엔 이득 작음 (T4).
- **Label smoothing**: 작은 α(0.01~0.05)+재캘리 전제로 T2 (딥). 큰 α는 LogLoss 직접 손해.
- **리밸런싱 경고**: class imbalance correction이 확률 캘리를 망침(arXiv:2202.09101) → **원분포 학습 + 사후 prior/캘리 보정**이 정석. 리샘플링 금지 원칙 재확인.

### 4.3 보조 타깃·soft label (T-e, T-f)

- **T-e 멀티태스크**: 딥 student는 멀티헤드(위치/스트라이크/헛스윙 보조 라벨, 주 타깃 loss 가중↑). GBDT는 공유표현 불가 → **보조타깃 OOF 예측을 피처 주입**(스태킹 표준). TabPFN-MT, MultiTab 근거.
- **T-f soft-label 증류**: born-again(student가 교사 능가 가능), Pocket-FM(TFM→GBDT 증류), soft GBM. `distill_target={hard, soft(T∈{1,2,4}), mix(λ∈{0.5,0.7})}` — 08 증류계획의 arm화. 증류 후 재캘리 필수.

### 4.4 샘플 가중

- recency weighting **T1** (§3.2와 동일 arm), 엔티티별 가중 T3, curriculum learning은 증거 혼재(랜덤과 동급인 반례 다수)로 T4 탐색 전용.

---

## 5. 크로스도메인·콜드스타트 (조사축 ⑤)

### 5.1 타 스포츠 '실행 정확도' 구조

| 도메인 | 구조 | → 전이 |
|---|---|---|
| **축구 xPass/xG** (StatsBomb) | **난이도 모델(상황만)과 능력 모델(선수 오프셋) 분리** — 능력=누적(실제−기대) 잔차의 shrinkage 추정 | **F16**: 투수ID 제외 '상황만' xSuccess 모델 + 투수별 (실제−기대) 누적 잔차 피처 — leak-free 능력 피처 |
| **NBA 슛** (tracking) | 저확률 슛일수록 수비 느슨 = **셀렉션 교란**. 궤적 RNN이 정적 피처를 능가 | 카운트가 의도 로케이션을 바꾸는 confounder → 상황×투수 교호작용 명시. 시퀀스 모델 근거 보강 |
| **테니스 서브** | 프로는 혼합전략, 피로가 전략 변경. 직전 fault 후 행동 조정 | **F19**에 행동조정 플래그(직전 볼넷/폭투 직후) 포함 |
| **골프 SG** (Broadie) | 상태(거리×라이)별 기대치 테이블 + 개인 편차 | 카운트×구종×존 '상태별 리그 평균 성공률 테이블 + 개인 오프셋' 저용량 베이스라인 (F15와 동계) |
| **다트** (Tibshirani JRSS-A) | 착점=2D 정규 산포, EM으로 개인 σ, **empirical Bayes로 소표본 shrink** | **M34** 생성모델 그대로 |

### 5.2 동적 스킬·hot-hand

- Elo/Glicko/TrueSkill=상태공간 모델의 근사 추론(arXiv:2308.02414) → **M26/F20**: 필터링된 제구 로짓 평균+분산 피처, 비활동 시 분산 증가, TrueSkill2식 다중 관측.
- **Miller & Sanjurjo (Econometrica 2018)**: "연속 성공 후 성공률" 추정량은 소표본 편향 — 보정하면 hot hand 실재. → **P7**: 우리 데이터에서 제구의 단기 자기상관을 **순열검정으로 먼저 진단**하고 단기 window 피처 arm의 prior를 조정. window 통계 피처 자체도 시퀀스 길이 보정/베이지안 평활 필요.

### 5.3 의료·신뢰성

- 병원 ID random effect ML(medRxiv 42병원), XGBoost+병원 랜덤효과(PMC11505330) → M25. 참조표본 밖 신규 병원=랜덤효과 prior 평균 0으로 예측(PMC2838162) = 신인 투수 콜드스타트의 통계적 정석.
- frailty(개체 취약성 승수)·gap time·누적 이벤트 수(마모 모델링 관례) → F19에 포함.

### 5.4 콜드스타트 3층 대응

1. **통계적 (최우선, 비용 최소)**: **F17** 베타-이항 shrinkage — 투수(×구종×존) 성공률을 계층 prior로 shrink한 posterior mean + **posterior 분산(불확실성)도 동시 주입**. Efron-Morris James-Stein(첫 45타석으로 잔여 시즌 예측에서 원시 평균 압도).
2. **임베딩**: **F22** Meta-Embedding(SIGIR'19)/MWUF(SIGIR'21)/GME — 신인 ID 임베딩을 속성 기반 generator로 초기화. 기존 딥 arm(F6/M12) 부착형.
3. **최적화**: MeLU식 few-shot 적응(첫 N구=support set) — 해커톤 규모에선 후순위.

---

## 6. 우선순위 종합 (구현비용 대비 기대효과, prior 기준)

**즉시 (라벨 확정 전에도 준비 가능):**
1. **P1** adversarial validation (2025 입력 보유 — 시프트 피처 목록이 이후 모든 결정에 영향)
2. **P5** CV 프로토콜 메타실험 (어떤 CV가 2024를 가장 잘 예측하는가)
3. **F17** shrinkage 피처 · **F16** 난이도-능력 분리 · **F15** 위치가치 그리드+Edge% · **F18** 미니 Kirby Index

**주력 스윕에 편입:**
4. **P2** recency decay · **P3** 100% refit · **C5** 최근윈도우 재캘리+prior-shift · **C6** 클립+3지표 리포트
5. **M25** GPBoost/MERF · **M24** GLMMNet (M15와 같은 브랜치에서 비교) · **T-f** soft-label 증류 arm
6. **E3/E4** 앙상블 운영 (prob/logit-mean·hill-climb, 스윕 top-k 재활용, rank-avg 금지)

**탐색 예산 (~20%):**
7. **M26/F20** 상태공간 · **M27~M34** 다양성원 롱샷 · **F21~F23** · **P4** pseudo-label(규정 확인 후) · **P7** hot-hand 진단 · **P8** 제출 페어링
8. **M35/M36**은 저-prior — 하드 제외 없이 공정 조건으로 소수 실험.

> 원칙 불변: 티어=예산 배분용 prior일 뿐, 채택은 시간분할 CV LogLoss 게이트가 판정. 롱샷에도 공정한 조건(임베딩·튜닝 예산).

---

## 부록 — 주요 출처

**모델**: GRANDE arXiv:2309.17130 · xRFM 2508.10053 · ModernNCA 2407.03257 · TabDPT 2410.18164 · LMMNN JMLR'23 · GLMMNet 2301.12710 · FFM RecSys'16 · EBM 1909.09223 · hyper-deep ensembles 2006.13570 · SWA 1803.05407 · TabArena 2506.16791 · TabReD 2406.19380 · temporal-shift ICML'25 2502.20260 · TabPFN-3 2605.13986 · Mitra 2510.21204
**야구**: xCTRL 2508.19184 · FanGraphs Location+/PitchingBot/Kirby Index · BP CSAA · STATS Pitch Intent · Edge%(Petti) · SIS 미트 위치 · IEEE 9859411(attention-LSTM) · MIT Sloan 2012/Transformer · PMC6673423(피로) · Prospects365(TTO)
**대회**: AMEX 1위 github.com/jxzly · Jane Street 1위 writeup · Ubiquant 1/2위 · Optiver 1위 · MLB PDE(H2O) · Home Credit 2024 · NVIDIA GM 플레이북 · Numerai FNC · Uber AV 2004.03045 · LightGBM 공식 튜닝 가이드
**캘리/손실**: Venn-Abers(Vovk) · focal-properness 2408.11598 · imbalance harm 2202.09101 · King & Zeng 2001 · Saerens prior-shift(2505.19068 계열) · Pocket-FM 2605.18654 · soft GBM 2006.04059 · TabPFN-MT 2605.20234
**크로스도메인**: StatsBomb xPass · Broadie SG · Tibshirani darts JRSS-A 2011 · darts EB 2302.10750 · Miller-Sanjurjo Econometrica 2018 · 상태공간 스킬 2308.02414 · TrueSkill2 · MeLU 1908.00413 · Meta-Embedding 1904.11547 · MWUF SIGIR'21 · Efron-Morris CASI Ch.7
