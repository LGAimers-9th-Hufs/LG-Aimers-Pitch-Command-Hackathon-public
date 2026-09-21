# 큰 LLM으로 모델을 학습/발견할 수 있나? — 해커톤용 판정

> '큰 LLM으로 에이전트 학습'을 이 과제(정형 확률예측)에 맞게: LLM 예측기(①)가 아니라 **증류(②)·LLM AutoML 에이전트(③)·합성데이터(④)·의사라벨(⑤)** 로 조사. 2026 우선 + 적대적 검증.
> ⚠️ 전부 '판정'이 아니라 시간분할 CV LogLoss 게이트로 검증할 **탐색 arm(prior)**.

---

## 결론 한 줄
큰 LLM을 "예측기"가 아니라 "학습 보조/파이프라인 탐색기"로 쓰는 것은 이 대규모(수백만 행)·캘리브레이션 확률(LogLoss/Brier) 과제에서 잘 튜닝된 GBDT를 **이길 근거는 없고 보완만 가능** — 다섯 방식 모두 하드 승리 근거 부족이며, 가장 값어치 있는 arm은 (1) **LLM 지식을 "행"이 아니라 "피처(사전확률·의미 임베딩)"로 주입해 콜드스타트 엔티티를 메우는 것**과 (2) **LLM AutoML 에이전트를 모델 대체가 아니라 피처·앙상블 아이디어 채굴 가속기로 쓰는 것**뿐이다. 이 둘조차 확정이 아니라 시간분할 CV LogLoss 게이트로 검증할 **낮은-중간 prior의 탐색 arm(사전확률)** 이다.

## 방식별 판정표

| 방식 | 대표 방법(연도) | GBDT 대비 | 유망도(★1-5) | 탐색 예산 | 어떻게(게이트) |
|------|----------------|-----------|--------------|-----------|----------------|
| ① LLM→테이블 예측기 지식증류 / LLM 사전확률을 피처로 | LLM-FE(2025), AutoElicit, Pocket-FM(2026), SERSAL(2025) | 대규모에서 **못 이김**, 콜드스타트에서만 보완 | ★★★ (피처 주입 route) / ★☆ (소프트라벨 증류) | 1일×2 arm | 정적 엔티티/컨텍스트 키당 LLM 1회 호출→수치 사전확률·임베딩 컬럼 추가; 엔티티 히스토리 깊이로 CV 층화, 시간분할 OOF LogLoss + ECE 개선 시에만 채택 |
| ② 테이블 FM(TabPFN/TabICL) 증류→빠른 학생 | Pocket-FM(2026), TabPFN-2.5(2025), TabTune(2026) | **비김/약간 짐** (교사가 이 regime에서 GBDT에 못 이김) | ★★ | 1–2일 probe | ≤50k 행 저-카디널리티 서브뷰 + 층화 OOF 소프트라벨(ICL 라벨누수 방지)→학생 학습, 온도 스케일링; 스택의 다양성 멤버로만, 시간분할 CV LogLoss 개선 시 유지 |
| ③ LLM AutoML 에이전트로 파이프라인 탐색 | AIDE, LightAutoDS-Tab(2025), AutoKaggle(2024-25), MLE-bench(2024-26) | **못 이김** (수렴점이 곧 튜닝된 GBDT), 탐색 가속으로 보완 | ★★★ | 예산 상한 고정, 다시드 | 누수-안전 시간분할·LogLoss 목적으로 에이전트 구동→피처/인코딩/앙상블 아이디어만 이식, 시간분할 OOF LogLoss 초과 개선 시에만 채택 |
| ④ LLM 합성/증강 학습데이터 | CLLM(2024), GReaT(2023), TabuLa(2023) | 대규모에서 **무효~약간 해로움** | ★☆ | 반나절~1일, 빠른 kill | 전체 증강 금지; 진짜 콜드스타트 슬라이스만 CLLM식 생성+엄격 큐레이션+합성행 다운웨이트+재캘리브레이션; 전체 시간분할 LogLoss·신뢰도곡선 개선 시에만 |
| ⑤ LLM 의사라벨/약지도/사전확률 생성기 | SERSAL(2024), Snorkel(2018), TabLLM(2023) | 의사라벨은 **짐**(수백만 골드라벨 존재), 사전확률-피처만 보완 | ★☆ (label) / ★★ (prior-feature) | 반나절 | 라벨 생성 금지; 콜드스타트 엔티티에 한해 pitch-time 정보로만 스칼라 사전확률 p_llm 캐시→GBDT 1컬럼; ≥5 rolling-origin 폴드에서 >2σ 개선 + ECE 비악화 시에만, 사전등록 |

## 가장 유망한 탐색 arm (권장)

worth-trying 이상만 추림.

**Arm A — LLM 지식을 "피처"로 주입 (방식 ①의 priors-as-features route). 유망도 ★★★**
- 왜: 증류의 천장은 교사이고, 대규모·고차원·고카디널리티 테이블에서 LLM/TFM 교사는 튜닝된 GBDT에 진다(Pocket-FM 2026: 고차원에서 +0.001 AUC ~51% 승률, 동전던지기). 따라서 전체 LogLoss를 증류로 이기는 길은 근거 부족. 그러나 LLM 세계지식을 **컬럼으로** 넣으면 base-rate 왜곡 없이 as-of 집계가 못 메우는 **콜드스타트(희귀 투수/타자, 신규 구종, 얕은 히스토리)** 갭을 보완할 수 있다(AutoElicit / LLM-FE 2025 템플릿: LLM 생성 피처가 XGBoost·MLP·TabPFN를 일관되게 개선). 이득은 집계에서 ~0으로 희석되므로 엔티티 히스토리 깊이로 CV를 층화해 확인.
- 게이트: 엔티티/컨텍스트 키당 LLM 1회 오프라인 호출(투수 아키타입, 구종 물리 사전확률, 타자 성향 요약, 카운트 레버리지 의미)→수치·짧은 임베딩을 컬럼으로. 추론 시 LLM 0(오프라인 캐시). 시간분할 OOF LogLoss 개선 **and** 최신 시간폴드 ECE/신뢰도곡선 비악화, 다시드/다폴드 재현 시에만 채택.
- 06 카탈로그 추가 ID 제안: **F13 (LLM-derived priors/embeddings as cold-start features)** — 피처 계열이므로 F 접두. 콜드스타트 층화 평가를 필수 메타데이터로.

**Arm B — LLM AutoML 에이전트를 아이디어 채굴 가속기로 (방식 ③). 유망도 ★★★**
- 왜: 에이전트가 tabular에서 수렴·제출하는 모델은 결국 튜닝된 GBDT 앙상블이다(AutoKaggle의 모델 스텝=RF/XGBoost 중 선택; LightAutoDS-Tab 2025의 승리 0.70→0.84는 모델링을 GBDT AutoML에 위임했기 때문). MLE-bench(2024-26): 최고 공개 에이전트도 75개 중 ~17%만 동메달. 즉 새 모델족은 안 나오지만 **피처·인코딩·앙상블·하이퍼파라미터 영역을 더 빨리** 발견할 수 있다 — 보완.
- 게이트: 누수-안전 시간분할·LogLoss 목적을 명시(시간분할임을 알려 look-ahead 피처 방지)하고 서브셋에 구동→후보 피처/전처리/스태킹만 수확해 자체 LightGBM/XGBoost/CatBoost에 이식. CV 노이즈 초과(다시드) 개선 시에만 채택, 예산 상한·kill 조건 고정.
- 06 카탈로그 추가 ID 제안: **M22 (agent-harvested features/ensembling recipes, validated in-house)** — 방법론/파이프라인 계열이므로 M 접두.

**Arm C(선택) — TFM 증류 학생을 스택 다양성 멤버로 (방식 ②). 유망도 ★★**
- 왜: 대규모 regime에 교사 우위가 없어 학생이 GBDT를 이길 순 없지만(Pocket-FM는 교사가 CatBoost에 뒤질 때 "graceful fail"), ≤21피처 저-차원 서브뷰에서 소폭(+0.011 AUC)·다양성 멤버 가치는 남는다. **단 AUC가 아니라 시간분할 LogLoss로만 게이트** (ScoringBench 2026: Brier-랭크 vs LogLoss-랭크 스피어만 ~0.15). TFM 메타러너 스태킹은 금지(Calibration-Trap: 최악 LogLoss).
- 06 카탈로그 추가 ID 제안: **M23 (OOF-distilled TFM student as stack diversity member)**.

## 낮은 prior지만 실험할 것

하드 제외 없이, 최소 비용·빠른 kill로 공정 게이트에 태울 것.

- **방식 ④ LLM 합성행 (★☆):** 전체 증강 금지(대규모에서 이득 소멸, base-rate 왜곡으로 LogLoss 악화 위험). **진짜 콜드스타트 슬라이스에만** CLLM(2024)식 생성 후 aleatoric-uncertainty 큐레이션→소량만 추가, 합성행 다운웨이트 + isotonic/Platt 재캘리브레이션. 전체 시간분할 OOF LogLoss/Brier가 폴드 노이즈 초과 개선 **and** 신뢰도곡선 비악화일 때만 유지. 콜드스타트 subslice의 정확도/F1 개선만으로는 REJECT. 예산 반나절~1일, 첫 실행이 시간폴드에서 baseline 못 이기면 즉시 kill.
- **방식 ⑤ LLM 사전확률-피처 (label 아님, ★★ / label route ★☆):** 의사라벨 절대 금지(수백만 골드라벨이 이미 있어 LLM 라벨은 엄격히 더 시끄럽고 miscalibration 주입; 2504.15432·2506.12468). 콜드스타트/희소 엔티티에 한해 pitch-time 정보로만 스칼라 p_llm을 오프라인 캐시→GBDT 1컬럼. ≥5 rolling-origin 폴드에서 >2σ LogLoss 개선 **and** ECE 비악화일 때만, 사전등록(fishing 방지). 기대값은 무개선이므로 1패스 후 kill하고 피처엔지니어링·엔티티 타깃인코딩·단조제약 튜닝에 재배분.
- **방식 ② 순수 대체용 증류:** 메인 베팅 금지. 오직 Arm C처럼 다양성 멤버/저-차원 서브뷰 probe로만 1–2일 배정.

## 2026 근거 요약
(실제 2026 논문 우선, 없으면 2025 명시)

- **Pocket Foundation Models (2026, arXiv 2605.18654):** 층화 OOF 소프트라벨로 TFM→GBDT 증류. 저-차원 소규모 +0.011 AUC, 고차원/대규모 +0.001(≈51% 승률, 동전던지기), 교사가 CatBoost에 뒤지면 오히려 악화. "distillation fails gracefully." 지표는 AUC(LogLoss/Brier 아님).
- **Ensembling TFMs — Diversity Ceiling & Calibration Trap (2026, arXiv 2605.18696):** TFM 스태킹이 정확도/ROC-AUC는 올리지만 **최악 LogLoss** 산출("경계 sharpening이 캘리브레이션 파괴"), 6개 TFM 근중복(Q=0.961). LogLoss 과제에 직접 유해.
- **ScoringBench (2026, arXiv 2603.29928):** proper scoring rule로 평가 시 지표별 랭킹 크게 변동, Brier-랭크 vs log-score-랭크 스피어만 ~0.15. AUC 개선이 LogLoss 개선을 보장하지 않음 → 정확히 채점 지표로 게이트해야.
- **LLM-FE (2025, arXiv 2503.14434):** LLM이 진화적 옵티마이저로 피처 변환 생성/정제, XGBoost·MLP·TabPFN 일관 개선. "distilled features" route(=Arm A)의 구체 템플릿.
- **BoostLLM (2026, arXiv 2605.06117):** LLM 구조적 사전확률+부스팅이 few-shot에서만 XGBoost 대등/상회. 대규모 우위 없음.
- **LightAutoDS-Tab (2025, arXiv 2507.13413):** LLM 에이전트가 고전 AutoML(LightAutoML/FEDOT)에 위임해 순수-LLM 에이전트(AIDE 0.70→0.84) 상회 — 승리 원천은 오케스트레이션·전처리, 모델링은 여전히 GBDT.
- **TML-Bench (2026, arXiv 2603.05764):** tabular DS 에이전트 전용 벤치. 랭킹은 유효제출+홀드아웃 점수 기준(캘리브레이션 아님), "큰 시간예산에서 개선"은 compute 반영일 뿐.
- **Illusion of Generalization in Tabular LMs (2026, arXiv 2602.04031)** 및 **OmniTabBench (2026, arXiv 2604.06814):** 둘 다 반증 — LM 일반화 과대평가, 대규모 수치 테이블에서 GBDT가 프론티어, 합성/FM은 소규모에서만 승리.
- **Where LLM Annotators Fail (2026, arXiv 2605.27913):** LLM 라벨 오류가 region-dependent 구조적이라 노이즈 보정 실패.
- (2025 근거) **Feeding LLM Annotations at Your Own Risk (2504.15432)**, **Instance-Dependent Label Noise (2506.12468)**: LLM 라벨 학습이 골드 baseline에 못 미치고 instance-dependent 노이즈가 과신 확률로 캘리브레이션 악화 → LogLoss/Brier 팽창. **TabPFN-2.5 (2511.08667, 실제 2025)**: ≤50k행/≤10k행에서만 우위, 50k+에서 XGBoost가 더 강함. **Light-Weight Benchmarks (2512.00888, 실제 2025)**: TFM ~0.8% 이득에 ~40,000배 지연.

## ⚠️ 검증·하이프 경고

- **"큰 LLM" 라벨 오귀속:** Pocket-FM(2605.18654)·TabPFN-2.5는 **테이블 FM→GBDT** 증류이지 chat-LLM→테이블 메커니즘이 아니다. 방식 ①의 라벨과 실제 증거 사이 간극 존재.
- **Bridge-Garden Dilemma(2026, 2605.26246) 토픽 오귀속:** LLM→LLM 텍스트 생성 증류(exposure bias)이지 tabular/GBDT·Brier/ECE 증거 아님. "hard+soft 혼합" 결과는 테이블 사전확률 근거로 전이 불가.
- **캘리브레이션 head-to-head 증거 전무:** 전 메커니즘에서 튜닝된 GBDT 대비 **Brier/ECE/신뢰도곡선** 직접 비교 논문이 하나도 없다. 모든 우위 주장은 AUC/정확도/리더보드-퍼센타일 기반 → **근거 부족**. 따라서 채택은 반드시 시간분할 CV LogLoss로만.
- **regime 한정 주장을 일반화하지 말 것:** CLLM(n<100)·BoostLLM/TabLLM(few-shot)·SERSAL(zero-gold-label)의 "승리"는 전부 데이터 희소 niche. 수백만 행 과제에 일반 우위로 제시하면 하이프.
- **AUC↔LogLoss 비상관(~0.15) & Calibration-Trap:** AUC를 올리는 증류/스태킹이 채점 지표(LogLoss)를 오히려 악화시킬 수 있다. TFM 메타러너 스태킹은 명시적 금지.
- **에이전트 category error:** LightAutoDS-Tab·AutoKaggle의 "승리"는 GBDT 도구 오케스트레이션/자동화 성공이지 LLM이 GBDT를 out-predict한 게 아니다. MLE-bench ~17% 메달률이 counter-hype 앵커.
- **독립성·재현성 경고:** Pocket-FM/Ensembling/TabTune은 동일 그룹(Lexsi Labs) 비-피어리뷰 preprint 상호인용. TabPFN-2.5의 "tuned tree 대폭 상회"는 벤더(Prior Labs) TabArena **정확도** 마케팅 주장. 에이전트는 run-to-run 고분산·비재현·누수(look-ahead) 리스크 — 단일시드 승리는 노이즈로 간주, 다시드/다폴드 유지 확인 후에만 shipping.