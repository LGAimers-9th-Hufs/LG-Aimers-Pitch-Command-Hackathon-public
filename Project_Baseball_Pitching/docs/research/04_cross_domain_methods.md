# 야구 밖 전이 가능 방법 조사 — GBDT를 이기거나 보완하는가?

> 문제를 일반화('투수×타자 + 맥락 + 시퀀스 → 이진 성공확률')해 CTR/추천·시퀀스·매치업·타스포츠·GBDT초월 도메인 조사. 2026 우선 + 적대적 검증.
> (헬스케어/금융 도메인은 조사 결과가 조작 placeholder로 확인돼 제외)

---

## 결론 한 줄
야구 밖에서 가장 유망한 이식은 (1) **두 엔티티(투수×타자) 상호작용을 겨냥한 CTR 계열의 학습 임베딩 + CAN 코-액션(co-action) 메커니즘**과 (2) **현재 상황을 쿼리로 삼아 과거 이벤트열을 주의(attention)로 요약하는 타깃-어텐션 시퀀스 인코더**이며, 둘 다 GBDT를 대체하기보다 **임베딩·로짓을 피처로 주입하거나 앙상블하는 "보완"** 형태로만 LogLoss/Brier 이득이 기대된다. 여기에 **계층적 베이지안 log5 shrinkage 매치업 사전확률**(구조적으로 캘리브레이션됨)을 GBDT 피처로 넣는 것이 가장 저위험·고ROI 이식이다. (근거: CTR 도메인 transfer_plan; Sequential 도메인 target-attention; Two-tower 도메인 hierarchical Bayesian log5)

## 도메인별 후보 방법 판정표

| 도메인 | 대표 방법(연도) | GBDT 대비 | 우리 과제 유망도 | 어떻게 이식 |
|---|---|---|---|---|
| CTR / 딥 추천 | CAN Co-Action Network(2022), Entity Embeddings(2016), DIN/TWIN 타깃-어텐션(2018/2023) | 보완 (drop-in은 짐; OmniTabBench 2026상 딥 CTR넷은 잘튜닝 GBDT를 일관되게 못 이김) | ★4 | 투수/타자 ID 임베딩(16-64d) 학습 → 컬럼으로 LightGBM에 export; CAN으로 투수×타자 매치업 코-액션; 타깃-어텐션으로 이전 투구열 요약. 반드시 사후 캘리브레이션 |
| 시퀀셜 행동 모델링 | DIN/DIEN(2018/19), BST(2019), SIM/TWIN-V2(2020/24), HSTU(2024) | 보완 (TabReD 2024/25: 시간분할·산업피처에선 GBDT/단순 MLP 승, 복잡 DL 패) | ★4 | 현재 (타자×카운트×상황)을 쿼리로 하는 타깃-어텐션 인코더 학습 → 로짓+풀링 임베딩을 GBDT 피처로 export(페북 GBDT+LR의 역방향). 히스토리 길면 SIM/TWIN 2단 검색 |
| 투-타워/매치업/콜드스타트 | 계층적 베이지안 log5(2018/2025), Entity 임베딩, 시간적 GNN(RelBench v2 2026) | 보완 (GNN은 AUROC로만 이김, 캘리브레이션 미보고; 베이지안 사전확률은 구조적 캘리) | ★4 | log5 partial-pooling 사전확률(캘리·콜드스타트 자동)을 as-of 컬럼으로 GBDT에 주입; GNN 노드 임베딩 추출→GBDT 피처. 종단 GNN은 사후 재캘리 필수 |
| 헬스케어/금융 시계열 리스크 | (없음 — placeholder "a") | 판정 불가 (근거 미해결) | ★1 | 이식 근거 없음. 제외 |
| 타 스포츠 기대성공(xG 등) | XGBoost xG(2025/26), 2단 발생/전환 분해, Seq2Event(2022), 컨텍스추얼 임베딩(2023) | 보완/동률 (타 스포츠도 단일이벤트 캘리 확률에선 GBDT가 SOTA로 수렴) | ★3 | P(발생)×P(성공\|발생) 2단 분해(별도 GBDT 헤드); masked-event 사전학습 임베딩을 피처로; 시퀀스 인코더 벡터 주입. 트래킹 없으면 GNN 스킵 |
| 2026 "GBDT 넘는가?" 종합 | TabM(2025), RealMLP(2024/25), 스태킹, CalArena(2026) | 동률+보완 (대체는 없음; TabM/RealMLP는 타이, 스택이 소폭 실이득) | ★4 | GBDT 앵커 유지 + TabM/RealMLP를 ID 엔티티 임베딩+시퀀스 인코딩으로 두 번째 베이스 학습 → 각자 재캘리 → OOT 로지스틱 메타러너 블렌드 후 재캘리 |

## 진짜 유망한 신규 후보 (권장)

**1. CAN 코-액션 네트워크 (2022) — worth-trying 중 가장 과제-정합적**
- 왜 잠재력: 우리 과제의 핵심은 두 고차원 익명 ID의 상호작용(투수×타자)이다. CAN은 엔티티 B 임베딩을 마이크로-MLP의 가중치로, 엔티티 A 임베딩을 입력으로 넣어 데카르트 곱을 저렴하게 근사한다 — 정확히 "이 팔 vs 이 방망이" 매치업 전용. GBDT는 이 상호작용을 손수 만든 as-of 통계로만 근사한다. 알리바바 배치서 +12% CTR / +8% RPM, 데카르트 곱 피처보다 적은 파라미터로 우위. (CAN 방법·논문)
- 이식: batter 임베딩→가중치, pitcher 임베딩→입력의 코-액션 유닛으로 매치업 벡터 생성 → FinalMLP/DS-MLP 2-스트림 헤드에 concat.

**2. 타깃-어텐션 시퀀스 인코더 (DIN/TWIN 계열) — 새 신호의 최선 후보**
- 왜 잠재력: GBDT가 원천적으로 못 먹는 것은 **순서 의존적 가변길이 이벤트열**이다. 타깃-어텐션은 현재 (타자·카운트·상황)을 쿼리로 과거 투구/타석열을 주의-풀링해, as-of 집계가 뭉개는 순서·셋업 패턴·매치업 기억을 포착한다. 도메인 전체에서 "진짜 새 리프트의 가장 명확한 자리"로 명시됨. (Sequential summary·transfer_plan)
- 이식: 각 토큰 = {구종/존/결과, 카운트, 베이스-아웃, 점수차, 레버리지, 시간갭}. 소형 트랜스포머 블록 + 학습 시간갭 위치 임베딩. 히스토리가 길면 SIM/TWIN 2단 GSU 검색→ESU 어텐션. 로짓+풀링 임베딩을 GBDT에 export.

**3. 계층적 베이지안 log5 shrinkage 매치업 사전확률 (2018/2025) — 가장 저위험 이식**
- 왜 잠재력: 유일하게 **구조적으로 캘리브레이션된** 매치업 확률을 내고, 콜드스타트를 리그 사전확률로의 shrink로 자동 처리한다. On-topic 2025 야구 논문(2511.17733)이 이 레시피가 캘리·저분산임을 보임(다만 마진 넘는 상호작용 자체는 GMP 16.73%→17.02%로 작음 — 그래서 "대체"가 아니라 사전확률을 피처로). 희소 ID에 이상적인 GBDT 입력. (Two-tower method·transfer_plan)
- 이식: pitcher_id·batter_id(및 pitcher×pitch_type)를 랜덤효과로, 로그오즈에서 결합, 엄격 as-of로 사후 성공확률 + 랜덤효과 추정치 + 사후분산을 새 컬럼으로 LightGBM에 주입.

**4. 학습 엔티티 임베딩 (2016) — 가장 값싼 승리 후보**
- 왜 잠재력: 임베딩은 ID-heavy 데이터에서 NN이 트리를 구조적으로 능가하는 유일한 지점(유사 엔티티 간 강도 공유). 단독으론 튜닝 GBDT와 타이(OmniTabBench 2026)지만, **학습된 투수/타자 임베딩을 컬럼으로 GBDT에 되먹이는 것**이 반복 보고된 가장 싼 이득. 단, MultiTab 2025는 "카테고리 피처가 지배할 때만 크게 도움"으로 온도조절.

## 기존 A/B/C 후보에 어떻게 통합

앞서 정한 GBDT 스택 후보에 아래를 **새 브랜치/멤버**로 부착한다.

- **후보 A = 잘 튜닝된 GBDT + as-of 피처(앵커):** 유지. 모든 신규 신호는 여기로 export되는 컬럼이거나 앙상블 멤버로만 진입. GBDT는 대규모·고차원에서 최고의 자연 캘리브레이션(OmniTabBench 2026)이라 앵커 유지가 정답.
- **후보 B = 딥 베이스러너(TabM/RealMLP 2-타워):** 여기에 (i) 투수/타자 엔티티 임베딩, (ii) CAN 코-액션 매치업 벡터, (iii) 타깃-어텐션 시퀀스 임베딩을 concat한 FinalMLP/DS-MLP 헤드로 확장. 즉 CTR·시퀀셜 도메인의 세 부품이 B의 내부 구조가 됨.
- **후보 C = 피처-주입/스태킹 레이어:** 세 갈래 새 브랜치를 컬럼으로 흡수 —
  1. 베이지안 log5 사후확률 + 랜덤효과 + 사후분산 (Two-tower 브랜치)
  2. 동결한 타깃-어텐션 인코더의 로짓·풀링 임베딩 (Sequential 브랜치)
  3. GNN/투-타워/xG 2단 헤드에서 뽑은 노드·컨텍스추얼 임베딩 (매치업/타 스포츠 브랜치)
- **결합·캘리브레이션(비협상):** BCE로 학습, 엄격 시간분할(train 과거 / valid 미래 / as-of 누수 0). 각 모델을 시간-전진 홀드아웃에서 isotonic/temperature로 재캘리 → OOT 예측 위 로지스틱 메타러너로 블렌드 → **블렌드 재캘리("calibration trap" 회피, Ensembling-TFM 2026)**. Brier/LogLoss + reliability diagram + per-count/per-leverage 슬라이스로 매 단계 베이스라인 대비 검증.

## 신중/제외

- **딥 CTR/DLRM/순수 MLP·트랜스포머 drop-in 대체 [loses]:** Grinsztajn·OmniTabBench 2026·TabReD — 시간분할·이질 대규모 tabular에서 트리 승, 딥 패 + 더 나쁜 캘리브레이션. **대체 기대 금지.**
- **DCNv2/xDeepFM/DeepFM 명시적 크로싱넷 [ties]:** GBDT 분할이 이미 저차 상호작용을 잡음 → 대개 엔지니어링 크로스를 재유도하는 수준. 델타 0.001-0.002.
- **HSTU/생성 추천(ULTRA-HSTU 2026) [complements이나 과잉]:** 수천만+ 이벤트·거대 컴퓨트 전제, 단일 이진 타깃 헤드 캘리 미검증. 재사용 파운데이션 인코더가 필요할 때만.
- **TabPFN-3/tabular foundation model [loses/미적용]:** ~1M 행 캡, 컨텍스트 길이 제한, 고차원 ID·시퀀스 네이티브 미지원. 예측기로 부적합, 오프라인 피처 증류에만.
- **GANDALF/GATE/NODE [loses], TabR 검색증강 [complements이나 비용]:** 전자는 TabM/RealMLP에 밀림·캘리 이점 없음; 후자는 수백만 행 검색 비용 과다.
- **헬스케어/금융 도메인 전체:** placeholder "a"로 근거 미해결 → **제외.**

Hype-flag: DS-MLP "…is All You Need" 및 FinalMLP "enhanced/powerful"는 딥 베이스라인 대비 AUC/LogLoss로만 측정, **GBDT 대조·캘리브레이션 평가 전무**. ContextGNN "20% 우위", RelBench v2 "관계구조 우위", HSTU "65.8% better/trillion-param", ULTRA-HSTU "21x faster/4-8% engagement"는 전부 랭킹/engagement 프레이밍으로 확률-캘리 head-to-head 아님.

## 2026 근거 요약 (실제 2026, 없으면 2025)

- **OmniTabBench (2026):** 3030-데이터셋 대규모 메타벤치. GBDT가 대규모·고차원서도 경쟁/우세, 딥·파운데이션 이득은 marginal·비용과다. 지배적 승자 없음 = complement-only 지지. **핵심 정직 앵커.**
- **RelBench v2 (2026):** 시간적 GNN vs LightGBM+수동FE. GNN이 AUROC/AP로 다수 승(user-churn 94.3 vs 83.9) — 단 **LogLoss/Brier·캘리브레이션 미보고**, 판별력 승이지 확률 승 아님.
- **DS-MLP / UniMixer (2026):** Criteo LogLoss 0.4366/AUC 0.8152로 딥 SOTA — **GBDT 베이스라인 없음**, 이득 0.001 수준.
- **CalArena (2026):** GBDT/딥/파운데이션 전반 사후 캘리(Platt/isotonic/temp) 벤치 — 딥 멤버는 블렌드 전 재캘리 필요.
- **Ensembling TFM: "Diversity Ceiling and Calibration Trap" (2026):** 나이브 앙상블이 캘리브레이션을 악화 → 최종 블렌드 재캘리 필수.
- **The Impacts of Increasingly Complex Matchup Models (2025):** 계층적 베이지안 log5 투수×타자×recency, log-loss/GMP로 평가, 구조적 캘리. 상호작용 복잡도의 GMP 이득은 작음(16.73%→17.02%)이나 사전확률 자체는 강함.
- **TabReD (2024/25):** 산업 피처+시간분할에서 GBDT·단순 MLP 승, 복잡 DL 패 — 시퀀셜 계열의 중립 정직 앵커.
- (2025) MCNet 단조 캘리브레이션, MultiTab, StatsBomb xG 재현(Brier ~0.068), xG+ 2단 XGBoost.

## ⚠️ 검증·하이프 경고

- **placeholder 논문 "a" (헬스케어/금융 도메인): 조작/자리표시 의심.** 제목이 단일 문자 "a", 저자·venue·식별자 전무로 해결 불가. 이 도메인의 "complement-only" 판정과 모든 캘리·GBDT 대조 주장은 **근거 없음** → 도메인 전체 제외.
- **연도 오기:** RealMLP는 실제 2024(NeurIPS 2024, arXiv 2407.04491)인데 dossier가 2025로 표기. "Learning Contextual Event Embeddings"는 호스팅 PDF 2022 / SSAC 발표 2023 — 컨퍼런스 연도 라벨은 방어 가능하나 주장은 "3개 스포츠북과 competitive"(우위 아님).
- **메트릭 미스매치(핵심):** TabM·RealMLP·T-MLP·TabR·OmniTabBench는 accuracy/RMSE/AUC로 벤치 — **어느 것도 캘리브레이션/proper scoring에서 GBDT head-to-head를 보고하지 않음.** 이들을 "GBDT를 캘리 확률에서 이긴다"의 근거로 인용하면 overreach.
- **차용 프레이밍 주의:** Ensembling-TFM의 헤드라인 "TFM이 튜닝 GBDT를 이긴다"는 선행연구 인용일 뿐 자체 결과 아님 — 실제 기여는 "calibration trap"(스태킹이 캘리 악화)로 **오히려 경고성**.
- **산업 A/B 수치 오독 주의:** DIEN 20.7% / SIM 7.1% / BST +7.57% 등은 배치된 딥 모델 대비 온라인 랭킹/engagement 리프트이지 GBDT 포함 전방법 대비 오프라인 정확도 승이 아님.
- **소표본/과대일반화:** Tennis Momentum(2509.22670)은 2경기만으로 홀드아웃·캘리·GBDT 베이스라인 없이 광범위 주장. Sloan 야구 트랜스포머(2025)는 LogLoss/Brier·GBDT 베이스라인 미보고 → **시사적 참고만**.
- **업계 "90%+ 정확도" 2026 블로그:** 마케팅, 캘리 확률 근거 아님 — 할인 처리.

핵심 종합: 어떤 외부 도메인 방법도 dossier 근거상 **잘 튜닝된 GBDT를 캘리브레이션 확률(LogLoss/Brier)에서 단독으로 이긴다는 직접 측정 증거가 없다.** 모든 "beats-GBDT" 주장은 랭킹(AUC/AP) 또는 삼각추정이며, 방어 가능한 결론은 "보완/앙상블 이득"과 "각 딥 멤버 재캘리 필수"뿐이다. 반드시 자체 A/B(베이스라인 GBDT vs GBDT+딥피처 vs 딥단독, 전진 시간분할·캘리 LogLoss/Brier)로 검증 후에만 복잡도를 추가할 것.