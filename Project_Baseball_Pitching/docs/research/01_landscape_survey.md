# 야구 투수 예측 — 파이프라인 landscape 조사 (2026 최신 우선)

> LG 트윈스 해커톤('다음 투구 제구 성공 확률') 대비 광범위 조사. 8개 접근각 중 7개 완료(1개 실패).
> ⚠️ 아래는 landscape 참고용. 과제 특화 우승 플레이북은 `02_playbook.md` 참고.

---

## TL;DR 추천

- **투구율 = 구종 비율/다음 구종 분포**: MLB Statcast + 시퀀스 모델이 정답. 최신·novel 지향이면 LLM 월드모델(Neural Sabermetrics, 2026.2), 안정·재현 지향이면 attention-LSTM(2022) 또는 XGBoost 시퀀싱 피처 베이스라인.
- **투구율 = 투구 결과율(스트라이크/볼/타격결과)**: pitch-level 궤적+카운트 피처에 그래디언트 부스팅(XGBoost/LightGBM/CatBoost) + expected run value(xRV/xwOBA) 회귀 타깃이 실무 SOTA(MDPI 2025, 정확도 0.89–0.93).
- **투구율 = 시즌/경기별 성적률(ERA/WHIP 등 rate-stat)**: 시계열 회귀. Temporal Fusion Transformer(2025)에 rest days·누적 투구수 등 워크로드 공변량을 넣는 구성이 최선.
- **공통 원칙**: 사전(pre-release) 피처만 사용해 누수를 피하고, 시간 기준 분할 + log-loss/macro-F1로 평가. 90%+ 정확도 수치는 대부분 누수(post-pitch)임을 유의.
- **KBO 데이터**: pitch-level 공개 데이터가 희소 → box-score/게임로그 기반 tabular(+뉴스 감성 융합)로 시작하고, TabPFN을 소표본 도전자로 병행. KBO 다음 구종 예측은 2026 문헌 공백이라 novelty 훅이 큼.

## 2026 최신 하이라이트

- **Neural Sabermetrics with World Model (arXiv 2602.07030, 2026.2)** — MLB 트래킹 10년+, 7M+ 투구 시퀀스(~3B 토큰)로 LLM을 자기회귀 월드모델로 사전학습. 다음 구종 ~64%, 타자 스윙 결정 78%(포스트시즌 OOD 평가). 다음 구종/구종 믹스(투구율) 예측에 가장 직접적인 2026 청사진.
- **Interpretable Pre-Release Pitch Type Anticipation from Broadcast 3D Kinematics (arXiv 2603.04874, 2026.3)** — 방송 영상 3D 포즈 → 229개 생체역학 피처 → 그래디언트 부스팅. 공 릴리스 *이전* 신체 역학만으로 8구종 80.4%(119,561 투구). 손목/몸통 기울기가 상위 신호, ~80%가 역학-only 상한.
- **Structure of Pitch-Pattern Motifs in MLB (arXiv 2601.11904, 2026.1)** — 12.4M 투구를 언어처럼 분석(엔트로피/Zipf/Heaps). **부정 결과**: 시퀀스 다양성은 ERA/승수와 뚜렷한 상관 없음 → "시퀀스 구조만으로 성적 예측" 가정에 대한 경고, 피처 설계용.
- **CNN-LSTM Player-Specific Pitch Type Prediction from Video (MDPI ASI, 2026)** — 방송 영상 기반 구종 인식을 시계열 분류로 프레이밍(Scherzer 2015–2020, 5구종). 동료심사 통과 2026 비전+시퀀스 파이프라인.
- **KFYO: Fusing vision and biomechanics for pitch evaluation (Heliyon, 2026)** — 투수/타자 포즈로 스트라이크존 추정해 스트라이크/볼 판정. YOLOv12 검출+칼만 추적+포즈. MLB 2026 ABS 도입 흐름과 맞물린 저비용 대안.
- **Scalable Injury-Risk Screening from Broadcast Video (arXiv 2603.04864, 2026)** — 방송 영상 생체역학 18지표로 부상 위험(Tommy John AUC 0.811, 주요 팔 부상 AUC 0.825). 시계열 예측용 피로/워크로드 공변량 소스로 유용.
- **Cross-individual generalizability of ball-speed models (arXiv 2605.05487, 2026.5)** — 투수 내 R²≈0.91이 투수 간 R²≈0.38로 붕괴(leave-one-subject-out, 50 투수). 신규 투수 일반화가 어렵다는 핵심 경고 → 사전학습/개인화 필요.
- **(2026 KBO)** ML-Based Classification of Team Playoff Advancement in KBO (Applied Sciences, 2025–2026) — 최근 몇 안 되는 KBO 투구지표 ML(10시즌; LogReg AUC 0.804, ERA·WHIP 지배). 단, 팀/시즌 단위이며 pitch-level 아님.

> 정직한 한계: 2026 야구 특화 논문은 대부분 **비전/생체역학(구종 타입·부상)** 에 몰려 있고, **순수 tabular 투구 결과** 및 **rate-stat 시계열 회귀(투구율)** 를 직접 다루는 2026 동료심사 논문은 희소합니다. 이 두 축은 2025 문헌(TFT, MDPI XGBoost) + 2025–2026 범용 tabular/시계열 SOTA를 전이하는 것이 현실적 최선입니다.

## 파이프라인·모델 비교표

| 접근법 | 대표 모델/논문(연도) | 적합한 투구율 정의 | 데이터 요구 | 장점 | 단점 | 성숙도 |
|---|---|---|---|---|---|---|
| LLM 월드모델(자기회귀 이벤트) | Neural Sabermetrics (2026) | 다음 구종/구종 믹스 비율 | 대규모 play-by-play(수백만 투구, ~3B 토큰) | 구종·스윙·결과 단일 모델 통합, 가장 novel, OOD 평가 | 미동료심사 프리프린트, 대규모 데이터·컴퓨트 필요, 블랙박스 | 신흥(프리프린트) |
| Attention/Stacked LSTM 시퀀스 | Decide the Next Pitch (2022) | 다음 구종 비율 | 투수별 다게임 pitch 시퀀스(pre-release 컨텍스트) | 재현성·해석 용이, 강한 표준 베이스라인 76.7% | 구종 정확도 상한(~40–50% 다중분류), 투수별 데이터 필요 | 성숙 |
| 그래디언트 부스팅(XGBoost/LGBM/CatBoost) | MDPI Applied Sci 15:7081 (2025) | 투구 결과율(스트라이크/볼/타격결과), 구종 tabular | pitch-level 궤적+카운트/컨텍스트(>~10k행) | 실무 SOTA 0.89–0.93, 빠름, SHAP 해석, 불균형 가중 | 시퀀스 정보는 lag 피처로만, 누수 위험(post-pitch), 결과는 고분산 | 성숙(실무 표준) |
| 기대값 회귀(xRV/xwOBA/ΔRV) | PitchProfiler/mlbpitchprofiler (2025–2026, 비동료심사) | 투구 품질/결과율(연속 타깃) | pitch-level 궤적+컨텍스트 | 클래스보다 안정·stabilize된 신호, Stuff+식 등급 | 단일 표준 정확도 없음, 산업 문서 위주 | 산업 성숙/학술 얇음 |
| Temporal Fusion Transformer | Pitcher Performance by TFT (2025) | 시즌/경기별 성적률(ERA/WHIP) | 경기 단위 시계열 + 정적/시변 공변량 | rate-stat 예측에 최적, 변수중요도 해석, RNN·투영시스템 상회 | 초록에 수치 미공개, rate-stat 회귀 벤치마크 희소, 짧은 시퀀스엔 부적합 | 신흥/제한 벤치 |
| Tabular 파운데이션 모델(TabPFN v2) | TabPFN v2 Nature (2025); 축구 xG (2026) | 소표본 구종/결과율, KBO 소규모 | 소·중형 표(≤10k행, ~500피처) | 학습 없이 in-context, 소표본 강세, 앙상블 멤버 | 대규모(수백만)엔 GBDT 우위, 스포츠서 GBDT와 거의 동률 | 신흥, 야구 벤치 없음 |
| 3D 포즈/생체역학 → 부스팅 | Bright et al. 2603.04874 (2026) | 구종 비율(릴리스 전 tell) | 방송 영상 + 포즈 추출 인프라 | 릴리스 전 예측 80.4%, 누수 없음, 해석 가능 | 무거운 CV 스택, ~80% 상한, 그립 변형엔 ball-flight 필요 | 신흥(2026 최전선) |
| CNN-LSTM / ST-GCN 비전 | MDPI ASI (2026); KBO ST-GCN | 구종 비율(영상 입력) | 방송 영상 프레임 | 비전+시퀀스 결합, 투수별 파인튜닝 | 투수별 특화, 정확도 중간(KBO 6클래스 63.4%) | 신흥 |
| 시계열 파운데이션 모델(Chronos/TimesFM) | 범용(2026), 야구 미발표 | 성적률 시계열(속도/사용률/ERA) | 시계열(제로샷 가능) | zero-shot 예측, 야구 미발표라 novelty | 야구 벤치마크 부재(검증 안 됨) | 범용 성숙/야구 공백 |
| 모티프/엔트로피 분석 | Pitch-Pattern Motifs (2026) | (예측 아님) 예측가능성 진단 | 대규모 시퀀스 | 투수별 예측가능성 prior, 피처 엔지니어링 | 예측기 아님, 다양성↔성적 상관 약함 | 진단 도구 |

## 상황별 추천

- **소량·콜드스타트(투수 1명/1시즌, 신규 투수)**: TabPFN v2를 1차 도전자로(학습 비용 ~0, 소표본 강세) + XGBoost 베이스라인, 둘을 스태킹. 시퀀스 모델(LSTM/Transformer)은 시퀀스가 짧아 이득 없음. 단 cross-pitcher 일반화 붕괴(R² 0.91→0.38, 2026)를 감안해 투수 임베딩/계층 구조를 고려.
- **대량 Statcast(수백만 투구)**: 그래디언트 부스팅(XGBoost/LightGBM/CatBoost)이 여전히 우위(파운데이션 모델보다 스케일 유리). 결과율은 xRV/xwOBA 회귀 타깃, 구종은 시퀀싱·카운트 피처. TabPFN은 대규모엔 부적합.
- **시퀀스 예측(다음 구종/at-bat 내 흐름)**: attention-LSTM(2022)을 정직한 베이스라인으로, 최신·novel 지향이면 LLM 월드모델(Neural Sabermetrics, 2026) 또는 소형 트랜스포머. 단 "시퀀스 순서만으로 성적 예측" 금지(모티프 2026 부정 결과) — 워크로드·상대·카운트 공변량 필수.
- **성적률 시계열(게임별 ERA/WHIP, 사용률 추이)**: TFT(2025)에 rest days·누적 투구수·직전 등판 이후 일수 등 피로/워크로드 공변량 투입. 미개척 novelty로 Chronos/TimesFM zero-shot을 XGBoost/Marcel과 벤치마크하는 것도 방어 가능한 새 기여.
- **KBO 데이터**: pitch-level Statcast 등가 공개 데이터가 사실상 부재 → box-score/게임로그 tabular(XGBoost) + 선택적으로 KoBERT 뉴스 감성 융합(0.65→0.74 사례). KBO 다음 구종 예측은 2026 문헌 공백이라 강한 novelty 훅. MLB에서 사전학습 후 전이/피처 엔지니어링 전략 권장.
- **영상만 있는 경우**: 2026 비전/포즈 파이프라인(Bright 2603.04874, KFYO)이 유일한 방향. 단 CV 인프라 부담·~80% 상한을 감수할 때만.

## 데이터·구현 노트

- **데이터 소스**: MLB Statcast(투구당 ~118컬럼)를 `pybaseball`의 `statcast(start,end)`로 Baseball Savant에서 수집(1일 단위 자동 청킹). KBO는 공개 pitch-level tracking이 없어 게임로그/박스스코어 + 뉴스(Naver) 감성/임베딩에 의존.
- **누수(leakage) 경고 — 최우선**: 구종/투구율 *예측*은 **사전(pre-release) 피처만** 사용 — 카운트, 점수, 주자 상태, 직전 구종, 투수/타자 좌우, shrunk prior 경향. **post-release ball-flight(구속·스핀·무브먼트)는 절대 사용 금지**. 화제의 "93.3% XGBoost(17구종, macro-F1 0.873)"는 GitHub 프로젝트로 post-pitch 스핀/릴리스 스피드를 써서 사실상 *분류*이지 예측이 아님 → 누수 상한이지 forecast 스킬 아님. 정직한 사전 다중분류 정확도는 ~40–50%, 이진 fastball/non-fastball ~63–78%.
- **평가**: 반드시 **시간 기준 분할**(예: 2024 중반까지 학습, 이후 검증) — 랜덤 분할 금지. 구종/결과는 심한 클래스 불균형이므로 **log-loss + macro-F1**을 함께 보고(단순 accuracy는 희소 구종 실패를 숨김). rate-stat 회귀는 stabilization/stickiness(교차연도 상관)와 Marcel/투영시스템 대비 벤치마크.
- **타깃 설계**: "투구율"이 결과율/성적률이면 하드 클래스보다 **기대값·비율 회귀 + 보정된 불확실성**(2026 distributional tabular FM, arXiv 2603.08206)이 더 예측적·안정적. 비율 타깃엔 확률 캘리브레이션 유의.
- **구현 매핑(리포지토리 노트북 연계)**: 코스의 `00_Hands_on_Tabular_ML.ipynb` 패러다임(로지스틱 → XGBoost → TabM → LLM 직렬화 → TabPFN)이 그대로 이식됨 — Statcast 행을 자연어로 직렬화해 소형 LLM에 통과시키는 Step 5는 LLM 스코어링(arXiv 2604.14321) 패턴과 동형. GBDT + TabPFN + linear 스태킹을 production 레시피로 권장.
- **모델 선택 요약**: pitcher-내 성능은 좋지만 pitcher-간 전이는 무너짐(2026) → 글로벌 모델 + 투수 임베딩/계층 베이지안(matchup 모델, 2025)로 보강.

## ⚠️ 검증 경고

- **"a" (2026)** — "Pitch velocity, biomechanics & injury-risk prediction" 앵글에 포함된 이 항목은 verify-flag에서 **likely-fabricated**로 표시됨: 제목이 알파벳 한 글자 'a', 초록·방법 모두 "test" 플레이스홀더로 실재하지 않는 항목. **미확인, 인용 금지.** 투구 속도/생체역학/부상 위험 주제에서 실제로 인용 가능한 관련 실물 논문은 2024 JSES/AJSM "Pitch-Tracking Metrics as a Predictor of Future Shoulder and Elbow Injuries in MLB Pitchers"이며, 2026 대안으로는 본 도시에의 Scalable Injury-Risk Screening(arXiv 2603.04864, 2026)을 사용할 것.
- **일반 인용 주의**: 도시에 내 arXiv ID(2602.07030, 2603.04874, 2601.11904, 2605.05487, 2603.04864 등)는 검색 반환·초록 확인으로 교차검증되었으나, **2602.07030(Neural Sabermetrics)와 2603.04874, KFYO 등 2026 최신 결과는 미동료심사 프리프린트/신규 저널**이므로 최종 인용 전 게재본·수치를 재확인할 것.
- **비논문 출처**: "mlb-pitch-prediction"(GitHub, 2025)과 PitchProfiler/mlbpitchprofiler(2025–2026)는 논문이 아닌 프로젝트/산업 문서 — 엔지니어링 템플릿·누수 경계 사례로만 사용하고 학술 근거로 인용하지 말 것.