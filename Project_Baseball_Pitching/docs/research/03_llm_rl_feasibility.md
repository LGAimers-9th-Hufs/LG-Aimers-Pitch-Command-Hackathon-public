# LLM + RL 결합 실효성 검증 — GBDT를 실제로 이기는가?

> LG 트윈스 해커톤('다음 투구 제구 성공 확률') 대상. 6개 결합기법을 2026 논문 근거로 판정 + 적대적 하이프 검증.

---

## 결론 한 줄
LLM+RL+ML 결합은 이 과제(대용량 Trackman 피치 로그 기반 제구 성공 확률을, LogLoss/Brier로 채점)의 **점수를 실제로 올린다는 근거가 없다** — GBDT(LightGBM/XGBoost/CatBoost) 스택을 확률 예측기로 유지하고, LLM은 오직 **오프라인 피처 아이디어 발굴**과 **의도(intent) 기반 라벨/피처 정의**에만 주변부로 쓰며, **RL은 (순차적 투구-시퀀싱 정책이 아닌 한) 통째로 제외**하라.

## 기법별 실효성 판정표

| 기법 | 실제 효과(근거) | GBDT 대비 | 이 과제 권장도(★1-5) | 리스크 | 쓴다면 어디에 |
|---|---|---|---|---|---|
| ① LLM-as-predictor (TabLLM/BoostLLM, 행 직렬화) | zero/few-shot 소데이터에서만 우세, 데이터 늘면 GBDT에 패배. TabLLM 자체가 "poor calibration·high variance" 보고, 직렬화가 구조적 규칙성 폐기 (TabLLM 2210.10723; Amazon survey 2402.17944) | **패배** (특히 대데이터·확률채점) | ★1 | 낮은 캘리브레이션 = LogLoss/Brier 직격, 고비용 | 안 씀 |
| ② LLM-as-feature-engineer (CAAFE/LLM-FE/OCTree) | LLM-FE 평균순위 1.42 vs XGB 3.95이나, 큰 이득은 소형/합성구조 데이터뿐, "대부분 0-2pp", GBDT가 여전히 예측기 (LLM-FE 2503.14434; Human-LLM FE 2601.21060 AUROC +1.3-1.5pp) | 예측기는 GBDT 유지, **한계적 보완** | ★2 | 타깃 누수(as-of/rolling 피처)·이득 종종 0 | 오프라인 1회성 피처 브레인스토밍 → 시간인지 CV로 검증 게이트 |
| ③ RL 기반 AutoML/피처선택/HPO | 랜덤서치가 RL NAS와 대등(Li&Talwalkar 1902.07638), 비RL OpenFE가 RL NFS/GRFG를 40/49서 압도(2211.12507); GBDT는 이미 정규화된 암묵적 피처선택. 어느 것도 proper score/캘리브레이션 최적화 안 함 | **패배 / 무의미** | ★1 | 불안정·고비용, LogLoss 개선 없음 | 안 씀 (HPO는 Optuna/랜덤서치로) |
| ④ Inverse RL / intent inference (pitch command) | 진짜 IRL은 야구 제구엔 부재. xCTRL(2508.19184)은 유용하나 **GMM/베이지안이지 IRL 아님**, GBDT와 대조 안 됨. IRL은 샘플 비효율·과적합. 어느 것도 proper score 대조 없음 | 예측기로는 **패러다임 불일치**; 의도추론 전처리는 조건부 유용 | ★2 (의도추론 피처화 한정 ★3) | IRL을 예측기로 오용, 라벨정의 없이 성공 정의 불가 | 라벨정의(의도 대비 거리)·의도조건 피처 → GBDT 투입 |
| ⑤ LLM/자기회귀 월드모델 (스포츠 시퀀스) | Neural Sabermetrics(2602.07030)는 next-pitch 정확도 64%/스윙 78%이나 **GBDT 대조 없음·정확도만(LogLoss/Brier 아님)**. 대데이터 태블러는 GBDT 우세(2502.17361) | 목표 지표서 **불일치/미입증** | ★1 (임베딩 피처추출 조건부 ★2) | as-of 누수, 손번역 이득이 lag/rolling로 이미 회수됨 | (선택) as-of 시퀀스 임베딩을 GBDT 추가컬럼으로 A/B |
| ⑥ RL-finetuned/추론 LLM (GRPO/RLHF) & 하이브리드 LLM+RL+ML | RLVP(WWW 2026)가 유일하게 확률보상 최적화하나 task-specific XGB를 "최대 55% 태스크"서만 상회=사실상 동률. PRPO는 정확도만·회귀/zero-shot 패배. LLM verbalized 확률은 체계적 miscalibration(2606.19509, 2509.15356) | 최선 **동률**, 10-100x 비용, 캘리브레이션 악화 | ★1 | proper score를 RL로 재유도=불안정, GBDT가 직접 목적함수로 최적화 가능 | 안 씀 (정적 확률예측); RL은 순차 정책 문제에만 |

## 실제로 도움되는 결합 (권장)
근거가 helps/conditional인 항목만:

- **LLM 오프라인 피처 아이디어 발굴 (②, 근거 helps/mixed)**: 스키마 + 야구 도메인 설명(카운트, leverage/WE 상태, 투수 피로/투구수, 직전 투구 시퀀싱, platoon, as-of rolling 제구율, Trackman 릴리스 일관성)을 LLM에 주고 후보 피처와 계산코드를 생성. **모든 후보를 기존 시간인지 CV(선택지표=LogLoss)로 게이트**, out-of-fold 개선 없는 것은 폐기. as-of/rolling 누수 감시 필수. 기대 효과 0-2pp, 종종 0 — 값싼 브레인스토밍 가속기이지 모델 업그레이드가 아님.
- **의도추론 기반 라벨/피처 (④ xCTRL 아이디어, conditional-yes)**: '제구 성공'은 의도 타깃 없이는 정의 불가 → GMM/베이지안 사후분포(카운트·포수 타깃·투수 성향 조건)로 타깃 추정, 성공=임계거리 이하로 원칙적 라벨 정의. 의도조건 피처(사후 타깃 좌표, 의도 불확실성/엔트로피, 구종·카운트별 as-of rolling xCTRL)를 **엄격 as-of**로 계산해 GBDT에 투입. 포수 글러브 타깃 트래킹이 있으면 관측 타깃 직접 사용(IRL 손댈 이유 거의 소멸).
- **(선택) 자기회귀 시퀀스 임베딩 (⑤, mixed·조건부)**: 피치별 로그로 사전학습한 시퀀스모델의 as-of 은닉상태 임베딩을 GBDT 추가컬럼으로. 단 **손제작 as-of history 피처를 이미 가진 더 강한 GBDT를 이겨야** 채택(축구 결과상 시퀀스 이득 대부분 lag/rolling로 회수됨).
- **비RL 자동 피처생성 OpenFE (③ 대안, helps)**: 자동화를 원하면 RL 대신 값싸고 강한 OpenFE 사용.
- **TabPFN v2/2.5를 별도 트랙·스택 멤버로**: LLM이 아닌 **합성-prior 트랜스포머**이며, cold-start(신인 <~50구)나 스택 다양성에서 검증 후 채택 — 이는 LLM 조사가 아니라 파운데이션모델 트랙.
- **후처리 캘리브레이션(isotonic/Platt) + reliability diagram**: proper score 과제의 핵심, RL 계열 어느 것도 다루지 않음.

## 하지 말 것 (근거 기반)
- **LLM을 확률 예측기로 사용 (①, 근거 hurts/mismatch)**: 데이터 규모에서 LogLoss·Brier 둘 다 GBDT에 패배, 출력이 miscalibrated.
- **RL 피처선택/파이프라인탐색/HPO (③, no)**: 랜덤서치·OpenFE가 RL을 이김, GBDT는 이미 정규화 피처선택. 시간낭비.
- **IRL을 예측기로 (④, mismatch)**: 보상/의도를 복원할 뿐 캘리브레이션 확률이 아님, GBDT 상회 근거 zero.
- **자기회귀 LLM 월드모델을 예측기로 (⑤, mismatch)**: LogLoss/Brier 대조 근거 없음, 2026 신호는 오히려 대데이터서 GBDT 우세.
- **GRPO/RL로 LLM 사후학습해 예측기로 (⑥, marginal)**: 최선 동률·10-100x 비용·캘리브레이션 악화.
- **볼트온 LLM+RL+ML 하이브리드 (⑥, no)**: proper-scoring 지도과제에서 RL은 GBDT가 직접 최적화하는 것을 불안정하게 재유도할 뿐. LLM은 주변부(피처, 희귀 게임상태 직렬화, SHAP 설명 번역)만.
- **hype-flag 걸린 근거로 결정하지 말 것**: BoostLLM(few-shot만), Dual-Agent RL(GBDT 대조 없이 다른 AutoFE만 이김), Multi-Component Reward FS(공정성 논문—예측 정확도와 카테고리 불일치).

## 추천 최종 설계
1. **점수 트랙(주력)**: LightGBM/XGBoost/CatBoost 스택 = 확률 예측기이자 캘리브레이션 소유자.
   - 목적함수를 직접 LogLoss(binary)로 설정 (Brier를 RL로 우회하지 말 것).
   - as-of 누수안전 피처: rolling 제구율, 카운트·leverage/WE 상태, 투수 피로/투구수, 직전 투구 시퀀싱, platoon, Trackman 릴리스 일관성.
   - count/leverage 상호작용 + 단조 제약, **투수/경기 단위 CV**로 누수 방지.
2. **라벨 정의 계층**: xCTRL식 GMM/베이지안 의도 사후분포로 '성공' 라벨 원칙화(관측 글러브 타깃 있으면 직접 사용).
3. **검증된 결합요소만 얹기(전부 CV 게이트 통과 시에만)**:
   - LLM 오프라인 피처 아이디어(②) → 시간인지 CV LogLoss 게이트.
   - 의도조건 피처(④) as-of.
   - (선택) OpenFE 자동 피처, 자기회귀 as-of 임베딩(강한 baseline 상회 시에만).
4. **모델 다양성**: TabPFN v2/2.5·TabM MLP를 스택 멤버/cold-start 애드온으로 head-to-head 검증 후 블렌딩.
5. **캘리브레이션 계층**: isotonic/Platt 후처리 + reliability diagram으로 LogLoss/Brier 검증.
6. **주변부 LLM**: 희귀 게임상태 직렬화, SHAP→스카우팅 설명 번역 (예측엔 미개입).
7. **RL**: 정적 확률예측에서 제외. 오직 투구-시퀀싱 정책·비용인지 측정획득 같은 **진짜 순차결정** 문제가 생기면 재검토.

## 2026 근거 요약
진짜 2026(없으면 2025) 근거:
- **aimultiple Tabular Models Benchmark (2026)**: 소데이터는 파운데이션모델(TabPFN 3) 선두, **대수치·고카디널리티 범주형에선 XGBoost/CatBoost 경쟁·승리**. TabPFN은 LLM 아님. *(단, 벤더 블로그 — 하단 경고 참조)*
- **Human-LLM Collaborative Feature Engineering, arXiv 2601.21060 (2026)**: AUROC 91.1→92.4-92.6%(~1.3-1.5pp). *(구체 수치 미검증)*
- **Neural Sabermetrics with World Model, arXiv 2602.07030 (2026)**: next-pitch 64%/스윙 78% 정확도, 신경 baseline 상회 — 그러나 **GBDT 대조·proper score 없음**.
- **RLVP, ACM WWW 2026 (DOI 10.1145/3774904.3792540)**: 확률보상 직접 최적화, 169-task 공동학습 후 task-specific XGB를 "최대 55% 태스크"서 상회 = 사실상 동률.
- **LLM Doesn't Know What It Doesn't Know, arXiv 2606.19509 (2026)**: 임상 태블러서 LLM verbalized confidence가 거의 상수·무정보, XGBoost 전반 승리(강한 부정 근거).
- **MO-IRL Global Intent Inference, arXiv 2603.07797 (2026)**: 궤적재구성 RMSE 27%↓ — off-domain(인간 팔뻗기), 확률과제 아님.
- **PipeBench, Nature Sci Reports 2026 (s41598-026-53722-x)**: 베이지안 최적화가 진화 AutoML의 ~97% 정확도를 훨씬 저비용에 도달, 랜덤서치 강력.
- **TabPFN-2.5, arXiv 2511.08667**: 2026으로 표기됐으나 **실제 2025년 11월 프리프린트**(하단 경고).

## ⚠️ 검증·하이프 경고
- **[가장 심각·조작] arXiv 2511.17733 미인용(reversed)**: '복잡 매치업 모델' 논문을 "복잡도 이득 미미·GBDT 경쟁적"으로 인용했으나, 실제 논문은 **더 복잡한 모델이 승률에서 유의미 이득(최대 ~1승/시즌)을 낸다고 결론**하며 GBDT/신경망을 전혀 다루지 않음(계층 베이지안 비교). 결론이 뒤집힌 인용 — 이 근거로 판단 금지.
- **벤더 블로그 오표기**: 'aimultiple Tabular Models Benchmark (2026)'은 **AIMultiple 벤더/마케팅 블로그**이지 동료심사 논문 아님. arXiv식 논문 제목으로 나열해 증거력 부풀림. 데이터 인용은 가능하나 '벤더 벤치마크'로 명시.
- **TabPFN-2.5 연도 오류 + 약한 baseline**: arXiv 2511 = 2025년 11월 제출(2026 개정)이므로 **2025 프리프린트**. 또 그 우위는 **default(미튜닝) XGBoost 대비** — RL 논문을 비판할 때와 동일한 약-baseline 함정.
- **BoostLLM(2605.06117)**: "wide range of shot counts서 XGBoost 매칭/상회"는 단일 산업팀(SinoPac) 프리프린트·독립 재현 없음·**few-shot 한정**. LLM이 GBDT를 이긴다는 논거로 쓰면 하이프.
- **Dual-Agent RL(2505.12628)**: 우위가 **다른 AutoFE(GRFG/NFS)에만**, 튜닝된 GBDT 대조 없음, downstream이 랜덤포레스트. 전형적 하이프 패턴.
- **OpenFE 과다-특정**: "40/49 데이터셋 +1.9% 평균"은 초록엔 없음(초록은 10개 벤치마크). 방향은 맞으나 정확 수치 미검증.
- **Multi-Component Reward FS(2510.09705)**: 공정성/편향 완화 논문 — 'RL 피처선택이 예측 개선?' 근거로 인용은 **카테고리 불일치**, 정확도 이득 없음.
- **PRPO(2510.17385)**: XGB/LightGBM/CatBoost/TabPFN "상회"는 **정확도만**, LogLoss/Brier/캘리브레이션 전무. cross-dataset zero-shot에선 XGB/TabPFN에 패배(32-shot 줘야 역전). 회귀 결과(TabPFN 0.1499 vs 0.1236)는 논문 표와 **불일치 의심**.
- **RLVP "55% 상회"**: '동률을 승리로 포장'. "up to" 표현 + 169-task 대규모 공동학습 요구가 깔끔한 우위 해석을 훼손.
- **xCTRL(2508.19184)**: "기존 제구 지표보다 예측력 높음"은 **동종 지표 내 비교**이지 GBDT/XGBoost 분류기 대조 아니며, **GMM 방법이지 IRL 아님** — IRL이 GBDT를 이긴다는 근거로 인용은 카테고리 오류.
- **Neural Sabermetrics**: baseline이 신경망 전용·지표가 정확도 — LogLoss/Brier로 GBDT를 이긴다는 head-to-head 없음.
- **Halawi RLVR(2505.17989) "o1 상회" 과장**: 초록은 정확도 매칭 + 캘리브레이션만 우위 주장. Brier 0.190 vs 0.202는 소폭, GBDT 대조 없고 인간(0.151) 하회.
- **Kaggle Grandmasters Playbook**: '2026' 표기이나 실제 **2025년 10월** 블로그. '850 실험/1위/LLM 에이전트 오케스트레이션'은 이 글이 아닌 별도 NVIDIA 기사 — 두 자료 혼동(conflation).
- **다수 head-to-head 수치**(ICU F1 59-65 vs 43, AUROC 0.847-0.894 vs GPT-4 0.60-0.63, TABULA-8B, pitch AUC ~0.968)는 2차 집계(clema.ai/aimultiple/MDPI)에서 나온 것으로 **정확 수치 미검증**(방향성만 신뢰).