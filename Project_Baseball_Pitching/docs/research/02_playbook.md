# LG Aimers x LG Twins 제구 성공 확률 예측 — 우승 플레이북

> 목표: 투수의 **다음 투구가 "제구 성공(command-success)"일 확률**을 투구 단위로 예측하는 이진 확률 문제. Train 2019–2024 / Hidden Test 2025 (시간분할). 아래는 dossier 근거만으로 구성한 실전 우승 전략입니다.

---

## TL;DR 우승 전략

1. **모델은 GBDT 3종(LightGBM+XGBoost+CatBoost) + TabPFN(v2/2.5) + TabM을 로지스틱 메타로 스태킹**하는 다패밀리 앙상블이 정석 — 시간분할·대용량 로그에서 가장 안정적(TabReD 2025).
2. **점수의 8할은 피처 엔지니어링**: 누수 안전 as-of 누적 제구율 + xCTRL식 "의도 vs 실행" 미스거리 피처 + 릴리스 일관성(트랙맨) + 카운트/레버리지 상호작용.
3. **핵심 창의 피처 = xCTRL(arXiv 2508.19184, 2025)**: 투수별 GMM으로 "의도한 타깃"을 추정하고 실제-의도 거리로 제구를 정량화. 리그평균 Location+보다 연도간 안정성(0.65 vs 0.48) 우수.
4. **검증은 무조건 시간순 forward-chaining(2019–22→23, →24)**, GAME 단위 그룹분할. 랜덤 KFold는 순위를 뒤집는다(TabReD·arXiv 2603.20315).
5. **제출물은 "확률"** → SMOTE 금지(캘리브레이션 악화, arXiv 2603.00208), 최신 시즌(2024)에서 isotonic/Platt 캘리브레이션, LogLoss/Brier로 최적화.
6. **가산점**: 상황·레버리지 조건부 "expected command", 릴리스 일관성 vs 조절력, SHAP 해석 레이어로 심사위원 어필.
7. **경고**: 라벨 정의(제구 성공)와 평가지표는 반드시 주최측/별첨 확인. 아래 가정(미스거리 ~6인치 온타깃)은 확정 아님.

---

## 추천 파이프라인 (아키텍처)

```
[Raw 투구 로그 2019-2024]
   │  ① 시간순 정렬 (pitcher → game → at-bat → pitch)
   ▼
[누수 안전 피처 생성]  ─ 모든 통계는 strict shift(1) 후 rolling/expanding
   ├ as-of 누적 제구율 (career/season/last-N/game/AB, EWMA, streak)
   ├ xCTRL식 의도-타깃 GMM 피처 (미스거리/posterior/entropy)
   ├ 트랙맨 릴리스 일관성 (release_x/z·extension·arm-slot rolling std, 현재편차)
   ├ 카운트·상황·매치업 상호작용 (count×pitch_type×runners, LI×count)
   └ 익명 ID 인코딩 (CatBoost ordered CTR / OOF target enc + 베이즈 수축)
   ▼
[베이스 모델 (다패밀리)]
   ├ LightGBM · XGBoost · CatBoost  (주력, 시간분할 최강)
   ├ TabPFN v2 / 2.5  (튜닝없는 캘리브레이션 좋은 파운데이션모델, ≤50k 서브샘플)
   ├ TabM (MLP-PLR 임베딩)  (딥 멤버, 시간분할서 GBDT급)
   └ (선택) GRU/LSTM-attention or 소형 causal Transformer  (아웃팅 내 시퀀스 브랜치)
   ▼
[스태킹]  OOF 확률 → 로지스틱/Ridge 메타러너 (2024 홀드아웃서 가중치 최적화)
   ▼
[캘리브레이션]  최신 시즌(2024)서 isotonic(데이터충분) / Platt·beta(희소)
   ▼
[제출]  2025에 적용, LogLoss/Brier 최적화 · reliability diagram 점검
```

**왜 이 선택인가**
- **GBDT 주력**: 시간분할·10만행+ 로그에서 XGB/LGBM/CatBoost가 평균순위 top-3(TabReD 2.9/3.1/3.4), 대부분의 딥모델은 시간분할서 무너짐.
- **다패밀리 블렌딩**: TabPFN·TabM은 GBDT와 **비상관(decorrelated)**이라 앙상블 이득의 원천. Kaggle 2025 우승 패턴(XGB/LGBM/CatBoost/TabPFN/NN 스택).
- **캘리브레이션이 곧 제출물**: 확률을 로그로스/Brier로 채점하므로 raw score를 반드시 보정. 로지스틱 메타러너 자체가 Platt 역할.
- **시퀀스는 대체가 아닌 증강**: 2026 motif 연구(arXiv 2601.11904)가 "시퀀스 문법 단독은 성능 예측력 제한적"이라 경고 → 시퀀스는 보조 피처/스택으로만.

---

## 모델 비교표

| 모델/접근 | 대표(연도) | 이 과제 적합성 | 장점 | 단점 | 우선순위 |
|---|---|---|---|---|---|
| **GBDT 3종** LightGBM/XGBoost/CatBoost | TabReD (2025) | ★★★★★ 시간분할·대용량 최강 | 안정적, 빠름, CatBoost가 익명 고cardinality ID를 ordered CTR로 native 처리, 로그로스 native | raw 확률은 캘리 필요, 딥 상호작용 수동 | **1순위 (주력)** |
| **TabPFN v2 / 2.5** | Prior Labs / TabPFN-2.5 (2025) | ★★★★ 서브샘플하면 강력 | 튜닝 없이 1-forward, out-of-box 캘리 우수, GBDT와 비상관 | 행/피처 상한(≤50k/2k), 서브샘플 필요 | **2순위 (블렌드)** |
| **TabM (MLP-PLR)** | TabM ICLR (2025) | ★★★★ 시간분할서 GBDT급 | 파라미터효율 MLP앙상블, 연속 트랙맨 피처에 PLR 임베딩, 엔티티임베딩 결합 가능 | 튜닝·학습비용, 단독으론 GBDT 미만 | **3순위 (딥 멤버)** |
| **TabICLv2 / TabPFN-3** | (2026) | ★★★ 확장형 파운데이션 | 더 큰 스케일, 확률품질 좋음 | 신규·검증 적음, 인프라 부담 | 선택(여유시) |
| **시퀀스 모델** GRU/LSTM-attn, causal Transformer | Attn-LSTM(2022), Sloan Transformer(2024?) | ★★★ 보조 증강용 | 아웃팅 내 피로·모멘텀 포착, CLS 상태를 스택 피처로 | 문법 단독 예측력 약함(motif 2026 경고), 데이터·튜닝 비용 | 보조(스태킹) |
| 순수 딥테이블 (FT-Transformer/SAINT/TabR) | — | ★★ 시간분할서 열세 | 표현력 | TabReD: 시간분할서 GBDT/MLP-PLR에 밀림 | 비권장 |
| 리샘플링(SMOTE/RUS/ROS) | — | ✗ 확률과제서 해로움 | (판별엔 중립) | Brier/로그로스 악화(arXiv 2603.00208) | **금지** |

---

## 피처 엔지니어링 플랜

> 원칙: 모든 히스토리·인코딩 통계는 **현재 투구 이전 데이터만**으로 계산(as-of). group별 `shift(1)` 후 `rolling()/expanding()` — 이 규율이 프라이빗 리더보드의 1차 차별화 요소.

### 1) 게임 상태 / 중요도 (WE·LI)
- 카운트를 **12단계 범주형**으로 인코딩(정수 2개 아님). ahead/behind/even, two-strike flag, first-pitch flag.
- outs, runners(주자상태), inning, score diff.
- **Leverage Index·Win Expectancy는 그대로 사용**(pre-pitch 게임상태의 함수라 누수 안전). 상호작용: `LI×count`, `LI×투수career제구`, high-leverage flag, WE-swing.
- 가설: 고레버리지에서 제구가 흔들리는가 → 트리가 `LI×count`로 분기하도록.

### 2) as-of 누적 히스토리 (지배적 신호)
- 투수 career/season **제구성공률 to-date**, last-N(N=5,10,25) rolling 제구율, 현재 게임/현재 타석 제구율.
- 카운트별·구종별·이닝/레버리지별 제구율(count-specific rate).
- 연속 성공/실패 streak, 최근 미스의 **EWMA(hot/cold 상태)**.
- 피로 프록시: 아웃팅 내 누적 투구수, 이닝, times-through-order, 휴식일, 시즌 누적 워크로드.
- **희소 bin은 empirical-Bayes 수축**: pitcher×count×pitch_type 셀이 희박 → 구종/리그 prior로 수축(xCTRL도 count-specific 분포를 count-agnostic prior로 수축, ≥250투구 요구). 2025 익명·저경력 투수 일반화에 필수.

### 3) 트랙맨 로그 (릴리스·무브먼트 물리)
- 유발 수직/수평 무브먼트, 스핀효율.
- **릴리스 일관성 = 직접적 제구 신호**: 투수별 `release_x/z`, extension, arm-slot 각의 rolling std/EWMA + **현재 투구의 최근 평균 대비 편차**. 낮은 변동성(특히 foot-plant 시점)이 좋은 제구(Driveline 2026).
- 속도·스핀의 시즌평균 대비 delta, 1차/2차 delta(직전 투구 대비).
- (선택) 릴리스 지표로 "예상 로케이션"을 예측하는 모델의 **잔차(예측-실제)** = 과/부족 제구 피처(Springer 2025).

### 4) xCTRL식 의도-타깃 피처 (최고 레버리지 창의 피처)
- 2019–2024로 **pitcher × pitch_type × batter_hand × count** 별 과거 로케이션에 **GMM(EM, K는 CV로 1–6)** 적합 → 각 컴포넌트 = 습관적 타깃존.
- 새 투구마다 파생: 가장 가까운 타깃까지 거리/Mahalanobis, **"온타깃" posterior 확률**, 타깃 개수·entropy, 그 맥락의 **expected xCTRL(미스거리)**.
- 저경력 투수는 구종/리그 prior로 EB 수축(2025 테스트가 익명·저히스토리일 수 있어 결정적).
- 라벨을 예측 미스거리의 **캘리브레이션된 로지스틱 함수**로 매핑 → "expected command"를 곧바로 확률로.

### 익명 ID 처리
- pitcher/batter/team ID를 **역사적 제구율로 인코딩하되 누수 안전하게**: CatBoost **ordered CTR**(랜덤순열서 앞쪽 행만) 또는 **K-fold OOF target encoding + 베이즈 수축**(글로벌/투수 prior로).
- 상호작용 인코딩: `pitcher_id×count`, `pitcher_id×pitch_type`, `pitcher_id×batter_hand`.
- **딥 멤버엔 엔티티 임베딩**(dim ≈ min(50, card^0.5)) — 원핫보다 희소 ID 일반화·과적합 억제, 출력 벡터를 GBDT 피처로 export 가능(Entity Embeddings 1604.06737; Universal Embeddings 2025).
- 2025 테스트: 인코딩을 2019–2024로 **freeze 후 시간순으로 on-the-fly 갱신**.

---

## 검증 & 캘리브레이션

### 시간기반 CV (2025 모사)
- **Forward-chaining / expanding window**: train 2019–22 → val 2023, train 2019–23 → val 2024. 2019–2024→2025 구조를 그대로 모사.
- **GAME 단위 그룹분할**(투구 단위 X) — 같은 경기·투수 누수 차단, GroupKFold by pitcher로 2025 일반화 갭 정직 추정.
- **랜덤 KFold 절대 금지**: 미래정보 누수로 모델 순위 왜곡·GBDT 마진 과장. TabReD·arXiv 2603.20315(rolling-origin이 순위 뒤집음)·arXiv 2512.06932(분할 전 시퀀스 생성 누수) 근거.
- 각 fold 내에서 인코딩·rolling 통계를 **그 fold의 과거만으로 재계산**.
- 시즌간 분포 드리프트(속도/스핀) 체크를 step 1으로(NVIDIA Grandmasters 2025).

### 지표 (라벨/지표 미확정 — 가정)
- **주력: LogLoss**(확률 proper score). 보조: **Brier**, **class-stratified Brier**(희소 클래스가 오차 대부분 은폐), AUC(랭킹시).
- reliability diagram + 캘리브레이션 slope≈1 모니터.
- **ECE는 진단용으로만**(binning·불연속 편향, arXiv 2406.04068). 블렌드 가중·early stopping은 실제 대회지표로 최적화.

### 캘리브레이션
- **리샘플링 금지**: SMOTE/RUS/ROS는 판별력 그대로면서 Brier/로그로스 악화(arXiv 2603.00208, 10데이터셋/605k). 자연 비율 + native 로그로스, 필요시 `scale_pos_weight`는 gradient 강조용만.
- 부득이 undersample시 **prior/intercept 보정**(log sampling-odds로 절편 조정, arXiv 2410.18144).
- **최신 시즌(2024)을 캘리 홀드아웃**으로 예약 → isotonic(소수 이벤트 충분시, GBDT 비선형 매핑에 최적) / Platt·beta(희소시) / temperature(TabM 등 net).
- 미래 드리프트(2025 신규투수·구속트렌드·ABS시대) 대응해 **가장 최근 데이터로 재캘리**.
- 블렌딩시 **로지스틱 메타러너가 곧 Platt 캘리브레이터** → 최종 블렌드에 isotonic 재보정.

---

## 💡 가산점: 새로운 제구력 평가 아이디어

심사위원을 감동시킬 "단순 로케이션 분류기가 아닌, 의도와 실행의 간극을 모델링" 서사.

1. **Expected Command (xCTRL 핵심)**: "공이 어디 갔나"가 아니라 **"의도한 곳과 실제의 거리"**로 제구를 정의. 투수별 GMM+베이즈 posterior 미스거리를 as-of 누적으로 산출 → 리그평균 Location+보다 연도간 안정(0.65 vs 0.48), FIP/IP/BB9 예측. 이 프레임이 창의성 점수의 백본.

2. **상황·레버리지 조건부 제구 안정성**: LI·WE·카운트로 제구율을 조건화하되 **hierarchical/empirical-Bayes로 강하게 수축**. clutch-command는 대부분 노이즈(FanGraphs 합의: "과거는 설명하나 미래 예측력 미미") → 큰 LI 상호작용보다 작고 안정적인 상호작용이 이긴다는 정직한 서사.

3. **일관성 vs 조절력(consistency-vs-adjustability)**: 제구를 순수 반복성이 아니라 **"통제된 변동성"**으로 재해석. foot-plant 시점 릴리스 변동은 낮게, 이후 국면은 전략적 조절 — 아웃팅 내 릴리스 분산 + 아웃팅 투구수를 피로/조절 프록시로(Driveline "Dampening Scores", 2026).

4. **Intent-residual(과/부족 제구)**: 릴리스 지표 기반 "stuff/location" 모델의 예측 로케이션 대비 실제 잔차 = 의도-실행 갭. xPV/xRV expected-vs-actual 전통, 원시 로케이션보다 정보량·안정성 우수.

5. **복합 command 스코어**: 단일 스칼라 대신 GMM-미스 + 잔차 + 일관성의 **상보적 서브메트릭 앙상블**(농구 자유투 앙상블 2512.08824의 설계패턴 차용).

6. **SHAP 해석 레이어**: 각 예측을 카운트·LI·직전미스·릴리스일관성으로 귀속 → 해석성 가산점 + 익명 데이터 sanity check + 피처선택. 2026 해석형 피칭 ML(arXiv 2603.04874, upper 64.9% vs lower 35.1%) 방법론 차용.

---

## 2026 최신 근거

**진짜 2026 (검증된 일반 방법론 — 이 과제에 직결):**
- **TabICLv2** (2026, arXiv 2602.11139) — 확장형 오픈 tabular 파운데이션 모델, 서브샘플 블렌드 멤버.
- **TabPFN-3 Technical Report** (2026, arXiv 2605.13986) — TabPFN 계열 최신, 단일-forward 캘리 분류기 SOTA.
- **Distributional Regression with Tabular Foundation Models (proper scoring rules)** (2026, arXiv 2603.08206) — 확률 품질(proper score·캘리)을 직접 평가 → 확률채점 과제에 직결.
- **Rolling-Origin Validation Reverses Model Rankings** (2026, arXiv 2603.20315) — 시간순 검증이 모델 순위를 뒤집음, 엄격 시간분할 정당화.
- **On Probability Estimation for Unbalanced Classification** (2026, Springer 10.1007/s42519-026-00572-5) — 불균형서 리샘플링보다 prior-correction/캘리.
- **Tipping the Balance (class imbalance correction)** (2026, arXiv 2603.00208) — 리샘플링이 Brier 악화 → 확률과제서 리밸런싱 금지 근거.
- **Lag and Rolling Features 가이드** (2026, AnalyticsVidhya) — shift(1) 후 rolling, 누수 안전 as-of 레시피.

**2026 야구 (인접/보조):**
- **The Interaction of Biomechanics and Command** (2026, Driveline 블로그) — 릴리스 일관성 vs 조절력, Dampening Scores. **주의: 산업 블로그, 비peer-review.**
- **Interpretable Pre-Release Pitch Type Anticipation from 3D Kinematics** (2026, arXiv 2603.04874) — 제구 아닌 구종이지만 해석/피처귀속 레이어 템플릿.
- **Structure of Pitch-Pattern Motifs in MLB** (2026, arXiv 2601.11904) — ~12.4M 투구, 시퀀스 문법은 성능과 연관 제한적 → **시퀀스를 주신호가 아닌 증강으로**라는 정직한 경고.

**핵심 야구 근거는 2025 (2026 command-확률 논문은 사실상 부재):**
- **xCTRL "Separating Intent from Execution"** (2025, arXiv 2508.19184) — **이 과제의 개념적 백본**. GMM+베이즈 의도추정, posterior-weighted 미스거리. 이 앵커를 2025로 정확히 인용할 것.
- **Context-Enhanced DL for Pitch Location** (2025, Springer 10.1007/s12283-025-00497-5) — 릴리스 지표→로케이션 예측, 잔차=command. 트랙맨 피처 청사진.
- **TabReD** (2025, arXiv 2406.19380) — 시간분할서 GBDT·MLP-PLR 최강, 랜덤분할이 순위 왜곡. 검증·모델선택 기초.
- **TabM** (2025, arXiv 2410.24210), **TabPFN-2.5** (2025, Prior Labs), **Kaggle 2025 스태킹 레시피**(Medium/NVIDIA Grandmasters).

**정직한 평가**: 2026 야구-제구-확률 전용 논문은 **존재하지 않음**. 엣지는 turnkey 솔루션이 아니라 **command 문헌(2025 xCTRL)이 지시하는 피처를, 검증된 2025–2026 일반 tabular 방법으로 모델링**하는 데 있음.

---

## ⚠️ 검증 경고 & 확인 필요

- **라벨·지표 미확정 (최우선 확인)**: "제구 성공"의 **정확한 정의**와 **평가지표**는 dossier에 없음 — **반드시 별첨/주최측 확인**. 본 플레이북은 (a) 라벨이 Statcast식 온타깃/미스거리(~6인치 이내, 리그평균 미스 ~11–12인치)에 대응한다는 **가정**, (b) 지표가 LogLoss/Brier(확률)라는 **가정** 위에 작성됨. 라벨이 다르면(예: 심판 콜존/헛스윙 유도) 피처·목적함수 재설계 필요.

- **인용 주의 (verify-flag)**: **"Transformer-Based Baseball Modeling for Pitch Outcome Prediction and Strategy Optimization" (2024, MIT Sloan, Declan Kneita)** — 논문 실재·제목/저자 일치하나 **연도 미확인**(사이트가 SSAC25/26 참조, 2025일 수 있음). 발표시 "연도 미확정"으로 표기 권장.

- **산업 블로그 주의**: Driveline "Interaction of Biomechanics and Command" (2026)는 **peer-review 아님**. 릴리스 일관성 피처의 동기로만 사용, 학술 근거로 과대인용 금지.

- **합성데이터 주의**: GitHub 벤치마크 "Baseball-Pitch-Sequence-Prediction" (2026, jman4162)는 **합성데이터** → 엔지니어링 템플릿으로만, 수치는 비권위적.

- **인접 논문 주의**: 2026 야구 arXiv(2603.04874 구종예측, 2603.04864 부상스크리닝)는 **command가 아닌 video/kinematics** 과제 — 해석 방법론만 전용, "제구 예측 근거"로 오인용 금지.

- **과대주장 금지**: "2026 next-pitch command-success 확률 예측 논문"은 존재하지 않음. 그런 문헌이 있다는 어떤 주장도 **조작으로 간주**.