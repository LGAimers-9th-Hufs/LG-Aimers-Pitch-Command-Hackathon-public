# Teacher→Student 증류 설계 (A100 학습 트랙)

> **확정 원칙**: 큰 모델 = **교사(teacher, 학습 도구)** · 제출물 = **내가 학습시킨 student 에이전트**. 추론/제출 시 교사 없음. student 학습은 **A100 GPU**로 진행.
> **불변 진실**: student가 무엇이든 최종 채점 **LogLoss에서 GBDT 앵커를 이겨야** 채택 (시간분할 CV 게이트). A100은 "무엇을 시도하나"(비용)를 풀지 "무엇이 이기나"(벤치)를 바꾸지 않음.
> 근거: `07_llm_to_train_model.md` (Pocket-FM 2026, Calibration-Trap 2026, ScoringBench 2026, LLM-FE 2025).

---

## 0. 증류의 천장 = 교사 품질 (설계의 출발점)
soft-label 증류로 얻는 student 상한은 **교사 성능**이다. 대규모·고차원·캘리브레이션 확률 regime에서 **교사(대형 LLM/TFM)가 GBDT를 못 이기면**, 그 예측을 모방하는 student도 못 이긴다(Pocket-FM 2026: 고차원 +0.001 AUC≈동전던지기). → **결론: "예측 증류"에만 베팅하지 말고, 교사에서 GBDT가 못 만드는 *지식/표현*을 뽑아 student를 강화하는 route를 우선한다.**

## 1. 무엇을 증류할 것인가 — 3가지 신호
| 신호 | 정의 | route | 위험 |
|---|---|---|---|
| **Soft-label(logit) KD** | 교사 확률/로짓을 student가 KL로 모방 (temperature T) | D2 | 교사 천장에 종속 |
| **Feature/Representation 증류** | 교사 임베딩·사전확률을 student 입력/중간층으로 | D1·D3 | 낮음(피처 주입) |
| **Data 증류** | 교사가 콜드스타트 슬라이스에 합성/증강 생성 | (보조) | base-rate 왜곡 |

## 2. 교사(teacher) 후보
- **대형 LLM** — 도메인 지식 → 콜드스타트 사전확률·의미 임베딩 (F13 route). 오프라인 1회 호출, 추론 시 0.
- **Tabular Foundation Model** (TabPFN v2/2.5, TabICLv2) — OOF soft-label 교사 (Pocket-FM식). ≤50k 서브뷰.
- **자체 대형 모델 (self-distillation / born-again)** — 같은 데이터의 큰 GBDT+DL 앙상블 → 작은/빠른 student. 교사=학생 도메인 동일이라 천장 문제 없음. **가장 안전한 A100 활용.**
- **시퀀스 자기지도 사전학습** (pitch2vec식: 투구열 masked prediction) → 임베딩 교사 → 시퀀스 student.

## 3. 학생(student) 후보 = 제출물 아키텍처
| ID | student | 증류 수용 | 비고 |
|---|---|---|---|
| **S1** | GBDT (증류 피처·soft-label 흡수) | feature/soft-label as 컬럼 | 최저 위험, A100 불필요 |
| **S2** | 신경망 tabular (TabM/RealMLP/FT-Transformer) | soft-label KD 직접 | A100 적합, 비상관 멤버 |
| **S3** | **시퀀스 에이전트** (transformer over pitch history) | 교사 임베딩+soft-label | "새 신호", A100 진가 |
| **S4** | 하이브리드 (시퀀스 인코더 + tabular head) | 혼합 | 상한 높음·복잡 |

## 4. 증류 손실 (A100)
```
student loss  L = α·BCE(y_true) + (1-α)·T²·KL( softmax(z_s/T) || softmax(z_t/T) )
  + β·MSE( feat_s, feat_t )        # feature distill (선택)
```
- 온도 **T**, 혼합 **α**, 피처 가중 **β** 는 시간분할 CV로 튜닝.
- **라벨 누수 방지 (핵심)**: 교사 soft-label은 반드시 **OOF/stratified**로 생성 (교사가 그 행을 학습에 안 본 상태). ICL/교사 라벨누수는 Pocket-FM도 경고.
- mixed precision(bf16) + A100로 S2/S3 대규모 학습.

## 5. A100 학습 파이프라인 (단계)
```
1. 데이터 → 누수안전 as-of 피처 (공통 레이어, sweep harness와 공유)
2. 교사 준비 (오프라인):
   - TFM/big-ensemble → OOF soft-label 캐시
   - LLM → 엔티티/컨텍스트별 사전확률·임베딩 캐시 (F13)
   - (S3) 시퀀스 자기지도 사전학습 → 임베딩
3. student 학습 (KD loss, bf16, A100)
4. 캘리브레이션 (temperature → isotonic/Platt, 최신 시즌 홀드아웃)
5. 게이트: 시간분할 CV LogLoss로 GBDT 앵커와 비교 → 이기면 채택/블렌드
```

## 6. 검증 & 함정 (07 근거, 비협상)
- **LogLoss로만 게이트** — AUC↔LogLoss 스피어만 ~0.15 (ScoringBench 2026). AUC 개선을 승리로 오인 금지.
- **Calibration-Trap** — 증류/스태킹이 캘리 악화 → 각 student 재캘리 + 블렌드 재캘리 (Ensembling-TFM 2026). **TFM 메타러너 스태킹 금지.**
- **교사 천장 체크** — 교사가 CV에서 GBDT 못 이기면 soft-label route(D2) 조기 kill, feature route(D1)로 전환.
- **콜드스타트 층화 평가** — 엔티티 히스토리 깊이로 CV 층화 (이득이 집계에서 희석됨).
- **의사라벨 금지** — 골드라벨 수백만 존재 → LLM 라벨은 노이즈·과신만.
- **재현성** — 다시드/다폴드 유지 확인 후에만 채택 (신경망·에이전트 고분산).

## 7. 권장 실행 순서 (증류 트랙, 각 단계 CV LogLoss 게이트)
| 단계 | 구성 | 위험/보상 | A100 |
|---|---|---|---|
| **D1** | S1(GBDT) + **F13**(LLM 콜드스타트 피처) + TFM soft-label 컬럼 | 최저위험·소보상 | 불필요 |
| **D2** | S2(TabM) + **soft-label KD**(TFM/big-ensemble 교사, OOF) | 중위험·다양성 멤버 | 권장 |
| **D3** | **S3(시퀀스 student)** + 교사 임베딩/soft-label | 고위험·"새 신호" 상한↑ | **핵심** |
| **D4** | **self-distillation(born-again)**: 큰 GBDT+DL 앙상블 → 작은 빠른 student | 중위험·배포효율 | 권장 |

## 8. 06 카탈로그 연결 (추가 ID)
- **S1~S4** = student 아키텍처 (제출물). **D1~D4** = 증류 학습 레시피.
- 교사 신호: F13(LLM 피처), M23(TFM 증류 학생), + 신규 **F15**(교사 OOF soft-label 컬럼), **M24**(self-distillation born-again student).
- 전부 sweep harness의 **동일 시간분할 CV 게이트**에 후보로 등록 → GBDT 앵커와 공정 비교.

## 9. 2026 근거
- **Pocket-FM** (2026, 2605.18654) — TFM→GBDT 증류, graceful fail, 지표는 AUC.
- **Ensembling TFMs: Calibration-Trap** (2026, 2605.18696) — 증류/스태킹이 최악 LogLoss.
- **ScoringBench** (2026, 2603.29928) — proper score로 게이트해야, AUC↔LogLoss 비상관.
- **LLM-FE** (2025, 2503.14434) — LLM 생성 피처가 XGB/MLP/TabPFN 개선 (F13 템플릿).
- **TabPFN-2.5** (2025, 2511.08667) — ≤50k행 우위, 대규모는 XGBoost.

---

### 한 줄 요약
**교사(대형 LLM/TFM/자체 큰 모델)에서 GBDT가 못 만드는 지식·표현·soft-label을 뽑아, A100으로 student(S1→S2→S3)를 D1→D4 순서로 학습·증류하고, 매 단계를 시간분할 CV LogLoss로 GBDT 앵커와 겨뤄 이길 때만 제출 후보로 채택한다. 교사는 학습 도구일 뿐, 제출물은 student다.**
