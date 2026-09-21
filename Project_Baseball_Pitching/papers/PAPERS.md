# 참고 논문 목록 (Papers)

3개 리포트(`01_landscape_survey`, `02_playbook`, `03_llm_rl_feasibility`)에서 인용한 논문.

- ✅ **arXiv 34편은 실제 존재 검증 완료**(arXiv API로 제목 대조) + PDF 다운로드 완료 → `papers/` 폴더에 있음.
- 🔒 **유료/비-arXiv 논문**은 다운로드 불가 → 아래 주소로 직접 받아서 이 폴더에 넣으면 됨.
- ⚠️ **비고 표시**: 검증 단계에서 잡힌 해석 주의사항(조작 ID는 없음, 전부 실재).

---

## A. 야구 — 제구·투수 예측 (핵심)

| 상태 | 논문 (연도) | 링크 | 비고 |
|---|---|---|---|
| ✅ PDF | **Separating Intent from Execution: A Probabilistic Approach to Pitch Location Accuracy** (xCTRL, 2025) | https://arxiv.org/abs/2508.19184 | **창의성 백본.** GMM/베이지안 의도추정(IRL 아님). 제구 지표 내 비교이지 GBDT 대조 아님 |
| ✅ PDF | Neural Sabermetrics with World Model (2026) | https://arxiv.org/abs/2602.07030 | next-pitch 64%. 정확도만 보고, LogLoss/GBDT 대조 없음 |
| ✅ PDF | Interpretable Pre-Release Pitch Type Anticipation from 3D Kinematics (2026) | https://arxiv.org/abs/2603.04874 | 누수 없는 예측(릴리스 전)의 모범, 해석 레이어 템플릿 |
| ✅ PDF | Structure of Pitch-Pattern Motifs in MLB (2026) | https://arxiv.org/abs/2601.11904 | **부정 결과**: 시퀀스 다양성 ≠ 성적 → 시퀀스는 보조로만 |
| ✅ PDF | Scalable Injury-Risk Screening in Baseball Pitching from Broadcast Video (2026) | https://arxiv.org/abs/2603.04864 | 워크로드/피로 공변량 소스 |
| ✅ PDF | Cross-individual Generalizability of Ball-Speed Prediction in Baseball (2026) | https://arxiv.org/abs/2605.05487 | **경고**: 투수 내 R²0.91 → 투수 간 0.38 붕괴 → 투수 임베딩 필요 |
| ✅ PDF | The Impacts of Increasingly Complex Matchup Models on Baseball Win Probability (2026) | https://arxiv.org/abs/2511.17733 | ⚠️ 03 리포트가 결론을 **반대로 인용**했던 논문(실제: 복잡모델이 이득). GBDT 비교 아닌 계층 베이지안 |
| 🔒 URL | Context-Enhanced DL for Pitch Location (Springer, 2025) | https://doi.org/10.1007/s12283-025-00497-5 | 릴리스 지표→로케이션, 잔차=command. Trackman 피처 청사진 |

## B. 정형데이터 모델 (파이프라인 주력)

| 상태 | 논문 (연도) | 링크 | 비고 |
|---|---|---|---|
| ✅ PDF | TabReD: Pitfalls in Tabular DL Benchmarks (2025) | https://arxiv.org/abs/2406.19380 | **검증 근거.** 시간분할서 GBDT/MLP-PLR 최강, 랜덤분할이 순위 왜곡 |
| ✅ PDF | TabM: Parameter-Efficient Ensembling (2025) | https://arxiv.org/abs/2410.24210 | 딥 스택 멤버, PLR 임베딩 |
| ✅ PDF | A Closer Look at TabPFN v2 (2025) | https://arxiv.org/abs/2502.17361 | TabPFN v2 분석·확장 |
| ✅ PDF | TabPFN-2.5 (2025.11 프리프린트) | https://arxiv.org/abs/2511.08667 | ⚠️ 우위는 미튜닝 XGBoost 대비(약-baseline 주의) |
| ✅ PDF | TabPFN-3 Technical Report (2026) | https://arxiv.org/abs/2605.13986 | 소데이터 파운데이션 SOTA |
| ✅ PDF | TabICLv2: Scalable Open Tabular Foundation Model (2026) | https://arxiv.org/abs/2602.11139 | 확장형 파운데이션 |
| ✅ PDF | Distributional Regression with Tabular Foundation Models (2026) | https://arxiv.org/abs/2603.08206 | **확률 품질(proper score) 직접 평가** → 확률채점 과제 직결 |
| 🔒 URL | TabPFN v2 — Accurate predictions on small data with a tabular foundation model (Nature, 2025) | https://www.nature.com/articles/s41586-024-08328-6 | 원 논문(Nature) |

## C. LLM for tabular

| 상태 | 논문 (연도) | 링크 | 비고 |
|---|---|---|---|
| ✅ PDF | TabLLM: Few-shot Classification of Tabular Data (2022) | https://arxiv.org/abs/2210.10723 | few-shot만 우세, 캘리 나쁨 |
| ✅ PDF | LLMs on Tabular Data: Prediction/Generation/Understanding — Survey (2024) | https://arxiv.org/abs/2402.17944 | 분류체계 |
| ✅ PDF | LLM-FE: Automated Feature Engineering as Evolutionary Optimizers (2025) | https://arxiv.org/abs/2503.14434 | ✅ 권장 사용처(피처 아이디어). 예측기는 GBDT |
| ✅ PDF | Human-LLM Collaborative Feature Engineering (2026) | https://arxiv.org/abs/2601.21060 | AUROC +1.3~1.5pp(수치 미검증) |
| ✅ PDF | Strengthening LLMs for Tabular Prediction with Structural Priors (PRPO, 2025) | https://arxiv.org/abs/2510.17385 | ⚠️ "상회"는 정확도만, LogLoss/캘리 전무 |
| ✅ PDF | BoostLLM: Boosting-inspired LLM Fine-tuning for Few-shot Tabular (2026) | https://arxiv.org/abs/2605.06117 | ⚠️ few-shot 한정·단일 산업팀·재현 없음 |
| ✅ PDF | LLM Predictive Scoring and Validation (2026) | https://arxiv.org/abs/2604.14321 | 텍스트→점수, 직렬화 스코어링 패턴 |

## D. RL / 자동 피처생성 / 의도추론

| 상태 | 논문 (연도) | 링크 | 비고 |
|---|---|---|---|
| ✅ PDF | OpenFE: Automated Feature Generation (2022) | https://arxiv.org/abs/2211.12507 | ✅ RL 대신 값싸고 강한 자동 피처생성 |
| ✅ PDF | Dual-Agent RL for Automated Feature Generation (2025) | https://arxiv.org/abs/2505.12628 | ⚠️ 우위가 다른 AutoFE에만, 튜닝 GBDT 대조 없음 |
| ✅ PDF | Multi-Component Reward + Policy Gradient for Feature Selection (2025) | https://arxiv.org/abs/2510.09705 | ⚠️ 공정성 논문 — 예측 개선 근거로는 카테고리 불일치 |
| ✅ PDF | Outcome-based RL to Predict the Future (RLVR, 2025) | https://arxiv.org/abs/2505.17989 | ⚠️ "o1 상회"는 정확도 매칭+캘리만, GBDT 대조 없음 |
| ✅ PDF | Toward Global Intent Inference by Inverse RL (2026) | https://arxiv.org/abs/2603.07797 | 진짜 IRL, 단 인간 팔동작·확률과제 아님 |

## E. 확률·캘리브레이션·검증·누수 (필수)

| 상태 | 논문 (연도) | 링크 | 비고 |
|---|---|---|---|
| ✅ PDF | Tipping the Balance: Class Imbalance Correction (2026) | https://arxiv.org/abs/2603.00208 | **SMOTE/리샘플링 금지 근거**(Brier 악화) |
| ✅ PDF | Platt Scaling for Calibration after Undersampling (2024) | https://arxiv.org/abs/2410.18144 | undersample시 절편 보정 |
| ✅ PDF | Reassessing How to Compare/Improve Calibration (2024) | https://arxiv.org/abs/2406.04068 | ECE 진단 한계 |
| ✅ PDF | Rolling-Origin Validation Reverses Model Rankings (2026) | https://arxiv.org/abs/2603.20315 | **시간순 검증 정당화** |
| ✅ PDF | Hidden Leaks in Time Series: Data Leakage in LSTM Eval (2025) | https://arxiv.org/abs/2512.06932 | 분할 전 시퀀스 생성 누수 경고 |
| ✅ PDF | Predicting LMs' Success at Zero-Shot Probabilistic Prediction (2025) | https://arxiv.org/abs/2509.15356 | LLM 확률예측 한계 |
| ✅ PDF | LLM Doesn't Know What It Doesn't Know (2026) | https://arxiv.org/abs/2606.19509 | **강한 부정 근거**: LLM verbalized confidence 무정보, XGBoost 승 |
| 🔒 URL | On Probability Estimation for Unbalanced Classification (Springer, 2026) | https://doi.org/10.1007/s42519-026-00572-5 | 리샘플링보다 prior-correction/캘리 |

## F. 기타 스포츠 (설계 패턴 참고)

| 상태 | 논문 (연도) | 링크 | 비고 |
|---|---|---|---|
| ✅ PDF | Commanding the Foul Shot: New Ensemble of Free Throw Metrics (2025) | https://arxiv.org/abs/2512.08824 | 상보적 서브메트릭 앙상블 설계 차용 |

## G. 유료/비-arXiv — 직접 다운로드 필요 (🔒)

| 논문 (연도) | 링크 | 비고 |
|---|---|---|
| Pitcher Performance Forecasting with Temporal Fusion Transformer (2025) | 검색: "Temporal Fusion Transformer pitcher performance 2025" | rate-stat 시계열 예측 |
| RLVP — RL with Verifiable Probability (ACM WWW, 2026) | https://doi.org/10.1145/3774904.3792540 | ⚠️ "55% 상회"=사실상 동률 |
| PipeBench — AutoML pipeline benchmark (Nature Sci Reports, 2026) | https://doi.org/10.1038/s41598-026-53722-x | 베이지안 최적화가 저비용에 우수 |
| Robustness Limits of LLMs on Tabular Predictions (PNAS Nexus, 2026) | https://academic.oup.com/pnasnexus (검색: "Robustness LLMs tabular predictions 2026") | LLM 테이블 예측 강건성 부족 |
| Pitch Outcome Prediction with Gradient Boosting (MDPI Applied Sciences 15:7081, 2025) | https://www.mdpi.com/search?q=pitch+outcome+gradient+boosting | xRV/xwOBA 회귀 |
| CNN-LSTM Player-Specific Pitch Type from Video (MDPI ASI, 2026) | https://www.mdpi.com/journal/asi | 비전+시퀀스 |
| KFYO — Fusing Vision & Biomechanics for Pitch Evaluation (Heliyon, 2026) | 검색: "KFYO pitch evaluation Heliyon 2026" | 스트라이크/볼 판정 |

### 비-논문 자료 (인용 주의 — 학술 근거 아님)
- **aimultiple Tabular Models Benchmark (2026)** — 벤더 마케팅 블로그 (동료심사 아님)
- **Driveline "Interaction of Biomechanics and Command" (2026)** — 산업 블로그
- **Kaggle Grandmasters Playbook (NVIDIA, 2025.10)** — 블로그 (스태킹 레시피 참고용)
- **MIT Sloan Transformer for Pitch Outcome (Kneita)** — 연도 미확정(2025/26)
