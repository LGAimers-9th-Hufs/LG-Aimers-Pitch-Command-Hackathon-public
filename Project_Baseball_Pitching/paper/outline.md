# Predicting Next-Pitch Command Success under Temporal Shift (working title)

> 논문 뼈대. 섹션 제목은 영문(LaTeX 1:1 전환용), 메모는 한국어.
> `<!-- AUTO:... -->` 마커 사이는 `python sweep/make_paper_assets.py`가 최신 run으로 자동 치환 — 손으로 수정 금지.
> 재료 매핑: 조사문서 01~10 + `sweep/` 하네스 + `results/`(run 기록).

## 1 Introduction

- 과제: 투수의 **다음 투구 제구 성공 확률**(pitch-level binary probability), train 2019–2024 → hidden test 2025, 익명 ID. ← `HACKATHON_TASK.md`
- 왜 어려운가: 시간 시프트(ABS 도입), 콜드스타트(신인/희귀 투수), 확률 캘리브레이션이 곧 지표(LogLoss). ← `02_playbook.md`
- 기여(초안): (i) 누수-안전 as-of 피처 + fold-aware cutoff 프로토콜, (ii) ~80 arm 카탈로그를 동일 시간분할 게이트로 공정 비교하는 successive-halving 하네스, (iii) 시프트·콜드스타트·캘리 관점의 실증 분석.

## 2 Related Work

- 야구 커맨드/제구 예측·측정(Command 계량, Kirby Index, Location+ 등) ← `01_landscape_survey.md`, `09_new_candidates_and_related_research.md` §2, `papers/`
- Tabular 학습·시간분할 검증(temporal holdout 대회 기법, adversarial validation) ← `04_cross_domain_methods.md`, `09` §3
- LLM·RL 접근의 한계 — negative result로 정리 ← `03_llm_rl_feasibility.md`, `07_llm_to_train_model.md`

## 3 Data & Task

- 데이터 계약(DataSpec: season/game/pitch 순서, context, release 로그) ← `sweep/config.py`
- 라벨 정의·공식 지표(별첨 확정 시 업데이트) ← `HACKATHON_TASK.md`
- 시간 구조: 시즌 단위 forward-chaining, 2024+ ABS era. ← `10_contextual_environmental_factors.md`

## 4 Method

- **Leak-safe as-of features**: shift(1) 기반 과거-전용 집계 + 베이즈 수축; target-파생 통계는 val 시즌 cutoff에서 동결(fold-aware). 플라시보 셔플로 누수 0 증명. ← `05_final_blueprint.md`, `sweep/features.py`, `sweep/test_leakage.py`
- **Forward-chaining CV + calibration**: train=과거 시즌, val=미래 시즌; train 내 최신 시즌 홀드아웃에 isotonic(기본), C5 변형(recent_window/prior_shift) A/B. ← `sweep/validation.py`, `09` §4
- **Candidate catalog & fair gate**: 모델 M0–M36 × 피처 F1–F33 × 캘리 C1–C6 — 하드 제외 없음, tier는 예산 배분만. ← `06_candidate_pipelines_catalog.md`, `10`
- **Successive halving**: rung0(1 fold) 스크리닝 → 생존자 전체 폴드 승격, 게이트=시간분할 CV LogLoss로 앵커(GBDT) 대비 판정. ← `sweep/sweep.py`
- **Distillation track**: teacher(LLM/TFM)는 훈련 도구, 제출물은 자체 학습 student(동일 게이트 통과 필수). ← `08_distillation_plan.md`

## 5 Experiments & Results

Figures: `paper/assets/leaderboard_logloss.pdf`(리더보드), `perfold_logloss.pdf`(시즌별 = ABS era 시프트), `reliability.pdf`(캘리 diagram), `calib_ab.pdf`(C5 A/B).
Ablation 원자료: `results/leaderboard_history.csv` (run × 후보 롱포맷, "arm X가 언제부터 앵커를 이겼나").

<!-- AUTO:leaderboard:begin -->
| # | candidate | tier | LogLoss | Brier | AUC | Cold | Δ anchor | gate |
|---|---|---|---|---|---|---|---|---|
| 1 | S2_mlp_student | T3 | 0.6749 | 0.2403 | 0.610 | 0.7052 | -0.0012 | **beats anchor** |
| 2 | D2_kd_student | T3 | 0.6760 | 0.2402 | 0.611 | 0.7076 | -0.0001 | **beats anchor** |
| 3 | M1_gbdt_anchor | T1 | 0.6762 | 0.2413 | 0.600 | 0.6946 | +0.0000 | anchor |
| 4 | M24_glmmnet | T3 | 0.6765 | 0.2390 | 0.620 | 0.6946 | +0.0004 | - |
| 5 | M15_bayes_log5 | T3 | 0.6781 | 0.2404 | 0.606 | 0.6989 | +0.0019 | - |
| 6 | M25_merf | T3 | 0.6793 | 0.2399 | 0.613 | 0.6847 | +0.0032 | - |
| 7 | M1F13_gbdt_llm | T4 | 0.6797 | 0.2426 | 0.593 | 0.6966 | +0.0035 | - |
| 8 | M1rich_gbdt_f15_19 | T2 | 0.6805 | 0.2409 | 0.610 | 0.6870 | +0.0043 | - |
<!-- AUTO:leaderboard:end -->

## 6 Discussion

- 분포 시프트: P1 adversarial validation AUC, as-of 히스토리 깊이가 주 구분자 ← `sweep/diagnostics.py`, run.json `diagnostics.adversarial`
- ABS era(2024+) 전후 per-season 성능 변화 ← `10`, `perfold_logloss`
- 콜드스타트: 얕은 히스토리 구간 LogLoss(`Cold` 열), 랜덤효과 계열(M24/M25)의 fallback 이점 ← `09` §5
- Limitations: 합성 검증 단계 결과는 파이프라인 증명용, 실데이터 결론 아님.

## 7 Reproducibility

모든 수치·figure는 `results/runs/<run_id>/run.json`에서 재생성 가능(`sweep/figures.py`).

<!-- AUTO:run_meta:begin -->
- run_id: `20260703-091215_synthetic`  (data_source=synthetic, seed=7)
- git: `3f55768618009d738bf3ddfeffd9663bc881a343` (branch=master, dirty=True)
- harness v0.1, timestamp 2026-07-03T09:12:15.692676+00:00
- adversarial validation AUC: full=0.844, context-only=0.493
<!-- AUTO:run_meta:end -->
