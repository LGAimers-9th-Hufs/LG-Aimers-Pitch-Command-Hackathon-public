# Partner 2 handoff

## 바로 쓸 파일

- 검증: `team_pred_2024.csv`, `team_pred_2023.csv`, `team_pred_2022.csv`, `team_pred_2021.csv`.
- 서빙: `serving/predict.py`의 `predict(test_df, model_dir)`.
- 정규화 상수: `serving/MANIFEST.json`의 `raw_mean_2024=0.4598991505`,
  `raw_std_2024=0.0353605790`.
- 재현: `train_partner2.py`; 상세 판단: `REPORT.md`.

## 채택 판단

- 팀 앵커 대비 2024 `s=0.04276`로 요청한 다양성 범위에 도달했다.
- 네 폴드 `gain_re`는 `+0.33 / +62.94 / +33.35 / +14.21`, 평균 `+27.71`로 모두 양수다.
- 그러나 핵심 2024 이득은 거의 0이고 첫 파트너와의 2024 s는 `0.02923`이다.
- 따라서 첫 파트너를 대체하기보다, 세 모델 공동 OOF 최적화에서 작은 세 번째 축으로 시험하는
  편이 타당하다. 2024 한 폴드의 고정캘리 최적 w=0.151을 그대로 제출 가중치로 쓰면 안 된다.

## 서빙 상태

- `season <= 2024` 전량 학습 완료, 2025 예측 base 상수 `0.4621786761`.
- 245,789행 1.759초, NaN 0, float64, [0,1] 통과.
- `predict.py`와 `selftest.py` AST 금지 호출 0건.
- 새 pitcher ID는 저장된 TrackMan 전역 중앙값 + confidence 0으로 폴백한다.
- 최종 hybrid 모델 gain의 28.86%가 TrackMan 궤적 피처다.

## 꼭 알아둘 위험

- 2023 raw score 0, SD 0.0782로 레짐 민감도가 크다.
- 엔티티 매칭은 실명 정답 없는 고신뢰 부분 매칭이며 2024 검증 행 커버리지는 58.87%다.
- 2024 `gain_re=+0.33`이므로 이 모델만으로 목표 격차를 여는 근거는 없다.
