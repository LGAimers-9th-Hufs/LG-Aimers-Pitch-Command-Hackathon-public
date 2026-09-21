# LG Aimers 9기 × LG 트윈스 해커톤 — 투수 제구 성공 확률 예측 (팀 ABS깡통존)

> Predicting the probability that a KBO pitcher's **next pitch hits its intended location** — research notes, experiment harness, submission packaging and compliance tooling from the LG Aimers 9th × LG Twins sports hackathon (DACON 236743, 2026-07 ~ 08). Final: Public **1058.60**, rank 269.

이 저장소는 대회 기간 중 팀 대표 저장소로 쓴 private 저장소의 **정제 스냅샷**(단일 커밋)입니다. 대회 제공 데이터·파일, 제출물, 모델 가중치는 들어 있지 않습니다 — 무엇을 왜 뺐는지는 [`PUBLISHING_NOTE.md`](PUBLISHING_NOTE.md). 팀 소개와 저장소 4개의 관계는 [org 프로필](https://github.com/LGAimers-9th-Hufs)에 있습니다.

## 1. 문제

| | |
|---|---|
| 과제 | 투구 한 개마다 "이 공이 의도한 제구에 성공할 확률"을 내는 **이진 확률 예측** |
| 학습 / 평가 | 2019–2024 시즌 약 148만 투구로 학습 → **2025 시즌**(정답 비공개) 예측. 학습과 평가 사이에 시즌 경계가 있는 시간 기반 분할 |
| 입력 4축 | 경기 상황(볼카운트·주자·이닝 …) · 경기 중요도(승리 확률·레버리지) · 투수/타자의 **과거 이력 누적치**(`asof_*`) · 트랙맨 추적 로그(구속·회전·릴리스) — 선수·팀 ID는 익명 |
| 지표 | `Score = max(0, 100000 × (1 − Brier / r(1−r)))` — 순위가 아니라 **확률값 자체**의 정확도가 점수 |
| 규정 | 평가 서버에서 test의 각 행을 **다른 행과 무관하게** 예측해야 함(행 간 집계·배치 보정 금지), 추론 10분 제한 |

## 2. 우리가 만든 것 — 세 문장

1. **LightGBM 앙상블 + 당해 시즌 성적 분리 피처.** 입력의 이력 누적치는 커리어 전체 값이라 2025 안에서도 계속 갱신됩니다. 학습 데이터로 "직전 시즌 말까지의 누적"을 만들어 빼면 **당해 시즌 성적만** 남는데, 선수 ID가 입력에 없어 트리 모델이 스스로 만들 수 없는 신호입니다. 이 한 가지로 공개 점수 911 → 953 (`sweep/inseason.py`, `docs/log/`).
2. **새 피처·모델의 채택은 시간 순 교차검증 게이트로만.** 3개 폴드 평균이 0보다 크고 폴드 간 표준편차보다도 커야 채택. 무작위 6개 컬럼(위약)이 느슨한 기준을 통과하는 것을 실측하고 기준을 올렸습니다(`docs/log/`, `sweep/test_regulation.py`).
3. **최종 제출은 팀원 네 모델의 확률 균등 평균.** `Brier(평균) = 평균 Brier − 예측 분산/4` 라는 항등식 때문에 서로 다른 모델을 평균하면 이득이 구조적으로 보장됩니다. 가중치는 리더보드로 맞추지 않고 균등 1/4로 고정했습니다(`submission/build_team_blend.py`).

## 3. 결과와 버린 것

| 단계 | Public 점수 | 비고 |
|---|---:|---|
| 팀 LightGBM 기준선 | 853 | 최근 시즌 위주 학습 + 후처리 수축 (`sweep/lgbm_family.py`) |
| + 당해 시즌 성적 분리 | 953 | 단일 최대 신호 |
| 팀 네 모델 균등 블렌드 (`submit_blendD4`) | **1058.60** | 규정 준수 최종 제출 |
| (배제) 리더보드 역산 상수 계보 | 1075.84 | 아래 §4 |

시도했지만 버린 것(각각 이유는 [`docs/00_KILL_LIST.md`](Project_Baseball_Pitching/docs/00_KILL_LIST.md)):
- **외부 사전학습 표 모델(TabPFN·TabICL)** — 라이선스, 행 간 참조(transductive) 규정 저촉, 10분 추론 한도 초과(`docs/research/16`, `sweep/tabicl_pilot.py`)
- **LLM/강화학습 계열** — Brier 지표에서 GBDT를 이길 근거 없음(`docs/03`)
- **트랙맨 물리 피처 확장** — 선수 ID 매칭까지는 성공했지만 게이트를 넘는 이득이 없음(`docs/log/15`, `sweep/trackman*.py`)
- **kNN·TabM·선형 잔차 등 다양성 후보** — 앙상블 안에서 이득이 흡수됨(`docs/log/`)
- **학습 창(시즌 범위) 베팅** — 배포 조건에서 재측정하면 이득이 사라짐(`docs/log/37`)

## 4. 정직 고지 — 규정 위반을 스스로 찾아 배제한 기록

8월 28일 팀 재감사에서, 공개 리더보드 점수를 여러 번 제출해 **역산한 상수**(확률 보정 기울기·절편, 블렌드 가중)가 당시 최고점 계보(1075.84)에 들어 있음을 확인했습니다. 규정의 "평가 데이터 전체를 보고 만든 사후 보정값" 금지에 해당한다고 판단해 그 계보 전체를 최종 산출물에서 배제하고, 상수 없이 다시 조립한 1058.60으로 마감했습니다. 경위와 회복 산수는 [`phase3/COMPLIANCE_DISCLOSURE.md`](Project_Baseball_Pitching/phase3/COMPLIANCE_DISCLOSURE.md)·[`docs/log/33`](Project_Baseball_Pitching/docs/log/33_compliance_and_team_blend.md)에, 제출 zip에서 그런 상수를 잡아내는 검사기는 [`submission/scan_probe_provenance.py`](Project_Baseball_Pitching/submission/scan_probe_provenance.py)에 있습니다.

## 5. 저장소 지도 (`Project_Baseball_Pitching/`)

| 경로 | 내용 |
|---|---|
| `docs/01`–`10`, `docs/research/` | 사전 리서치 — 관련 연구 지형, 플레이북, LLM/RL 실효성 판정, 후보 모델·피처 카탈로그, 외부 사전학습 모델 조사 |
| `docs/log/11`–`37` | **실험 일지**(날짜순) — 실데이터 첫 실행, 트랙맨 선수 ID 매칭, 당해 시즌 분리, 컴플라이언스 사건, 팀 블렌드, 마무리 |
| `docs/01_DECISIONS.md`, `docs/00_KILL_LIST.md` | 되돌리기 어려운 결정의 근거 / 죽인 아이디어와 이유 |
| `sweep/` | 모델링 하네스(약 75개 모듈): 데이터 로더·시간 분할(`real_data.py`), LightGBM 계열(`lgbm_family.py`), 당해 시즌 분리(`inseason.py`), 앙상블 조립(`eb_carrier.py`), 트랙맨 매칭(`trackman*.py`), 규정 검사(`test_regulation.py`) |
| `submission/` | 제출 패키징·검증: 블렌드 조립(`build_team_blend.py`), 서버 조건 재현 검증(`verify_submission.py`), 역산 상수 검사기(`scan_probe_provenance.py`), 후보 조합 탐색(`combo_search.py`) |
| `phase3/` | 재현 패키지 수집기(`tools/collect_package.py`; 최종 제출을 245,789행에서 RMS 0.000000으로 재현), 솔루션 발표 자료(PPT·그림), 컴플라이언스 공시 |
| `results/` | 게이트·지표 json, 실험 점수표(`leaderboard_history.csv`), 리더보드 제출 원장(`lb_history_260829.md`) |
| `papers/PAPERS.md` | 참고 논문 목록 |

처음이라면 이 순서로: [`docs/log/33`](Project_Baseball_Pitching/docs/log/33_compliance_and_team_blend.md)(무슨 일이 있었고 어떻게 마감했나) → [`sweep/inseason.py`](Project_Baseball_Pitching/sweep/inseason.py)(핵심 피처) → [`submission/build_team_blend.py`](Project_Baseball_Pitching/submission/build_team_blend.py)(최종 제출이 어떻게 조립되나).

## 6. 재현

```bash
# 1) 대회 데이터(train.csv, test.csv, trackman_history.csv, sample_submission.csv)를
#    Project_Baseball_Pitching/data/ 에 둔다 — 저장소에는 없음
pip install -r Project_Baseball_Pitching/requirements.txt
cd Project_Baseball_Pitching
python sweep/test_regulation.py                       # 행 독립성·패리티 검사
python sweep/trackman.py --sig && python sweep/trackman.py   # 트랙맨 선수 매칭표 재생성 (공개본 제외분)
python submission/build_team_blend.py --help          # 제출 zip 조립 → scan_probe_provenance.py → verify_submission.py
```

공개본에서 제외한 파생 산출물(트랙맨 매칭표·선수별 프로필·모델 가중치)은 위 스크립트로 다시 만들어집니다. 평가 서버 조건(L4 GPU·6 vCPU·28 GB·추론 10분)은 `docs/SUBMISSION_COMPLIANCE.md`.

## 7. 팀 저장소와 공개 범위

최종 블렌드의 네 모델 중 둘(`clookup`, `cmoe`)은 팀원 저장소 `lg-aimers-ysy`, 하나(`tm3L`)는 `LG_Aimers_JTT`의 트랙맨 파이프라인을 여기서 행 독립 방식으로 다시 빌드한 것, 하나(`physmix`)는 이 저장소의 앙상블입니다. 팀원 저장소의 공개본(`-public`)은 각자 검토를 마치는 대로 org에서 공개됩니다.

데이콘 안내(2026-09-21)에 따라 **대회 제공 데이터·파일 자체, 개인정보·민감정보는 제외**했습니다. 대회 데이터에서 파생한 선수별 집계표·ID 연결표, 모델 가중치, 제출 zip도 보수적으로 제외했고, 문서 속 익명 팀 ID ↔ 실제 구단 매핑은 익명화했습니다. `CLAUDE.md`는 대회 기간 중 코딩 에이전트에게 준 작업 지침 원문으로, 당시 상태를 그대로 담고 있습니다.
