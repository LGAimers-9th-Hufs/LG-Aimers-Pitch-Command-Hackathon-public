# PUBLISHING_NOTE — `LG-Aimers-Pitch-Command-Hackathon-public`

원본: `LGAimers-9th-Hufs/LG-Aimers-Pitch-Command-Hackathon` @ `e45e50b27ffc` · 생성: `publish/export.py` (fresh history, 단일 커밋)

> 데이콘(2026-09-21): "대회 종료 이후 참가자가 직접 작성한 코드는 개인 GitHub 등을 통해 공개하셔도 무방하며 … 대회에서 제공된 데이터 및 파일 자체, 개인정보·민감정보 등 외부 공개가 제한되는 내용은 포함하여 공개할 수 없으므로 반드시 제외해 주시기 바랍니다."

이 저장소는 대회 기간 중 사용한 private 저장소의 **정제 스냅샷**(단일 커밋)입니다. 원본 히스토리는 공개하지 않습니다.

제외한 것(데이콘 안내: 대회 제공 데이터·파일 자체, 개인정보·민감정보 공개 불가):
- 대회 제공 파일: `data/`(train/test/trackman_history/sample_submission — 원래부터 미추적), 주최 측 베이스라인 노트북(`baseline/`), 과제 설명 PDF
- 대회 데이터에서 파생한 선수별 집계·ID 연결표: `results/trackman/*`, `results/phase3/profiles.json`, `results/dsf/soft_tm_*.json`, `results/dsf_entity/entity_final.json` — `sweep/trackman*.py`, `sweep/dsf_*.py`로 재생성
- 제출 zip·모델 가중치 전부(`submission/dist/`, `phase3/dist/`, `results/codex_partner*/serving/*.txt`) — 각 `submission/build_*.py`로 재생성
- 팀 내부 조율 문서(실명·출석 요청, 메일 초안), 팀원의 사적 제출 메모 전사, 팀원 zip 단위 판정 파일
- LG AI 커리큘럼 강의자료와 그 파생 요약(`1_`~`6_`, `INDEX.md`, `RESEARCH_SUMMARY.md`) — 저작물
- 문서 내 익명 `pitcher_team_id` ↔ 실제 구단 매핑(`docs/log/15` §2)은 구단명을 익명화

컴플라이언스 사건(`phase3/COMPLIANCE_DISCLOSURE.md`, `docs/log/33`)은 팀 단위 서술로 남겼고 특정 팀원에게 귀속하는 문장은 일반화했습니다.

## 제외 내역 (사유별 파일 수 / 크기)

| 제외 사유 | 파일 수 | 크기 | 예시 |
|---|---:|---:|---|
| LG AI 강의자료(저작물) | 19 | 69.7 MB | `5_LLM_Application_and_Evaluation/『LLM Application & Evaluation』 강의자료.pdf` |
| 제출물 디렉터리(`dist/`) | 5 | 66.2 MB | `Project_Baseball_Pitching/submission/dist/submit_blendD4.zip` |
| LightGBM 텍스트 모델 | 20 | 5.4 MB | `Project_Baseball_Pitching/results/codex_partner2/serving/hybrid_model.txt` |
| 명시 제외 파일(내부 조율 문서·판정 파일·강의 요약·데이터 설명서 사본 등) | 13 | 1.7 MB | `해커톤주제.pdf` |
| 선수별 트랙맨 프로필(대회 데이터 파생) | 4 | 1.2 MB | `Project_Baseball_Pitching/results/phase3/profiles.json` |
| 트랙맨 ID 매칭표·프로필·구단 매핑(대회 데이터 파생) | 8 | 0.5 MB | `Project_Baseball_Pitching/results/trackman/tm_profile_tier1.json` |
| 선수별 잔차 보정표(대회 데이터 파생) | 1 | 0.4 MB | `Project_Baseball_Pitching/results/dsf_entity/entity_final.json` |
| 공개 수출 도구(금지어 목록 포함, private 원본 전용) | 13 | 0.3 MB | `publish/manifests/lg-aimers-ysy-public.json` |
| 주최 측 베이스라인 노트북(대회 제공 파일) | 2 | 0.0 MB | `Project_Baseball_Pitching/baseline/[Baseline_Inference]_RandomForest를 활용한 모델 학습 및 피쳐엔지니어링 (추론).ipynb` |

전체 목록은 원본(private) 저장소의 `publish/manifests/` manifest(json)에 기록되어 있습니다.
