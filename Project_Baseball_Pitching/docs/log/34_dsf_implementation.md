# 34 — JTT-DSF 구현 및 시간순 검증 (2026-08-13)

## 결론

- 안전한 공식 앵커는 계속 `FINAL_submit_blend = 1032.6717450165`다.
- 점수 상승 우선 선택에 따라 공격 후보 `FINAL_SUBMIT_260813.zip`을 만들었다.
- 계층 선수×상황 residual은 7개 forward split 모두 양수였다.
- 그러나 1100 승격 기준인 가중 `+90`은 통과하지 못했다. 1100을 보장하거나 공식 예상점수로
  표현하지 않는다.

## 구현

### 동적 상태

`sweep/dsf_features.py`

- 누적 as-of 카운터를 시즌말끼리 차분해 시즌 관측으로 복원
- 최근 시즌 가중 상태 mean/variance/trend와 Beta posterior 생성
- 투수 성공/실패형, 타자, 구종 mix를 모두 frozen artifact + 현재 행으로 갱신
- 학습/서빙 공통 feature 계약과 state-only 부분공간 제공

### 소프트 TrackMan

`sweep/dsf_trackman.py`

- target 시즌 이전 시그니처만으로 행동 지문 비용을 재계산
- hard ID 하나 대신 top-k 후보 posterior 유지
- 물리 profile posterior mean/variance, entropy, top1, coverage를 frozen lookup으로 생성

### 시간순 residual pipeline

`sweep/dsf_pipeline.py`

- ENS-9 chronological cache의 level/slope 직교 shape residual만 학습
- V2022/V2023/V2024 고정 cap/gamma 검증
- gate 실패 시 final artifact 생성을 거부해 champion을 자동 보호

### 계층 entity residual

`sweep/dsf_entity_residual.py`

- 투수, 타자, 투수×count-group, 타자×count-group, 투수×same-hand의 강수축 lookup
- 같은 시즌 앞달→뒷달 forward split 6개와 안정 체제 2022→2023 split
- 이전 시즌 residual을 `0.30^age`로 약하게 전달
- 추론은 현재 행 key와 frozen lookup만 사용

## 실측 결과

### Dynamic-state residual — 기각

| Fold | best gain |
|---|---:|
| V2022 | -3.61 |
| V2023 | -31.02 |
| V2024 | -0.44 |

현재 JTT의 in-season/EB 표현과 중복됐다. 재캘리 불변 오라클 여유도 최대 약 +5였다.

### Dynamic + soft TrackMan residual — 기각

| Fold | best gain |
|---|---:|
| V2022 | -4.61 |
| V2023 | -34.60 |
| V2024 | +0.65 |

soft posterior를 넣어도 반복 가능한 전이 신호가 없었다.

### Temporal hierarchical entity residual — incremental GO

최종 공격 설정: `hier_context`, cap `0.150`, gamma `0.30`, history weight `0.45`.

과거 fold 기준선도 raw ENS-9가 아니라 frozen affine/segment를 적용한 exact-family 배포 형태로
수정해 학습-서빙 확률 공간의 불일치를 줄였다. 선택 목적함수는 2025에 가까운 구간을 우선해
`2024 동일시즌 50% + 안정 교차시즌 25% + 이전 동일시즌 25%`로 고정했다.

| Split | gain |
|---|---:|
| 2022 month≤6 → later | +22.11 |
| 2022 month≤8 → later | +50.40 |
| 2023 month≤6 → later | +154.61 |
| 2023 month≤8 → later | +119.37 |
| 2024 month≤6 → later | +17.88 |
| 2024 month≤8 → later | +37.61 |
| 2022 → 2023 | +29.57 |
| 평균 / 최악 | +61.65 / +17.88 |

V2024 forward 평균은 `+27.74`다. 전 split 양수라 공격 후보 조건은 통과했지만,
1100용 강한 gate(V2024 평균 +30 및 전체 +90)는 통과하지 못했다.

## 패키지 및 검증

- 원본 보존: `open/experiments_backup/FINAL_SUBMIT_260812_anchor_1032.671745.zip`
- 신규 후보: `submission/dist/FINAL_submit_blend_DSF_ATTACK.zip`
- 현재 선택본: `open/FINAL_SUBMIT_260813.zip`
- SHA-256: `3488F90AC6F9678165DE637C6B7F4552DC64D733707B26F1A8985A58FA107BCA`
- 생성기: `submission/build_dsf_entity_submission.py`
- 모델 artifact: `results/dsf_entity/entity_final.json`
- 단위/독립성 테스트: `sweep/test_dsf.py` — 5/5 통과
- 제출 검증: 60,000행, 규정 AST/형식/안전망 전부 통과, 정상 경로 4.5초

로컬 추정기는 `1073.84`를 냈지만 2024 proxy가 artifact 학습 행을 다시 보므로 과대평가된다.
이 수치는 공식 기대점수나 1100 근거로 쓰지 않는다. 정직한 근거는 위 forward split뿐이다.

## 재현

```powershell
$py = 'D:\Vscode file\LG_baseball\.venv\Scripts\python.exe'
& $py sweep\test_dsf.py
& $py sweep\dsf_pipeline.py --validate --with-trackman
& $py sweep\dsf_entity_residual.py --validate --attack-refine --fit-final
& $py submission\build_dsf_entity_submission.py --output submission\dist\FINAL_submit_blend_DSF_ATTACK.zip
& $py submission\verify_submission.py submission\dist\FINAL_submit_blend_DSF_ATTACK.zip --proxy 60000 --threads 6 --no-server-env
```
