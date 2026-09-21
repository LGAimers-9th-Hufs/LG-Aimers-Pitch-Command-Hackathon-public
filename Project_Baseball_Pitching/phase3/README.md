# Phase 3 제출 패키지 — 재현 학습 코드 + 환경 명세

> **이 파일은 저장소 쪽 작업 지도다.** 실제로 주최에 내는 문서는
> [`package_README.md`](package_README.md)(= 패키지 안에서 `README.md`)이고,
> 패키지는 `python phase3/tools/collect_package.py`로 만든다.

## 0. 규칙과 일정 (2026-09-03 대회 페이지 확인 — 원문 전사는 [`RULES_PHASE3.md`](RULES_PHASE3.md))

필수 제출물 4종, `dacon@dacon.io` 메일, 코드 확장자 `.py`/`.ipynb`, 인코딩 UTF-8.

1. 학습 코드 개발 환경(OS) 및 라이브러리 버전
2. **Private Score 재현용 학습 코드** (추론 코드는 리더보드 제출본으로 대체)
3. 자유 형식의 솔루션 PPT
4. 팀원들의 오프라인 해커톤(Phase3) 참가 여부 기재

| 일정 | |
|---|---|
| 코드·PPT 제출 | 2026-09-02 12:00 ~ **2026-09-07(월) 10:00** |
| 코드 검증 | ~ 2026-09-11(금) |
| 진출자 발표 | 2026-09-14(월) |
| 오프라인 해커톤 | 2026-09-19(토)~20(일), LG인화원 |

## 0-1. 이 폴더의 구성

| 파일 | 역할 |
|---|---|
| `RULES_PHASE3.md` | 대회 규칙 §3·§2-4 원문 전사 + 제출물 대조 체크리스트 |
| `package_README.md` | **주최 제출용 패키지의 첫 문서** (환경·재현 절차·실측 결과·시드/HP) |
| `TEAM_REQUEST.md` | 팀원 요청 4항목 (참가 여부·재현 보고·실명·재배포 동의) |
| `COMPLIANCE_DISCLOSURE.md` | 규정 저촉 경위와 시정 (패키지에도 동봉) |
| `SOLUTION_OUTLINE.md` | 발표 원고 전문 S1~S11 |
| `PPT_OUTLINE.md` | 슬라이드 18장 구성안 |
| `tools/collect_package.py` | 패키지 수집기 — 폐포·커밋 고정·UTF-8 검사·zip |
| `package/`, `dist/` | 수집기 산출물 (gitignore) |

## 0-2. 먼저 읽을 것

**[`COMPLIANCE_DISCLOSURE.md`](COMPLIANCE_DISCLOSURE.md)** — 우리 팀 공개 리더보드 상위 제출물
일부가 규정 §5에 저촉되는 방식(리더보드 점수 역산)으로 만들어졌던 경위와, 최종 산출물에서
그것을 제거한 과정. **불리한 사실을 먼저 적었다.**

## 1. 최종 산출물의 구조

최종 제출물은 **서로 다른 파이프라인에서 나온 레그들의 확률 평균**이다.
현 규정 준수 최고 = `submit_blendD4.zip` (공개 **1058.6047851923**, 2026-08-29).

```
submit_blendD4.zip
├── model/legs/clookup/     ← 팀원 ysy: clean_regime + 계층 잔차 룩업                  (LB 1049.4562)
├── model/legs/cmoe/        ← 팀원 ysy: Challenger + 3-seed FactorTabM + TrackMan MoE (LB 1027.5280)
├── model/legs/physmix/     ← 본 저장소: ENS-9 (LGBM/ET/선형 5멤버 + 보정기 + physmix) (LB 1009.2645)
├── model/legs/tm3L/        ← 팀원 JTT 트랙맨 v3 모델 + 본 저장소의 §5 행 독립 재빌드   (역산 940.31)
├── model/legs/MANIFEST.json  ← 레그별 원본 SHA-256 · 가중치 · provenance
├── script.py               ← 각 레그를 자기 폴더에서 원본 script.py로 실행 → row_id 병합 → 평균
└── requirements.txt        ← 레그 requirements 합집합
```

가중치는 **균등 1/4 고정**이다. 리더보드에서 역산하지 않았다 —
Brier 확률 평균의 항등식 `Score(p̄) = 평균Score + C·E_i[(p_i − p̄)²]` (`C = 1e5/(r(1−r))`)이
학습 데이터만으로 이득을 보장하므로, `d = S_B − S_A`가 작을 때 최적 가중은 구조적으로 0.5 근방이다.
(첫 팀 블렌드 `submit_blendA3.zip` = clookup 대신 cregime, tm3L 없이 균등 1/3 — 1057.2623262209.
이 실측이 프록시이득→실현이득 배율 **ρ = 42.423/29.677 = 1.4295**의 측정점이다.)

**레그를 바이트 그대로 담는다.** 각 레그의 `script.py`는 한 줄도 고치지 않았고, 자기 cwd에서
`./data/test.csv`를 읽어 `./output/submission.csv`를 쓰는 서버 계약 그대로 실행된다.
train/serve skew는 정의상 0이다.

## 2. 재현 절차

### 2.1 레그별 학습 (각 레그는 독립적으로 재현된다)

| 레그 | 저장소 | 학습 진입점 |
|---|---|---|
| `clookup` | `LGAimers-9th-Hufs/lg-aimers-ysy` | `clean_regime`(`pipelines/dsf_upgrade/build_regime_submission.py`, 공식 `train.csv`로 처음부터 재학습) + 계층 잔차 lookup 층 (`pipelines/clean_forest/train_lookup_final.py` → `build_lookup_submission.py`) |
| `cmoe` | `LGAimers-9th-Hufs/lg-aimers-ysy` | `pipelines/dsf_upgrade/build_clean_moe_submission.py` |
| `physmix` | 본 저장소 | `python submission/build_lgbm_ensemble.py --ens9 --callocal --tag physmix` |
| `tm3L` | 모델 = `LGAimers-9th-Hufs/LG_Aimers_JTT` `trackman_pipeline/`(v3) · 서빙 재작성 = 본 저장소 | `python submission/build_tm3_rowwise.py` (원본 v3의 test-배치 통계 3종을 train 유래 상수로 치환 — `docs/log/35` §3, 원본 `build()` 파리티 0.000e+00) |

`physmix`가 의존하는 학습 코드:
`sweep/lgbm_family.py`(june 계열 LGBM + 피처 53) · `sweep/inseason.py`(당해 시즌 분해 is4) ·
`sweep/nn_member.py`(원핫 선형 멤버) · `sweep/eb_carrier.py`(보정기) ·
`sweep/tm_physmix.py`(트랙맨 물리 × 행 갱신 mix) · `sweep/real_data.py`(로더 + SERVE 피처 블록).

### 2.2 블렌드 조립

```bash
python submission/build_team_blend.py --tag blendD4 \
  --leg clookup=<clean_lookup/submission.zip> \
  --leg cmoe=<clean_moe/submission.zip> \
  --leg physmix=submission/dist/submit_ens9_physmix.zip \
  --leg tm3L=submission/dist/submit_tm3L.zip
```

빌더는 원본 zip 엔트리를 **바이트째** 복사하고 `MANIFEST.json`에 SHA-256을 남긴 뒤
**빌드 직후 재대조**한다(해시 봉인 아티팩트가 개행 변환으로 조용히 파괴된 전례가 있어서다).

### 2.3 검증 (전부 통과해야 제출)

```bash
python submission/verify_submission.py submission/dist/submit_blendD4.zip --proxy 245789 --threads 6
python submission/scan_probe_provenance.py submission/dist/submit_blendD4.zip   # BLOCK 0건
python sweep/test_regulation.py                                                 # 행 독립성·파리티
```

실측(2026-08-29): 구조 · §5 배치 통계 부재(AST) · 행 수 245,789 · `row_id` 순서 · 확률 범위 ·
런타임 **3레그 60.3초 · 5레그 71.5초**(서버 제한 600초) 전부 통과. 엔트리 SHA-256 불일치 0건.
**구성 대조**(A3): 블렌드 출력 vs 세 레그 산술평균 최대 절대차 `2.220e-16` = 원소 단위 일치.

**Phase 3 재현 실측(2026-09-03)** — 상세는 [`package_README.md`](package_README.md) §5:
physmix 재학습 105초에 **모델 가중치 4파일 SHA-256 비트 일치** · 보정기 완전 일치 ·
예측 대조 RMS `0.000000` / tm3L 5엔트리 전부 비트 일치 / blendD4 재조립 **91엔트리 중 88 비트 일치**
(차이 3개는 서빙 템플릿 상위호환 + provenance 문구 시정, 예측 영향 0) /
scan BLOCK 0 · verify 245,789행 통과(정상 195.9초·서버핀 21.0초) · test_regulation 전부 통과.

⚠ **재현이 한 번 깨졌다가 원인을 특정했다.** `results/trackman/matches.csv`를 2026-08-30
연구 라운드(GMM Tier 재컷, 커밋 `3917b5f`)가 덮어써 Tier 1이 **407 → 179명**으로 줄었고,
physmix 물리 보정기 프로필이 400 → 179투수가 됐다. 제출본이 실제로 쓴 판은 커밋 `17fc655`이며
수집기가 그 판을 고정 동봉한다. **연구 라운드가 공유 아티팩트를 제자리에서 덮어쓰면 배포본
재현이 조용히 깨진다** — 아티팩트에도 코드와 같은 버전 고정이 필요하다.

## 3. 개발 환경

| | 학습(로컬) | 평가 서버 |
|---|---|---|
| OS | Windows 11 Pro 26200 | Ubuntu 22.04 |
| Python | 3.13.7 | 3.11.15 |
| numpy | 2.2.6 | **2.x** (주최 `rf.pkl`이 `numpy._core` 참조 = 2.x 직렬화 실측) |
| scikit-learn | 1.8.0 | 1.8.0 |
| scipy | 1.17.0 | 1.15.3 (사전설치) |
| pandas | 3.0.0 (08-29 기록은 2.3.3 — 그 차이에도 재현 비트 일치) | 2.0.3 기본 / requirements로 2.3.3 |
| LightGBM | 4.6.0 | `requirements.txt`로 설치 (`lightgbm==4.6.0`) |
| torch | 2.11.0+cu128 | 2.7.1+cu128 (사전설치, requirements에 넣지 않음) |
| 하드웨어 | — | NVIDIA L4 22.4GiB · 6 vCPU · RAM 28GB · CUDA 12.8 |

제출 zip의 `requirements.txt` = 레그 requirements 합집합:
`joblib==1.5.3` · `lightgbm==4.6.0` · `pandas==2.3.3` · `scikit-learn==1.8.0`.
torch는 서버 사전설치본을 쓴다(`clean_moe`가 이 구성으로 채점된 것이 실증).

**커스텀 클래스를 pickle에 넣지 않는다** — 서버에 우리 모듈이 없어 언피클이 실패한다.
`_OffsetBoost`/`_SeedAvg` 같은 래퍼는 순수 sklearn 객체의 dict로 분해해 동봉하고 `script.py`가 조합한다.

## 4. 규정 준수 근거

- 행 독립성(§5): 각 레그가 `test.csv`의 **한 행만 보고** 예측한다. 블렌드 층은 레그 출력을
  `row_id`로 병합해 가중 평균할 뿐 행 간 참조가 없다. `verify_submission`이 AST로
  `mean/groupby/value_counts/transform/rolling/expanding/cumsum/rank/shift` 호출 부재를 강제하고,
  `sweep/test_regulation.py::test_row_independence`가 단일 행 결과 == 전체 프레임 결과임을 증명한다.
- 사용 금지 정보(§6): 현재 투구의 위치·판정·결과·구종·트랙맨 측정값 미사용. 2025 트랙맨 미사용.
  입력은 공식 `train.csv` · 평가환경 `test.csv` · `trackman_history.csv`(2019~2024)뿐.
- 리더보드 역산 상수: **없음**. 근거와 경위는 `COMPLIANCE_DISCLOSURE.md`.

아티팩트별 허용 근거표는 `docs/SUBMISSION_COMPLIANCE.md`.

## 5. 발표 자료

[`SOLUTION_OUTLINE.md`](SOLUTION_OUTLINE.md) — 솔루션 PPT 원고.
