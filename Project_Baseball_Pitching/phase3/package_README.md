# LG Aimers 9기 × LG 트윈스 해커톤 — Phase 3 재현 학습 코드

**팀 ABS깡통존** · 투구 제구 성공 확률 예측 (train 2019–2024 → test 2025)

> 대회 규칙 §3 "[제출 파일 목록] o (필수) Private Score 재현용 학습 코드"에 대응하는 패키지다.
> 추론 코드는 규칙대로 **리더보드에 제출한 추론 코드로 대체**되며, 이 패키지는 그 제출물의
> `model/` 폴더를 처음부터 다시 만들어내는 학습 코드다.

---

## 0. 검증 대상 제출물

| | |
|---|---|
| 파일 | `submit_blendD4.zip` |
| 공개 점수 | **1058.6047851923** |
| 구조 | 서로 다른 파이프라인 4레그의 **확률 균등평균 (각 1/4)** |
| 제출 시각 | 2026-08-29 |

### ⚠ 먼저 밝히는 것 — 우리가 더 높은 점수를 방어하지 않는 이유

리더보드에 남은 우리 팀의 최고 점수는 **1075.8374148399**(`factor_tabm_router`)다.
그러나 2026-08-28 팀 내부 재감사에서 **그 계보에 리더보드 점수를 역산해 박은 상수**가
있음이 확인됐다. 규칙 §5 금지 목록의 "평가 데이터 전체를 보고 만든 사후 보정값"에 해당한다.

⇒ 우리는 그 계보를 **최종 산출물에서 전면 배제**하고, 규정 준수가 기계적으로 확인된
`submit_blendD4.zip`을 재현 대상으로 제출한다. 공개 점수 기준 **17.23점**을 스스로 반납한 것이다
(1075.8374 - 1058.6048).
경위·범위·시정의 전문은 **[`COMPLIANCE_DISCLOSURE.md`](COMPLIANCE_DISCLOSURE.md)** 에 있다.
불리한 사실이라 먼저 적었다.

### 레그 구성

| 레그 | 가중 | 단독 공개점수 | 만든 사람 | 학습 코드 |
|---|---:|---:|---|---|
| `clookup` | 1/4 | 1049.4562 | 팀원 (ysy) | `partners/lg-aimers-ysy/pipelines/clean_forest/` |
| `cmoe` | 1/4 | 1027.5280 | 팀원 (ysy) | `partners/lg-aimers-ysy/pipelines/dsf_upgrade/build_clean_moe_submission.py` |
| `physmix` | 1/4 | 1009.2645 | 본 저장소 | `submission/build_lgbm_ensemble.py` |
| `tm3L` | 1/4 | (미제출, 역산 940.3) | 모델 = 팀원(JTT) · 서빙 재작성 = 본 저장소 | `partners/LG_Aimers_JTT/trackman_pipeline/` + `submission/build_tm3_rowwise.py` |

가중치는 **균등 1/4 고정**이며 리더보드에서 역산하지 않았다. Brier 확률 평균의 항등식

```
Score(p̄) = 평균Score + C · E_i[(p_i − p̄)²]          C = 1e5/(r(1−r)) ≈ 4·10⁵
최적 가중  w* = 0.5 + d/(2K)      (d = 레그 점수차, K = 4e5·RMS²)
```

에서 `d`가 작을 때 `w*`가 0.5에 붙으므로, 균등 가중은 **학습 데이터만으로** 정당화된다.

---

## 1. 팀 구성과 오프라인 해커톤(Phase 3) 참가 여부

> 규칙 §3 "[제출 파일 목록] o (필수) 팀원들의 오프라인 해커톤(Phase3) 참가 여부 기재"

| 팀원 | 담당 | Phase 3 참가 |
|---|---|---|
| (TEAM_MEMBER_1) | physmix 레그 · tm3L 행독립 재빌드 · 블렌드 조립 · 규정 검증 도구 | (참가/불참) |
| (TEAM_MEMBER_2) | clookup · cmoe 레그 (clean_forest / dsf_upgrade 파이프라인) | (참가/불참) |
| (TEAM_MEMBER_3) | tm3L 원본 모델 (trackman_pipeline v3) | (참가/불참) |

---

## 2. 개발 환경

> 규칙 §3 "[제출 파일 목록] o (필수) 학습 코드 개발 환경(OS) 및 라이브러리 버전"

### 2.1 학습(로컬)

| | |
|---|---|
| OS | Windows 11 Pro 10.0.26200 |
| Python | 3.13.7 |
| numpy | 2.2.6 |
| pandas | 3.0.0 |
| scikit-learn | 1.8.0 |
| scipy | 1.17.0 |
| LightGBM | 4.6.0 |
| joblib | 1.5.3 |
| torch | 2.11.0+cu128 (팀원 레그 `cmoe`의 FactorTabM 학습에만 사용) |
| GPU | 학습에 필수가 아님 — `physmix`·`tm3L`은 CPU만으로 재현된다 |

`requirements_train.txt`에 같은 핀이 들어 있다.

### 2.2 추론(대회 평가 서버) — 참고

제출 zip의 `requirements.txt`는 다음 4핀이고, torch는 서버 사전설치본을 쓴다.

```
joblib==1.5.3
lightgbm==4.6.0
pandas==2.3.3
scikit-learn==1.8.0
```

서버 환경: Ubuntu 22.04 · Python 3.11.15 · NVIDIA L4 · 6 vCPU · RAM 28GB · CUDA 12.8.
주최 베이스라인 `rf.pkl`이 `numpy._core`를 참조하므로 서버 numpy는 2.x다(실측 확인).

**커스텀 클래스를 pickle에 넣지 않는다.** 서버에 우리 모듈이 없어 언피클이 실패하기 때문이다.
래퍼는 순수 sklearn 객체의 dict로 분해해 동봉하고 `script.py`가 조립한다.

---

## 3. 데이터 배치

데이터는 주최 배포본이므로 패키지에 넣지 않았다. `open.zip`을 풀어 다음 위치에 둔다.

```
data/data/train.csv
data/data/test.csv
data/data/sample_submission.csv
data/data/trackman_history.csv
```

`sweep/real_data.py`가 자기 위치 기준 `parent.parent/"data"/"data"`를 보므로 **경로가 고정**이다.
`sweep/`는 평면 sibling import 패키지라 폴더를 재배치하면 전부 깨진다.

---

## 4. 재현 절차

패키지 루트에서 순서대로 실행한다. 전체 20분 이내(CPU)에 끝난다.

```bash
pip install -r requirements_train.txt

# 1) physmix 레그 학습 (약 105초)
python submission/build_lgbm_ensemble.py --ens9 --callocal --tag ens9_physmix

# 2) tm3L 레그 — 팀원 v3 모델의 §5 행독립 재빌드 (약 60초)
python submission/build_tm3_rowwise.py

# 3) 최종 제출물 조립 — 레그를 바이트 그대로 담고 SHA-256을 봉인한다
python submission/build_team_blend.py --tag blendD4 \
  --leg clookup=partners/lg-aimers-ysy/model_artifacts/clean_lookup/submission.zip \
  --leg cmoe=partners/lg-aimers-ysy/model_artifacts/clean_moe/submission.zip \
  --leg physmix=submission/dist/submit_ens9_physmix.zip \
  --leg tm3L=submission/dist/submit_tm3L.zip

# 4) 규정·서버계약 검증 (전부 통과해야 제출)
python submission/scan_probe_provenance.py submission/dist/submit_blendD4.zip   # BLOCK 0건
python submission/verify_submission.py submission/dist/submit_blendD4.zip --proxy 245789 --no-server-env
python sweep/test_regulation.py
```

`--callocal`은 캘리브레이션 상수의 출처를 **로컬 재적합값**으로 명시하는 플래그다.
`--calfitted`(리더보드 역산)와 둘 중 하나를 명시하지 않으면 빌드가 실패하도록 되어 있다 —
과거에 provenance 문장이 사실과 어긋난 채 제출된 사고(디스클로저 §3)의 재발 방지 장치다.

`--no-server-env`는 서버 핀 버전 가상환경 검사를 건너뛴다(그 venv는 용량 때문에 미동봉).

### 4.1 팀원 레그(clookup·cmoe)의 재학습

두 레그는 팀원 저장소의 파이프라인으로 학습된다. 코드가 `partners/lg-aimers-ysy/`에
커밋 `e76e6f1` 기준으로 동봉돼 있고, 학습 산출물 `submission.zip`도 함께 들어 있어
위 3)의 조립이 그 자리에서 돌아간다. 처음부터 재학습하려면:

```bash
cd partners/lg-aimers-ysy
python pipelines/dsf_upgrade/build_regime_submission.py      # clean_regime (clookup의 베이스)
python pipelines/clean_forest/train_lookup_final.py          # 계층 잔차 lookup 층
python pipelines/clean_forest/build_lookup_submission.py     # -> clean_lookup.zip
python pipelines/dsf_upgrade/build_clean_moe_submission.py   # -> clean_moe.zip
```

기대 SHA-256: `clean_lookup.zip = c5b04786…5753`, `clean_moe.zip = 9d61bf8b…e61e`
(전체 값은 `SHA256SUMS.txt`). 자세한 설명은 각 폴더의 `README.md`와 `pipelines/*/PIPELINE.md`.

---

## 5. 재현 실측 결과 (2026-09-03, 이 패키지 트리에서 직접 실행)

**이 표는 주장이 아니라 측정이다.** 위 4절의 명령을 이 패키지에서 그대로 실행해
원 제출물과 대조한 결과다.

| 단계 | 결과 |
|---|---|
| physmix 재학습 | 105초. 모델 가중치 4개 파일 **SHA-256 비트 일치** (`allraw_l15.txt`·`et_l100_d28.joblib`·`june_l15.txt`·`june_l7.txt`), `requirements.txt` 일치 |
| physmix 보정기 | `pm_corrector`(프로필 400투수·계수 13·스케일) **완전 일치**, 최대 절대차 `0.000e+00` |
| physmix 예측 대조 | `verify_submission --compare`: **RMS차 0.000000 · 최대 0.000000** |
| tm3L 재빌드 | 5개 엔트리 **전부 SHA-256 비트 일치** |
| blendD4 조립 | 91개 엔트리 중 **88개 비트 일치** (차이 3개는 아래 참조) |
| **최종 예측 대조** | 245,789행 전량에서 원 제출물과 **RMS차 `0.000000` · 최대 `0.000000`** (평균 0.49561 vs 0.49561) = **원소 단위 동일** |
| `scan_probe_provenance` | **BLOCK 0건** (REVIEW 8건은 전부 부정문·준수 선언문) |
| `verify_submission --proxy 245789` | 전 항목 통과 · 정상경로 195.9초 / 서버핀 21.0초 (서버 제한 600초) |
| `sweep/test_regulation.py` | 행 독립성·타깃 셔플 플라시보·폴드 분리 등 **전부 통과** |

### 5.1 남은 차이 3개의 정체 (전부 예측에 영향 없음)

1. **`model/legs/physmix/script.py`** — 서빙 템플릿이 8월 이후 확장됐다(326행 → 742행).
   추가분은 ENS-10/ENS-11용 `dt_correction`·`c11_correction` 분기이며, 이 분기는
   메타데이터에 해당 키(`dt_corrector`·`c11_corrector`)가 있을 때만 실행된다.
   physmix 메타데이터에는 그 키가 없으므로 **실행 경로가 동일**하다.
   위 표의 예측 대조 `RMS 0.000000`이 이를 실측으로 증명한다.
2. **`model/legs/physmix/model/metadata.json`** — `provenance` 문장 하나만 다르다.
   `"…refit for THIS blend on the same protocol"` → `"…on the fit-2023 -> val-2024 protocol"`.
   이는 디스클로저 §3의 시정 조치로 문구를 정확하게 고친 것이다. 숫자는 동일(1.04, −0.01).
3. **`model/legs/MANIFEST.json`** — 위 두 파일의 SHA를 기록하므로 따라서 달라진다.

### 5.2 🔎 재현하면서 우리가 발견한 것 — 공유 아티팩트 드리프트

이 재현 작업 자체가 결함을 하나 잡아냈다. 기록을 남긴다.

`physmix`의 물리 보정기는 트랙맨 엔티티 매칭 결과 `results/trackman/matches.csv`의
**Tier 1** 투수만 쓴다. 그런데 2026-08-30의 별개 연구 라운드(트랙맨 재설계, GMM 기반 Tier
재컷)가 **같은 경로를 덮어썼다.** 투수↔트랙맨 ID 배정 자체는 바뀌지 않았지만
`tier` 컬럼이 바뀌어 **Tier 1이 407명 → 179명**으로 줄었다.

그 상태로 재현하면 보정기 프로필이 400투수 → 179투수가 되고 계수·스케일이 전부 달라진다.
즉 **HEAD 코드로는 제출본이 재현되지 않았다.** 원인을 git 이력에서 특정한 뒤,
제출본이 실제로 쓴 판(커밋 `17fc655`, Tier 1 = 407명)을 이 패키지에 고정 동봉했다
(`results/trackman/matches.csv`, 수집기 `phase3/tools/collect_package.py`의 `ARTIFACTS`).
그 상태에서 위 5절의 비트 일치가 나온다.

**교훈**: 연구 라운드가 공유 아티팩트를 제자리에서 덮어쓰면 배포본 재현이 조용히 깨진다.
아티팩트에도 코드와 같은 버전 고정이 필요하다.

### 5.3 환경 차이에 대한 메모

원 제출본은 2026-08-08에, 이 재현은 2026-09-03에 만들었다. 그 사이 로컬 pandas가
**2.3.3 → 3.0.0**으로 올라갔는데도 LightGBM 텍스트 부스터와 ExtraTrees joblib이
비트 단위로 같았다. 학습 경로에 pandas 버전 의존성이 없다는 실측 근거다.

---

## 6. 규정 준수

### 6.1 규칙 §2-4 (추론 코드의 평가 데이터 예측 원칙)

> "평가 데이터의 다른 행이나 전체 평가 데이터의 분포를 이용해 특정 행의 예측값을 보정하거나
> 생성하는 방식은 정상적인 추론 절차로 인정되지 않습니다."

세 문장 각각에 기계 검사가 하나씩 붙어 있다.

| 규칙 | 강제 수단 | 결과 |
|---|---|---|
| 각 행은 독립적인 예측 대상 | `sweep/test_regulation.py::test_row_independence` — 단일 행 예측 == 전체 프레임의 해당 행, 그리고 10% 부분집합 == 전체 | 통과 |
| 입력 변수 + 공식 학습 데이터만 사용 | 외부 데이터·사전학습 가중치 0건. 입력은 공식 `train.csv`·`test.csv`·`trackman_history.csv`(2019–2024)뿐 | — |
| 다른 행/전체 분포로 보정 금지 | `submission/verify_submission.py`의 AST 검사 — `mean/groupby/value_counts/transform/rolling/expanding/cumsum/rank/shift` 호출 부재 강제 | 통과 |

이력 피처는 전부 **train에서 만든 고정 룩업 + 행 단위 조회** 형태다. 예를 들어 당해 시즌
성적 분해(`is_*`)는 공식 `asof_pitcher_n`(커리어 누적)에서 동봉 상수(그 투수의 직전 시즌 말
누적)를 빼는 행 단위 뺄셈이다. test의 다른 행을 보지 않는다.

`tm3L` 레그는 팀원 원본의 서빙 코드가 이 원칙을 세 곳에서 위반하고 있었다
(`league_level(test)` 배치 추정 · `recalibrate()` 배치 평균 이동 · `nanmedian(test 컬럼)`).
학습된 모델은 바이트 그대로 두고 **이 3종만 train에서 계산한 동봉 상수로 치환**했다
(`submission/build_tm3_rowwise.py`; 원본 `build()` 대비 파리티 `0.000e+00`).

### 6.2 리더베이스 역산 상수

**없다.** 기계 검사기 `submission/scan_probe_provenance.py`가 제출 zip을 열어 역산 전용
metadata 키·알려진 역산 상수값·provenance 문자열 서명을 찾는다. 최종 산출물은 **BLOCK 0건**이다.
경위와 시정 전문은 [`COMPLIANCE_DISCLOSURE.md`](COMPLIANCE_DISCLOSURE.md).

### 6.3 누수 방지

as-of 통계를 전 시즌에 한 번에 계산하면 검증 시즌 행이 같은 시즌의 미래 타깃을 본다.
폴드별 재계산 + 검증 시즌 엔티티 통계 동결로 수정했고, **타깃 셔플 플라시보**로 증명했다 —
검증 라벨을 무작위로 섞어도 피처가 비트 단위로 불변이다(`test_regulation.py`).

---

## 7. 시드·하이퍼파라미터 (physmix 레그)

> 규칙 부연: "코드에 Random Seed, Hyperparameter 등 코드 재현을 위한 값들을 꼭 기재"

전부 `submission/build_lgbm_ensemble.py` 상단과 `sweep/lgbm_family.py`에 상수로 박혀 있다.

**학습 시즌 = 2024 단독** (253,507행). 전 시즌 풀링 대비 로컬 +530점 — 이 대회 최대 레버다.
단, 이것은 법칙이 아니라 레짐 베팅이다(§8 참조).

**공통 LightGBM 파라미터** (`sweep/lgbm_family.py::BASE_PARAMS`)

```
objective=regression   metric=l2   boosting=gbdt          ← Brier가 곧 제곱오차이므로 지표 직접 최소화
learning_rate=0.03     feature_fraction=0.85
lambda_l1=0.5          lambda_l2=8.0                      ← 강정칙
num_threads=4
```

**멤버 5개** (`MEMBERS_ENS7` — 가중은 `sweep/ens4_weights.py`의 Caruana 그리디를 동결)

| 이름 | 종류 | 가중 | 학습 범위 | 입력 | 하이퍼파라미터 · 시드 |
|---|---|---:|---|---|---|
| `nn_lin_bis` | 선형 | 0.475 | 2024 | 78컬럼 | `hidden=() dropout=0 epochs=60 lr=1e-3 wd=1e-4` · **seed 0** |
| `et_l100_d28` | ExtraTrees | 0.250 | 2024 | 57컬럼 | `n_estimators=200 min_samples_leaf=100 max_depth=28 max_features=0.7` · **random_state 9** |
| `allraw_l15` | LightGBM | 0.125 | 전 시즌 | 57컬럼 | `num_leaves=15 min_data_in_leaf=1000 num_iterations=400` · **seed 57** |
| `june_l15` | LightGBM | 0.100 | 2024 | 57컬럼 | 위와 동일 · **seed 57** |
| `june_l7` | LightGBM | 0.050 | 2024 | 57컬럼 | `num_leaves=7 min_data_in_leaf=1500 num_iterations=500` · **seed 49** |

**보정기** — 트랙맨 `w=0.5` (ridge λ=1000) · physmix `w=0.5` (ridge λ=1000, 계수 13)
**당해 시즌 분해** `K_IS = 100` · **pEB** `kappa = 100`, `league = 0.535228220260797`
**캘리브레이션** `center=0.5, slope=1.04, shift=-0.01`
 — `fit ≤2023 → val 2024` 프로토콜의 로컬 재적합값 `(1.035270, −0.010359)`과 반올림 범위에서 일치한다.
   리더보드 입력이 전혀 없다. 더 높은 점수의 `calP`(1015.70)를 쓰지 않은 이유가 이것이다.

**피처 78개** = 공식 44 + 범주형 3 + 파생 6 + 당해시즌 분해 4 + 타자 당해분 6 + pEB 3 + cxp 12.
트리 멤버는 앞 57컬럼만 본다.
**`season`·`pitcher_id`·`batter_id`는 제거한다** — 시즌이 갈수록 분포가 이동해 2025 행이
train 어디에도 없는 구간에 놓인다(그런 컬럼 하나가 2025에서 −192점인 것을 실측했다).

---

## 8. 알려진 한계

- **레짐 의존성이 최대 미측정 리스크다.** "최신 1시즌 학습"과 동결 캘리 상수는 2025가 안정
  레짐이라는 데 건 베팅이다. `fit 2022 → val 2023`에서는 부호가 완전히 뒤집혀 최신 1시즌이
  꼴찌였다(−1215 vs −467). 진폭 ±400~750점. **2025는 맞춘 것이지 통제한 것이 아니다.**
- **앙상블 다양성은 정보가 고정되면 포화한다.** 유능한 레그들의 정직한 불일치(RMS)는
  0.024~0.029에 5번 수렴했다. 그 지점에서 추가 점수 추구를 산수로 종료했다.
- **도넛 기하·트랙맨 매칭은 현상 검증까지만 갔다.** 게이트를 통과하지 못한 것은 싣지 않았다.

자세한 서술은 [`docs/SOLUTION_OUTLINE.md`](docs/SOLUTION_OUTLINE.md)(발표 원고 전문).

---

## 9. 파일 지도

```
sweep/                  학습·검증 모듈 24개 (평면 sibling import — 재배치 금지)
  real_data.py            로더 + 규정 안전 피처층 + 시간분할 폴드
  lgbm_family.py          june853 계열 LGBM 레시피 + 피처 53
  inseason.py             당해 시즌 성적 분해(is4) — 단일 최대 신호
  inseason_full.py        타자판 분해(bis)
  nn_member.py            선형 멤버
  eb_carrier.py           경험적 베이즈 블록(pEB)
  interaction_carrier.py  카운트 × 당해 이탈 곱(cxp)
  tm_member.py            트랙맨 프로필 보정기
  tm_physmix.py           트랙맨 구종군 물리 × 행 갱신 mix 보정기
  trackman.py             투수-시즌 엔티티 매칭 (헝가리안)
  trackman_v2.py          매칭 v2 + 사후 연결 (provenance)
  test_regulation.py      행 독립성·파리티·폴드 분리 기계 검사
submission/             빌더·검증기 11개
results/                재현에 필요한 입력 아티팩트 (§5.2 참조)
  trackman/matches.csv    커밋 17fc655 고정 — 제출본이 쓴 판
  phase3/profiles.json    트랙맨 투수 프로필
partners/               팀원 저장소 (커밋 고정 복사)
  lg-aimers-ysy/          @ e76e6f1 — clookup·cmoe 학습 코드 + 학습 산출물
  LG_Aimers_JTT/          @ d7ecbbc — trackman_pipeline v3 (tm3L 원본 모델)
docs/                   규칙 전사·발표 원고·준수 근거표·과제 스펙
COMPLIANCE_DISCLOSURE.md  규정 저촉 경위와 시정 (먼저 읽을 것)
SHA256SUMS.txt          이 패키지 전 파일의 SHA-256
```
