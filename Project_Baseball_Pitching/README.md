# LG Aimers 9기 × LG 트윈스 해커톤 — 작업 저장소

투구 단위 **제구 성공 확률** 예측. train 2019~2024 → hidden test 2025.
지표 `Score = max(0, 100000 × (1 − Brier / (r(1−r))))`, 수료 기준 **549.51**.

---

## 지금 어디에 있나 (2026-09-03)

**대회 종료(09-02). 현재 단계 = Phase 3 코드·PPT 제출 (마감 09-07 10:00).**

| | 상태 |
|---|---|
| **규정 준수 팀 최고 = 제출 검증 대상** | **1058.6047851923** — `submission/dist/submit_blendD4.zip` |
| 팀 역대 최고 | 🚫 1075.8374148399 (`factor_tabm_router`) — **제출 금지**, 아래 참조 |
| 점수 추구 | **종결**(08-30). 전수 열거 4,082조합의 유일 양수 후보 blendF5도 실측 1056.79 |
| 현재 목표 | Phase 3 제출물 품질 — 재현 가능성 + 규정 준수 입증 |
| 대회 정보 | 라벨·지표·데이터·규정·제출환경 전부 확정 → `docs/HACKATHON_TASK.md` |
| Phase 3 | **`phase3/`** — 규칙 전사·패키지 수집기·재현 실측·발표 원고·규정 준수 공개 |

### Phase 3 제출물 만들기

```bash
python phase3/tools/collect_package.py     # -> phase3/dist/ABS_kkangtongzone_phase3_code.zip
```

재현 실측(09-03): physmix 모델 가중치 **비트 일치** · tm3L 5엔트리 **비트 일치** ·
blendD4 재조립 후 245,789행 예측 대조 **RMS 0.000000**. 상세 → `phase3/package_README.md` §5.

### 🚨 2026-08-28 컴플라이언스 사건 — 먼저 읽을 것

팀원 재감사가 우리 챔피언 계보(`submit_exact1` → `FINAL_submit_blend` → DSF 계열)에서
**리더보드 점수를 역산해 박은 상수**를 발견했다. 규정 별첨 §5 금지 목록의 마지막 항목
**"평가 데이터 전체를 보고 만든 사후 보정값"**에 해당하고, 우리
`docs/SUBMISSION_COMPLIANCE.md:57`이 이미 스스로 금지해 둔 행위였다.

⇒ **1035~1075 구간의 팀 제출물 전량이 사용 불가.** 경위 전문은
`phase3/COMPLIANCE_DISCLOSURE.md`, 대응 기록은 `docs/log/33_compliance_and_team_blend.md`.

**경계선**: 사전 등록한 후보를 제출해 점수 좋은 쪽을 남기는 것(**선택**)은 허용된다.
점수에서 최적값을 풀어 상수로 박는 것(**적합**)은 위반이다.

---

## 5분 만에 파악하기

1. **`docs/log/33_compliance_and_team_blend.md`** — 최신 진실. 사건 + 회복 경로 + 실측표
2. **`phase3/COMPLIANCE_DISCLOSURE.md`** — 무엇이 왜 위반이었고 어떻게 시정했나
3. **`docs/00_KILL_LIST.md`** — 죽은 아이디어 등록부. **새 아이디어 전에 먼저 검색할 것**
4. **`docs/01_DECISIONS.md`** — 되돌리기 어려운 결정과 근거 (D-01~D-61)
5. `docs/HACKATHON_TASK.md` — 확정 스펙

`docs/research/01~13`은 실데이터 도착 **이전**의 조사다. 상당수가 규정으로 무효화됐으니
반드시 킬 리스트로 걸러 읽을 것.

---

## 현재 산출물 = 팀 블렌드 blendD4

최종 제출물은 **서로 다른 파이프라인에서 나온 레그들의 확률 균등평균**이다.
근거는 추정이 아니라 정확한 항등식이다.

```
Score(p̄) = 평균Score + C · E_i[(p_i − p̄)²]        C = 1e5/(r(1−r)) ≈ 4·10⁵
기대점수 = 평균LB + 프록시이득 / 1.4295            ← ρ는 blendA3 실측(42.423/29.677)
```

| 레그 | 출처 | LB |
|---|---|---:|
| `clookup` | 팀원 ysy — clean_regime + 계층 잔차 룩업 | 1049.4562 |
| `cmoe` | 팀원 ysy — Challenger + 3-seed FactorTabM + TrackMan MoE | 1027.5280 |
| `physmix` | 본 저장소 — ENS-9 (LGBM/ET/선형 5멤버 + 보정기 + physmix) | 1009.2645 |
| `tm3L` | 팀원 JTT 트랙맨 v3 + 본 저장소의 §5 행독립 재빌드 | (역산 940.31) |

**가중치는 균등 1/4 고정.** 리더보드로 역산하지 않는다 — `w* = 0.5 + d/(2K)`가 `d`가 작을 때
0.5에 붙으므로 균등 가중은 학습 데이터만으로 정당화된다.

**항등식은 예측 도구로 검증됐다**: blendA3에서 ρ=1.4295를 실측한 뒤, 전수 열거(4,082조합)의
유일 양수 후보 blendF5의 기대 1059.20 대비 실측 1056.79 — **실현 분산이득이 예측의 94.6%**.
이 프로젝트에서 사전 예측이 맞은 유일한 계열이다(적합이 아니라 대수라서).

**1100은 산수로 닫혔다**: 도달하려면 `LB ≥ 1000` 이면서 기존 평균과의 `RMS ≥ 0.0344`인 새
레그가 필요한데, 유능한 레그들의 정직한 RMS는 0.024~0.029에 **5번 수렴**했다.

---

## 폴더 구조

```
docs/         스펙·결정·폐기 등록부 · research/(사전 조사 01~13) · log/(11~34, 33이 최신 진실)
phase3/       ★규칙 전사(RULES) · 패키지 수집기(tools/) · package_README · 발표 원고 · 규정 준수 공개
baseline/     주최측 베이스라인 노트북 2종 (제출 계약의 원본)
data/         대회 배포 데이터 (gitignore) — open.zip + data/*.csv
partners/     팀원 저장소 clone 3개 (gitignore, 2.9GB)
sweep/        실행 코드 ~75개. real_data.py(피처) · lgbm_family.py · inseason.py(is4) ·
              eb_carrier.py(ENS-9) · tm_physmix.py · trackman*.py · test_regulation.py
              dsf_*.py(08-13 추가) · _retired/(합성 시절)
submission/   ★scan_probe_provenance.py · ★leg_matrix.py · ★build_team_blend.py ·
              proxy_rms.py · build_lgbm_ensemble.py · verify_submission.py · dist/
results/      compliance_scan.json(119개 판정) · leg_matrix/(RMS 캐시) · ens*/ · dual/ · dsf*/
papers/       참고논문 (PDF는 gitignore, PAPERS.md만 추적)
paper/        논문 뼈대
```

⚠ `data/` · `sweep/` · `results/` · `paper/`의 **위치는 고정**이다.
`real_data.py`·`reporting.py`가 `parent.parent` 상대경로로 이들을 찾고, sweep 모듈은
평면 sibling import를 쓴다. 코드 폴더를 옮기거나 하위 폴더로 쪼개면 전부 깨진다.

---

## 실행 방법

```bash
cd Project_Baseball_Pitching

# 0) 제출 후보의 규정 위반 서명 검사 — 모든 제출 전 필수. BLOCK 0건이어야 한다
python submission/scan_probe_provenance.py submission/dist/submit_blendA3.zip

# 1) 규정 준수 · 파리티 검증 (~1분)
python sweep/test_regulation.py

# 2) 레그 간 불일치(RMS) 행렬 + 블렌드 채산성 — 제출 슬롯 0개
python submission/leg_matrix.py --auto --rows 30000 --audit --anchor cregime

# 3) 팀 블렌드 zip 조립 (레그는 바이트 그대로 담긴다)
python submission/build_team_blend.py --tag blendA3 \
  --leg cregime=partners/lg-aimers-ysy/model_artifacts/clean_regime/submission.zip \
  --leg cmoe=partners/lg-aimers-ysy/model_artifacts/clean_moe/submission.zip \
  --leg physmix=submission/dist/submit_ens9_physmix.zip

# 4) 전체 245,789행 실검증 (3레그 약 60초)
python submission/verify_submission.py submission/dist/submit_blendA3.zip --proxy 245789 --threads 6
```

`verify_submission.py`는 임시 폴더에 zip을 풀고 서버와 같은 방식으로 `script.py`를 실행한다.
**모델 파일을 일부러 손상시킨 폴백 경로까지 검사**한다 — `script.py` 실행 오류는 제출 횟수를
차감하므로, 어떤 경우에도 `submission.csv`가 나와야 하기 때문이다.

⚠ 팀원 저장소를 clone할 때는 반드시 `-c core.autocrlf=false`. 개행 변환이 LightGBM 텍스트
부스터를 파괴해 레그가 **무음으로 빠진다**(D-60 실사고).

---

## 절대 어기면 안 되는 것

1. **행 독립성(§5)** — test의 다른 행을 쓰는 피처·사후 보정 금지. 예측 배치의 평균을 보고
   확률을 이동시키는 것도 위반이다. 보정 상수는 train에서 산출해 **얼려서** 행 단위로 적용한다.
   `test_regulation.py::test_row_independence`가 이를 기계적으로 증명한다.
2. **리더보드 점수를 역산해 상수로 박지 않는다(§5)** — 그것이 08-28 사건의 원인이다.
   공개 점수는 **사전 등록된 후보 중 선택**에만 쓴다.
   `scan_probe_provenance.py`가 이를 기계적으로 강제한다.
3. **제출 zip 내부 레이아웃** — 최상위는 `model/` · `script.py` · `requirements.txt` 뿐.
   여분 폴더가 하나라도 있으면 설치 오류.
4. **`script.py`는 죽으면 안 된다** — 설치 실패는 제출 횟수 미차감이지만 실행 오류는 차감된다.
   생성되는 `script.py`의 콘솔 출력은 **전부 ASCII**로 둔다(비UTF8 로케일에서 크래시한 전례).
