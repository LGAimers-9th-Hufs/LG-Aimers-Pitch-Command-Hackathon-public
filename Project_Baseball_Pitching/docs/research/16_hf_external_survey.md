# 16. "외부의 힘" — HF 사전학습 표형 모델 전면 재조사 + TabICLv2 파일럿 (2026-08-30)

> 유저 요청: "이론 최적화 말고 외부의 힘(HF 등)을 받아보자. API 금지, 규정 준수."
> 위치 확인: 이 채널은 라운드5(TabPFN-3 파일럿 3구성 전패, `14_round5_new_methods.md` §0-b)와
> 라운드6(Beyond-IID 계보 킬, `15_round6_latest_methods.md`)으로 이미 닫혀 있었다. 본 라운드는
> ① 2026-08-30 기준 웹 재조사로 "판을 뒤집는 신규"가 없는지 확인하고 ② 유일한 라이선스
> 청정 미실측 후보 TabICLv2를 실측해 채널을 완결한다. 웹 조사는 병렬 에이전트 2기
> (인벤토리/증거), 전 항목 fetch 검증.

## §1. 2026-08 인벤토리 — 로컬 가중치로 쓸 수 있는 표형 FM

| 모델 | 라이선스/게이트 | 스케일 | 판정 |
|---|---|---|---|
| TabPFN-2.5 / 2.6 / **3** (Prior Labs) | **전부 PriorLabs 로그인 게이트 + 비상업** ("model, derivatives, and outputs cannot be used for any commercial or production purpose") | v3: 1M×200 (H100) | **사용 불가 확정.** 대회 제출·주최 코드 검증·zip 재배포 전부 저촉. 로컬 `~/.cache/tabpfn/auth_token` 실존 = 패키지가 런타임 토큰을 강제한다는 물증(인터넷 차단 서버 리스크). 라운드5 "license-cleared" 메모는 라운드6에 이어 재차 오판 확인 |
| TabPFN v2 공개판 (`Prior-Labs/TabPFN-v2-clf`) | Apache계+표기, 비게이트 | ~10K행 | 컨텍스트 1만행 — 1.475M행 대비 실익 없음 (라운드3 판정 유지) |
| **TabICLv2** (`jingang/TabICL`, ckpt 2026-02-12, 110MB) | **BSD-3 · 비게이트 · 순수 합성 사전학습** | ~500K 컨텍스트 (1M은 VRAM 50GB) | **유일 실행 후보** → §3 파일럿. pip `tabicl==2.1.1`(torch 2.7.1 호환), `model_path=` 완전 오프라인, 공식 파인튜닝(`tabicl[finetune]`) |
| TabDPT-Turbo v1.2 (`Layer6/TabDPT`) | Apache-2.0, 비게이트 | 행별 faiss KNN 컨텍스트(행 캡 없음) | 보류 — 실데이터(OpenML 123종, 스포츠 없음 확인) 사전학습이라 "외부 데이터" 스토리 열위 + retrieval 계보는 시간 시프트에서 최악(TabReD/LAMDA) + Beyond-IID 킬 중복 |
| Mitra (`autogluon/mitra-classifier`) | Apache-2.0 | 10K행 캡 | 소규모 전용 — 탈락 |
| LimiX (`stable-ai/LimiX-16M`) | 가중치 "상업은 별도 승인" | — | 라이선스 탈락 (라운드5 판정 유지) |
| Google TabFM (`google/tabfm-1.0.0-pytorch`, 2026-06) | `tabfm-non-commercial-v1.0` | ≤500피처 | 라이선스 탈락 |
| Nori (`Synthefy/Nori`, 2026-06) | Apache-2.0, 합성 | 회귀 전용 | 와일드카드만(Brier=MSE 트릭) — TabICL 사망 시 무의미라 미실행 |
| TabuLa-8B / CARTE / TabSTAR / ContextTab | Llama3 / — / — / 실데이터 | 8B LLM·문자열 지향 | 속도·오염·적합성 탈락 |
| xRFM (ICLR 2026) | 사전학습 아님(커널) | 무제한 | "외부의 힘" 범위 밖, NN-계열 d는 TabM 실측(0.027~0.029 = d 벽 수렴)이 이미 대변 |
| AutoGluon 1.6.x `extreme` (2026-08-05) | FM 4종 포트폴리오 | **FM은 ≤500K 캡** | 1.475M행에선 FM 미가동 → 자동화 우회 불가 |

## §2. 대규모+시간시프트 실증 — 문헌 최종 정리 (전부 fetch 검증)

- **Beyond-IID (arXiv:2606.30410, 2026-06-29, TabPFN/AutoGluon 진영 저자들)**: "TFMs fail to
  compete with traditional tree-based and deep learning models on non-IID (temporal, grouped),
  large-scale ... datasets." TabPFN-2.6·TabDPT는 100K행 초과 시 아예 RF로 대체 평가.
  TabICLv2는 대형에서 돌긴 했으나 튜닝 GBDT+RealMLP에 패배. 우리 세팅(1.475M행·연단위
  시프트·엔지니어드 피처)이 정확히 TFM 패배 구간.
- **TabReD (arXiv:2406.19380)**: XGBoost 평균순위 2.2 > MLP-PLR 2.5 > LGBM 2.6; FM 부재;
  retrieval류는 시간 시프트에서 특히 붕괴. 단 시간분할이 격차를 압축(GBDT 우위 ~1%)
  — "FM 레그가 참패하진 않는다"의 근거이기도 했으나, 우리 v3 실측은 참패였다.
- **LAMDA (arXiv:2502.20260, ICML 2025)**: recency 컨텍스트(최근 1만행)가 TabPFNv2를
  개선하나 우승은 못 시킴. 트리가 가장 안정.
- **Realistic-Eval (arXiv:2505.16226)**: TabPFN v2는 "small-scale, covariate-shifted,
  class-balanced 전용" — 열린 환경에선 트리 우세.
- 벤더 반론의 한계: TabPFN-3 "1M행 튜닝 GBDT 격파"(arXiv:2605.13986)는 IID 벤치·H100·
  비상업 가중치·독립 재현 0건. TabICLv2 논문(arXiv:2602.11139) 스스로 "distribution
  shifts ... left to future work".
- 대회 전례(2025-26): ML Contests 집계 — 우승 솔루션 중 TabPFN 1건(소규모 의료, 피처
  생성기 역할). Kaggle S6E2 1위가 TabICL을 ~150 OOF 레그 중 소가중 멤버로 채택(가장
  근접한 ≥10만행 전례). ≥10만행에서 FM이 하중을 받친 검증 사례 0건.
- 야구 사전학습 모델: pitch-level 공개 모델 0건(HF 포함). 존재하는 것은 원명 Statcast
  컬럼 요구 취미 repo뿐 — 익명화 78피처에 접속 불가. 종결.

## §3. TabICLv2 파일럿 (`sweep/tabicl_pilot.py`, `results/tabicl/`)

- 프로토콜 = `tabpfn3_pilot.py` 동일(쌍대 LGBM·프록시 30K 쿼리·d vs 캐시 레그·속도 외삽)
  + **§5 행 독립 프로브**(1행 단독 vs 배치 예측 편차 — column-then-row attention이 쿼리
  행끼리 섞으면 점수 무관 즉사).
- 사전 등록: 구성 ① c23 100K(v3 직접 비교) ② c24x 100K(2024에서 프록시 블록 제외 —
  ICL 메모리제이션 왜곡 차단). n_est=4. **G1**: 쌍대 delta ≥ −1·SE (v3는 −2.7~−4.3 SE로
  탈락). **G0**: 245,789행 투영 ≤ ~7분. 통과 시에만 표준 3-fold 로컬 → 전이 →
  `E₅=(4026.56+S)/5+(59.42+64000·d²)/1.4295`, E ≥ 1058.60+5일 때만 블렌드 1슬롯.

### 실측 결과 (RTX 3080 10GB, `results/tabicl/pilot_c23_e4.json` · `pilot_c23k100_q8k.json`)

| 항목 | 실측 | 판정 |
|---|---|---|
| fit (ctx 100K) | 4.4~4.7s, VRAM 108MB | — |
| **속도 G0** | **18.8~19.2s/1k → 245,789행 투영 77~79분** (VRAM peak 4.9GB) | **킬** — 10분 한도의 ~8×. n_est 4→1이어도 ~19분, ctx 반감 병행해도 경계선인데 L4는 3080보다 대역폭 열세 |
| **§5 행 독립** | 같은 행 단독 vs 50행 배치 = **4.82e-02** · 2000행 배치 vs 50행 = 2.98e-02 · **같은 배치 반복 = 0.00e+00** | **킬** — 비결정성이 아니라 **transductive 구조**(쿼리 배치 조성이 예측을 결정론적으로 변경 = test 행 간 참조). SAINT를 즉사시킨 클래스, DACON "단일 행 == 전체" 검사 그대로 걸림 |
| **정확도 G1** (8K 쿼리) | LGBM(동일 ctx) 445.85 vs TabICL **579.02** → 쌍대 **+133.17 (SE 225.85)** | **통과** — v3의 −433~−2120(−2.7~−4.3 SE)과 정반대. 이 데이터에서 컨텍스트 대등 GBDT와 동급을 찍은 **첫 FM** (mean 0.506, sd 0.036로 스프레드는 과소) |
| 구성 ② c24x | 미실행 | 킬 2건(속도·§5)이 컨텍스트 무관 아키텍처 속성 → 판정 불변이라 생략 |
| d vs 캐시 레그 | 미측정 (30K 런은 속도 abort) | moot |

## §4. 판정 — 채널 최종 종결 (4중 봉인)

"외부의 힘(사전학습 표형 FM)" 채널은 서로 독립인 4개 축으로 봉인된다:

1. **라이선스/게이트**: 성능 상위 계열(TabPFN 2.5/2.6/3, Google TabFM, LimiX) 전부
   비상업·로그인 게이트 — 제출물 사용 불가.
2. **§5 transductive**: 유일 청정 후보 TabICLv2가 구조상 쿼리 배치를 섞는다(실측 4.8e-2,
   결정론). 행별 예측 전환은 시간이 폭발하고, 라이브러리 내부 수술은 검증 불가 리스크.
3. **10분 추론 한도**: v3 11분 · TabICLv2 77~79분(로컬 3080 기준) — 둘 다 초과.
4. **정확도 상한**: Beyond-IID 계보 킬은 TabICLv2 실측(쌍대 +133±226)이 **부분 반증**
   — 단, "컨텍스트 대등 LGBM과 동급"일 뿐이고 배포 가능 컨텍스트(≤~30만행)로는 1.475M행
   풀스택 레그(S 1009~1049)에 수백 점 미달이라, ②③이 없어도 E 산수는 음수였다.

**결론: 슬롯 0 소모, 제출 없음, 채널 종결.** 남긴 것 — ① TabICLv2가 이 데이터에서 FM
최초로 정확도 게이트를 통과했다는 실측(후속 대회에서 §5류 제약·시간 한도가 없다면 재평가
가치 있음), ② `sweep/tabicl_pilot.py`의 행 독립 프로브(단독 vs 배치 vs 반복 3중 대조)는
transductive 모델 판별 표준 절차로 재사용 가능, ③ ckpt는 `data/tabicl/`(gitignored) 보존.
