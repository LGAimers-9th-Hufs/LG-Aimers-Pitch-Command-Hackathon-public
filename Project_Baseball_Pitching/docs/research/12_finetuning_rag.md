# 12. 파인튜닝·RAG 타당성 판정 — 대회 합법 형태와 재론 방지 기록

작성 2026-08-06. 방법: 2방향 병렬 웹조사(R5 RAG/retrieval, R6 파인튜닝), 인용 전수 실존 확인(arXiv abs/HF 모델카드/GitHub/DOI). 목적은 열광이 아니라 **판정** — 각 경로의 가능/조건부/불가와 근거를 남겨 재론을 방지한다.
관련: `13_trackman_train_usage.md` §6(N12 초안 — 본 문서 §2가 설계 v2로 개정), `11_other_domains_round3.md`, `docs/log/14_baseline_gap.md`(콜드스타트 비중 4~6% — kNN 기대치 캡의 근거), `00_KILL_LIST.md`.

## §0 판정 요약

| 경로 | 판정 | 한 줄 근거 |
|---|---|---|
| RAG = **train-only frozen bank kNN** | **가능 — 유일한 합법 RAG 형태, 설계 v2 확정(§2)** | 뱅크=train 고정물+행 단위 쿼리 = z_table의 일반화. 우승 전례 2건 + 산업 20년(PECOTA) |
| TabPFN-2.5 / TabPFN-3 | **불가 (라이선스 단독 종결)** | 가중치 라이선스가 상업/프로덕션/경쟁 벤치마킹 금지 + 패키지 로그인 토큰 강제 |
| TabPFN v2 제로샷/컨텍스트 | 조건부 — **실익 낮음** | 라이선스 청정(29MB ckpt, 오프라인 로드 가능)이나 컨텍스트 10k 상한 + 시간이동 반증 |
| TabPFN 파인튜닝 | **불가 (실익 기준 종결)** | 라이선스-청정 모델로 100만행·시간분할에서 GBDT를 이긴 실증 0건 + 명시적 반증 논문 |
| **LLM 파인튜닝(LoRA/SFT)** | **불가 확정 (재론 금지)** | few-shot 밖 실증 0 + 2026 재평가가 "일반화는 오염 아티팩트" 판정. F13 사망 유지·강화 |
| **TabM → 순수 numpy forward 배포** | **가능 — 파인튜닝 계열 유일 실전 등급(§4, N34)** | Apache-2.0, 동형 벤치(TabReD) DL 1위, torch 없이 서빙 가능(requirements 불변) |
| TabICL v2 50k-컨텍스트 제로샷 | 조건부(차선, N35) | BSD-3·오프라인 로드 가능하나 temporal shift 반증 + 콜드스타트 무기여 — 로컬 게이트 선행 절대 |
| D4 자기증류(HGB→numpy MLP) | 가능하나 실익 최하 | sklearn이 이미 네이티브 배포 — 남는 가치는 fallback 승격뿐 |
| wheel 동봉(오프라인 설치) | 가능(전례 확립) — 단 정찰 선행 | Kaggle 표준 패턴. pip 없이 wheel unzip + sys.path가 더 안전 |

## §1 RAG의 합법 형태 — 왜 train-only frozen bank인가

- 규정 §5는 test 행 간 상호참조를 금지하지만 **train(2019–24) 유래 고정 룩업의 zip 동봉 + 행 단위 조회는 명시적 합법**(SUBMISSION_COMPLIANCE의 z_table 전례). M31(ModernNCA) 기각 사유는 "검색 뱅크가 test를 참조하면 위반"이었으므로 train-only 뱅크는 그 기각에 저촉되지 않는다.
- **z_table과의 차이 = 존재 이유**: z_table 키는 pitcher_id → 2025 미등장 신인은 default로 추락(정보 0). kNN 키는 **프로필 공간 좌표**(asof 재조합 12~16dim) → 신인의 행 값으로 "비슷한 기존 투수들"의 수축 로짓을 빌려온다. 투수 경계를 넘는 유사도.
- **기대치 캡(문서 14)**: 2025 콜드스타트 비중은 4~6%뿐 — 순수 콜드 채널의 상한은 수십 점. 이 캡을 뚫는 것이 §2의 잔차 뱅크(N32, 전 층 headroom)다.
- 전례(전부 실존 확인): Kaggle **Home Credit 2018 1위** — `neighbors_target_mean_500`(EXT_SOURCE 4차원 공간 500-NN 타깃 평균, 피처 중요도 3위); **Otto 2015 1위** — kNN 메타피처 다수(k=2~102, 5-fold OOF); **PECOTA**(2003~) — 신인·마이너리거를 최유사 선수 100명의 실제 후속 성적으로 예측; **CARMELO** — 양의 유사도 전원 감쇠 가중. 학술: TabR(arXiv:2307.14338), ModernNCA(2407.03257, ICLR'25)가 "이웃 신호는 1급"임을 확립.
- **반대 실증(설계 제약으로 흡수)**: TabReD(2406.19380, ICLR'25) + Cai&Ye(2502.20260, ICML'25) — 검색 기반 방법은 랜덤 분할에서 최대 이득, **시간 시프트 하에서 유의하게 하락**(이웃 노후화). 대응 = 시즌 내 z-score 좌표(era drift 소거) + recency 가중 ρ^Δs + forward 게이트 검증. kNN-LM(1911.00172)의 λ-보간·datastore 교체 관점은 참고(테이블 직접 전례는 부재 — 정직 표기).

## §2 N12 설계 v2 — 문서 13 §6 초안에 대한 교정 10건 (R5 문헌 검토 결과)

1. **신뢰도는 거리가 아니라 평균 가중치에**: w_i ∝ K(d_i)·cred_i·ρ^Δseason. 거리에 섞으면 "저이력 투수는 모두와 멀다"는 왜곡 기하 — 콜드 쿼리가 이웃을 못 빌림(설계 목적과 충돌).
2. **쿼리 좌표도 credibility 수축**: 신인의 asof 좌표는 극단으로 튐 → 자기 표본수 기준 리그 평균으로 수축 후 검색. 부수 효과: 극초반 신인은 보정이 0으로 **우아하게 퇴화**(현 default 행동과 연속).
3. **좌표 = 시즌 내 z-score**(era drift 1차 방어).
4. **k 단일값 금지**: {10, 25, 50, 100} + 적응 커널(k/2번째 이웃 거리) 스윕. 이론(Samworth arXiv:1101.5783, Ann.Stat. 2012): k* ≍ n^{4/(d+4)}, n=2.6k·d_eff 4–8 → k* 14–51. Brier(확률)는 분류보다 큰 k 선호(CARMELO는 수백 명 감쇠 가중).
5. **하드 오프셋 금지(269 사태 경로)**: knn_logit/knn_dist/knn_std를 **HGB 입력 피처**로(모델이 신뢰 시점을 학습). recal 5항(KN) 형태는 보조 arm으로 병행 게이트.
6. **잔차 뱅크가 상한이 더 높다**(→N32): knn_logit은 베테랑 층에서 z_table과 공선(정보 증분이 콜드 층 국한). 이웃 **잔차**는 "현 모델의 국소 편향"이라는 직교 신호 — 전 층 headroom.
7. **자기참조 누출 차단(필수)**: 폴드별 뱅크는 그 폴드 train 시즌만으로 재구축(배포 형상 복제) + leave-own-pitcher-out. Otto/Home Credit의 OOF 규율 — 안 지키면 게이트가 거짓말한다.
8. **콜드 경로 마스킹 학습(초안 최대 구멍, DropoutNet NeurIPS 2017 번역)**: 피처만 추가하면 train에 "z=default 행"이 희소해 HGB가 knn 피처 사용법을 못 배움. 학습 시 무작위 투수 부분집합의 z→default 마스킹 + 해당 투수 제외 knn 계산 증강으로 2025 신인 상황을 시뮬레이션.
9. **뱅크 단위 = 투수-시즌(~2.6k) 유지**(커리어는 에이징을 뭉갬; game_type 분할은 F 파단 전례와 충돌 — game_type은 모델 피처 소관).
10. **런타임**: float32 청크 GEMM(2.6k 뱅크 ≈ 20.5 GFLOP → 수 초; 50k 뱅크도 30–60초). **float16 금지**(numpy f16은 BLAS 미경유 — 수십 배 느려짐; 뱅크 166KB라 양자화 실익 0). 동봉은 npz(allow_pickle=False).
- 구현 주의(row-independence 비트 검사): build_features 안에서는 GEMM 전개식(‖x‖²+‖b‖²−2xb) 대신 **(q−B)² 직접합**(행별 elementwise — 배치 크기와 무관하게 결정론적)을 서브청크로. GEMM은 recal-KN 경로에서만.
- **사전 진단(반나절, 채널 생사 판정)**: refV24 OOF 잔차에 대해 "이웃 평균 잔차 vs 자기 잔차"의 held-out 상관(Hidden Heterogeneity 진단, arXiv:2202.01840 — 모델이 분해 못한 하위모집단이 클수록 국소 신호에 잔여 가치. 전역 isotonic이 죽었는데 국소 이웃 신호가 살아 있을 수 있는 이유의 문헌적 설명). ~0이면 kNN 채널 전체 조기 폐기.

## §3 신규 kNN arms

| arm | 무엇 | 근거 | 층 | 비용 | 게이트 |
|---|---|---|---|---|---|
| **N32 knn_residual** | 뱅크 값 = 현 모델 OOF 잔차의 투수-시즌 수축 평균 → 이웃 가중 평균 잔차("이 프로필 유형에서 모델이 체계적으로 얼마나 빗나가나")를 피처/보정항으로. β는 게이트로 추정(하드 동결 금지 — K4 교훈) | Analog Ensemble(Delle Monache MWR 2013 — 기상 운영계 15년: 유사 상황의 과거 오차로 확률 보정 + 이웃 산포=불확실성), Diff-KNN(MAKE 2025, DOI 10.3390/make7040131), SBA(Bella 2009) | 전 층 국소 편향(베테랑 해상도 포함) | N12 인프라 위 +0.5일 | N12와 한 스윕(4피처 arm 포함 — 상호 잠식 확인). 주 지표 refV24 |
| **N33 rookie_comps_z** | 모델 구조 불변 외과 패치: z_table 조회 실패(2025 신인) 시에만 프로필 이웃 k명의 수축 로짓 가중 평균으로 default 대체. train 측도 동일 규칙로 재학습(입력 분포 일치). knn_dist 게이트(뱅크에서 멀면 default 유지 — Long-Tail Crisis arXiv:2503.22426의 "검색 가치는 뱅크 조밀 영역에 집중" 경고 흡수) | PECOTA/CARMELO(산업 표준), Ferrari Dacrema RecSys'19(잘 튜닝된 이웃 기법의 경쟁력) | ③콜드 전담 | 1일 | score_cold 전용 게이트(역방향 폴드에서 "처음 등장 투수" 층 분리 채점). **베테랑 행은 비트 불변 → 전체 리스크 구조적 캡** |
| N12 스윕 부속 | {dist,std}만 / {logit}만 / 전체 피처 조합 arm — "프로필 공간 희소 지대에서 base rate로 수축"하는 게이팅 신호 학습 | AnEn 산포, Long-Tail Crisis | 과신 제어 | 0 | N12 스윕 내 |
| (최후순위) lambda_blend | p = λ(d)·p_model + (1−λ(d))·p_knn — N12/N32/N33 전부 실패 시에만. λ 상수화 금지(전역 재캘리와 등가로 퇴화 — 기각 전례 인접) | kNN-LM — 단 테이블 직접 전례 부재 | ③ | 0.5일 | 거리 게이트형만 |

## §4 파인튜닝 판정 상세 (재론 방지 기록)

### 4.1 불가 확정 — LLM 파인튜닝(LoRA/SFT)
- 원전 TabLLM(arXiv:2210.10723, PMLR v206): 우위는 **~256행 이하 few-shot 한정**, 그 이상 GBDT 회복 — 논문 자인. 최신 후속 BoostLLM(arXiv:2605.06117, 2026)도 여전히 low-data 프레임, 100만행 실증 없음.
- 결정적 반증: **"The Illusion of Generalization in Tabular Language Models"(arXiv:2602.04031, 2026)** — Tabula-8B를 165개 데이터셋 재평가: 이진/범주 분류 lift ≈ 0, 고성능 사례는 train-test 오염 아티팩트.
- 우리 데이터는 LLM 이점 성립 조건(의미 있는 텍스트, 소수 샷)이 전무: 익명 정수 ID + 수치 47컬럼 + 147만 행. **F13 사망 판정 유지·강화.** 재론 조건: "익명 수치 100만행·시간분할에서 GBDT 상회"를 보인 새 실증 등장 시에만.

### 4.2 불가 — TabPFN 계열
- **TabPFN-2.5/3**: HF `Prior-Labs/tabpfn_2_5` 모델카드 — 라이선스가 상업/프로덕션/경쟁 벤치마킹 금지. 패키지 2.5+는 로그인 토큰 강제(인터넷 차단 서버에서 동작 불명). TabPFN-3(arXiv:2605.13986)은 1M행을 H100에서 GBDT 상회 주장하나 동일 비상업 — 성능 실증 자체가 전부 라이선스 불가 모델에 귀속.
- **TabPFN v2**(라이선스 청정, ckpt 29MB, 오프라인 로드 issue #174): 컨텍스트 10k 상한 — 147만 행의 0.7%. 컨텍스트 압축 문헌(TuneTables 2402.11137, LoCalPFN 2406.05207, 2502.02527)은 "무작위 서브샘플은 대형에서 저하, kNN-컨텍스트 필수"인데 행별 컨텍스트 재구성은 24.6만 forward → 10분 초과. 결정타: **arXiv 2506.08982 "On Finetuning Tabular Foundation Models" — "gradual temporal shift + rich feature에서 TabPFNv2 불안정, 기존 방법(GBDT) 우위"** = 우리 세팅 그대로. 파인튜닝 실증도 50k행까지만.
- 콜드스타트 정직 노트: TFM의 few-shot 이점이 우리 갭을 메우려면 asof 19종에 없는 정보원이 필요한데, 익명 정수 ID에서 ICL이 추가로 읽을 것이 없다. 그 채널의 정답은 여전히 트랙맨 매칭(N13)과 kNN(N12).

### 4.3 가능 — N34 tabm_numpy (파인튜닝 계열 유일 실전 등급)
- **무엇**: TabM(arXiv:2410.24210, yandex-research/tabm, Apache-2.0)을 로컬 torch로 학습 → 가중치 npz 덤프 → **script.py에서 순수 numpy forward**. TabM forward = 멤버별 elementwise 스케일 벡터 × 공유 Linear → ReLU → 멤버 출력 평균(선택적 piecewise-linear 임베딩도 bin-edge 룩업+선형보간 = numpy 구현 가능). **requirements.txt 3줄 불변(D-08 준수), 커스텀 pickle 불요.** 24.6만 행 × (k=32, hidden 512, 3층) ≈ CPU numpy ~2분.
- **근거**: TabReD(2406.19380 — 시간분할 산업 데이터 8종, **우리와 동형 세팅**)에서 TabM이 DL 중 최고; TabArena(2506.16791) 개별 모델 상위 = TabM·LightGBM·RealMLP. 정직한 기대치: TabReD 결론은 "GBDT ≈ MLP+임베딩" — 대폭 우위가 아니라 (a) HGB와 오차 상관 낮은 **블렌드 다양성**, (b) M-arm 추가 비용 저렴이 가치.
- **게이트**: 하네스에 TabM-mini arm 추가 → refV24+역방향. 본명은 단독이 아니라 **GLM+HGB와의 블렌드 arm**(Ranjan&Gneiting 확률 평균). 통과 시에만 npz+numpy forward 서빙 이식.
- ModernNCA(retrieval형·배포 무거움)·RealMLP(TabM 하위호환) 제외 권고.

### 4.4 조건부 — N35 tabicl_zeroshot (차선)
- TabICL v2(arXiv:2502.05564, ICML 2025; BSD-3, 토큰 불요, model_path 오프라인 로드): 사전학습 컨텍스트 ~48–60k행. "TabArena에서 튜닝된 GBDT를 ~80% 데이터셋 상회, TabPFN-2.5보다 10배 빠름" 주장. **train 고정 서브샘플 50k(최근 시즌 가중)를 zip 동봉하는 배포형은 규정 합법.**
- 조건: ① 로컬 게이트에서 HGB 근처(격차 <0.5%)가 먼저 나와야 함(temporal shift 반증 + 콜드스타트 무기여로 사전확률 낮음) ② torch 필요 → §5 wheel 동봉 경로 — TabICL 게이트 통과 후에만 논의. ckpt 크기 미확인.

### 4.5 D4 자기증류 — 존치하되 최하 순위
- sklearn 모델은 서버에 이미 네이티브 배포 가능 → "배포 안전" 동기 소멸. 남는 가치 = fallback(순수 numpy 로지스틱)을 numpy MLP 증류물로 승격. 근거는 건재(DeepGBM KDD'19, Menon ICML'21 — soft label=저분산 Bayes 추정, 2508.20224 — teacher 캘리↔학생 품질)하나 refinement 게이트 개선 없으면 채택 금지(시드평균 기각과 동일 기준).

## §5 wheel 동봉·정찰 (torch가 필요해질 때만)
- Kaggle 오프라인 대회 표준 패턴 실존: wheel을 데이터로 동봉 → `pip install --no-index --find-links`. **상위 패턴: wheel은 zip이므로 압축 해제 후 `sys.path.insert` — pip/권한 리스크 제거.** torch CUDA wheel ~2.3–2.8GB(분리형 4–5GB) — 10GB 예산 내.
- 유일한 실질 불확실성 = 서버의 requirements.txt 처리 방식(사전 설치 환경 추정). **저비용 정찰**: requirements 불변 + script.py에 `try: import torch` 로그 + **예측 출력은 현행 제출물과 완전 동일**(점수 리스크 0) — 단 로그 가시성 여부를 팀 채널로 먼저 확인. 제출권 차감 리스크 때문에 N35가 로컬 게이트를 통과하기 전에는 집행 금지.

## §6 랭킹과 Phase B 연결

1. **N12 v2 + N32 + 스윕 부속** — 한 번의 인프라(뱅크·쿼리·마스킹 학습)로 3개 arm 동시 게이트. HH 사전 진단(반나절)이 관문. [Phase B S2]
2. **N33 rookie_comps_z** — S2 실패 시에도 단독 생존 가능(리스크 캡 구조).
3. **N34 tabm_numpy** — 독립 트랙(로컬 torch 학습). 블렌드 arm으로 게이트. [S5 병렬 옵션]
4. N35 tabicl — N34까지 끝난 뒤 여유 시.
5. 불가 확정(TabPFN-2.5/3·TabPFN FT·LLM FT)은 **KILL_LIST 등재 권고** — 근거 링크 포함.

## §7 인용 검증 상태
직접 확인: TabPFN-2.5(2511.08667)·HF 모델카드(라이선스)·TabPFN-v2-clf(29MB)·오프라인 issue #174 · 2506.08982 · TabPFN-3(2605.13986) · TabM(2410.24210)+GitHub · TabReD(2406.19380) · TabArena(2506.16791) · TabICL(2502.05564)+GitHub · LoCalPFN(2406.05207) · TuneTables(2402.11137) · TabLLM(2210.10723) · BoostLLM(2605.06117) · Illusion of Generalization(2602.04031) · 2508.20224 · Menon ICML'21 · DeepGBM KDD'19 · TabR(2307.14338) · ModernNCA(2407.03257) · Cai&Ye(2502.20260) · kNN-LM(1911.00172) · Long-Tail Crisis(2503.22426) · Samworth(1101.5783) · Conn&Li(1711.09200) · AnEn(MWR 2013) · 2103.04530 · Diff-KNN(DOI 리졸브; 본문 403 — 수치 83.5%는 스니펫 기반) · HH(2202.01840) · SBA(Springer 2009) · DropoutNet(NeurIPS 2017 PDF) · Ferrari Dacrema(1907.06902) · Kaggle 오프라인 wheel 전례.
미확인 표기: L4 실측 추론시간(전부 추정) · 서버 torch 유무 · TabICL ckpt 크기 · L40S 1.15초/2000행(블로그) · Home Credit 피처 세부(Kaggle 원문 JS — Medium 재구성 경유).
