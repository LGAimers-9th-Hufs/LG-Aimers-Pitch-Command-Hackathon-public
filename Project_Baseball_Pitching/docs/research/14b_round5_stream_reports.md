# 14b — 라운드 5 스트림 보고서 원문 보존 (2026-08-29)

> `14_round5_new_methods.md`의 근거 자료. 세 병렬 조사 에이전트(S4 내부 감사 · S1 아키텍처 ·
> S2 대회 레시피+KBO 규칙)의 최종 보고서를 그대로 보존한다 — 재감사 방지용 file:line 증거와
> 출처 링크가 여기에만 있다. 판정의 최신 상태는 14의 티어표(§0-b 킬 후기 포함)가 정본.

---

# [S4] 내부감사 — 미사용 정보축 보고

**전제(실측)**: 현 블렌드 = blendD4(clookup+cmoe+physmix+tm3L) 1058.60 (`docs/log/35:93`). 5번째 레그 스펙은 정직-d 기준으로 **S=1000→d≥0.0344 / S=1045→d≥0.0314** (`35:104-106`). 정직 d 사다리: 같은스택 튜닝 0.004–0.019 · 타저자 GBDT tm3L 0.0238(`35:19`) · 타저자 MLP 0.0261(`results/leg_matrix/summary.json:37`) · 최대 관측 cmoe|tm3L 0.0306. 로컬↔LB: physmix 로컬 897.77→LB 1009.26(`28:159`), ENS-4 877.82→984.08(`22:86`). **단일축은 물리적으로 부족 — 축 적층만이 답이다.**

## ① `sweep/pitch_align.py` — 실행됨, 채널 절반만 개통

- **무엇**: Tier1 투수의 등판을 (월,요일,투구수) 튜플로 정확 대응, 등판 내 train행(asof_pitcher_n순)↔TM행(pitch_no순) 1:1 정렬 (`pitch_align.py:2-15,96-112`).
- **실행 기록**: `results/ens11/pitch_align_audit.json` — 유일등판 **29,043 · 투구 923,503쌍 · 상태열 일치 99.19% · 커버 87.75%**. 구종군 일치 94.2%<99% kill로 `go:false`였으나 D-53에 confusion이 단일 셀(컷패스트볼 분류 차이)임을 근거로 **판정 변경 GO** (`27:80-88`). idx npz는 go=false라 **미저장**(현 디렉토리에 없음) — `ens11_response.py:59-60`의 `aligned_pairs()`가 인라인 재계산하므로 재실행 비용 ~수 분.
  *(세션 후기: `--force-save`로 916,049쌍 npz 실체화 완료 — `results/ens11/pitch_align_idx.npz`)*
- **개통된 것**: β 스칼라 반응모델 +0.97, encctx28 재적합은 V21 −7.8로 기각 (`27:105-115`).
- **미개통(=이 채널의 잔여)**: (a) **물리→실패유형(4클래스) 반응** — `27:88`에 "미구현"으로 명시된 그대로 남음. (b) **특권정보 증류(LUPI)**: 실물리 916k쌍을 입력으로 보는 teacher → 물리 없는 student가 로짓 모사. β 선형 1자유도만 했고 비선형 teacher-student는 미시도. (c) 정렬쌍으로 **투수별 물리민감도 룩업**(물리편차×맥락→성공 기울기, per-pid EB수축) — 행 갱신 가중(count·hand)과 곱해져 판별식 ② 통과.
- 점수: +1~3급(β가 +0.97 실증) / d: TM 유래 재성형—어느 레그에도 없는 정보 / §5: 학습전용, 서빙은 E[phys] 산술(`ens11_response.py:9-11`) / 비용 1~2일 / kill: 위약(β y-셔플) 미분리 시. **판별식: 통과(②)**.

## ② TM-as-labels — 시도된 것과 안 된 것의 정확한 경계

- **된 것**: `commandnet.py` 인코더(TM 1.79M행, 구종군 CE+물리z 7회귀, `:10-14`)는 라벨-프리 통과(CE +6.2%), **NN student는 사망**(주셀 −8.7 ≈ 위약 −9.6, `27:74-75`), ridge 캐리어 전환 시 +2.26/+8.35/+1.07/+1.32 (`enc_carrier_gate.json`). `aux_mtl.py` 공식-only MTL은 kill이나 **MTL−위약 = +17.7 = 보조감독 신호 실재** (`27:34-35`). OOF 스킵 교정도 이미 구현됨(`commandnet.py:296-320`).
- **안 된 것 (설계 공백 3)**: (1) **가중치 전이 사전학습**: 인코더 출력 22컬럼을 피처로 '먹였을' 뿐, 트렁크 가중치를 train 라벨로 **fine-tune**한 적 없음. (2) **정렬 물리를 보조 타깃으로**: aux_mtl의 타깃은 복원 3/4클래스뿐 — 916k행의 실측 7축 물리를 회귀 보조손실로 쓰는 조합 미시도(정보량이 복원 라벨보다 큼). (3) **레그 단위 판정 자체가 미시도**: 모든 NN 실험이 ENS-9 멤버교체 증분(w .475 동결)으로만 측정됨(`commandnet.py:17`) — 레그의 성립 조건(단독 점수+d)은 다른 시험이다.
- §5: 서빙 = {pid:emb} 룩업+numpy forward, TabM 전례 합법(`commandnet.py:21`) / 비용 3~4일 / kill: GBDT-증류 없는 단독 NN 로컬 <850 시(자체 NN 상한 834 실측, `35:112`). **판별식: 통과(인코더 출력이 count·month 등 행 맥락 함수, `commandnet.py:69-84`)**.

## ③ F29 완성 — 트리가 이미 절반을 흡수, 잔여는 '가중'이지 '피처'가 아니다

- 문서 설계: era×카운트·era×높이×구종군 (`docs/research/10:220`), 근거 = 0-2존이 3-0존의 64%인 사람심판 패턴이 2024+ 소멸(`10:134-135`), 카운트×era는 심판ID 없이 포착(`10:154-155`), 대응 서열 F29→F31→C5(`10:143-145`).
- 챔피언 현황: `era_abs` 스칼라뿐(`real_data.py:220`), season은 드랍(`lgbm_family.py:53`).
- **빌드 가능 목록**: era×{count_bucket, count_diff, two_strike, three_ball, ahead, behind} · era×p_ball_minus_strike · era×same_hand · era×asof mix 3률. **불가**: era×높이×구종군(그 투구의 높이·구종 자체가 §6 금지+미제공), 2025 존v2·클락 서브플래그(train에 2025 없음 — 사망).
- **정직 판정**: era_abs(24.5만행)와 count는 둘 다 피처라 min_data_in_leaf 1000의 트리도 분할로 도달 가능 — 명시 피처의 신규 정보는 근소. **살아있는 잔여는 F31(2024 가중 스윕 + 전시즌 학습)**: "최신 1시즌만"은 ±400~750점 레짐 베팅이고(june853 메모), 학습창이 바뀌면 예측 자체가 크게 달라진다 = **d 발생원으로서의 era축**. 비용 0.5일 / kill: 3폴드 게이트 / **판별식: 피처형은 ① 통과(test에선 era=1 상수이나 count가 행마다 변함), 단 신규성 약 — '가중형'으로 채택 권고**.

## ④ v4 시퀀스 신호의 행 합법 그림자 (정의: `partners/.../v4/lg_features.py:112-151`)

원본은 train 행 정렬로 복원(테스트에선 §5 위반 — v4 zip이 실제로 그 이유로 레그 불가, `35:16`).

| 신호 | 판정 | 근거 |
|---|---|---|
| pitch_in_game | **사망(정확값)** | 경기 시작 시점의 n은 어떤 train-fixed 베이스로도 제공 불가. is4의 n_cur는 시즌 내 위치일 뿐(`lg_features.py:127`의 gstart가 미지). `n_cur mod 평균경기부하`는 수학적으론 행 갱신이나 오차 누적 — 위약 게이트 필수의 절름발이 |
| fatigue(전경기 부하) | **사망** | prev1_game_success_rate는 있어도 투구수는 없음. per-pid 평균부하는 상수(①불통과), 곱할 합법 행 가중 부재 |
| pitch_in_inning | **사망(신규성)** | 하한 ≈ f(outs_before, num_runners_on, balls+strikes) — 전부 기존 컬럼의 결정함수 = 트리 소관 |
| pa_pitch_no / foul_count | **사망** | floor(balls+strikes)는 이미 입력. 파울 수 실현값 복원 불가 |
| **is_starter** | **살아있음(형태)** | P(선발\|pid) train 룩업(상수) × row inning·n_cur → ② 통과. **보직은 44컬럼 어디에도 없는 신규 정보**("선발이 7회에 있다=깊은 등판" vs "불펜 7회"를 현 모델은 구분 못 함) |
| **game_no** | **살아있음(형태)** | est_game_no = n_cur ÷ train당 경기당투구수(pid). n_cur = asof_pitcher_n − 시즌말 룩업(is4 인프라 그대로, `knn_residual.py:94-95`)이 행마다 갱신 → ① 통과. 시즌 진행·등판 밀도(월 결합 시 강등/부상 프록시) |
| prev_game_pitches / roll3 | **사망** | 상동(fatigue) |

비용 0.5~1일 / kill: 위약(pid 셔플 룩업) 미분리.

## ⑤ N9 (`docs/00_KILL_LIST.md:98`) — 원안 사망, 구장(park) 재해석만 생존

- 원안: pitcher_team_id×season 평균 reverse rate = 포수/배터리 프록시. **판별식 ① 불통과**(투수-시즌 내 상수), ②는 game_dayofweek 가중(백업포수 요일)으로 형식상 가능하나 — **프레이밍 채널 자체가 2024+ ABS로 소멸**(`10:151-153`, F34도 "2019~23만 활성"으로 설계됨 `10:225`). test=2025 페이오프 ≈ 0. **사망**.
- **생존 변형 = 구장축(F28, `10:219`)**: top_bottom="T"면 투수가 홈팀 → **park_id = (T ? pitcher_team_id : batter_team_id)**. 행마다(원정/홈 교대) 변해 ① 통과, train-fixed park×성공률 룩업 합법. 트리는 "조건부 컬럼 스왑"을 두 고차 ID 원시값에서 스스로 합성하지 못한다. 마운드·돔 여부는 2025에도 지속되는 물리 신호. 실데이터 구현 0건(park는 `_retired/`에만 존재 — `sweep/_retired/features.py:306`). 비용 0.5일 / kill: 구장별 편차가 EB수축 후 무의미하면.

## ⑥ `sweep/knn_residual.py` — 보정기는 사망, '피처 뱅크'는 미시도

- 구현: fit시즌 잔차뱅크 + leave-own-pitcher-out + GPU top-k + 수축평균 **가산 보정**(`knn_residual.py:20-31,62-78`). 게이트 결과: **전 구성 평균 음수(최선 −4.2), 순수효과 +2~21로 국소구조 실재하나 분산이 승리** (`23:48-54`).
- 미시도: 문서 12가 교정으로 명시한 **"HGB 피처 형태가 기본"**(knn_logit/dist/std를 입력 피처로, `12:34-37`) — sweep 전체에 knn_logit 구현 0건(grep 확인). Home Credit/Otto 전례의 형태다.
- 레그급 판정: 신호 크기(+1~2 손익분기)가 실측된 채널이라 **주축은 못 된다**. 단 피처형은 모델이 신뢰 시점을 학습하므로 보정기보다 하방이 없다 — 조합재 D급. §5: 질의 = 행 자신 피처 vs train-fixed 뱅크, 합법(추론 10분 내 GPU 가능 — 로컬 3폴드 367s). 비용 1일(인프라 재사용) / kill: 위약(잔차 셔플 뱅크) / **판별식: 통과(질의 좌표가 행마다 다름)**.

## ⑦ month/team 학습측 피처 — 프로브 금지와 무관한 합법 채널, 구현 0건

- 현황: game_month·요일·팀ID 전부 원시 통과(`real_data.py:94,101`, 드랍 안 됨 `lgbm_family.py:53`). **금지된 것은 month LB 프로브 계보**(분할축 역산 — `32_month_axis_jtt.md`, 메모리 확정)이지 train측 month 피처가 아니다 — 이 구분을 문서에 남길 것.
- 합법 후보: (a) **투수×월 곡선 룩업**(train, EB수축) — 월이 시즌 내 행마다 변해 ① 통과, asof 어디에도 없는 정보. (b) 팀 수비/park는 ⑤로 흡수. (c) batter_team×시즌 상대타선 룩업 — batter_id asof가 이미 개인을 담아 잔여 약함. (d) 요일×p_starter(6선발 로테이션 슬롯) — ② 통과, 소형. 비용 0.5일 / kill: 위약 6컬럼과 미분리(난수 위약이 2SE를 뚫는 함정 유의 — 로컬 게이트 맹점 메모).

## ⑧ Passthrough-15 감사 (정직 판정 — 대부분 중복)

- **run_top/bot_before**: run_total·score_diff_home과 선형 항등(top=(total−diff)/2) — **완전 중복**.
- **score_diff_home**: score_diff_pitcher_team×top_bottom 부호 항등 — 중복.
- **away_win_expectancy**: 실측 home+away ∈ [99.9,100.1] — **완전 중복(반올림 잡음뿐)**.
- **asof_pitcher_pitchmix_n**: 실측 ratio≡1.000, 투수-시즌 내 SD 0 — **asof_pitcher_n과 동일체, 정보 0**. F32(커버리지 신호) 가설 사망.
- **runner_on_1b / num_runners_on**: 원시 입력 존재. 미구축 잔여 = F33 도루위협 플래그(1루단독×박빙×비RISP, `10:156-157`) — 트리 학습 가능 조합이라 신규성 소.
- **outs_before**: 원시 입력, 잔여 없음. **game_dayofweek**: ⑦-(d) 외 잔여 없음.
- 결론: 이 15컬럼엔 레그를 만들 정보가 **없다**. 유일한 진짜 미사용 원시 조합 = top_bottom×팀ID(⑤ park).

## TOP-3 조합 (새 레그 1개 기준, 예측 d는 정직-d 사다리 대비)

**1위 — "TM-표현 NN 레그": TM 사전학습 인코더(가중치 전이) + 정렬 916k 물리·복원라벨 보조손실 + 자체 GBDT 앙상블 소프트타깃 증류 (①+②)**
- 점수 메커니즘: 증류가 점수를 teacher(로컬 ~898) 근방에 고정 — 순수 NN 상한 834(`35:112`)의 해법. 예상 로컬 870~895 → LB 975~1005.
- d 메커니즘: 함수족(MLP 0.0261) + 입력표현(TM emb/물리 — 어느 레그에도 없음) + 감독(증류 자유도+보조손실 +17.7 실재) 3축 적층. **예측 d 0.029~0.036**. doc33:200 표가 근거: d 0.0354면 **S=960.3로도 1100 도달** — 점수를 조금 팔아 d를 사는 유일한 좌표.
- kill: 3폴드 단독점수 <860 또는 위약(pid 경로교환) 미분리. 비용 3~4일.

**2위 — "전시즌·era-가중 독립 GBDT 레그": june+is4 코어, ENS-9 비상속, F31 가중 스윕 + era×count + ④생존형(est_game_no·p_starter×inning) + ⑤park + ⑦월곡선 + condphys core4·enc10을 보정기가 아닌 피처로**
- 점수: is4-k100 계열 기반(877.82→984 실증) + 피처 게이트 통과분 — 로컬 885~905.
- d: 타-피처기저 GBDT(tm3L 0.0238이 준거) + **학습창 레짐 베팅**(±400~750점급 예측 변화의 원천)으로 **0.025~0.032**. 1위보다 점수 안전, d 하한 낮음.
- kill: 전시즌 학습이 V24 폴드에서 −50 이상 열세면 가중 상향으로 후퇴, 그래도 미달 시 폐기. 비용 2~3일.

**3위 — 1·2위 어느 쪽이든 공통 탑재하는 "그림자 블록"(④⑤⑦ + ⑥ 피처형 kNN)**
- 단독으론 전부 +1~2급이나, **§5 합법 형태로는 어떤 팀원 레그도 못 가진 정보**(v4 seq는 위반이라 서빙 불가, `35:16`)라서 d 가산이 순수하다. 예측 d 가산 +0.002~0.005. 비용 1.5일, 1·2위에 병합.

**감사 결론**: 정보축 재고는 ①정렬 물리감독(부분 개통) ②NN×TM 표현 ③학습창 레짐 ④보직·시즌진행 그림자 ⑤park뿐이며, 나머지(⑧ 원시열·N9 원안·pitchmix_n)는 실측으로 닫혔다. 0.031+는 이들 중 **3축 이상 동시 적층**에서만 산술이 성립한다.

---

# [S1] 연구 보고: 새 레그 후보 아키텍처 (2025–26 증거 기준)

## 핵심 결론 먼저

**TabPFN-3 (2026-05-12, Prior Labs)가 유일하게 킬 리스트 재개방 요건을 충족한다.** 재개방 근거(신규 반증):
- 기술보고서 arXiv:2605.13986 (PDF 83p 직접 추출·검증): **1M행×200피처 지원**, 내부 대규모 벤치(100K–1M행, 13개 데이터셋)에서 **8시간 튜닝된 LightGBM/XGBoost/CatBoost를 단일 forward pass로 상회**. 이 중 회귀 4개는 명시적 **temporal split**. v3는 **temporal prior(이산시간 동적 SCM) + OOD/외삽 prior**를 사전학습에 추가 — 구버전 킬 사유(2506.08982)를 정면으로 겨냥한 설계 변경.
- 독립 검증 arXiv:2606.30410 "Beyond IID"(TabArena 팀)는 TabPFN-**2.6**·TabICLv2·TabDPT까지만 평가하고 "temporal·대규모에서는 튜닝 GBDT가 지배" 결론 → **v2.6 이하·TabICL·TabDPT·Mitra의 킬은 유지, v3만 미판정 공백**. v3의 temporal 우위는 **벤더 자체 벤치만 존재**(그것도 회귀 한정) — 우리 로컬 게이트가 첫 독립 검증이 된다.
  *(세션 후기: 그 첫 독립 검증이 같은 날 실행됐고 **3구성 전패로 킬** — 14 §0-b.)*

나머지 지형(전부 검증): TabReD arXiv:2406.19380 · ICML'25 arXiv:2502.20260 (github.com/LAMDA-Tabular/Tabular-Temporal-Shift) 공통 결론 = temporal split에서 **GBDT+MLP-PLR만 정상**, retrieval 계열은 처방(Fourier temporal embedding)을 넣어도 역전 없음. 순수 함수족 교체로는 우리가 실측한 상한(MLP 834/RMS 0.026)을 깰 근거가 2026년 문헌에도 없다.

## 후보 카드 (4장)

### 1. TabPFN-3 (zero-shot ICL 레그)
- (i) 점수: TabArena 전체(51개셋) 승률 80%+ (vs LightGBM(T+E) 82%, XGBoost(T+E) 85%). 내부 대규모 벤치(100K–1M) 전 구간 1위. 우리 조건(1.47M×47)은 컨텍스트 축소 필요 — 2024시즌(~245K) 또는 2023–24(~500K).
- (ii) d: 유일하게 "함수족 교체"가 아니라 **정보 사용 방식 교체** — 8조 토큰 합성 SCM prior에 대한 베이지안 ICL. temporal/OOD prior는 시즌 경계 외삽에서 GBDT와 정반대 거동(Fig 26).
- (iii) 합법성/서빙: HF Prior-Labs/tabpfn_3, tabpfn-3-license-v1.0 — "연구·평가·Data Science Competition 허용, 상업 금지" = 팀 규칙 "비상업 이상 OK" 충족. **합성 데이터로만 사전학습** → 외부 실데이터 오염 제로. `pip install tabpfn` + ckpt 동봉(`model_path=` 오프라인 로드). 소비자 GPU 공식 지원(8–16GB), cached-predict는 쿼리가 train KV에만 attend → §5 충족(패리티로 확증).
- (iv) 비용: 학습 0시간, 구현 2–4일. 최대 리스크 = 10분 추론 한도(증류 경로 존재).
- (v) 킬: ① 시간분할 로컬 <901(두 컨텍스트 모두) ② RMS <0.031 ③ L4 추론 한도 초과+증류도 미달 ④ 라이선스 저촉.

### 2. xRFM (재귀 특징 기계 = AGOP 특징학습 + 국소 커널)
- (i) arXiv:2508.10053 (ICLR 2026): TALENT 회귀 100셋 1위, 이진분류 120셋 2위(0.856 vs RealMLP 0.839), 70K–500K행 검증, O(n log n). 단 **temporal 증거 전무**.
- (ii) 제3의 함수족(커널+AGOP 특징행렬, leaf 60K 국소모델). 다양성 실측 문헌 없음.
- (iii) **MIT**, github.com/dmbeaglehole/xRFM, `predict_proba` 제공. 서빙: leaf별 npz 분해 + ~150줄 재구현(+1일).
- (iv) 3–5일. 1.5M은 논문 검증 범위(500K) 초과 — 파일럿 필수.
- (v) 킬: ① 로컬 <901 ② RMS <0.031 ③ 1.5M 학습 >24h/OOM ④ 확률 캘리 열화.

### 3. GRANDE (경사하강 하드 축정렬 트리 앙상블)
- (i) arXiv:2309.17130 (ICLR 2024) + 2025–26 PyTorch 재작성(github.com/s-marton/grande, MIT). 원논문 19개 이진셋 MRR 0.702 vs CatBoost 0.570. TabArena 순위는 UNVERIFIED.
- (ii) 축정렬 편향(정확도) + 다른 학습 동역학(end-to-end SGD, 인스턴스별 leaf 가중) → "정확도는 트리, 오차는 비트리" 가설. 실측 증거 없음.
- (iii) MIT, state_dict 추출 가능. (iv) 1.5M 학습시간 UNVERIFIED — 파일럿 반나절. 구현 2–3일. (v) 킬: 파일럿 환산 >8h / 로컬 <901 / RMS <0.031.

### 4. RealMLP-TD (문헌상 최강 DL — 이중탈락 예상, 최후 보루)
- (i) TabArena v1 단일모델 1위(Elo 1569 > LGBM 1532); temporal에서 살아남는 DL = MLP-PLR 계열. 단 Beyond IID에서도 temporal·대규모는 GBDT 우위.
- (ii) 우리 실측이 직접 반박 — 자체 MLP 상한 로컬 834, 팀메이트 MLP RMS 0.026.
- (iii) Apache-2.0 (pytabkit). (iv) 1–2일. (v) 킬: RMS <0.031 (즉사 예상) — 전멸 시에만.

## Q3: 오차 비상관 문헌 판정 (2025–26)

- TabArena arXiv:2506.16791 — 전 모델 가중 앙상블이 모든 단일 계열 상회, *"the battle between GBDTs and deep learning is a false dichotomy"*; NN·GBDT 쌍방이 가중을 받음 = 계열 간 상보성의 정량 증거(IID 중형 기준).
- Beyond IID — temporal 제약 하 "정확도 유지한 채 비상관" 가능한 계열은 극소수.
- LoMETab arXiv:2605.14365: rank-r 곱셈 어댑터로 멤버 다양성 수 배 제어 가능하나 성능 TabM 동급 — 계열 내부 다양성이라 무관, TabM 킬 유지.
- OmniTabBench arXiv:2604.06814 (3,030셋): 계열별 승리 영역 지도는 있으나 오차 상관 분석 없음.

## 건드리지 말 것 (킬 사유 명기)

- TabPFN≤2.6·TabICL(v2)·TabDPT·Mitra: Beyond IID 직접 실측 전패. TabDPT는 retrieval 킬 중복. Mitra 소규모 전용. LimiX(arXiv:2509.03505): 증거 전무 + 라이선스 별도 승인.
- TabR/ModernNCA: retrieval 킬 유지 — temporal embedding 처방 후에도 미역전(저장소 Table A).
- TabM/LoMETab: 게이트 실측 킬(솔로 750) + 반증 없음.
- FT-Transformer: TabReD 중위권. **SAINT: intersample attention = test 행 간 참조, §5 위반 소지 자체로 탈락.**
- NODE/GANDALF: GRANDE 열위. DCNv2/FinalMLP/FFM: CTR 계열, TabReD 비상위 + entity-embedding 킬 저촉. EBM: 가법+쌍대 상한으로 901 개연성 낮음.

## 최종 순위

1순위 TabPFN-3 *(→ 실측 킬, 14 §0-b)* · 2순위 xRFM · 3순위 GRANDE(파일럿 조건부) · 4순위 RealMLP(보루).
주요 출처: arXiv 2605.13986 · 2606.30410 · 2506.16791 · 2406.19380 · 2502.20260 · 2508.10053 · 2309.17130 · 2605.14365 · 2604.06814 · 2509.03505 · 2511.08667 · HF Prior-Labs/tabpfn_3 · github PriorLabs/tabpfn, dmbeaglehole/xRFM, s-marton/grande. 전 arXiv ID 실개방 검증. UNVERIFIED 3건: GRANDE TabArena 정확 순위 · EBM TabArena 순위 · HF tabpfn_3 게이트 여부.

---

# [S2] 연구 보고: 미래-홀드아웃 대회 고고학 + KBO 2025 규칙

**요약**: ① 2024–2026 스포츠 Brier 대회(March Mania 3개년)는 전부 "XGBoost+레이팅+캘리브레이션" 우승 — GBDT를 크게 이기는 공개 레시피는 스포츠 대회에 **없다**. ② 일반 표형 대회의 2025 실질 변화는 표형 파운데이션 모델이며, **국내 DACON 미래-홀드아웃(전력사용량, 미래 1주)에서 TabPFN 단독 Private 1위 선례**(dacon.io/competitions/official/236531/codeshare/12783, 2025-08-29 게시). ③ KBO 2025 규칙은 전부 공식 소스로 확정(아래 표 — 14 §3에 전재). ④ xCTRL(2508.19184)보다 새로운 pre-pitch command 논문 **없음**(피인용 1건 = 테니스 2607.28500). ⑤ +130 갭의 최유력 불법 가설 = "test 배치에서 2025 in-season as-of 재구성".

## 후보 카드 요약 (전문 카드 6장 중 채택분은 14 티어표에 반영)

- **카드1 TFM 레그**: TabPFN-2.5(arXiv:2511.08667)/TabICL(2502.05564) 근거 + TabPFN-TS(2501.02945) 방증 + DACON 선례. *(S1과 통합, v3로 실행 → 킬)*
- **카드2 이종 3단 스태킹+힐클라임**: Chris Deotte 2025-04 Playground 우승(72모델 3단 스택), NVIDIA 그랜드마스터 플레이북. *(우리 D-35 실측 킬과 충돌 — 재론 불가로 판정)*
- **카드3 Numerai식 era-강건 학습**: era boosting(forum.numer.ai/t/era-boosted-models/189) · feature neutralization(forum.numer.ai/t/1059, arXiv:2303.07925 지지). 중화는 원형(era별 회귀=배치 통계) 금지 → **train에서 β 적합·동결 후 행 단위 적용**으로 변형해야 §5 합법. T2 레그에 반영.
- **카드4 룰-인코딩 era 피처**: zone_top_pct(2024=56.35→2025=55.75)·zone_bottom_pct·clock_enforced 등 season-level 상수. 유일 검증 경로 = **2023→2024 폴드(실제 룰 쇼크)**. 트리는 수치 외삽 불가 → 델타 피처 형태 필수.
- **카드5 커리어-시퀀스 Transformer**: 투구 시퀀스 유효성 arXiv:2606.17345 + NFL BDB 2026 예측 트랙(train 2023-24→live 2025) 선례. **train-고정 룩업으로만 시퀀스 구성해야 합법**; 2025 이력이 2024말에 잘리는 것이 킬 조건.
- **카드6 사전등록 이산 후보 + LB 선택**: ABS 존 이동이라는 ex-ante 근거로 레벨 후보 2~3개 등록·선택. **연속 최적화·대칭쌍 역산 금지, COMPLIANCE_DISCLOSURE 사전 기록 필수.** 후보 간 차이 ±1~2점이면 축 소진.

## KBO 2025 규칙 확인 (전부 규칙 공지 기반 — 상세 표는 14 §3)

ABS 2025 = 상단 55.75%·하단 27.04%(각 −0.6%p ≈ 1cm 하향, 존 크기 불변·전체 하향), 시즌 중 추가 변경 없음 (6차 실행위 2024-12-04; KBO 공지 2025-05-08 bdSe=11496). 피치클락: 2024 시범(18/23초, 경고만) → 2025 정식(20/25초, 위반 볼/스트라이크, 타석 간 33초, 견제 제한 없음 — MLB와 차이). 공인구 규격 불변(반발계수 측정치는 결과 통계 인접 — 피처 금지). 연장 12→11회. 체크스윙 판독 1군은 2026부터. ⚠ KBO 규정 페이지의 "20초/9초"는 표 파싱 오류 — 복수 언론(20/25초) 기준.

## 미래-홀드아웃 고고학 결론

- March Mania 2025 우승 = XGBoost 단일(lr 0.0093, depth 4)+29피처+수동 오버라이드(공식 write-up 검증). 2026 4위 = XGB+LGBM 0.65/0.35 + [0.025,0.975] 클리핑 + 대칭 증강 + 시즌-GroupKFold. 2024/2026 1위 write-up 본문 접근 실패(UNVERIFIED).
- Rohlik v2(미래 2주) 1위 = LGBM+XGB, 호라이즌별 직접 예측 14모델(+0.4), 평균 인코딩 — 역시 GBDT+FE.
- Jane Street/Optiver 계열의 핵심(예측 기간 중 온라인 재학습)은 라벨 피드백 전제 → 우리 환경 원천 불가·불법.
- 시사점: 미래-홀드아웃에서 GBDT를 크게 이긴 사례는 (a) 시퀀스/공간 데이터의 NN(BDB 2026), (b) 소중규모 TFM(DACON 전력 1위)뿐.

## 1198 설명 가설 (+130 ≈ ΔBrier 3.5e-4 = is4의 3.3배)

**합법**: ① as-of/당해시즌 분해의 심화(세그먼트·상대별) — 가장 개연성 높은 합법 단일 요인 ② TFM/이종 레그 다양성(+수십) ③ 사전등록 대량 선택(+20~30 상한, private에서 일부 증발).
**불법(설명력 순)**: ① test 배치 2025 in-season as-of 재구성(단독 +100급, 최유력) ② 배치 평균/분위 캘리(존 이동 시 4e5·δ² 레버리지) ③ LB 역산 상수. ①②는 Phase 3 재현 코드 제출에서 노출되는 유형 — 최종 순위는 집행 강도에 달렸다.
