# 11. 타 도메인 3차 조사 — 운동제어·축구·정밀스포츠·산업통계

작성 2026-08-06. 방법: 4방향 병렬 웹조사(R1 인간 운동제어 / R2 축구 xG 캘리브레이션 / R3 정밀 조준 스포츠 / R4 산업·통계), 전 제안에 3중 필터(§5 행 독립성 / 데이터 부재 사실 / 서버 환경) 자가검증 강제, 인용은 arXiv abs·DOI·PDF 직접 확인(미확인은 명기, 철회 논문 1건 배제: arXiv:2607.01722).
관련: `docs/log/14_baseline_gap.md`(갭 분해 — 우선순위의 근거), `13_trackman_train_usage.md`(N11–N13), `12_finetuning_rag.md`(N32–N35), `00_KILL_LIST.md`(상위 문서).
**번호 규약**: 신규 arm은 N14부터(N5/6/8 결번 유지). 기존 M/F 번호와 무관.

## §0 요약

- 3개 도메인(운동제어·탄도학·야구 최신 문헌)이 **같은 지점으로 수렴**: 제구는 결과율이 아니라 **조준점 a + 산포 σ의 2물리량**이고, 우리가 가진 실패 3유형 비율(asof_pitcher_middle/ball/reverse_rate)에서 이 좌표계를 **폐형식으로 역산**할 수 있다(과식별 → 잔차도 피처). → Family A (N14–N21).
- 축구·야구·골프의 시즌 간 지속성 수치가 **차등 수축**의 정량 prior를 준다: 산포형 지표 r=0.65 > 결과형 0.48(야구 xCTRL), 슈터 52%/GK 74%(축구), 퍼팅≈노이즈(골프). 현 z_table의 두 결함(모멘트법 τ² 과소추정, 시즌 경계 미분리)이 "스프레드 20% 과수축"의 기술적 원인 후보로 특정됨. → Family B (N22–N27).
- 공정능력지수(Cpk)의 MLB 투수 직접 적용 전례(2026)가 확보됨 — asof 3률의 probit 변환 피처는 비용이 pandas 몇 줄. → Family C (N28–N29).
- 사망 선고 다수 확보(§1.5) — 특히 **릴리스 포인트 일관성은 제구(BB/9)를 예측하지 못한다**는 실증은 트랙맨 프로필(N13)의 목표 변수를 바꾼다.

## §1 횡단 발견 (개별 arm보다 중요)

### 1.1 조준-산포 분해의 3중 수렴
- **운동제어**(R1): Harris & Wolpert 1998 *Nature* — 운동 노이즈 ∝ 제어신호 크기. Trommershäuser 2003 JOSA-A / Landy 2012 J Neurosci — 자기 σ와 벌점 기하로부터 최적 조준점이 도출되고, σ가 크면 조준을 후퇴(과소보상 경향).
- **탄도학**(R3): 착점 N₂(μ,σ²I)에서 임의 조준점·반경의 명중확률은 Marcum Q(비중심 χ²) 폐형식(Grubbs 1964 *Oper. Res.*; DiDonato & Jarnagin 1961 *Math. Comp.*; 현대 구현 CRAN shotGroups). 양궁 실기록에서 반경이 Rayleigh를 근접하게 따름(Dall & Park 2022) = 등방 2D 정규 가정의 실측 타당성.
- **야구 최신**(R1·R3 독립 확인): Ludwig, Brill & Wyner, **xCTRL** (arXiv:2508.19184, 2025) — 의도/실행 분리, **제구(실행오차)의 시즌 간 r=0.65 vs 결과형 Location+ 0.48 vs Stuff+ 0.73**, bin당 250구 안정화, 1인치 개선 ≈ FIP −0.3.
- **우리만의 확장 지점(문헌 공백 = 가산점 서사)**: 기존 문헌의 벌점영역은 표적 *바깥*에만 있다. 우리 성공영역은 **도넛(한복판도 실패)** — 이때 ∂P/∂σ의 부호가 조준점에 따라 반전된다(코너 공략: 정밀도=미덕 / 한복판 공략: 정밀도=독). 트리는 σ를 입력으로 받지 않는 한 이 구조를 학습할 수 없다.

### 1.2 "능력 vs 전략"의 채널별 차등 지속성
- σ(산포)는 신체 능력이라 sticky(r 0.65), a(조준 편향)는 포수·전략 의존이라 이월이 약하다 — Kirby Index(연간 R² 0.50)가 결과형 Location+(0.39)를 상회(FanGraphs, 산업). 반례 축: 골프 퍼팅 스킬은 사실상 노이즈(Brill & Wyner arXiv:2506.21822 — 553명 중 유의 3명) = **채널별 신호량이 극단적으로 다르므로 채널별 차등 수축이 정답**이라는 횡단 교훈.
- 축구 정량 prior: Böttger & Vischer 2026(뮌스터 dp 3/2026, REML 메타분석, 8시즌 3,202 선수-시즌) — finishing 선수간 이질성 I²≈21%, **시즌→시즌 신호 유지율 슈터 ≈52% / GK ≈74%**. "한 시즌 초과성과는 능력 지표로 못 쓴다."

### 1.3 스프레드 20% 과수축의 기술적 원인 후보 2개 (R2)
현 z_table(Bühlmann–Straub)과 문헌 표준의 차이 4가지 중 결함 후보:
1. **모멘트법 τ² 과소추정** — 공변량 설명분·이분산 하에서 between-분산이 과소추정되어 수축이 과해짐. 문헌 표준은 REML 또는 **홀드아웃 예측우도 직접 스캔**(Miss It Like Messi JQAS 2024: Dirichlet α=30을 홀드아웃 로그우도로 선택).
2. **시즌 내 수축(k)과 시즌 경계 지속(ρ)의 미분리** — "④스프레드 과수축"과 "②전이 과신"이 모순이 아니라 **파라미터 2개**였다는 해석. 유지율 52–74%가 ρ의 외부 prior.
부수: 확률 스케일 선형 credibility(현행)보다 로짓 스케일 random intercept가 표준(극단 확률 왜곡 회피).

### 1.4 난이도층/레벨층 분업 — 현 구조의 문헌 지지
- Robberechts & Davis MLSA 2020: xG 난이도 모델은 **5시즌 데이터 필요, 최근 시즌만 쓰면 개선 없음**(난이도층에 감쇠를 걸면 손해라는 반대증거).
- Dixon & Coles 1997 *JRSS-C*: 레벨·강도 파라미터에만 가중우도 감쇠 ξ(예측우도로 선택, 반감기≈1시즌) — 레이어가 아니라 모델 내부 갱신의 30년 전례.
- Opta 실무(2023-02): 드리프트 대응은 재캘리 레이어가 아니라 **모델 버전 개정**(xG 값 67% 변경). Davis et al. *Machine Learning* 2024: "1–2시즌 넘은 데이터는 관련성을 잃을 수 있으나, 노후화 연구 자체가 거의 없다."
→ "HGB 잔차부스트(난이도)는 풀데이터, GLM(레벨)만 감쇠·갱신"인 현 구조와 정확히 동형. F31(recency-decay)의 정밀화 방향 = **적용 대상을 레벨층으로 한정**.

### 1.5 사망 선고 모음 (재제안 금지)
| 후보 | 판정 | 근거 |
|---|---|---|
| **릴리스 포인트 일관성 피처** | 트랙맨 목표변수에서 **강등** | Wakamiya 2024(Front Sports Act Living, MLB): RP 산포는 K/9·HR/9·xFIP 예측하나 **BB/9 예측 0**(구속만 R² 1.4%). Kusafuka 2020: RP 게인 1(1cm→1cm) vs **릴리스 각도 게인 ~30**(1°→30cm). 제구는 위치가 아니라 방향 문제 → N13 프로필은 무브먼트/구속 일관성 우선으로 교체 |
| 투수별 이방성 공분산(2×2) | 불필요 | Kusafuka 2022(J Mot Behav): 릴리스 파라미터 간 협응적 공변동 유의하게 없음 — **스칼라 σ가 충분통계량** (n=14 한계 명기) |
| 손잡이 주효과 제구 arm | 기각 | 좌/우완 BB% 차이 무시 수준(블로그, 미확인) — 손은 조준 이동 δa(ctx)에만 흡수 |
| lag-1 오차수정, 경기 내 피로 | 원리적 불가 | 투구순번·game_id 부재 (Kusafuka 2025 Sci Rep의 ACF1-SD r=0.54 구조는 실재하나 구현 불가) |
| 연령 곡선 | 불가 | 연령 컬럼 없음 |
| beta 3-파라미터 **후처리** | 기각 | Manokhin & Grønhaug 2026(arXiv:2601.19944, TabArena 21분류기): "Platt·isotonic류 후처리는 강한 현대 타뷸러 모델의 proper scoring을 체계적으로 열화". 단 비대칭 항(ln p̂, ln(1−p̂))을 **GLM 입력 피처로 넣는 in-model 형태**(Kull AISTATS 2017)는 합법 각주 |
| Venn-Abers(C4) | 보류 유지, 제출 소비 금지 | 동 논문: 이미 보정된 모델엔 Brier +2.2% 열화. 교환가능성 위반(시간 전이)으로 validity 보장 소멸 |

## §2 Family A — 조준-산포 분해 (asof 3률의 행 단위 물리 좌표화)

공통 규정 논리: 입력은 자기 행의 asof 값 + train 동결 그리드/룩업 → §5 자명 통과. 전 arm이 레벨층(β) 불변 — 4e5·δ² 비용을 건드리지 않음.

| arm | 무엇 | 근거(핵심) | 층 | 비용 | 게이트 |
|---|---|---|---|---|---|
| **N14 marcum_asof** ★T1 | p_middle=1−Q₁(d/σ,r₀/σ), p_ball=Q₁(d/σ,R₁/σ), p_reverse=Φ(−(a+m)/σ) — 3식 2미지수 **과식별 역산** → 행 단위 (σ̂, â, misfit) 3피처. 기하 상수(R₁/r₀, m/r₀)는 train 풀링 동결, 역산은 200×200 그리드 보간(np.i0 급수 or scipy.stats.ncx2 — sklearn 의존성으로 존재). z_table 수축 rate에 적용(Φ⁻¹ 폭주 방지) | Grubbs 1964, DiDonato&Jarnagin 1961, Dall&Park 2022(Rayleigh 실측), M34/N1의 수치 시뮬레이션을 폐형식으로 교체 | ④해상도 | 1일 | rung0(최신 폴드) 3피처 추가 vs 미추가 → refV24+역방향 |
| **N15 donut_signflip** ★T1 | 투수별 (a_p,σ_p) + 문맥별 조준이동 δa(ctx) 룩업 → 행마다 P_donut(a_p+δa(ctx), σ_p) 피처 1개. **∂P/∂σ 부호 반전**(1.1) 주입 — 트리가 원리적으로 못 배우는 구조. **사전 관측검정(0.25일): 3볼 카운트에서 σ̂ 상위군의 실제 성공률이 하위군보다 높은가 — 음성이면 즉시 폐기** | Trommershäuser 2003, Landy 2012, Harris&Wolpert 1998 | ④(②) | N14 위 +0.5일 | 사전검정 → rung0 → 전용 진단: 투수별 예측/관측 스프레드비 0.80→1.0 이동 여부 |
| **N16 sticky_split** ★T1~2 | 시즌 경계에서 σ̂ 약수축·â 강수축 후 폐형식으로 3률 **재구성** — 도넛 기하를 보존하는 구조적 수축. 하이퍼 (λ_σ, λ_a) 2개 | xCTRL r 0.65 vs 0.48, Kirby 0.50 vs 0.39, 골프 반례(채널별 차등이 정답) | ②③ | N14 위 +1일 | **train 내 자체 사전검증**: 연도쌍 r(σ̂) vs r(â) 측정 — σ̂이 높지 않으면 폐기 (이 측정 자체가 논문 Figure) |
| N17 credk_closed | 콜드스타트 K를 정보량으로 폐형식화: delta-method로 Var(σ̂\|n)=v(σ,a)/n → K=E[v]/Var_between. 단기/장기 창별 (σ̂_short, σ̂_long) 멀티스케일 | Taylor&Grubbs 1975(사격 그룹통계 추정 효율), 골프 EB | ③ | N14 위 +0.5일 | 신인·저표본 슬라이스 refinement 별도 리포트 |
| N18 effort_sigma | σ_row = σ_p·exp(β·e(ctx)), e는 2차 허용(70% MTS 봉우리형). **경로A: 트랙맨은 문맥→노력도 map "제작"에만 사용, 채점엔 미사용**(2025 부재 무관 — 트랙맨의 최저가 활용처). 경로B: e를 도넛 모델 내 자유 파라미터로 | Freeston&Rooney 2014(70% MTS 최적, 속도↑→수직오차↑), Kusafuka 2020(1m/s↔20cm), 2022(고속에서 보상 공변동 소멸=볼록) | ④① | 1일(B)/2일(A) | N15 채택 후에만. 3볼/2스트라이크 서브그룹 부호 확인 |
| N19 press_li | LI×σ̂, LI×â, LI×log(1+n) 상호작용 피처(부호 강제 금지 — 프로는 초킹 작고 이질적) | 다트 JEBO 2020(결승 레그 −2.5~−7.7pp, 프로 5.3% vs 아마 18.1% — 숙련이 완충), 바이애슬론 Leonelli 2024, 테니스 Klaassen&Magnus JASA 2001. R4의 야구 측: 투수 체계적 초크 부재(Otten 2013) — 기대 소폭 | ①미시 | 0.5일 | rung0, 노이즈면 즉폐기 |
| N20 odisp_phi | 투수별 월 블록 성공률의 베타-이항 과산포 φ̂ 룩업 — "일관성의 일관성"(상태 변동형 투수 식별). misfit(N14)과 상보 — 상관 높으면 한쪽만 | Phillips 2012(크리켓: 변동성이 숙련 판별축), Jamil 2022(위치 밴드가 결과 가름 → 산포의 시간 변동에 잔여 정보) | ④보조 | <1일 | rung0, 다중공선 정리 |
| N21 ell_strike (조건부) | asof_pitcher_strike_rate를 4번째 식으로 → (a,b,σ) 또는 (a,σ_h,σ_v) just-identified 확장. **식별 한계 정직 보고: 5률 전부로도 완전 타원 모델은 비식별**(좌표 없는 데이터의 이론 상한 — 논문 '한계'절 감) | Grubbs 1964(타원 본론), Park 2014(양궁 분산 구조 분해) | ④ | 2일 | **N14 통과 & misfit 중요도 상위일 때만**(misfit 큼 = 확장모델 존재 증명) |

## §3 Family B — credibility·시즌 전이 정밀화 (z_table 업그레이드)

Phase B의 S1(N11 z→GLM)과 **한 묶음으로 구현** 권장 — 같은 z 생성 코드를 고치는 일이다.

| arm | 무엇 | 근거 | 층 | 비용 | 게이트 |
|---|---|---|---|---|---|
| **N22 k_pll_scan** ★T1 | 수축강도 k를 모멘트법 대신 **홀드아웃 예측우도(Brier) 직접 스캔**. 가능하면 로짓 스케일: GLM offset + 투수 원핫 L2 로지스틱(=MAP random intercept, C가 수축강도) | Miss It Like Messi JQAS 2024(α=30 홀드아웃 선택 전례), Bayes-xG arXiv:2311.13707, §1.3 | ④③ | 0.5~1일 | k 격자 → refV24+역방향, oracle 662 갭 잠식률 보고 |
| **N23 rho_seasongate** ★T1 | 서버 적용 오프셋 = **ρ·z(≤2024)** — 시즌 경계 감쇠를 k와 분리. ρ∈{1.0, 0.85, 0.75, 0.6, 0.5} | Böttger&Vischer 2026(유지율 52–74%), xCTRL 0.65 | ②④ | 극소 | (k,ρ) 2D 공동 스캔 |
| N24 decay_split | 시간감쇠(F31)를 **GLM 레벨층에만** 적용, HGB는 풀데이터. ξ는 미래 시즌 예측우도로 선택 | Dixon&Coles 1997(반감기≈1시즌), Robberechts&Davis 2020(난이도층 감쇠 금지 반대증거) | ② | 소 | ξ∈{0, 반감기 2/1/0.5시즌} vs F31 전층 감쇠 A/B |
| N25 hcred2 | Jewell 2단 계층: 수축 목적지를 리그평균 → **투수 유형평균**(양손×game_type×믹스 분위 버킷)으로 | Jewell 1975(IIASA RM-75-24), Campo&Antonio 2023(actuaRE, arXiv:2206.15244), Efron&Morris 1975 | ③④ | 중 | n버킷 슬라이스 refinement + 그룹 분산비의 연도 안정성(역방향) |
| N26 n_bucket_bias | log(1+커리어 투구수)·신인 더미·저표본 버킷을 **명시 입력 피처**로 — 저표본 체계 편향을 모델 내부에서 흡수 | Davis&Robberechts arXiv:2401.09940(xG는 저볼륨 슈터 체계적 과대예측 — 표본량=캘리 오차 축) | ③④① | 극소 | refV24 + 그룹별 reliability slope |
| N27 thermo_cred | 온도계 기울기 × 월 교호 — 시즌 초 수축, 중반 이후 풀 기울기(GLM 내부 항 = 레이어 아님) | Benz&Lopez 2021(급성 레짐은 부분 풀링), DC 계열 관행 | ②③ | 소 | 2024 가상 서버 역방향에서 월별 Brier 분해(4–5월 개선/후반 비악화) |

## §4 Family C·D — 공정능력 피처·그룹 캘리 + 프로토콜

| arm | 무엇 | 근거 | 층 | 비용 | 게이트 |
|---|---|---|---|---|---|
| **N28 cpk_asof** ★T1(최저가) | asof 3률의 probit 변환 3종: 수율지수 capa=⅓Φ⁻¹((1+s)/2)(Boyles), 도넛 양측 최약마진 cpk_row=min(z_in,z_out)/3 + **비대칭도 z_in−z_out**(z_in=Φ⁻¹(1−middle), z_out=Φ⁻¹(1−ball−reverse)) — 실패 유형이 "한복판형/이탈형"인지의 투수 유형 정보, Taguchi 합성손실. z_table 수축 rate에 적용 + ε=1/(n+2) 클리핑 | Boyles 1991 JQT(S_pk↔수율 1:1), **RSM 2026 MLB 투수 WHIP Cpk+CUSUM(DOI 10.1080/15438627.2026.2632610 — 재부상 4–9경기 조기감지)**, Perakis&Xekalaki(부적합비율 기반 지수), Tahan&Lévesque(원형 공차대) | ④③ | pandas 10–20줄 | F-arm rung0. N14와 같은 재료(3률 변환)의 저가 버전 — 같은 배치로 게이트 |
| N29 spc_drift | prev-k vs career의 **2-비율 z**(1/√n 스케일링이 신규 정보), 단조성 d1>d3>d5(가속 드리프트), EWMA 근사(train 고정계수) | Hawkins 1987(자기시동 CUSUM = 모수 미지 표준화, 콜드스타트 동형), RSM 2026 | ②④ | 극소 | 원시 prev 컬럼 대비 ablation 순증 요구 |
| N30 mc_offset | **감사 선행(무료)**: 그룹수집 C={game_type×카운트×손, LI버킷, n버킷}의 그룹조건부 캘리 오차를 forward 폴드에서 측정 → \|δ_g\|>2SE & 2022/23/24 부호 일치 그룹만 train 147만 **전체** 적합 이산상수 dict 동봉. 기각 레이어와의 차이 = train 전체 적합(소표본 수축 원인 없음) + 이산 그룹 상수(연속 외삽 없음) + n버킷이 콜드스타트 명시 그룹 | HKRR ICML 2018, Kim+ PNAS 2022(universal adaptability — 이동 모집단 전이 보장), **Hansen+ NeurIPS 2024(arXiv:2406.06487): 잘 보정된 모델은 대체로 이미 multicalibrated → 감사 비면 즉시 폐기** | ①세그먼트 | 소 | 감사 → refV24+역방향. 감사 결과가 비면 제출 소비 금지 |
| P9 closed_test_beta | Vergouwe closed testing을 **제출 의사결정 규칙**으로: {β=0, β=−0.02, +기울기}를 역방향 폴드 우도비로 선별 — "언제 절편만, 언제 그 이상"의 원칙 | Vergouwe 2017 Stat Med, Riley 2021(slope 재추정은 절편보다 훨씬 큰 표본 요구 = 절편-only 우선의 정량 근거) | ①(전략) | 소 | 서버 코드 불변 |
| P10 subgroup_audit | n버킷×월×game_type reliability slope·그룹 Brier 분해를 refV24/역방향에서 상시 산출(진단 전용) — 전역 캘리 양호가 서브그룹 편향을 가림 | Davis&Robberechts 2024, Davis+ ML 2024 | 진단 | 극소 | 산출물이 타 arm 채택 판단 입력 |
| N31 coldstart_debut | 2025 미등장 ID 폴백을 전역평균 → **데뷔 코호트 (a,σ)·성공률 분포 + 월 인덱스 학습 기울기**로 | Kusafuka 2023(경험=표적 구분 능력: 신인은 reverse형 실패 상대 열위 가설), Wu 2014 Nat Neurosci(변동 큰 개인=빠른 학습→신인 기울기 가파름) | ③ | 0.75일 | 미등장 ID 서브그룹 전용 refinement 1차 + 전체 비악화. **선결 ID 지속성은 검증 완료**(pitchmix_n 단조 100%) |
| T5 소액 | chronic_load_sigma(만성부하→σ 승수; Bradbury 2012 — ERA 지표라 약함) / volat_feat(성공률 시즌 내 분산; Andersen&Carstensen 2026 간접) / li_pressure 잔여 | — | — | 각 0.5일 | T5 예산 1회, 기각 시 미련 금지 |

## §5 통합 우선순위 (갭 분해 문서 14와 결합)

문서 14의 결론 — 갭의 최대 단일 원인은 레벨(β 과충전, −127), 조건부 성능은 이미 +35 우위, **콜드스타트는 2025의 4~6%뿐**(상한 수십 점), 안정 우위 채널은 warm 95% 층의 투수 해상도 — 를 반영한 실행 순서:

1. **B-패밀리 합본**(N11 z→GLM + N22 k_pll + N23 ρ + N26 n_bucket): 같은 코드 경로, warm 층 해상도 직격. [S1 확장]
2. **3률 변환 배치**(N28 cpk + N29 spc + N14 marcum 3피처): 전부 행 단위 변환 — 한 게이트 런. [S1.5 신설]
3. **N15 signflip**(사전 관측검정 먼저) → 통과 시 N16 sticky/N17 credk.
4. kNN 패밀리(N12 v2 — 문서 12) — 콜드 캡 인지하되 잔차 뱅크는 전 층 headroom.
5. N13 트랙맨(목표변수 교체 반영) — 타임박스·kill criteria 불변.
6. N24/N25/N27/N30/N31 — 1~4 결과 보고 후.

## §6 가산점 서사 (발표용, 합본)

"성공률은 결과이고 제구력은 파라미터다. 우리는 투수의 실패 3유형 비율만으로 탄도학의 offset-circle 이론(Grubbs 1964)을 거꾸로 돌려, 투수를 **산포 σ(신체 능력)와 조준 공격성 a(전략)의 2차원 좌표**로 사상했다 — 좌표 데이터 없이. 최신 문헌(xCTRL, Kirby Index)은 산포형 지표가 결과형보다 훨씬 sticky함(r 0.65 vs 0.48)을 보였지만 전부 트래킹 좌표가 필요하다. 우리 지표는 박스스코어 수준의 집계율만 있는 어떤 리그에서도 산출되는 산포형 제구 지표다. 그리고 이 대회의 성공영역은 도넛이다 — 한복판도 실패다. Harris & Wolpert(1998, Nature)의 signal-dependent noise가 말하듯 세게 던질수록 흩어지는데, 도넛에서는 흩어짐이 항상 악이 아니다: 코너를 겨눌 때 σ는 독이지만 3볼에서 한복판을 겨누는 순간 σ는 공을 구멍 밖 도넛으로 흩뿌려주는 약이 된다. ∂P/∂σ의 부호 반전 — σ를 보지 못하는 어떤 트리도 배울 수 없는 구조를, 우리는 train에서 역산한 (a, σ) 룩업 하나로 주입한다. 부산물로 야구계의 통념 하나가 뒤집힌다: 릴리스 포인트 일관성은 제구의 지표가 아니다(BB/9 예측력 0, Wakamiya 2024) — 손이 어디 있느냐가 아니라 공이 어느 방향으로 떠나느냐(각도 게인 30배)의 문제이기 때문이다."

## §7 인용 검증 상태

- **직접 확인(arXiv abs/DOI/PDF)**: Harris&Wolpert 1998 · Todorov&Jordan 2002 · Trommershäuser 2003 · Landy 2012 · Kusafuka 2020/2022/2023/2025 · Freeston&Rooney 2014 · Wakamiya 2024 · Bradbury&Forman 2012 · Wu 2014 · xCTRL 2508.19184 · Brill&Wyner 2506.21822 · Grubbs 1964 · DiDonato&Jarnagin 1961 · Taylor&Grubbs 1975 · Dall&Park 2022 · Park 2014 · Klein Teeselink JEBO 2020(PDF 본문) · Leonelli 2411.02000 · Klaassen&Magnus 2001 · Mecheri 2016 · Shimizu&Konaka 2024 · Phillips 2012 · Jamil 2022 · Haugh&Wang JQAS 2024(=2302.10750 게재 확정, 서지 갱신) · Böttger&Vischer 2026 · Davis&Robberechts 2401.09940 · Bayes-xG 2311.13707 · Miss It Like Messi 2308.01523(JQAS 2024) · Robberechts&Davis MLSA 2020 · Davis+ ML 2024 · Dixon&Coles 1997 · Benz&Lopez 2012.14949 · Boyles 1991 · Hawkins 1987 · Shi-Ma-Lin 2016 · HKRR 2018 · Hansen+ 2406.06487 · Kim+ PNAS 2022 · Manokhin&Grønhaug 2601.19944 · Kull AISTATS 2017 · Vergouwe 2017 · Riley 2021 · Jewell 1975 · Sundt 1988 · Campo&Antonio 2206.15244 · Efron&Morris 1975 · Otten 2013 · Jane 2022 · Hsu 2019.
- **부분/2차 확인**: RSM 2026 투수 Cpk(DOI 리졸브 확인, 저자명 페이월) · Gerber&Jones 1975 · Hunter 2018 · Ovadia 2019 · Kim-Ghorbani-Zou 2019 · Leamon 2018(산업) · Kirby Index(산업) · ElHabr/penaltyblog(기술 블로그 — 명시 구분).
- **미확인·배제**: "릴리스 1ms=2피트"(1차 출처 미특정 — 발표 사용 금지) · 좌/우완 BB%·연령 곡선(블로그) · 538 클럽 SPI 방법론(사이트 폐쇄) · xG Brier 0.0686 단일치(고교생 저널 — 신뢰 하위 명시) · **arXiv:2607.01722 철회 확인 → 배제**.
