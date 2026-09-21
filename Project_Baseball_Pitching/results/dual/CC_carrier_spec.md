# C-C — 제3 캐리어: is4×맥락 잔차 ridge 보정기 스펙 (설계서, 구현·측정 전)

*(2026-08-08, Claude 측 독립 조사 C-C. 설계만 — 코드·측정 없음. 근거 파일은 전부 경로로 인용.)*

## 0. 한 줄 요약과 권고

**권고안 = 안 2 (is4 전면 ⊗ 맥락, 25항) × 타깃 B (TM 보정 차감 잔차) × 사전등록 주 셀
(λ=1000, w2=0.5), 배포 후보는 w2∈{0.25, 0.5}만.**
기대값은 로컬 +1.5~+3.5 (단독 +10 도달 확률 ~5–10%, 3폴드 D-25+위약 통과 확률 ~25–35%).
근거: 같은 정보(cxp)가 선형 캐리어에서 +2.1~+7.2로 실재 확인됐고(V21 −0.4), 같은 운반형(잔차
ridge)이 TM 정보로 ENS 위 +3.4를 실현했으며, 비모수 잔차 운반(kNN)은 순수효과 +2~21이 실재해도
분산에 져서 −4.2였다 — **저차원 모수 기저가 분산을 통제하면서 같은 층을 노리는 마지막 형태**다.

---

## 1. 동결 불변식 (게이트 전제)

- **ENS-7 완전 동결**: 멤버 5종 가중 `ENS7_W = {nn_lin_bis .475, et_l100_d28 .25, allraw_l15 .125,
  june_l15 .10, june_l7 .05}` + TM 보정기 `TM_W=0.5` + 배포 캘리 (slope 1.04, shift −0.01)
  전부 불변 (`sweep/eb_carrier.py` L56–59). 멤버 재학습·재가중·캘리 재적합 일절 없음.
- 신규 보정기는 **그 위 한계 증분**: 판정식
  `Δ = deploy_score(yv, ens_full + w2·corr2) − deploy_score(yv, ens_full)`,
  `ens_full = Σ ENS7_W·P + TM_W·corr_tm`, `deploy_score` = `eb_carrier.deploy_score`(동결 캘리).
- 폴드 = fit 시즌 v−1 → val 시즌 v. **네 폴드 전부 멤버 캐시 존재**
  (`results/ens7/preds_val{2021,2022,2023,2024}.npz` — y·corr·멤버 5종 동봉, `eb_carrier.get_members`).
- 채택 = 3폴드(V24/23/22) 평균>0 **및** 평균>폴드SD **및** 위약 분리 **및** V21 감사 무붕괴,
  제출은 패키지 합계 +10 규칙에 종속.
- **이중 운반 금지 전방 규칙**: corr2가 채택되면 같은 cxp 정보를 nn_lin 입력으로 다시 넣는 안은
  자동 소멸한다(역도 동일). 한 정보는 한 운반로만.

## 2. 왜 "잔차 ridge"가 cxp 정보의 남은 운반인가

같은 정보(투수 당해 혁신 × 카운트)의 운반 3형태 실측:

| 운반 | 실측 | 병리 |
|---|---|---|
| 트리 직삽 (is4+확장 컬럼) | ±13 구조 섭동, D-25 반복 실패 (`docs/log` D-25) | 트리 재배치 노이즈가 신호를 삼킴 |
| 선형 멤버 입력 (nn_lin+cxp) | +2.12/+7.23/+2.94/−0.40 (V24/23/22/21, `results/interaction/gate.json`) | 멤버 재학습(섭동) + 블렌드 흡수(w=0.475 희석), 시그모이드 척도 |
| 잔차 보정기 (TM 전례) | +1.18/+4.55/+4.45 ≈ **+3.4** (w0.5, `results/tm_member/gate.json`) | — 유일하게 ENS 수준 실현 |

잔차 보정기가 구조적으로 깨끗한 이유 세 가지:
1. **무섭동**: 동결 멤버를 하나도 재학습하지 않는다 → 판정이 곧 한계 증분(±13 섭동 소멸).
2. **확률 척도 직결**: ridge가 잔차(확률 차)를 직접 적합 → Brier(=점수식)와 같은 척도.
   nn_lin은 로지스틱 척도에서 배우고 블렌드가 다시 섞는다.
3. **분산 통제가 명시적**: 이득은 `2·w2·Cov(corr2, resid)`로 선형, 비용은 `w2²·E[corr2²]`로
   제곱 (점수 환산 ≈ 4e5×, memory Brier 레버리지). kNN(−4.2)은 이웃평균의 E[corr2²]가 커서
   순수효과 +2~21을 전부 반납했다 — **p≤32 모수 기저는 같은 Cov를 훨씬 작은 E[corr2²]로 담는다.**

주의(정직한 상한): june 트리 5멤버는 이미 is4 4컬럼을 본다(base 57에 포함). corr2가 노리는 것은
트리 해상도(min_data 1000–2500) **아래**의 매끄러운 곱 구조뿐이다. 흡수 법칙(docs/log/24 §2)상
이 이득은 "새 정보~정제 경계"라 전이율 1.0을 기대하면 안 된다 — EV를 cxp 실측(+2~3)에 앵커한 이유.

## 3. 설계행렬 옵션 (2+1안)

공통 규약:
- 인자 중심화 상수는 **고정 리터럴 또는 fit 유래 동봉 상수**(배치 통계 아님 — nn_lin served
  spec의 mean/scale 전례, `build_lgbm_ensemble.py` train_linear): `is_smc = is_sm−0.5`,
  `is_sharec = is_share−0.5`, `is_delta`(이미 0 중심), `is_lognz = is_logn − mean_fit(is_logn)`,
  `hand ∈ {0,1}`(TM ctx 동형), `cc = count_code/11`, `C12` = 카운트 12상태 원핫.
- 곱을 만든 뒤 **컬럼별 scale-only 표준화**(fit sd로 나눔, 중심화 없음 → 희소성 보존).
  sd 벡터는 fit에서 산출해 동봉 — λ가 전 컬럼에 동일한 의미를 갖게 하는 장치다
  (is_delta 원척도는 1e-2급, is_logn은 O(1~8) — 표준화 없이는 λ가 컬럼마다 다른 수축이 된다).
- 절편 1열 포함(표준화 제외). is4 값은 X의 IS_COLS(순서 계약, `lgbm_family.py` L92)를 그대로 읽는다.
  K_IS=100 (ENS-7 계약).

### 안 1 — "cxp 최소 이식" (15항)

```
D = [ C12 ⊗ is_delta ]   12항   ← interaction_carrier.py L60 cxp_0..11과 산술 동일
  + [ is_delta ]           1항   (주효과 — ridge가 곱을 주효과에 직교화하도록)
  + [ hand · is_delta ]    1항   (hand_x_pd, 같은 파일 L66)
  + 절편                   1항
```
- **cxp와의 관계 = 문자 그대로 대체 운반.** 실측 앵커가 가장 직접적(+2.1/+7.2/+2.9/−0.4가
  이 12컬럼 자체의 실적). 분산 최소.
- 희소성: C12 곱 컬럼 j의 비영 비율 = 카운트 셀 점유율(0-0 ~25% … 3-0 ~1%, 추정치 — 실행 시
  fit에서 출력할 것). 셀별 유효 norm² = n_j·Var(is_delta|j) → λ가 자동 신뢰도 수축으로 작동(§5).
- 약점: is_sm/is_share/is_logn 축의 곱 구조를 버린다 — TM 보정기 ctx에 is_sm이 들어가 유효했던
  전례와 어긋나는 절약.

### 안 2 — "is4 전면 ⊗ 맥락" (25항) ★권고

```
D = 안1의 15항
  + [ is_smc, is_sharec, is_lognz ] ⊗ (1, cc, hand)   9항   (TM 설계 동형: 프로필 자리에 is4)
  = 24항 + 절편 = 25항
```
- is_delta에는 검증된 원핫 해상도(cxp 그대로), 나머지 3인자에는 TM ctx 동형의 연속 맥락 —
  항 수를 TM(31)보다 작게 유지하면서 is4 네 축을 전부 운반.
- `is_smc ⊗ (1, cc, hand)`는 corr_tm의 `TM6 ⊗ is_smc`와 **컬럼은 분리**되고 상관으로만 겹친다
  (corr_tm에는 TM6 인자 없는 is_sm×count 항이 없다) → 겹침은 §4의 타깃 B로 처리.
- 추가 맥락 후보 중 **배제**: li/inning/주자(사전 실측 없음 + D-25 컬럼 비용), balls 단독(cc·C12에
  포함), fb_rate(TM ctx에서 이미 소비 — corr_tm과 겹침 확대 방지).

### 안 3 — "안 2 + 구종믹스 당해 블록" (31항, 스트레치 전용)

```
D = 안2의 24항 + [ ism_fb_d, ism_br_d, ism_os_d ] ⊗ (1, hand)   6항 + 절편 = 31항
```
- `inseason_full.py` BLOCKS "m"(pitchmix, K_M=200)의 `_d` 3컬럼. **문서화된 역풍 2건**:
  트리 직삽 `is4+mix`는 is4 대비 V24 −17.9/V23 +1.0/V22 −3.7(`results/inseason/full_gate.json`),
  EB log-ratio 재표현은 전 폴드 음수(`results/eb_carrier/gate.json` ebB100+mix). 운반형이 달라
  "혹시"는 남지만 **prior는 음수** — 안 2가 살았을 때만 +6항 증분 arm으로 1회 측정.
- 유일하게 서빙 룩업이 추가되는 안(§8): `inseason_mix` {pid: [n_end, k_fb, k_br, k_os]} ~792엔트리.

## 4. 잔차 타깃 — A(june_fit_resid) vs B(TM 차감), 권고 B

| | 타깃 A: `r = y − p_june(fit, in-sample)` | 타깃 B: `r₂ = r − TM_W·corr_tm(fit)` |
|---|---|---|
| 정의 | `tm_member.june_fit_resid` 그대로 (기존 TM 보정기와 동일 타깃) | A에서 **배포되는 그대로의** TM 보정(w=0.5, 커버 밖 0)을 뺀 것 |
| 배포 동형 | ✅ (fit=v−1 학습 → v 적용, TM 전례) | ✅ **더 정확** — 배포식 `ens + 0.5·corr_tm + w2·corr2`에서 corr2가 학습한 대상이 정확히 "0.5·corr_tm이 남긴 것" |
| 공선성 | ⚠ corr_tm과 corr2가 **같은 전체 잔차**를 각자 적합 → 상관된 방향(is_sm·count 경유)을 이중 운반, 그 성분은 `0.5β_tm + w2β_2`로 과보정 | ✅ 순차(부스팅) 직교화로 구조적 제거 |
| 비용 | 0 (resid 재사용) | 폴드당 TM ridge 재해석 1회: `TMM.design`(1.4M행×31) + 31×31 solve — 수 초. 코드가 이미 `get_members` V21 분기와 빌더에 있다 |
| 함정 | — | corr_tm(fit)은 그 ridge의 in-sample 값이라 미세 낙관이 섞인다. λ=1000·31계수·~16만 커버행이면 유효 자유도가 미미해 무시 가능. corr_tm이 파단 폴드에서 흔들리면 corr2 타깃도 흔들린다(V23에서 관찰할 것) |

**권고: 타깃 B를 주 판정으로, A를 같은 실행의 대조 arm으로**(비용 0 — resid 공유, solve만 2벌).
- 해석 규칙까지 사전 등록: `Δ_A ≈ Δ_B` → 겹침 미미(어느 쪽이든 무방, B 유지).
  `Δ_A ≫ Δ_B` → A의 초과분은 TM과의 이중 운반(과보정 위험) — **B만 신뢰**.
  `Δ_A ≪ Δ_B` → corr_tm 차감이 노이즈를 걷어낸 것 — B 채택 강화.
- 배제한 대안 2종: (i) **ens 잔차(y − ens_fit)** — 멤버 fit-행 예측이 캐시에 없고, ET(l100)의
  in-sample 잔차는 준보간이라 타깃을 왜곡하며, june 잔차→ENS 적용의 척도 불일치는 w2 격자가
  흡수한다(TM 전례로 실증). (ii) **TM+is4 동시 재적합** — corr_tm 계수를 다시 푸는 순간 ENS-7
  동결 위반. 제외.

## 5. 격자 근거 — λ∈{300, 1000, 3000} × w2∈{0.25, 0.5, 1.0}

**w2 (분산 통제의 1차 레버):**
- 점수 항등식: `Δscore ≈ 4e5·[2·w2·Cov − w2²·E[corr2²]]` (r(1−r)≈0.25). 이득 선형·비용 제곱 →
  과대 w는 비대칭적으로 위험하다. kNN 교훈의 정량형: 순수효과가 실재해도(+2~21) w를 줄여도
  손익분기 +1~2에서 캡 — **분산이 큰 운반은 w로 못 살린다**(docs/log/23 §3).
- TM 실측이 같은 격자에서 내부 극대를 보였다: V24에서 w1.0이 −1.33으로 뒤집힘, w0.5 최적
  (`results/tm_member/gate.json`) → {0.25, 0.5, 1.0} 유지. **1.0은 진단 전용**(w축 곡률로
  Cov 대 분산 비율을 읽는다), 배포 후보는 0.25/0.5만.
- 사전 등록 주 셀 w2=0.5: TM 전례의 최적점이자, corr2 분산이 TM(31항)과 같은 급(≤32항)이라
  같은 체급의 w가 1차 추정으로 합리적.

**λ (셀별 자동 신뢰도 수축):**
- 표준화 후 컬럼 Gram ≈ N_fit(~2.3e5) — 전 지지 컬럼엔 λ/N ≈ 0.001~0.013으로 사실상 무수축,
  **희소 셀 곱 컬럼**(norm² = n_j·Var)에만 유효하게 작동한다. 3-0 셀(~1%, ~2.3e3행)이면
  λ=1000에서 ~30%, λ=3000에서 ~55% 수축 — is4의 K 수축과 같은 역할을 ridge가 셀 크기에
  비례해 자동 수행한다.
- 격자의 구조적 의미: λ=1000 ≈ 트리 min_data(1000–1500)와 같은 신뢰 granularity(TM 실증점),
  3000 = 트리보다 보수적(안전측), 300 = 트리 아래 해상도 공격(안1의 희소 셀에서만 차이 예상).
  반 데케이드 스텝이면 min 격자로 충분 — λ 민감도가 크면 그 자체가 "신호가 희소 셀에 있다"는
  진단이다.
- **선택 규칙 사전 등록**: (λ, w2)는 점수 평균 최대가 아니라 **V24 주판정 + V23/V22 개선분
  maximin**으로 고른다(ens5_select의 스케일 폭주 교훈, docs/log/23 §2). 폴드마다 다른 셀이
  이기면 선택 낙관으로 간주하고 주 셀(1000, 0.5)로 회귀한다.

## 6. 위약 설계 (컬럼·계수 수 매칭)

모든 실항이 is4(또는 ism) 인자를 포함하므로 위약은 그 인자만 교란한다. **맥락(C12·cc·hand)은
절대 건드리지 않는다** — 카운트 주변분포는 실·위약에서 동일해야 한다.

- **위약 P1 (주) — 투수 이력 경로 교환**: `IS.base_for(lut, pid, season, shuffle_seed=s)`
  (inseason.py L90–100, is4_shuf 전례 seed=7)로 투수↔직전시즌말 누적 대응만 파괴 →
  `L.add_inseason`으로 is4 재생성 → **같은 곱, 같은 항 수, 같은 표준화 절차(위약 자신의 fit sd로
  재산출), 같은 λ·w2 격자, 같은 solve**. 시즌 내 행별 갱신 구조(범위·단조성)는 보존되고
  "그 투수 자신의 당해 폼"이라는 정보만 죽는다. 2시드(eb_carrier 전례 — 스크린은 1시드,
  후보 확정 시 2시드). 안 3의 ism 인자는 `IF.end_lookup` 사전을 `eb_carrier.entity_traj_map`
  패턴으로 pid 순열 재매핑해 동일 처리.
- **위약 P2 (보조) — is4 인자 행 셔플**: 시즌 내에서 is4 4컬럼을 행 순열(interaction_carrier
  L97–98 PL 패턴)한 뒤 같은 곱 형성, 1시드. P1과의 차이 = 행 내 맥락 정렬까지 파괴 —
  "임의의 같은 척도 회귀변수도 이만큼 낸다"를 걸러낸다(난수 6컬럼이 2SE를 통과한 위약 실측
  교훈, memory 로컬 게이트 맹점).
- **매칭 규칙(명문화)**: 위약 설계행렬은 실arm과 (N×p) 동형 — 항 수·절편·표준화 절차·λ·w2·타깃
  산출 절차 전부 동일. 다른 것은 오직 투수↔이력 대응 하나. 멤버·corr_tm·ens_full은 실arm과
  동일하게 동결(위약은 corr2 입력만 교란).
- **판정**: `Δinfo = Δ(real) − mean(Δ(P1 시드들))`이 채택 (λ,w2)에서 3폴드 평균 양수이고
  음수 폴드가 1개 이하일 것 + P2가 real과 유의 분리될 것. real이 D-25를 통과해도 위약이
  같이 통과하면 기각(채널 부재).

## 7. 실행 시퀀스 — 신규 스크립트 1개 `sweep/is_corrector.py`

**재사용**: `eb_carrier.get_members / deploy_score / ENS7_W / TM_W`(멤버 캐시 4폴드 완비),
`tm_member.june_fit_resid / load_profile / design`(타깃 B의 corr_tm(fit) 재구성),
`inseason.career_end_lookup / base_for`(실·위약 base), `lgbm_family.build_features / K_IS=100`,
`inseason_full.end_lookup / build_block`(안3만). 신규 로직은 설계행렬 빌더 + solve 루프뿐(~150줄).

```python
# 의사코드
L.K_IS = 100; df = rd.load_train()
lut = IS.career_end_lookup(df); nb, kb = IS.base_for(lut, pid, season)
X = L.build_features(df, base=(nb, kb))          # 57열 (is4 포함)
IS_real = X[IS_COLS]                              # 실 인자
IS_pl = {s: add_inseason(base_for(shuffle_seed=s)) for s in (0, 1)}   # 위약 P1

def build_design(ISblk, df, opt, fit):            # 안1/안2(/안3)
    factors = 중심화(ISblk); ctx = (C12, cc, hand)
    D = 곱 조립 + 절편; scale = D[fit].std(); return D / scale, scale

for v in vals:                                    # 2024 → 2024,2023,2022 → +2021
    P, corr_v, yv = get_members(df, X, y, season, v)         # 캐시 히트
    ens_full = Σ ENS7_W·P + TM_W·corr_v; s_ref = deploy_score(yv, ens_full)
    resid = TMM.june_fit_resid(X, y, fit)                    # 타깃 A (지배 비용: lgbm 2회)
    Df, covf = TMM.design(df, X, TMM.load_profile(), fit)    # 타깃 B용 corr_tm(fit)
    c_tm = solve(A'A + 1000·I, A'r);  corr_fit = (Df@c_tm)·(covf) 
    rB = resid − TM_W·corr_fit[fit]
    for arm in (안1, 안2, [안3], P1_s0, P1_s1, P2):
        Dz, sc = build_design(...)
        for target in (rB, resid):                # B 주판정, A 대조
            for lam in (300, 1000, 3000):
                c2 = solve(Dz[fit]'Dz[fit] + lam·I, Dz[fit]'target)
                for w2 in (0.25, 0.5, 1.0):
                    Δ = deploy_score(yv, ens_full + w2·(Dz[val]@c2)) − s_ref
# 출력: 폴드×arm×λ×w2 표 + Δinfo 표 + maximin 선택 → results/is_corr/gate.json
```

**명령 순서와 게이트**
1. `python sweep/is_corrector.py --vals 2024` — **V24 스크린** (~6–8분; 비용 지배 = june_fit_resid
   lgbm 2회). 통과선: 어떤 (λ, w2∈{0.25,0.5})에서 Δ ≥ +1.5 **및** P1 1시드와 분리. 미달 → 종료
   (kNN 손익분기 전례상 스크린 미달을 폴드 추가로 구제하지 않는다).
2. `--vals 2024,2023,2022` — 3폴드 확정 + P1 2시드·P2 (~20분, resid×3이 지배). D-25: 평균>0 &
   평균>SD & §6 위약 판정.
3. `--vals 2024,2023,2022,2021` — **V21 감사** (~+5분, preds_val2021.npz 캐시 존재). cxp가 V21
   −0.40이었으므로 V21 붕괴(< −3) 시 채택 보류. 총 소요 ≈ **30–40분**.
4. 통과 시: 다른 생존 후보와 합산해 패키지 +10 충족 여부 → 충족 시에만 §8 편입·제출.

## 8. §5 서빙 가능성 — 가능 (안1·2는 신규 룩업 0개)

**행 단위 산술 + 동봉 상수만으로 완결된다.** corr2(row) = `w2 · Σ_k c_k · D_k(row)/scale_k`:
- D_k의 재료 = balls/strikes(행), hand 2컬럼(행), is4 4컬럼 — is4는 서빙 템플릿이 **이미**
  `inseason_base` 룩업(792 투수, `submit_ens7_bis_tm.zip` metadata 실측)으로 계산하고 있다
  (`script_lgbm_template.py` L55–76). test 행 간 참조 없음, 배치 통계 없음 — tm_correction과
  동일한 적법성 등급.
- 배포 동형성: 빌더 fit=2024 행의 is4 base = 2023말, 서빙 2025 행의 base = 마지막 기록 시즌말
  (=2024말) — 게이트의 v−1→v 오프셋과 정확히 같다. 중심화 상수·scale은 fit에서 산출해 동봉
  (nn_lin served spec 전례로 §5 적법 확립).
- **TM 대비 이점**: 커버리지 마스크 불요 — is4는 신인 포함 전 행에서 정의된다(base(0,0) = 커리어가
  곧 당해, 학습과 동일 의미론). 커버 0.66의 희석이 없다.

**`build_lgbm_ensemble.py` 변경 개요** (--ens7 분기 확장 또는 --ens8):
1. tm_coef solve 직후: `corr_fit` 전체 계산(1줄) → `rB` → fit 행 설계행렬 + scale 산출 →
   `c2` solve → `meta["is_corrector"] = {"terms": 항 정의 리스트, "coef": [...25], "scale": [...],
   "centers": {...}, "weight": w2, "target": "B"}` (수 KB).
2. 파리티 게이트 추가(D-19 −96점 사고 재발 방지): 템플릿 `is_correction()` vs 빌더 계산을
   20k 프로브 행에서 RMS < 1e-9 — 기존 tm 파리티 블록(L299–318) 복제.
3. 템플릿: `is_correction(test, X, meta)` 신설(~25줄, tm_correction 미러 — X에서 is4 읽고 곱 조립,
   scale 나눗셈, coef 내적) → `raw = raw + tm_correction(...) + is_correction(...)`;
   try/except 개별 격리(보정 실패 = 0 가산, 안전망 불변). 추론 비용 O(N×32) matvec — 10분 한도에
   무의미. `test_regulation.banned_calls` 통과 형태(집계 연산 0) 유지.
- **안 3 채택 시에만**: `inseason_mix` 룩업(~792 엔트리, `IF.end_lookup` blocks "m"을 serve_base와
  같은 "마지막 기록 시즌말"로 붕괴) + 템플릿 build_features에 ism 블록(isb 블록 미러, ~15줄) 추가.
  서빙 표면이 늘어나는 유일한 안 — 안 2까지는 **zip 내용물 변화가 metadata 수 KB뿐**이다.

## 9. 예상 EV·리스크 (사전 등록 포함)

**EV (실측 앵커 3점 삼각측량)**
- 정보량 상한: cxp 선형 운반 +2.12/+7.23/+2.94/−0.40 — 안정 레짐 실현분 ~+2~3.
- 운반형 실현율: TM 잔차 ridge가 ENS 위 +3.4 실현(동일 게이트·동일 w0.5) — 운반형은 검증됨.
- 분산 하한 경고: kNN 잔차 −4.2(순수효과 실재에도) — 모수화·λ·w로 분산을 눌러야만 양수가 남는다.
- ⇒ **중심 추정 +1.5~+3.5 (안 2, λ=1000, w2 0.25~0.5)**. 단독 +10 확률 ~5–10%(cxp 정보량이
  근본 제약), D-25+위약 통과 확률 ~25–35%. 30–40분 비용 대비, cxp 정보의 **마지막 미시도
  운반형을 닫는다**는 소진 가치가 실질 절반이다.

**리스크와 방어**
1. TM과 이중 운반(과보정) → 타깃 B 구조적 차단 + A/B 대조 진단(§4 해석 규칙).
2. V21 취약(cxp −0.40 전례) → V21 감사 필수, 붕괴 시 보류(§7-3).
3. 희소 셀 노이즈(λ=300 × 딥카운트) → λ 자동 수축 논증(§5) + 배포 λ는 1000 미만 금지.
4. 격자 선택 낙관(3설계×2타깃×3λ×3w2 = 54셀) → **주 셀 사전 등록: 안2×B×λ1000×w2 0.5.**
   나머지는 전부 감도 분석으로 격하, 주 셀이 죽고 다른 셀만 살면 "선택 낙관"으로 기각.
5. 위약 통과 착시(난수 6컬럼 전례) → Δinfo 이중 위약(P1 경로교환 + P2 행셔플), 컬럼 수 완전 매칭(§6).
6. 서빙 파리티(D-19 −96 사고 계급) → 빌더 RMS < 1e-9 게이트 없이는 zip 생성 금지(§8-2).
7. 파단 레짐(V23) 왜곡 → V23은 스트레스 폴드로만 사용, maximin에는 개선분으로만 반영(기존 규약).
