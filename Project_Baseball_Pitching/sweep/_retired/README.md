# 폐기된 모듈 (합성 데이터 시절)

실데이터·규정 확정(2026-08-05) 이전에 만든 하네스다. **실행 경로에서 쓰지 않는다.**
지우지 않고 남기는 이유는 ① 설계 기록 ② 아래 "구조 가능" 2건 때문이다.

폐기 사유는 `docs/00_KILL_LIST.md` §4에 등록돼 있다.

| 파일 | 왜 폐기됐나 |
|---|---|
| `config.py` | `DataSpec`이 합성 스키마 전용. 실데이터엔 `game_id`·`pitch_index`·`pitch_no`·릴리스 컬럼이 아예 없어 컬럼명 매핑으로 이어붙일 수 없다 |
| `features.py` | 자체 as-of 피처 레이어 전부. **2025 test에서 계산 자체가 불가능**(§5 행 독립성)하고, 주최가 `asof_*` 19컬럼을 leave-current-out으로 제공한다 |
| `candidates.py` | 위 두 모듈의 표현(`flat_f13`/`flat_plus`/`seq` 등)에 의존 |
| `sweep.py` | successive halving. LogLoss(작을수록 좋음)로 정렬·비교·delta가 하드코딩돼 있어 대회 지표(클수록 좋음)에서 **순위가 뒤집힌다** |
| `run_smoke.py` | 합성 end-to-end 진입점 |
| `diagnostics.py` | P1 adversarial validation. 아이디어는 유효하나 2025 입력이 로컬에 5행뿐이라 원형은 실행 불가 |
| `test_leakage.py` | 합성 플라시보. 실경로에서 증명할 명제가 "미래 라벨이 새지 않았다"에서 **"어떤 행도 다른 행에 의존하지 않는다"**로 바뀌었다 → `sweep/test_regulation.py`가 대체 |
| `make_paper_assets.py` | LogLoss 기준 표를 생성. 지표 교체 후 재작성 필요 |

## 구조 가능(salvage) — 옮겨 쓸 값이 있는 것

1. **`candidates.py:_BayesLog5`** — 투수율·타자율·리그평균의 log5 사후확률. `p_asof_rate`/`b_asof_rate`
   참조를 공식 `asof_pitcher_success_rate`/`asof_batter_success_rate`로 바꾸면 **행 단위로 합법**이다.
2. **`candidates.py:_MERFLite`의 적합된 투수별 절편 dict** — train에서 만든 **고정 룩업**이라
   zip에 동봉해 조회하면 합법이다. 원본 ID를 트리에 직접 넣는 것(2025에 train 최대 초과 ID 실재,
   은닉 시간지표)의 안전한 대체재.

`features.py`의 `_beta_posterior_var`(베타-이항 사후분산)도 `(rate, n)` 쌍만 있으면 계산되므로
행 단위로 옮길 수 있다 — Phase F arm C의 재료.
