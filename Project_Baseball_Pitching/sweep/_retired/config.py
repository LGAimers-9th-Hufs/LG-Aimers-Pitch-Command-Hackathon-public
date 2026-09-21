"""
데이터 계약(DataSpec) + 합성 데이터 생성기.

실데이터를 꽂을 때는 아래 DataSpec의 컬럼 이름만 실제 데이터에 맞추면 됨.
합성 생성기는 하네스 스모크 테스트용(실데이터 도착 전 파이프라인 검증).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class DataSpec:
    # 식별/시간/그룹
    target: str = "command_success"       # 이진 타깃 (제구 성공=1). 실데이터 라벨명으로 교체.
    pitcher_id: str = "pitcher_id"
    batter_id: str = "batter_id"
    season: str = "season"                # 시간 기준(시즌). 시간분할 CV의 축.
    game_id: str = "game_id"              # 그룹 분할 단위(누수 차단).
    order: str = "pitch_index"            # 경기 내 투구 순서(as-of 정렬용).
    pitch_no: str = "pitch_no"            # 등판 내 투구 번호(F19 피로 피처). 없으면 as-of로 파생.
    team_id: str = "team_id"              # 투수 소속 팀ID(익명). F28 구장 프록시용. 없으면 skip.
    home_team_id: str = "home_team_id"    # 홈팀ID = 익명 구장. F28용. 없으면 skip.
    is_home: str = "is_home"              # 투수가 홈팀인지. F28용. 없으면 skip.
    abs_era_start: int = 2024             # F29: ABS 도입 시즌(2024 1군 전면). 존 v2=2025.
    # 맥락/중요도 피처(예측 시점에 존재하는 pre-release 정보만)
    context: list = field(default_factory=lambda: [
        "balls", "strikes", "outs", "inning", "score_diff", "on_base", "leverage",
    ])
    # Trackman 릴리스 로그(F18 미니 Kirby Index용). 과거 투구의 값만 사용(현재 투구 실측 금지).
    release: list = field(default_factory=lambda: [
        "release_x", "release_z", "release_angle",
    ])


def make_synthetic(n_pitchers=60, n_batters=120, seasons=(2019, 2020, 2021, 2022, 2023, 2024, 2025),
                   pitches_per_season=4000, seed=7) -> pd.DataFrame:
    """제구 성공 확률 과제를 흉내낸 합성 pitch-level 데이터.

    - 투수별 잠재 제구력(skill) + 카운트/레버리지 효과 + 노이즈 → 잠재 로짓 → 이진 타깃.
    - as-of 피처가 실제로 예측력을 갖도록 skill을 시간에 걸쳐 일관되게 둠(단, 컬럼으로 노출 안 함).
    - 2025 시즌 = hidden test 흉내(라벨은 있지만 CV에서 미래로 취급).
    """
    rng = np.random.default_rng(seed)
    pitcher_skill = rng.normal(0.0, 0.6, n_pitchers)      # 잠재 제구력(비노출)
    batter_diff = rng.normal(0.0, 0.3, n_batters)          # 타자 난이도(비노출)
    n_teams = 10
    pitcher_team = rng.integers(0, n_teams, n_pitchers)   # 투수 소속팀(익명 팀ID)
    stadium_eff = rng.normal(0.0, 0.08, n_teams)          # 홈팀ID=구장 효과(비노출, F28 신호)
    # 투수별 릴리스 평균 위치/각도(비노출) + 릴리스 일관성(고skill=저분산 → 미니 Kirby 신호)
    pit_rel_x = rng.normal(0.0, 1.0, n_pitchers)
    pit_rel_z = rng.normal(6.0, 0.5, n_pitchers)
    pit_rel_ang = rng.normal(0.0, 3.0, n_pitchers)
    pit_rel_sd = np.clip(0.25 * (1.0 - 0.9 * pitcher_skill), 0.05, None)  # skill↑ → SD↓
    # 데뷔 시즌: 다수는 첫 시즌부터, 일부는 후반 데뷔(콜드스타트/희귀 투수 생성 → F13 arm 검증용)
    debut_idx = rng.integers(0, len(seasons), n_pitchers)
    debut_idx[: int(n_pitchers * 0.6)] = 0                 # 60%는 원년부터
    rows = []
    for si, season in enumerate(seasons):
        n = pitches_per_season
        active = np.where(debut_idx <= si)[0]              # 이 시즌까지 데뷔한 투수만 등판
        pid = active[rng.integers(0, len(active), n)]
        bid = rng.integers(0, n_batters, n)
        game = rng.integers(0, 120, n) + si * 1000         # 시즌별 고유 game_id
        balls = rng.integers(0, 4, n)
        strikes = rng.integers(0, 3, n)
        outs = rng.integers(0, 3, n)
        inning = rng.integers(1, 10, n)
        score_diff = rng.integers(-6, 7, n)
        on_base = rng.integers(0, 8, n)                    # 주자 상태 비트(0~7)
        leverage = np.round(np.abs(rng.normal(1.0, 0.7, n)), 2)
        # 등판 내 투구 번호(피로 프록시): (game, pitcher)별 순번. pitch_index(arange) 시간순.
        pitch_no = (pd.DataFrame({"g": game, "p": pid})
                    .groupby(["g", "p"]).cumcount().to_numpy())
        # 릴리스 실측(현재 투구) — 과거값만 F18에 쓰이며, 신호는 skill→SD 경로로 전달.
        # F26 신호: 등판 내 투구수에 비례해 릴리스 높이가 뜨는 피로 드리프트 추가.
        rel_sd = pit_rel_sd[pid]
        release_x = pit_rel_x[pid] + rng.normal(0, 1, n) * rel_sd
        release_z = (pit_rel_z[pid] + 0.004 * pitch_no + rng.normal(0, 1, n) * rel_sd)
        release_angle = pit_rel_ang[pid] + rng.normal(0, 1, n) * rel_sd
        # F28 신호: 홈/원정 + 홈팀ID(=익명 구장) 효과
        team = pitcher_team[pid]
        is_home = rng.integers(0, 2, n)
        opp = (team + rng.integers(1, n_teams, n)) % n_teams
        home_team = np.where(is_home == 1, team, opp)
        # 잠재 로짓: 제구력↑ 성공↑, 3볼/2스트 압박↓, 고레버리지 약간↓, 피로(등판 내 투구수)↓
        # + F29 era 신호(2024+ ABS로 base rate 이동) + F28 구장 효과 + 홈 이점
        era_abs = 1.0 if season >= 2024 else 0.0
        logit = (0.9 * pitcher_skill[pid] - 0.5 * batter_diff[bid]
                 - 0.15 * balls - 0.10 * (strikes == 2)
                 - 0.08 * (leverage - 1.0) - 0.010 * pitch_no
                 + 0.18 * era_abs + stadium_eff[home_team] + 0.05 * is_home
                 + rng.normal(0, 0.5, n) + 0.2)
        p = 1.0 / (1.0 + np.exp(-logit))
        y = (rng.random(n) < p).astype(int)
        df = pd.DataFrame({
            "season": season, "game_id": game, "pitch_index": np.arange(n),
            "pitcher_id": pid, "batter_id": bid, "pitch_no": pitch_no,
            "team_id": team, "home_team_id": home_team, "is_home": is_home,
            "balls": balls, "strikes": strikes, "outs": outs, "inning": inning,
            "score_diff": score_diff, "on_base": on_base, "leverage": leverage,
            "release_x": release_x, "release_z": release_z, "release_angle": release_angle,
            "command_success": y,
        })
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    # 전역 시간 순서 키(정렬 안정성)
    out = out.sort_values(["season", "game_id", "pitch_index"]).reset_index(drop=True)
    return out
