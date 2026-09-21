완료했습니다. 모든 산출물은 `results/codex_research/`에만 작성했습니다.

- [REPORT.md](G:/26-LGAI/Project_Baseball_Pitching/results/codex_research/REPORT.md)
- [HANDOFF.md](G:/26-LGAI/Project_Baseball_Pitching/results/codex_research/HANDOFF.md)
- [research_calc.py](G:/26-LGAI/Project_Baseball_Pitching/results/codex_research/research_calc.py)
- [partition_design.py](G:/26-LGAI/Project_Baseball_Pitching/results/codex_research/partition_design.py)
- [probe_math.py](G:/26-LGAI/Project_Baseball_Pitching/results/codex_research/probe_math.py)
- [partner_analysis.py](G:/26-LGAI/Project_Baseball_Pitching/results/codex_research/partner_analysis.py)

핵심 결론:

- train 비중 헤시안을 가정하면 `k`셀 partition은 최소 `k-1` probe + 꼭짓점, 즉 정확히 `k`슬롯이다.
- 최선 partition은 3·4월 병합 `month_m34`: 7슬롯, V2024 오라클 `+11.15`, 계획 EV `+10~20`.
- 파트너의 상수 레벨 오차는 post-blend 절편으로 정확히 흡수된다. 다만 기준은 `s` 하나가 아니라 `S_B^aff + s⊥`다.
- 파트너 가중·level·slope는 4 probe + 배포, 총 5슬롯으로 동시에 닫힌다.
- 현재 Codex 파트너 중간 후보는 추정 이득 `+9.73`. 172점을 단독 회수하려면 `s≈0.04481`가 필요한데 현재 V2024 실측은 `0.02691`이다.
- 형상은 quadratic만 보존했다. 로짓축은 quadratic/cubic 공간과 사실상 동일하고, 3차 고유분은 최대 `+1.63`.
- 27슬롯 기본 실행안과 축별 중단 조건·실패 조건을 HANDOFF에 넣었다.

2026 문헌까지 확인했지만, target sample이나 target importance weight 없이 새로 적용할 합법적 방법은 찾지 못했습니다. [PMLR 2026 distribution-shift theory](https://proceedings.mlr.press/v313/bhattacharjee26a.html), [PMLR 2025 covariate-shift risk control](https://proceedings.mlr.press/v266/almeida25a.html).