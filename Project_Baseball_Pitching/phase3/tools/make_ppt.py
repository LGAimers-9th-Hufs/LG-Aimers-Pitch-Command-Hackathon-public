# -*- coding: utf-8 -*-
"""솔루션 PPT 생성 (python-pptx).

    python phase3/tools/figures.py      # 먼저 그림을 만든다
    python phase3/tools/make_ppt.py     # -> phase3/ABS_kkangtongzone_solution.pptx

구성은 `phase3/PPT_OUTLINE.md`, 원고는 `phase3/SOLUTION_OUTLINE.md`.
슬라이드당 주장 1개 + 실측 수치 1~2개. 발표자 노트에 원고의 해당 문단을 넣는다.
수치는 전부 원고에서 출처가 확인된 것만 쓴다.
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent.parent
FIG = ROOT / "phase3" / "figures"
OUT = ROOT / "phase3" / "ABS_kkangtongzone_solution.pptx"

FONT = "맑은 고딕"
INK = RGBColor(0x0B, 0x0B, 0x0B)
INK2 = RGBColor(0x52, 0x51, 0x4E)
INK3 = RGBColor(0x7C, 0x7B, 0x76)
ACCENT = RGBColor(0x2A, 0x78, 0xD6)
WARN = RGBColor(0xEB, 0x68, 0x34)
PAPER = RGBColor(0xFC, 0xFC, 0xFB)
RULE = RGBColor(0xE5, 0xE4, 0xE0)

W, H = Inches(13.333), Inches(7.5)

TEAM = "ABS깡통존"
MEMBERS = "(팀원 실명 3인 — TEAM_REQUEST.md 회신 후 채운다)"

# (제목, 리드 한 줄, 불릿들, 근거 수치 한 줄, 그림 파일 or None, 발표자 노트)
SLIDES = [
    ("표지", None, [], None, None, None),   # 1. 별도 처리

    ("문제의 기하 — Brier가 무엇을 보상하는가",
     "레벨 정합이 신호 발굴보다 먼저다.",
     ["Brier는 제곱오차라 분해가 정확하다: 신뢰도 + 분해능 − 불확실성",
      "예측 평균이 δ 어긋날 때 비용 ≈ 4·10⁵·δ² — 2차로 커진다",
      "첫 제출 269점의 해부: 시즌 성공률 하락을 그대로 외삽한 레벨 오판 −168,"
      " 거기에 isotonic이 분해능을 −186 파괴했다. 무보정이었다면 ~641점이었다."],
     "교훈 1 — 확률 예측 대회에서 캘리브레이션은 후처리가 아니라 설계다.",
     "fig1_brier_geometry.png",
     "지표의 기하부터 말한다. 우리 첫 제출이 269점이었고, 그 원인이 모델이 아니라 레벨이었다는 것이 "
     "이 프로젝트 전체의 방향을 정했다."),

    ("규정이 설계를 결정한다",
     "test 행 독립 원칙이 우리 피처 설계를 전부 규정했다.",
     ["규칙 §2-4: 평가 데이터의 다른 행이나 전체 분포로 예측값을 보정할 수 없다",
      "⇒ 참가자가 직접 만드는 as-of 이력 피처는 2025에서 계산 자체가 불가능하다",
      "유일한 이력 소스 = 주최가 준 공식 asof 19컬럼 + train에서 만든 고정 룩업의 행 단위 조회"],
     "새 피처 제안마다 첫 질문: “이 값을 2025 test의 한 행만 보고 계산할 수 있는가?”",
     None,
     "규정을 제약이 아니라 설계 입력으로 다뤘다는 점을 강조한다."),

    ("⭐ 새로운 시각 ① — 당해 시즌 폼 분해",
     "주최가 준 카운터는 평가 기간 안에서도 갱신된다.",
     ["asof_pitcher_n은 커리어 누적이고 한 행에서 k = round(rate × n)이 정확히 복원된다",
      "공개 test 표본 실증: 투수 21813의 asof_n 3,465 vs train 2024년 말 누적 3,085 — 2025에 이미 380구",
      "train으로 만든 직전 시즌 말 누적 (n_base, k_base)를 동봉해 행 단위로 뺀다 → 당해 시즌분만 남는다"],
     "공개 리더보드 911 → 953 (+41.9). 프로젝트 전체에서 단일 최대 신호였다.",
     "fig2_inseason_gate.png",
     "이 슬라이드가 가산점 축의 첫 번째다. 합법적으로만 얻을 수 있는 신호이고, 규정을 정확히 읽은 "
     "결과로 나왔다는 점을 말한다."),

    ("①의 야구적 해석 — 제구력은 커리어 평균이 아니라 올해의 폼이다",
     "is_delta = 올해 이 투수는 자기 평소보다 잘 넣고 있는가.",
     ["부상 복귀·메커닉 변화·나이 곡선은 시즌 단위로 반영되는데 커리어 누적은 그것을 평활해 버린다",
      "트리가 스스로 못 만든다 — pitcher_id가 피처셋에서 제거돼 있어 투수별 기준선을 세울 수 없다",
      "베테랑은 당해분이 커리어의 10% 남짓이라 누적률에 희석된다"],
     "3폴드 +92.5 / +190.3 / +86.1 (평균 +123, 폴드 간 SD 58.4). 같은 크기의 위약은 붙지 않았다.",
     None,
     "야구 도메인 언어로 다시 말해 심사위원이 '새로운 제구력 평가 시각'으로 읽게 한다."),

    ("⭐ 새로운 시각 ② — 제구는 조준과 산포의 2물리량, 그리고 성공 영역은 도넛이다",
     "너무 정확하면 한복판에 꽂혀 실패한다.",
     ["라벨은 심판 콜이 아니라 위치 기반 — 한복판·크게 벗어남·포수 요구 반대가 실패다",
      "조준점 a와 산포 σ의 2차원 정규분포로 두면 도넛 확률이 Marcum Q 함수의 차로 닫힌 형식이 된다",
      "원판이 목표면 ∂P/∂σ < 0 이지만, 도넛에서는 조준이 한복판에 가까울 때 부호가 뒤집힌다"],
     "현상은 실재한다 — 3볼 카운트의 σ 기울기 +0.29, z = 3.48 (사전 관측검정 통과).",
     None,
     "축 정렬 분할로는 재현하기 어려운 구조라는 점을 말한다. 다음 슬라이드에서 '그럼에도 싣지 않았다'로 넘긴다."),

    ("②의 정직한 판정 — 물리적으로 옳아도 증분이 0이면 싣지 않는다",
     "게이트를 통과하지 못한 것은 최종 산출물에 넣지 않았다.",
     ["시간분할 CV에서 −7.8, 폴드 가드 전패",
      "asof 율에서 유도되는 어떤 피처도(재수축·probit·물리 역산) 트리 위에서 증분이 0이었다",
      "트리가 원시 컬럼에서 이미 같은 정보를 뽑고 있었다는 뜻이다"],
     "교훈 — 물리적으로 옳은 이야기가 예측 증분을 보장하지 않는다. 판정은 게이트에 맡겼다.",
     None,
     "'현상의 실재'와 '탑재 가치'는 다른 판정이라는 것이 우리 발표의 일관된 프레임이다."),

    ("⭐ 새로운 시각 ③ — 익명 ID 사이의 엔티티 매칭",
     "교집합이 0인 두 ID 체계를 커리어 시그니처로 연결했다.",
     ["train의 익명 pitcher_id 792명 ↔ 공식 trackman_history의 pitcher_trackman_id 906명, 교집합 0",
      "행 단위 매칭은 불가능함을 먼저 증명했다(유일 식별 블록 1.9%)",
      "구종 믹스·투구 손·등판 구조·투구량으로 커리어 지문을 만들고 Hungarian 알고리즘으로 배정"],
     "4중 검증 통과 — 팀×손 permutation z = 26.2 · 두 방법 합치 92.9% · Tier1 행가중 78.2% ·"
     " 팀 컬럼 없이 10개 프랜차이즈 블록이 순도 ~99%로 분리",
     None,
     "익명화된 데이터셋에 외부 로그를 결합하는 일반적 방법론이다. 다만 그 위에 세운 프로필 피처가 "
     "게이트를 넘지 못해 최종 제출물에는 싣지 않았다."),

    ("⭐ 새로운 시각 ④ — 검증 프로토콜 자체가 산출물이다",
     "난수 6컬럼이 2SE 게이트를 통과한다.",
     ["균등 난수 6컬럼을 진짜 피처와 같은 결측 패턴·커버리지로 넣어 게이트를 돌렸다",
      "val 2022에서 +11.2 ± 9.9 — 폴드 내 2SE 기준으로는 '통과'였다",
      "컬럼을 추가하는 행위 자체가 폴드 간 ±13의 변동을 만든다 (feature_fraction 추첨과 분할 경쟁이 바뀐다)"],
     "⇒ 채택 기준을 “3폴드 평균 > 0 그리고 평균 > 폴드 간 SD”로 바꾸고,"
     " 모든 새 arm에 같은 컬럼 수의 난수 위약을 붙였다.",
     "fig3_placebo.png",
     "이 슬라이드가 가산점 축의 네 번째다. 방법론 자체가 재사용 가능한 산출물이라는 주장."),

    ("누수 방지의 기계적 증명 — 타깃 셔플 플라시보",
     "실데이터 없이도 누수 부재를 증명할 수 있다.",
     ["as-of 통계를 전 시즌에 한 번에 계산하면 검증 시즌 행이 같은 시즌의 미래 타깃을 본다",
      "폴드별 재계산 + 검증 시즌 엔티티 통계 동결로 수정",
      "증명: 검증 라벨을 무작위로 섞어도 수정판의 피처는 비트 단위로 불변, 누수판은 변한다"],
     "sweep/test_regulation.py가 이 검사를 자동으로 돌린다 — 행 독립성·파리티·폴드 분리 전부 통과.",
     None,
     "규정 준수를 '문서'가 아니라 '기계 검사'로 강제한다는 우리 원칙의 첫 사례다."),

    ("모델 — 무엇이 실제로 값을 했나",
     "앵커는 평범하다. 값을 한 것은 학습 창과 지표 직접 최소화다.",
     ["LightGBM 2종(num_leaves 7·15) + ExtraTrees + 선형 멤버의 확률 평균, objective=regression"
      " (Brier가 곧 제곱오차이므로 지표를 직접 최소화)",
      "피처 78개 = 공식 44 + 범주형 3 + 파생 6 + 당해시즌 4 + 타자 당해분 6 + pEB 3 + cxp 12",
      "season·pitcher_id·batter_id는 제거 — 그런 컬럼 하나가 2025에서 −192점인 것을 실측했다"],
     "학습 시즌: fit 2023 → val 2024에서 2023 단독 707.9 vs 2019~2023 전체 319.0 — 530점 차.",
     None,
     "다만 이것은 법칙이 아니라 레짐 베팅이다. fit 2022 → val 2023에서는 부호가 완전히 뒤집혀 "
     "최신 1시즌이 꼴찌였다(−1215 vs −467). 2025는 맞춘 것이지 통제한 것이 아니다."),

    ("채널 상한을 먼저 재는 규율",
     "오라클 상한이 현 점수보다 낮으면 그 채널은 파지 않는다.",
     ["투수 해상도 오라클(그 시즌 실제 성공률을 안다고 가정) = 723.9",
      "투수 + 타자 = 757.9 · 타자 단독 78.2 · 상황(카운트·아웃·주자) 24.6",
      "실제 앙상블 = 773.5 — 우리 점수가 이미 투수 오라클보다 높다"],
     "⇒ asof_*가 식별 정보를 이미 다 담고 있다. 투수 ID·계층 모델·랜덤 효과 계열은 팔 곳이 없다."
     " 이 반나절짜리 측정이 며칠을 아꼈다.",
     None,
     "판별 질문: 이 값이 한 투수의 한 시즌 안에서 행마다 변하는가? 아니오면 포화 채널이다."),

    ("최신 방법도 같은 게이트를 통과해야 한다",
     "성능이 좋아도 규정에 못 들어가는 아키텍처가 있다.",
     ["TabPFN-3 (2026 공개 표형 파운데이션 모델): 같은 맥락 LGBM 대비 쌍대 3폴드 전패 (−433 / −2120 / −933)",
      "TabICLv2 (BSD-3·순수 합성 사전학습): 쌍대 +133 ± 226 — 파운데이션 모델이 정확도 축을 통과한 첫 사례",
      "그런데 단독 예측 vs 배치 예측이 4.8e-2 어긋나고 같은 배치 반복은 0.0 — 결정론적 test 행 간 참조 = §2-4 위반"],
     "⇒ 행 독립 3중 프로브(단독 / 배치 / 반복)를 신규 모델의 표준 선행검사로 채택했다.",
     None,
     "라이선스·속도(한도의 8배)도 걸렸지만 결정적인 것은 규정이었다. 아키텍처 수준의 성질이라 "
     "구성을 바꿔도 피할 수 없다."),

    ("앙상블 — 확률 평균의 이득은 항등식이다",
     "적합이 아니라 대수다. 그래서 사전에 예측할 수 있다.",
     ["Score(p̄) = 평균Score + C·E_i[(p_i − p̄)²],  C = 1e5/(r(1−r)) ≈ 4·10⁵",
      "최적 가중 w* = 0.5 + d/(2K) — d가 작으면 0.5에 붙는다 ⇒ 균등 가중이 학습 데이터만으로 정당화된다",
      "파트너는 점수가 아니라 불일치(RMS)로 고른다"],
     "blendA3에서 ρ = 42.423/29.677 = 1.4295를 측정한 뒤, 사전 예측은 blendF5 한 번뿐이었고"
     " 실현율이 94.6%였다.",
     "fig6_identity.png",
     "정직하게 그렸다. A3는 ρ를 맞춘 점, D4는 S(tm3L)을 역산한 점이라 정의상 대각선 위에 있다. "
     "진짜 시험은 F5 하나였다."),

    ("다양성의 상한 — 정직한 d의 벽",
     "좋은 모델은 같은 정보의 같은 E[y|x]로 간다.",
     ["레그 추가의 이득 = 0.8·gain₄ + 64000·d²  (d = 새 레그와 기존 평균의 RMS)",
      "설계 축을 전부 뒤집은 독립 파이프라인도 불일치가 0.0134로 수렴했다",
      "우리 코드를 보지 않고 만든 파이프라인만 0.026~0.034를 냈다 — 최종 제출물이 세 사람 것인 이유"],
     "유능한 레그의 정직한 d는 0.024~0.029에 5중 수렴. 전수 열거 4,082조합의 유일 양수 후보까지"
     " 실측해 점수 추구를 산수로 종료했다.",
     "fig5_breakeven.png",
     "측정 규약의 함정도 여기서 나왔다 — d는 배포본과 같은 학습 창의 예측으로 재야 한다. "
     "학습 창이 다른 모델을 섞어 재면 d가 ~1.5배 부풀고 기대점수가 허상이 된다."),

    ("규정 준수 — 우리가 저지른 것과 시정",
     "불리한 사실을 먼저 말한다.",
     ["공개 리더보드 점수로 캘리 상수와 블렌드 가중치를 역산해 박은 제출물이 있었다"
      " (§5 “평가 데이터 전체를 보고 만든 사후 보정값”)",
      "팀 내부 재감사가 발견했고, 우리는 규정 원문과 대조해 판정이 옳음을 확인했다",
      "시정: 해당 계보를 최종 산출물에서 전면 배제 (공개 점수 약 17점 반납)"],
     "기계 검사기 scan_probe_provenance.py를 만들어 모든 신규 제출에 강제한다 — BLOCK 0건이 조건.",
     None,
     "경계선: 사전 등록한 후보를 제출해 점수 좋은 쪽을 남기는 것(선택)은 리더보드의 존재 이유다. "
     "점수에서 최적값을 풀어 상수로 박는 것(적합)은 모델을 평가 라벨의 함수로 만든다. "
     "우리는 이 구분을 문서로 갖고 있으면서도 한 번 넘었다. 규율은 기계 검사로 강제해야 한다."),

    ("최종 산출물 — 세 사람의 서로 다른 파이프라인 4레그",
     "레그를 바이트 그대로 담고, 가중은 균등 1/4로 고정했다.",
     ["clookup 1049.46 (팀원) · cmoe 1027.53 (팀원) · physmix 1009.26 (본 저장소) · tm3L (역산 940.31)",
      "각 레그의 script.py는 한 줄도 고치지 않았다 — train/serve skew가 정의상 0이다",
      "블렌드 층은 레그 출력을 row_id로 병합해 평균할 뿐, 행 간 참조가 없다"],
     "submit_blendD4.zip = 1058.6047851923. 재현 검증: 모델 가중치 비트 일치,"
     " 245,789행 예측 대조 RMS 0.000000.",
     "fig4_disagreement.png",
     "가장 많이 어긋나는 쌍이 가장 많이 벌어준다는 것을 행렬로 보여준다. "
     "cmoe↔tm3L의 0.0306이 이 블렌드의 이득 대부분을 만든다."),

    ("한계와 다음",
     "우리 상한은 규정을 지킨 상태에서의 상한이다.",
     ["레짐 의존성이 최대 미측정 리스크 — 최신 1시즌 학습과 동결 캘리는 2025가 안정 레짐이라는 데 건 베팅이다",
      "세그먼트 × 당해 시즌 오라클: 평가 라벨을 보면 +206점, 규정 안에서 실현하면 −17점"
      " — 그 차이가 곧 §5 위반으로만 얻어지는 몫이다",
      "벽을 옮기는 것은 모델링이 아니라 새 정보 축(예: 투구 위치 컬럼)뿐이다"],
     "오라클로 닫은 채널 13층을 문서로 남겼다 — 같은 데이터셋을 다시 다루는 사람이 며칠을 아낄 수 있다.",
     None,
     "파단 레짐이었다면 −1,200점대를 맞을 수 있었다는 것을 숨기지 않는다."),
]


def set_bg(slide):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = PAPER


def add_text(slide, left, top, width, height, runs, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP, line=1.25):
    """runs = [(text, size, bold, color, space_after_pt), ...]  단락 하나씩."""
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for i, (text, size, bold, color, after) in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line
        p.space_after = Pt(after)
        r = p.add_run()
        r.text = text
        r.font.name = FONT
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
    return tb


def add_rule(slide, left, top, width):
    from pptx.enum.shapes import MSO_SHAPE
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, Emu(9525))
    s.fill.solid()
    s.fill.fore_color.rgb = RULE
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def title_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_text(slide, Inches(1.0), Inches(1.5), Inches(11.3), Inches(1.0),
             [("LG Aimers 9기 × LG 트윈스 해커톤", 16, False, INK2, 0)])
    add_text(slide, Inches(1.0), Inches(2.0), Inches(11.3), Inches(1.4),
             [("투수 제구 성공 확률 예측", 40, True, INK, 0)])
    add_rule(slide, Inches(1.0), Inches(3.35), Inches(11.3))
    add_text(slide, Inches(1.0), Inches(3.6), Inches(11.3), Inches(1.8),
             [("점수를 만든 것은 모델 아키텍처가 아니라 세 가지였다", 20, True, ACCENT, 10),
              ("① 평가 지표의 기하 — 레벨 정합이 신호 발굴보다 먼저다", 15, False, INK, 4),
              ("② 주최가 준 카운터가 평가 기간 안에서도 갱신된다는 사실 — 당해 시즌 폼을 합법적으로 복원했다",
               15, False, INK, 4),
              ("③ 모델을 늘리는 대신 서로 다르게 틀리는 모델을 평균하는 것 — 그 이득은 정확한 항등식이다",
               15, False, INK, 0)])
    add_text(slide, Inches(1.0), Inches(5.9), Inches(11.3), Inches(1.2),
             [("팀 %s  ·  %s" % (TEAM, MEMBERS), 13, False, INK2, 4),
              ("최종 제출물 submit_blendD4.zip — 공개 리더보드 1058.6047851923", 13, True, INK, 4),
              ("2026-09  ·  코드·PPT 제출본", 12, False, INK3, 0)])
    return slide


def content_slide(prs, n, title, lead, bullets, evidence, figure, notes):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_text(slide, Inches(0.7), Inches(0.42), Inches(11.9), Inches(0.9),
             [(title, 26, True, INK, 0)])
    add_rule(slide, Inches(0.7), Inches(1.32), Inches(11.9))
    y = Inches(1.5)
    if lead:
        add_text(slide, Inches(0.7), y, Inches(11.9), Inches(0.5),
                 [(lead, 17, True, ACCENT, 0)])
        y = Inches(2.05)

    has_fig = figure is not None and (FIG / figure).exists()
    text_w = Inches(6.0) if has_fig else Inches(11.9)
    runs = [("· " + b, 14, False, INK, 9) for b in bullets]
    add_text(slide, Inches(0.7), y, text_w, Inches(3.6), runs, line=1.3)

    if has_fig:
        # 가로 5.85in · 세로 4.30in 상자에 비율을 지켜 맞춘다 (하단 규칙선 6.05in을 넘지 않게)
        from PIL import Image
        iw, ih = Image.open(FIG / figure).size
        box_w, box_h = 5.85, 4.30
        scale = min(box_w / iw, box_h / ih)
        w_in, h_in = iw * scale, ih * scale
        left = Inches(7.0 + (box_w - w_in) / 2)
        top = Inches(1.55 + (box_h - h_in) / 2)
        slide.shapes.add_picture(str(FIG / figure), left, top,
                                 width=Inches(w_in), height=Inches(h_in))

    if evidence:
        add_rule(slide, Inches(0.7), Inches(6.05), Inches(11.9))
        add_text(slide, Inches(0.7), Inches(6.18), Inches(11.9), Inches(0.9),
                 [(evidence, 13, True, WARN, 0)], line=1.25)

    add_text(slide, Inches(12.35), Inches(6.95), Inches(0.6), Inches(0.35),
             [(str(n), 11, False, INK3, 0)], align=PP_ALIGN.RIGHT)
    if notes:
        slide.notes_slide.notes_text_frame.text = notes
    return slide


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    missing = [s[4] for s in SLIDES if s[4] and not (FIG / s[4]).exists()]
    if missing:
        print("   [경고] 그림 없음(먼저 figures.py 실행): %s" % missing)

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    title_slide(prs)
    for i, (title, lead, bullets, evidence, figure, notes) in enumerate(SLIDES[1:], start=2):
        content_slide(prs, i, title, lead, bullets, evidence, figure, notes)
    prs.save(OUT)
    print(">> 완성 %s  (슬라이드 %d장, %.2f MB)"
          % (OUT, len(prs.slides._sldIdLst), OUT.stat().st_size / 1e6))


if __name__ == "__main__":
    main()
