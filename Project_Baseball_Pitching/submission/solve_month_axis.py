# -*- coding: utf-8 -*-
"""month 분할축 프로브 결과 → 최적 오프셋 역산 (D-61).

    python submission/solve_month_axis.py --s0 1032.6717450165 --scores d1=1035.12,d5=1033.4

산수
----
직교정규 기저에서 방향 j의 계수를 c라 하면  `S(c) = S₀ + a_j·c − cost·c²`.
동봉 map은 `cost = 1.0`(설계가중 기준)이 되도록 스케일돼 있다. 프로브는 c=1 이므로

    a_j = ΔS_j + cost            (ΔS_j = S_j − S₀)
    최적 c*_j = a_j / (2·cost)
    회수액   G_j = a_j² / (4·cost)      · 방향이 직교하므로 G = Σ G_j

⚠ `cost`는 **평가셋의 월 분포**에 의존한다. 설계가중 기준 1.0이지만 train 시즌별로 재보면
0.63~1.25로 흔들린다(`check_month_probe.py` ③). `--cost`로 감도를 볼 것.
⚠ 방향별 부호: a_j < 0 이면 최적 c*도 음수다(반대 방향). 그대로 실으면 된다.
"""
from __future__ import annotations
import argparse

NDIR = 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s0", type=float, default=1032.6717450165, help="앵커 LB 점수")
    ap.add_argument("--scores", required=True, help="'d1=1035.12,d5=1033.4' — 측정한 방향만")
    ap.add_argument("--cost", type=float, default=1.0, help="방향당 무신호 비용")
    args = ap.parse_args()

    meas = {}
    for tok in args.scores.split(","):
        k, v = tok.split("=")
        j = int(k.strip().lstrip("dD"))
        if not 1 <= j <= NDIR:
            raise SystemExit("방향 번호는 1~%d" % NDIR)
        meas[j] = float(v)

    print("앵커 S₀ = %.10f · 비용 = %.4f\n" % (args.s0, args.cost))
    print("%-5s %16s %10s %10s %10s %10s"
          % ("방향", "실측 S_j", "ΔS_j", "a_j", "최적 c*", "회수액 G_j"))
    print("-" * 68)
    ts = [0.0] * NDIR
    G = 0.0
    for j in range(1, NDIR + 1):
        if j not in meas:
            continue
        ds = meas[j] - args.s0
        a = ds + args.cost
        c = a / (2 * args.cost)
        g = a * a / (4 * args.cost)
        ts[j - 1] = c
        G += g
        print("%-5s %16.10f %10.4f %10.4f %10.6f %10.4f" % ("d%d" % j, meas[j], ds, a, c, g))

    print("-" * 68)
    print("%-5s %16s %10s %10s %10s %10.4f" % ("합계", "", "", "", "", G))
    print()
    print("배포 예상 점수 = %.4f  (앵커 대비 %+.4f)" % (args.s0 + G, G))
    if G <= 0:
        print("🚫 회수액 0 이하 — 이 축은 조용하다. 배포하지 말 것.")
        return
    print()
    print("배포 명령:")
    print("  python submission/build_month_probe.py --deploy %s --tag moStar"
          % ",".join("%.10g" % t for t in ts))
    print()
    print("검증:")
    print("  python submission/verify_submission.py submission/dist/submit_moStar.zip --proxy 245789")
    print()
    print("⚠ 비용 감도 — 평가셋 월 분포가 설계와 다르면 G가 이만큼 움직인다:")
    for c_ in (0.75, 1.0, 1.25):
        g2 = sum(((meas[j] - args.s0) + c_) ** 2 / (4 * c_) for j in meas)
        print("     cost=%.2f → G=%7.4f  (배포 예상 %.4f)" % (c_, g2, args.s0 + g2))


if __name__ == "__main__":
    main()
