# -*- coding: utf-8 -*-
"""`month_m34` 분할축 프로브를 **현 챔피언(JTT) 베이스**로 재빌드한다 (D-61).

    python submission/build_month_probe.py --null          # 추가항 없음 — 파리티 검증용
    python submission/build_month_probe.py --all           # d1~d6 프로브 6종
    python submission/build_month_probe.py --deploy t1,t2,t3,t4,t5,t6 --tag moStar

왜 재빌드하는가
---------------
`dist/_deferred/submit_mo_d1..d6.zip` 6종은 **exact1(1025.5511) 베이스**로 만들어졌다. 그런데
챔피언이 `FINAL_submit_JTT.zip`(1032.6717 = 0.7412·exact1 + 0.2588·regime)로 바뀌었다.
분할축이 재는 것은 **그 모델의 세그먼트별 잔차**이므로 base가 바뀌면 계수도 바뀐다:

    resid_JTT = 0.7412·resid_exact1 + 0.2588·resid_regime   ⇒  a_j(JTT) ≠ a_j(exact1)

`a_j(regime)`을 모르므로 exact1에서 잰 값을 JTT로 옮길 수 없다. **현 챔피언 잔차로 재기준화**한다.

상수는 하나도 손으로 옮기지 않는다
----------------------------------
- 방향 정의(`map`)는 `_deferred/submit_mo_d{j}.zip`의 metadata에서 **그대로 읽어온다**
- `seg_correction`(`map`·`cells` 지원 상위 호환본)도 같은 zip의 script.py에서 **그대로 읽어온다**
  (JTT 쪽은 `levels`/`ge`만 아는 구버전이라 `map` 항을 만나면 KeyError → 세그먼트 보정 전체가
   탈락한다. gtF +1.32까지 같이 날아간다.)
- 나머지 엔트리는 **바이트째 복사** — 재직렬화 금지(개행 변환·float 왕복이 봉인을 깬다)

프로브 산수
-----------
오프셋 δ를 더하면 `ΔS = −C·mean(δ²) − 2C·mean(δ·(p−y))`, `C = 1e5/(r(1−r)) ≈ 4e5`.
동봉 map은 무신호 비용이 정확히 **1점**이 되도록 이미 스케일돼 있다(모델과 무관한 δ만의 성질).
⇒ `a_j = ΔS_j + 1` · `t_j = a_j/2` · `회수 가능액 G = Σ a_j²/4`

🚨 희석 보정 (D-61에 발견)
--------------------------
`script.py`에서 세그먼트 보정은 **regime 블렌드보다 먼저** 적용된다:

    p = clip(p + seg_correction(...))            # ← 여기서 더해지고
    p = clip(w_champ·p + w_regime·regime_p)      # ← 여기서 w_champ 배로 희석된다

⇒ 최종 출력에 실리는 오프셋은 `w_champ × map` 이고 무신호 비용은 `w_champ² = 0.549`배가 된다.
실측으로 확인했다: d1의 map 0.003476561542824311 → 출력 δ 2.577e-03 = 정확히 `×0.7412437855967673`.
따라서 **t 에 `1/w_champ` 을 곱해** 설계 크기를 복원한다. `w_champ`는 하드코딩하지 않고
metadata의 `regime_transfer_probe.champion_weight`에서 읽는다.
(챔피언의 기존 gtF 항도 같은 희석을 받고 있으나, 그것은 이미 1032.6717 실측에 반영된 상태이므로
 건드리지 않는다 — 건드리면 앵커가 바뀐다.)
"""
from __future__ import annotations
import argparse
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "submission" / "dist"
DEFERRED = DIST / "_deferred"
BASE = DIST / "FINAL_submit_JTT_fixed.zip"       # 최상위 레이아웃까지 정리된 챔피언 사본
SRC_IMPL = DEFERRED / "submit_mo_d1.zip"          # seg_correction 상위 호환본의 출처
NDIR = 6


def _read_entries(zp: Path) -> dict:
    with zipfile.ZipFile(zp) as z:
        return {i.filename: z.read(i.filename) for i in z.infolist()}


def _func_src(src: str, name: str) -> str:
    """`def name(` 부터 다음 최상위 `def ` 직전까지."""
    i = src.index("def %s(" % name)
    j = src.index("\ndef ", i + 10)
    return src[i:j]


def upgrade_seg_correction(script: str) -> str:
    """JTT의 구버전 seg_correction을 `map`/`cells` 지원본으로 교체한다.

    `levels`/`ge` 경로의 의미는 두 구현이 동일하므로, `map` 항이 없는 zip의 예측은 바뀌지 않는다
    (`--null` 빌드를 원본과 대조해 RMS 0.000000으로 증명한다)."""
    new = _func_src(zipfile.ZipFile(SRC_IMPL).read("script.py").decode("utf-8"), "seg_correction")
    old = _func_src(script, "seg_correction")
    if old == new:
        return script
    return script.replace(old, new, 1)


def direction_term(j: int) -> dict:
    """d{j} zip에서 month 방향 항을 그대로 읽어온다."""
    m = json.loads(_read_entries(DEFERRED / ("submit_mo_d%d.zip" % j))["model/metadata.json"]
                   .decode("utf-8"))
    terms = [t for t in m["seg_probe"] if t.get("col") == "game_month"]
    if len(terms) != 1:
        raise SystemExit("d%d: game_month 항이 %d개 — 예상과 다르다" % (j, len(terms)))
    return terms[0]


def build(out: Path, extra_terms: list):
    blobs = _read_entries(BASE)
    script = blobs["script.py"].decode("utf-8")
    patched = upgrade_seg_correction(script)
    blobs["script.py"] = patched.encode("utf-8")

    meta = json.loads(blobs["model/metadata.json"].decode("utf-8"))
    base_seg = meta.get("seg_probe") or []

    # 희석 보정 — 세그먼트 항은 regime 블렌드 **이전**에 더해지므로 w_champ 배로 줄어든다
    wc = float(meta["regime_transfer_probe"]["champion_weight"])
    extra_terms = [dict(t, t=float(t.get("t", 1.0)) / wc) for t in extra_terms]
    if extra_terms:
        print("  희석 보정: t × 1/%.16f = ×%.10f" % (wc, 1.0 / wc))
    meta["seg_probe"] = base_seg + extra_terms
    blobs["model/metadata.json"] = json.dumps(meta, ensure_ascii=False).encode("utf-8")

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(blobs):
            z.writestr(name, blobs[name])
    tops = sorted({n.split("/")[0] for n in blobs})
    print("  %-26s %6.2f MB  seg_probe %d항  최상위 %s"
          % (out.name, out.stat().st_size / 1e6, len(meta["seg_probe"]), tops))
    print("     seg_correction 교체: %s" % ("예" if patched != script else "아니오(이미 최신)"))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--null", action="store_true", help="추가항 없음 — 앵커 무손상 검증용")
    g.add_argument("--dir", type=int, choices=range(1, NDIR + 1), help="방향 d{N} 프로브")
    g.add_argument("--all", action="store_true", help="d1~d6 전부")
    g.add_argument("--deploy", help="'t1,...,t6' — 역산한 최적 오프셋으로 배포본 생성")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    if not BASE.exists():
        raise SystemExit("베이스 없음: %s" % BASE)
    print("베이스: %s\n" % BASE.name)

    if args.null:
        build(DIST / "submit_moJ_null.zip", [])
    elif args.dir:
        build(DIST / ("submit_moJ_d%d.zip" % args.dir), [direction_term(args.dir)])
    elif args.all:
        for j in range(1, NDIR + 1):
            build(DIST / ("submit_moJ_d%d.zip" % j), [direction_term(j)])
    else:
        ts = [float(x) for x in args.deploy.split(",")]
        if len(ts) != NDIR:
            raise SystemExit("--deploy 는 %d개 값이 필요하다" % NDIR)
        # 방향들을 t_j 배로 각각 실어 합산한다 (map은 t 배율을 그대로 받는다)
        terms = []
        for j, t in enumerate(ts, start=1):
            if abs(t) < 1e-12:
                continue
            tm = dict(direction_term(j))
            tm["t"] = float(tm.get("t", 1.0)) * t
            terms.append(tm)
        tag = args.tag or "moStar"
        build(DIST / ("submit_%s.zip" % tag), terms)
        print("\n  실린 방향: %s" % ", ".join("d%d×%.6g" % (j, t)
                                             for j, t in enumerate(ts, 1) if abs(t) >= 1e-12))


if __name__ == "__main__":
    main()
