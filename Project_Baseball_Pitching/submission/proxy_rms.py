# -*- coding: utf-8 -*-
"""시점 정합 프록시로 제출물 간 예측 불일치(RMS)를 측정한다 — **제출 슬롯 0개**.

    python submission/proxy_rms.py --rows 30000

왜 필요한가 (D-61)
------------------
블렌드 이득은 `이득 = (K+d)²/(4K)`, `K = 1e5·mean(Δ²)/(r(1−r)) ≈ 4e5·RMS²`, `d = S₃ − S_anchor`.
즉 **파트너는 점수가 아니라 불일치로 고른다.** 그런데 RMS를 2025에서 잴 수 없으므로 2024 행을
프록시로 쓴다.

⚠ 그냥 돌리면 **틀린다.** 동봉 `inseason_base`는 2025 추론용(= 2024 시즌 말 누적)이라 2024 행에
적용하면 `is_n = max(asof_pitcher_n − base, 0)`이 **전 행 0으로 붕괴**한다(실측 100%). 당해 시즌
분해(is4) 블록이 죽은 채로 비교하게 되고, 실제로 `RMS(regime, champ)`가 0.024404로 나왔으나
2025 실측은 0.016306이었다(K 기준 2.24배 과대).

⇒ 2024 행에는 **2023 시즌 말 base**를 넣어야 2025 서빙(직전 시즌 말 base + 당해 누적)과 동형이다.
룩업 생성은 `sweep/inseason.py`의 `career_end_lookup`을 그대로 쓴다.

게이트
------
패치된 프록시로 regime 레그를 역산해 `RMS(regime, champ)`를 재고 **2025 실측 0.016306**과 대조한다.
오차 15% 초과면 프록시를 믿지 않고 LB 프로브로 직행한다.
"""
from __future__ import annotations
import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "submission" / "dist"
sys.path.insert(0, str(ROOT / "sweep"))

PROXY_SEASON = 2024                 # 프록시 행이 속한 시즌 (base는 2023 말이 된다)
C_SCORE = 4e5                       # 1e5/(r(1−r)), r≈0.494
S_ANCHOR = 1032.6717450165          # FINAL_submit_JTT.zip LB 실측
W_CHAMP = 0.7412437855967673        # JTT 안의 챔피언 가중치
W_REGIME = 0.25875621440323276
RMS_2025_REF = 0.016306             # LB 3점으로 측정된 RMS(regime, champ) — 게이트 기준
GATE_TOL = 0.15

# (표시명, zip 경로, LB 실측 점수 or None)
ZIPS = [
    ("exact1",       DIST / "submit_exact1.zip",             1025.5511136843),
    ("JTT",          DIST / "FINAL_submit_JTT.zip",          S_ANCHOR),
    ("ens7_bis_tm",  DIST / "submit_ens7_bis_tm.zip",        1003.5142600363),
    ("ens4_is_k100", DIST / "submit_ens4_inseason_k100.zip",  984.0806860421),
    ("june853",      DIST / "aimers_sub_june_verzip.zip",     853.5697653812),
    ("baselineRF",   ROOT / "data" / "baseline_submit.zip",   549.5119345223),
]


# --------------------------------------------------------------------------- 룩업
def _last_row_per_season(df: pd.DataFrame, id_col: str, n_col: str, rate_cols: list):
    """{(id, season): [n, k₁, k₂...]} — 시즌 안에서 `n_col`이 최대인 행의 누적값.

    `career_end_lookup`(투수)과 달리 마지막 투구 자신을 더하지 않는다. 라벨이 있는 것은
    control_success 하나뿐이라 middle 계열은 더할 수 없고, 1투구 차이는 RMS 진단에 무의미하다.
    """
    n = pd.to_numeric(df[n_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    d = pd.DataFrame({"i": df[id_col].to_numpy(), "s": df["season"].to_numpy(), "n": n})
    for j, rc in enumerate(rate_cols):
        r = pd.to_numeric(df[rc], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        d["k%d" % j] = np.rint(r * n)
    last = d.loc[d.groupby(["i", "s"])["n"].idxmax()]
    ks = ["k%d" % j for j in range(len(rate_cols))]
    return {(int(i), int(s)): [float(nn)] + [float(v) for v in row]
            for i, s, nn, row in zip(last["i"], last["s"], last["n"], last[ks].to_numpy())}


def _base_before(lut: dict, season: int) -> dict:
    """{str(id): [...]} — `season` **미만**에서 가장 최근 시즌 말 누적. 없으면 키 자체를 뺀다
    (서빙 코드가 '룩업에 없음 = 신인 = (0,0)'으로 처리하므로 동형이다)."""
    by_id = {}
    for (i, s), v in lut.items():
        if s < season:
            by_id.setdefault(i, {})[s] = v
    return {str(i): list(h[max(h)]) for i, h in by_id.items()}


def build_proxy_bases(df: pd.DataFrame, season: int):
    """프록시 시즌용 (투수 base, 타자 base)."""
    import inseason                                     # sweep/inseason.py
    # 투수: 실제 빌더와 같은 함수를 쓴다(마지막 투구 +1 보정 포함)
    pit_lut = inseason.career_end_lookup(df)
    pit = _base_before({k: list(v) for k, v in pit_lut.items()}, season)
    bat_lut = _last_row_per_season(
        df, "batter_id", "asof_batter_n",
        ["asof_batter_success_rate", "asof_batter_middle_rate"])
    bat = _base_before(bat_lut, season)
    return pit, bat


# --------------------------------------------------------------------------- 패치
def patch_meta(work: Path, pit: dict, bat: dict) -> list:
    """work/ 안의 metadata를 프록시 시점으로 교체. 존재하는 키만 건드린다."""
    touched = []
    mpath = work / "model" / "metadata.json"
    if mpath.exists():
        m = json.loads(mpath.read_text(encoding="utf-8"))
        if "inseason_base" in m:
            m["inseason_base"] = {k: v[:2] for k, v in pit.items()}
            touched.append("inseason_base")
        if "inseason_batter" in m:
            m["inseason_batter"] = {k: v[:3] for k, v in bat.items()}
            touched.append("inseason_batter")
        if isinstance(m.get("pkg"), dict) and "peb_base" in m["pkg"]:
            m["pkg"]["peb_base"] = {k: v[:2] for k, v in pit.items()}
            touched.append("pkg.peb_base")
        mpath.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")

    rpath = work / "model" / "regime_metadata.json"
    if rpath.exists():
        r = json.loads(rpath.read_text(encoding="utf-8"))
        if "pitcher_inseason_base" in r:
            r["pitcher_inseason_base"] = {k: v[:2] for k, v in pit.items()}
            touched.append("regime.pitcher")
        if "batter_inseason_base" in r:
            r["batter_inseason_base"] = {k: v[:3] for k, v in bat.items()}
            touched.append("regime.batter")
        rpath.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
    return touched


def run_zip(name: str, zpath: Path, sub: pd.DataFrame, tmp: Path,
            pit: dict, bat: dict, id_col: str, target: str, patch: bool):
    work = tmp / name
    work.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(work)
    touched = patch_meta(work, pit, bat) if patch else []
    (work / "data").mkdir(exist_ok=True)
    test = sub.drop(columns=[target])
    test.to_csv(work / "data" / "test.csv", index=False, encoding="utf-8")
    pd.DataFrame({id_col: test[id_col], target: 0.5}).to_csv(
        work / "data" / "sample_submission.csv", index=False, encoding="utf-8")
    r = subprocess.run([sys.executable, "script.py"], cwd=work, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=3600)
    out = work / "output" / "submission.csv"
    if r.returncode != 0 or not out.exists():
        print("   !! 실패 rc=%s\n   %s" % (r.returncode, (r.stderr or "")[-700:].replace("\n", "\n   ")))
        return None
    for ln in r.stdout.splitlines():
        if "!!" in ln:
            print("   ⚠ " + ln)
    print("   패치: %s" % (", ".join(touched) if touched else "(해당 키 없음)"))
    return pd.read_csv(out)[target].to_numpy(dtype=float)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=30000)
    ap.add_argument("--no-patch", action="store_true", help="패치 없이(=망가진 프록시) 대조용")
    args = ap.parse_args()

    import real_data as rd
    print(">> train 로드 중...")
    df = rd.load_train()
    print(">> %d 시즌 base 룩업 생성 중..." % (PROXY_SEASON - 1))
    pit, bat = build_proxy_bases(df, PROXY_SEASON)
    print("   투수 %d명 · 타자 %d명" % (len(pit), len(bat)))

    sub = df[df[rd.SEASON] == PROXY_SEASON].head(args.rows).reset_index(drop=True)
    print(">> 프록시 = %d 시즌 %s행\n" % (PROXY_SEASON, f"{len(sub):,}"))

    # 정합성 즉시 확인: is_n 이 살아 있는가
    pn = sub["asof_pitcher_n"].to_numpy(dtype=float)
    nb = np.array([pit.get(str(int(p)), [0.0])[0] for p in sub["pitcher_id"]], dtype=float)
    isn = np.maximum(pn - nb, 0.0)
    print("   is_n == 0 비율 : %.1f%%  (패치 전에는 100.0%%)" % (100 * (isn == 0).mean()))
    print("   is_n 중앙값    : %.0f\n" % np.median(isn))

    tmp = Path(tempfile.mkdtemp(prefix="proxyrms_"))
    preds = {}
    try:
        for name, zp, _ in ZIPS:
            if not zp.exists():
                print("[skip] %s — 파일 없음" % name)
                continue
            print("[run ] %s" % name)
            p = run_zip(name, zp, sub, tmp, pit, bat, rd.ID, rd.TARGET, not args.no_patch)
            if p is not None:
                preds[name] = p

        champ, jtt = preds.get("exact1"), preds.get("JTT")
        print("\n" + "=" * 76)
        print("🚦 게이트 — regime 레그 역산 후 2025 실측 RMS와 대조")
        print("=" * 76)
        ratio = None
        if champ is not None and jtt is not None:
            regime = (jtt - W_CHAMP * champ) / W_REGIME
            rms = float(np.sqrt(((regime - champ) ** 2).mean()))
            ratio = rms / RMS_2025_REF
            print("  프록시 RMS(regime, champ) : %.6f" % rms)
            print("  2025 실측 (LB 3점)        : %.6f" % RMS_2025_REF)
            print("  비율                      : %.3f×  (곡률 K 기준 %.2f×)" % (ratio, ratio ** 2))
            ok = abs(ratio - 1.0) <= GATE_TOL
            print("  판정                      : %s (허용 ±%.0f%%)"
                  % ("✅ 통과 — 프록시 신뢰" if ok else "🚫 실패 — LB 프로브로 직행", 100 * GATE_TOL))
        else:
            print("  exact1 또는 JTT 실행 실패 — 게이트 판정 불가")

        print("\n" + "=" * 76)
        print("세 번째 레그 후보 — JTT 앵커와의 RMS 및 채산성")
        print("=" * 76)
        print("%-14s %10s %10s %10s %9s %10s %10s"
              % ("후보", "LB", "RMS", "손익분기", "K", "이득", "보정이득"))
        print("-" * 76)
        for name, _zp, s3 in ZIPS:
            if name in ("exact1", "JTT") or name not in preds or s3 is None:
                continue
            rms = float(np.sqrt(((preds[name] - jtt) ** 2).mean()))
            d = s3 - S_ANCHOR
            need = math.sqrt(-d / C_SCORE)
            K = C_SCORE * rms ** 2
            g = (K + d) ** 2 / (4 * K) if K > -d else None
            Kc = K / (ratio ** 2) if ratio else None      # 게이트 비율로 보정한 곡률
            gc = ((Kc + d) ** 2 / (4 * Kc)) if (Kc and Kc > -d) else None
            print("%-14s %10.2f %10.6f %10.6f %9.1f %10s %10s"
                  % (name, s3, rms, need, K,
                     ("%+.2f" % g) if g else "손해",
                     ("%+.2f" % gc) if gc else ("손해" if Kc else "—")))
        print("\n앵커 = JTT %.4f · 이득 = (K+d)²/(4K) · 보정이득 = 게이트 비율로 K를 나눈 것" % S_ANCHOR)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
