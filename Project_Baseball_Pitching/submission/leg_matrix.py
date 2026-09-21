# -*- coding: utf-8 -*-
"""여러 제출 zip을 같은 프록시 행에 돌려 **레그 간 불일치(RMS) 행렬**과 블렌드 채산성을 낸다.

    python submission/leg_matrix.py --auto --rows 30000
    python submission/leg_matrix.py --leg calP=submission/dist/submit_ens9_calP.zip \
                                    --leg cregime=partners/.../clean_regime/submission.zip \
                                    --anchor cregime --rows 30000

왜 RMS인가 (D-61)
-----------------
Brier 채점에서 확률 평균의 점수는 **정확한 항등식**이다(적합이 아니라 대수):

    Score(p̄) = 평균Score + 1e5·RMS²/(4V),     V = r(1−r),  C = 1e5/V ≈ 4e5
    이득(w)   = w·d + K·w(1−w),                d = S_B − S_A,  K = 4e5·RMS²
    손익분기   RMS > √(|d|/4e5)

⇒ **파트너는 점수가 아니라 불일치로 고른다.** 점수가 앵커에 가까울수록 문턱이 급격히 낮아진다.

⚠ 시점 정합 (D-61에서 K를 2.24배 과대평가했던 함정)
----------------------------------------------------
동봉 `inseason_base`가 2025 추론용(2024 시즌 말 누적)이면 2024 프록시 행에서 `is_n`이 전부 0으로
붕괴한다. `proxy_rms.build_proxy_bases`가 2023 말 base로 갈아끼운다. 우리 패키지 키만 패치되므로,
**팀원 패키지에 시즌 의존 룩업이 있으면 그 레그의 RMS는 왜곡된다** — `--audit`이 후보를 표시한다.

⚠ 가중치를 LB로 역산하지 말 것
------------------------------
`w*`는 **참고값**이다. 점수에서 `w`를 풀어 배포하면 §5 "평가 데이터 전체를 보고 만든 사후 보정값"
위반이다(팀 재감사가 `regime_transfer_probe`를 그 이유로 금지). 사전 등록 후보 중 LB로 **고르는**
것만 허용된다. `d`가 작으면 `w* = 0.5 + d/(2K)`가 0.5에 붙으므로 **w=0.5는 train만으로 정당화된다.**
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import shutil
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
sys.path.insert(0, str(ROOT / "sweep"))
sys.path.insert(0, str(ROOT / "submission"))

import proxy_rms as PR                                             # noqa: E402

CACHE = ROOT / "results" / "leg_matrix"
C_SCORE = 4e5                       # 1e5/(r(1−r)), r ≈ 0.494
PARTNERS = ROOT / "partners"
DIST = ROOT / "submission" / "dist"

# 규정 준수 + LB 실측 점수가 있는 레그 (점수 None = 미채점)
# 점수 출처 정본 = results/lb_history_260829.md (2026-08-29 제출 로그 전사) — 반올림 금지
AUTO_LEGS = [
    # name,        zip,                                                          LB
    ("cregime",   PARTNERS / "lg-aimers-ysy/model_artifacts/clean_regime/submission.zip",  1045.9643163789),
    ("clookup",   PARTNERS / "lg-aimers-ysy/model_artifacts/clean_lookup/submission.zip",  1049.4561885154),
    ("cmoe",      PARTNERS / "lg-aimers-ysy/model_artifacts/clean_moe/submission.zip",     1027.5280314742),
    ("calP",      DIST / "submit_ens9_calP.zip",                                 1015.7036703211),
    ("physmix",   DIST / "submit_ens9_physmix.zip",                              1009.2644920688),
    ("bis_tm",    DIST / "submit_ens7_bis_tm.zip",                               1003.5142600363),
    ("is_k100",   DIST / "submit_ens4_inseason_k100.zip",                         984.0806860421),
    ("june853",   DIST / "aimers_sub_june_verzip.zip",                            853.5697653812),
    # 구 등록값 939.0은 submit_trackman_ensemble(939.07)/residual(939.63)과 혼동된 오기였다
    ("ysy_v3",    PARTNERS / "lg-aimers-ysy/model_artifacts/submit_v3/submission.zip",      822.0412364012),
    ("ysy_cbb",   PARTNERS / "lg-aimers-ysy/model_artifacts/submit_catboost_blend/submission.zip", 937.6278793313),
    # ysy_mlp 점수는 blendE5(1039.5281171389)에서의 역산 추정(±20) — 손익분기 907.6 미달, 레그 사망
    ("ysy_mlp",   PARTNERS / "lg-aimers-ysy/model_artifacts/submit_mlp/submission.zip",     812.27),
    ("ysy_trkm",  PARTNERS / "lg-aimers-ysy/model_artifacts/submit_trackman_ensemble/submission.zip", 939.0700120243),
    ("ysy_resid", PARTNERS / "lg-aimers-ysy/model_artifacts/submit_residual_ensemble/submission.zip", 939.6308843588),
    ("ysy_stbl",  PARTNERS / "lg-aimers-ysy/model_artifacts/submit_stable_features/submission.zip",  None),
    ("ysy_rshr",  PARTNERS / "lg-aimers-ysy/model_artifacts/submit_regime_shared/submission.zip",    None),
    ("grok_v6",   PARTNERS / "LG_Aimers_JTT/GROK/submit_v6.zip",                  716.8233531487),
    ("grok_v5",   PARTNERS / "LG_Aimers_JTT/GROK/submit_v5_grok.zip",             728.408076724),
    ("jtt_tm4",   PARTNERS / "LG_Aimers_JTT/trackman_pipeline/data/v4/submit_v4.zip", None),
    ("jtt_tm3",   PARTNERS / "LG_Aimers_JTT/trackman_pipeline/data/v3/submit_v3.zip", 996.2366887286),
    ("jtt_tm2",   PARTNERS / "LG_Aimers_JTT/trackman_pipeline/data/v2/submit_v2.zip", 912.1317478883),
    # SEQ_DIST_260822(1055.18)은 scan BLOCK 10건 — base/champion에 exact1 역산 상수 전체 포함. 사용 금지.
    # tm3L = jtt_tm3의 행 독립 재빌드(제출본은 dist/submit_tm3L.zip). 여기 등록된 것은
    # 측정 전용 p23 변형 — hist가 pickle 안이라 patch_meta_deep이 못 바꾸므로, 2023년말
    # hist23.json을 동봉해 다른 레그와 같은 시점 기준으로 비교한다(jtt_tm* as-is의 RMS는
    # 당시즌 복원 블록이 죽은 채 잰 값이라 부풀려져 있다 — D-61과 같은 경로).
    # tm3L 점수는 직접 실측이 아니라 blendD4(1058.6047851923)에서의 역산 추정(±10, ρ=1.4295 가정)
    ("tm3L",      ROOT / "results/leg_matrix/measure/tm3L_p23.zip",               940.31),
    ("baseRF",    ROOT / "data" / "baseline_submit.zip",                          549.5119345223),
]

# 시즌 의존 룩업 서명 — 프록시 시점 정합이 깨질 수 있는 키
SEASON_LOOKUP_KEYS = ("inseason", "career_end", "season_end", "_base", "peb_base",
                      "prev_season", "last_season")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_season_lookup(zpath: Path):
    """시즌 의존 룩업 후보를 표시 (프록시 왜곡 위험)."""
    hits = set()
    try:
        with zipfile.ZipFile(zpath) as z:
            for info in z.infolist():
                if not info.filename.lower().endswith(".json") or info.file_size > 80_000_000:
                    continue
                try:
                    obj = json.loads(z.read(info).decode("utf-8", errors="replace"))
                except Exception:
                    continue
                stack = [obj]
                while stack:
                    cur = stack.pop()
                    if isinstance(cur, dict):
                        for k, v in cur.items():
                            kl = str(k).lower()
                            if any(s in kl for s in SEASON_LOOKUP_KEYS):
                                hits.add(f"{info.filename}:{k}")
                            if isinstance(v, (dict, list)) and len(hits) < 12:
                                stack.append(v)
                    elif isinstance(cur, list) and len(cur) < 200:
                        stack.extend(x for x in cur if isinstance(x, (dict, list)))
    except Exception:
        pass
    return sorted(hits)


# 동봉 룩업 키 → 어느 base 사전을 넣을지. 패키지마다 이름이 다르다.
PITCHER_BASE_KEYS = ("inseason_base", "pitcher_inseason_base", "peb_base",
                     "pitcher_career_end", "pitcher_season_end")
BATTER_BASE_KEYS = ("inseason_batter", "batter_inseason_base",
                    "batter_career_end", "batter_season_end")


def _entry_width(d):
    """기존 값의 리스트 길이를 그대로 보존한다 (패키지마다 [n,k] / [n,k1,k2])."""
    for v in d.values():
        return len(v) if isinstance(v, list) else 0     # 0 = 리스트가 아님 → 패치 금지
    return None


def patch_meta_deep(work: Path, pit: dict, bat: dict):
    """work/ 아래 **모든** JSON을 뒤져 시즌 말 base 룩업을 프록시 시점으로 교체.

    `proxy_rms.patch_meta`는 `model/metadata.json`·`model/regime_metadata.json` 두 경로만
    본다. 팀원 패키지는 `regime/model/metadata.json`처럼 중첩돼 있어 **조용히 패치가 빠지고**,
    그러면 당해 시즌 분해(is4) 블록이 죽은 채로 비교하게 되어 RMS가 부풀려진다
    (D-61에서 K를 2.24배 과대평가한 바로 그 경로). 그래서 경로 무관하게 키로 찾는다.
    """
    touched = []
    for jp in sorted(work.rglob("*.json")):
        try:
            if jp.stat().st_size > 400_000_000:
                continue
            obj = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        changed = False

        def walk(node, path=""):
            nonlocal changed
            if not isinstance(node, dict):
                return
            for k in list(node.keys()):
                v = node[k]
                kl = str(k).lower()
                if isinstance(v, dict) and v and kl in PITCHER_BASE_KEYS:
                    w = _entry_width(v)
                    if not w:
                        print(f"   [warn] {jp.name}:{k} 값이 리스트가 아니다 — 패치 생략")
                        continue
                    node[k] = {kk: vv[:w] for kk, vv in pit.items()}
                    touched.append(f"{jp.name}:{path}{k}")
                    changed = True
                elif isinstance(v, dict) and v and kl in BATTER_BASE_KEYS:
                    w = _entry_width(v)
                    if not w:
                        print(f"   [warn] {jp.name}:{k} 값이 리스트가 아니다 — 패치 생략")
                        continue
                    node[k] = {kk: vv[:w] for kk, vv in bat.items()}
                    touched.append(f"{jp.name}:{path}{k}")
                    changed = True
                elif isinstance(v, dict) and len(v) < 200:
                    walk(v, f"{path}{k}.")

        walk(obj)
        if changed:
            jp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return touched


def _find_root(work: Path) -> Path:
    """zip 최상위에 script.py가 없으면 한 단계 내려간다."""
    if (work / "script.py").exists():
        return work
    subs = [d for d in work.iterdir() if d.is_dir()]
    for d in subs:
        if (d / "script.py").exists():
            return d
    return work


def run_leg(name, zpath, sub, pit, bat, id_col, target, rows, force=False):
    """레그 하나를 프록시에 돌려 예측 반환 (캐시)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = f"{name}_{rows}_{sha256(zpath)[:12]}"
    npy = CACHE / f"{key}.npy"
    if npy.exists() and not force:
        print(f"[cache] {name}")
        return np.load(npy)

    tmp = Path(tempfile.mkdtemp(prefix=f"leg_{name}_"))
    try:
        work = tmp / name
        work.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(work)
        work = _find_root(work)
        touched = patch_meta_deep(work, pit, bat)
        (work / "data").mkdir(exist_ok=True)
        test = sub.drop(columns=[target])
        test.to_csv(work / "data" / "test.csv", index=False, encoding="utf-8")
        pd.DataFrame({id_col: test[id_col], target: 0.5}).to_csv(
            work / "data" / "sample_submission.csv", index=False, encoding="utf-8")
        import subprocess
        r = subprocess.run([sys.executable, "script.py"], cwd=work, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=7200)
        out = work / "output" / "submission.csv"
        if r.returncode != 0 or not out.exists():
            print(f"[FAIL ] {name} rc={r.returncode}")
            print("        " + (r.stderr or "")[-500:].replace("\n", "\n        "))
            return None
        p = pd.read_csv(out)[target].to_numpy(dtype=float)
        print(f"[run  ] {name:<10} n={len(p):,} mean={p.mean():.6f} sd={p.std():.6f}"
              f"  패치={','.join(touched) if touched else '-'}")
        np.save(npy, p)
        return p
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", action="append", default=[], metavar="NAME=PATH[:SCORE]")
    ap.add_argument("--auto", action="store_true", help="AUTO_LEGS 목록 사용")
    ap.add_argument("--anchor", default=None, help="기준 레그 이름 (기본 = 최고 점수)")
    ap.add_argument("--rows", type=int, default=30000)
    ap.add_argument("--force", action="store_true", help="캐시 무시")
    ap.add_argument("--audit", action="store_true", help="시즌 의존 룩업 표시")
    args = ap.parse_args()

    legs = []
    if args.auto:
        legs += [(n, p, s) for n, p, s in AUTO_LEGS]
    for spec in args.leg:
        name, _, rest = spec.partition("=")
        path, _, sc = rest.rpartition(":")
        if not path or not Path(path).exists():
            path, sc = rest, ""
        legs.append((name, Path(path), float(sc) if sc else None))
    legs = [(n, p, s) for n, p, s in legs if p.exists()]
    missing = [n for n, p, s in (AUTO_LEGS if args.auto else []) if not p.exists()]
    if missing:
        print("[skip] 파일 없음: " + ", ".join(missing))
    if not legs:
        print("레그가 없다"); return 1

    import real_data as rd
    print(">> train 로드")
    df = rd.load_train()
    print(f">> {PR.PROXY_SEASON - 1} 시즌 말 base 룩업")
    pit, bat = PR.build_proxy_bases(df, PR.PROXY_SEASON)
    sub = df[df[rd.SEASON] == PR.PROXY_SEASON].head(args.rows).reset_index(drop=True)
    pn = sub["asof_pitcher_n"].to_numpy(dtype=float)
    nb = np.array([pit.get(str(int(p)), [0.0])[0] for p in sub["pitcher_id"]], dtype=float)
    isn = np.maximum(pn - nb, 0.0)
    print(f">> 프록시 {len(sub):,}행 · is_n==0 {100*(isn==0).mean():.1f}% "
          f"(패치 전 100%) · 중앙값 {np.median(isn):.0f}\n")

    if args.audit:
        print("=" * 78)
        print("시즌 의존 룩업 감사 — 표시된 레그는 프록시 RMS가 왜곡될 수 있다")
        print("=" * 78)
        for n, p, _ in legs:
            h = audit_season_lookup(p)
            print(f"  {n:<10} {', '.join(h[:4]) if h else '(없음)'}")
        print()

    preds, scores = {}, {}
    for n, p, s in legs:
        arr = run_leg(n, p, sub, pit, bat, rd.ID, rd.TARGET, args.rows, args.force)
        if arr is not None and len(arr) == len(sub):
            preds[n] = arr
            scores[n] = s

    names = list(preds)
    if len(names) < 2:
        print("\n측정 가능한 레그가 2개 미만이다"); return 1

    # ---------------------------------------------------------------- RMS 행렬
    print("\n" + "=" * 78)
    print("레그 간 불일치 RMS (프록시 %s행)" % f"{len(sub):,}")
    print("=" * 78)
    print("%-10s" % "" + "".join("%9s" % n[:8] for n in names))
    for a in names:
        row = "%-10s" % a[:10]
        for b in names:
            row += "%9.5f" % float(np.sqrt(((preds[a] - preds[b]) ** 2).mean()))
        print(row)

    # ------------------------------------------------------------ 채산성 (2-way)
    anchor = args.anchor
    if anchor is None:
        scored = [(s, n) for n, s in scores.items() if s is not None]
        anchor = max(scored)[1] if scored else names[0]
    if anchor not in preds:
        print(f"\n앵커 {anchor} 예측 없음"); return 1
    S_A = scores.get(anchor)
    print("\n" + "=" * 78)
    print(f"2-way 채산성 — 앵커 {anchor}"
          + (f" (LB {S_A:.4f})" if S_A else " (LB 미채점 — d 계산 불가)"))
    print("=" * 78)
    print("%-10s %11s %9s %10s %8s %9s %9s"
          % ("레그", "LB", "RMS", "손익분기", "K", "w=0.5", "w* 이득"))
    print("-" * 78)
    for n in names:
        if n == anchor:
            continue
        rms = float(np.sqrt(((preds[n] - preds[anchor]) ** 2).mean()))
        K = C_SCORE * rms ** 2
        s = scores.get(n)
        if s is None or S_A is None:
            print("%-10s %11s %9.5f %10s %8.1f %9s %9s"
                  % (n[:10], "미채점" if s is None else f"{s:.1f}", rms, "—", K, "—", "—"))
            continue
        d = s - S_A
        need = math.sqrt(-d / C_SCORE) if d < 0 else 0.0
        g_half = 0.5 * d + K * 0.25
        w_star = min(max(0.5 + d / (2 * K), 0.0), 1.0) if K > 0 else 0.0
        g_star = w_star * d + K * w_star * (1 - w_star)
        print("%-10s %11.4f %9.5f %10.5f %8.1f %9.2f %9.2f"
              % (n[:10], s, rms, need, K, g_half, g_star))
    print("\n손익분기 = √(|d|/4e5) · K = 4e5·RMS² · 이득(w) = w·d + K·w(1−w)")
    print("⚠ w*는 참고값이다. LB 점수로 w를 역산해 배포하면 §5 위반 —"
          " 사전 등록 후보 중 고르기만 허용된다.")

    # -------------------------------------------------- 다레그 균등평균 (구조적 보장)
    print("\n" + "=" * 78)
    print("다레그 균등 평균 — 점수가 있는 레그만, 앵커 포함 상위부터 누적")
    print("=" * 78)
    ranked = sorted([n for n in names if scores.get(n) is not None],
                    key=lambda n: -scores[n])
    print("%-4s %-34s %11s %10s %10s" % ("M", "구성", "평균LB", "분산이득", "기대점수"))
    print("-" * 78)
    for m in range(2, len(ranked) + 1):
        grp = ranked[:m]
        P = np.stack([preds[n] for n in grp])
        pbar = P.mean(axis=0)
        var_term = float(((P - pbar) ** 2).mean())      # E_i[(p_i − p̄)²]
        mean_lb = float(np.mean([scores[n] for n in grp]))
        gain = C_SCORE * var_term
        print("%-4d %-34s %11.2f %10.2f %10.2f"
              % (m, "+".join(g[:6] for g in grp)[:34], mean_lb, gain, mean_lb + gain))
    print("\n항등식 Score(p̄) = 평균Score + C·E_i[(p_i−p̄)²] — 적합이 아니라 대수다.")
    print("약한 레그를 넣으면 평균LB가 선형으로 깎이고 분산이득은 2차로 는다.")

    out = CACHE / "summary.json"
    out.write_text(json.dumps({
        "rows": int(len(sub)), "anchor": anchor,
        "scores": {k: v for k, v in scores.items()},
        "rms": {f"{a}|{b}": float(np.sqrt(((preds[a] - preds[b]) ** 2).mean()))
                for a, b in itertools.combinations(names, 2)},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n>> 저장 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
