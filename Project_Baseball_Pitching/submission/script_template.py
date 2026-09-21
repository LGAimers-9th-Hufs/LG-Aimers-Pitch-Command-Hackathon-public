# -*- coding: utf-8 -*-
"""평가 서버가 실행하는 추론 스크립트 v2. `build_submission.py`가 `script.py`로 동봉한다.

서버 계약: CWD = zip 루트, 입력 `./data/{test,sample_submission}.csv`, 출력 `./output/submission.csv`.

규정 준수 (data_description.md §5): 각 행은 독립적으로 예측한다. 이 파일에는 test 배치 전체를
보는 연산이 없다 — 캘리브레이션은 전부 train에서 산출해 동봉한 **상수**(calib.json)이고,
투수 룩업은 train 유래 고정 테이블을 행 단위로 조회한다(§5가 명시 허용하는 형태).

강건성: base.joblib 언피클 실패 → 순수 numpy 로지스틱 폴백. 그마저 실패 → 상수 안전망.
`script.py` 실행 오류는 제출 횟수를 차감하므로 어떤 경우에도 submission.csv를 만든다.
"""
import importlib.util
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"
DATA_DIR = "./data"
MODEL_DIR = "./model"
OUT_PATH = "./output/submission.csv"
CHUNK = 200_000


def _load_serve_features():
    """real_data.py에서 바이트 그대로 잘라온 피처 코드 — 학습·추론이 같은 함수를 쓴다."""
    path = os.path.join(MODEL_DIR, "serve_features.py")
    spec = importlib.util.spec_from_file_location("serve_features", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _predict_one(bundle, src):
    """단일 멤버 예측 — base.joblib은 순수 sklearn 객체 dict이고 kind별로 조합한다."""
    kind = bundle.get("kind", "single")
    parts = []
    for i in range(0, len(src), CHUNK):
        chunk = src.iloc[i:i + CHUNK]
        if kind == "offset":       # GLM 오프셋 부스팅: p = clip(p0 + 잔차트리)
            p0 = bundle["glm"].predict_proba(chunk[bundle["glm_cols"]])[:, 1]
            p = np.clip(p0 + bundle["gbr"].predict(chunk), 1e-3, 1 - 1e-3)
        elif kind == "avg":        # 시드 평균: 확률 평균(sum/len — 행 단위)
            ps = [m.predict_proba(chunk)[:, 1] for m in bundle["models"]]
            p = sum(ps) / len(ps)
        else:
            p = bundle["est"].predict_proba(chunk)[:, 1]
        parts.append(p)
    return np.concatenate(parts) if parts else np.zeros(0)


def _predict_bundle(bundle, X, raw, default_rep):
    """kind='blend'면 멤버별로 제 표현(feat82/raw47)과 제 β를 쓰고 **확률**을 가중평균한다.

    확률 평균이어야 Brier 볼록성 항등식이 성립한다:
        Brier(p̄) = 평균 Brier − mean((p1−p2)²)/4  → 다양성 이득이 구조적으로 보장된다.
    (로짓 평균에는 이 보장이 없다.) 전부 행 단위 연산이라 §5에 저촉되지 않는다.
    """
    if bundle.get("kind") != "blend":
        rep = bundle.get("rep", default_rep)
        return _predict_one(bundle, raw if rep == "raw47" else X)

    total = np.zeros(len(X), dtype=np.float64)
    wsum = 0.0
    for mb in bundle["members"]:
        q = _predict_one(mb, raw if mb.get("rep") == "raw47" else X)
        b = float(mb.get("beta", 0.0))
        if b:                      # 멤버별 사전등록 레벨 스칼라(제출 이력과 동일한 예측 재현)
            q = _sigmoid(_logit(q) + b)
        w = float(mb.get("weight", 1.0))
        total += w * q
        wsum += w
        print("  blend member %s rep=%s w=%.2f beta=%+.3f mean=%.4f"
              % (mb.get("name", "?"), mb.get("rep"), w, b, float(q.sum() / len(q))))
    return total / wsum


def _fallback_predict(X, calib):
    fb = calib["fallback"]
    A = X.to_numpy(dtype=np.float64, copy=True)
    med = np.asarray(fb["median"], dtype=np.float64)
    idx = np.where(np.isnan(A))
    A[idx] = np.take(med, idx[1])
    A = (A - np.asarray(fb["mean"])) / np.asarray(fb["scale"])
    return _sigmoid(A @ np.asarray(fb["coef"]) + fb["intercept"])


def main():
    with open(os.path.join(MODEL_DIR, "calib.json"), "r", encoding="utf-8") as f:
        calib = json.load(f)

    sf = _load_serve_features()
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), encoding="utf-8-sig")
    sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"), encoding="utf-8-sig")
    print("test=%d  submission=%d" % (len(test), len(sub)))

    X = sf.build_features(test).reindex(columns=calib["feature_order"])
    raw = test[sf.CAT_COLS + sf.OFFICIAL_NUM]

    # ---- 베이스 예측 (언피클 실패 시 폴백) ----
    try:
        import joblib
        bundle = joblib.load(os.path.join(MODEL_DIR, "base.joblib"))
        p = _predict_bundle(bundle, X, raw, calib.get("rep"))
        print("base model OK (kind=%s)" % bundle.get("kind"))
    except Exception:
        traceback.print_exc()
        print("!! base 모델 사용 불가 → numpy 폴백")
        p = _fallback_predict(X, calib)

    # ---- 재캘리 레이어: 전부 train 유래 상수, 행 단위 적용 ----
    beta = float(calib.get("beta", 0.0))
    recal = calib.get("recal")
    if recal:
        ztab = calib.get("z_table", {})
        zdef = float(calib.get("z_default", 0.0))
        # 투수 룩업: 행의 pitcher_id 하나로 조회 (train 유래 고정 테이블 — §5 허용)
        z = test["pitcher_id"].astype(str).map(ztab).astype(float).to_numpy()
        z = np.where(np.isnan(z), zdef, z)
        t = test[calib["thermo_col"]].to_numpy(dtype=float)
        t = np.where(np.isnan(t), recal["t_center"], t)
        p = _sigmoid(recal["a"] + recal["b"] * _logit(p)
                     + recal["c"] * (z - recal["z_center"])
                     + recal["d"] * (t - recal["t_center"]) + beta)
    elif beta:
        p = _sigmoid(_logit(p) + beta)

    lo, hi = calib.get("clip", [0.001, 0.999])
    p = np.clip(p, lo, hi)

    # ---- sample_submission의 row_id 순서로 병합 ----
    pred = dict(zip(test[ID_COL].tolist(), p.tolist()))
    out, missing = [], 0
    for rid, cur in zip(sub[ID_COL], sub[TARGET_COL]):
        v = pred.get(rid)
        if v is None:
            missing += 1
            out.append(cur)
        else:
            out.append(v)
    if missing:
        print("경고: 예측 없는 row_id %d건 — placeholder 유지" % missing)
    sub[TARGET_COL] = out

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    sub.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print("Saved %s (rows=%d)" % (OUT_PATH, len(sub)))


def safety_net():
    """main()이 어떤 이유로든 실패해도 유효한 제출 파일을 남긴다(실행 오류 = 횟수 차감)."""
    try:
        with open(os.path.join(MODEL_DIR, "calib.json"), "r", encoding="utf-8") as f:
            c = json.load(f)
        const = float(_sigmoid(np.array([c.get("z_default", 0.0)]))[0])
    except Exception:
        const = 0.5
    sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"), encoding="utf-8-sig")
    sub[TARGET_COL] = const
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    sub.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print("!! 안전망 발동: 상수 %.4f (rows=%d)" % (const, len(sub)))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        safety_net()
