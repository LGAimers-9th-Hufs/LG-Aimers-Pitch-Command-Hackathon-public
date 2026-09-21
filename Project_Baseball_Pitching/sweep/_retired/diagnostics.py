"""
사전 진단 (S0) — 실데이터 sweep 전에 돌리는 시프트/CV 진단.

P1 Adversarial Validation: train(과거 시즌) vs val(최신 시즌) 행을 피처로 구분하는 분류기의
  AUC로 **분포 시프트**를 측정. AUC≈0.5면 시프트 없음, 높으면 상위 기여 피처가 시프트원.
  대응(스윕 arm): 상위 피처 drop / 상대화 / AV 확률을 recency sample_weight로.
  (근거: AMEX 등 시간분할 대회 우승팀 공통 기법. 09 §3.2)
P5 CV 프로토콜 비교(경량 스캐폴드): 시즌 GroupKFold vs expanding window 폴드 구조를 요약.
  실데이터에서 "어떤 CV가 최종 시즌을 가장 잘 예측하는가"를 돌릴 엔트리.
"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from features import build_features
from validation import make_time_folds


def adversarial_validation(df, spec, feature_set="full", top_k=6, seed=0):
    """train(<val 최신시즌) vs val(최신시즌) 판별 AUC + 상위 기여 피처.
    feature_set: "full"(as-of 포함) | "context"(맥락만, sanity≈0.5)."""
    X, _, meta = build_features(df, spec, rich=True)
    seasons = sorted(meta[spec.season].unique())
    val_season = seasons[-1]
    z = (meta[spec.season].to_numpy() == val_season).astype(int)  # 1=val, 0=train
    drop = [c for c in ("_pid", "_bid") if c in X.columns]
    Xf = X.drop(columns=drop)
    if feature_set == "context":
        cols = [c for c in spec.context if c in Xf.columns]
        Xf = Xf[cols]

    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                         max_leaf_nodes=31, random_state=seed)
    oof = cross_val_predict(clf, Xf, z, cv=3, method="predict_proba")[:, 1]
    auc = roc_auc_score(z, oof)

    top = []
    if feature_set == "full" and auc > 0.55:  # 시프트 있을 때만 기여 피처 계산
        clf.fit(Xf, z)
        imp = permutation_importance(clf, Xf, z, n_repeats=3, random_state=seed,
                                     scoring="roc_auc", max_samples=min(4000, len(Xf)))
        order = np.argsort(imp.importances_mean)[::-1][:top_k]
        top = [(Xf.columns[i], float(imp.importances_mean[i])) for i in order]
    return {"val_season": val_season, "auc": auc, "top_features": top, "feature_set": feature_set}


def run_adversarial(df, spec):
    print(">> [P1] Adversarial Validation (train vs 최신 시즌)")
    ctx = adversarial_validation(df, spec, feature_set="context")
    print(f"   context-only AUC={ctx['auc']:.3f}  (sanity: 인위적 시프트 없으면 ≈0.5)")
    full = adversarial_validation(df, spec, feature_set="full")
    print(f"   full(as-of 포함) AUC={full['auc']:.3f}  (val={full['val_season']})")
    if full["top_features"]:
        print("   시프트 상위 피처(drop/상대화 후보):")
        for name, v in full["top_features"]:
            print(f"     - {name:<18} imp={v:.4f}")
    else:
        print("   → 유의한 시프트 없음(합성은 정상). 실데이터에선 구속·구종 트렌드 피처가 걸릴 수 있음.")
    full["context_auc"] = ctx["auc"]  # 기록용 sanity 지표 병기
    return full


def cv_protocol_summary(df, spec):
    """P5 스캐폴드: expanding window 폴드 구조 요약(실데이터에서 GroupKFold와 A/B할 엔트리)."""
    _, _, meta = build_features(df, spec)
    folds = make_time_folds(meta, spec, min_train_seasons=3)
    print(">> [P5] CV 프로토콜(expanding window) 폴드 구조")
    summary = []
    for tr_idx, va_idx, tr_seasons, va_season in folds:
        print(f"   train={list(tr_seasons)} → val={va_season}  "
              f"(train {len(tr_idx)} / val {len(va_idx)} 행)")
        summary.append({"train_seasons": [int(s) for s in tr_seasons],
                        "val_season": int(va_season),
                        "n_train": len(tr_idx), "n_val": len(va_idx)})
    print("   실데이터: 시즌 GroupKFold(스크리닝) vs 위 expanding(최종 게이트) A/B 예정.")
    return summary
