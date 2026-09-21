"""
후보 레지스트리. 각 후보 = (name, tier, kind, build()).
build()는 fit(X,y)/predict_proba(X)를 갖는 모델을 반환.

카탈로그(06/08) ID와 매핑:
  M0  로지스틱 baseline
  M1  GBDT 앵커 (sklearn HistGradientBoosting; lightgbm/xgboost 있으면 자동 추가)
  S2  신경망 tabular student (torch MLP, hard label)
  D2  증류 student (torch MLP + teacher soft-label KD)  ← teacher=GBDT, OOF 소프트라벨
optional: lightgbm(M1b)/xgboost(M1c)는 설치돼 있으면 자동 등록.

새 러너(TabM, 시퀀스 student S3, self-distillation M24 등)는 여기에 build()만 추가하면
동일한 시간분할 CV 게이트로 자동 편입됨.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict


@dataclass
class Candidate:
    name: str
    tier: str
    kind: str
    build: Callable
    needs: str = "flat"   # 입력 표현: "flat"(as-of 피처) | "flat_f13"(+LLM prior) | "seq"(투구열 텐서)


# ---------- baselines (항상 사용 가능) ----------
def _logistic():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def _gbdt_anchor():
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        random_state=0)


# ---------- torch student (S2) + 증류 student (D2) ----------
class _TorchMLP:
    """작은 MLP. soft_from(teacher)로 KD 소프트라벨 학습 가능."""
    def __init__(self, hidden=64, epochs=40, lr=1e-3, alpha=0.5, kd_teacher=None, seed=0):
        self.hidden, self.epochs, self.lr = hidden, epochs, lr
        self.alpha, self.kd_teacher, self.seed = alpha, kd_teacher, seed

    def fit(self, X, y):
        import torch
        torch.manual_seed(self.seed)
        self.scaler = StandardScaler().fit(X)
        Xt = torch.tensor(self.scaler.transform(X), dtype=torch.float32)
        yt = torch.tensor(np.asarray(y), dtype=torch.float32).view(-1, 1)

        # KD: teacher OOF 소프트라벨(누수 방지: cross_val_predict)
        soft = None
        if self.kd_teacher is not None:
            oof = cross_val_predict(self.kd_teacher, X, y, cv=3, method="predict_proba")[:, 1]
            soft = torch.tensor(np.clip(oof, 1e-6, 1 - 1e-6), dtype=torch.float32).view(-1, 1)

        d = Xt.shape[1]
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, self.hidden), torch.nn.ReLU(),
            torch.nn.Linear(self.hidden, self.hidden), torch.nn.ReLU(),
            torch.nn.Linear(self.hidden, 1))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        bce = torch.nn.BCEWithLogitsLoss()
        n = Xt.shape[0]
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, 4096):
                idx = perm[i:i + 4096]
                opt.zero_grad()
                logit = self.net(Xt[idx])
                loss = bce(logit, yt[idx])
                if soft is not None:  # KD: hard + soft 혼합
                    loss = self.alpha * loss + (1 - self.alpha) * bce(logit, soft[idx])
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, X):
        import torch
        Xt = torch.tensor(self.scaler.transform(X), dtype=torch.float32)
        with torch.no_grad():
            p = torch.sigmoid(self.net(Xt)).numpy().ravel()
        return np.column_stack([1 - p, p])


# ---------- M15: 계층 베이지안 log5 (매치업 구조적 사전확률) ----------
class _BayesLog5:
    """log5 매치업 결합: as-of 투수 제구율(p)·타자 유발율(b)·리그평균(g)으로
        P = (p·b/g) / (p·b/g + (1-p)(1-b)/(1-g))
    의 사후확률 로짓을 만들고, 로지스틱으로 슬로프/절편만 학습(구조적 캘리).
    as-of 율은 이미 베이즈 수축(부분 풀링)된 값(features._asof_rate) → 계층 사전확률 역할.
    입력: flat 피처(p_asof_rate, b_asof_rate 컬럼 사용)."""
    def __init__(self):
        self.g = 0.5

    def _log5_logit(self, X):
        eps = 1e-4
        p = np.clip(X["p_asof_rate"].to_numpy(), eps, 1 - eps)
        b = np.clip(X["b_asof_rate"].to_numpy(), eps, 1 - eps)
        g = min(max(self.g, eps), 1 - eps)
        num = p * b / g
        den = num + (1 - p) * (1 - b) / (1 - g)
        w = np.clip(num / den, eps, 1 - eps)
        return np.log(w / (1 - w)).reshape(-1, 1)

    def fit(self, X, y):
        y = np.asarray(y)
        self.g = float(y.mean())
        self.lr = LogisticRegression(max_iter=1000).fit(self._log5_logit(X), y)
        return self

    def predict_proba(self, X):
        return self.lr.predict_proba(self._log5_logit(X))


# ---------- S3/D3: 시퀀스 student (경량 GRU over 투구열) + 교사 soft-label KD ----------
class _SeqGRU:
    """투수별 as-of 투구열(build_sequences) → 소형 GRU → 마지막 은닉 → 로짓.
    경량 버전(CPU 스모크용). A100에선 hidden/epochs↑·Transformer로 확장.

    D3(soft-label KD): kd=True면 입력이 (S, Xflat) 튜플(needs=seq_kd). 교사(flat GBDT)의
    OOF 소프트라벨(cross_val_predict, 누수 방지)을 hard label과 혼합 학습. 08 증류계획 D3 트랙.
    """
    def __init__(self, hidden=32, epochs=15, lr=1e-3, seed=0, kd=False, alpha=0.5, kd_teacher_fn=None):
        self.hidden, self.epochs, self.lr, self.seed = hidden, epochs, lr, seed
        self.kd, self.alpha, self.kd_teacher_fn = kd, alpha, kd_teacher_fn

    def fit(self, S, y):
        import torch
        torch.manual_seed(self.seed)
        # KD: 입력이 (S, Xflat) 튜플 → 교사 OOF 소프트라벨 생성
        soft = None
        if isinstance(S, tuple):
            S, Xflat = S
            if self.kd:
                teacher = (self.kd_teacher_fn or _gbdt_anchor)()
                oof = cross_val_predict(teacher, Xflat, y, cv=3, method="predict_proba")[:, 1]
                soft = np.clip(oof, 1e-6, 1 - 1e-6)
        S = np.asarray(S, dtype=np.float32)
        flat = S.reshape(-1, S.shape[-1])
        self.mean = flat.mean(0)
        self.std = flat.std(0) + 1e-6
        St = torch.tensor((S - self.mean) / self.std, dtype=torch.float32)
        yt = torch.tensor(np.asarray(y), dtype=torch.float32).view(-1, 1)
        softt = torch.tensor(soft, dtype=torch.float32).view(-1, 1) if soft is not None else None

        F = S.shape[-1]
        self.gru = torch.nn.GRU(F, self.hidden, batch_first=True)
        self.head = torch.nn.Linear(self.hidden, 1)
        params = list(self.gru.parameters()) + list(self.head.parameters())
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=1e-5)
        bce = torch.nn.BCEWithLogitsLoss()
        n = St.shape[0]
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, 2048):
                idx = perm[i:i + 2048]
                opt.zero_grad()
                _, h = self.gru(St[idx])
                logit = self.head(h[-1])
                loss = bce(logit, yt[idx])
                if softt is not None:  # KD: hard + soft 혼합
                    loss = self.alpha * loss + (1 - self.alpha) * bce(logit, softt[idx])
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, S):
        import torch
        if isinstance(S, tuple):
            S = S[0]
        S = np.asarray(S, dtype=np.float32)
        St = torch.tensor((S - self.mean) / self.std, dtype=torch.float32)
        with torch.no_grad():
            _, h = self.gru(St)
            p = torch.sigmoid(self.head(h[-1])).numpy().ravel()
        return np.column_stack([1 - p, p])


# ---------- M24: GLMMNet/LMMNN (엔티티 임베딩 랜덤효과 NN) ----------
class _GLMMNet:
    """투수/타자 **엔티티 임베딩 + 분산 정규화(랜덤효과 shrinkage prior)** + flat 피처 MLP → 로짓.

    M25(GBT판 랜덤효과)의 NN 짝. 미등장 엔티티(콜드스타트)는 예약된 unknown 인덱스(≈0 임베딩)로
    fallback. needs=flat_id(원본 _pid/_bid 컬럼 필요; 트리 피처가 아니라 임베딩 조회에만 사용).
    """
    def __init__(self, emb_dim=8, hidden=64, epochs=40, lr=1e-3, re_l2=1e-3, seed=0):
        self.emb_dim, self.hidden, self.epochs = emb_dim, hidden, epochs
        self.lr, self.re_l2, self.seed = lr, re_l2, seed

    @staticmethod
    def _idx(ids, mapping):
        return np.array([mapping.get(int(v), 0) for v in ids], dtype=np.int64)  # 미등장→0

    def fit(self, X, y):
        import torch
        torch.manual_seed(self.seed)
        pid, bid = X["_pid"].to_numpy(), X["_bid"].to_numpy()
        Xf = X.drop(columns=[c for c in ("_pid", "_bid", "_season") if c in X.columns])
        self.scaler = StandardScaler().fit(Xf)
        # id→index 맵(train). 인덱스 0 = unknown 예약.
        self.pmap = {int(v): i + 1 for i, v in enumerate(np.unique(pid))}
        self.bmap = {int(v): i + 1 for i, v in enumerate(np.unique(bid))}
        Xt = torch.tensor(self.scaler.transform(Xf), dtype=torch.float32)
        pit = torch.tensor(self._idx(pid, self.pmap))
        bit = torch.tensor(self._idx(bid, self.bmap))
        yt = torch.tensor(np.asarray(y), dtype=torch.float32).view(-1, 1)

        d = Xt.shape[1]
        self.emb_p = torch.nn.Embedding(len(self.pmap) + 1, self.emb_dim)
        self.emb_b = torch.nn.Embedding(len(self.bmap) + 1, self.emb_dim)
        torch.nn.init.normal_(self.emb_p.weight, std=0.1)
        torch.nn.init.normal_(self.emb_b.weight, std=0.1)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d + 2 * self.emb_dim, self.hidden), torch.nn.ReLU(),
            torch.nn.Linear(self.hidden, 1))
        params = (list(self.net.parameters()) + list(self.emb_p.parameters())
                  + list(self.emb_b.parameters()))
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=1e-5)
        bce = torch.nn.BCEWithLogitsLoss()
        n = Xt.shape[0]
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, 4096):
                idx = perm[i:i + 4096]
                opt.zero_grad()
                ep, eb = self.emb_p(pit[idx]), self.emb_b(bit[idx])
                logit = self.net(torch.cat([Xt[idx], ep, eb], dim=1))
                # 랜덤효과 분산 정규화(임베딩 L2 = shrinkage prior)
                re_pen = self.re_l2 * (ep.pow(2).mean() + eb.pow(2).mean())
                loss = bce(logit, yt[idx]) + re_pen
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, X):
        import torch
        pid, bid = X["_pid"].to_numpy(), X["_bid"].to_numpy()
        Xf = X.drop(columns=[c for c in ("_pid", "_bid", "_season") if c in X.columns])
        Xt = torch.tensor(self.scaler.transform(Xf), dtype=torch.float32)
        pit = torch.tensor(self._idx(pid, self.pmap))
        bit = torch.tensor(self._idx(bid, self.bmap))
        with torch.no_grad():
            ep, eb = self.emb_p(pit), self.emb_b(bit)
            p = torch.sigmoid(self.net(torch.cat([Xt, ep, eb], dim=1))).numpy().ravel()
        return np.column_stack([1 - p, p])


# ---------- M25: GBT + 가우시안 랜덤효과 하이브리드 (GPBoost/MERF-lite) ----------
class _MERFLite:
    """GBDT(고정효과: 상황·as-of 피처) + 투수별 랜덤 절편(로짓 스케일, 베이즈 수축).

    faithful-light MERF: (1) ID 제외 피처로 GBDT 적합, (2) OOF 예측의 로짓 잔차를
    투수별 평균(수축 k)으로 집계 = 랜덤 절편 BLUP 근사, (3) 예측 = GBDT 로짓 + 랜덤절편.
    **미등장 투수(콜드스타트)는 절편 0**(=prior 평균으로 자동 fallback) — random-effect의 핵심 이점.
    입력: flat_id 표현(원본 _pid/_bid 컬럼 포함). _pid/_bid는 트리 입력에서 제외.
    """
    def __init__(self, shrink_k=30.0, seed=0):
        self.shrink_k, self.seed = shrink_k, seed

    @staticmethod
    def _drop_ids(X):
        return X.drop(columns=[c for c in ("_pid", "_bid", "_season") if c in X.columns])

    def fit(self, X, y):
        y = np.asarray(y)
        self.gm = float(y.mean())
        Xf = self._drop_ids(X)
        self.gbdt = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
            random_state=self.seed).fit(Xf, y)
        # OOF 로짓 잔차 → 투수별 랜덤 절편(수축). 누수 방지: cross_val_predict.
        oof = np.clip(cross_val_predict(
            HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
                l2_regularization=1.0, random_state=self.seed),
            Xf, y, cv=3, method="predict_proba")[:, 1], 1e-4, 1 - 1e-4)
        resid = _logit(y * 0.998 + 0.001) - _logit(oof)   # 관측(평활)−예측, 로짓 스케일
        pid = X["_pid"].to_numpy()
        self.intercept = _shrunk_group_mean(pid, resid, self.shrink_k)
        return self

    def predict_proba(self, X):
        Xf = self._drop_ids(X)
        base = np.clip(self.gbdt.predict_proba(Xf)[:, 1], 1e-6, 1 - 1e-6)
        pid = X["_pid"].to_numpy()
        b = np.array([self.intercept.get(int(k), 0.0) for k in pid])  # 미등장→0
        p = 1.0 / (1.0 + np.exp(-(_logit(base) + b)))
        return np.column_stack([1 - p, p])


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _shrunk_group_mean(keys, vals, k):
    """그룹별 수축 평균: n_g/(n_g+k) * mean_g (전역평균 0 기준). 반환 dict{key:intercept}."""
    import pandas as pd
    s = pd.Series(vals).groupby(keys)
    out = {}
    for key, grp in s:
        n = len(grp)
        out[int(key)] = (n / (n + k)) * float(grp.mean())
    return out


# ---------- F31: 시즌 recency-decay 가중 GBDT ----------
class _EraWeighted:
    """GBDT + sample_weight = decay^(fit 최신시즌 − season). F31(ABS 체제 표본 상향)의 일반화.
    하드 threshold(season>=2024)는 캘리 프로토콜이 train 최신 시즌을 fit에서 홀드아웃해
    전 폴드에서 공허(가중 전부 1.0 → 앵커류와 비트동일 = 죽은 arm)였음. decay 가중은
    fit에 들어오는 모든 시즌에 차등 적용되어 게이트에서 측정 가능. 하드 2024+ 가중의
    원형은 제출 직전 P3(100% refit) A/B로 평가. needs=flat_id(_season은 트리 입력 제외)."""
    def __init__(self, decay=0.8, seed=0):
        self.decay, self.seed = decay, seed

    def fit(self, X, y):
        season = X["_season"].to_numpy()
        sw = np.power(float(self.decay), (season.max() - season).astype(float))
        Xf = X.drop(columns=[c for c in ("_pid", "_bid", "_season") if c in X.columns])
        self.gbdt = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
            random_state=self.seed).fit(Xf, y, sample_weight=sw)
        return self

    def predict_proba(self, X):
        Xf = X.drop(columns=[c for c in ("_pid", "_bid", "_season") if c in X.columns])
        return self.gbdt.predict_proba(Xf)


def get_candidates():
    cands = [
        Candidate("M0_logistic", "T1", "linear", _logistic),
        Candidate("M1_gbdt_anchor", "T1", "gbdt", _gbdt_anchor),
        Candidate("S2_mlp_student", "T3", "nn", lambda: _TorchMLP(epochs=40)),
        Candidate("D2_kd_student", "T3", "distill",
                  lambda: _TorchMLP(epochs=40, alpha=0.5, kd_teacher=_gbdt_anchor())),
        # M15: 계층 베이지안 log5 매치업 사전확률(구조적 캘리). flat 피처 사용.
        Candidate("M15_bayes_log5", "T3", "bayes", _BayesLog5),
        # F13: LLM 콜드스타트 사전확률·임베딩을 얹은 GBDT. needs=flat_f13.
        Candidate("M1F13_gbdt_llm", "T4", "gbdt", _gbdt_anchor, needs="flat_f13"),
        # S3: 시퀀스 student(경량 GRU). needs=seq. A100서 확장(D3).
        Candidate("S3_seq_student", "T3", "seq", lambda: _SeqGRU(epochs=15), needs="seq"),
        # M1+rich: GBDT 앵커 + 09 신규 피처(F15/F17/F18/F19). needs=flat_plus.
        Candidate("M1rich_gbdt_f15_19", "T2", "gbdt", _gbdt_anchor, needs="flat_plus"),
        # M25: GBT + 가우시안 랜덤효과 하이브리드(GPBoost/MERF-lite). needs=flat_id.
        Candidate("M25_merf", "T3", "hybrid", lambda: _MERFLite(), needs="flat_id"),
        # M24: 엔티티 임베딩 랜덤효과 NN(GLMMNet/LMMNN). M25의 NN 짝. needs=flat_id.
        Candidate("M24_glmmnet", "T3", "nn", lambda: _GLMMNet(epochs=40), needs="flat_id"),
        # D3: 시퀀스 student + 교사(GBDT) soft-label KD. needs=seq_kd((Seq, Xflat)).
        Candidate("D3_seq_kd", "T3", "distill",
                  lambda: _SeqGRU(epochs=15, kd=True, alpha=0.5, kd_teacher_fn=_gbdt_anchor),
                  needs="seq_kd"),
        # M1+ctx: GBDT + 10 조사 맥락 피처(F24~F33: 경험곡선·복귀·드리프트·압박·구장·era). needs=flat_ctx.
        Candidate("M1ctx_gbdt_f24_33", "T2", "gbdt", _gbdt_anchor, needs="flat_ctx"),
        # F31: 시즌 recency-decay 가중 GBDT — test(2025)에 가까운 체제 표본 상향. needs=flat_id.
        Candidate("M1w_f31_recency", "T2", "gbdt", lambda: _EraWeighted(decay=0.8), needs="flat_id"),
    ]
    # optional GBDT (설치돼 있으면 자동 등록)
    try:
        import lightgbm as lgb  # noqa
        cands.append(Candidate("M1b_lightgbm", "T1", "gbdt",
            lambda: __import__("lightgbm").LGBMClassifier(
                n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8,
                colsample_bytree=0.8, reg_lambda=1.0, verbosity=-1)))
    except Exception:
        pass
    try:
        import xgboost as xgb  # noqa
        cands.append(Candidate("M1c_xgboost", "T1", "gbdt",
            lambda: __import__("xgboost").XGBClassifier(
                n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
                colsample_bytree=0.8, reg_lambda=1.0, eval_metric="logloss")))
    except Exception:
        pass
    return cands


ANCHOR = "M1_gbdt_anchor"  # 게이트 기준(모든 후보는 이걸 이겨야 채택)
