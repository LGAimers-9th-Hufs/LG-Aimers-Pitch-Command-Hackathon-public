# -*- coding: utf-8 -*-
"""W1-B — 보조라벨 factorized 멀티태스크 logit-skip student (3주 프로그램 Track B).

    python sweep/aux_mtl.py --vals 2024                  # V24 스크린
    python sweep/aux_mtl.py --vals 2024,2023,2022 --seeds 3

## 합법성 (탐사 확정)
§6은 현재 투구 정보를 **"입력으로"** 금지 — 학습 타깃(보조 손실) 사용은 금지 문구 없음.
서빙 zip에는 라벨·next-row 룩업 미동봉(추론 입력 = 기존 78컬럼 그대로) → §5 행 단위 유지.
D-02의 T-e 기각 전제("보조 라벨 부재")는 D-44 복원 실증으로 사실 오류 판명.

## 구조 (Codex 자문 반영)
- 타깃: `recover_events` 차분 복원 — 구종군 3(mix 연속행) · 실패유형 4(**y=0 행 한정** —
  1[y=0] 마스크가 1−y 재학습을 차단하는 factorized 설계의 핵심).
- carrier = **logit-skip**: `logit(p) = logit(p_pkg) + δθ(Z)` — 검증된 선형 멤버(nn_lin_pkg)를
  고정 스킵으로 보존, δ·트렁크에 강한 L2. 실패해도 멤버 파괴 없음.
- 판정 = ENS-9 멤버 교체 증분(w .475 동결). **kill: MTL − main-only < +2 → 공식-only 종료**
  (같은 입력의 MTL은 수렴 법칙 — 본셀은 W2의 TM 표현 결합형 CommandNet).
- 위약 = 보조라벨 시즌 내 행 셔플(마스크 보존) — 이득이 라벨 정보인지 정규화 잡음인지 분리.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import real_data as rd            # noqa: E402
import lgbm_family as L           # noqa: E402
import inseason as IS             # noqa: E402
import inseason_full as IF        # noqa: E402
import nn_member as NM            # noqa: E402
from nn_carrier import BIS        # noqa: E402
from eb_carrier import (get_members9, deploy_score,   # noqa: E402
                        ENS9_W, TM_W, W_PM, eb_block)
from interaction_carrier import build_products   # noqa: E402
from auxlabel_oracle import recover_events   # noqa: E402

OUT_DIR = HERE.parent / "results" / "ens11"
P_SUCC = [("asof_pitcher_success_rate", "succ")]
HID = 64
EPOCHS = 40
LAM_T = 0.3           # 구종군 CE 가중
LAM_F = 0.3           # 실패유형 CE 가중
WD_TRUNK = 3e-3       # δ 경로 강 L2 (Codex)


class MTLStudent(nn.Module):
    def __init__(self, d, hid=HID):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d, hid), nn.GELU())
        self.delta = nn.Linear(hid, 1)
        self.head_t = nn.Linear(hid, 3)
        self.head_f = nn.Linear(hid, 4)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, z, logit_pkg):
        h = self.trunk(z)
        return (logit_pkg + self.delta(h).squeeze(-1),
                self.head_t(h), self.head_f(h))


def train_student(Z, logit_pkg, y, tgt_t, m_t, tgt_f, m_f, lam_t, lam_f, seed,
                  epochs=EPOCHS, bs=8192, lr=1e-3):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MTLStudent(Z.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD_TRUNK)
    n = len(Z)
    Zt = torch.tensor(Z, dtype=torch.float32)
    Lp = torch.tensor(logit_pkg, dtype=torch.float32)
    Yt = torch.tensor(y, dtype=torch.float32)
    Tt = torch.tensor(tgt_t, dtype=torch.long)
    Ft = torch.tensor(tgt_f, dtype=torch.long)
    Mt = torch.tensor(m_t)
    Mf = torch.tensor(m_f)
    ce = nn.CrossEntropyLoss(reduction="mean")
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            z = Zt[idx].to(dev)
            lg, ht, hf = model(z, Lp[idx].to(dev))
            p = torch.sigmoid(lg)
            loss = ((p - Yt[idx].to(dev)) ** 2).mean()
            mt = Mt[idx]
            if lam_t > 0 and mt.any():
                loss = loss + lam_t * ce(ht[mt.to(dev)], Tt[idx][mt].to(dev))
            mf = Mf[idx]
            if lam_f > 0 and mf.any():
                loss = loss + lam_f * ce(hf[mf.to(dev)], Ft[idx][mf].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model, dev


def predict_student(model, dev, Z, logit_pkg, bs=65536):
    out = np.empty(len(Z))
    with torch.no_grad():
        for i in range(0, len(Z), bs):
            z = torch.tensor(Z[i:i + bs], dtype=torch.float32).to(dev)
            lp = torch.tensor(logit_pkg[i:i + bs], dtype=torch.float32).to(dev)
            lg, _, _ = model(z, lp)
            out[i:i + bs] = torch.sigmoid(lg).cpu().numpy()
    return np.clip(out, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vals", default="2024")
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(">> 준비 (K_IS=100) + 보조라벨 복원")
    L.K_IS = 100.0
    df = rd.load_train()
    ev, cover, mix_ok, _y = recover_events(df)
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()

    T3 = np.stack([np.nan_to_num(ev[nm], nan=0.0) for nm in ("fb", "br", "os")], axis=1)
    F4 = np.stack([np.nan_to_num(ev[nm], nan=0.0) for nm in ("rev", "mid", "ball", "strk")], axis=1)
    m_t = mix_ok & cover & (T3.sum(axis=1) == 1.0)
    m_f = cover & (y == 0) & (F4.sum(axis=1) == 1.0)
    tgt_t = T3.argmax(axis=1)
    tgt_f = F4.argmax(axis=1)
    print(f"   구종군 타깃 {m_t.mean():.3f} 커버 · 실패유형 타깃 {m_f.mean():.3f} 커버")

    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), season)
    X = L.build_features(df, base=(nb, kb)).reset_index(drop=True)
    for tag_, ent_, ncol_, rc_, K_ in IF.BLOCKS:
        if tag_ == "b":
            lb = IF.end_lookup(df, ent_, ncol_, rc_)
            B_bis = IF.build_block(df, tag_, ent_, ncol_, rc_, K_, lb)[BIS].reset_index(drop=True)
    Bps, _, _ = eb_block(df, "pitcher_id", "asof_pitcher_n", P_SUCC, 100, "ebps")
    PR = build_products(df, X, B_bis)
    CXP = [c for c in PR.columns if c.startswith("cxp_")]
    XA = pd.concat([X, B_bis, Bps, PR[CXP]], axis=1)

    res = []
    for v in [int(s) for s in args.vals.split(",")]:
        t1 = time.time()
        fit, val = (season == v - 1), (season == v)
        P, corr, corr_pm_v, yv = get_members9(v)
        others = sum(P[m] * w for m, w in ENS9_W.items() if m != "nn_lin_pkg")
        w_lin = ENS9_W["nn_lin_pkg"]
        base_rest = others + TM_W * corr + W_PM * corr_pm_v
        s_ref = deploy_score(yv, base_rest + w_lin * P["nn_lin_pkg"])

        # 선형 멤버 재학습(스킵 로짓용 fit/val 예측 동시 확보)
        st, oh = NM.prep_fit(XA[fit]), NM.onehot_fit(XA[fit])
        Z_fit = np.concatenate([NM.prep_apply(XA[fit], st), NM.onehot_apply(XA[fit], oh)], axis=1)
        Z_val = np.concatenate([NM.prep_apply(XA[val], st), NM.onehot_apply(XA[val], oh)], axis=1)
        mdl_lin, dev0 = NM.train_mlp(Z_fit, y[fit], hidden=(), dropout=0.0, epochs=60,
                                     lr=1e-3, wd=1e-4, seed=0)
        p_pkg_fit = np.clip(NM.predict_mlp(mdl_lin, dev0, Z_fit), 1e-6, 1 - 1e-6)
        p_pkg_val = np.clip(NM.predict_mlp(mdl_lin, dev0, Z_val), 1e-6, 1 - 1e-6)
        lg_fit = np.log(p_pkg_fit / (1 - p_pkg_fit))
        lg_val = np.log(p_pkg_val / (1 - p_pkg_val))
        s_par = deploy_score(yv, base_rest + w_lin * p_pkg_val)
        print(f"\n[val {v}] ENS-9 동결 기준 {s_ref:.2f} · 재학습 파리티 {s_par - s_ref:+.2f}"
              f" [{time.time()-t1:.0f}s]")

        # 위약: 보조라벨 fit-시즌 내 셔플 (마스크 위치 보존, 값만 재배열)
        rngp = np.random.default_rng(5)
        fidx = np.where(fit)[0]
        tgt_t_pl, tgt_f_pl = tgt_t.copy(), tgt_f.copy()
        it_ = fidx[m_t[fidx]]
        tgt_t_pl[it_] = tgt_t[rngp.permutation(it_)]
        if_ = fidx[m_f[fidx]]
        tgt_f_pl[if_] = tgt_f[rngp.permutation(if_)]

        arms = [("main-only", 0.0, 0.0, tgt_t, tgt_f),
                ("MTL", LAM_T, LAM_F, tgt_t, tgt_f),
                ("MTL·위약", LAM_T, LAM_F, tgt_t_pl, tgt_f_pl)]
        for name, lt, lf, tt_, tf_ in arms:
            ds = []
            for sd_ in range(args.seeds):
                mdl, dev = train_student(
                    Z_fit, lg_fit, y[fit], tt_[fit], m_t[fit], tf_[fit], m_f[fit],
                    lt, lf, seed=11 + sd_)
                p_stu = predict_student(mdl, dev, Z_val, lg_val)
                ds.append(deploy_score(yv, base_rest + w_lin * p_stu) - s_ref)
            d_arr = np.array(ds)
            res.append(dict(val=v, arm=name, d=round(float(d_arr.mean()), 3),
                            d_sd=round(float(d_arr.std()), 3), n_seed=args.seeds))
            print(f"  {name:12s}  Δ {d_arr.mean():+7.2f}"
                  + (f" ±{d_arr.std():.2f}" if args.seeds > 1 else "")
                  + f"  [{time.time()-t1:.0f}s]")

    t = pd.DataFrame(res)
    (OUT_DIR / "aux_mtl_gate.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    piv = t.pivot_table(index="arm", columns="val", values="d")
    print("\n" + piv.to_string(float_format=lambda x: f"{x:+.2f}"))
    print("\nkill 판정: MTL − main-only < +2 → 공식-only 종료 (본셀 = W2 CommandNet)")
    print(f">> 저장 {OUT_DIR/'aux_mtl_gate.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
