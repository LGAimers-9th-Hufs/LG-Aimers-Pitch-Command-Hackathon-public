# -*- coding: utf-8 -*-
"""W2-E1 — **TM 조건부 encoder + CommandNet** (3주 프로그램 본명 트랙, Codex 최우선 권고).

    python sweep/commandnet.py --enc-eval            # 라벨-프리 검증 (인코더 자체 품질)
    python sweep/commandnet.py --gate --vals 2024    # V24 스크린 (멤버 교체 게이트)
    python sweep/commandnet.py --gate --vals 2024,2023,2022,2021 --seeds 3

## 구조

**encoder** (트랙맨 179만 행 supervised): 투수 embedding(12) + 행 맥락(카운트·이닝·아웃·월·
초말·시즌·양손 26) → MLP(64) → 구종군 CE(3) + 물리 z 회귀(7: velo/drag/relh/ext/spin/ivb/hb,
시즌 내 표준화, 결측 마스크). **폴드별 TM season<v 만 학습**(배포 동형 — 2025 서빙은 ≤2024).

**CommandNet student**: [78컬럼 표준화 피처 + encoder 출력(구종확률 3 + 물리예측 7 + emb 12 +
커버 1)] → 트렁크(64) → **OOF 스킵 로짓**(W1-B 붕괴 교훈: in-sample 스킵 + 학습 δ = 이중적합
−622 → fit 시즌 2-fold OOF로 교정) + factorized 보조 헤드(구종군·1[y=0] 실패유형).
판정 = ENS-9 nn_lin_pkg 멤버 교체 증분(w .475 동결). 위약 = 투수 embedding 경로 교환(커버 보존).

## 수렴 법칙 비해당 근거
입력 자체가 바뀐다 — 트랙맨 유래 표현(물리 예측·embedding)은 78컬럼 어디에도 없는 정보.
서빙 = {pid: emb} 룩업 + 인코더 가중치(순수 numpy forward) + 행 산술 — §5 합법(TabM 전례).
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
import trackman as TM             # noqa: E402
import nn_member as NM            # noqa: E402
from nn_carrier import BIS        # noqa: E402
from eb_carrier import (get_members9, deploy_score,   # noqa: E402
                        ENS9_W, TM_W, W_PM, eb_block)
from interaction_carrier import build_products   # noqa: E402
from auxlabel_oracle import recover_events   # noqa: E402
from tm_intent import tier1_map   # noqa: E402
from tm_condphys import MEAS7, RAW_OF, GROUPS   # noqa: E402

OUT_DIR = HERE.parent / "results" / "ens11"
EMB = 12
HID_E = 96
HID_S = 64
EPOCH_E = 20
EPOCH_S = 40
LAM_PHYS = 1.0
LAM_T = 0.3
LAM_F = 0.3
WD_S = 3e-3
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ================================================================ 맥락 인코딩 (양측 공통)
def ctx_matrix(month, dow_unused, cc, inning, outs, top, season, hh, ph):
    """26컬럼 맥락 — train 행과 tm 행이 같은 함수로 만든다(동형 보장)."""
    n = len(cc)
    C = np.zeros((n, 12), dtype=np.float32)
    C[np.arange(n), np.clip(cc, 0, 11)] = 1.0
    Mo = np.zeros((n, 8), dtype=np.float32)
    Mo[np.arange(n), np.clip(month, 3, 10) - 3] = 1.0
    return np.concatenate([
        C, Mo,
        (np.clip(inning, 1, 10) / 10.0)[:, None].astype(np.float32),
        (np.clip(outs, 0, 2) / 2.0)[:, None].astype(np.float32),
        top[:, None].astype(np.float32),
        ((season - 2019) / 5.0)[:, None].astype(np.float32),
        hh[:, None].astype(np.float32),
        ph[:, None].astype(np.float32),
    ], axis=1)


def tm_ctx_targets(tm):
    """트랙맨 행 → (ctx 26, pid_idx용 tm_id, 구종군 타깃, 물리 z 타깃 7 + 마스크,
    행별 as-of 커리어 물리평균 7 [career<row시즌 — 인코더 입력용])."""
    d = tm[tm["pitch_type_group"].isin(GROUPS)].copy()
    d["drag"] = d["rel_speed"] - d["zone_speed"]
    grp = d.groupby(rd.SEASON)
    Zt = np.empty((len(d), 7), dtype=np.float32)
    for j, m in enumerate(MEAS7):
        col = "drag" if m == "drag" else RAW_OF[m]
        mu = grp[col].transform("mean")
        sd = grp[col].transform("std")
        Zt[:, j] = ((d[col] - mu) / sd).to_numpy(dtype=np.float32)
    # as-of 커리어 물리평균 (시즌 단위 접두 — 서빙도 같은 정의의 룩업)
    d["_z_tmp"] = 0.0
    cell = pd.DataFrame({"tmid": d["pitcher_trackman_id"].to_numpy(),
                         "ssn": d[rd.SEASON].to_numpy(int)})
    for j in range(7):
        cell[f"z{j}"] = Zt[:, j]
    for j, g in enumerate(GROUPS):                     # 구종군 as-of 믹스도 접두(기준선 주입)
        cell[f"g{j}"] = (d["pitch_type_group"].to_numpy() == g).astype(float)
    agg = cell.groupby(["tmid", "ssn"]).agg(
        **{f"s{j}": (f"z{j}", "sum") for j in range(7)},
        **{f"n{j}": (f"z{j}", "count") for j in range(7)},
        **{f"gs{j}": (f"g{j}", "sum") for j in range(3)},
        gn=("g0", "count")).reset_index()
    prefix = {}
    for tmid, g in agg.groupby("tmid"):
        g = g.sort_values("ssn")
        cs = np.cumsum(g[[f"s{j}" for j in range(7)]].to_numpy(float), axis=0)
        cn = np.cumsum(g[[f"n{j}" for j in range(7)]].to_numpy(float), axis=0)
        gs = np.cumsum(g[[f"gs{j}" for j in range(3)]].to_numpy(float), axis=0)
        gn = np.cumsum(g["gn"].to_numpy(float))
        ss = g["ssn"].to_numpy(int)
        for qs in range(int(ss.min()) + 1, 2025 + 1):
            k = int(np.searchsorted(ss, qs))
            if k == 0:
                continue
            with np.errstate(invalid="ignore"):
                ph = np.where(cn[k - 1] > 0, cs[k - 1] / np.maximum(cn[k - 1], 1), 0.0)
                mx = (gs[k - 1] + 10.0 / 3) / (gn[k - 1] + 10.0)
            prefix[(tmid, qs)] = np.concatenate([ph, mx]).astype(np.float32)
    key = list(zip(cell["tmid"], cell["ssn"]))
    Pm = np.zeros((len(d), 10), dtype=np.float32)
    cache = {}
    for i, k in enumerate(key):
        v = cache.get(k)
        if v is None:
            v = prefix.get(k)
            cache[k] = v if v is not None else False
        if v is not False and v is not None:
            Pm[i] = v
    ctx = ctx_matrix(
        d["game_month"].to_numpy(int), None,
        (d["balls_before"].astype(int) * 3 + d["strikes_before"].astype(int)).to_numpy(int),
        d["inning"].to_numpy(int), d["outs_before"].to_numpy(int),
        (d["top_bottom"].astype(str).str.upper().str[0] == "T").to_numpy(float),
        d[rd.SEASON].to_numpy(int),
        (d["batter_hand"] == "Right").to_numpy(float),
        (d["pitcher_hand"] == "Right").to_numpy(float))
    gg = d["pitch_type_group"].map({g: i for i, g in enumerate(GROUPS)}).to_numpy(int)
    ctx = np.concatenate([ctx, Pm], axis=1)          # 26 + 10 (as-of 물리 7 + 믹스 3 주입)
    return d, ctx, gg, Zt, prefix


class Encoder(nn.Module):
    def __init__(self, n_ent, d_ctx):
        super().__init__()
        self.emb = nn.Embedding(n_ent + 1, EMB)          # 마지막 인덱스 = 미커버 fallback
        self.trunk = nn.Sequential(nn.Linear(EMB + d_ctx, HID_E), nn.GELU())
        self.head_t = nn.Linear(HID_E, 3)
        self.head_p = nn.Linear(HID_E, 7)

    def forward(self, eidx, ctx):
        h = self.trunk(torch.cat([self.emb(eidx), ctx], dim=1))
        return self.head_t(h), self.head_p(h)


def train_encoder(ctx, eidx, gg, Zt, n_ent, seed=0, epochs=EPOCH_E, bs=16384, lr=1.5e-3):
    torch.manual_seed(seed)
    model = Encoder(n_ent, ctx.shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    n = len(ctx)
    Ct = torch.tensor(ctx)
    Et = torch.tensor(eidx, dtype=torch.long)
    Gt = torch.tensor(gg, dtype=torch.long)
    Pt = torch.tensor(np.nan_to_num(Zt, nan=0.0))
    Mt = torch.tensor(~np.isnan(Zt))
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            ei = Et[idx].clone()
            drop = torch.rand(len(ei)) < 0.05          # fallback 임베딩도 학습(미커버 대비)
            ei[drop] = n_ent
            ht, hp = model(ei.to(DEV), Ct[idx].to(DEV))
            m = Mt[idx].to(DEV)
            mse = (((hp - Pt[idx].to(DEV)) ** 2) * m).sum() / m.sum().clamp(min=1)
            loss = ce(ht, Gt[idx].to(DEV)) + LAM_PHYS * mse
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def encoder_features(model, ent_index, df, mmap_inv, prefix, pl_map=None):
    """train/val 행 → [type_prob 3 | phys 7 | emb 12 | cover 1] (22+1컬럼). §5 행 단위."""
    pid = df["pitcher_id"].to_numpy().astype(int)
    if pl_map is not None:
        pid = np.array([pl_map.get(p, p) for p in pid])
    tmid = np.array([mmap_inv.get(p, -1) for p in pid])
    eidx = np.array([ent_index.get(t, len(ent_index)) for t in tmid])
    cover = (tmid >= 0) & (eidx < len(ent_index))
    ssn_row = df[rd.SEASON].to_numpy(int)
    Pm = np.zeros((len(df), 10), dtype=np.float32)
    cache = {}
    for i, (t, s) in enumerate(zip(tmid, ssn_row)):
        if t < 0:
            continue
        v = cache.get((t, s))
        if v is None:
            v = prefix.get((t, s))
            cache[(t, s)] = v if v is not None else False
        if v is not False and v is not None:
            Pm[i] = v
    ctx = ctx_matrix(
        df["game_month"].to_numpy(int), None,
        (df["balls_before"].astype(int) * 3 + df["strikes_before"].astype(int)).to_numpy(int),
        df["inning"].to_numpy(int), df["outs_before"].to_numpy(int),
        (df["top_bottom"].astype(str).str.upper().str[0] == "T").to_numpy(float),
        df[rd.SEASON].to_numpy(int),
        (df["batter_hand"].to_numpy(int) == 2).astype(float),
        (df["pitcher_hand"].to_numpy(int) == 2).astype(float))
    ctx = np.concatenate([ctx, Pm], axis=1)
    out = np.zeros((len(df), 23), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(df), 65536):
            sl = slice(i, min(i + 65536, len(df)))
            ht, hp = model(torch.tensor(eidx[sl], dtype=torch.long).to(DEV),
                           torch.tensor(ctx[sl]).to(DEV))
            out[sl, 0:3] = torch.softmax(ht, dim=1).cpu().numpy()
            out[sl, 3:10] = hp.cpu().numpy()
            out[sl, 10:22] = model.emb(torch.tensor(eidx[sl], dtype=torch.long).to(DEV)).cpu().numpy()
    out[:, 22] = cover.astype(np.float32)
    out[~cover, 10:22] = 0.0
    return out, cover


# ================================================================ student
class CommandNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d, HID_S), nn.GELU())
        self.delta = nn.Linear(HID_S, 1)
        self.head_t = nn.Linear(HID_S, 3)
        self.head_f = nn.Linear(HID_S, 4)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, z, logit_pkg):
        h = self.trunk(z)
        return (logit_pkg + self.delta(h).squeeze(-1), self.head_t(h), self.head_f(h))


def train_student(Z, lg, y, tt, mt, tf, mf, lam_t, lam_f, seed,
                  epochs=EPOCH_S, bs=8192, lr=1e-3):
    torch.manual_seed(seed)
    model = CommandNet(Z.shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD_S)
    ce = nn.CrossEntropyLoss()
    n = len(Z)
    Zt = torch.tensor(Z, dtype=torch.float32)
    Lt = torch.tensor(lg, dtype=torch.float32)
    Yt = torch.tensor(y, dtype=torch.float32)
    Tt = torch.tensor(tt, dtype=torch.long)
    Ft = torch.tensor(tf, dtype=torch.long)
    Mt_ = torch.tensor(mt)
    Mf_ = torch.tensor(mf)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            lgp, ht, hf = model(Zt[idx].to(DEV), Lt[idx].to(DEV))
            loss = ((torch.sigmoid(lgp) - Yt[idx].to(DEV)) ** 2).mean()
            m1 = Mt_[idx]
            if lam_t > 0 and m1.any():
                loss = loss + lam_t * ce(ht[m1.to(DEV)], Tt[idx][m1].to(DEV))
            m2 = Mf_[idx]
            if lam_f > 0 and m2.any():
                loss = loss + lam_f * ce(hf[m2.to(DEV)], Ft[idx][m2].to(DEV))
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def predict_student(model, Z, lg, bs=65536):
    out = np.empty(len(Z))
    with torch.no_grad():
        for i in range(0, len(Z), bs):
            sl = slice(i, min(i + bs, len(Z)))
            lgp, _, _ = model(torch.tensor(Z[sl], dtype=torch.float32).to(DEV),
                              torch.tensor(lg[sl], dtype=torch.float32).to(DEV))
            out[sl] = torch.sigmoid(lgp).cpu().numpy()
    return np.clip(out, 0, 1)


def oof_logits(XA, y, fit, val, seed=0):
    """fit 시즌 2-fold OOF 스킵 로짓 + val 로짓(전체 fit 학습). W1-B 붕괴 교정."""
    fidx = np.where(fit)[0]
    rng = np.random.default_rng(seed)
    half = rng.permutation(len(fidx)) % 2
    lg_fit = np.empty(len(fidx))
    for h in (0, 1):
        tr_i = fidx[half != h]
        te_i = fidx[half == h]
        st, oh = NM.prep_fit(XA.iloc[tr_i]), NM.onehot_fit(XA.iloc[tr_i])
        Ztr = np.concatenate([NM.prep_apply(XA.iloc[tr_i], st),
                              NM.onehot_apply(XA.iloc[tr_i], oh)], axis=1)
        mdl, dev0 = NM.train_mlp(Ztr, y[tr_i], hidden=(), dropout=0.0, epochs=60,
                                 lr=1e-3, wd=1e-4, seed=0)
        Zte = np.concatenate([NM.prep_apply(XA.iloc[te_i], st),
                              NM.onehot_apply(XA.iloc[te_i], oh)], axis=1)
        p = np.clip(NM.predict_mlp(mdl, dev0, Zte), 1e-6, 1 - 1e-6)
        lg_fit[half == h] = np.log(p / (1 - p))
    st, oh = NM.prep_fit(XA[fit]), NM.onehot_fit(XA[fit])
    Zf = np.concatenate([NM.prep_apply(XA[fit], st), NM.onehot_apply(XA[fit], oh)], axis=1)
    mdl, dev0 = NM.train_mlp(Zf, y[fit], hidden=(), dropout=0.0, epochs=60,
                             lr=1e-3, wd=1e-4, seed=0)
    Zv = np.concatenate([NM.prep_apply(XA[val], st), NM.onehot_apply(XA[val], oh)], axis=1)
    p_val = np.clip(NM.predict_mlp(mdl, dev0, Zv), 1e-6, 1 - 1e-6)
    return lg_fit, np.log(p_val / (1 - p_val)), p_val, (st, oh, Zf, Zv)


# ================================================================ 드라이버
def enc_eval():
    """라벨-프리 검증: TM <2023 학습 → 2023 TM 행 CE/MSE vs 고정 기준."""
    print(">> 트랙맨 로드")
    tm = TM.load_trackman()
    d, ctx, gg, Zt, _prefix = tm_ctx_targets(tm)
    ssn = d[rd.SEASON].to_numpy(int)
    tr, te = ssn < 2023, ssn == 2023
    ents = np.sort(d.loc[tr, "pitcher_trackman_id"].unique())    # 학습 엔티티만 — 신규는 fallback
    eindex = {e: i for i, e in enumerate(ents)}
    eidx = d["pitcher_trackman_id"].map(eindex).fillna(len(ents)).to_numpy(int)
    print(f"   학습 {tr.sum():,} · 검증(2023) {te.sum():,} · 엔티티 {len(ents)}"
          f" · 검증 fallback {float((eidx[te] == len(ents)).mean()):.3f}")

    t1 = time.time()
    model = train_encoder(ctx[tr], eidx[tr], gg[tr], Zt[tr], len(ents))
    print(f"   인코더 학습 [{time.time()-t1:.0f}s]")

    with torch.no_grad():
        ht, hp = [], []
        for i in range(0, int(te.sum()), 65536):
            sl = np.where(te)[0][i:i + 65536]
            a, b = model(torch.tensor(eidx[sl], dtype=torch.long).to(DEV),
                         torch.tensor(ctx[sl]).to(DEV))
            ht.append(torch.log_softmax(a, dim=1).cpu().numpy())
            hp.append(b.cpu().numpy())
    ht = np.concatenate(ht)
    hp = np.concatenate(hp)
    gg_te = gg[te]
    ce_enc = float(-ht[np.arange(len(gg_te)), gg_te].mean())

    # 기준 1: 투수별 <2023 경험 믹스 (κ=30)
    mix = pd.DataFrame({"e": eidx[tr], "g": gg[tr]}).groupby(["e", "g"]).size().unstack(fill_value=0)
    mix = mix.reindex(columns=[0, 1, 2], fill_value=0)
    glob = mix.sum(axis=0).to_numpy(float)
    glob = glob / glob.sum()
    P = (mix.to_numpy(float) + 30 * glob) / (mix.sum(axis=1).to_numpy(float)[:, None] + 30)
    row = pd.Series(np.arange(len(mix)), index=mix.index)
    ei_te = pd.Series(eidx[te]).map(row).to_numpy()
    Pte = np.where(np.isnan(ei_te)[:, None], glob[None, :], P[np.nan_to_num(ei_te, nan=0).astype(int)])
    ce_base = float(-np.log(np.clip(Pte[np.arange(len(gg_te)), gg_te], 1e-9, 1)).mean())

    m_te = ~np.isnan(Zt[te])
    mse_enc = float((((hp - np.nan_to_num(Zt[te], nan=0.0)) ** 2) * m_te).sum() / m_te.sum())
    base_p = pd.DataFrame({"e": eidx[tr]})
    zmu = pd.DataFrame(np.nan_to_num(Zt[tr], nan=np.nan)).groupby(eidx[tr]).mean()
    zmu_te = zmu.reindex(pd.Series(eidx[te])).to_numpy()
    zmu_te = np.nan_to_num(zmu_te, nan=0.0)
    mse_base = float((((zmu_te - np.nan_to_num(Zt[te], nan=0.0)) ** 2) * m_te).sum() / m_te.sum())

    print(f"\n[라벨-프리] 구종군 CE: encoder {ce_enc:.4f} vs 투수믹스 기준 {ce_base:.4f} "
          f"({(1 - ce_enc / ce_base) * 100:+.1f}%, 통과선 ≥3%)")
    print(f"[라벨-프리] 물리 MSE: encoder {mse_enc:.4f} vs 투수평균 기준 {mse_base:.4f} "
          f"({(1 - mse_enc / mse_base) * 100:+.1f}%, 통과선 ≥5%)")


def gate(vals, seeds):
    print(">> 준비 (K_IS=100) + 보조라벨 + 트랙맨")
    L.K_IS = 100.0
    df = rd.load_train()
    ev, cover_ev, mix_ok, _ = recover_events(df)
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()
    T3 = np.stack([np.nan_to_num(ev[nm], nan=0.0) for nm in ("fb", "br", "os")], axis=1)
    F4 = np.stack([np.nan_to_num(ev[nm], nan=0.0) for nm in ("rev", "mid", "ball", "strk")], axis=1)
    m_t = mix_ok & cover_ev & (T3.sum(axis=1) == 1.0)
    m_f = cover_ev & (y == 0) & (F4.sum(axis=1) == 1.0)
    tgt_t = T3.argmax(axis=1)
    tgt_f = F4.argmax(axis=1)

    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), season)
    X = L.build_features(df, base=(nb, kb)).reset_index(drop=True)
    for tag_, ent_, ncol_, rc_, K_ in IF.BLOCKS:
        if tag_ == "b":
            lb = IF.end_lookup(df, ent_, ncol_, rc_)
            B_bis = IF.build_block(df, tag_, ent_, ncol_, rc_, K_, lb)[BIS].reset_index(drop=True)
    Bps, _, _ = eb_block(df, "pitcher_id", "asof_pitcher_n",
                         [("asof_pitcher_success_rate", "succ")], 100, "ebps")
    PR = build_products(df, X, B_bis)
    CXP = [c for c in PR.columns if c.startswith("cxp_")]
    XA = pd.concat([X, B_bis, Bps, PR[CXP]], axis=1)

    mmap = tier1_map()
    mmap_inv = {int(p): int(t) for t, p in mmap.items()}
    tm = TM.load_trackman()
    d_tm, ctx_tm, gg_tm, Zt_tm, prefix_tm = tm_ctx_targets(tm)
    ssn_tm = d_tm[rd.SEASON].to_numpy(int)
    pids_cov = sorted(mmap_inv)

    res = []
    for v in vals:
        t1 = time.time()
        fit, val = (season == v - 1), (season == v)
        P, corr, corr_pm_v, yv = get_members9(v)
        others = sum(P[m] * w for m, w in ENS9_W.items() if m != "nn_lin_pkg")
        w_lin = ENS9_W["nn_lin_pkg"]
        base_rest = others + TM_W * corr + W_PM * corr_pm_v
        s_ref = deploy_score(yv, base_rest + w_lin * P["nn_lin_pkg"])

        # 폴드 인코더 (TM season < v)
        tr_tm = ssn_tm < v
        ents_f = np.sort(d_tm.loc[tr_tm, "pitcher_trackman_id"].unique())
        eindex = {e: i for i, e in enumerate(ents_f)}
        eidx_f = d_tm.loc[tr_tm, "pitcher_trackman_id"].map(eindex).to_numpy(int)
        enc = train_encoder(ctx_tm[tr_tm], eidx_f, gg_tm[tr_tm], Zt_tm[tr_tm], len(ents_f))
        E_real, covE = encoder_features(enc, eindex, df, mmap_inv, prefix_tm)
        print(f"\n[val {v}] ENS-9 동결 기준 {s_ref:.2f} · encoder 커버 {covE[val].mean():.3f}"
              f" [{time.time()-t1:.0f}s]")

        # OOF 스킵 로짓
        lg_fit, lg_val, p_val_lin, _ = oof_logits(XA, y, fit, val)
        s_par = deploy_score(yv, base_rest + w_lin * p_val_lin) - s_ref
        print(f"   OOF/전량 lin 파리티 Δ {s_par:+.2f}")

        st, oh = NM.prep_fit(XA[fit]), NM.onehot_fit(XA[fit])
        Zb_fit = np.concatenate([NM.prep_apply(XA[fit], st), NM.onehot_apply(XA[fit], oh)], axis=1)
        Zb_val = np.concatenate([NM.prep_apply(XA[val], st), NM.onehot_apply(XA[val], oh)], axis=1)

        # 위약: 커버 보존 pid 경로 교환 (embedding·물리 예측이 다른 투수 것)
        rngp = np.random.default_rng(7)
        perm = rngp.permutation(len(pids_cov))
        pl_map = {pids_cov[i]: pids_cov[perm[i]] for i in range(len(pids_cov))}
        E_pl, _ = encoder_features(enc, eindex, df, mmap_inv, prefix_tm, pl_map=pl_map)

        def zcat(Zb, E, mask):
            return np.concatenate([Zb, E[mask]], axis=1).astype(np.float32)

        arms = [("student-noenc", None),
                ("CommandNet(주셀)", E_real),
                ("CommandNet·위약", E_pl)]
        for name, E in arms:
            ds = []
            n_seed = seeds if name.startswith("CommandNet(") else 1
            for sd_ in range(n_seed):
                Zf = Zb_fit if E is None else zcat(Zb_fit, E, fit)
                Zv = Zb_val if E is None else zcat(Zb_val, E, val)
                mdl = train_student(Zf, lg_fit, y[fit], tgt_t[fit], m_t[fit],
                                    tgt_f[fit], m_f[fit], LAM_T, LAM_F, seed=21 + sd_)
                p_stu = predict_student(mdl, Zv, lg_val)
                ds.append(deploy_score(yv, base_rest + w_lin * p_stu) - s_ref)
            d_arr = np.array(ds)
            res.append(dict(val=v, arm=name, d=round(float(d_arr.mean()), 3),
                            d_sd=round(float(d_arr.std()), 3), n_seed=n_seed))
            print(f"  {name:18s}  Δ {d_arr.mean():+7.2f}"
                  + (f" ±{d_arr.std():.2f}" if n_seed > 1 else "")
                  + f"  [{time.time()-t1:.0f}s]")

    t = pd.DataFrame(res)
    (OUT_DIR / "commandnet_gate.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    piv = t.pivot_table(index="arm", columns="val", values="d")
    print("\n" + piv.to_string(float_format=lambda x: f"{x:+.2f}"))
    core = t[(t.arm == "CommandNet(주셀)") & t.val.isin([2024, 2023, 2022])]
    if len(core) >= 3:
        print(f"\n[주셀] 평균 {core.d.mean():+.2f} · SD {core.d.std():.2f} · "
              f"kill선(평균 ≥+3 & >SD) {'통과' if core.d.mean() >= 3 and core.d.mean() > core.d.std() else '미달'}")
    print(f">> 저장 {OUT_DIR/'commandnet_gate.json'}")


def carrier_gate(vals):
    """E1 대안 운반: 인코더 출력을 **검증된 캐리어**로 — ①선형 멤버 직입 ②ridge 보정기."""
    import tm_member as TMM
    from nn_carrier import train_lin
    from is_corrector import ridge_corr
    from tm_physmix import build_group_profile, row_z

    print(">> 준비 (K_IS=100) + 트랙맨 인코더")
    L.K_IS = 100.0
    df = rd.load_train()
    y = df[rd.TARGET].to_numpy().astype(float)
    season = df[rd.SEASON].to_numpy()
    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), season)
    X = L.build_features(df, base=(nb, kb)).reset_index(drop=True)
    for tag_, ent_, ncol_, rc_, K_ in IF.BLOCKS:
        if tag_ == "b":
            lb = IF.end_lookup(df, ent_, ncol_, rc_)
            B_bis = IF.build_block(df, tag_, ent_, ncol_, rc_, K_, lb)[BIS].reset_index(drop=True)
    Bps, _, _ = eb_block(df, "pitcher_id", "asof_pitcher_n",
                         [("asof_pitcher_success_rate", "succ")], 100, "ebps")
    PR = build_products(df, X, B_bis)
    CXP = [c for c in PR.columns if c.startswith("cxp_")]
    XA = pd.concat([X, B_bis, Bps, PR[CXP]], axis=1)

    mmap = tier1_map()
    mmap_inv = {int(p): int(t) for t, p in mmap.items()}
    tm = TM.load_trackman()
    d_tm, ctx_tm, gg_tm, Zt_tm, prefix_tm = tm_ctx_targets(tm)
    ssn_tm = d_tm[rd.SEASON].to_numpy(int)
    pids_cov = sorted(mmap_inv)
    prof_pm = build_group_profile()
    Z_pm, cov_pm = row_z(df, prof_pm)
    cc_ = (df["balls_before"].to_numpy() * 3 + df["strikes_before"].to_numpy()).astype(int)
    cn_ = cc_ / 11.0
    hd_ = (df["pitcher_hand"].to_numpy() == df["batter_hand"].to_numpy()).astype(float)
    pm_cols = []
    for mi in range(4):
        for ctx2 in (np.ones(len(df)), cn_, hd_):
            pm_cols.append(Z_pm[:, mi] * ctx2)
    D_pm = np.stack(pm_cols, axis=1)
    prof_tm6 = TMM.load_profile()

    res = []
    for v in vals:
        t1 = time.time()
        fit, val = (season == v - 1), (season == v)
        P, corr, corr_pm_v, yv = get_members9(v)
        others = sum(P[m] * w for m, w in ENS9_W.items() if m != "nn_lin_pkg")
        w_lin = ENS9_W["nn_lin_pkg"]
        base_rest = others + TM_W * corr + W_PM * corr_pm_v
        s_ref = deploy_score(yv, base_rest + w_lin * P["nn_lin_pkg"])
        ens9 = base_rest + w_lin * P["nn_lin_pkg"]

        tr_tm = ssn_tm < v
        ents_f = np.sort(d_tm.loc[tr_tm, "pitcher_trackman_id"].unique())
        eindex = {e: i for i, e in enumerate(ents_f)}
        eidx_f = d_tm.loc[tr_tm, "pitcher_trackman_id"].map(eindex).to_numpy(int)
        enc = train_encoder(ctx_tm[tr_tm], eidx_f, gg_tm[tr_tm], Zt_tm[tr_tm], len(ents_f))
        E_real, covE = encoder_features(enc, eindex, df, mmap_inv, prefix_tm)
        rngp = np.random.default_rng(7)
        perm = rngp.permutation(len(pids_cov))
        pl_map = {pids_cov[i]: pids_cov[perm[i]] for i in range(len(pids_cov))}
        E_pl, _ = encoder_features(enc, eindex, df, mmap_inv, prefix_tm, pl_map=pl_map)
        print(f"\n[val {v}] ENS-9 동결 기준 {s_ref:.2f} [{time.time()-t1:.0f}s]")

        # ---- ① 선형 멤버 직입 (bis 성공 경로): 파리티 / +enc10(type3+phys7) / 위약
        ENC10 = [f"enc_{i}" for i in range(10)]
        E10 = pd.DataFrame(E_real[:, 0:10], columns=ENC10)
        E10p = pd.DataFrame(E_pl[:, 0:10], columns=ENC10)
        for name, B in (("lin파리티", None), ("lin+enc10", E10), ("lin+enc10·위약", E10p)):
            XAe = XA if B is None else pd.concat([XA, B], axis=1)
            p_lin = train_lin(XAe, y, fit, val)
            d_ = deploy_score(yv, base_rest + w_lin * p_lin) - s_ref
            res.append(dict(val=v, arm=name, d=round(float(d_), 3)))
            print(f"  {name:16s}  Δ {d_:+7.2f}  [{time.time()-t1:.0f}s]")

        # ---- ② ridge 보정기 (rCpm 타깃): enc10 plain
        residA = TMM.june_fit_resid(X, y, fit)
        Df_tm, covf_tm = TMM.design(df, X, prof_tm6, fit)
        A_tm = Df_tm[fit & covf_tm]
        G = A_tm.T @ A_tm + 1000.0 * np.eye(A_tm.shape[1])
        c_tm = np.linalg.solve(G, A_tm.T @ residA[covf_tm[fit]])
        corr_tm_fit = Df_tm[fit] @ c_tm
        corr_tm_fit[~covf_tm[fit]] = 0.0
        residB = residA - TM_W * corr_tm_fit
        corr_pm_fit = ridge_corr(D_pm, fit, fit, residB, 1000.0)
        corr_pm_fit[~cov_pm[fit]] = 0.0
        rCpm = residB - W_PM * corr_pm_fit
        for name, E in (("corr+enc10", E_real), ("corr+enc10·위약", E_pl)):
            corr2 = ridge_corr(E[:, 0:10].astype(np.float64), fit, val, rCpm, 1000.0)
            corr2[~covE[val] if E is E_real else ~covE[val]] = 0.0
            for w in (0.25, 0.5):
                key = f"w{w:g}"
                dd = deploy_score(yv, ens9 + w * corr2) - s_ref
                res.append(dict(val=v, arm=f"{name}·{key}", d=round(float(dd), 3)))
            print(f"  {name:16s}  " + "  ".join(
                f"w{w:g}: {deploy_score(yv, ens9 + w * corr2) - s_ref:+7.2f}" for w in (0.25, 0.5)))

    t = pd.DataFrame(res)
    (OUT_DIR / "enc_carrier_gate.json").write_text(t.to_json(orient="records"), encoding="utf-8")
    print(f"\n>> 저장 {OUT_DIR/'enc_carrier_gate.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc-eval", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--carrier", action="store_true")
    ap.add_argument("--vals", default="2024")
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.enc_eval:
        enc_eval()
    if args.gate:
        gate([int(s) for s in args.vals.split(",")], args.seeds)
    if args.carrier:
        carrier_gate([int(s) for s in args.vals.split(",")])
    print(f"\n({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
