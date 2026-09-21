# -*- coding: utf-8 -*-
"""T1.5 — TM-표현 NN **레그** (라운드5 결정, docs/research/14 §0-b 이후 주력).

    python sweep/tm_repr_leg.py --fold 2024 --stage prep
    python sweep/tm_repr_leg.py --fold 2024 --stage arm --arm legP --seeds 2
    python sweep/tm_repr_leg.py --fold 2024 --stage arm --arm legP_pl

죽은 실험들과의 차이 (S4 감사 — docs/research/14b [S4] ②의 설계 공백 3):
  1) **레그 단위 판정**: 솔로 점수(deploy_score) + V24에선 프록시 30k행 d까지 —
     기존 NN 실험은 전부 ENS-9 멤버 교체 증분(w 동결)으로만 측정됐다.
  2) **정렬 실물리 보조손실**: pitch_align 916,049쌍의 같은-투구 실측 z(7축)를
     회귀 보조타깃으로. aux_mtl(복원 라벨만)·student(인코더 출력 피처만)와 다른 감독.
  3) 선형 백본 + zero-init 델타 트렁크: nn_lin(로컬 834) 계보를 보존한 채
     보조감독이 트렁크만 조형. 깊은 MLP 단독(750)의 함정 회피.

위약(사전 등록): 인코더 = pid 경로 교환(기존 pl_map 장치) · 정렬물리 = pid 내 행 셔플
(투수 수준 정보는 보존, 투구 수준 대응만 파괴 — 이 채널의 신규분이 정확히 그것이다).
킬: 3폴드 솔로 <860 또는 위약 미분리.

§5: 서빙 = {pid:emb} 룩업 + 인코더/레그 가중치 forward + 행 산술 (TabM·cmoe 전례 합법).
정렬쌍·TM은 학습 전용 — 서빙 입력에 2025 TM 없음.
"""
from __future__ import annotations

import argparse
import json
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
from eb_carrier import eb_block                        # noqa: E402
from interaction_carrier import build_products         # noqa: E402
from auxlabel_oracle import recover_events             # noqa: E402
from tm_intent import tier1_map                        # noqa: E402
import commandnet as CN           # noqa: E402

OUT = HERE.parent / "results" / "tm_repr"
ALIGN_NPZ = HERE.parent / "results" / "ens11" / "pitch_align_idx.npz"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def leg_score(y, p):
    """레그 게이트 점수 = 합법 레거시 캘리(1.04, -0.01) 적용 후 score.

    eb_carrier.deploy_score는 금지된 LB 역산 상수(slope 1.14202)를 적용하므로
    새 레그 게이트에 쓰지 않는다 — 역사적 로컬 라인(805/834/877.8)과도 비교 불가.
    """
    q = np.clip(0.5 + 1.04 * (np.asarray(p, dtype=float) - 0.5) - 0.01, 1e-6, 1 - 1e-6)
    return L.score(y, q)


# ================================================================ 데이터 조립 (prep)
def build_xa(df, season):
    """commandnet.gate()의 XA 조립을 그대로 — is4 base는 (pid, 시즌) 커리어말이라 폴드 안전."""
    L.K_IS = 100.0
    lut = IS.career_end_lookup(df)
    nb, kb = IS.base_for(lut, df["pitcher_id"].to_numpy(), season)
    X = L.build_features(df, base=(nb, kb)).reset_index(drop=True)
    B_bis = None
    for tag_, ent_, ncol_, rc_, K_ in IF.BLOCKS:
        if tag_ == "b":
            lb = IF.end_lookup(df, ent_, ncol_, rc_)
            B_bis = IF.build_block(df, tag_, ent_, ncol_, rc_, K_, lb)[BIS].reset_index(drop=True)
    Bps, _, _ = eb_block(df, "pitcher_id", "asof_pitcher_n",
                         [("asof_pitcher_success_rate", "succ")], 100, "ebps")
    PR = build_products(df, X, B_bis)
    CXP = [c for c in PR.columns if c.startswith("cxp_")]
    return pd.concat([X, B_bis, Bps, PR[CXP]], axis=1)


def aligned_phys_rows(df, d, Zt):
    """pitch_align 쌍 → df 행 위치별 실측 물리 z(7) (미정렬 NaN).

    pitch_align.py:52-56과 동일한 순서 계약: train_idx는 df.sort_values(row_id) 순의 위치.
    tm_idx는 TM.load_trackman() 프레임의 원 인덱스 라벨. (d, Zt) = CN.tm_ctx_targets 산출 재사용.
    """
    z = np.load(ALIGN_NPZ)
    train_idx, tm_idx = z["train_idx"], z["tm_idx"]
    Zdf = pd.DataFrame(Zt, index=d.index)
    Za = Zdf.reindex(tm_idx).to_numpy(dtype=np.float32)      # 필터된 구종군 밖 = NaN
    order = df.sort_values(rd.ID).index.to_numpy()
    pos = order[train_idx]                                    # df 원 위치
    out = np.full((len(df), 7), np.nan, dtype=np.float32)
    out[pos] = Za
    return out


def prep(v):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f">> prep fold v={v}")
    df = rd.load_train()
    y = df[rd.TARGET].to_numpy().astype(np.float32)
    season = df[rd.SEASON].to_numpy()
    fit = season == v - 1
    val = season == v

    print(">> XA (X+is4 | bis | ebps | cxp)")
    XA = build_xa(df, season)
    print(f"   XA {XA.shape}  [{time.time()-t0:.0f}s]")

    print(">> 보조라벨 복원")
    ev, cover_ev, mix_ok, _ = recover_events(df)
    T3 = np.stack([np.nan_to_num(ev[nm], nan=0.0) for nm in ("fb", "br", "os")], axis=1)
    F4 = np.stack([np.nan_to_num(ev[nm], nan=0.0) for nm in ("rev", "mid", "ball", "strk")], axis=1)
    m_t = mix_ok & cover_ev & (T3.sum(axis=1) == 1.0)
    m_f = cover_ev & (y == 0) & (F4.sum(axis=1) == 1.0)
    tgt_t = T3.argmax(axis=1)
    tgt_f = F4.argmax(axis=1)

    print(">> 트랙맨 + 폴드 인코더 (TM season < v)")
    mmap = tier1_map()
    mmap_inv = {int(p): int(t) for t, p in mmap.items()}
    tm = TM.load_trackman()
    d_tm, ctx_tm, gg_tm, Zt_tm, prefix_tm = CN.tm_ctx_targets(tm)
    ssn_tm = d_tm[rd.SEASON].to_numpy(int)
    tr_tm = ssn_tm < v
    ents = np.sort(d_tm.loc[tr_tm, "pitcher_trackman_id"].unique())
    eindex = {e: i for i, e in enumerate(ents)}
    eidx = d_tm.loc[tr_tm, "pitcher_trackman_id"].map(eindex).to_numpy(int)
    enc = CN.train_encoder(ctx_tm[tr_tm], eidx, gg_tm[tr_tm], Zt_tm[tr_tm], len(ents))
    E_real, cov = CN.encoder_features(enc, eindex, df, mmap_inv, prefix_tm)
    pids_cov = sorted(mmap_inv)
    rngp = np.random.default_rng(7)
    perm = rngp.permutation(len(pids_cov))
    pl_map = {pids_cov[i]: pids_cov[perm[i]] for i in range(len(pids_cov))}
    E_pl, _ = CN.encoder_features(enc, eindex, df, mmap_inv, prefix_tm, pl_map=pl_map)
    print(f"   enc 커버 fit {cov[fit].mean():.3f} / val {cov[val].mean():.3f}  [{time.time()-t0:.0f}s]")

    print(">> 정렬 실물리 (fit 행만 사용)")
    phys = aligned_phys_rows(df, d_tm, Zt_tm)
    pm = ~np.isnan(phys[:, 0])
    print(f"   정렬 커버: 전체 {pm.mean():.3f} · fit 행 {pm[fit].mean():.3f}")
    # 위약: fit 내 같은 pid 안에서 정렬물리 행 셔플 (투구 수준 대응만 파괴)
    phys_pl = phys.copy()
    pid = df["pitcher_id"].to_numpy()
    rng2 = np.random.default_rng(11)
    fi = np.where(fit & pm)[0]
    dfp = pd.Series(fi).groupby(pid[fi])
    for _, g in dfp:
        gi = g.to_numpy()
        phys_pl[gi] = phys[rng2.permutation(gi)]

    print(">> 표준화 (fit 기준)")
    st, oh = NM.prep_fit(XA[fit]), NM.onehot_fit(XA[fit])
    Zb_fit = np.concatenate([NM.prep_apply(XA[fit], st), NM.onehot_apply(XA[fit], oh)], axis=1)
    Zb_val = np.concatenate([NM.prep_apply(XA[val], st), NM.onehot_apply(XA[val], oh)], axis=1)

    # d-변형 입력: 챔피언 basin 중첩 블록(bis·ebps·cxp) 제거판 — 교환비(점수↓ vs d↑) 실측용
    drop = set(BIS) | {c for c in XA.columns if c.startswith(("ebps", "cxp_"))}
    XA2 = XA[[c for c in XA.columns if c not in drop]]
    st2, oh2 = NM.prep_fit(XA2[fit]), NM.onehot_fit(XA2[fit])
    Zc_fit = np.concatenate([NM.prep_apply(XA2[fit], st2), NM.onehot_apply(XA2[fit], oh2)], axis=1)
    Zc_val = np.concatenate([NM.prep_apply(XA2[val], st2), NM.onehot_apply(XA2[val], oh2)], axis=1)
    print(f"   Zb {Zb_fit.shape[1]}열 / Zc(basin 제거 {len(drop)}컬럼) {Zc_fit.shape[1]}열")

    print(">> KD teacher (june풍 LGBM, fit 학습)")
    import lightgbm as lgb
    mt_ = lgb.train(dict(objective="binary", metric="binary_logloss", learning_rate=0.05,
                         num_leaves=15, min_data_in_leaf=1000, feature_fraction=0.85,
                         bagging_fraction=0.8, bagging_freq=1, verbosity=-1, seed=0),
                    lgb.Dataset(XA[fit], label=y[fit]), num_boost_round=600)
    p_tea_fit = mt_.predict(XA[fit]).astype(np.float32)
    p_tea_val = mt_.predict(XA[val]).astype(np.float32)
    s_tea = leg_score(y[val], p_tea_val)
    print(f"   teacher 솔로 {s_tea:.2f}")

    np.savez_compressed(
        OUT / f"w{v}.npz",
        Zb_fit=Zb_fit.astype(np.float32), Zb_val=Zb_val.astype(np.float32),
        Zc_fit=Zc_fit.astype(np.float32), Zc_val=Zc_val.astype(np.float32),
        E_fit=E_real[fit], E_val=E_real[val], Epl_fit=E_pl[fit], Epl_val=E_pl[val],
        y_fit=y[fit], y_val=y[val],
        phys_fit=phys[fit], physpl_fit=phys_pl[fit],
        tgt_t_fit=tgt_t[fit], m_t_fit=m_t[fit], tgt_f_fit=tgt_f[fit], m_f_fit=m_f[fit],
        p_tea_fit=p_tea_fit, s_tea=np.array([s_tea]))
    print(f">> saved {OUT}/w{v}.npz  [{time.time()-t0:.0f}s]")


# ================================================================ 레그 모델 (arm)
def self_lin_params(model):
    return list(model.lin.parameters())


def self_trunk_params(model):
    ps = []
    for m in (model.trunk, model.delta, model.head_p, model.head_t, model.head_f):
        ps += list(m.parameters())
    return ps


class LegNet(nn.Module):
    def __init__(self, dz, hid=64):
        super().__init__()
        self.lin = nn.Linear(dz, 1)
        self.trunk = nn.Sequential(nn.Linear(dz, hid), nn.GELU())
        self.delta = nn.Linear(hid, 1)
        self.head_p = nn.Linear(hid, 7)
        self.head_t = nn.Linear(hid, 3)
        self.head_f = nn.Linear(hid, 4)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, z):
        h = self.trunk(z)
        logit = self.lin(z).squeeze(-1) + self.delta(h).squeeze(-1)
        return logit, self.head_p(h), self.head_t(h), self.head_f(h)


def train_leg(Z, y, phys, tt, mt, tf, mf, p_tea, *, lam_p, lam_t, lam_f, w_kd,
              seed, epochs=60, bs=4096, lr=1e-3, hid=64):
    torch.manual_seed(seed)
    model = LegNet(Z.shape[1], hid).to(DEV)
    # nn_lin(805~834) 검증 레시피 이식: 출력 바이어스 = 기저율 로짓, 가중치 0.1배,
    # OneCycleLR. 이 두 가지가 없으면 선형 백본이 수렴 못 한다(-634/-1972 실측).
    p0 = float(np.mean(y))
    with torch.no_grad():
        model.lin.bias.fill_(float(np.log(p0 / (1 - p0))))
        model.lin.weight.mul_(0.1)
    # wd 분리: 선형 백본 1e-4(nn_lin 계보) / 트렁크·헤드 3e-3(CommandNet 수준).
    opt = torch.optim.AdamW([
        {"params": self_lin_params(model), "weight_decay": 1e-4},
        {"params": self_trunk_params(model), "weight_decay": 3e-3},
    ], lr=lr)
    steps = int(np.ceil(len(Z) / bs)) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    ce = nn.CrossEntropyLoss()
    n = len(Z)
    Zt = torch.tensor(Z)
    Yt = torch.tensor(y)
    Pt = torch.tensor(np.nan_to_num(phys, nan=0.0))
    Pm = torch.tensor(~np.isnan(phys[:, 0]))
    Tt = torch.tensor(tt, dtype=torch.long)
    Ft = torch.tensor(tf, dtype=torch.long)
    Mt_ = torch.tensor(mt)
    Mf_ = torch.tensor(mf)
    Kt = torch.tensor(p_tea) if p_tea is not None else None
    for _ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            logit, hp, ht, hf = model(Zt[idx].to(DEV))
            p = torch.sigmoid(logit)
            loss = ((p - Yt[idx].to(DEV)) ** 2).mean()
            if w_kd > 0 and Kt is not None:
                loss = loss + w_kd * ((p - Kt[idx].to(DEV)) ** 2).mean()
            if lam_p > 0:
                m = Pm[idx].to(DEV)
                if m.any():
                    mse = (((hp - Pt[idx].to(DEV)) ** 2).mean(dim=1) * m).sum() / m.sum()
                    loss = loss + lam_p * mse
            if lam_t > 0:
                m1 = Mt_[idx]
                if m1.any():
                    loss = loss + lam_t * ce(ht[m1.to(DEV)], Tt[idx][m1].to(DEV))
            if lam_f > 0:
                m2 = Mf_[idx]
                if m2.any():
                    loss = loss + lam_f * ce(hf[m2.to(DEV)], Ft[idx][m2].to(DEV))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    return model


def predict_leg(model, Z, bs=65536):
    out = np.empty(len(Z), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(Z), bs):
            sl = slice(i, min(i + bs, len(Z)))
            logit, _, _, _ = model(torch.tensor(Z[sl]).to(DEV))
            out[sl] = torch.sigmoid(logit).cpu().numpy()
    return np.clip(out, 1e-6, 1 - 1e-6)


ARMS = {
    #            enc      phys     lam_p lam_t lam_f w_kd   base
    "lin":      ("none",  None,    0.0,  0.0,  0.0,  0.0),
    "leg0":     ("real",  None,    0.0,  0.3,  0.3,  0.0),
    "legP":     ("real",  "real",  1.0,  0.3,  0.3,  0.0),
    "legP_pl":  ("pl",    "pl",    1.0,  0.3,  0.3,  0.0),
    "legPD":    ("real",  "real",  1.0,  0.3,  0.3,  0.5),
    "legPnb":   ("real",  "real",  1.0,  0.3,  0.3,  0.0),   # basin 제거 입력(Zc)
}


def arm(v, name, seeds):
    t0 = time.time()
    w = np.load(OUT / f"w{v}.npz")
    enc_mode, phys_mode, lam_p, lam_t, lam_f, w_kd = ARMS[name]
    if name == "legPnb":
        Zb_fit, Zb_val = w["Zc_fit"], w["Zc_val"]
    else:
        Zb_fit, Zb_val = w["Zb_fit"], w["Zb_val"]
    if enc_mode == "none":
        Zf, Zv = Zb_fit, Zb_val
    elif enc_mode == "real":
        Zf = np.concatenate([Zb_fit, w["E_fit"]], axis=1)
        Zv = np.concatenate([Zb_val, w["E_val"]], axis=1)
    else:
        Zf = np.concatenate([Zb_fit, w["Epl_fit"]], axis=1)
        Zv = np.concatenate([Zb_val, w["Epl_val"]], axis=1)
    phys = (np.full((len(Zf), 7), np.nan, dtype=np.float32) if phys_mode is None
            else w["phys_fit"] if phys_mode == "real" else w["physpl_fit"])
    p_tea = w["p_tea_fit"] if w_kd > 0 else None
    y_fit, y_val = w["y_fit"], w["y_val"]
    print(f">> arm {name} v={v}  Z {Zf.shape}  teacher_solo {float(w['s_tea'][0]):.2f}")

    scores, preds = [], []
    for sd in range(seeds):
        mdl = train_leg(Zf, y_fit, phys, w["tgt_t_fit"], w["m_t_fit"],
                        w["tgt_f_fit"], w["m_f_fit"], p_tea,
                        lam_p=lam_p, lam_t=lam_t, lam_f=lam_f, w_kd=w_kd, seed=21 + sd)
        p = predict_leg(mdl, Zv)
        s = leg_score(y_val, p)
        scores.append(s)
        preds.append(p)
        print(f"   seed {sd}: solo {s:.2f}  [{time.time()-t0:.0f}s]")
    p_mean = np.mean(preds, axis=0)
    s_bag = leg_score(y_val, p_mean)
    rep = {"fold": v, "arm": name, "seeds": seeds,
           "solo_mean": round(float(np.mean(scores)), 2),
           "solo_sd": round(float(np.std(scores)), 2),
           "solo_bag": round(s_bag, 2),
           "teacher": round(float(w["s_tea"][0]), 2)}

    # V24: 프록시 30k행 d (leg_matrix 캐시와 같은 행 = season 2024 head 30000)
    if v == 2024:
        import glob
        dvs = {}
        p30 = p_mean[:30000]
        legs = {}
        for nm in ("clookup", "cmoe", "physmix", "tm3L", "cregime", "ysy_mlp"):
            fs = sorted(glob.glob(str(HERE.parent / "results/leg_matrix" / f"{nm}_30000_*.npy")))
            if fs:
                legs[nm] = np.load(fs[-1])
                dvs[nm] = round(float(np.sqrt(np.mean((p30 - legs[nm]) ** 2))), 5)
        if all(k in legs for k in ("clookup", "cmoe", "physmix", "tm3L")):
            D4 = np.mean([legs[k] for k in ("clookup", "cmoe", "physmix", "tm3L")], axis=0)
            dvs["D4_mean"] = round(float(np.sqrt(np.mean((p30 - D4) ** 2))), 5)
        rep["rms_vs_legs"] = dvs
        print("   d: " + "  ".join(f"{k}={x}" for k, x in dvs.items()))

    outp = OUT / f"gate_{v}_{name}.json"
    outp.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    np.save(OUT / f"pred_{v}_{name}.npy", p_mean)
    print(f">> {rep}")
    print(f">> saved {outp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--stage", choices=["prep", "arm"], required=True)
    ap.add_argument("--arm", default="legP", choices=list(ARMS))
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()
    if args.stage == "prep":
        prep(args.fold)
    else:
        arm(args.fold, args.arm, args.seeds)


if __name__ == "__main__":
    main()
