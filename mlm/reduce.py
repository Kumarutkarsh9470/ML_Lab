"""Dimension reduction in front of any prediction model, for two-stage (MSE) and E2E training.

A model "KIND<k>-HEAD" first maps the numeric characteristics to k components and then applies HEAD
(Linear or NN1-NN5) to [components, industry dummies]. The projection is fit on each window's training
data only:

    PCA  top-k eigenvectors of the characteristic covariance (unsupervised)
    PLS  SIMPLS weight vectors (supervised by covariance with returns)
    LDA  top-k Linear Discriminant Analysis directions separating each day's return deciles
    SFA  the k slowest directions (persistence-aware): smallest eigenvalues of C_delta v = lambda C v, where
         C_delta is the covariance of day-to-day changes. Slow components imply low signal turnover.

Each component is scaled to unit variance on the training data. With learnable=True the projection W is a
parameter, kept near its starting value by the penalty ||W - W0||^2 / ||W0||^2 (decision-aware PCA under E2E).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg as sla
import torch
import torch.nn as nn

from .data import Q, load_panel, panel_features, split_window
from .linear import _centered, _lda_directions, day_targets, simpls, sufficient_stats
from .metrics import oos_r2
from .models import make_model
from .train import TrainConfig, fit_mse, predict, to_tensors


@torch.no_grad()
def persistence_cov(df, X, num, device, chunk=250_000):
    """Covariance of day-to-day changes of the numeric features, over consecutive rows of the same stock."""
    order = np.lexsort((df["date"].to_numpy(), df["permno"].to_numpy()))
    perm = df["permno"].to_numpy()[order]
    same = torch.from_numpy(perm[1:] == perm[:-1])
    Xs = X[torch.from_numpy(order)][:, torch.from_numpy(num)]
    S = torch.zeros(len(num), len(num), dtype=torch.float64, device=device)
    n = 0
    for s in range(0, len(Xs) - 1, chunk):
        e = min(s + chunk + 1, len(Xs))
        x = Xs[s:e].to(device).float().mul_(1.0 / Q)
        d = (x[1:] - x[:-1])[same[s:e - 1].to(device)]
        S += (d.T @ d).double()
        n += len(d)
    return (S / max(n, 1)).cpu().numpy()


def directions(kind, k, st, num, C_delta=None, alpha=0.2):
    """Projection matrix (numeric features x k) for one reduction, scaled so each component has unit variance."""
    C, c, _ = _centered(st, 0)
    Cn = C[np.ix_(num, num)]
    if kind == "PCA":
        T = np.linalg.eigh(Cn)[1][:, ::-1][:, :k]
    elif kind == "PLS":
        P = len(num)
        R, s = np.zeros((P, k)), c[num].copy()
        V = np.zeros((P, k))
        for a in range(k):  # SIMPLS weight vectors on the numeric block
            r = s / np.sqrt(s @ Cn @ s)
            p = Cn @ r
            v = p - V[:, :a] @ (V[:, :a].T @ p)
            v /= np.linalg.norm(v)
            s = s - v * (v @ s)
            R[:, a], V[:, a] = r, v
        T = R
    elif kind == "LDA":
        T = _lda_directions(st, num, alpha)[0][:, :k]
    elif kind == "SFA":
        reg = 1e-6 * np.trace(Cn) / len(Cn) * np.eye(len(Cn))
        T = sla.eigh(C_delta + reg, Cn + reg)[1][:, :k]  # smallest ratio of change variance to level variance
    else:
        raise ValueError(kind)
    T = T / np.sqrt(np.einsum("ik,ij,jk->k", T, Cn, T))[None, :]
    return T


class Reduced(nn.Module):
    """x -> [(x_num - mu) W, x_dummies] -> head."""

    def __init__(self, head, n_features, num, dum, T, mu, learnable=False):
        super().__init__()
        self.register_buffer("num", torch.as_tensor(num, dtype=torch.long))
        self.register_buffer("dum", torch.as_tensor(dum, dtype=torch.long))
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer("W0", torch.as_tensor(T, dtype=torch.float32))
        self.W = nn.Parameter(self.W0.clone(), requires_grad=learnable)
        self.head = make_model(head, T.shape[1] + len(dum))

    def set_learnable(self, flag):
        self.W.requires_grad_(flag)

    def penalty(self):
        return ((self.W - self.W0) ** 2).sum() / (self.W0 ** 2).sum()

    def forward(self, x):
        z = (x[:, self.num] - self.mu) @ self.W
        return self.head(torch.cat([z, x[:, self.dum]], dim=1))


def parse(name):
    """'PCA20-NN3' -> ('PCA', 20, 'NN3')."""
    red, head = name.split("-", 1)
    kind = red.rstrip("0123456789")
    return kind, int(red[len(kind):]), head


def build(name, features, st, C_delta=None, alpha=0.2):
    kind, k, head = parse(name)
    num = np.array([i for i, f in enumerate(features) if not f.startswith("d_")])
    dum = np.array([i for i, f in enumerate(features) if f.startswith("d_")], dtype=int)
    T = directions(kind, k, st, num, C_delta, alpha)
    mu = st["sx"][num] / st["n"]
    return Reduced(head, len(features), num, dum, T, mu)


def load_reduced(path, name, n_features):
    """Rebuild a saved Reduced model from its state dict (shapes are read from the saved tensors)."""
    state = torch.load(path, map_location="cpu")
    _, _, head = parse(name)
    m = Reduced(head, n_features, state["num"].numpy(), state["dum"].numpy(), state["W0"].numpy(), state["mu"].numpy())
    m.load_state_dict(state)
    return m


def run_reduced(names, panel_dir, windows, out_dir, cfg=None, features=None, verbose=True):
    """Two-stage (MSE) training of reduced models on every window; same outputs as mlm.train.run_rolling."""
    cfg = cfg or TrainConfig()
    out_dir = Path(out_dir)
    features = features or panel_features(panel_dir)
    log_path = out_dir / "reduced_log.csv"
    for w in windows:
        todo = [m for m in names if not (out_dir / m / f"{w['name']}.parquet").exists()]
        if not todo:
            continue
        df = load_panel(panel_dir, w["train_start"], w["test_end"], features)
        tr, va, te = split_window(df, w, cfg.embargo)
        Xtr, ytr = to_tensors(tr, features, cfg)
        Xva, yva = to_tensors(va, features, cfg)
        Xte = to_tensors(te, features, cfg, labels=False)
        y, rank, cls = day_targets(tr)
        st = sufficient_stats(Xtr, np.c_[y, rank], cls, cfg.device)
        num = np.array([i for i, f in enumerate(features) if not f.startswith("d_")])
        C_delta = persistence_cov(tr, Xtr, num, cfg.device) if any(parse(m)[0] == "SFA" for m in todo) else None
        va_meta = va[["date", "permno", "me", "target"]].reset_index(drop=True)
        te_meta = te[["date", "permno", "me", "target"]].reset_index(drop=True)
        n_tr, n_va = len(tr), len(va)
        if cfg.data_on_device and cfg.device != "cpu":
            Xtr, ytr, Xva, Xte = (t.to(cfg.device) for t in (Xtr, ytr, Xva, Xte))
        del df, tr, va, te
        for m in todo:
            t0 = time.time()
            torch.manual_seed(cfg.seed)
            model, info = fit_mse(build(m, features, st, C_delta), Xtr, ytr, Xva, yva, cfg)
            d = out_dir / m
            d.mkdir(parents=True, exist_ok=True)
            res = te_meta.assign(pred=predict(model, Xte, cfg.device) / cfg.target_scale)
            va_meta.assign(pred=predict(model, Xva, cfg.device) / cfg.target_scale).to_parquet(d / f"{w['name']}_val.parquet", index=False)
            res.to_parquet(d / f"{w['name']}.parquet", index=False)
            torch.save(model.state_dict(), d / f"{w['name']}.pt")
            row = dict(window=w["name"], model=m, n_train=n_tr, n_val=n_va, **info,
                       test_r2=oos_r2(res["pred"], res["target"]), seconds=round(time.time() - t0, 1))
            pd.DataFrame([row]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
            if verbose:
                print(f"{w['name']} {m:12s} epochs={info['epochs']:3d} test_R2={row['test_r2']:+.4%} ({row['seconds']}s)")
    return pd.read_csv(log_path) if log_path.exists() else None
