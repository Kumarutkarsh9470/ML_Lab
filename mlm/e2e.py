"""End-to-end (E2E) training: the portfolio problem is a layer of the network (Sections 3.2-3.4, 4.2-4.4).

The lower-level problem of eqs. (15) and (20), with the smoothing term squared (PAPER_ISSUES.md A2, A3):

    w* = argmin_w  -w'rhat + zeta * ||w||^2 + gamma/2 * ||w - w_prev||_1
         s.t.      0 <= w <= u,  sum(w) = 1,      u = 1 / (10% of N)

is separable apart from the budget constraint, so its solution is w_i(tau) = clip(S_i(rhat_i - tau), 0, u)
for one threshold tau, where S_i is the 1-D minimizer (a soft threshold around w_prev_i when gamma > 0).
With gamma = 0 this is the Euclidean projection of rhat / (2 zeta) onto the capped simplex.

Its exact Jacobian: on the "free" stocks F (strictly inside the bounds and off the w_prev plateau),
dw/drhat = (I - 11'/|F|) / (2 zeta); elsewhere it is zero. That replaces the generic KKT differentiation of
cvxpylayers (Section 3.3) with an O(N) computation, and gives the same answer.

Objectives (E2EConfig.objective):
    "top"     long-only top decile, eq. (15); with gamma > 0 the cost-aware eq. (20)   (Sections 4.2, 4.3)
    "meansd"  mean-standard-deviation with cap 0.1 and diagonal covariance, eq. (21)   (Section 4.4)
    "ppp"     parametric portfolio policy: capped softmax of the scores, same mean-SD utility, no layer
"""
import copy
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data import Q, load_panel, panel_features, split_window
from .linear import GRIDS as LINEAR_GRIDS
from .linear import day_targets
from .meansd import capped_softmax, meansd_portfolio, solve_meansd, solve_threshold
from .models import MODELS, make_model


def _weights(v, tau, mask, u, p, gamma, zeta):
    a = v - tau
    c = gamma / 2
    s = torch.where(a > 2 * zeta * p + c, (a - c) / (2 * zeta),
                    torch.where(a < 2 * zeta * p - c, (a + c) / (2 * zeta), p))
    return torch.minimum(torch.clamp(s, min=0.0), u) * mask, a


class PortfolioLayer(torch.autograd.Function):
    """Batched over days: v (B, N) predictions, mask (B, N), u (B, 1) caps, p (B, N) previous weights."""

    ITERS = 60  # bisection steps: the threshold's bracket shrinks by 2^-60

    @staticmethod
    def forward(ctx, v, mask, u, p, gamma, zeta):
        vd = v.double()
        big = vd.abs().max().item() + 2 * zeta * 1.0 + gamma + 1.0
        lo = torch.full((v.shape[0], 1), -big, dtype=torch.float64, device=v.device)
        hi = torch.full_like(lo, big)
        md, ud, pd_ = mask.double(), u.double(), p.double()
        for _ in range(PortfolioLayer.ITERS):
            mid = (lo + hi) / 2
            w, _ = _weights(vd, mid, md, ud, pd_, gamma, zeta)
            over = w.sum(1, keepdim=True) > 1
            lo, hi = torch.where(over, mid, lo), torch.where(over, hi, mid)
        w, a = _weights(vd, (lo + hi) / 2, md, ud, pd_, gamma, zeta)
        c = gamma / 2
        if gamma > 0:
            on_plateau = (a <= 2 * zeta * pd_ + c) & (a >= 2 * zeta * pd_ - c)
        else:
            on_plateau = torch.zeros_like(md, dtype=torch.bool)
        free = (w > 1e-12) & (w < ud - 1e-12) & ~on_plateau & (md > 0)
        ctx.save_for_backward(free)
        ctx.zeta = zeta
        return w.to(v.dtype)

    @staticmethod
    def backward(ctx, g):
        (free,) = ctx.saved_tensors
        f = free.to(g.dtype)
        n = f.sum(1, keepdim=True).clamp(min=1)
        gm = (g * f).sum(1, keepdim=True) / n
        return (g - gm) * f / (2 * ctx.zeta), None, None, None, None, None


def portfolio(v, mask, u, p=None, gamma=0.0, zeta=1.0):
    p = torch.zeros_like(v) if p is None else p
    return PortfolioLayer.apply(v, mask, u, p, float(gamma), float(zeta))


def solve_day(v, u, p, gamma, zeta):
    """The same lower-level problem for one day in NumPy, solved exactly (for the sequential cost-aware case)."""
    w, free, _ = solve_threshold(v, np.full(len(v), 2 * zeta), gamma / 2, p, u)
    return w, free


class _GivenSolution(torch.autograd.Function):
    """Returns precomputed weights w and back-propagates with the free-set Jacobian of the top-decile layer."""

    @staticmethod
    def forward(ctx, v, w, free, zeta):
        ctx.save_for_backward(free)
        ctx.zeta = zeta
        return w

    @staticmethod
    def backward(ctx, g):
        (free,) = ctx.saved_tensors
        f = free.to(g.dtype)
        gm = (g * f).sum(1, keepdim=True) / f.sum(1, keepdim=True).clamp(min=1)
        return (g - gm) * f / (2 * ctx.zeta), None, None, None


@dataclass
class E2EConfig:
    objective: str = "top"       # "top" (eqs. 15/20), "meansd" (eq. 21) or "ppp"
    lr: float = 1e-3             # Adam, as in Appendix A.2.2 (the paper uses 1e-4 for eta < 1 in Section 4.4)
    days_per_step: int = 8       # gradient accumulated over 8 cross-sections (Appendix A.2.2)
    max_epochs: int = 100
    patience: int = 10
    top_frac: float = 0.1        # "top": cap u = 1 / (10% of N), the long-only top decile
    cap: float = 0.1             # "meansd"/"ppp": absolute cap w <= 0.1 (eq. 21)
    eta: float = 1.0             # "meansd"/"ppp": risk aversion
    zeta_rel: tuple = (0.2,)     # "top": smoothing relative to the spread of the warm-start predictions;
    #                              several values are compared on validation (PAPER_ISSUES.md A2)
    gamma: float = 0.0           # transaction cost rate (fraction)
    da_mu: float = 0.0           # decision-aware PCA: penalty mu * ||W - W0||^2 / ||W0||^2 (0 keeps W fixed)
    warm_start: bool = True      # False trains from a fresh initialization (PAPER_ISSUES.md D1)
    target_scale: float = 100.0  # networks work in percent units, like the MSE models
    seed: int = 0
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


class Days:
    """Rows of one split, grouped by day, with everything the portfolio layer needs on the device."""

    def __init__(self, df, features, device, permno_index, top_frac=0.1, sigma=None):
        self.top_frac = top_frac
        order = np.lexsort((df["permno"].to_numpy(), df["date"].to_numpy()))
        df = df.iloc[order].reset_index(drop=True)
        self.df = df[["date", "permno", "me", "target"]]
        self.X = torch.from_numpy(np.ascontiguousarray(df[features].to_numpy(np.int8)))
        y, _, _ = day_targets(df)
        self.y = torch.from_numpy(y.astype(np.float32))
        self.sigma = None if sigma is None else torch.from_numpy(np.asarray(sigma, np.float32)[order])
        self.gid = torch.from_numpy(df["permno"].map(permno_index).to_numpy(np.int64))
        _, self.start = np.unique(df["date"].to_numpy(), return_index=True)
        self.stop = np.append(self.start[1:], len(df))
        self.device = device

    def __len__(self):
        return len(self.start)

    def block(self, d0, d1):
        """Days d0..d1-1 as padded (B, Nmax) tensors, plus the flat rows to feed the network."""
        a, b = self.start[d0], self.stop[d1 - 1]
        X = self.X[a:b].to(self.device, non_blocking=True).float().mul_(1.0 / Q)
        sizes = self.stop[d0:d1] - self.start[d0:d1]
        B, N = len(sizes), int(sizes.max())
        row = torch.from_numpy(np.repeat(np.arange(B), sizes)).to(self.device)
        col = torch.from_numpy(np.concatenate([np.arange(s) for s in sizes])).to(self.device)
        pad = lambda flat, fill=0.0: torch.full((B, N), fill, device=self.device, dtype=flat.dtype).index_put_((row, col), flat.to(self.device))
        mask = pad(torch.ones(b - a))
        y = pad(self.y[a:b])
        gid = pad(self.gid[a:b], 0)
        sig = pad(self.sigma[a:b], 1.0) if self.sigma is not None else None
        u = torch.from_numpy(1.0 / (self.top_frac * sizes)).float().to(self.device)[:, None]
        return X, row, col, mask, y, gid, u, sig


def _pad(pred, row, col, B, N):
    out = torch.zeros(B, N, device=pred.device, dtype=pred.dtype)
    out[row, col] = pred
    return out


def _day_objective(cfg, w, y, mask, p_prev, dropped, sig):
    """Realized daily objective: return, minus costs, minus eta x ex-ante volatility for mean-SD utility."""
    obj = (w * y).sum(1)
    if cfg.gamma > 0:
        obj = obj - cfg.gamma / 2 * (((w - p_prev).abs() * mask).sum(1) + dropped)
    if cfg.objective in ("meansd", "ppp"):
        obj = obj - cfg.eta * torch.sqrt(((w * sig) ** 2).sum(1) + 1e-18)
    return obj


def epoch_pass(model, days, cfg, zeta, prev, train, opt=None, order=None, collect=False):
    """One pass over the days. Returns the mean daily objective, yesterday's weights and (collect) a daily log.

    With gamma > 0 days are processed in calendar order and each day starts from the previous day's weights
    (detached: an input of the lower-level problem); otherwise blocks of days can be shuffled.
    """
    model.train(train)
    steps = [(d, min(d + cfg.days_per_step, len(days))) for d in range(0, len(days), cfg.days_per_step)]
    if order is not None:
        steps = [steps[i] for i in order]
    total, count, log = 0.0, 0, []
    cap = None
    for d0, d1 in steps:
        X, row, col, mask, y, gid, u, sig = days.block(d0, d1)
        B, N = mask.shape
        if cfg.objective != "top":
            u = torch.full_like(u, cfg.cap)
        with torch.set_grad_enabled(train):
            v = _pad(model(X).squeeze(-1) / cfg.target_scale, row, col, B, N)
            # in order when costs link the days, and when logging (so turnover is measured against yesterday)
            sequential = cfg.gamma > 0 or collect
            if not sequential:
                if cfg.objective == "top":
                    W = portfolio(v, mask, u, zeta=zeta)
                elif cfg.objective == "meansd":
                    W = meansd_portfolio(v, sig, mask, u, cfg.eta)
                else:
                    W = capped_softmax(v * cfg.target_scale, mask, u)
                Pm = torch.zeros_like(W)
                dropped = torch.zeros(B, device=W.device)
            else:  # days depend on each other through yesterday's weights: solve them in order
                Ws, Ps, drops = [], [], []
                gn, mn = gid.cpu().numpy(), mask.cpu().numpy()
                for k in range(B):
                    n_k = int(mn[k].sum())
                    ids = gn[k, :n_k]
                    p = np.zeros(N)
                    p[:n_k] = prev[ids]
                    pt = torch.from_numpy(p).float().to(v.device)[None]
                    if cfg.objective == "top":
                        w_np, free = solve_day(v[k, :n_k].detach().double().cpu().numpy(), float(u[k, 0]), p[:n_k], cfg.gamma, zeta)
                        Wf, Ff = np.zeros(N), np.zeros(N, bool)
                        Wf[:n_k], Ff[:n_k] = w_np, free
                        wk = _GivenSolution.apply(v[k:k + 1], torch.from_numpy(Wf).float().to(v.device)[None],
                                                  torch.from_numpy(Ff).to(v.device)[None], zeta)
                    elif cfg.objective == "meansd":
                        wk = meansd_portfolio(v[k:k + 1], sig[k:k + 1], mask[k:k + 1], u[k:k + 1], cfg.eta, pt, cfg.gamma)
                    else:
                        wk = capped_softmax(v[k:k + 1] * cfg.target_scale, mask[k:k + 1], u[k:k + 1])
                    drops.append(prev.sum() - p[:n_k].sum())
                    prev = np.zeros_like(prev)
                    prev[ids] = wk[0, :n_k].detach().double().cpu().numpy()
                    Ws.append(wk)
                    Ps.append(pt)
                W, Pm = torch.cat(Ws), torch.cat(Ps)
                dropped = torch.tensor(drops, dtype=torch.float32, device=W.device)
            obj = _day_objective(cfg, W, y, mask, Pm, dropped, sig)
            if train:
                loss = -obj.mean()
                if cfg.da_mu > 0 and hasattr(model, "penalty"):
                    loss = loss + cfg.da_mu * model.penalty()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        total += float(obj.detach().sum())
        count += B
        if collect:
            gross = (W * y).sum(1).detach().cpu().numpy()
            to = 0.5 * ((((W - Pm).abs() * mask).sum(1)) + dropped).detach().cpu().numpy()
            sd = torch.sqrt(((W * (sig if sig is not None else 0)) ** 2).sum(1)).detach().cpu().numpy() if sig is not None else np.zeros(B)
            dates = days.df["date"].to_numpy()[days.start[d0:d1]]
            log += [dict(date=dt, gross=g, net=g - cfg.gamma * t, to_p=t, sd_exante=s, objective=o)
                    for dt, g, t, s, o in zip(dates, gross, to, sd, obj.detach().cpu().numpy())]
    return total / count, prev, (pd.DataFrame(log) if collect else None)


def train_e2e(model, tr, va, cfg, zeta, n_ids):
    """Train on the decision loss; early stopping on the validation objective (Appendix A.2.2)."""
    model.to(cfg.device)
    if cfg.da_mu > 0 and hasattr(model, "set_learnable"):
        model.set_learnable(True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    rng = np.random.default_rng(cfg.seed)
    zeros = lambda: np.zeros(n_ids)  # yesterday's weights by stock id (CPU, used by the sequential solves)
    with torch.no_grad():
        best, _, _ = epoch_pass(model, va, cfg, zeta, zeros(), train=False)  # the warm start itself
    start_val, best_state, best_epoch, bad = best, copy.deepcopy(model.state_dict()), 0, 0
    for epoch in range(1, cfg.max_epochs + 1):
        n_steps = -(-len(tr) // cfg.days_per_step)
        order = None if cfg.gamma > 0 else rng.permutation(n_steps)
        epoch_pass(model, tr, cfg, zeta, zeros(), train=True, opt=opt, order=order)
        with torch.no_grad():
            val, _, _ = epoch_pass(model, va, cfg, zeta, zeros(), train=False)
        if val > best:
            best, best_epoch, bad = val, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model, dict(epochs=epoch, best_epoch=best_epoch, val_obj=best, val_obj_warm=start_val)


@torch.no_grad()
def predict_rows(model, X, device, batch=200_000):
    model.eval()
    return torch.cat([model(X[s:s + batch].to(device).float().mul_(1.0 / Q)).squeeze(-1).cpu()
                      for s in range(0, len(X), batch)]).numpy()


class _Scaled(nn.Module):
    """Wraps a model that predicts in return units so that it outputs percent units, like the networks."""

    def __init__(self, inner, scale):
        super().__init__()
        self.inner, self.scale = inner, scale

    def penalty(self):
        return self.inner.penalty()

    def set_learnable(self, flag):
        self.inner.set_learnable(flag)

    def forward(self, x):
        return self.inner(x) * self.scale


def warm_model(name, warm_dir, window, n_features, cfg):
    """The saved two-stage model `name` for `window`, as a module with outputs in percent units."""
    from .mixture import GRID as MIX_GRID
    from .mixture import load_mixture
    from .reduce import load_reduced
    d = Path(warm_dir) / name
    base = name.split("-SVR")[0]
    if base in MODELS + ["RFF"]:
        m = make_model(base, n_features)
        if cfg.warm_start:
            m.load_state_dict(torch.load(d / f"{window}.pt", map_location="cpu"))
        return m
    if name in LINEAR_GRIDS:  # closed-form coefficients -> a linear layer
        beta = np.load(d / f"{window}_beta.npy")
        m = nn.Linear(n_features, 1)
        with torch.no_grad():
            m.weight.copy_(torch.as_tensor(beta[1:] * cfg.target_scale, dtype=torch.float32)[None])
            m.bias.fill_(float(beta[0] * cfg.target_scale))
        return m
    if name in MIX_GRID:
        return _Scaled(load_mixture(d / f"{window}.pt"), cfg.target_scale)
    return load_reduced(d / f"{window}.pt", name, n_features)  # "PCA20-NN3" and the like


def load_sigma(unranked_dir, frame, col="f_ivol252"):
    """Ex-ante daily volatility for the mean-SD objective: one-year residual volatility (Section 4.4 uses a
    one-year diagonal estimate); missing values get the day's median."""
    years = sorted(frame["date"].dt.year.unique())
    s = pd.concat([pd.read_parquet(Path(unranked_dir) / f"{y}.parquet", columns=["date", "permno", col]) for y in years])
    out = frame[["date", "permno"]].merge(s, on=["date", "permno"], how="left")[col]
    med = out.groupby(frame["date"].values).transform("median")
    return out.fillna(med).fillna(0.02).clip(lower=0.002).to_numpy()


def run_e2e(models, panel_dir, windows, warm_dir, out_dir, cfg, features=None, tag="E2E", verbose=True,
            unranked_dir="D:/ML_Markowitz_data/panel_cn_unranked"):
    """E2E version of each saved two-stage model, warm-started from it (Appendix A.2.2).

    models are names of saved two-stage models in warm_dir: the paper's networks, closed-form linear
    families, reduced models ("PCA20-NN3") and mixtures ("GMM-Experts"). Writes out_dir/<tag>-<model>/:
    <window>.parquet (test predictions), <window>_daily.parquet (the realized test portfolio, day by day)
    and <window>.pt. For "top", zeta = zeta_rel x (spread of the warm-start predictions) / u, and with
    several zeta_rel values the one with the best validation objective is kept.
    """
    out_dir, warm_dir = Path(out_dir), Path(warm_dir)
    features = features or panel_features(panel_dir)
    log_path = out_dir / f"{tag}_log.csv"
    for w in windows:
        todo = [m for m in models if not (out_dir / f"{tag}-{m}" / f"{w['name']}.parquet").exists()]
        if not todo:
            continue
        df = load_panel(panel_dir, w["train_start"], w["test_end"], features)
        tr_df, va_df, te_df = split_window(df, w, embargo=2)
        need_sigma = cfg.objective in ("meansd", "ppp")
        sig = {k: load_sigma(unranked_dir, f) if need_sigma else None for k, f in (("tr", tr_df), ("va", va_df), ("te", te_df))}
        ids = {p: i for i, p in enumerate(np.unique(df["permno"]))}
        tr = Days(tr_df, features, cfg.device, ids, cfg.top_frac, sig["tr"])
        va = Days(va_df, features, cfg.device, ids, cfg.top_frac, sig["va"])
        te = Days(te_df, features, cfg.device, ids, cfg.top_frac, sig["te"])
        del df, tr_df, va_df, te_df
        for m in todo:
            t0 = time.time()
            torch.manual_seed(cfg.seed)
            best = None
            for zr in (cfg.zeta_rel if cfg.objective == "top" else (None,)):
                model = warm_model(m, warm_dir, w["name"], len(features), cfg).to(cfg.device)
                zeta = 1.0
                if zr is not None:
                    rhat = predict_rows(model, tr.X[: min(len(tr.X), 500_000)], cfg.device) / cfg.target_scale
                    zeta = zr * max(float(np.std(rhat)), 1e-6) / (1.0 / (cfg.top_frac * np.median(tr.stop - tr.start)))
                model, info = train_e2e(model, tr, va, cfg, zeta, len(ids))
                if best is None or info["val_obj"] > best[1]["val_obj"]:
                    best = (model, info, zeta, zr)
            model, info, zeta, zr = best
            pred = predict_rows(model, te.X, cfg.device) / cfg.target_scale
            with torch.no_grad():
                _, _, daily = epoch_pass(model, te, cfg, zeta, np.zeros(len(ids)), train=False, collect=True)
            d = out_dir / f"{tag}-{m}"
            d.mkdir(parents=True, exist_ok=True)
            te.df.assign(pred=pred).to_parquet(d / f"{w['name']}.parquet", index=False)
            daily.to_parquet(d / f"{w['name']}_daily.parquet", index=False)
            torch.save(model.state_dict(), d / f"{w['name']}.pt")
            row = dict(window=w["name"], model=m, objective=cfg.objective, eta=cfg.eta, gamma=cfg.gamma, zeta_rel=zr,
                       zeta=zeta, da_mu=cfg.da_mu, warm_start=cfg.warm_start, **info, seconds=round(time.time() - t0, 1))
            pd.DataFrame([row]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
            if verbose:
                print(f"{w['name']} {tag}-{m}: epochs {info['epochs']} (best {info['best_epoch']}), validation objective "
                      f"{100 * info['val_obj_warm']:.3f}% -> {100 * info['val_obj']:.3f}% ({row['seconds']}s)")
    return pd.read_csv(log_path) if log_path.exists() else None


def backtest_meansd(pred, sigma, eta, gamma=0.0, cap=0.1):
    """Two-stage benchmark for Section 4.4 (MSE-Opt): solve eq. (21) each day with the given forecasts.

    pred: date, permno, pred, target; sigma aligned with pred rows. Returns the daily realized portfolio.
    """
    df = pred.assign(sigma=sigma).sort_values(["date", "permno"])
    dates, start = np.unique(df["date"].to_numpy(), return_index=True)
    stop = np.append(start[1:], len(df))
    P, R, Y, S = (df[c].to_numpy() for c in ("permno", "pred", "target", "sigma"))
    prev, rows = (P[:0], np.zeros(0)), []
    for dt, a, b in zip(dates, start, stop):
        perm = P[a:b]
        _, i, j = np.intersect1d(perm, prev[0], assume_unique=True, return_indices=True)
        p = np.zeros(b - a)
        p[i] = prev[1][j]
        dropped = prev[1].sum() - p.sum()
        w, _, s = solve_meansd(R[a:b].astype(np.float64), S[a:b].astype(np.float64), eta, cap, p, gamma)
        to = 0.5 * (np.abs(w - p).sum() + dropped)
        g = float(w @ Y[a:b])
        rows.append(dict(date=dt, gross=g, net=g - gamma * to, to_p=to, sd_exante=s))
        prev = (perm, w)
    return pd.DataFrame(rows).set_index("date")
