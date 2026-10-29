"""Linear models and linear dimension reduction, fit in closed form from sufficient statistics."""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg as sla
import torch

from .data import Q, load_panel, panel_features, split_window

N_CLASSES = 10
LAMBDAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
GRIDS = {
    "OLS": [{}],
    "Ridge": [{"lam": l} for l in LAMBDAS],
    "Ridge-rank": [{"lam": l} for l in LAMBDAS],
    "PCR": [{"k": k} for k in (5, 10, 20, 40, 80)],
    "PLS": [{"k": k} for k in (1, 2, 3, 5, 8, 12, 20)],
    "LDA": [{"m": m, "alpha": a} for m in (1, 3, 5, 9) for a in (0.05, 0.2, 0.5)],
    "LDA-top": [{"alpha": a} for a in (0.01, 0.05, 0.2, 0.5)],
}


def day_targets(df, winsor=(0.005, 0.995)):
    """Winsorized return, within-day rank in [-1, 1], and within-day decile class (0 = lowest)."""
    g = df.groupby("date")["target"]
    q = g.quantile(list(winsor)).unstack()
    y = df["target"].clip(df["date"].map(q[winsor[0]]), df["date"].map(q[winsor[1]]))
    pct = g.rank(method="first", pct=True)
    cls = (np.ceil(pct * N_CLASSES) - 1).clip(0, N_CLASSES - 1)
    return y.to_numpy(np.float64), (2 * pct - 1).to_numpy(np.float64), cls.to_numpy(np.int64)


def _x(X, s, e, device):
    return X[s:e].to(device, non_blocking=True).float().mul_(1.0 / Q)


@torch.no_grad()
def sufficient_stats(X, Y, cls, device, chunk=250_000):
    """n, sum x, X'X, X'Y and per-class sums, in float64 (chunks are multiplied in float32 on the device)."""
    P, T = X.shape[1], Y.shape[1]
    XtX = torch.zeros(P, P, dtype=torch.float64, device=device)
    sx = torch.zeros(P, dtype=torch.float64, device=device)
    XtY = torch.zeros(P, T, dtype=torch.float64, device=device)
    csum = torch.zeros(N_CLASSES, P, dtype=torch.float64, device=device)
    Yt, Ct = torch.from_numpy(Y), torch.from_numpy(cls)
    for s in range(0, len(X), chunk):
        x = _x(X, s, s + chunk, device)
        yv = Yt[s:s + chunk].to(device).float()
        XtX += (x.T @ x).double()
        sx += x.sum(0).double()
        XtY += (x.T @ yv).double()
        csum.index_add_(0, Ct[s:s + chunk].to(device), x.double())
    cnt = np.bincount(cls, minlength=N_CLASSES).astype(np.float64)
    return dict(n=float(len(X)), sx=sx.cpu().numpy(), XtX=XtX.cpu().numpy(), XtY=XtY.cpu().numpy(),
                sy=Y.sum(0), csum=csum.cpu().numpy(), cnt=cnt)


def _centered(st, t):
    """Covariance of x and covariance of x with target column t."""
    n, mu = st["n"], st["sx"] / st["n"]
    C = st["XtX"] / n - np.outer(mu, mu)
    c = st["XtY"][:, t] / n - mu * st["sy"][t] / n
    return C, c, mu


def _solve(A, b, eps=1e-8):
    return np.linalg.solve(A + eps * np.trace(A) / len(A) * np.eye(len(A)), b)


def _project_regress(C, c, T):
    """Regress on z = x T and map the coefficients back: beta = T gamma."""
    return T @ _solve(T.T @ C @ T, T.T @ c)


def _rescale(beta, C, c):
    """Least-squares slope of the return on the score x'beta, so the prediction is in return units."""
    return beta * (beta @ c) / (beta @ C @ beta)


def simpls(C, c, k):
    """PLS1 coefficients from the covariance matrices (de Jong, 1993)."""
    P = len(c)
    R, V, q = np.zeros((P, k)), np.zeros((P, k)), np.zeros(k)
    s = c.copy()
    for a in range(k):
        r = s / np.sqrt(s @ C @ s)
        p = C @ r
        v = p - V[:, :a] @ (V[:, :a].T @ p)
        v /= np.linalg.norm(v)
        s = s - v * (v @ s)
        R[:, a], V[:, a], q[a] = r, v, c @ r
    return R @ q


def _lda_directions(st, num, alpha):
    """Generalized eigenvectors of the between- vs (shrunk) within-class scatter of the numeric features."""
    n, cnt = st["n"], st["cnt"]
    mu = st["sx"][num] / n
    mk = st["csum"][:, num] / np.maximum(cnt, 1)[:, None]
    Sw = (st["XtX"][np.ix_(num, num)] - (cnt[:, None, None] * np.einsum("ki,kj->kij", mk, mk)).sum(0)) / n
    Sw = (1 - alpha) * Sw + alpha * np.trace(Sw) / len(num) * np.eye(len(num))
    d = mk - mu
    Sb = (cnt[:, None, None] * np.einsum("ki,kj->kij", d, d)).sum(0) / n
    vals, vecs = sla.eigh(Sb, Sw)
    return vecs[:, ::-1], Sw, mk


def fit_family(st, family, params, num, dum, Craw, craw, eig_num):
    """Coefficient vector (over all features) for one family and hyperparameter setting."""
    P = len(craw)
    avg_var = np.trace(Craw) / P
    if family == "OLS":
        return _solve(Craw, craw)
    if family == "Ridge":
        return np.linalg.solve(Craw + params["lam"] * avg_var * np.eye(P), craw)
    if family == "Ridge-rank":
        Cr, cr, _ = _centered(st, 1)
        return _rescale(np.linalg.solve(Cr + params["lam"] * avg_var * np.eye(P), cr), Craw, craw)
    if family == "PCR":
        k = min(params["k"], len(num))
        T = np.zeros((P, k + len(dum)))
        T[num, :k] = eig_num[:, :k]
        T[dum, k:] = np.eye(len(dum))
        return _project_regress(Craw, craw, T)
    if family == "PLS":
        return simpls(Craw, craw, min(params["k"], P))
    if family == "LDA":
        vecs, _, _ = _lda_directions(st, num, params["alpha"])
        m = params["m"]
        T = np.zeros((P, m + len(dum)))
        T[num, :m] = vecs[:, :m]
        T[dum, m:] = np.eye(len(dum))
        return _project_regress(Craw, craw, T)
    if family == "LDA-top":
        _, Sw, mk = _lda_directions(st, num, params["alpha"])
        cnt = st["cnt"]
        rest = (mk[:-1] * cnt[:-1, None]).sum(0) / cnt[:-1].sum()
        w = np.zeros(P)
        w[num] = np.linalg.solve(Sw, mk[-1] - rest)
        return _rescale(w, Craw, craw)
    raise ValueError(family)


@torch.no_grad()
def predict_many(X, B, b0, device, chunk=250_000):
    """Predictions of several coefficient vectors at once: X (n x P, int8) times B (P x m)."""
    Bt = torch.from_numpy(B.astype(np.float32)).to(device)
    out = [(_x(X, s, s + chunk, device) @ Bt).cpu() for s in range(0, len(X), chunk)]
    return torch.cat(out).numpy().astype(np.float64) + b0


def daily_scores(df, preds, y, rank):
    """Validation metrics for each column of preds: MSE, mean daily rank IC, mean top-decile return."""
    P = pd.DataFrame(preds, index=df.index)
    d = df["date"]
    pr = P.groupby(d).rank(pct=True)
    a = pr.sub(pr.groupby(d).transform("mean"))
    b = pd.Series(rank, index=df.index)
    b = b - b.groupby(d).transform("mean")
    ic = (a.mul(b, axis=0).groupby(d).sum() / np.sqrt((a ** 2).groupby(d).sum().mul((b ** 2).groupby(d).sum(), axis=0)))
    top = pr.gt(0.9).mul(pd.Series(y, index=df.index), axis=0).groupby(d).sum() / pr.gt(0.9).groupby(d).sum()
    return pd.DataFrame({"mse": ((P.sub(y, axis=0)) ** 2).mean().to_numpy(),
                         "ic": ic.mean().to_numpy(), "top": top.mean().to_numpy()})


def run_linear(panel_dir, windows, out_dir, families=None, criterion="ic", features=None, embargo=2,
               device=None, verbose=True):
    """Fit every family on every window, choose each family's hyperparameters on validation, save test predictions."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    features = features or panel_features(panel_dir)
    families = families or list(GRIDS)
    num = np.array([i for i, f in enumerate(features) if not f.startswith("d_")])
    dum = np.array([i for i, f in enumerate(features) if f.startswith("d_")], dtype=int)
    grid_path = out_dir / "linear_grid.csv"
    for w in windows:
        todo = [f for f in families if not (out_dir / f / f"{w['name']}.parquet").exists()]
        if not todo:
            continue
        t0 = time.time()
        df = load_panel(panel_dir, w["train_start"], w["test_end"], features)
        tr, va, te = split_window(df, w, embargo)
        to_x = lambda d: torch.from_numpy(np.ascontiguousarray(d[features].to_numpy(np.int8)))
        ytr, rtr, ctr = day_targets(tr)
        yva, rva, _ = day_targets(va)
        st = sufficient_stats(to_x(tr), np.c_[ytr, rtr], ctr, device)
        Craw, craw, mu = _centered(st, 0)
        ybar = st["sy"][0] / st["n"]
        eig_num = np.linalg.eigh(Craw[np.ix_(num, num)])[1][:, ::-1]
        t_fit = time.time()
        configs, betas = [], []
        for fam in todo:
            for params in GRIDS[fam]:
                configs.append((fam, params))
                betas.append(fit_family(st, fam, params, num, dum, Craw, craw, eig_num))
        B = np.column_stack(betas)
        b0 = ybar - mu @ B
        Xva = to_x(va)
        scores = daily_scores(va, predict_many(Xva, B, b0, device), yva, rva)
        scores.insert(0, "params", [str(p) for _, p in configs])
        scores.insert(0, "family", [f for f, _ in configs])
        scores.insert(0, "window", w["name"])
        Xte = to_x(te)
        for fam in todo:
            idx = scores.index[scores["family"] == fam]
            best = scores.loc[idx, criterion].idxmin() if criterion == "mse" else scores.loc[idx, criterion].idxmax()
            scores.loc[idx, "chosen"] = idx == best
            pred = predict_many(Xte, B[:, [best]], b0[[best]], device)[:, 0]
            (out_dir / fam).mkdir(parents=True, exist_ok=True)
            va[["date", "permno", "me", "target"]].assign(
                pred=predict_many(Xva, B[:, [best]], b0[[best]], device)[:, 0]).to_parquet(
                out_dir / fam / f"{w['name']}_val.parquet", index=False)
            te[["date", "permno", "me", "target"]].assign(pred=pred).to_parquet(out_dir / fam / f"{w['name']}.parquet", index=False)
            np.save(out_dir / fam / f"{w['name']}_beta.npy", np.r_[b0[best], B[:, best]])
        scores.to_csv(grid_path, mode="a", header=not grid_path.exists(), index=False)
        if verbose:
            ch = scores[scores["chosen"] == True]
            print(f"{w['name']}: {len(tr):,} train rows, stats {t_fit - t0:.0f}s, total {time.time() - t0:.0f}s | "
                  + ", ".join(f"{r.family} {r.params} ic={r.ic:.4f}" for r in ch.itertuples()))
    return pd.read_csv(grid_path) if grid_path.exists() else None
