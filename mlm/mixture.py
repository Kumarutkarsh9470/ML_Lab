"""Mixture of linear experts with a GMM gate (two-stage closed form, and a module reusable by E2E)."""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture

from .data import Q, load_panel, panel_features, split_window
from .linear import _centered, daily_scores, day_targets, sufficient_stats

GRID = {"GMM-Experts": [{"k": k, "lam": l} for k in (2, 3, 4) for l in (0.1, 1.0)],
        "Size-Experts": [{"k": 3, "lam": l} for l in (0.1, 1.0)]}


class MixtureExperts(nn.Module):
    """Gate from a GMM on PCA scores (fixed buffers) times K linear experts (parameters)."""

    def __init__(self, num, mu, V, gmm, betas, b0, size_col=None, size_edges=None):
        super().__init__()
        self.register_buffer("num", torch.as_tensor(num, dtype=torch.long))
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer("V", torch.as_tensor(V, dtype=torch.float32))
        if gmm is not None:
            self.register_buffer("means", torch.as_tensor(gmm.means_, dtype=torch.float32))
            self.register_buffer("prec_chol", torch.as_tensor(gmm.precisions_cholesky_, dtype=torch.float32))
            self.register_buffer("log_w", torch.as_tensor(np.log(gmm.weights_), dtype=torch.float32))
        self.size_col = size_col
        if size_edges is not None:
            self.register_buffer("size_edges", torch.as_tensor(size_edges, dtype=torch.float32))
        self.beta = nn.Parameter(torch.as_tensor(betas, dtype=torch.float32))
        self.b0 = nn.Parameter(torch.as_tensor(b0, dtype=torch.float32))

    def gate(self, x):
        if self.size_col is not None:
            s = x[:, self.size_col].contiguous()
            k = torch.bucketize(s, self.size_edges)
            return torch.nn.functional.one_hot(k, self.beta.shape[0]).float()
        z = (x[:, self.num] - self.mu) @ self.V
        y = torch.einsum("nd,kde->nke", z, self.prec_chol) - torch.einsum("kd,kde->ke", self.means, self.prec_chol)[None]
        logdet = torch.log(torch.diagonal(self.prec_chol, dim1=1, dim2=2)).sum(1)
        logp = self.log_w + logdet - 0.5 * (y ** 2).sum(2)
        return torch.softmax(logp, dim=1)

    def forward(self, x):
        return ((x @ self.beta.T + self.b0) * self.gate(x)).sum(1, keepdim=True)


def load_mixture(path):
    """Rebuild a saved MixtureExperts module (for E2E warm starts)."""
    from types import SimpleNamespace
    saved = torch.load(path, map_location="cpu")
    st, size_col = saved["state"], saved["size_col"]
    gmm = None if size_col is not None else SimpleNamespace(
        means_=st["means"].numpy(), precisions_cholesky_=st["prec_chol"].numpy(), weights_=np.exp(st["log_w"].numpy()))
    edges = st["size_edges"].numpy() if size_col is not None else None
    m = MixtureExperts(st["num"].numpy(), st["mu"].numpy(), st["V"].numpy(), gmm, st["beta"].numpy(), st["b0"].numpy(),
                       size_col, edges)
    m.load_state_dict(st)
    return m


@torch.no_grad()
def weighted_stats(X, y, W, device, chunk=250_000):
    """Per-expert sums: n_k, sum w x, X' diag(w) X, X' diag(w) y, sum w y."""
    K, P = W.shape[1], X.shape[1]
    XtX = torch.zeros(K, P, P, dtype=torch.float64, device=device)
    sx = torch.zeros(K, P, dtype=torch.float64, device=device)
    Xty = torch.zeros(K, P, dtype=torch.float64, device=device)
    yt, Wt = torch.from_numpy(y.astype(np.float32)), torch.from_numpy(W.astype(np.float32))
    for s in range(0, len(X), chunk):
        x = X[s:s + chunk].to(device).float().mul_(1.0 / Q)
        w, yy = Wt[s:s + chunk].to(device), yt[s:s + chunk].to(device)
        for k in range(K):
            xw = x * w[:, k:k + 1]
            XtX[k] += (xw.T @ x).double()
            sx[k] += xw.sum(0).double()
            Xty[k] += (xw.T @ yy).double()
    return dict(n=W.sum(0), sx=sx.cpu().numpy(), XtX=XtX.cpu().numpy(), Xty=Xty.cpu().numpy(),
                sy=(W * y[:, None]).sum(0))


def fit_experts(ws, lam, avg_var):
    K, P = ws["sx"].shape
    betas, b0 = np.zeros((K, P)), np.zeros(K)
    for k in range(K):
        n, mu = ws["n"][k], ws["sx"][k] / ws["n"][k]
        C = ws["XtX"][k] / n - np.outer(mu, mu)
        c = ws["Xty"][k] / n - mu * ws["sy"][k] / n
        betas[k] = np.linalg.solve(C + lam * avg_var * np.eye(P), c)
        b0[k] = ws["sy"][k] / n - mu @ betas[k]
    return betas, b0


@torch.no_grad()
def predict_module(model, X, device, chunk=250_000):
    model.eval().to(device)
    return torch.cat([model(X[s:s + chunk].to(device).float().mul_(1.0 / Q)).squeeze(-1).cpu()
                      for s in range(0, len(X), chunk)]).numpy().astype(np.float64)


def run_mixture(panel_dir, windows, out_dir, families=None, n_pc=10, sample=200_000, criterion="ic",
                features=None, device=None, seed=0, verbose=True):
    """Fit each family's grid per window, choose on validation, save test/validation predictions and the module."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    features = features or panel_features(panel_dir)
    families = families or list(GRID)
    num = np.array([i for i, f in enumerate(features) if not f.startswith("d_")])
    size_col = features.index("f_size") if "f_size" in features else None
    grid_path = out_dir / "mixture_grid.csv"
    to_x = lambda d: torch.from_numpy(np.ascontiguousarray(d[features].to_numpy(np.int8)))
    for w in windows:
        todo = [f for f in families if not (out_dir / f / f"{w['name']}.parquet").exists()]
        if not todo:
            continue
        t0 = time.time()
        df = load_panel(panel_dir, w["train_start"], w["test_end"], features)
        tr, va, te = split_window(df, w)
        Xtr, Xva, Xte = to_x(tr), to_x(va), to_x(te)
        ytr, rtr, ctr = day_targets(tr)
        yva, rva, _ = day_targets(va)
        st = sufficient_stats(Xtr, np.c_[ytr, rtr], ctr, device)
        C, _, mu_all = _centered(st, 0)
        avg_var = np.trace(C) / len(C)
        mu = mu_all[num]
        V = np.ascontiguousarray(np.linalg.eigh(C[np.ix_(num, num)])[1][:, ::-1][:, :n_pc])
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(tr), min(sample, len(tr)), replace=False)
        Zs = ((Xtr[idx][:, num].numpy().astype(np.float64) / Q) - mu) @ V
        rows, mods = [], []
        for fam in todo:
            for params in GRID[fam]:
                if fam == "GMM-Experts":
                    gmm = GaussianMixture(params["k"], covariance_type="full", reg_covar=1e-4, random_state=seed).fit(Zs)
                    mod = MixtureExperts(num, mu, V, gmm, np.zeros((params["k"], len(features))), np.zeros(params["k"]))
                else:
                    if size_col is None:
                        continue
                    s = Xtr[idx][:, size_col].numpy().astype(np.float32) / Q
                    edges = np.quantile(s, [1 / 3, 2 / 3])
                    mod = MixtureExperts(num, mu, V, None, np.zeros((3, len(features))), np.zeros(3), size_col, edges)
                mod.to(device)
                Wtr = torch.cat([mod.gate(Xtr[a:a + 250_000].to(device).float().mul_(1 / Q)).cpu()
                                 for a in range(0, len(Xtr), 250_000)]).numpy()
                betas, b0 = fit_experts(weighted_stats(Xtr, ytr, Wtr, device), params["lam"], avg_var)
                with torch.no_grad():
                    mod.beta.copy_(torch.as_tensor(betas)), mod.b0.copy_(torch.as_tensor(b0))
                sc = daily_scores(va, predict_module(mod, Xva, device)[:, None], yva, rva).iloc[0]
                rows.append(dict(window=w["name"], family=fam, params=str(params), **sc.to_dict()))
                mods.append(mod.cpu())
        grid = pd.DataFrame(rows)
        for fam in todo:
            sub = grid[grid.family == fam]
            if sub.empty:
                continue
            best = sub[criterion].idxmin() if criterion == "mse" else sub[criterion].idxmax()
            grid.loc[sub.index, "chosen"] = sub.index == best
            mod = mods[best]
            d = out_dir / fam
            d.mkdir(parents=True, exist_ok=True)
            for split, frame, X in (("", te, Xte), ("_val", va, Xva)):
                frame[["date", "permno", "me", "target"]].assign(pred=predict_module(mod, X, device)).to_parquet(
                    d / f"{w['name']}{split}.parquet", index=False)
            torch.save({"state": mod.cpu().state_dict(), "k": mod.beta.shape[0], "size_col": mod.size_col}, d / f"{w['name']}.pt")
        grid.to_csv(grid_path, mode="a", header=not grid_path.exists(), index=False)
        if verbose:
            ch = grid[grid.chosen == True]
            print(f"{w['name']}: {time.time() - t0:.0f}s | " + ", ".join(f"{r.family} {r.params} ic={r.ic:.4f}" for r in ch.itertuples()))
    return pd.read_csv(grid_path) if grid_path.exists() else None
