"""Section 4.4: the mean-standard-deviation portfolio layer, with an exact gradient, and a PPP benchmark."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

_POOL = ThreadPoolExecutor(8)


def solve_threshold(r, a, c, p, u):
    """Exact w_i = clip(S_i(r_i - tau), 0, u) with sum(w) = 1, by sorting breakpoints."""
    n = len(r)
    h = p > 0
    e = np.concatenate([r - (a * u + c), r - (a * p + c), r[h] - (a[h] * p[h] - c), r[h] + c])
    ds = np.concatenate([-1 / a, 1 / a, -1 / a[h], 1 / a[h]])
    order = np.argsort(e, kind="stable")
    e, ds = e[order], ds[order]
    slope = np.cumsum(ds)
    F = n * u + np.concatenate([[0.0], np.cumsum(slope[:-1] * np.diff(e))])
    k = int(np.searchsorted(-F, -1.0))
    k = min(max(k, 1), len(e) - 1)
    tau = e[k - 1] + (1.0 - F[k - 1]) / slope[k - 1] if slope[k - 1] != 0 else e[k - 1]
    A = r - tau
    s = np.where(A > a * p + c, (A - c) / a, np.where(A < a * p - c, (A + c) / a, p))
    w = np.clip(s, 0.0, u)
    plateau = (A <= a * p + c) & (A >= a * p - c) & (c > 0)
    free = (w > 1e-12) & (w < u - 1e-12) & ~plateau
    return w, free, tau


_S_CACHE = {}


def solve_meansd(r, sigma, eta, u, p=None, gamma=0.0, xtol=1e-10, key=None):
    """Mean-SD portfolio for one day: returns weights, the free set and the portfolio volatility s."""
    from scipy.optimize import brentq
    p = np.zeros_like(r) if p is None else p
    c = gamma / 2
    var = sigma ** 2

    def at(s):
        w, free, tau = solve_threshold(r, eta * var / s, c, p, u)
        return w, free, np.sqrt(w @ (var * w))

    h = lambda x: np.log(max(at(np.exp(x))[2], 1e-300)) - x
    lo_lim, hi_lim = np.log(1e-8), np.log(float(sigma.max()))
    ck = None if key is None else (key, eta, gamma)
    if ck in _S_CACHE:
        x0, step = _S_CACHE[ck], 0.05
        lo, hi = max(x0 - step, lo_lim), min(x0 + step, hi_lim)
        hlo, hhi = h(lo), h(hi)
        while hlo <= 0 and lo > lo_lim:
            step *= 4
            lo = max(x0 - step, lo_lim)
            hlo = h(lo)
        while hhi >= 0 and hi < hi_lim:
            step *= 4
            hi = min(x0 + step, hi_lim)
            hhi = h(hi)
    else:
        lo, hi = lo_lim, hi_lim
        hlo, hhi = h(lo), h(hi)
    x = brentq(h, lo, hi, xtol=xtol) if hlo > 0 > hhi else (lo if hlo <= 0 else hi)
    if ck is not None:
        _S_CACHE[ck] = x
    s = np.exp(x)
    w, free, _ = at(s)
    return w, free, s


def _vjp(g, w, free, s, sigma, eta):
    """Vector-Jacobian product g' dw*/drhat for one day, from the implicit function theorem."""
    out = np.zeros_like(w)
    if not free.any():
        return out
    b = 1.0 / (eta * sigma[free] ** 2)
    wf, gf = w[free], g[free]
    A, WF = b.sum(), wf.sum()
    VF = (sigma[free] ** 2 * wf ** 2).sum()
    Dn = s - VF / s + WF ** 2 / (eta * s * A)
    d1, d2 = -(s * WF / (eta * A)) / Dn, (s / eta) / Dn
    c1, c2 = 1 / A + WF / (s * s * A) * d1, WF / (s * s * A) * d2
    G1, G2 = -s * (gf * b).sum(), (gf * wf).sum() / s
    K1, K2 = G1 * c1 + G2 * d1, G1 * c2 + G2 * d2
    out[free] = b * (s * gf + K1) + wf * K2
    return out


class MeanSDLayer(torch.autograd.Function):
    """Batched over days (padded): rhat (B, N), sigma (B, N), mask (B, N), u (B, 1), p (B, N)."""

    @staticmethod
    def forward(ctx, rhat, sigma, mask, u, p, eta, gamma, keys):
        R, S, M, U, P = (t.detach().double().cpu().numpy() for t in (rhat, sigma, mask, u, p))
        W, FREE, SD = np.zeros_like(R), np.zeros(R.shape, bool), np.zeros(len(R))

        def one(k):
            n = int(M[k].sum())
            key = None if keys is None else keys[k]
            W[k, :n], FREE[k, :n], SD[k] = solve_meansd(R[k, :n], S[k, :n], eta, U[k, 0], P[k, :n], gamma, key=key)

        if len(R) > 1:
            list(_POOL.map(one, range(len(R))))
        else:
            one(0)
        ctx.saved = (W, FREE, SD, S, M, eta)
        return torch.from_numpy(W).to(rhat.device, rhat.dtype)

    @staticmethod
    def backward(ctx, g):
        W, FREE, SD, S, M, eta = ctx.saved
        G = g.detach().double().cpu().numpy()
        out = np.zeros_like(W)
        for k in range(len(W)):
            n = int(M[k].sum())
            out[k, :n] = _vjp(G[k, :n], W[k, :n], FREE[k, :n], SD[k], S[k, :n], eta)
        return torch.from_numpy(out).to(g.device, g.dtype), None, None, None, None, None, None, None


def meansd_portfolio(rhat, sigma, mask, u, eta, p=None, gamma=0.0, keys=None):
    """Mean-SD layer; keys (one per day, e.g. the date) warm-start the volatility search across epochs."""
    p = torch.zeros_like(rhat) if p is None else p
    return MeanSDLayer.apply(rhat, sigma, mask, u, p, float(eta), float(gamma), None if keys is None else tuple(keys))


def capped_softmax(z, mask, u, iters=8):
    """PPP mapping from scores to long-only, fully invested weights with a cap (no optimization layer)."""
    w = torch.softmax(z.masked_fill(mask == 0, float("-inf")), dim=1)
    for _ in range(iters):
        capped = w >= u
        excess = (w - u).clamp(min=0).sum(1, keepdim=True)
        free_mass = (w * (~capped)).sum(1, keepdim=True).clamp(min=1e-12)
        w = torch.where(capped, u.expand_as(w), w + excess * w / free_mass * (~capped))
    return w * mask
