"""Check the E2E portfolio layer: solution against a generic solver, gradient against finite differences."""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlm.e2e import portfolio  # noqa: E402

rng = np.random.default_rng(0)
torch.set_default_dtype(torch.float64)


def reference(v, u, p, gamma, zeta):
    """Generic solve of min -v'w + zeta||w||^2 + gamma/2 ||w-p||_1, 0<=w<=u, sum w = 1 (w = p + a - b split)."""
    n = len(v)
    obj = lambda x: -v @ x[:n] + zeta * x[:n] @ x[:n] + gamma / 2 * (x[n:2 * n] + x[2 * n:]).sum()
    cons = [{"type": "eq", "fun": lambda x: x[:n].sum() - 1},
            {"type": "eq", "fun": lambda x: x[:n] - p - x[n:2 * n] + x[2 * n:]}]
    x0 = np.r_[np.full(n, 1 / n), np.maximum(1 / n - p, 0), np.maximum(p - 1 / n, 0)]
    res = minimize(obj, x0, bounds=[(0, u)] * n + [(0, None)] * (2 * n), constraints=cons, method="SLSQP",
                   options={"ftol": 1e-14, "maxiter": 1000})
    return res.x[:n], res.fun


worst = 0.0
for trial in range(60):
    n = int(rng.integers(20, 60))
    u = 1 / (0.1 * n)
    v = rng.normal(0, 0.01, n)
    gamma = float(rng.choice([0.0, 0.002, 0.01]))
    zeta = float(rng.choice([0.002, 0.01, 0.05]))
    p = rng.dirichlet(np.ones(n)) if gamma > 0 else np.zeros(n)
    p = np.minimum(p, u)
    w = portfolio(torch.tensor(v)[None], torch.ones(1, n), torch.tensor([[u]]), torch.tensor(p)[None], gamma, zeta)[0].numpy()
    f = lambda w: -v @ w + zeta * w @ w + gamma / 2 * np.abs(w - p).sum()
    w_ref, _ = reference(v, u, p, gamma, zeta)
    assert abs(w.sum() - 1) < 1e-9 and w.min() >= -1e-12 and w.max() <= u + 1e-12
    gap = f(w) - f(w_ref)
    worst = max(worst, gap)
    assert gap < 1e-9, (trial, gap)
print(f"solution: 60 random problems, never worse than SLSQP (worst objective gap {worst:.1e})")

# gradient of a smooth function of w* with respect to v, against central finite differences
for gamma in (0.0, 0.004):
    n, zeta = 40, 0.01
    u = 1 / (0.1 * n)
    v = torch.tensor(rng.normal(0, 0.01, n))[None].requires_grad_(True)
    p = torch.tensor(np.minimum(rng.dirichlet(np.ones(n)), u))[None] if gamma else torch.zeros(1, n)
    y = torch.tensor(rng.normal(0, 0.02, n))
    loss = lambda vv: (portfolio(vv, torch.ones(1, n), torch.tensor([[u]]), p, gamma, zeta)[0] * y).sum()
    loss(v).backward()
    g = v.grad[0].numpy()
    h, fd = 1e-7, np.zeros(n)
    for i in range(n):
        e = torch.zeros(1, n)
        e[0, i] = h
        fd[i] = (loss(v.detach() + e) - loss(v.detach() - e)).item() / (2 * h)
    err = np.max(np.abs(g - fd)) / np.max(np.abs(fd))
    print(f"gradient (gamma={gamma}): max rel. error vs finite differences {err:.1e}, "
          f"{int((g != 0).sum())} of {n} stocks with nonzero gradient")
    assert err < 1e-4
# the NumPy one-day solver (used for the cost-aware model) must agree with the batched layer
import time  # noqa: E402

from mlm.e2e import solve_day  # noqa: E402

worst, t_np = 0.0, 0.0
for trial in range(200):
    n = int(rng.integers(500, 5000))
    u = 1 / (0.1 * n)
    v = rng.normal(0, 0.01, n)
    gamma, zeta = float(rng.choice([0.0, 0.002, 0.01])), float(rng.choice([0.002, 0.01, 0.05]))
    p = np.minimum(rng.dirichlet(np.ones(n)) * rng.uniform(0.6, 1.0), u) if gamma else np.zeros(n)
    t0 = time.perf_counter()
    w_np, free = solve_day(v, u, p, gamma, zeta)
    t_np += time.perf_counter() - t0
    w_t = portfolio(torch.tensor(v)[None], torch.ones(1, n), torch.tensor([[u]]), torch.tensor(p)[None], gamma, zeta)[0].numpy()
    worst = max(worst, np.abs(w_np - w_t).max())
    assert abs(w_np.sum() - 1) < 1e-10
print(f"NumPy one-day solver: max weight difference vs batched layer {worst:.1e}; {1000 * t_np / 200:.2f} ms per day")
assert worst < 1e-8
print("E2E layer checks passed")
