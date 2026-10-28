"""Check the mean-SD layer (Section 4.4): solution vs a generic solver, gradient vs finite differences."""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlm.e2e import solve_day  # noqa: E402
from mlm.meansd import capped_softmax, meansd_portfolio, solve_meansd, solve_threshold  # noqa: E402

rng = np.random.default_rng(0)

# 1. exact threshold solver agrees with the bisection solver used by the zeta layer
worst = 0.0
for _ in range(200):
    n = int(rng.integers(50, 3000))
    u, zeta, gamma = 1 / (0.1 * n), float(rng.choice([0.002, 0.02])), float(rng.choice([0.0, 0.004]))
    r = rng.normal(0, 0.01, n)
    p = np.minimum(rng.dirichlet(np.ones(n)), u) if gamma else np.zeros(n)
    w1, _, _ = solve_threshold(r, np.full(n, 2 * zeta), gamma / 2, p, u)
    w2, _ = solve_day(r, u, p, gamma, zeta)
    worst = max(worst, np.abs(w1 - w2).max())
    assert abs(w1.sum() - 1) < 1e-10
print(f"exact threshold solver vs bisection: max weight difference {worst:.1e}")
assert worst < 1e-8


# 2. mean-SD solution vs SLSQP on small problems
def ref(r, sig, eta, u, p, gamma):
    n = len(r)
    f = lambda w: -r @ w + eta * np.sqrt(w @ (sig ** 2 * w) + 1e-18) + gamma / 2 * np.abs(w - p).sum()
    best = None
    for s in range(8):
        x0 = rng.dirichlet(np.ones(n))
        res = minimize(f, np.minimum(x0, u) / np.minimum(x0, u).sum(), bounds=[(0, u)] * n,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], method="SLSQP",
                       options={"ftol": 1e-14, "maxiter": 2000})
        if best is None or res.fun < best.fun:
            best = res
    return best.fun, f


worst = 0.0
for trial in range(30):
    n = int(rng.integers(15, 30))
    u, eta = 0.1 if n > 12 else 1.0, float(rng.choice([0.1, 1.0, 5.0]))
    gamma = float(rng.choice([0.0, 0.004]))
    r, sig = rng.normal(0.001, 0.01, n), rng.uniform(0.01, 0.04, n)
    p = np.minimum(rng.dirichlet(np.ones(n)), u) if gamma else np.zeros(n)
    w, free, s = solve_meansd(r, sig, eta, u, p, gamma)
    fref, f = ref(r, sig, eta, u, p, gamma)
    assert abs(w.sum() - 1) < 1e-9 and w.max() <= u + 1e-12 and abs(s - np.sqrt(w @ (sig ** 2 * w))) < 1e-8
    worst = max(worst, f(w) - fref)
    assert f(w) - fref < 1e-7, (trial, f(w) - fref)
print(f"mean-SD solution: 30 problems, never worse than SLSQP by more than {worst:.1e}")

# 3. gradient of a smooth loss of w* against central finite differences
torch.set_default_dtype(torch.float64)
for gamma in (0.0, 0.004):
    n, eta, u = 60, 2.0, 0.1
    r = torch.tensor(rng.normal(0.002, 0.01, n))[None].requires_grad_(True)
    sig = torch.tensor(rng.uniform(0.01, 0.04, n))[None]
    p = torch.tensor(np.minimum(rng.dirichlet(np.ones(n)), u))[None] if gamma else None
    y = torch.tensor(rng.normal(0, 0.02, n))
    L = lambda rr: (meansd_portfolio(rr, sig, torch.ones(1, n), torch.tensor([[u]]), eta, p, gamma)[0] * y).sum()
    L(r).backward()
    g = r.grad[0].numpy()
    h, fd = 1e-7, np.zeros(n)
    for i in range(n):
        e = torch.zeros(1, n)
        e[0, i] = h
        fd[i] = (L(r.detach() + e) - L(r.detach() - e)).item() / (2 * h)
    err = np.max(np.abs(g - fd)) / np.max(np.abs(fd))
    print(f"mean-SD gradient (gamma={gamma}): max rel. error vs finite differences {err:.1e}, {int((g != 0).sum())} nonzero")
    assert err < 1e-4

# 4. PPP capped softmax: valid weights
z = torch.tensor(rng.normal(0, 3, (4, 50)))
w = capped_softmax(z, torch.ones(4, 50), torch.full((4, 1), 0.1))
print(f"capped softmax: sums {w.sum(1).numpy().round(6)}, max weight {w.max():.4f}")
assert torch.allclose(w.sum(1), torch.ones(4)) and w.max() <= 0.1 + 1e-9
print("mean-SD checks passed")
