"""Check the greedy solver for eq. (19) against scipy's LP solver on random problems."""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlm.portfolio import solve_tc_lp  # noqa: E402

rng = np.random.default_rng(1)
worst = 0.0
for trial in range(300):
    n = rng.integers(10, 80)
    u = 1.0 / (0.1 * n) * rng.uniform(0.6, 1.0)  # cap may be below yesterday's weights
    rhat = rng.normal(0, 0.01, n)
    p = rng.dirichlet(np.ones(n)) * rng.uniform(0.5, 1.0)  # some weight left the universe
    p = np.minimum(p, 1.5 * u)
    gamma = rng.choice([0.0, 0.001, 0.002, 0.004, 0.02])
    w = solve_tc_lp(rhat, p, u, gamma)
    obj = lambda w: rhat @ w - gamma / 2 * np.abs(w - p).sum()
    # variables [w, a, b] with w - p = a - b, a, b >= 0
    c = np.concatenate([-rhat, gamma / 2 * np.ones(n), gamma / 2 * np.ones(n)])
    A_eq = np.vstack([np.hstack([np.eye(n), -np.eye(n), np.eye(n)]),
                      np.concatenate([np.ones(n), np.zeros(2 * n)])])
    b_eq = np.concatenate([p, [1.0]])
    bounds = [(0, u)] * n + [(0, None)] * (2 * n)
    ref = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    assert ref.status == 0
    assert abs(w.sum() - 1) < 1e-9 and w.min() >= -1e-12 and w.max() <= u + 1e-9
    gap = obj(ref.x[:n]) - obj(w)
    worst = max(worst, gap)
    assert gap < 1e-10, (trial, gap)
print(f"300 random problems: greedy matches linprog, worst objective gap {worst:.2e}")
