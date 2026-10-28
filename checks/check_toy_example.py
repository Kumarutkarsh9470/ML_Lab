"""Recompute the four-asset example of Section 3.1 and Table 1."""
import numpy as np
from scipy.optimize import minimize

x = np.array([[0.3, 0.1], [0.35, 0.2], [0.1, 0.9], [0.2, 1.0]])
r = np.array([0.32, 0.39, 0.92, 1.04])
S = np.array([[0.0064, 0.00288, 0, 0], [0.00288, 0.0144, 0, 0.0084],
              [0, 0, 0.36, 0.294], [0, 0.0084, 0.294, 0.49]])
print("true r from stated pricing rule:", np.r_[x[:2] @ [1, 0.2], x[2:] @ [0.2, 1]], "vs stated", r)
print("Sigma eigenvalues:", np.linalg.eigvalsh(S).round(5))

theta = np.linalg.lstsq(x, r, rcond=None)[0]
print("OLS (no intercept):", theta.round(4), "paper: [0.652, 0.927]")
xi = np.c_[np.ones(4), x]
print("OLS with intercept:", np.linalg.lstsq(xi, r, rcond=None)[0].round(4))

thetas = {"O": np.array([0.652, 0.927]), "L": np.array([0.684, 0.880]), "S": np.array([0.619, 0.973])}
paper_rhat = {"O": [0.288, 0.413, 0.899, 1.057], "L": [0.293, 0.416, 0.861, 1.017], "S": [0.283, 0.411, 0.938, 1.097]}
for k, th in thetas.items():
    print(f"rhat_{k}:", (x @ th).round(3), "paper:", paper_rhat[k], " MSE:", round(np.mean((x @ th - r) ** 2), 6))


def solve(rhat, eta):
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    obj = lambda w: -(w @ rhat - eta * w @ S @ w)
    best = None
    for s in range(20):  # multi-start for safety
        w0 = np.random.default_rng(s).dirichlet(np.ones(4))
        res = minimize(obj, w0, bounds=[(0, 1)] * 4, constraints=cons, method="SLSQP", options={"ftol": 1e-14})
        if best is None or res.fun < best.fun:
            best = res
    return best.x


paper = {10: [31.97, 32.01, 31.90], 1: [62.84, 62.53, 62.90]}
for eta in [10, 1]:
    out = []
    for k in "OLS":
        w = solve(x @ thetas[k], eta)
        out.append(100 * (w @ r - eta * w @ S @ w))
        print(f"eta={eta} g_{k}: w={w.round(4)}")
    print(f"eta={eta} realized utility %: {np.round(out, 2)}  paper: {paper[eta]}")
