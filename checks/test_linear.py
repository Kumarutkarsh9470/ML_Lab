"""Check the closed-form linear family against scikit-learn on simulated data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlm import linear as L  # noqa: E402
from mlm.data import Q  # noqa: E402

rng = np.random.default_rng(0)
n_days, per_day, p_num, p_dum = 60, 400, 30, 4
n = n_days * per_day
A = rng.normal(size=(p_num, p_num)) / np.sqrt(p_num)
Z = rng.normal(size=(n, p_num)) @ A  # correlated features
Xn = np.clip(Z / 3, -1, 1)
D = np.eye(p_dum)[rng.integers(0, p_dum, n)]
Xf = np.c_[Xn, D]
X8 = np.round(Xf * Q).astype(np.int8)
Xq = X8.astype(np.float64) / Q  # what the model actually sees
beta_true = rng.normal(size=p_num + p_dum) * 0.01
df = pd.DataFrame({"date": np.repeat(pd.bdate_range("2020-01-01", periods=n_days), per_day),
                   "target": Xq @ beta_true + 0.05 * rng.standard_t(5, n)})
y, rank, cls = L.day_targets(df)
assert np.bincount(cls).min() == np.bincount(cls).max() == n // 10, "deciles should be equal-sized"
st = L.sufficient_stats(torch.from_numpy(X8), np.c_[y, rank], cls, "cpu")
C, c, mu = L._centered(st, 0)
num, dum = np.arange(p_num), np.arange(p_num, p_num + p_dum)
eig = np.linalg.eigh(C[np.ix_(num, num)])[1][:, ::-1]
fit = lambda fam, **kw: L.fit_family(st, fam, kw, num, dum, C, c, eig)


def close(a, b, tol, name):
    err = np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12)
    print(f"{name:34s} max rel. error {err:.2e}")
    assert err < tol, name


# every row has one industry dummy, so OLS coefficients are not unique; compare predictions
b = fit("OLS")
close(Xq @ b + (y.mean() - mu @ b), LinearRegression().fit(Xq, y).predict(Xq), 1e-4, "OLS predictions vs sklearn")
lam = 0.1
avg_var = np.trace(C) / len(C)
close(fit("Ridge", lam=lam), Ridge(alpha=lam * avg_var * n).fit(Xq, y).coef_, 1e-5, "Ridge vs sklearn")
for k in (1, 3, 8):
    close(fit("PLS", k=k), PLSRegression(n_components=k, scale=False).fit(Xq, y).coef_.ravel(), 1e-4, f"PLS k={k} vs sklearn")
k = 6
pca = PCA(k).fit(Xq[:, num])
Zp = np.c_[pca.transform(Xq[:, num]), Xq[:, dum]]
ref = LinearRegression().fit(Zp, y)
pred_ref = ref.predict(Zp)
b = fit("PCR", k=k)
pred = Xq @ b + (y.mean() - mu @ b)
close(pred, pred_ref, 1e-4, "PCR k=6 predictions vs sklearn")
# Without shrinkage both are textbook LDA. (With shrinkage sklearn standardizes first and also shrinks the
# total scatter, so its between-class matrix differs by construction.)
lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage=None).fit(Xq[:, num], cls)
vecs, _, _ = L._lda_directions(st, num, 0.0)
for m in (1, 3):
    P1, P2 = Xq[:, num] @ vecs[:, :m], Xq[:, num] @ lda.scalings_[:, :m]
    q1, q2 = np.linalg.qr(P1 - P1.mean(0))[0], np.linalg.qr(P2 - P2.mean(0))[0]
    cc = np.linalg.svd(q1.T @ q2, compute_uv=False)
    print(f"{f'LDA subspace m={m} vs sklearn':34s} canonical corr min {cc.min():.6f}")
    assert cc.min() > 0.999
# daily rank IC against scipy
yhat = Xq @ fit("Ridge", lam=0.01)
sc = L.daily_scores(df, yhat[:, None], y, rank)
ref_ic = df.assign(p=yhat).groupby("date").apply(lambda g: spearmanr(g["p"], g["target"])[0]).mean()
print(f"{'daily rank IC vs scipy':34s} {sc['ic'][0]:.6f} vs {ref_ic:.6f}")
assert abs(sc["ic"][0] - ref_ic) < 1e-6
print("all linear-family checks passed")
