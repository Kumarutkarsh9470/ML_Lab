"""Checks for the extension modules on the synthetic panel (CPU): reductions, GMM experts, SVR/RFF, tuned MSE-Opt."""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mlm.data import rolling_windows  # noqa: E402
from mlm.metrics import nw_test  # noqa: E402
from mlm.mixture import MixtureExperts, run_mixture  # noqa: E402
from mlm.portfolio import decile_returns, top_portfolio, tuned_kappa  # noqa: E402
from mlm.reduce import load_reduced, run_reduced  # noqa: E402
from mlm.train import TrainConfig, load_predictions, run_rolling  # noqa: E402

# 1. the torch GMM gate equals sklearn's posterior
rng = np.random.default_rng(0)
Z = np.r_[rng.normal(0, 1, (500, 4)), rng.normal(2, 0.5, (500, 4))]
gmm = GaussianMixture(3, covariance_type="full", random_state=0).fit(Z)
mod = MixtureExperts(np.arange(4), np.zeros(4), np.eye(4), gmm, np.zeros((3, 4)), np.zeros(3))
gate = mod.gate(torch.tensor(Z, dtype=torch.float32)).numpy()
err = np.abs(gate - gmm.predict_proba(Z)).max()
print(f"GMM gate vs sklearn predict_proba: max abs diff {err:.1e}")
assert err < 1e-4

panel = ROOT / "data_synth" / "panel"
w = rolling_windows("2014-01-01", "2014-06-30")
out = Path(tempfile.mkdtemp())
cfg = TrainConfig(max_epochs=3, patience=2, device="cpu")
# 2. reductions with Linear and NN3 heads, including the persistence-aware one
names = ["PCA5-Linear", "PCA5-NN3", "PLS3-NN3", "LDA3-NN3", "SFA5-NN3"]
log = run_reduced(names, panel, w, out, cfg, verbose=False)
print(log.groupby("model")["test_r2"].mean().mul(100).round(3).to_dict())
m = load_reduced(out / "PCA5-NN3" / "2014Q1.pt", "PCA5-NN3", 9)
print("reloaded reduced model, W shape", tuple(m.W.shape), "penalty at start", float(m.penalty()))
# 3. GMM experts and size experts
grid = run_mixture(panel, w, out, device="cpu", sample=20_000, verbose=False)
print("mixture chosen:", grid[grid.chosen == True][["window", "family", "params", "ic"]].to_dict("records"))  # noqa: E712
# 4. SVR variants via the loss option
run_rolling(["Linear", "RFF"], panel, w, out, TrainConfig(max_epochs=3, patience=2, device="cpu", loss="eps", eps=0.5,
            weight_decay=1e-4), verbose=False, tag="-SVR")
for name in ["Linear-SVR", "RFF-SVR", "GMM-Experts", "Size-Experts"] + names:
    t, v = load_predictions(out, name), load_predictions(out, name, "val")
    assert len(t) and len(v) and t.pred.notna().all(), name
print("all variants wrote test and validation predictions")
# 5. tuned MSE-Opt and the new reporting columns
t, v = load_predictions(out, "PCA5-NN3"), load_predictions(out, "PCA5-NN3", "val")
kappa, chosen = tuned_kappa(t, v, w, 0.002)
print("tuned kappa per window:", chosen.to_dict())
opt = top_portfolio(t, 0.002, optimize=True, kappa=kappa)
print("tuned MSE-Opt net daily mean", round(opt["net"].mean(), 5), "| H-L raw present:", "H-L raw" in decile_returns(t).columns)
print(nw_test(opt["net"]).round(4).to_dict())
shutil.rmtree(out)
print("extension checks passed")
