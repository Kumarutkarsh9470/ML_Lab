"""End-to-end smoke test of every E2E variant on the synthetic panel (CPU, few epochs): objectives, warm starts
from each saved model type, decision-aware PCA, training from scratch, zeta tuning and the mean-SD benchmark."""
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mlm.data import rolling_windows  # noqa: E402
from mlm.e2e import E2EConfig, backtest_meansd, load_sigma, run_e2e  # noqa: E402
from mlm.linear import run_linear  # noqa: E402
from mlm.mixture import run_mixture  # noqa: E402
from mlm.reduce import run_reduced  # noqa: E402
from mlm.train import TrainConfig, load_predictions, run_rolling  # noqa: E402

panel = ROOT / "data_synth" / "panel"
w = rolling_windows("2014-01-01", "2014-03-31")
tmp = Path(tempfile.mkdtemp())
warm, out, unr = tmp / "warm", tmp / "e2e", tmp / "unranked"
# fake one-year volatilities for the mean-SD objective
for y in range(2010, 2017):
    f = panel / f"{y}.parquet"
    if f.exists():
        d = pd.read_parquet(f, columns=["date", "permno"])
        unr.mkdir(parents=True, exist_ok=True)
        d.assign(f_ivol252=np.random.default_rng(y).uniform(0.01, 0.04, len(d))).to_parquet(unr / f"{y}.parquet")
cfg = TrainConfig(max_epochs=2, patience=1, device="cpu")
run_rolling(["Linear", "NN3"], panel, w, warm, cfg, verbose=False)
run_linear(panel, w, warm, families=["Ridge"], device="cpu", verbose=False)
run_reduced(["PCA5-NN3", "PCA5-Linear"], panel, w, warm, cfg, verbose=False)
run_mixture(panel, w, warm, families=["GMM-Experts"], device="cpu", sample=20_000, verbose=False)

base = dict(max_epochs=2, patience=1, device="cpu")
runs = [
    ("top", ["NN3", "Ridge", "PCA5-NN3", "GMM-Experts"], E2EConfig(**base, zeta_rel=(0.1, 0.5))),
    ("top-tc", ["NN3"], E2EConfig(**base, gamma=0.002)),
    ("da-pca", ["PCA5-Linear", "PCA5-NN3"], E2EConfig(**base, da_mu=1.0)),
    ("scratch", ["NN3"], E2EConfig(**base, warm_start=False)),
    ("meansd", ["NN3"], E2EConfig(**base, objective="meansd", eta=1.0)),
    ("meansd-tc", ["NN3"], E2EConfig(**base, objective="meansd", eta=1.0, gamma=0.002)),
    ("ppp", ["NN3"], E2EConfig(**base, objective="ppp", eta=1.0)),
]
for tag, models, c in runs:
    t0 = time.time()
    log = run_e2e(models, panel, w, warm, out, c, tag=tag, verbose=False, unranked_dir=unr)
    for m in models:
        p = pd.read_parquet(out / f"{tag}-{m}" / "2014Q1.parquet")
        dly = pd.read_parquet(out / f"{tag}-{m}" / "2014Q1_daily.parquet")
        assert p.pred.notna().all() and dly.net.notna().all() and len(dly) == p.date.nunique(), (tag, m)
    print(f"{tag:10s} {models}: {time.time() - t0:.0f}s, daily net {dly.net.mean():+.5f}, turnover {dly.to_p.mean():.3f}"
          + (f", chosen zeta_rel {log.zeta_rel.tolist()}" if tag == "top" else ""))
# two-stage mean-SD benchmark (MSE-Opt for Section 4.4)
t = load_predictions(warm, "NN3")
bt = backtest_meansd(t, load_sigma(unr, t), eta=1.0, gamma=0.002)
print(f"mean-SD MSE-Opt backtest: {len(bt)} days, net {bt.net.mean():+.5f}, ex-ante sd {bt.sd_exante.mean():.4f}")
shutil.rmtree(tmp)
print("E2E variant checks passed")
