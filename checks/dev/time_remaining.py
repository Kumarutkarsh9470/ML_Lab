"""Time one window (the largest, 2023Q4) of every remaining run type, to estimate the full runs."""
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlm.data import rolling_windows  # noqa: E402
from mlm.e2e import E2EConfig, run_e2e  # noqa: E402
from mlm.mixture import run_mixture  # noqa: E402
from mlm.reduce import run_reduced  # noqa: E402
from mlm.train import TrainConfig, run_rolling  # noqa: E402

PANEL, PRED = Path("D:/ML_Markowitz_data/panel_cn"), Path("D:/ML_Markowitz_data/preds/cn")
w = [x for x in rolling_windows("2014-01-01", "2023-12-31") if x["name"] == (sys.argv[1] if len(sys.argv) > 1 else "2023Q4")]
tmp = Path(tempfile.mkdtemp())
times = {}


def clock(label, fn):
    t = time.time()
    fn()
    times[label] = time.time() - t
    print(f"{label:38s} {times[label]:6.0f} s", flush=True)


clock("reduced: PCA20-NN3 (MSE)", lambda: run_reduced(["PCA20-NN3"], PANEL, w, tmp, TrainConfig(), verbose=False))
clock("reduced: SFA20-NN3 (MSE)", lambda: run_reduced(["SFA20-NN3"], PANEL, w, tmp, TrainConfig(), verbose=False))
clock("GMM + size experts (all settings)", lambda: run_mixture(PANEL, w, tmp, verbose=False))
clock("RFF-SVR", lambda: run_rolling(["RFF"], PANEL, w, tmp, TrainConfig(loss="eps", weight_decay=1e-4), verbose=False, tag="-SVR"))
for label, cfg, tag in [("E2E top, NN3, one zeta", E2EConfig(), "t1"),
                        ("E2E top, NN3, 3 zetas", E2EConfig(zeta_rel=(0.05, 0.2, 1.0)), "t3"),
                        ("E2E with costs 20 bps, NN3", E2EConfig(gamma=0.002), "tc"),
                        ("E2E mean-SD, NN3, eta=1, gross", E2EConfig(objective="meansd", eta=1.0), "msd"),
                        ("E2E mean-SD, NN3, eta=1, 20 bps", E2EConfig(objective="meansd", eta=1.0, gamma=0.002), "msdtc"),
                        ("PPP, NN3, eta=1, gross", E2EConfig(objective="ppp", eta=1.0), "ppp")]:
    clock(label, lambda: run_e2e(["NN3"], PANEL, w, PRED, tmp, cfg, tag=tag, verbose=False))
shutil.rmtree(tmp)
