"""Time one window (the largest, 2023Q4) of the mean-SD, mean-SD with costs and PPP runs (Section 4.4)."""
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlm.data import rolling_windows  # noqa: E402
from mlm.e2e import E2EConfig, run_e2e  # noqa: E402

PANEL, PRED = Path("D:/ML_Markowitz_data/panel_cn"), Path("D:/ML_Markowitz_data/preds/cn")
w = [x for x in rolling_windows("2014-01-01", "2023-12-31") if x["name"] == (sys.argv[1] if len(sys.argv) > 1 else "2023Q4")]
tmp = Path(tempfile.mkdtemp())
for label, cfg, tag in [("E2E mean-SD, NN3, eta=1, gross", E2EConfig(objective="meansd", eta=1.0), "msd"),
                        ("E2E mean-SD, NN3, eta=1, 20 bps", E2EConfig(objective="meansd", eta=1.0, gamma=0.002), "msdtc"),
                        ("PPP, NN3, eta=1, gross", E2EConfig(objective="ppp", eta=1.0), "ppp")]:
    t = time.time()
    run_e2e(["NN3"], PANEL, w, PRED, tmp, cfg, tag=tag, verbose=False)
    print(f"{label:38s} {time.time() - t:6.0f} s", flush=True)
shutil.rmtree(tmp)
