"""GPU-resident training must reproduce a finished window, and be faster. Compares against saved 2014Q1 results."""
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlm.data import rolling_windows  # noqa: E402
from mlm.train import TrainConfig, run_rolling  # noqa: E402

PANEL, PRED = Path("D:/ML_Markowitz_data/panel_cn"), Path("D:/ML_Markowitz_data/preds/cn")
w = rolling_windows("2014-01-01", "2014-03-31")
ref_log = pd.read_csv(PRED / "train_log.csv")
for on_gpu in (True,):
    out = Path(tempfile.mkdtemp())
    t0 = time.time()
    log = run_rolling(["NN3"], PANEL, w, out, TrainConfig(data_on_device=on_gpu), verbose=False)
    secs = time.time() - t0
    new = pd.read_parquet(out / "NN3" / "2014Q1.parquet")
    old = pd.read_parquet(PRED / "NN3" / "2014Q1.parquet")
    ref = ref_log[(ref_log.window == "2014Q1") & (ref_log.model == "NN3")].iloc[0]
    print(f"data on GPU={on_gpu}: {secs:.0f}s total (model {log.seconds.iloc[0]:.0f}s) vs {ref.seconds:.0f}s before; "
          f"epochs {log.epochs.iloc[0]} vs {ref.epochs}; val MSE {log.val_mse.iloc[0]:.6f} vs {ref.val_mse:.6f}; "
          f"max |pred diff| {np.max(np.abs(new.pred.values - old.pred.values)):.2e}, "
          f"pred corr {np.corrcoef(new.pred, old.pred)[0, 1]:.6f}")
    shutil.rmtree(out)
