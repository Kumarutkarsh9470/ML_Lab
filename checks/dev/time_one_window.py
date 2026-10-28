"""Time one rolling window on the China panel: data loading, the linear family, and one epoch of NN training."""
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlm.data import load_panel, panel_features, rolling_windows, split_window  # noqa: E402
from mlm.linear import run_linear  # noqa: E402
from mlm.models import make_model  # noqa: E402
from mlm.train import TrainConfig, fit_mse, predict, to_tensors  # noqa: E402

PANEL = Path("D:/ML_Markowitz_data/panel_cn")
which = sys.argv[1] if len(sys.argv) > 1 else "2014Q1"
w = [x for x in rolling_windows("2014-01-01", "2023-12-31") if x["name"] == which][0]
feats = panel_features(PANEL)

t = time.time()
df = load_panel(PANEL, w["train_start"], w["test_end"], feats)
tr, va, te = split_window(df, w)
t_load = time.time() - t
print(f"window {which}: train {len(tr):,}, val {len(va):,}, test {len(te):,} rows, {len(feats)} features; load {t_load:.0f}s")

tmp = Path(tempfile.mkdtemp())
t = time.time()
run_linear(PANEL, [w], tmp, verbose=False)
t_lin = time.time() - t
shutil.rmtree(tmp)
print(f"linear family (7 families, 33 settings), incl. its own load: {t_lin:.0f}s")

cfg = TrainConfig(max_epochs=1, patience=1)
t = time.time()
Xtr, ytr = to_tensors(tr, feats, cfg)
Xva, yva = to_tensors(va, feats, cfg)
t_tensor = time.time() - t
for name in ["Linear", "NN3"]:
    torch.manual_seed(0)
    m = make_model(name, len(feats))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.time()
    fit_mse(m, Xtr, ytr, Xva, yva, cfg)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"{name}: one epoch incl. validation pass {time.time() - t:.1f}s (tensor prep {t_tensor:.0f}s, once per window)")
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
