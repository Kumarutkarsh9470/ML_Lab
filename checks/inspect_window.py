"""Inspect saved predictions for given model/window pairs: R^2, rank IC, spread, duplicates."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlm.metrics import oos_r2  # noqa: E402

P = Path("D:/ML_Markowitz_data/preds/cn")
for arg in sys.argv[1:]:
    m, w = arg.split(":")
    d = pd.read_parquet(P / m / f"{w}.parquet")
    pr, tr = d.groupby("date")["pred"].rank(), d.groupby("date")["target"].rank()
    ic = pd.concat([pr, tr], axis=1).groupby(d["date"]).corr().xs("pred", level=1)["target"].mean()
    print(f"{m:6s} {w}: rows {len(d):,}  R2 {100 * oos_r2(d.pred, d.target):+.3f}%  pred sd {d.pred.std():.5f}  "
          f"target sd {d.target.std():.4f}  rank IC {ic:.3f}  duplicates {d.duplicated(['date', 'permno']).sum()}")
