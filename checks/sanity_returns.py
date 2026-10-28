"""Where do the long-only returns come from? By year, excluding microcaps, and the size tilt of the top decile."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlm import metrics  # noqa: E402
from mlm.metrics import perf  # noqa: E402
from mlm.portfolio import decile_returns  # noqa: E402
from mlm.train import load_predictions  # noqa: E402

PRED = Path("D:/ML_Markowitz_data/preds/cn")
metrics.ANN = json.loads(Path("D:/ML_Markowitz_data/panel_cn/meta.json").read_text())["ann"]
pd.set_option("display.width", 220)
for m in sys.argv[1:] or ["OLS", "Ridge-rank"]:
    p = load_predictions(PRED, m)
    dec = decile_returns(p)
    by_year = dec[["High(H)", "Low(L)"]].groupby(dec.index.year).mean() * metrics.ANN * 100
    # universe-average return: what an equal-weighted "buy everything" portfolio earns vs the CSI 500
    by_year["All stocks"] = p.groupby("date")["target"].mean().groupby(lambda d: d.year).mean() * metrics.ANN * 100
    print(f"== {m}: annualized % by year (equal-weighted, market-adjusted vs CSI 500)")
    print(by_year.round(1).T.to_string())
    size_pct = p.groupby("date")["me"].rank(pct=True)
    top = p.groupby("date")["pred"].rank(pct=True) > 0.9
    print(f"average market-cap percentile of the top decile: {size_pct[top].mean():.2f} (0.5 = neutral)")
    big = p[size_pct > 0.3]
    h_all, h_big = perf(dec["High(H)"]), perf(decile_returns(big)["High(H)"])
    print(f"top decile, all stocks: Avg {h_all['Avg %']:.1f}% SR {h_all['SR']:.2f} | "
          f"excluding bottom 30% by market cap: Avg {h_big['Avg %']:.1f}% SR {h_big['SR']:.2f}")
    ew = perf(p.groupby("date")["target"].mean())
    print(f"equal-weighted universe vs CSI 500: Avg {ew['Avg %']:.1f}% SR {ew['SR']:.2f}\n")
