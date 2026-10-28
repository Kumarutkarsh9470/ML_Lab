"""Out-of-sample comparison of every model with predictions, over a common test period."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlm import metrics  # noqa: E402
from mlm.linear import GRIDS  # noqa: E402
from mlm.metrics import nw_tstat, oos_r2, perf  # noqa: E402
from mlm.models import MODELS  # noqa: E402
from mlm.portfolio import decile_returns, top_portfolio  # noqa: E402
from mlm.train import load_predictions  # noqa: E402

PRED = Path("D:/ML_Markowitz_data/preds/cn")
metrics.ANN = json.loads(Path("D:/ML_Markowitz_data/panel_cn/meta.json").read_text())["ann"]
start, end = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("2014-01-01", "2023-12-31")
names = [m for m in MODELS + list(GRIDS) if (PRED / m).exists()]
rows, tops = {}, {}
for m in names:
    p = load_predictions(PRED, m)
    p = p[(p.date >= start) & (p.date <= end)]
    if p.empty:
        continue
    pr = p.groupby("date")["pred"].rank(pct=True)
    tr = p.groupby("date")["target"].rank(pct=True)
    ic = pd.concat([pr, tr], axis=1).groupby(p["date"]).corr().xs("pred", level=1)["target"]
    dec = decile_returns(p)
    tp = top_portfolio(p)
    tops[m] = tp["gross"]
    h, hl = perf(dec["High(H)"]), perf(dec["H-L"])
    rows[m] = {"days": p.date.nunique(), "R2 %": 100 * oos_r2(p.pred, p.target), "IC": ic.mean(),
               "IC t": nw_tstat(ic), "H Avg %": h["Avg %"], "H SD %": h["SD %"], "H SR": h["SR"],
               "L Avg %": perf(dec["Low(L)"])["Avg %"], "H-L Avg %": hl["Avg %"], "H-L SR": hl["SR"],
               "TO %": 100 * tp["to_p"].mean()}
t = pd.DataFrame(rows).T
ref = "Linear" if "Linear" in tops else t.index[0]
t[f"t (H vs {ref})"] = [nw_tstat(tops[m] - tops[ref]) if m != ref else np.nan for m in t.index]
pd.set_option("display.width", 220)
print(f"test period {start} to {end}; annualization {metrics.ANN:.1f} days")
print(t.sort_values("H SR", ascending=False).round(3).to_string())
