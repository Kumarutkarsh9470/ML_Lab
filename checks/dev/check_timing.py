"""Where does the high R^2 come from? Day-level vs cross-sectional fit, and CSI 500 alignment with the stocks."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlm import cn  # noqa: E402

P = Path("D:/ML_Markowitz_data/preds/cn")
log = pd.read_csv(P / "train_log.csv")
print("test R2 % by window (NN3):", (log[log.model == "NN3"].set_index("window")["test_r2"] * 100).round(2).to_dict())

for m, w in [("NN3", "2017Q1"), ("NN3", "2014Q2"), ("Linear", "2017Q1")]:
    d = pd.read_parquet(P / m / f"{w}.parquet")
    day = d.groupby("date")[["pred", "target"]].mean()
    tot = (d.target ** 2).sum()
    dm = d.groupby("date")[["pred", "target"]].transform("mean")
    xs = 1 - ((d.target - dm.target - (d.pred - dm.pred)) ** 2).sum() / ((d.target - dm.target) ** 2).sum()
    print(f"{m} {w}: corr(day-mean pred, day-mean target) = {day.pred.corr(day.target):.3f}; "
          f"day-mean share of target variance = {(len(d) / len(day) * (day.target ** 2).sum()) / tot:.2f}; "
          f"cross-sectional R2 = {100 * xs:+.3f}%")

# alignment: same-day correlation of CSI 500 return with the average stock return, at lags -2..2
J = Path("D:/ML_Markowitz_data/raw_cn/joshuaxql/cn_data")
C = Path("D:/ML_Markowitz_data/raw_cn/chenditc/qlib_bin")
codes = sorted(p.name for p in (J / "features").iterdir())[::10]
W = cn.load_wide(J, codes, ["close", "factor"], "2015-01-01", "2018-12-31")
ret = (W["close"] * W["factor"]).pct_change(fill_method=None).mean(axis=1)
idx = cn.load_wide(C, ["I"], ["close"], "2015-01-01", "2018-12-31", ["sh000905"])["close"]["I"].pct_change(fill_method=None)
both = pd.concat([ret.rename("stocks"), idx.rename("csi500")], axis=1).dropna()
print("corr(avg stock return_t, CSI500 return_{t+k}):",
      {k: round(both.stocks.corr(both.csi500.shift(-k)), 3) for k in (-2, -1, 0, 1, 2)})
