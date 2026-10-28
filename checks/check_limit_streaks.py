"""Test: is the high R^2 (and part of the top-decile return) from new listings locked at the upper price limit?"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlm import cn  # noqa: E402
from mlm.metrics import oos_r2  # noqa: E402

P = Path("D:/ML_Markowitz_data/preds/cn")
U = Path("D:/ML_Markowitz_data/panel_cn_unranked")
J = Path("D:/ML_Markowitz_data/raw_cn/joshuaxql/cn_data")
w = sys.argv[1] if len(sys.argv) > 1 else "2017Q1"
d = pd.read_parquet(P / "NN3" / f"{w}.parquet")
year = d.date.dt.year.iloc[0]
u = pd.read_parquet(U / f"{year}.parquet", columns=["date", "permno", "code", "f_age", "f_uphits21"])
d = d.merge(u, on=["date", "permno"], how="left")
d["age_days"] = np.expm1(d["f_age"])

# is the stock locked at its upper limit all day on t+1 (open = high = low = up_limit)? then it cannot be bought
codes = sorted(d.code.dropna().unique())
L = cn.load_wide(J, codes, ["open", "high", "low", "up_limit"], d.date.min(), d.date.max() + pd.Timedelta(days=10))
locked = (L["low"] >= L["up_limit"] - 0.005) & (L["high"] <= L["up_limit"] + 0.005)
nxt = locked.shift(-1)  # locked on the next trading day
d["locked_next"] = [bool(nxt.at[t, c]) if c in nxt.columns and t in nxt.index and pd.notna(nxt.at[t, c]) else False
                    for t, c in zip(d.date, d.code)]

new = d.age_days < 60
top = d.groupby("date")["pred"].rank(pct=True) > 0.9
print(f"window {w}: {len(d):,} stock-days; listed < 60 days: {new.mean():.1%}; locked limit-up on t+1: {d.locked_next.mean():.1%}")
print(f"  share of top decile that is listed < 60 days: {new[top].mean():.1%}; locked on t+1: {d.locked_next[top].mean():.1%}")
print(f"  mean target: all {d.target.mean():.4f} | locked on t+1 {d.target[d.locked_next].mean():.4f} | new listings {d.target[new].mean():.4f}")
for lab, keep in [("all", np.ones(len(d), bool)), ("excluding listed < 60 days", ~new),
                  ("excluding locked on t+1", ~d.locked_next), ("excluding both", ~new & ~d.locked_next)]:
    k = d[keep]
    top_k = k.groupby("date")["pred"].rank(pct=True) > 0.9
    print(f"  {lab:28s} R2 {100 * oos_r2(k.pred, k.target):+6.2f}%   top-decile mean daily return {100 * k.target[top_k].mean():.3f}%")
