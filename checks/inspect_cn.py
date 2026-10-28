"""Inspect the two Qlib dumps: layout, field conventions, and cross-source agreement."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
J = Path(r"D:\ML_Markowitz_data\raw_cn\joshuaxql\cn_data")
C = Path(r"D:\ML_Markowitz_data\raw_cn\chenditc\qlib_bin")


def head(p, n=5):
    print(f"--- {p.relative_to(p.parents[1])}")
    with open(p, encoding="utf-8") as f:
        for _ in range(n):
            print("   ", f.readline().rstrip())


def read_bin(root, code, field):
    cal = pd.to_datetime(pd.read_csv(root / "calendars" / "day.txt", header=None)[0])
    a = np.fromfile(root / "features" / code / f"{field}.day.bin", dtype="<f4")
    start = int(a[0])
    return pd.Series(a[1:], index=cal[start:start + len(a) - 1].values, name=field)


for root in (J, C):
    cal = pd.read_csv(root / "calendars" / "day.txt", header=None)[0]
    print(root.name, "calendar:", cal.iloc[0], "to", cal.iloc[-1], len(cal), "days")
    head(root / "instruments" / "all.txt", 3)
head(J / "instruments" / "st.txt", 5)
head(J / "instruments" / "csi500.txt", 3)
head(J / "stock_basic.csv", 4)
print("industry names:", json.loads((J / "industry_names.json").read_text(encoding="utf-8")) if (J / "industry_names.json").exists() else None)
ind = sorted((J / "industry").iterdir())
print("industry dir:", [p.name for p in ind[:5]], "...", len(ind))
sub = ind[0]
if sub.is_dir():
    print("  inside", sub.name, [p.name for p in sorted(sub.iterdir())[:5]])
else:
    head(sub, 4)
fin = sorted((J / "financial").iterdir())
print("financial dir:", [p.name for p in fin[:3]], "...", len(fin))
f0 = fin[0]
if f0.is_dir():
    files = sorted(f0.iterdir())
    print("  inside", f0.name, len(files), [p.name for p in files[:6]], [p.stat().st_size for p in files[:6]])
feat = sorted((J / "features").iterdir())
print("features dirs:", [p.name for p in feat[:3]], "...", [p.name for p in feat[-3:]], len(feat))
cfeat = sorted((C / "features").iterdir())
print("chenditc features:", [p.name for p in cfeat[:3]], len(cfeat), [p.name for p in sorted((C / "features" / "sh000905").iterdir())])

# conventions on one large stock with known splits/dividends
code_j, code_c = "600519.SH", "sh600519"
dj = pd.concat([read_bin(J, code_j, f) for f in ["open", "close", "high", "low", "volume", "vwap", "factor", "total_mv", "circ_mv", "up_limit", "down_limit"]], axis=1)
dc = pd.concat([read_bin(C, code_c, f) for f in ["open", "close", "volume", "amount", "vwap", "factor", "change", "adjclose"]], axis=1)
print("\njoshuaxql 600519:\n", dj.loc["2014-06-20":"2014-07-02"])
print("chenditc 600519:\n", dc.loc["2014-06-20":"2014-07-02"])
both = dj[["close", "vwap", "factor", "volume"]].join(dc[["close", "vwap", "factor", "volume", "amount"]], lsuffix="_j", rsuffix="_c", how="inner").loc["2010":"2023"]
print("raw close if j_close / j_factor:", (both.close_j / both.factor_j).loc["2014-06-25":"2014-06-27"].round(2).tolist())
r_j = both.vwap_j.pct_change(); r_c = both.vwap_c.pct_change()
print(f"vwap daily return corr j vs c: {r_j.corr(r_c):.5f}; median |diff| {np.nanmedian(np.abs(r_j - r_c)):.2e}")
print("ratio close_j/close_c describe:", (both.close_j / both.close_c).describe().round(4).to_dict())

# delisted stock and up/down limits
for code in ["600087.SH", "000024.SZ"]:
    s = read_bin(J, code, "close")
    print(code, "bars:", s.notna().sum(), s.dropna().index.min().date(), "to", s.dropna().index.max().date())
idx = read_bin(C, "sh000905", "close")
print("CSI500 (chenditc) close:", idx.loc["2010-01-04":"2010-01-08"].round(2).tolist(), "last", idx.dropna().index.max().date())
lim = dj.loc["2015-07-01":"2015-07-10", ["close", "up_limit", "down_limit", "factor"]]
print("limits vs close:\n", lim)
