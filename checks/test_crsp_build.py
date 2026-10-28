"""Smoke test of mlm.crsp on fake files shaped like the WRDS tables."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlm import crsp, data  # noqa: E402

tmp = Path(sys.argv[1])
raw = tmp / "raw"
raw.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
dates = pd.bdate_range("2007-01-01", "2011-12-31")
N = 300
for y in range(2007, 2012):
    d = dates[dates.year == y]
    df = pd.DataFrame({"permno": np.tile(np.arange(10000, 10000 + N), len(d)),
                       "date": np.repeat(d, N)})
    n = len(df)
    df["ret"] = rng.standard_t(4, n) * 0.02
    df.loc[rng.random(n) < 0.01, "ret"] = np.nan
    df["prc"] = np.abs(rng.normal(30, 20, n)) * np.where(rng.random(n) < 0.05, -1, 1)
    df["vol"] = np.where(rng.random(n) < 0.03, 0, rng.integers(100, 1_000_000, n)).astype(float)
    df["shrout"] = rng.integers(1_000, 500_000, n).astype(float)
    df["cfacpr"] = 1.0
    df["askhi"] = df["prc"].abs() * 1.02
    df["bidlo"] = df["prc"].abs() * 0.98
    df["shrcd"], df["exchcd"] = 11, 1
    df["siccd"] = rng.choice([100, 1311, 2834, 3571, 4911, 5000, 5311, 6020, 7372, 9999, 0], n)
    df = df[~((df.permno == 10005) & (df.date > "2010-06-30"))]  # delisted stock
    df.to_parquet(raw / f"dsf_{y}.parquet", index=False)
pd.DataFrame({"date": dates, "vwretd": rng.normal(0.0004, 0.01, len(dates))}).to_parquet(raw / "dsi.parquet")
pd.DataFrame({"permno": [10005, 10006], "dlstdt": pd.to_datetime(["2010-07-02", "2011-01-01"]),
              "dlret": [-0.3, np.nan]}).to_parquet(raw / "dsedelist.parquet")

dsf, mkt = crsp.load_raw(raw)
assert ((dsf.permno == 10005) & (dsf.ret == np.float32(-0.3))).sum() == 1, "delisting row missing"
panel, numeric, dummies = crsp.build_features(dsf, mkt, "2010-01-01", "2010-12-31")
print(panel.shape, len(numeric), len(dummies))
assert panel.groupby(["date", "permno"]).size().max() == 1
assert (panel[dummies].sum(axis=1) <= 1).all()

# hand check of the target and two features for one stock-day
s = dsf[dsf.permno == 10010].set_index("date")
d0 = pd.Timestamp("2010-06-15")
row = panel[(panel.permno == 10010) & (panel.date == d0)].iloc[0]
t2 = s.index[s.index.get_loc(d0) + 2]
assert np.isclose(row.target, s.ret[t2] - mkt[t2], atol=1e-6), (row.target, s.ret[t2] - mkt[t2])
past = s.ret.loc[:d0].iloc[-21:]
assert np.isclose(row.f_r21, np.expm1(np.log1p(past.dropna()).sum()), atol=1e-5)
assert np.isclose(row.f_vol21, past.std(), atol=1e-5)
print("target and features match hand computation")

osap = pd.DataFrame({"permno": np.repeat(np.arange(10000, 10000 + N), 13),
                     "yyyymm": np.tile([200912] + [201000 + m for m in range(1, 13)], N)})
osap["SigA"] = rng.normal(size=len(osap)).astype(np.float32)
osap["SigB"] = rng.normal(size=len(osap)).astype(np.float32)
feats = crsp.write_ranked_panel(panel, tmp / "panel", numeric, dummies, osap)
p = data.load_panel(tmp / "panel", "2010-01-01", "2010-12-31")
print(p.shape, p[feats].dtypes.unique())
assert len(feats) == len(numeric) + 2 + len(dummies)
x = p[feats[:len(numeric) + 2]] / 127
assert x.min().min() >= -1 and x.max().max() <= 1
# OSAP signal of month m is used during month m+1
o = osap.set_index(["permno", "yyyymm"]).SigA
r = p[(p.date == "2010-03-10")].set_index("permno")["o_SigA"]
raw_feb = o.xs(201002, level="yyyymm").reindex(r.index)
assert np.corrcoef(r.rank(), raw_feb.rank())[0, 1] > 0.999
print("OSAP lag and ranking OK")
