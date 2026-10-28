"""Quick test of the China panel build on a subset of stocks, with hand checks."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlm import cn  # noqa: E402
from mlm.data import load_panel  # noqa: E402

RAW = Path(r"D:\ML_Markowitz_data\raw_cn")
J, C = RAW / "joshuaxql" / "cn_data", RAW / "chenditc" / "qlib_bin"
out_dir = Path(sys.argv[1])
codes = sorted(p.name for p in (J / "features").iterdir())
codes = sorted(set(codes[::37]) | {"600519.SH", "000001.SZ", "600087.SH", "000024.SZ"})
t = time.time()
panel, numeric, dummies = cn.build_panel(J, C, out_dir, codes=codes)
print(f"build: {time.time() - t:.0f}s, {panel.shape}")
assert panel.groupby(["date", "permno"]).size().max() == 1

# hand check: target for Moutai on 2014-06-20 spans the 2014-06-25 ex-date? use 2014-06-23: VWAP 24th -> 25th
cal = cn.calendar(J)
v = cn.load_wide(J, ["600519.SH"], ["vwap", "factor"], "2014-06-01", "2014-07-31")
idx = cn.load_wide(C, ["I"], ["close"], "2014-06-01", "2014-07-31", ["sh000905"])["close"]["I"]
vadj = v["vwap"]["600519.SH"] * v["factor"]["600519.SH"]
d0 = pd.Timestamp("2014-06-23")
d1, d2 = vadj.index[vadj.index.get_loc(d0) + 1], vadj.index[vadj.index.get_loc(d0) + 2]
expect = vadj[d2] / vadj[d1] - 1 - (idx[d2] / idx[d1] - 1)
got = panel[(panel.code == "600519.SH") & (panel.date == d0)]["target"].iloc[0]
print(f"Moutai target {d0.date()} (VWAP {d1.date()} -> {d2.date()}, across the 10-for-1 bonus issue): {got:.5f} vs hand {expect:.5f}")
assert abs(got - expect) < 1e-5
row = panel[(panel.code == "600519.SH") & (panel.date == "2010-03-01")].iloc[0]
print("Moutai 2010-03-01 mom12m (needs 2009 history):", row.f_mom12m, " beta252:", row.f_beta252, " size:", row.f_size)
assert np.isfinite(row.f_mom12m) and np.isfinite(row.f_beta252)
print("delisted 600087 last panel date:", panel[panel.code == "600087.SH"].date.max().date())
st = panel[(panel.code == "000004.SZ") & (panel.date.between("2010-01-04", "2011-06-07"))]
print("ST 000004.SZ rows during its ST interval (expect 0):", len(st))
assert len(st) == 0
print("industry dummies sum per row (should be <= 1):", panel[dummies].sum(axis=1).max())

names = cn.field_names_from_manifest(RAW / "joshuaxql" / "manifest.json")
t = time.time()
cn.save_unranked(panel, numeric, dummies, out_dir / "unranked")
feats = cn.write_cn_panel(out_dir / "unranked", J, out_dir, names)
print(f"write: {time.time() - t:.0f}s, {len(feats)} features")
p = load_panel(out_dir, "2014-04-24", "2014-04-28")
r = p[p.permno == 600519].set_index("date")
# Moutai's 2014Q1 report (EPS 3.56) was announced 2014-04-25: usable from the next trading day only
raw = panel[(panel.code == "600519.SH") & (panel.date.between("2014-04-24", "2014-04-29"))][["date", "close_raw"]]
ev, fin = cn.pit_events(J, ["600519.SH"], names)
asof = pd.merge_asof(raw.sort_values("date"), ev, on="date", by=None, allow_exact_matches=False)
print(asof[["date", "fin_eps", "fin_bps"]].to_string(index=False))
assert asof.set_index("date").loc["2014-04-25", "fin_eps"] == np.float32(14.58)
assert asof.set_index("date").loc["2014-04-28", "fin_eps"] == np.float32(3.56)
print("point-in-time alignment OK")
