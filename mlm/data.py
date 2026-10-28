"""Panel storage, rolling windows, and a synthetic panel for testing the pipeline.

A panel is one parquet file per year plus meta.json. Each row is a (date, permno) pair in the
investable universe with:
    me      market cap at the close of `date` ($ millions)
    target  market-adjusted return earned by a position chosen with information up to `date`
    f_*     characteristics, as cross-sectional ranks in [-1, 1] stored as int8 (value * 127)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ID_COLS = ["date", "permno", "me", "target"]
Q = 127  # int8 scale for ranked features


def rank_quantize(df, cols, by="date", chunk=40):
    """Cross-sectional rank of each column mapped to [-1, 1]; missing values become 0 (the median).

    Columns are processed `chunk` at a time to bound the memory of the float64 temporaries.
    """
    g = df.groupby(by)
    parts = []
    for s in range(0, len(cols), chunk):
        c = cols[s:s + chunk]
        r = g[c].rank(method="average")
        n = g[c].transform("count")
        x = 2 * (r - 1) / (n - 1).where(n > 1) - 1
        parts.append((x.fillna(0.0) * Q).round().astype(np.int8))
    return pd.concat(parts, axis=1)


def write_panel_year(df, panel_dir, year, features):
    panel_dir = Path(panel_dir)
    panel_dir.mkdir(parents=True, exist_ok=True)
    df[ID_COLS + features].to_parquet(panel_dir / f"{year}.parquet", index=False)


def write_meta(panel_dir, features, **info):
    (Path(panel_dir) / "meta.json").write_text(json.dumps({"features": features, **info}, indent=1))


def panel_features(panel_dir):
    return json.loads((Path(panel_dir) / "meta.json").read_text())["features"]


def load_panel(panel_dir, start, end, features=None):
    panel_dir = Path(panel_dir)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    features = features or panel_features(panel_dir)
    parts = [pd.read_parquet(f, columns=ID_COLS + features)
             for y in range(start.year, end.year + 1)
             if (f := panel_dir / f"{y}.parquet").exists()]
    df = pd.concat(parts, ignore_index=True)
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    return df.sort_values(["date", "permno"]).reset_index(drop=True)


def rolling_windows(first_test, last_test, train_months=39, val_months=9, test_months=3):
    """Windows advancing one quarter at a time: 39 months train, 9 months validation, next quarter test (Section 4.1)."""
    out = []
    t0 = pd.Timestamp(first_test)
    while t0 <= pd.Timestamp(last_test):
        out.append(dict(
            name=f"{t0.year}Q{(t0.month - 1) // 3 + 1}",
            train_start=t0 - pd.DateOffset(months=train_months + val_months),
            val_start=t0 - pd.DateOffset(months=val_months),
            test_start=t0,
            test_end=t0 + pd.DateOffset(months=test_months) - pd.Timedelta(days=1),
        ))
        t0 = t0 + pd.DateOffset(months=test_months)
    return out


def split_window(df, w, embargo=2):
    """Train/validation/test rows for one window.

    The last `embargo` trading days before each boundary are dropped: their labels are returns
    realized after the boundary, so keeping them would leak future data into training.
    """
    d = df["date"]
    tr = df[(d >= w["train_start"]) & (d < w["val_start"])]
    va = df[(d >= w["val_start"]) & (d < w["test_start"])]
    te = df[(d >= w["test_start"]) & (d <= w["test_end"])]
    if embargo:
        tr = tr[tr["date"] < np.sort(tr["date"].unique())[-embargo]]
        va = va[va["date"] < np.sort(va["date"].unique())[-embargo]]
    return tr, va, te


def make_synthetic_panel(panel_dir, start="2010-01-01", end="2016-12-31", n_stocks=600, seed=0):
    """Simulated panel in the same format as the CRSP build, for testing the pipeline end to end.

    Expected returns are nonlinear and differ by firm size, echoing the four-asset example in
    Section 3.1: large caps load mainly on f_value, small caps on f_growth. f_rev is a fast-moving
    (high-turnover) signal, and f_noise* carry no information.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    T, N = len(dates), n_stocks
    phi = np.array([0.99, 0.99, 0.98, 0.98, 0.6, 0.95, 0.95, 0.9], dtype=np.float32)
    names = ["f_value", "f_growth", "f_c2", "f_c3", "f_rev", "f_c5", "f_noise6", "f_noise7"]
    z = np.empty((T, N, len(phi)), np.float32)
    z[0] = rng.standard_normal((N, len(phi)))
    lme = np.empty((T, N), np.float32)
    lme[0] = 6 + 2 * rng.standard_normal(N)
    s_phi, s_me = np.sqrt(1 - phi ** 2), 2 * np.sqrt(1 - 0.999 ** 2)
    for t in range(1, T):
        z[t] = phi * z[t - 1] + s_phi * rng.standard_normal((N, len(phi)))
        lme[t] = 6 + 0.999 * (lme[t - 1] - 6) + s_me * rng.standard_normal(N)

    large = lme > np.median(lme, axis=1, keepdims=True)
    f = [z[..., k] for k in range(len(phi))]
    mu = (5e-4 * np.where(large, np.tanh(2 * f[0]) + 0.3 * f[1], 0.3 * f[0] + np.tanh(2 * f[1]))
          + 4e-4 * f[2] * f[3] - 5e-4 * f[4] + 2e-4 * f[5])
    noise = np.where(large, 0.015, 0.03) * rng.standard_t(4, (T, N)) / np.sqrt(2)

    df = pd.DataFrame({
        "date": np.repeat(dates.values, N),
        "permno": np.tile(np.arange(10001, 10001 + N, dtype=np.int32), T),
        "me": np.exp(lme).ravel(),
        "target": (mu + noise).astype(np.float32).ravel(),
        **{n: f[k].ravel() for k, n in enumerate(names)},
        "f_size": lme.ravel(),
    })
    df = df[rng.random(len(df)) > 0.03].reset_index(drop=True)  # stocks occasionally missing from the universe
    features = names + ["f_size"]
    for c in features:
        df.loc[rng.random(len(df)) < 0.05, c] = np.nan  # missing characteristics
    df[features] = rank_quantize(df, features)
    for y, g in df.groupby(df["date"].dt.year):
        write_panel_year(g, panel_dir, y, features)
    write_meta(panel_dir, features, source="synthetic", seed=seed)
    return features
