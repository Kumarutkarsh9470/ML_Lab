"""US version of the panel: CRSP daily data from WRDS, daily characteristics, optional OSAP signals.

The paper's 415 daily China A-share characteristics are not public. This builds:
  * 21 characteristics computed daily from CRSP prices and volumes (size, reversal, momentum,
    volatility, beta, liquidity, ...), using data up to the close of each date;
  * optionally, the ~200 monthly Open Source Asset Pricing signals (Chen and Zimmermann),
    each applied to every day of the following month;
  * 10 SIC-division dummies in place of the 31 Shenwan industries.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ID_COLS, Q, rank_quantize, write_meta, write_panel_year

DSF_SQL = """
SELECT a.permno, a.date, a.ret, a.prc, a.vol, a.shrout, a.cfacpr, a.askhi, a.bidlo,
       b.shrcd, b.exchcd, b.siccd
FROM crsp.dsf AS a
JOIN crsp.dsenames AS b
  ON a.permno = b.permno
 AND a.date BETWEEN b.namedt AND COALESCE(b.nameendt, '9999-12-31')
WHERE a.date BETWEEN '{y}-01-01' AND '{y}-12-31'
  AND b.shrcd IN (10, 11)
  AND b.exchcd IN (1, 2, 3)
"""

NUM_COLS = ["ret", "prc", "vol", "shrout", "cfacpr", "askhi", "bidlo"]
SIC_EDGES = [10, 15, 20, 40, 50, 52, 60, 70, 90]  # SIC division boundaries on the 2-digit major group
SIC_NAMES = ["agri", "mining", "constr", "manuf", "transp_util", "wholesale", "retail", "finance",
             "services", "public"]


def download(db, raw_dir, start_year, end_year):
    """Download crsp.dsf (common stocks on NYSE/AMEX/Nasdaq), crsp.dsi and crsp.dsedelist. Existing files are kept."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for y in range(start_year, end_year + 1):
        f = raw_dir / f"dsf_{y}.parquet"
        if not f.exists():
            df = db.raw_sql(DSF_SQL.format(y=y), date_cols=["date"])
            df.to_parquet(f, index=False)
            print(f"dsf {y}: {len(df):,} rows")
    lo, hi = f"{start_year}-01-01", f"{end_year}-12-31"
    if not (raw_dir / "dsi.parquet").exists():
        db.raw_sql(f"SELECT date, vwretd FROM crsp.dsi WHERE date BETWEEN '{lo}' AND '{hi}'",
                   date_cols=["date"]).to_parquet(raw_dir / "dsi.parquet", index=False)
    if not (raw_dir / "dsedelist.parquet").exists():
        db.raw_sql(f"SELECT permno, dlstdt, dlret FROM crsp.dsedelist WHERE dlstdt BETWEEN '{lo}' AND '{hi}'",
                   date_cols=["dlstdt"]).to_parquet(raw_dir / "dsedelist.parquet", index=False)


def load_raw(raw_dir):
    raw_dir = Path(raw_dir)
    dsf = pd.concat([pd.read_parquet(f) for f in sorted(raw_dir.glob("dsf_*.parquet"))], ignore_index=True)
    dsf["permno"] = dsf["permno"].astype(np.int32)
    for c in NUM_COLS:
        dsf[c] = pd.to_numeric(dsf[c], errors="coerce").astype(np.float32)
    dsf["siccd"] = pd.to_numeric(dsf["siccd"], errors="coerce")
    dsf = dsf.drop_duplicates(["date", "permno"])  # overlapping dsenames ranges
    mkt = pd.read_parquet(raw_dir / "dsi.parquet").set_index("date")["vwretd"].astype(np.float64).sort_index()
    dsf = _add_delisting_returns(dsf, pd.read_parquet(raw_dir / "dsedelist.parquet"), mkt.index)
    return dsf.sort_values(["date", "permno"]).reset_index(drop=True), mkt


def _add_delisting_returns(dsf, dl, calendar):
    """Book each delisting return on the trading day after the stock's last trade.

    The added row has no price, so it never enters the investable universe, but it does reach the
    target of the decisions made just before the delisting.
    """
    dl = dl.dropna(subset=["dlret"]).copy()
    dl["permno"] = dl["permno"].astype(np.int32)
    last = dsf.groupby("permno")["date"].max().rename("last").reset_index()
    dl = dl.merge(last, on="permno")
    pos = calendar.searchsorted(dl["last"], side="right")
    ok = pos < len(calendar)
    rows = pd.DataFrame({"permno": dl["permno"].to_numpy()[ok],
                         "date": calendar[pos[ok]],
                         "ret": pd.to_numeric(dl["dlret"], errors="coerce").to_numpy(np.float32)[ok]})
    return pd.concat([dsf, rows], ignore_index=True)


def _beta_ivol(R, M, window, minp):
    """Rolling CAPM beta and residual volatility, using only the days each stock has a return."""
    Mm = pd.DataFrame(np.where(R.notna(), M.to_numpy()[:, None], np.nan), index=R.index, columns=R.columns)
    roll = lambda X: X.rolling(window, min_periods=minp).mean()
    mr, mm = roll(R), roll(Mm)
    cov = roll(R * Mm) - mr * mm
    beta = cov / (roll(Mm * Mm) - mm * mm)
    ivol = np.sqrt((roll(R * R) - mr * mr - beta * cov).clip(lower=0))
    return beta, ivol


def build_features(dsf, mkt, panel_start, panel_end, min_price=1.0, exec_lag=1):
    """Long panel of investable (date, permno) rows with me, target and unranked characteristics.

    target is the return over day t+1+exec_lag minus the CRSP value-weighted market return. With
    exec_lag=1, a decision made at the close of day t is traded at the close of t+1 and earns the
    t+2 return, the close-to-close analogue of the paper's VWAP(t+1) to VWAP(t+2) execution.
    """
    wide = lambda c: dsf.pivot(index="date", columns="permno", values=c)
    R, P, V, S = wide("ret"), wide("prc").abs(), wide("vol"), wide("shrout")
    M = mkt.reindex(R.index)

    rows = dsf[(dsf["date"] >= panel_start) & (dsf["date"] <= panel_end)]
    ii, jj = R.index.get_indexer(rows["date"]), R.columns.get_indexer(rows["permno"])
    take = lambda X: np.asarray(X, dtype=np.float32)[ii, jj]

    me = P * S / 1e3  # $ millions (shrout is in thousands)
    target = R.sub(M, axis=0).shift(-(1 + exec_lag))
    base = dict(prc=take(P), ret=take(R), me=take(me), target=take(target))
    keep = ((base["prc"] >= min_price) & np.isfinite(base["ret"]) & (base["me"] > 0)
            & np.isfinite(base["target"]))
    ii, jj, rows = ii[keep], jj[keep], rows[keep]
    out = pd.DataFrame({"date": rows["date"].to_numpy(), "permno": rows["permno"].to_numpy(),
                        "me": base["me"][keep], "target": base["target"][keep]})

    lr = np.log1p(R.clip(lower=-0.9999))
    cum = lambda w, minp, lag=0: np.expm1(lr.rolling(w, min_periods=minp).sum().shift(lag))
    turn = V / (S * 1e3)
    dv = P * V
    padj = P / wide("cfacpr")
    specs = {
        "size": lambda: np.log(me),
        "lprice": lambda: np.log(P),
        "r1": lambda: R,
        "r5": lambda: cum(5, 4),
        "r21": lambda: cum(21, 15),
        "mom6": lambda: cum(105, 80, 21),        # months t-6 to t-1
        "mom12": lambda: cum(231, 180, 21),      # months t-12 to t-1
        "mom36_12": lambda: cum(504, 380, 252),  # months t-36 to t-12
        "vol21": lambda: R.rolling(21, min_periods=15).std(),
        "vol63": lambda: R.rolling(63, min_periods=45).std(),
        "maxret21": lambda: R.rolling(21, min_periods=15).max(),
        "skew63": lambda: R.rolling(63, min_periods=45).skew(),
        "beta252": lambda: _beta_ivol(R, M, 252, 120)[0],
        "ivol63": lambda: _beta_ivol(R, M, 63, 45)[1],
        "turn21": lambda: turn.rolling(21, min_periods=15).mean(),
        "std_turn21": lambda: turn.rolling(21, min_periods=15).std(),
        "dvol21": lambda: np.log(dv.rolling(21, min_periods=15).mean()),
        "illiq21": lambda: (R.abs() / dv.where(dv > 0)).rolling(21, min_periods=15).mean(),
        "zero21": lambda: (V == 0).astype(np.float32).where(V.notna()).rolling(21, min_periods=15).sum(),
        "hl21": lambda: ((wide("askhi") - wide("bidlo")) / P).rolling(21, min_periods=15).mean(),
        "high52": lambda: padj / padj.rolling(252, min_periods=180).max(),
    }
    for name, fn in specs.items():
        x = take(fn())
        x[~np.isfinite(x)] = np.nan
        out[f"f_{name}"] = x
        print(f"  {name}: {np.isfinite(x).mean():.1%} non-missing")

    div = np.digitize(rows["siccd"].to_numpy() // 100, SIC_EDGES)
    div[~np.isfinite(rows["siccd"].to_numpy()) | (rows["siccd"].to_numpy() <= 0)] = -1
    dummies = [f"d_{n}" for n in SIC_NAMES]
    for k, c in enumerate(dummies):
        out[c] = (div == k).astype(np.int8)
    return out, [f"f_{n}" for n in specs], dummies


def load_osap(raw_dir, start_ym=200912):
    """Monthly OSAP signals (permno, yyyymm, signals), downloaded once and cached. Needs `openassetpricing`."""
    f = Path(raw_dir) / "osap_signals.parquet"
    if not f.exists():
        import openassetpricing as oap
        import polars as pl
        sig = oap.OpenAP().dl_all_signals("polars")
        missing = {"permno", "yyyymm"} - set(sig.columns)
        if missing:
            raise ValueError(f"OSAP download lacks columns {missing}; first columns: {sig.columns[:10]}")
        sig.filter(pl.col("yyyymm") >= start_ym).write_parquet(f)
    sig = pd.read_parquet(f)
    sig["permno"] = sig["permno"].astype(np.int32)
    sig["yyyymm"] = sig["yyyymm"].astype(np.int32)
    cols = [c for c in sig.columns if c not in ("permno", "yyyymm")]
    sig[cols] = sig[cols].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    return sig


def _merge_osap(g, osap):
    """Signals dated month m (known at its end) are used on every day of month m+1."""
    prev = g["date"].dt.to_period("M") - 1
    ym = (prev.dt.year * 100 + prev.dt.month).astype(np.int32).to_numpy()
    sig = osap[osap["yyyymm"].isin(np.unique(ym))]
    sig = sig.rename(columns={c: f"o_{c}" for c in sig.columns if c not in ("permno", "yyyymm")})
    return g.assign(yyyymm=ym).merge(sig, on=["permno", "yyyymm"], how="left").drop(columns="yyyymm")


def write_ranked_panel(panel, panel_dir, numeric, dummies, osap=None, **info):
    """Rank-transform characteristics within each day and write one parquet file per year."""
    if osap is not None:
        numeric = numeric + [f"o_{c}" for c in osap.columns if c not in ("permno", "yyyymm")]
    features = numeric + dummies
    for y, g in panel.groupby(panel["date"].dt.year):
        if osap is not None:
            g = _merge_osap(g, osap)
        g = g.reset_index(drop=True)
        ranked = rank_quantize(g, numeric)
        out = pd.concat([g[ID_COLS], ranked, (g[dummies] * Q).astype(np.int8)], axis=1)
        write_panel_year(out, panel_dir, y, features)
        print(f"{y}: {len(out):,} rows, {out['date'].nunique()} days, {len(out) / out['date'].nunique():,.0f} stocks/day")
    write_meta(panel_dir, features, source="crsp", **info)
    return features
