"""China A-share panel built from two free Qlib-format dumps, the paper's own market.

joshuaxql/qlib_data (Hugging Face): raw daily prices, VWAP, adjustment factor, market caps, price limits,
ST intervals, Shenwan level-1 industry membership and point-in-time financial indicators (163 Tushare
fina_indicator fields with announcement dates), 2010 onward, delisted stocks included.
chenditc/investment_data (GitHub): adjusted prices back to 2000, used for the 2007-2009 history that
long-horizon characteristics need, and the CSI 500 index (sh000905).

Conventions (checked on 600519.SH): joshuaxql prices and VWAP are raw CNY, `factor` is the cumulative
adjustment factor (adjusted price = raw * factor), volume is in lots of 100 shares, total_mv / circ_mv are
in 10,000 CNY. chenditc prices are already adjusted and its amount is in 1,000 CNY.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ID_COLS, Q, rank_quantize, write_meta, write_panel_year

PIT_REC = np.dtype([("field", "<u4"), ("date", "<u4"), ("period", "<u4"), ("value", "<f8"),
                    ("next", "<u4"), ("extra", "<u4")])
PIT_HEADER = 72  # "QLPITD2\0", version, record size, 16-byte id, record count (u8 at byte 32), 32-byte hash
J_FIELDS = ["close", "high", "low", "volume", "vwap", "factor", "total_mv", "circ_mv", "up_limit", "down_limit"]
C_FIELDS = ["close", "high", "low", "amount", "volume"]
SPLICE = pd.Timestamp("2010-01-04")  # first joshuaxql day; earlier history comes from chenditc


def calendar(root):
    return pd.DatetimeIndex(pd.to_datetime(pd.read_csv(Path(root) / "calendars" / "day.txt", header=None)[0]))


def chenditc_code(code):
    """'600519.SH' -> 'sh600519'."""
    return f"{code[-2:].lower()}{code[:6]}"


def load_wide(root, codes, fields, start, end, dir_names=None):
    """dates x codes float32 frames, one per field. Each .day.bin is float32 values preceded by the start index."""
    root = Path(root)
    cal = calendar(root)
    lo, hi = cal.searchsorted(pd.Timestamp(start)), cal.searchsorted(pd.Timestamp(end), side="right")
    out = {f: np.full((hi - lo, len(codes)), np.nan, np.float32) for f in fields}
    for j, name in enumerate(dir_names or codes):
        d = root / "features" / name
        for f in fields:
            p = d / f"{f}.day.bin"
            if not p.exists():
                continue
            a = np.fromfile(p, dtype="<f4")
            s, v = int(a[0]), a[1:]
            i0, i1 = max(s, lo), min(s + len(v), hi)
            if i1 > i0:
                out[f][i0 - lo:i1 - lo, j] = v[i0 - s:i1 - s]
    return {f: pd.DataFrame(m, index=cal[lo:hi], columns=codes) for f, m in out.items()}


def interval_mask(path, codes, dates):
    """Boolean dates x codes mask from a Qlib instruments/industry file of (code, start, end) lines."""
    m = np.zeros((len(dates), len(codes)), bool)
    col = {c: j for j, c in enumerate(codes)}
    for code, s, e in pd.read_csv(path, sep="\t", header=None).itertuples(index=False):
        j = col.get(code)
        if j is not None:
            m[dates.searchsorted(pd.Timestamp(s)):dates.searchsorted(pd.Timestamp(e), side="right"), j] = True
    return m


def industry_codes(root, codes, dates):
    """Shenwan level-1 industry index (0..30) per date and stock, -1 if unknown."""
    files = sorted((Path(root) / "industry").glob("*.txt"))
    ind = np.full((len(dates), len(codes)), -1, np.int8)
    for k, f in enumerate(files):
        ind[interval_mask(f, codes, dates)] = k
    return ind, [f.stem for f in files]


def pit_events(root, codes, field_names):
    """For each stock and announcement date, the latest reported value of every financial indicator.

    A record replaces the current value only if its report period is the latest seen so far, so a late
    filing for an old quarter does not overwrite newer data.
    """
    frames = []
    for code in codes:
        p = Path(root) / "financial" / code / "pit.data"
        if not p.exists():
            continue
        raw = p.read_bytes()
        n = int(np.frombuffer(raw, "<u8", 1, 32)[0])
        a = np.frombuffer(raw, PIT_REC, n, PIT_HEADER)
        df = pd.DataFrame({"field": a["field"], "date": a["date"], "period": a["period"], "value": a["value"]})
        df = df.sort_values(["date", "period"], kind="stable")
        df = df[df["period"] >= df.groupby("field")["period"].cummax()]
        w = df.pivot_table(index="date", columns="field", values="value", aggfunc="last").ffill()
        w.insert(0, "code", code)
        frames.append(w.reset_index())
    ev = pd.concat(frames, ignore_index=True)
    ev["date"] = pd.to_datetime(ev["date"].astype(str), format="%Y%m%d")
    ev = ev.rename(columns={k: f"fin_{v}" for k, v in field_names.items()})
    fin = [f"fin_{v}" for v in field_names.values()]
    for c in fin:
        if c not in ev:
            ev[c] = np.nan
    ev[fin] = ev[fin].astype(np.float32)
    return ev[["date", "code"] + fin].sort_values("date").reset_index(drop=True), fin


def _splice(early, late, dates, first_late):
    """`early` before `first_late`, `late` from it on; both are reindexed to `dates`."""
    use_late = (dates >= first_late)[:, None]
    e, l = early.reindex(dates), late.reindex(dates)
    return pd.DataFrame(np.where(use_late, l.to_numpy(), e.to_numpy()), index=dates, columns=late.columns)


def _rolling_moments(R, M, window, minp):
    """Rolling CAPM beta, residual volatility, downside beta and coskewness against the market."""
    have = R.notna()
    Mm = pd.DataFrame(np.where(have, M.to_numpy()[:, None], np.nan), index=R.index, columns=R.columns)
    roll = lambda X: X.rolling(window, min_periods=minp).mean()
    er, em = roll(R), roll(Mm)
    erm, emm = roll(R * Mm), roll(Mm * Mm)
    cov, var_m, var_r = erm - er * em, emm - em * em, roll(R * R) - er * er
    beta = cov / var_m
    ivol = np.sqrt((var_r - beta * cov).clip(lower=0))
    # E[(r - er)(m - em)^2] / (sd_r * var_m)
    erm2 = roll(R * Mm * Mm)
    cosk = (erm2 - 2 * em * erm + em * em * er - er * emm + 2 * er * em * em - er * em * em) / (np.sqrt(var_r) * var_m)
    down = Mm.where(Mm < 0)
    Rd = R.where(down.notna())
    rd = lambda X: X.rolling(window, min_periods=minp // 3).mean()
    dbeta = (rd(Rd * down) - rd(Rd) * rd(down)) / (rd(down * down) - rd(down) ** 2)
    return beta, ivol, cosk, dbeta


def build_panel(j_root, c_root, panel_dir, panel_start="2010-01-01", panel_end="2023-12-31",
                hist_start="2007-01-01", data_end="2024-01-31", codes=None, min_listing_days=60,
                drop_locked_next=True, verbose=True):
    """Long panel of investable stock-days with me, target, price/volume characteristics and industry dummies.

    Tradability (not in the paper; see PAPER_ISSUES.md): a stock-day is dropped if the stock has traded for
    fewer than `min_listing_days` days since listing (new A-share listings often rise by the daily limit for
    weeks), or, with drop_locked_next, if it is locked at its upper price limit for the whole of day t+1, when
    a buy order at VWAP cannot be filled. Without these rules a large part of the top decile's return comes
    from untradeable stocks. `codes` restricts the stock list (for quick tests).
    """
    log = print if verbose else (lambda *a, **k: None)
    j_root, c_root = Path(j_root), Path(c_root)
    codes = codes or sorted(p.name for p in (j_root / "features").iterdir() if p.is_dir())
    log(f"{len(codes)} stocks")

    J = load_wide(j_root, codes, J_FIELDS, SPLICE, data_end)
    # chenditc is loaded a month past the splice only to cross-check the two sources; it is used before SPLICE
    H = load_wide(c_root, codes, C_FIELDS, hist_start, SPLICE + pd.DateOffset(months=1), [chenditc_code(c) for c in codes])
    idx = load_wide(c_root, ["CSI500"], ["close"], hist_start, data_end, ["sh000905"])["close"]["CSI500"]
    dates = H["close"].index[H["close"].index < SPLICE].append(J["close"].index)
    log(f"{len(dates)} trading days, {dates[0].date()} to {dates[-1].date()}")
    re = lambda X: X.reindex(dates)

    # adjusted prices from each source; returns are spliced, so the two price scales never mix
    bar_j = re(J["close"]).notna()
    padj_j = re(J["close"] * J["factor"])
    ret_j = padj_j.ffill().pct_change(fill_method=None).where(bar_j)
    ret_h = re(H["close"]).ffill().pct_change(fill_method=None).where(re(H["close"]).notna())
    after_splice = dates[dates > SPLICE][0]  # the 2010-01-04 return needs the 2009 close, so it comes from chenditc
    R = _splice(ret_h, ret_j, dates, after_splice).astype(np.float32)
    check = dates[dates > SPLICE][:20]
    a, b = ret_h.reindex(check).to_numpy(), ret_j.reindex(check).to_numpy()
    ok = np.isfinite(a) & np.isfinite(b)
    log(f"source check on the first 20 days of 2010: {ok.sum():,} stock-days in both, return corr "
        f"{np.corrcoef(a[ok], b[ok])[0, 1]:.4f}, median |diff| {np.median(np.abs(a[ok] - b[ok])):.1e}")
    M = idx.reindex(dates).pct_change(fill_method=None).astype(np.float32)

    amount = _splice(re(H["amount"]) * 1000, re(J["vwap"]) * re(J["volume"]) * 100, dates, SPLICE)
    vol_any = _splice(re(H["volume"]), re(J["volume"]), dates, SPLICE)
    hl = _splice(re(H["high"]) / re(H["low"]), re(J["high"]) / re(J["low"]), dates, SPLICE) - 1
    close_raw, total_mv, circ_mv = re(J["close"]), re(J["total_mv"]), re(J["circ_mv"])
    turn = re(J["volume"]) * 100 / (circ_mv * 1e4 / close_raw)

    # target: VWAP(t+1) -> VWAP(t+2) return minus the CSI 500 close-to-close return over day t+2
    vadj = re(J["vwap"] * J["factor"])
    target = (vadj / vadj.shift(1) - 1).sub(M, axis=0).shift(-2)

    # universe on day t: traded (bar with positive volume), not ST, target available
    st = interval_mask(j_root / "instruments" / "st.txt", codes, dates)
    universe = bar_j.to_numpy() & (re(J["volume"]).to_numpy() > 0) & ~st & np.isfinite(target.to_numpy())
    universe &= ((dates >= pd.Timestamp(panel_start)) & (dates <= pd.Timestamp(panel_end)))[:, None]
    n0 = universe.sum()
    basic = pd.read_csv(j_root / "stock_basic.csv", usecols=["ts_code", "list_date"]).set_index("ts_code")
    list_pos = dates.searchsorted(pd.to_datetime(basic["list_date"].reindex(codes)).fillna(dates[0]).values)
    trading_age = np.arange(len(dates))[:, None] - list_pos[None, :]
    universe &= trading_age >= min_listing_days
    n1 = universe.sum()
    if drop_locked_next:
        up = re(J["up_limit"])
        locked = ((re(J["low"]) >= up - 0.005) & (re(J["high"]) <= up + 0.005)).to_numpy()
        locked_next = np.vstack([locked[1:], np.zeros((1, locked.shape[1]), bool)])
        universe &= ~locked_next
    log(f"tradability: {n0 - n1:,} stock-days dropped as listed < {min_listing_days} trading days, "
        f"{n1 - universe.sum():,} as locked at the upper limit on t+1")
    ii, jj = np.nonzero(universe)
    log(f"universe: {len(ii):,} stock-days, {len(ii) / len(np.unique(ii)):,.0f} stocks per day")
    take = lambda X: np.asarray(X, dtype=np.float32)[ii, jj]
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # empty rolling windows before a stock lists

    lr = np.log1p(R.clip(lower=-0.9999))
    cum = lambda w, minp, lag=0: np.expm1(lr.rolling(w, min_periods=minp).sum().shift(lag))
    rmean = lambda X, w, minp: X.rolling(w, min_periods=minp).mean()
    P = np.exp(lr.fillna(0).cumsum()).where(R.notna())  # return index, consistent across the splice
    up_hit = (close_raw >= re(J["up_limit"]) - 0.005).astype(np.float32).where(bar_j)
    dn_hit = (close_raw <= re(J["down_limit"]) + 0.005).astype(np.float32).where(bar_j)
    listed = R.notna().cumsum() > 0
    zero = ((vol_any == 0) | (R.isna() & listed)).astype(np.float32)
    list_date = pd.to_datetime(basic["list_date"].reindex(codes))
    age = pd.DataFrame((dates.values[:, None] - list_date.values[None, :]) / np.timedelta64(1, "D"),
                       index=dates, columns=codes)
    m252 = _rolling_moments(R, M, 252, 120)
    m63 = _rolling_moments(R, M, 63, 40)

    specs = {
        "size": lambda: np.log(total_mv), "size_float": lambda: np.log(circ_mv),
        "float_ratio": lambda: circ_mv / total_mv, "lprice": lambda: np.log(close_raw),
        "age": lambda: np.log1p(age.clip(lower=0)),
        "r1": lambda: R, "r5": lambda: cum(5, 4), "mom1m": lambda: cum(21, 15),
        "mom2m": lambda: cum(21, 15, 21), "mom3m": lambda: cum(63, 45),
        "mom6m": lambda: cum(105, 75, 21), "mom12m": lambda: cum(231, 160, 21),
        "intmom": lambda: cum(126, 90, 126), "lrreversal": lambda: cum(504, 350, 252),
        "maxret21": lambda: R.rolling(21, min_periods=15).max(), "maxret63": lambda: R.rolling(63, min_periods=45).max(),
        "maxret252": lambda: R.rolling(252, min_periods=160).max(), "minret21": lambda: R.rolling(21, min_periods=15).min(),
        "vol21": lambda: R.rolling(21, min_periods=15).std(), "vol63": lambda: R.rolling(63, min_periods=45).std(),
        "vol252": lambda: R.rolling(252, min_periods=160).std(),
        "skew21": lambda: R.rolling(21, min_periods=15).skew(), "skew63": lambda: R.rolling(63, min_periods=45).skew(),
        "beta252": lambda: m252[0], "beta252_sq": lambda: m252[0] ** 2, "ivol252": lambda: m252[1],
        "coskew252": lambda: m252[2], "downbeta252": lambda: m252[3], "beta63": lambda: m63[0], "ivol63": lambda: m63[1],
        "dvol21": lambda: np.log(rmean(amount, 21, 15)), "dvol252": lambda: np.log(rmean(amount, 252, 160)),
        "illiq21": lambda: rmean(R.abs() / amount.where(amount > 0), 21, 15),
        "illiq252": lambda: rmean(R.abs() / amount.where(amount > 0), 252, 160),
        "turn5": lambda: rmean(turn, 5, 4), "turn21": lambda: rmean(turn, 21, 15), "turn252": lambda: rmean(turn, 252, 160),
        "std_turn21": lambda: turn.rolling(21, min_periods=15).std(),
        "turn_trend": lambda: rmean(turn, 21, 15) / rmean(turn, 252, 160),
        "abturn": lambda: rmean(turn, 5, 4) / rmean(turn, 252, 160),
        "zero21": lambda: zero.where(listed).rolling(21, min_periods=15).sum(),
        "zero252": lambda: zero.where(listed).rolling(252, min_periods=160).sum(),
        "hl21": lambda: rmean(hl, 21, 15), "high52": lambda: P / P.rolling(252, min_periods=160).max(),
        "uphits21": lambda: up_hit.rolling(21, min_periods=15).sum(), "dnhits21": lambda: dn_hit.rolling(21, min_periods=15).sum(),
        "uphits252": lambda: up_hit.rolling(252, min_periods=160).sum(),
    }
    out = pd.DataFrame({"date": dates[ii], "code": np.asarray(codes)[jj],
                        "me": take(total_mv) / 100, "target": take(target), "close_raw": take(close_raw)})
    out["permno"] = out["code"].str[:6].astype(np.int32)
    for name, fn in specs.items():
        x = take(fn())
        x[~np.isfinite(x)] = np.nan
        out[f"f_{name}"] = x
        log(f"  {name}: {np.isfinite(x).mean():.1%} non-missing")
    ind, ind_names = industry_codes(j_root, codes, dates)
    ind = ind[ii, jj]
    dummies = [f"d_{n.split('.')[0]}" for n in ind_names]
    for k, c in enumerate(dummies):
        out[c] = (ind == k).astype(np.int8)
    log(f"industry known for {np.mean(ind >= 0):.1%} of rows")
    numeric = [f"f_{n}" for n in specs]
    return out, numeric, dummies


VALUATION = {"bm": "fin_bps", "ep": "fin_eps", "ep_q": "fin_q_eps", "cfp": "fin_cfps", "ocfp": "fin_ocfps",
             "sp": "fin_total_revenue_ps", "ebitp": "fin_ebit_ps", "fcfp": "fin_fcff_ps", "retp": "fin_retainedps",
             "undistp": "fin_undist_profit_ps"}


def save_unranked(out, numeric, dummies, unranked_dir):
    """Store build_panel's output by year, so ranking can run (and resume) one year at a time."""
    unranked_dir = Path(unranked_dir)
    unranked_dir.mkdir(parents=True, exist_ok=True)
    for y, g in out.groupby(out["date"].dt.year):
        g.to_parquet(unranked_dir / f"{y}.parquet", index=False)
    (unranked_dir / "meta.json").write_text(json.dumps({"numeric": numeric, "dummies": dummies}, indent=1))


def write_cn_panel(unranked_dir, j_root, panel_dir, field_names, overwrite=False, verbose=True):
    """Add point-in-time fundamentals and valuation ratios, rank within each day, write one file per year.

    Years already in panel_dir are skipped unless overwrite=True.
    """
    log = print if verbose else (lambda *a, **k: None)
    unranked_dir, panel_dir = Path(unranked_dir), Path(panel_dir)
    meta = json.loads((unranked_dir / "meta.json").read_text())
    numeric, dummies = meta["numeric"], meta["dummies"]
    files = sorted(unranked_dir.glob("*.parquet"))
    codes = sorted(set().union(*(pd.read_parquet(f, columns=["code"])["code"].unique() for f in files)))
    events, fin = pit_events(j_root, codes, field_names)
    log(f"fundamentals: {len(events):,} announcement events, {len(fin)} indicators")
    vals = [f"v_{k}" for k in VALUATION]
    features = numeric + fin + vals + dummies
    days_per_year = {}
    for f in files:
        y = int(f.stem)
        days_per_year[y] = pd.read_parquet(f, columns=["date"])["date"].nunique()
        if (panel_dir / f"{y}.parquet").exists() and not overwrite:
            log(f"{y}: already written, skipped")
            continue
        g = pd.read_parquet(f).sort_values("date")
        g = pd.merge_asof(g, events, on="date", by="code", allow_exact_matches=False)  # announced before day t
        for k, src in VALUATION.items():
            g[f"v_{k}"] = (g[src] / g["close_raw"]).astype(np.float32)
        g = g.sort_values(["date", "permno"]).reset_index(drop=True)
        ranked = rank_quantize(g, numeric + fin + vals)
        panel = pd.concat([g[ID_COLS], ranked, (g[dummies] * Q).astype(np.int8)], axis=1)
        write_panel_year(panel, panel_dir, y, features)
        cover = g[fin].notna().mean().mean()
        log(f"{y}: {len(panel):,} rows, {g['date'].nunique()} days, {len(panel) / g['date'].nunique():,.0f} stocks/day, "
            f"fundamentals coverage {cover:.0%}")
        del g, ranked, panel
    write_meta(panel_dir, features, source="cn", ann=float(np.mean(list(days_per_year.values()))),
               days_per_year=days_per_year)
    return features


def crosscheck_kaggle(j_root, kaggle_dir, n=300, seed=0, start="2011-01-01", end="2022-12-30"):
    """Compare daily VWAP returns with the Kaggle StockChina files (JoinQuant data, an independent source).

    Kaggle's `avg` is a forward-adjusted VWAP rounded to 0.01 CNY, so for stocks whose adjusted price is
    small the rounding alone causes noticeable differences; the median gap is the useful number.
    """
    codes = sorted(p.name for p in (Path(j_root) / "features").iterdir() if p.is_dir())
    pick = list(np.random.default_rng(seed).choice(codes, n, replace=False))
    W = load_wide(j_root, pick, ["vwap", "factor"], start, end)
    rj_all = (W["vwap"] * W["factor"]).pct_change(fill_method=None)
    pairs = []
    for c in pick:
        f = Path(kaggle_dir) / f"{c[:6]}.{'XSHG' if c.endswith('SH') else 'XSHE'}.csv"
        if not f.exists():
            continue
        k = pd.read_csv(f, index_col=0, parse_dates=True)
        k = k[(k["paused"] == 0) & (k["avg"] > 0)]
        rk = k["avg"].pct_change(fill_method=None).loc[start:end]
        both = pd.concat([rj_all[c].rename("j"), rk.rename("k")], axis=1, join="inner").dropna()
        pairs.append(both.assign(code=c))
    df = pd.concat(pairs)
    d = (df["j"] - df["k"]).abs()
    return pd.Series({"stocks compared": df["code"].nunique(), "stock-days": len(df),
                      "return corr": df["j"].corr(df["k"]), "median |diff|": d.median(),
                      "share |diff| < 0.1%": (d < 1e-3).mean(), "share |diff| < 1%": (d < 1e-2).mean()})


def field_names_from_manifest(manifest_path):
    m = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return {int(k): v["name"] for k, v in m["fields"].items()}
