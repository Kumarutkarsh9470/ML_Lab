"""Two-stage baseline, stage two: turn predictions into portfolios (Sections 4.2 and 4.3)."""
import numpy as np
import pandas as pd

GROUPS = ["High(H)"] + [str(k) for k in range(2, 10)] + ["Low(L)"]


def decile_returns(pred, weight="ew", n_groups=10):
    """Daily returns of portfolios sorted into deciles on the prediction, highest (H) first.

    H-L is (H - L) / 2, a long-short portfolio with unit gross exposure. That is the convention
    Table 2 implies: for the linear MSE model, (34.2 + 59.5) / 2 = 46.9.
    """
    df = pred[["date", "pred", "target", "me"]].copy()
    pct = df.groupby("date")["pred"].rank(method="first", pct=True)
    df["g"] = n_groups + 1 - np.ceil(pct * n_groups).clip(1, n_groups).astype(int)  # 1 = highest
    if weight == "ew":
        out = df.groupby(["date", "g"])["target"].mean().unstack()
    else:
        df["wr"] = df["target"] * df["me"]
        s = df.groupby(["date", "g"])[["wr", "me"]].sum()
        out = (s["wr"] / s["me"]).unstack()
    out.columns = GROUPS
    out["H-L"] = (out["High(H)"] - out["Low(L)"]) / 2   # the paper's convention (unit gross exposure)
    out["H-L raw"] = out["High(H)"] - out["Low(L)"]     # long H, short L, $1 each (PAPER_ISSUES.md A7)
    return out


def solve_tc_lp(rhat, p, u, gamma):
    """Exact solution of the long-only top-decile problem with a turnover penalty (eq. 19):

        max_w  rhat'w - gamma/2 * ||w - p||_1   s.t.  sum(w) = 1,  0 <= w <= u.

    The objective is separable and concave piecewise linear in each w_i, so with a single budget
    constraint the problem is a fractional knapsack: keeping held weight (up to p_i) is worth
    rhat_i + gamma/2, and buying more (up to u) is worth rhat_i - gamma/2. Filling the most valuable
    pieces first is optimal. With gamma = 0 this gives 1/(10%·N) to the top 10% of stocks.
    """
    n = len(rhat)
    hold = np.minimum(p, u)
    lens = np.concatenate([hold, u - hold])
    slopes = np.concatenate([rhat + gamma / 2, rhat - gamma / 2])
    order = np.argsort(-slopes, kind="stable")  # for ties, an asset's "keep" piece comes before its "buy" piece
    cum = np.cumsum(lens[order])
    k = int(np.searchsorted(cum, 1.0 - 1e-12))
    take = np.zeros(2 * n)
    take[order[:k]] = lens[order[:k]]
    take[order[k]] = 1.0 - (cum[k - 1] if k else 0.0)
    return take[:n] + take[n:]


def _align(perm, prev):
    """Yesterday's weights on today's universe, plus the weight in stocks that left it (sold at a cost)."""
    pp, pw = prev
    _, i, j = np.intersect1d(perm, pp, assume_unique=True, return_indices=True)
    p = np.zeros(len(perm))
    p[i] = pw[j]
    return p, pw.sum() - pw[j].sum()


def _turnover(w, p, dropped):
    """One-sided turnover, 0.5 * ||w_t - w_{t-1}||_1. The cost gamma/2 * ||.||_1 in eq. (19) equals gamma * this."""
    return 0.5 * (np.abs(w - p).sum() + dropped)


def top_portfolio(pred, gamma=0.0, optimize=False, top_frac=0.1, kappa=None):
    """Daily long-only top-decile portfolio with transaction costs at rate gamma (a fraction, e.g. 0.002 = 20 bps).

    optimize=False is MSE-Top: hold the raw top decile and pay the costs.
    optimize=True is MSE-Opt: solve eq. (19) each day, given yesterday's weights.
    kappa (a scalar, or a Series indexed by date) scales the cost rate used inside the optimization only,
    for the tuned MSE-Opt benchmark; realized costs are always charged at gamma.
    Columns: gross and net returns, portfolio turnover (TO-P) and raw-signal turnover (TO-S).
    """
    df = pred.sort_values(["date", "permno"])
    dates, start = np.unique(df["date"].to_numpy(), return_index=True)
    stop = np.append(start[1:], len(df))
    P = df["permno"].to_numpy()
    R = df["pred"].to_numpy(np.float64)
    Y = df["target"].to_numpy(np.float64)
    empty = (P[:0], np.zeros(0))
    prev_raw, prev = empty, empty
    rows = []
    for d, a, b in zip(dates, start, stop):
        perm, rhat, y = P[a:b], R[a:b], Y[a:b]
        u = 1.0 / (top_frac * len(perm))
        w_raw = solve_tc_lp(rhat, np.zeros(len(perm)), u, 0.0)
        to_s = _turnover(w_raw, *_align(perm, prev_raw))
        p, dropped = _align(perm, prev)
        k = 1.0 if kappa is None else (kappa.get(d, 1.0) if isinstance(kappa, (pd.Series, dict)) else kappa)
        w = solve_tc_lp(rhat, p, u, gamma * k) if optimize else w_raw
        to_p = _turnover(w, p, dropped)
        gross = float(w @ y)
        rows.append((d, gross, gross - gamma * to_p, to_p, to_s))
        prev_raw, prev = (perm, w_raw), (perm, w)
    return pd.DataFrame(rows, columns=["date", "gross", "net", "to_p", "to_s"]).set_index("date")


KAPPAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def tuned_kappa(test_pred, val_pred, windows, gamma, kappas=KAPPAS):
    """Tuned MSE-Opt (PAPER_ISSUES.md D2): per window, the cost multiplier kappa with the best validation net return.

    MSE forecasts are shrunk toward zero, so feeding them into eq. (19) as they are under-reacts or over-reacts
    to costs; a practitioner would tune the trade-off. Only validation data is used for the choice. Returns
    a Series of kappa by test date and the per-window choices.
    """
    chosen, by_date = {}, {}
    for w in windows:
        v = val_pred[val_pred["window"] == w["name"]]  # this window's own validation predictions
        t = test_pred[(test_pred.date >= w["test_start"]) & (test_pred.date <= w["test_end"])]
        if v.empty or t.empty:
            continue
        nets = {k: top_portfolio(v, gamma, optimize=True, kappa=k)["net"].mean() for k in kappas}
        chosen[w["name"]] = max(nets, key=nets.get)
        by_date.update({d: chosen[w["name"]] for d in t.date.unique()})
    return pd.Series(by_date), pd.Series(chosen)
