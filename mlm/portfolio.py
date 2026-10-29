"""Two-stage baseline, stage two: turn predictions into portfolios (Sections 4.2 and 4.3)."""
import numpy as np
import pandas as pd

GROUPS = ["High(H)"] + [str(k) for k in range(2, 10)] + ["Low(L)"]


def decile_returns(pred, weight="ew", n_groups=10):
    """Daily returns of portfolios sorted into deciles on the prediction, highest (H) first."""
    df = pred[["date", "pred", "target", "me"]].copy()
    pct = df.groupby("date")["pred"].rank(method="first", pct=True)
    df["g"] = n_groups + 1 - np.ceil(pct * n_groups).clip(1, n_groups).astype(int)
    if weight == "ew":
        out = df.groupby(["date", "g"])["target"].mean().unstack()
    else:
        df["wr"] = df["target"] * df["me"]
        s = df.groupby(["date", "g"])[["wr", "me"]].sum()
        out = (s["wr"] / s["me"]).unstack()
    out.columns = GROUPS
    out["H-L"] = (out["High(H)"] - out["Low(L)"]) / 2
    out["H-L raw"] = out["High(H)"] - out["Low(L)"]
    return out


def solve_tc_lp(rhat, p, u, gamma):
    """Exact solution of the long-only top-decile problem with a turnover penalty (eq. 19)."""
    n = len(rhat)
    hold = np.minimum(p, u)
    lens = np.concatenate([hold, u - hold])
    slopes = np.concatenate([rhat + gamma / 2, rhat - gamma / 2])
    order = np.argsort(-slopes, kind="stable")
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
    """One-sided turnover, 0.5 * ||w_t - w_{t-1}||_1."""
    return 0.5 * (np.abs(w - p).sum() + dropped)


def top_portfolio(pred, gamma=0.0, optimize=False, top_frac=0.1, kappa=None):
    """Daily long-only top-decile portfolio with transaction costs at rate gamma (a fraction, e.g. 0.002 = 20 bps)."""
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
    """Tuned MSE-Opt: per window, the cost multiplier kappa with the best validation net return."""
    chosen, by_date = {}, {}
    for w in windows:
        v = val_pred[val_pred["window"] == w["name"]]
        t = test_pred[(test_pred.date >= w["test_start"]) & (test_pred.date <= w["test_end"])]
        if v.empty or t.empty:
            continue
        nets = {k: top_portfolio(v, gamma, optimize=True, kappa=k)["net"].mean() for k in kappas}
        chosen[w["name"]] = max(nets, key=nets.get)
        by_date.update({d: chosen[w["name"]] for d in t.date.unique()})
    return pd.Series(by_date), pd.Series(chosen)
