"""Performance statistics, following the conventions the paper's tables imply."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

ANN = 252  # trading days per year; set from the panel's meta.json (about 242 for A-shares, as in the paper)


def oos_r2(pred, target):
    """Out-of-sample R^2 against a zero forecast, as in Gu, Kelly and Xiu (2020)."""
    pred, target = np.asarray(pred, np.float64), np.asarray(target, np.float64)
    return 1 - np.sum((target - pred) ** 2) / np.sum(target ** 2)


def perf(r, ann=None):
    """Annualized mean, volatility (both in %) and Sharpe ratio of daily excess returns."""
    ann = ann or ANN
    r = r.dropna()
    avg, sd = r.mean() * ann * 100, r.std() * np.sqrt(ann) * 100
    return pd.Series({"Avg %": avg, "SD %": sd, "SR": avg / sd})


def drawdown(r):
    """Largest one-day loss and maximum drawdown of the cumulative sum of daily returns.

    The paper uses cumulative sums (Table 3 reports drawdowns above 100%), and its Length column
    counts calendar days from peak to trough, inclusive (2015-06-19 to 2015-07-06 gives 18).
    """
    r = r.dropna()
    cum = r.cumsum()
    dd = cum.cummax() - cum
    end = dd.idxmax()
    begin = cum.loc[:end].idxmax()
    return pd.Series({"MDD-1D %": -r.min() * 100, "MDD %": dd.max() * 100,
                      "MDD Begin": begin.date(), "MDD End": end.date(), "Length": (end - begin).days + 1})


def nw_tstat(x):
    """Newey-West t-statistic of the mean, with floor(T^(1/4)) lags (Section 4.2)."""
    x = pd.Series(x).dropna().to_numpy(np.float64)
    res = sm.OLS(x, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": int(len(x) ** 0.25)})
    return float(res.tvalues[0])


def nw_test(x):
    """Newey-West t-statistic with one-sided (mean > 0) and two-sided p-values.

    The paper marks t = 1.74 as significant at 5%, which holds only one-sided (PAPER_ISSUES.md C5); we
    report both and call a difference significant only when the two-sided test agrees.
    """
    from scipy.stats import norm
    t = nw_tstat(x)
    return pd.Series({"t": t, "p one-sided": 1 - norm.cdf(t), "p two-sided": 2 * (1 - norm.cdf(abs(t)))})
