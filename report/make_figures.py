"""Figures and summary numbers for the report and slides, computed from the saved out-of-sample predictions.

Re-run after new training finishes: every model with predictions in PRED is picked up automatically.
Writes report/figures/*.pdf, *.png and report/results_summary.json.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mlm import metrics  # noqa: E402
from mlm.linear import GRIDS  # noqa: E402
from mlm.metrics import nw_tstat, oos_r2, perf  # noqa: E402
from mlm.models import MODELS  # noqa: E402
from mlm.portfolio import decile_returns, top_portfolio  # noqa: E402
from mlm.train import load_predictions  # noqa: E402

PRED = Path("D:/ML_Markowitz_data/preds/cn")
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
metrics.ANN = json.loads(Path("D:/ML_Markowitz_data/panel_cn/meta.json").read_text())["ann"]
START, END = "2014-01-01", "2023-12-31"

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical slots 1-4
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})

SKIP = ("MSD-", "PPP-", "E2E-tc", "E2E-scratch")  # portfolio-level runs, evaluated in make_decision.py
names = [m for m in MODELS + [f"E2E-{m}" for m in MODELS] + list(GRIDS) if (PRED / m).exists()]
names += sorted(d.name for d in PRED.iterdir() if d.is_dir() and d.name not in names and not d.name.startswith(SKIP))
rows, cum, years = {}, {}, {}
PLOT_ONLY = "--plot-only" in sys.argv  # redraw from the cached numbers, without recomputing
if PLOT_ONLY:
    names = []
for m in names:
    p = load_predictions(PRED, m)
    p = p[(p.date >= START) & (p.date <= END)]
    if p["date"].nunique() < 2000:  # only models with the full 2014-2023 test period
        continue
    pr = p.groupby("date")["pred"].rank(pct=True)
    tr = p.groupby("date")["target"].rank(pct=True)
    ic = pd.concat([pr, tr], axis=1).groupby(p["date"]).corr().xs("pred", level=1)["target"]
    dec = decile_returns(p)
    tp = top_portfolio(p)
    size = p.groupby("date")["me"].rank(pct=True)
    big = perf(decile_returns(p[size > 0.3])["High(H)"])
    h, hl = perf(dec["High(H)"]), perf(dec["H-L"])
    rows[m] = {"IC": ic.mean(), "IC_t": nw_tstat(ic), "R2": 100 * oos_r2(p.pred, p.target),
               "H_avg": h["Avg %"], "H_sd": h["SD %"], "H_sr": h["SR"], "L_avg": perf(dec["Low(L)"])["Avg %"],
               "HL_avg": hl["Avg %"], "HL_sr": hl["SR"], "TO": 100 * tp["to_p"].mean(),
               "H_ex30_avg": big["Avg %"], "H_ex30_sr": big["SR"]}
    cum[m] = dec["High(H)"].cumsum()
    years[m] = (dec["High(H)"].groupby(dec.index.year).mean() * metrics.ANN * 100).round(1).to_dict()
    print(f"{m:12s} IC {rows[m]['IC']:.4f}  top decile {rows[m]['H_avg']:.1f}% SR {rows[m]['H_sr']:.2f}")

if PLOT_ONLY:
    S = json.loads((ROOT / "report" / "results_summary.json").read_text())
    tab = pd.DataFrame(S["models"]).T
    cum = {m: s for m, s in pd.read_parquet(FIG / "cum.parquet").items()}
else:
    tab = pd.DataFrame(rows).T
    json.dump({"period": [START, END], "ann": metrics.ANN, "models": tab.round(6).to_dict(orient="index"),
               "years": {m: {str(k): v for k, v in d.items()} for m, d in years.items()}},
              open(ROOT / "report" / "results_summary.json", "w"), indent=1)
    pd.DataFrame(cum).to_parquet(FIG / "cum.parquet")
FIG1 = ["Linear", "NN3", "OLS", "Ridge", "Ridge-rank", "PLS", "LDA", "PCA20-Linear", "PCA20-NN3", "LDA9-NN3",
        "GMM-Experts", "Size-Experts", "Linear-SVR", "E2E-Linear", "E2E-NN3", "E2E-Ridge", "DA0.1-PCA20-Linear"]
order = tab.loc[[m for m in FIG1 if m in tab.index]].sort_values("H_sr").index.tolist()

# Figure 1: statistical accuracy (rank IC) vs portfolio quality (top-decile Sharpe), same model order
fig, axes = plt.subplots(1, 2, figsize=(7.0, 0.8 + 0.27 * len(order)), sharey=True)
for ax, col, title, fmt in [(axes[0], "IC", "Mean daily rank IC", "{:.3f}"),
                            (axes[1], "H_sr", "Long-only top decile: Sharpe ratio", "{:.2f}")]:
    vals = tab.loc[order, col]
    ax.barh(range(len(order)), vals, height=0.62,  # orange: trained end to end
            color=[SERIES[1] if m.startswith(("E2E-", "DA")) else SERIES[0] for m in order])
    for i, v in enumerate(vals):
        ax.text(v, i, " " + fmt.format(v), va="center", fontsize=8, color=INK2)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(0, vals.max() * 1.22)
axes[0].set_yticks(range(len(order)))
axes[0].set_yticklabels(order, color=INK2)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"fig_ic_vs_sharpe.{ext}", bbox_inches="tight")

# Figure 2: cumulative top-decile return (sum of daily market-adjusted returns) for up to four models
pick = [m for m in ["NN3", "E2E-NN3", "PCA20-Linear", "DA0.1-PCA20-Linear", "OLS"] if m in cum][:4]
fig, ax = plt.subplots(figsize=(7.0, 2.8))
ends = {}
for m, c in zip(pick, SERIES):
    s = cum[m].dropna() * 100
    ax.plot(s.index, s.values, color=c, lw=1.5, label=m)
    ends[m] = (s.index[-1], s.values[-1])
# end labels, nudged apart so they never overlap
gap = 0.06 * (max(v for _, v in ends.values()) - min(0, min(v for _, v in ends.values())))
placed = []
for m, (x, y) in sorted(ends.items(), key=lambda kv: -kv[1][1]):
    y_lab = min([y] + [p - gap for p in placed])
    placed.append(y_lab)
    ax.text(x, y_lab, f"  {m}", color=INK2, fontsize=8, va="center")
ax.legend(loc="upper left", frameon=False, fontsize=8, ncol=len(pick), handlelength=1.6)
ax.set_ylabel("cumulative return, %", fontsize=8)
ax.yaxis.grid(True, color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.set_title("Long-only top decile, equal weight, market-adjusted (vs CSI 500), before costs", fontsize=9, color=INK, loc="left")
ax.set_xlim(cum[pick[0]].index[0], cum[pick[0]].index[-1] + pd.Timedelta(days=420))
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"fig_cumulative.{ext}", bbox_inches="tight")
print("figures written to", FIG)
