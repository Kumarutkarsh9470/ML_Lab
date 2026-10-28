"""Writes the three project notebooks from the cell sources below."""
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(sys.argv[1])


def nb(cells):
    book = nbf.v4.new_notebook()
    book.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    book.cells = [nbf.v4.new_markdown_cell(s.strip()) if k == "md" else nbf.v4.new_code_cell(s.strip())
                  for k, s in cells]
    return book


nb00 = [
    ("md", """
# 00 · Build the US daily panel from WRDS

The paper uses daily China A-share data with 415 characteristics, which is not public. This notebook builds the
closest US equivalent from CRSP:

| | Paper | Here |
|---|---|---|
| Universe | A-shares, excluding ST and suspended stocks | CRSP common stocks (share codes 10, 11) on NYSE/AMEX/Nasdaq, price ≥ $1 |
| Period | 2010 to 2023, daily | 2010 to 2023, daily |
| Target | VWAP(t+1) to VWAP(t+2) return minus CSI 500 | close(t+1) to close(t+2) return minus CRSP value-weighted market |
| Characteristics | 415 daily | 21 computed daily from CRSP, plus optionally about 200 monthly OSAP signals (used with a one-month lag) |
| Industries | 31 Shenwan dummies | 10 SIC-division dummies |

Each characteristic is ranked across stocks within each day to [-1, 1], with missing values set to 0, and stored as int8.

**Before running:** `pip install wrds` (and `pip install openassetpricing` if `USE_OSAP = True`). The first
`wrds.Connection` asks for your WRDS password and offers to save it in a pgpass file. Downloading 2007 to 2024
takes a while; files already in `data/raw` are skipped. Building the features needs about 8 GB of RAM.
"""),
    ("code", """
from pathlib import Path

import pandas as pd

from mlm import crsp
from mlm.data import load_panel, panel_features

WRDS_USERNAME = "your_wrds_username"
RAW_DIR, PANEL_DIR = Path("data/raw"), Path("data/panel")
DOWNLOAD_YEARS = (2007, 2024)   # 3 years of history before 2010 for mom36_12; 2024 for the t+2 returns at the end of 2023
PANEL_START, PANEL_END = "2010-01-01", "2023-12-31"
MIN_PRICE = 1.0
EXEC_LAG = 1                    # trade at the close of t+1, earn the t+2 return
USE_OSAP = False                # start with the 21 daily characteristics; True adds the OSAP monthly signals
"""),
    ("code", """
import wrds

db = wrds.Connection(wrds_username=WRDS_USERNAME)
crsp.download(db, RAW_DIR, *DOWNLOAD_YEARS)
db.close()
"""),
    ("code", """
dsf, mkt = crsp.load_raw(RAW_DIR)
panel, numeric, dummies = crsp.build_features(dsf, mkt, PANEL_START, PANEL_END, MIN_PRICE, EXEC_LAG)
del dsf
osap = crsp.load_osap(RAW_DIR) if USE_OSAP else None
features = crsp.write_ranked_panel(panel, PANEL_DIR, numeric, dummies, osap,
                                   min_price=MIN_PRICE, exec_lag=EXEC_LAG, osap=USE_OSAP)
print(len(features), "features")
"""),
    ("md", "## Sanity checks: stocks per day by year (compare with Table A.1) and target distribution"),
    ("code", """
counts = panel.groupby("date").size()
pd.DataFrame({"Avg stocks per day": counts.groupby(counts.index.year).mean().round(0),
              "Days": counts.groupby(counts.index.year).size()})
"""),
    ("code", """
panel["target"].describe(percentiles=[0.001, 0.01, 0.5, 0.99, 0.999])
"""),
]

nb01 = [
    ("md", """
# 01 · Baseline stage 1: train the MSE prediction models

The paper's benchmark is the two-stage "predict then optimize" approach. Stage 1, here, fits a return prediction
model by minimizing mean squared error. Stage 2 (notebook 02) sorts stocks on the predictions and builds portfolios.

Setup, following Section 4.1 and Appendix A.2:
* **Rolling windows** that move one quarter at a time: 39 months of training, 9 months of validation, then predict the
  next quarter. Two trading days before each boundary are dropped, because their labels are realized after it.
* **Models:** Linear and NN1 to NN5, with hidden layers of 32, 16, 8, 4 and 2 units; NNk uses the first k. Inputs pass
  through LayerNorm, and each hidden layer has ReLU, BatchNorm and 5% dropout.
* **Training:** Adam with learning rate 1e-3; early stopping after 10 epochs without improvement in validation MSE,
  keeping the best epoch.

Choices the paper does not report: batch size 10,000; at most 100 epochs; labels winsorized at the 0.5% and 99.5%
quantiles within each day, for training and validation only (evaluation uses raw returns); training in percent units.

`DATA_SOURCE = "synthetic"` runs the whole pipeline on simulated data. Switch it to `"wrds"` after running notebook 00.
Finished model-window pairs are saved and skipped on reruns, so a long run can be interrupted and resumed.
"""),
    ("code", """
from pathlib import Path

import pandas as pd

from mlm.data import make_synthetic_panel, panel_features, rolling_windows
from mlm.train import TrainConfig, run_rolling

DATA_SOURCE = "cn"   # "cn" (China A-shares, notebook 00_build_data_cn), "synthetic" (pipeline test), "wrds" (US)
SOURCES = {
    "cn": dict(panel="D:/ML_Markowitz_data/panel_cn", preds="D:/ML_Markowitz_data/preds/cn"),
    "synthetic": dict(panel="data_synth/panel", preds="preds/synthetic"),
    "wrds": dict(panel="data/panel", preds="preds/wrds"),
}
PANEL_DIR, PRED_DIR = (Path(SOURCES[DATA_SOURCE][k]) for k in ("panel", "preds"))

# Paper scale: all six models, test quarters 2014Q1 to 2023Q4 (40 windows). For a quick check use
# MODELS = ["Linear", "NN3"] and LAST_TEST = "2014-12-31".
MODELS = ["Linear", "NN1", "NN2", "NN3", "NN4", "NN5"]
FIRST_TEST, LAST_TEST = "2014-01-01", "2023-12-31"
FEATURES = None   # None uses every feature in the panel; or pass a list of column names
CFG = TrainConfig()
CFG
"""),
    ("code", """
if DATA_SOURCE == "synthetic" and not (PANEL_DIR / "meta.json").exists():
    make_synthetic_panel(PANEL_DIR)

features = FEATURES or panel_features(PANEL_DIR)
windows = rolling_windows(FIRST_TEST, LAST_TEST)
print(f"{len(features)} features, {len(windows)} windows")
pd.DataFrame(windows).set_index("name")
"""),
    ("code", """
log = run_rolling(MODELS, PANEL_DIR, windows, PRED_DIR, CFG, features)
"""),
    ("md", "## Training summary (averages over windows)"),
    ("code", """
log.groupby("model")[["epochs", "best_epoch", "val_mse", "test_r2", "seconds"]].mean().round(4)
"""),
]

nb02 = [
    ("md", """
# 02 · Baseline stage 2: portfolios from the MSE predictions

Reproduces the MSE (benchmark) side of the paper's portfolio tables:

* **Table 2:** equal-weighted decile portfolios sorted on predicted returns, rebalanced daily. H-L is (H - L) / 2,
  which is the convention the paper's numbers imply. **Table A.4:** the same, value weighted.
* **Table A.3:** the long-only top decile: turnover, largest one-day loss and maximum drawdown.
* **Table 3:** transaction costs of 0 to 40 bps. *MSE-Top* holds the raw top decile and pays the costs. *MSE-Opt* solves
  the paper's turnover-penalized problem (eq. 19) each day.

Conventions: returns are excess of the market and annualized with 252 days. SR = Avg / SD. Drawdowns use cumulative
sums of daily returns, and Length is calendar days from peak to trough inclusive, as in the paper. Turnover is one-sided,
0.5·‖w_t − w_{t−1}‖₁, so the daily cost is γ × turnover. t-statistics are Newey-West with ⌊T^¼⌋ lags.
"""),
    ("code", """
from pathlib import Path

import json

import matplotlib.pyplot as plt
import pandas as pd

from mlm import metrics
from mlm.linear import GRIDS as LINEAR_GRIDS
from mlm.metrics import drawdown, nw_tstat, oos_r2, perf
from mlm.models import MODELS as ALL_MODELS
from mlm.portfolio import decile_returns, top_portfolio
from mlm.train import load_predictions

DATA_SOURCE = "cn"
SOURCES = {
    "cn": dict(panel="D:/ML_Markowitz_data/panel_cn", preds="D:/ML_Markowitz_data/preds/cn"),
    "synthetic": dict(panel="data_synth/panel", preds="preds/synthetic"),
    "wrds": dict(panel="data/panel", preds="preds/wrds"),
}
PANEL_DIR, PRED_DIR = (Path(SOURCES[DATA_SOURCE][k]) for k in ("panel", "preds"))
metrics.ANN = json.loads((PANEL_DIR / "meta.json").read_text()).get("ann", 252)  # about 242 for A-shares
print("annualization:", round(metrics.ANN, 1), "days")
RES_DIR, FIG_DIR = Path("results") / DATA_SOURCE, Path("figures") / DATA_SOURCE
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [m for m in ALL_MODELS + list(LINEAR_GRIDS) if (PRED_DIR / m).exists()]
TC_MODEL = "NN3" if "NN3" in MODELS else MODELS[-1]   # the paper uses NN3 for Table 3
TCRS = [0, 10, 20, 30, 40]                             # transaction cost rates in bps

preds = {m: load_predictions(PRED_DIR, m) for m in MODELS}
pd.DataFrame({m: {"first": p.date.min().date(), "last": p.date.max().date(), "days": p.date.nunique(),
                  "stocks/day": round(len(p) / p.date.nunique())} for m, p in preds.items()}).T
"""),
    ("md", "## Out-of-sample R² (%, pooled against a zero forecast)"),
    ("code", """
r2 = pd.Series({m: 100 * oos_r2(p["pred"], p["target"]) for m, p in preds.items()}, name="OOS R2 %")
r2.to_csv(RES_DIR / "oos_r2.csv")
r2.round(3)
"""),
    ("md", "## Table 2 analogue: equal-weighted decile portfolios (MSE)"),
    ("code", """
def decile_table(weight):
    return pd.concat({m: decile_returns(p, weight).apply(perf).T for m, p in preds.items()}, axis=1)

t2 = decile_table("ew")
t2.to_csv(RES_DIR / "table2_mse_deciles_ew.csv")
t2.round(2)
"""),
    ("md", "## Table A.4 analogue: value-weighted decile portfolios (MSE)"),
    ("code", """
ta4 = decile_table("vw")
ta4.to_csv(RES_DIR / "tableA4_mse_deciles_vw.csv")
ta4.round(2)
"""),
    ("md", "## Table A.3 analogue: long-only top decile (MSE, no costs)"),
    ("code", """
ta3 = pd.DataFrame({m: pd.concat([pd.Series({"Turnover %": tp["to_p"].mean() * 100}), perf(tp["gross"]),
                                  drawdown(tp["gross"])])
                    for m, p in preds.items() for tp in [top_portfolio(p)]}).T
ta3.to_csv(RES_DIR / "tableA3_mse_top.csv")
ta3
"""),
    ("md", "## Table 3 analogue: MSE-Top and MSE-Opt under transaction costs"),
    ("code", """
p = preds[TC_MODEL]
rows, daily = [], {}
for tcr in TCRS:
    top, opt = top_portfolio(p, tcr / 1e4), top_portfolio(p, tcr / 1e4, optimize=True)
    daily[tcr] = top["net"], opt["net"]
    for method, d in [("MSE-Top", top), ("MSE-Opt", opt)]:
        rows.append(pd.concat([
            pd.Series({"TCR": tcr, "Method": method}),
            perf(d["net"]).set_axis(["Ann Ret %", "Ann SD %", "Ann SR"]),
            pd.Series({"TO-P %": d["to_p"].mean() * 100, "TO-S %": d["to_s"].mean() * 100}),
            drawdown(d["net"]),
        ]))
    diff = opt["net"] - top["net"]  # identical portfolios at 0 bps; the paper reports 0 there
    rows[-1]["t-stat Opt vs Top"] = nw_tstat(diff) if diff.abs().max() > 1e-10 else 0.0
t3 = pd.DataFrame(rows).set_index(["TCR", "Method"])
t3.to_csv(RES_DIR / f"table3_mse_tc_{TC_MODEL}.csv")
t3
"""),
    ("code", """
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
dec = decile_returns(p)
dec[["High(H)", "Low(L)", "H-L"]].cumsum().plot(ax=axes[0])
axes[0].set_title(f"{TC_MODEL}: cumulative decile returns, gross")
for tcr, color in zip([0, 20, 40], ["C0", "C1", "C3"]):
    top, opt = daily[tcr]
    top.cumsum().plot(ax=axes[1], color=color, label=f"MSE-Top {tcr} bps")
    opt.cumsum().plot(ax=axes[1], color=color, ls="--", label=f"MSE-Opt {tcr} bps")
axes[1].set_title(f"{TC_MODEL}: long-only top decile, net of costs")
axes[1].legend(fontsize=8)
for ax in axes:
    ax.set_ylabel("cumulative excess return")
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "baseline_cumulative.png", dpi=150)
"""),
    ("md", """
## Reading against the paper

These are the benchmark numbers the E2E model will be compared with. On the paper's A-share data, the MSE results are:

* NN3 top decile: Avg 40.0%, SD 20.6%, SR 1.94, turnover 60.6%.
* NN3 H-L: Avg 53.1%, SR 8.88. Most of the H-L return comes from the short (Low) leg.
* MSE-Top falls below zero at 30 bps. MSE-Opt cuts portfolio turnover (TO-P), but its signal turnover (TO-S) stays fixed.

US levels will differ: costs, short-sale access and the signal set are different. What should carry over are the
patterns: H-L driven by the short leg, the gap between MSE-Top and MSE-Opt growing with costs, and TO-S not changing.
"""),
]

nb00cn = [
    ("md", """
# 00 · Build the China A-share daily panel (the paper's market)

Two free Qlib-format dumps, both unpacked under `D:/ML_Markowitz_data/raw_cn`:

* **joshuaxql/qlib_data** (Hugging Face, `cn_data.zip`, SHA-256 checked): raw daily prices, VWAP, adjustment factor,
  total and float market cap, price limits, ST intervals, Shenwan level-1 industries (31, as in the paper),
  point-in-time financial indicators with announcement dates, 2010 onward, delisted stocks included.
* **chenditc/investment_data** (GitHub, `qlib_bin.tar.gz`): adjusted prices back to 2000, used for 2007-2009 history so
  long-horizon characteristics exist in 2010, and the CSI 500 index.

| | Paper | Here |
|---|---|---|
| Universe | A-shares, excluding ST and suspended stocks | Same; ST and suspension (no bar or zero volume) on decision day t |
| Target | VWAP(t+1) to VWAP(t+2) return minus the CSI 500 | Same; the CSI 500 leg uses its close-to-close return over day t+2 |
| Characteristics | 415 daily | 47 price/volume, 163 point-in-time financial indicators, 10 valuation ratios |
| Industries | 31 Shenwan dummies | 31 Shenwan level-1 dummies, membership as of day t |

A financial value is usable from the trading day **after** its announcement. Each characteristic is ranked across
stocks within each day to [-1, 1], with missing values set to 0, and stored as int8. Building the panel takes about
15-30 minutes and 6-8 GB of RAM; it is written to D:.
"""),
    ("code", """
from pathlib import Path

import pandas as pd

from mlm import cn
from mlm.data import panel_features

RAW = Path("D:/ML_Markowitz_data/raw_cn")
J_ROOT, C_ROOT = RAW / "joshuaxql" / "cn_data", RAW / "chenditc" / "qlib_bin"
UNRANKED_DIR = Path("D:/ML_Markowitz_data/panel_cn_unranked")   # raw characteristics, kept for later checks
PANEL_DIR = Path("D:/ML_Markowitz_data/panel_cn")
KAGGLE_DIR = Path("D:/ML_Markowitz_data/kaggle_stockchina/original_data")   # optional independent check (JoinQuant data)
PANEL_START, PANEL_END = "2010-01-01", "2023-12-31"
"""),
    ("md", "Step 1: price/volume characteristics, target and universe for all stocks (skipped if already saved)."),
    ("code", """
if not (UNRANKED_DIR / "meta.json").exists():
    panel, numeric, dummies = cn.build_panel(J_ROOT, C_ROOT, PANEL_DIR, PANEL_START, PANEL_END)
    cn.save_unranked(panel, numeric, dummies, UNRANKED_DIR)
    del panel
"""),
    ("md", "Step 2: add point-in-time fundamentals and valuation ratios, rank within each day, one year at a time. "
           "Years already written are skipped."),
    ("code", """
field_names = cn.field_names_from_manifest(RAW / "joshuaxql" / "manifest.json")
features = cn.write_cn_panel(UNRANKED_DIR, J_ROOT, PANEL_DIR, field_names)
print(len(features), "features in total")
"""),
    ("md", "## Sanity checks: stocks per day by year (paper's Table A.1) and target distribution"),
    ("code", """
u = pd.concat([pd.read_parquet(f, columns=["date", "target"]) for f in sorted(UNRANKED_DIR.glob("*.parquet"))])
paper_a1 = {2010: 1841, 2011: 2169, 2012: 2393, 2013: 2462, 2014: 2527, 2015: 2736, 2016: 2932, 2017: 3345,
            2018: 3612, 2019: 3741, 2020: 4015, 2021: 4485, 2022: 4869, 2023: 5196}
counts = u.groupby("date").size()
pd.DataFrame({"Avg stocks per day": counts.groupby(counts.index.year).mean().round(0),
              "Paper Table A.1": pd.Series(paper_a1), "Days": counts.groupby(counts.index.year).size()})
"""),
    ("code", """
print(f"{len(u):,} stock-days in total (paper text: 5,962,768; its Table A.1 implies about 11.2 million)")
u["target"].describe(percentiles=[0.001, 0.01, 0.5, 0.99, 0.999])
"""),
    ("md", "## Independent check of the VWAP returns against the Kaggle StockChina files"),
    ("code", """
cn.crosscheck_kaggle(J_ROOT, KAGGLE_DIR) if KAGGLE_DIR.exists() else "Kaggle folder not found, skipped"
"""),
]

nb01b = [
    ("md", """
# 01b · Linear models and linear dimension reduction (two-stage, closed form)

The improved linear baselines and the dimension-reduction methods. All of them predict with a single coefficient
vector, so each is fit in closed form from one pass over the window's training data (n, Σx, X'X, X'y and per-class
sums, accumulated on the GPU). That makes it cheap to run every family with a hyperparameter grid on every window.

| Family | What it is | Grid (chosen per window on validation) |
|---|---|---|
| OLS | least squares, as the paper's Linear but fit exactly | none |
| Ridge | L2 penalty λ × average feature variance | λ ∈ {0.001, 0.01, 0.1, 1, 10} |
| Ridge-rank | ridge on each day's return rank, rescaled to return units | same λ |
| PCR | regression on the top-K principal components of the numeric features, plus industry dummies | K ∈ {5, 10, 20, 40, 80} |
| PLS | partial least squares (SIMPLS) | K ∈ {1, 2, 3, 5, 8, 12, 20} |
| LDA | regression on the top-m Linear Discriminant Analysis directions separating each day's return deciles | m ∈ {1, 3, 5, 9}, shrinkage α ∈ {0.05, 0.2, 0.5} |
| LDA-top | Fisher discriminant for "top decile vs the rest", aimed at the long-only portfolio | α ∈ {0.01, 0.05, 0.2, 0.5} |

Everything (PCA loadings, LDA directions, coefficients) is fit on the training set only. The grid is chosen on
the validation set by **mean daily rank IC**, and the test quarter is never used for any choice. Every setting
tried is saved in `linear_grid.csv`, so the full grid can be reported (see PAPER_ISSUES.md on multiple testing).
The estimators are checked against scikit-learn in `checks/test_linear.py`.
"""),
    ("code", """
from pathlib import Path

import pandas as pd

from mlm.data import panel_features, rolling_windows
from mlm.linear import GRIDS, run_linear

PANEL_DIR = Path("D:/ML_Markowitz_data/panel_cn")
PRED_DIR = Path("D:/ML_Markowitz_data/preds/cn")
FIRST_TEST, LAST_TEST = "2014-01-01", "2023-12-31"   # all 40 windows
FAMILIES = list(GRIDS)
CRITERION = "ic"
windows = rolling_windows(FIRST_TEST, LAST_TEST)
print(f"{len(windows)} windows, {len(panel_features(PANEL_DIR))} features, families: {FAMILIES}")
"""),
    ("code", """
grid = run_linear(PANEL_DIR, windows, PRED_DIR, FAMILIES, CRITERION)
"""),
    ("md", "## Chosen settings and validation scores"),
    ("code", """
chosen = grid[grid["chosen"] == True]
print(chosen.groupby("family")["params"].agg(lambda s: s.value_counts().head(3).to_dict()))
chosen.groupby("family")[["ic", "top", "mse"]].mean().sort_values("ic", ascending=False)
"""),
]

nb01c = [
    ("md", """
# 01c · Our extensions, two-stage (statistical) training

Everything here is fit on each window's training data only, with choices made on validation, exactly like
the baselines. Outputs go to the same prediction folder, so notebook 02 evaluates them next to the paper's models.

| Name | What it is | Why |
|---|---|---|
| PCA20-Linear, PCA20-NN3 | 20 principal components of the numeric characteristics, then a linear or NN3 head | unsupervised reduction of 220 collinear characteristics |
| PLS5-NN3 | 5 PLS (SIMPLS) directions, then NN3 | reduction supervised by covariance with returns |
| LDA9-NN3 | the 9 Linear Discriminant Analysis directions separating each day's return deciles, then NN3 | reduction supervised by the decile structure the portfolio uses |
| SFA20-NN3 | the 20 slowest directions (smallest change-to-level variance) | persistence-aware: low-turnover signals, for the cost experiment |
| GMM-Experts | linear experts gated by a GMM on 10 PCA scores (K = 2-4) | different pricing rules for different kinds of firms, as in the paper's own example |
| Size-Experts | the same with hard size terciles | checks whether the GMM gate adds anything over size |
| Linear-SVR, RFF-SVR | ε-insensitive loss with L2 penalty; linear and random-Fourier-feature kernel | robust and nonlinear two-stage baselines |

The reductions are also used as starting points for E2E training (notebook 03), including decision-aware PCA.
"""),
    ("code", """
from pathlib import Path

from mlm.data import rolling_windows
from mlm.mixture import run_mixture
from mlm.reduce import run_reduced
from mlm.train import TrainConfig, run_rolling

PANEL_DIR = Path("D:/ML_Markowitz_data/panel_cn")
PRED_DIR = Path("D:/ML_Markowitz_data/preds/cn")
windows = rolling_windows("2014-01-01", "2023-12-31")
REDUCED = ["PCA20-Linear", "PCA20-NN3", "PLS5-NN3", "LDA9-NN3", "SFA20-NN3"]
"""),
    ("code", "run_reduced(REDUCED, PANEL_DIR, windows, PRED_DIR, TrainConfig())"),
    ("code", "run_mixture(PANEL_DIR, windows, PRED_DIR)"),
    ("code", """
svr = TrainConfig(loss="eps", eps=0.5, weight_decay=1e-4)  # tube of 0.5% daily return, in percent units
run_rolling(["Linear", "RFF"], PANEL_DIR, windows, PRED_DIR, svr, tag="-SVR", log_name="svr_log.csv")
"""),
]

nb03 = [
    ("md", """
# 03 · End-to-end (E2E) training

The portfolio problem is a layer of the network; the exact layer and its gradient are in `mlm/e2e.py` and
`mlm/meansd.py` and are checked in `checks/test_e2e_layer.py`, `checks/test_meansd.py` and
`checks/test_e2e_runs.py`. Every E2E model starts from its trained two-stage version (Appendix A.2.2).

| Section | Paper | Runs |
|---|---|---|
| A | Table 2 / A.3 | E2E for Linear and NN1-NN5, long-only top decile; smoothing ζ chosen on validation |
| B | Table 3 | NN3 trained with costs of 10-40 bps (eq. 20) |
| C | our extensions | E2E from ridge, PCA20-NN3 and GMM experts; decision-aware PCA (learnable projection, penalty μ) |
| D | PAPER_ISSUES D1 | NN3 trained from scratch, to check the warm start is not doing the work |
| E | Table 4 / 5 | mean-SD E2E and PPP across risk aversion η, with and without 20 bps costs |

**Run each section only after its time estimate** (measured on one window first).
"""),
    ("code", """
from pathlib import Path

from mlm.data import rolling_windows
from mlm.e2e import E2EConfig, run_e2e
from mlm.models import MODELS

PANEL_DIR = Path("D:/ML_Markowitz_data/panel_cn")
PRED_DIR = Path("D:/ML_Markowitz_data/preds/cn")   # two-stage models (warm starts) and E2E outputs
windows = rolling_windows("2014-01-01", "2023-12-31")
ZETA_GRID = (0.05, 0.2, 1.0)   # smoothing, chosen per model and window on validation (PAPER_ISSUES A2)
"""),
    ("md", "## A. E2E for the paper's models (Tables 2 and A.3)"),
    ("code", "run_e2e(MODELS, PANEL_DIR, windows, PRED_DIR, PRED_DIR, E2EConfig(zeta_rel=ZETA_GRID), tag='E2E')"),
    ("md", "## B. Transaction costs (Table 3): NN3 trained with each cost rate"),
    ("code", """
for bps in (10, 20, 30, 40):
    run_e2e(["NN3"], PANEL_DIR, windows, PRED_DIR, PRED_DIR, E2EConfig(gamma=bps / 1e4), tag=f"E2E-tc{bps}")
"""),
    ("md", "## C. E2E for our models, and decision-aware PCA"),
    ("code", """
run_e2e(["Ridge", "PCA20-NN3", "GMM-Experts"], PANEL_DIR, windows, PRED_DIR, PRED_DIR, E2EConfig(zeta_rel=ZETA_GRID), tag="E2E")
for mu in (0.1, 1.0, 10.0):   # strength of the pull toward the PCA loadings
    run_e2e(["PCA20-Linear", "PCA20-NN3"], PANEL_DIR, windows, PRED_DIR, PRED_DIR, E2EConfig(da_mu=mu), tag=f"DA{mu:g}")
"""),
    ("md", "## D. Warm start check: NN3 from scratch (first 8 windows)"),
    ("code", "run_e2e(['NN3'], PANEL_DIR, windows[:8], PRED_DIR, PRED_DIR, E2EConfig(warm_start=False), tag='E2E-scratch')"),
    ("md", "## E. Risk aversion (Tables 4 and 5): mean-SD E2E and PPP"),
    ("code", """
ETAS = (0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
for eta in ETAS:
    lr = 1e-4 if eta < 1 else 1e-3   # as in Appendix A.2.2
    for gamma, g in ((0.0, "gross"), (0.002, "net")):
        run_e2e(["NN3"], PANEL_DIR, windows, PRED_DIR, PRED_DIR, E2EConfig(objective="meansd", eta=eta, gamma=gamma, lr=lr),
                tag=f"MSD-{g}-eta{eta:g}")
        run_e2e(["NN3"], PANEL_DIR, windows, PRED_DIR, PRED_DIR, E2EConfig(objective="ppp", eta=eta, gamma=gamma, lr=lr),
                tag=f"PPP-{g}-eta{eta:g}")
"""),
]

nb04 = [
    ("md", """
# 04 · Evaluation of every model: paper replication, extensions and fixes

Out-of-sample test quarters 2014Q1-2023Q4, China A-shares with the tradability rules of PAPER_ISSUES A15.
Each table reports what the paper reports, plus the fixes in PAPER_ISSUES.md: one- and two-sided p-values
(C5), raw H-L next to (H-L)/2 (A7), a validation-tuned MSE-Opt benchmark (D2), gross and net results at every
risk aversion (C2, C6), realized frontiers and volatility-matched tests (D3), the warm-start check (D1) and a
Holm adjustment across the E2E-vs-MSE comparisons. Sections whose runs are not finished are skipped.
"""),
    ("code", """
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from mlm import metrics
from mlm.data import rolling_windows
from mlm.e2e import backtest_meansd, load_sigma
from mlm.linear import GRIDS as LINEAR
from mlm.metrics import drawdown, nw_test, oos_r2, perf
from mlm.mixture import GRID as MIXTURE
from mlm.models import MODELS
from mlm.portfolio import decile_returns, top_portfolio, tuned_kappa
from mlm.train import load_predictions

PANEL_DIR, PRED_DIR = Path("D:/ML_Markowitz_data/panel_cn"), Path("D:/ML_Markowitz_data/preds/cn")
UNRANKED = Path("D:/ML_Markowitz_data/panel_cn_unranked")
RES, FIG = Path("results/cn"), Path("figures/cn")
RES.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
metrics.ANN = json.loads((PANEL_DIR / "meta.json").read_text())["ann"]
windows = rolling_windows("2014-01-01", "2023-12-31")
EXT = ["PCA20-Linear", "PCA20-NN3", "PLS5-NN3", "LDA9-NN3", "SFA20-NN3", "Linear-SVR", "RFF-SVR"] + list(MIXTURE)
have = lambda m: (PRED_DIR / m).exists() and any((PRED_DIR / m).glob("20*Q*.parquet"))
two_stage = [m for m in MODELS + list(LINEAR) + EXT if have(m)]
e2e = [d.name for d in sorted(PRED_DIR.iterdir()) if d.name.startswith(("E2E-", "DA")) and not d.name.startswith(("E2E-tc", "E2E-scratch")) and have(d.name)]
print("two-stage:", two_stage); print("E2E:", e2e)
"""),
    ("md", "## Table 2 analogue: decile portfolios, accuracy, and E2E vs its MSE starting point"),
    ("code", """
rows, tops = {}, {}
for m in two_stage + e2e:
    p = load_predictions(PRED_DIR, m)
    pr, tr = p.groupby("date")["pred"].rank(), p.groupby("date")["target"].rank()
    ic = pd.concat([pr, tr], axis=1).groupby(p["date"]).corr().xs("pred", level=1)["target"]
    dec, tp = decile_returns(p), top_portfolio(p)
    tops[m] = tp["gross"]
    h, l, hl, hlr = (perf(dec[c]) for c in ("High(H)", "Low(L)", "H-L", "H-L raw"))
    rows[m] = {"R2 %": 100 * oos_r2(p.pred, p.target), "IC": ic.mean(), "IC t": nw_test(ic)["t"],
               "H Avg %": h["Avg %"], "H SD %": h["SD %"], "H SR": h["SR"], "L Avg %": l["Avg %"],
               "H-L Avg %": hl["Avg %"], "H-L SR": hl["SR"], "H-L raw Avg %": hlr["Avg %"], "TO %": 100 * tp["to_p"].mean()}
t2 = pd.DataFrame(rows).T
# E2E vs the two-stage model it started from: difference in daily top-decile return
base = lambda m: m.split("-", 1)[1] if m.startswith("E2E-") else m.split("-", 1)[1]
tests = {m: nw_test(tops[m] - tops[base(m)]) for m in e2e if base(m) in tops}
if tests:
    tt = pd.DataFrame(tests).T
    tt["p two-sided (Holm)"] = multipletests(tt["p two-sided"], method="holm")[1]
    t2 = t2.join(tt.add_prefix("vs MSE: "))
t2.to_csv(RES / "table2_all.csv"); t2.round(3)
"""),
    ("md", "## Table 3 analogue: transaction costs (NN3)"),
    ("code", """
nn3, nn3_val = load_predictions(PRED_DIR, "NN3"), load_predictions(PRED_DIR, "NN3", "val")
rows = []
for bps in (0, 10, 20, 30, 40):
    g = bps / 1e4
    kappa, chosen = tuned_kappa(nn3, nn3_val, windows, g) if bps else (None, None)
    runs = {"MSE-Top": top_portfolio(nn3, g), "MSE-Opt": top_portfolio(nn3, g, optimize=True)}
    if bps:
        runs["MSE-Opt (tuned)"] = top_portfolio(nn3, g, optimize=True, kappa=kappa)
    e2e_name = "E2E-NN3" if bps == 0 else f"E2E-tc{bps}-NN3"
    if have(e2e_name):  # E2E signal, same cost-aware construction as MSE-Opt
        runs["E2E"] = top_portfolio(load_predictions(PRED_DIR, e2e_name), g, optimize=bps > 0)
    for k, d in runs.items():
        r = pd.concat([pd.Series({"TCR": bps, "Method": k}), perf(d["net"]).set_axis(["Ann Ret %", "Ann SD %", "Ann SR"]),
                       pd.Series({"TO-P %": 100 * d["to_p"].mean(), "TO-S %": 100 * d["to_s"].mean()}), drawdown(d["net"])])
        if k != "MSE-Top":
            r = pd.concat([r, nw_test(d["net"] - runs["MSE-Opt"]["net"]).add_prefix("vs MSE-Opt: ")]) if k != "MSE-Opt" else r
        rows.append(r)
t3 = pd.DataFrame(rows).set_index(["TCR", "Method"]); t3.to_csv(RES / "table3_all.csv"); t3.round(3)
"""),
    ("md", "## Tables 4 and 5 analogue: risk aversion (mean-SD), E2E vs MSE-Opt vs PPP"),
    ("code", """
ETAS = (0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
daily = lambda tag: pd.concat([pd.read_parquet(f) for f in sorted((PRED_DIR / f"{tag}-NN3").glob("*_daily.parquet"))]).set_index("date").sort_index() \\
    if (PRED_DIR / f"{tag}-NN3").exists() else None
sig = load_sigma(UNRANKED, nn3)
rows, series = [], {}
for eta in ETAS:
    for g, gamma in (("gross", 0.0), ("net", 0.002)):
        cands = {"MSE-Opt": backtest_meansd(nn3, sig, eta, gamma), "E2E": daily(f"MSD-{g}-eta{eta:g}"), "PPP": daily(f"PPP-{g}-eta{eta:g}")}
        for k, d in cands.items():
            if d is None:
                continue
            r = d["net"] if gamma else d["gross"]
            series[(eta, g, k)] = r
            st = perf(r)
            rows.append({"eta": eta, "type": g, "method": k, "Daily ret %": 100 * r.mean(), "Daily vol %": 100 * r.std(),
                         "Utility %": 100 * (r.mean() - eta * r.std()), "Ann Ret %": st["Avg %"], "Ann Vol %": st["SD %"], "SR": st["SR"]})
t4 = pd.DataFrame(rows).pivot_table(index=["eta", "method"], columns="type").round(3) if rows else None
t4
"""),
    ("code", """
# Table 5: pooled tests across eta. Volatility-matched (scaled to 20% a year) return, and realized utility.
out = {}
for g in ("gross", "net"):
    for bench in ("MSE-Opt", "PPP"):
        vm, ut = [], []
        for eta in ETAS:
            a, b = series.get((eta, g, "E2E")), series.get((eta, g, bench))
            if a is None or b is None:
                continue
            scale = lambda r: r * 0.20 / (r.std() * np.sqrt(metrics.ANN))
            vm.append(scale(a) - scale(b))
            ut.append((a - eta * a.std()) - (b - eta * b.std()))
        if vm:
            out[(g, f"E2E vs {bench}")] = pd.concat([nw_test(pd.concat(vm)).add_prefix("vol-matched "), nw_test(pd.concat(ut)).add_prefix("utility ")])
pd.DataFrame(out).T.round(4) if out else "Section E of notebook 03 has not been run yet"
"""),
    ("code", """
# realized efficient frontiers (Figure 2)
if series:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = {"E2E": "#2a78d6", "MSE-Opt": "#eb6834", "PPP": "#1baf7a"}
    for k, c in colors.items():
        for g, ls in (("gross", "--"), ("net", "-")):
            pts = [(series[(e, g, k)].std() * np.sqrt(metrics.ANN), series[(e, g, k)].mean() * metrics.ANN) for e in ETAS if (e, g, k) in series]
            if pts:
                x, y = zip(*pts)
                ax.plot(x, y, ls, marker="o", ms=3, color=c, label=f"{k} ({g})")
    ax.set_xlabel("annualized standard deviation"); ax.set_ylabel("annualized market-adjusted return")
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.3)
    fig.savefig(FIG / "frontier.png", dpi=150, bbox_inches="tight")
"""),
    ("md", "## Warm-start check (PAPER_ISSUES D1): E2E from scratch vs warm-started, same windows"),
    ("code", """
if have("E2E-scratch-NN3") and have("E2E-NN3"):
    s = top_portfolio(load_predictions(PRED_DIR, "E2E-scratch-NN3"))["gross"]
    w = tops["E2E-NN3"].loc[s.index]
    pd.DataFrame({"from scratch": perf(s), "warm start": perf(w)}).T.join(pd.DataFrame([nw_test(w - s)], index=["warm start"]))
"""),
]

for name, cells in [("00_build_data_wrds", nb00), ("00_build_data_cn", nb00cn), ("01_train_mse", nb01),
                    ("01b_train_linear", nb01b), ("01c_train_extensions", nb01c), ("02_evaluate_baseline", nb02),
                    ("03_train_e2e", nb03), ("04_evaluate_all", nb04)]:
    if len(sys.argv) > 2 and name not in sys.argv[2:]:  # optionally write only the named notebooks
        continue
    nbf.write(nb(cells), ROOT / f"{name}.ipynb")
    print("wrote", name)
