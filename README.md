# Learning for the Decision: replicating and extending *Machine Learning Meets Markowitz*

A replication and extension of Wang, Gao, Harvey, Liu and Tao (2026), *Machine Learning Meets Markowitz*
(SSRN 5947774), on the paper's own market: daily China A-shares, 2010 to 2023, built entirely from free data.

The paper argues that return forecasts should be trained for the portfolio they feed, not for mean squared
error, and does so by putting the portfolio optimizer inside the network (end-to-end, E2E, learning). This
repository contains:

- **A free-data replication of the paper's setting.** It matches the paper's market, frequency, VWAP execution,
  ST and suspension filters, Shenwan industries and rolling windows.
- **An audit of the paper.** The audit recovers conventions the paper leaves unstated and finds numbers that
  contradict each other, claims the tables don't support, and a tradability problem the paper doesn't address.
  Each item has a resolution; see [PAPER_ISSUES.md](PAPER_ISSUES.md).
- **An exact, GPU-friendly differentiable portfolio layer.** It replaces the paper's generic convex-optimization
  differentiation, and covers the top decile, transaction costs and mean–standard deviation, all with closed-form
  gradients.
- **Extensions in the prediction layer.**
  - closed-form linear models (OLS, ridge, ridge on ranks)
  - dimension reduction: PCA, PLS and Linear Discriminant Analysis, plus persistence-aware components
  - decision-aware PCA
  - GMM mixtures of linear experts
  - linear and kernel SVR

The report ([report/report.pdf](report/report.pdf)) and slides ([report/ML_Markowitz_presentation.pptx](report/ML_Markowitz_presentation.pptx))
summarize the work.

## Status

| Part | Status |
|---|---|
| Data pipeline (China A-shares, tradability rules) | done |
| Two-stage MSE models: Linear, NN1–NN5 (paper) | done, 40 test quarters |
| Closed-form linear family and reductions: OLS, Ridge, Ridge-rank, PCR, PLS, LDA, LDA-top | done, 40 test quarters |
| Reduced networks (PCA, PLS, LDA, persistence-aware), GMM and size experts, linear and kernel SVR | done, 40 test quarters |
| E2E: Linear, NN3, Ridge, GMM experts, PCA20-NN3; decision-aware PCA (μ = 0.1, 1) | done, 40 test quarters |
| E2E with 20 bps costs (NN3); mean–SD E2E at η = 1 (NN3); NN3 from scratch (8 windows) | done |
| Remaining grid: NN1, NN2, NN4, NN5 E2E; costs of 10, 30, 40 bps; other η; PPP | implemented (`run_real.py` stages A, B, E), not yet run |

### Results (2014–2023, 40 test quarters)

Long-only, equal-weight top decile, before costs. Returns are market-adjusted (vs the CSI 500) and annualized.
Each E2E model is compared with the two-stage model it starts from; *t* is Newey–West.

| Model | Rank IC: two-stage → E2E | Top decile, % a year: two-stage → E2E | Sharpe: two-stage → E2E | *t* |
|---|---|---|---|---|
| Linear | 0.067 → 0.055 | 31.1 → 31.8 | 1.36 → 1.37 | 0.42 |
| NN3 | 0.061 → 0.056 | 29.0 → 31.5 | 1.23 → 1.35 | 1.69 |
| Ridge | 0.069 → 0.053 | 27.8 → 31.9 | 1.23 → 1.36 | 2.17 |
| PCA20-Linear, decision-aware PCA (μ = 0.1) | 0.056 → 0.048 | 21.8 → 30.4 | 1.00 → 1.36 | 4.96 |

- **E2E lowers statistical accuracy but raises portfolio returns,** the paper's argument, in our data.
- **Decision-aware PCA gives the largest gain,** and is the only comparison that survives a Holm adjustment.
- **With 20 bps costs,** NN3 trained with costs earns 24.7% net with 7% daily turnover. The comparisons:
  - the MSE top decile: 15.2% with 28% turnover
  - MSE-Opt: 22.1%
  - validation-tuned MSE-Opt: 23.2% (difference not significant, *t* = 1.2)
- **Mean–SD at η = 1:** Sharpe 1.36 for E2E against 1.28 for MSE-Opt (not significant).

Full tables: [report/report.pdf](report/report.pdf); numbers in `report/results_summary.json` and
`report/results_decision.json`.

## Data (free, no registration)

The notebooks expect the data under `D:/ML_Markowitz_data/`. Change the paths at the top of each notebook to use
another location.

| Source | File | Use |
|---|---|---|
| [joshuaxql/qlib_data](https://huggingface.co/datasets/joshuaxql/qlib_data) (Hugging Face) | `cn_data.zip`, 842 MB, SHA-256 `50d6fbb3…7757` | raw daily prices, VWAP, adjustment factors, market cap, price limits, ST intervals, Shenwan industries, 163 point-in-time financial indicators; 2010 onward; delisted stocks included |
| [chenditc/investment_data](https://github.com/chenditc/investment_data/releases) (GitHub release `2026-10-27`) | `qlib_bin.tar.gz`, 543 MB | 2007–2009 price history and the CSI 500 index |
| [Kaggle StockChina](https://www.kaggle.com/datasets/stevenchen116/stockchina) (optional) | `original_data/` | independent check of the VWAP returns |

Unpack the first two to `raw_cn/joshuaxql/cn_data` and `raw_cn/chenditc/qlib_bin`. Also save the Hugging Face
`manifest.json` next to `cn_data`; it holds the names of the financial fields.

## How to run

```
pip install -r requirements.txt
```

| Notebook | What it does | Time (RTX 3050 laptop GPU) |
|---|---|---|
| `00_build_data_cn.ipynb` | builds the ranked daily panel (2010–2023, 251 characteristics) | ~35 min |
| `01_train_mse.ipynb` | the paper's two-stage models (Linear, NN1–NN5), 40 rolling windows | ~80 min |
| `01b_train_linear.ipynb` | closed-form linear family and reductions | ~15 min |
| `01c_train_extensions.ipynb` | reduced networks, GMM experts, SVR | ~2.5 h |
| `02_evaluate_baseline.ipynb` | Tables 2, A.3, A.4 and 3 for the two-stage models | ~5 min |
| `03_train_e2e.ipynb` | all E2E runs (sections A–E) | ~45 h in full; `run_real.py` runs a 6 h subset |
| `04_evaluate_all.ipynb` | every table with the fixes from PAPER_ISSUES.md | ~15 min |

Choices and training runs are saved per window, so interrupted runs resume where they stopped.
`run_real.py` runs notebooks 01c and 03 as a script (`bash run_chain.sh C2 A2 B2 E2 D` is the subset reported here,
restarting after a crash). `report/make_figures.py`, `report/make_tables.py`, `report/make_decision.py` and
`report/build_deck.js` regenerate the report's figures, tables and slides from the saved predictions.

`00_build_data_wrds.ipynb` is an alternative US version (CRSP via WRDS); it needs a WRDS account. Its code is
tested on simulated CRSP-shaped files in `checks/test_crsp_build.py`.

## Layout

```
mlm/            the library
  cn.py           China panel: Qlib readers, splicing, point-in-time fundamentals, tradability rules
  data.py         panel storage, rolling windows, synthetic test panel
  models.py       Linear, NN1-NN5, random-Fourier-feature model
  train.py        two-stage training (MSE, SVR and Huber losses), rolling runs
  linear.py       closed-form OLS / ridge / PCR / PLS / LDA from sufficient statistics
  reduce.py       PCA / PLS / LDA / slow-feature reductions in front of any head; decision-aware PCA
  mixture.py      GMM mixture of linear experts
  portfolio.py    decile portfolios, exact cost-aware top decile (eq. 19), tuned MSE-Opt
  e2e.py          E2E training with the exact portfolio layer (eqs. 15, 20, 21) and PPP
  meansd.py       mean-standard-deviation layer with closed-form gradient
  metrics.py      performance, drawdowns, Newey-West tests
checks/         tests of every estimator, layer and gradient, plus diagnostics used in the audit
report/         LaTeX report, slides and the scripts that build them
results/cn, figures/cn   tables and figures from notebook 02
```

## Tests

Run any file in `checks/` with `python checks/<file>.py`. Highlights:
- **Estimators:** the closed-form estimators match scikit-learn (`test_linear.py`).
- **Exact cost-aware solver:** matches `scipy.optimize.linprog` (`test_lp.py`).
- **Portfolio layers:** both are never worse than a generic solver, and their gradients match finite
  differences (`test_e2e_layer.py`, `test_meansd.py`).
- **Paper's own numbers:** its four-asset example reproduces exactly (`check_toy_example.py`).

## Main differences from the paper

All are listed with reasons in [PAPER_ISSUES.md](PAPER_ISSUES.md):
- **Characteristics:** 47 price/volume, 163 point-in-time fundamentals, 10 valuation ratios and 31 Shenwan
  industries, against the paper's 415 characteristics.
- **Tradability:** a stock's first 60 trading days after listing are excluded, and so are stock-days locked at the
  upper price limit on the execution day.
- **Smoothing:** the smoothing term is squared, and is chosen on the validation set.
- **Labels:** a two-day embargo at each split boundary.
- **Training details the paper does not report:** fixed and documented (batch size 10,000, labels winsorized at
  0.5% and 99.5%).
