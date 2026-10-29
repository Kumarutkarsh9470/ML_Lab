# Machine Learning Meets Markowitz

Replication and extension of Wang, Gao, Harvey, Liu and Tao (2026), *Machine Learning Meets Markowitz*
(SSRN 5947774), on daily China A-shares from 2010 to 2023, using only free data. By Kumar Utkarsh and
Pratham Saluja, Indian Institute of Technology Guwahati.

The paper trains return forecasts for the portfolio they feed rather than for mean squared error, by placing
the portfolio optimizer inside the network (end-to-end, E2E, learning). This repository contains:

- a free-data rebuild of the paper's setting: VWAP execution, ST and suspension filters, Shenwan industries,
  point-in-time fundamentals, tradability rules and 40 rolling windows
- exact differentiable portfolio layers for the top-decile, transaction-cost and mean-SD problems
- closed-form linear models and dimension reduction: OLS, ridge, PCR, PLS and LDA
- decision-aware PCA, reduced networks, GMM mixtures of experts and SVR

## Results

Out of sample, 2014 to 2023 (40 test quarters). Long-only, equal-weight top decile, before costs,
market-adjusted against the CSI 500 and annualized. Each E2E model is compared with the two-stage
model it starts from; *t* is the Newey-West t-statistic of the difference.

| Model | Rank IC | Return, % a year | Sharpe | *t* |
|---|---|---|---|---|
| Linear | 0.067 to 0.055 | 31.1 to 31.8 | 1.36 to 1.37 | 0.42 |
| NN3 | 0.061 to 0.056 | 29.0 to 31.5 | 1.23 to 1.35 | 1.69 |
| Ridge | 0.069 to 0.053 | 27.8 to 31.9 | 1.23 to 1.36 | 2.17 |
| PCA20-Linear, decision-aware | 0.056 to 0.048 | 21.8 to 30.4 | 1.00 to 1.36 | 4.96 |

E2E training lowers statistical accuracy and raises portfolio returns. With 20 bps costs, NN3 trained with
costs earns 24.7% a year net with 7% daily turnover, against 15.2% and 28% for the MSE top decile.

## Data

All sources are free and need no registration. The code expects them under `D:/ML_Markowitz_data/`; change
the paths at the top of each notebook to use another location.

| Source | Contents |
|---|---|
| [joshuaxql/qlib_data](https://huggingface.co/datasets/joshuaxql/qlib_data) | daily prices, VWAP, market cap, price limits, ST intervals, Shenwan industries, 163 point-in-time financial indicators |
| [chenditc/investment_data](https://github.com/chenditc/investment_data/releases) | 2007 to 2009 price history and the CSI 500 index |
| [Kaggle StockChina](https://www.kaggle.com/datasets/stevenchen116/stockchina) | optional independent check of VWAP returns |

Unpack the first two to `raw_cn/joshuaxql/cn_data` and `raw_cn/chenditc/qlib_bin`, with the Hugging Face
`manifest.json` saved next to `cn_data`.

## Usage

```
pip install -r requirements.txt
```

| Step | What it does | Time |
|---|---|---|
| `00_build_data_cn.ipynb` | builds the daily panel (2,953 stocks a day, 251 characteristics) | 35 min |
| `01_train_mse.ipynb` | two-stage Linear and NN1 to NN5 | 80 min |
| `01b_train_linear.ipynb` | closed-form linear models and reductions | 15 min |
| `01c_train_extensions.ipynb` | reduced networks, mixtures of experts, SVR | 2.5 h |
| `02_evaluate_baseline.ipynb` | tables for the two-stage models | 5 min |
| `03_train_e2e.ipynb` | end-to-end training | 6 h to 45 h |
| `04_evaluate_all.ipynb` | all tables | 15 min |

Times are for an RTX 3050 laptop GPU. Every run saves per window and resumes where it stopped.
`run_real.py` runs notebooks 01c and 03 as a script, and `run_chain.sh` restarts it after a crash:

```
bash run_chain.sh C2 A2 B2 E2 D
```

## Layout

| Module | Contents |
|---|---|
| `mlm/cn.py` | China panel: Qlib readers, point-in-time fundamentals, tradability rules |
| `mlm/data.py` | panel storage and rolling windows |
| `mlm/models.py` | Linear, NN1 to NN5, random-Fourier-feature model |
| `mlm/train.py` | two-stage training with MSE, SVR or Huber loss |
| `mlm/linear.py` | closed-form OLS, ridge, PCR, PLS and LDA |
| `mlm/reduce.py` | PCA, PLS, LDA and slow-feature reductions; decision-aware PCA |
| `mlm/mixture.py` | GMM mixture of linear experts |
| `mlm/portfolio.py` | decile portfolios and the exact cost-aware top decile |
| `mlm/e2e.py` | end-to-end training with the exact portfolio layers |
| `mlm/meansd.py` | mean-SD portfolio layer with its exact gradient |
| `mlm/metrics.py` | performance, drawdowns and Newey-West tests |
