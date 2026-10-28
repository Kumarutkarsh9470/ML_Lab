# Inconsistencies in the paper, and how this replication handles them

Paper: Wang, Gao, Harvey, Liu and Tao, *Machine Learning Meets Markowitz* (draft of April 17, 2026).
Each numeric item was recomputed from the paper's own tables; the scripts are in `checks/`. Items are grouped by
what they affect.

- **A**: changes what we implement
- **B**: numbers in the paper that contradict each other
- **C**: claims the tables don't support
- **D**: fairness of the comparison (added controls)
- **E**: typos

## A. Ambiguities that change the implementation

| # | Where | Issue | Our fix |
|---|---|---|---|
| A1 | Tables 2, A.4, A.6 captions | Say stocks are sorted on "predicted one-quarter-ahead returns" "at the start of each test quarter". The text (4.1, 4.2) and the Table 2 caption itself say daily prediction and daily rebalancing. | Daily prediction and daily rebalancing. The quarterly wording looks copied from a monthly-data template. |
| A2 | Eq. (15), (24) | The smoothing term is written ζ‖w‖₂ (an unsquared norm), and ζ is never reported. An unsquared norm is not strongly convex, so the solution can stay at a vertex and gradients can still vanish. | Use ζ‖w‖₂² (strongly convex, so the solution changes smoothly with the predictions). Tune ζ on the validation set and report how sensitive results are to it. |
| A3 | Eq. (20) and Section 4.4 | The cost-aware lower-level problem (eq. 20) has no smoothing term, yet it is still a linear program with kinks, so it needs one too. | Add ζ‖w‖₂² there as well, with the same ζ. |
| A4 | Eq. (19), Table 3 caption | ‖w_t − w_{t−1}‖₁ is called "two-side turnover", while Table 3 says TO-P is "one-sided". | Turnover = ½‖Δw‖₁ and cost = γ × turnover. Table 3 confirms this: MSE-Top's drop from 0 to 10 bps is 14.7 pp = 0.001 × 60.6% × 242. |
| A5 | Section 4.2 vs Table 2 caption | SR is called a "daily average Sharpe ratio" in the text and an "annualized" one in the caption. | Annualized: SR = Avg / SD of annualized numbers, which matches all of Table 2. |
| A6 | Implied by Table 4 | The annualization factor is never stated. The tables imply about 243 days (annual / daily = 242 to 243). | Use the sample's actual average number of trading days per year (about 242 for A-shares). |
| A7 | Table 2 | "H-L" is described as the long-short portfolio, but the numbers equal (H − L)/2. Linear MSE: (34.2 + 59.5)/2 = 46.9. | Report (H − L)/2 to match the paper, and also the raw H − L. |
| A8 | Section 4.1 | Missing-value handling and feature scaling are not described, even though the paper cites Chen & McCoy (2024) on exactly this risk. | Rank each characteristic within each day to [−1, 1] and set missing values to 0, the cross-sectional median (Gu, Kelly and Xiu, 2020). |
| A9 | Appendix A.2 | Not reported: batch size and maximum epochs for the MSE models, number of seeds, and how ζ was tuned. | Batch size 10,000, at most 100 epochs, patience 10, fixed seeds, all documented. A multi-seed ensemble is applied to both MSE and E2E if time allows. |
| A10 | Section 4.1 | Labels are returns at t+2, but no gap is left between the training, validation and test periods, so about 2 days of labels at each boundary are realized after it. | Drop the last 2 trading days before each boundary. |
| A11 | Section 4.4, Table 4 caption | The position cap changes from 10/N (Sections 4.2–4.3) to an absolute w ≤ 0.1. The Table 4 caption adds a "max participation constraint" that is never defined. | Follow eq. (21): w ≤ 0.1, long-only, fully invested, diagonal covariance estimated over 1 year. Treat the "participation constraint" as the cap. |
| A12 | Section 4.4 | PPP must give long-only, capped, fully invested weights, but the paper doesn't say how its network output becomes such weights. | Softmax output with iterative capping at 0.1, using the same NN3. Documented as our choice. |
| A13 | Footnotes 16, 22 | The CSI 500 is called "the broader Chinese A-share market", but it is a mid-cap index. Execution at VWAP ignores China's ±10% (or ±20%) daily price limits and next-day suspensions. A stock locked at its limit for a whole day can't be traded. | Match the paper (CSI 500, VWAP(t+1) to VWAP(t+2)) as the main specification. Robustness checks: an all-share market benchmark, and excluding buys of stocks at their upper limit (and sales of stocks at their lower limit) on day t+1. |
| A14 | Section 4.1, footnote 25 | ST and suspended stocks are excluded, but the paper doesn't say on which date. Newly listed stocks, which in China often rise by the limit for many days, are not mentioned. | Apply the exclusions on decision day t, plus the tradability rules in A15. |
| A15 | Not addressed in the paper (**found in our replication**) | Without tradability rules, much of the long-only return comes from stocks no one could buy. In the 2017Q1 test quarter, 12% of NN3's top decile are stocks listed fewer than 60 days earlier, and 8% are locked at the upper limit for all of day t+1. Those stocks average +9.3% over the target day. Excluding them, the top decile earns −0.04% a day instead of +0.76%, and out-of-sample R² falls from +15% to about −1%. In 2014 (an IPO freeze) the effect is nil. | Drop stock-days in a stock's first 60 trading days, and stock-days locked at the upper limit on t+1 (the buy order fails). The panel and every model are rebuilt with these rules; the earlier results are kept as `*_v1` for comparison. Whether the paper's own numbers carry this bias can't be checked without its data, but its universe description gives no such filter. |

## B. Numbers in the paper that contradict each other

| # | Where | Evidence |
|---|---|---|
| B1 | Section 4.1 vs Table A.1 | The text has 5,962,768 stock-days. The yearly averages in Table A.1 add up to 46,323 stocks per day across the 14 years; at 242 days a year that is about **11.2 million**, 1.9 times as many. |
| B2 | Table 2, NN3 E2E | H-L = 54.9, but (H − L)/2 = (43.6 + 63.1)/2 = **53.35**. Its Avg/SD gives 9.31, not the reported SR of 9.05. Every other cell matches to within rounding. |
| B3 | Table 3, MSE-Opt at 40 bps | The drawdown runs from 2015-06-10 to 2018-02-05, which is **972** calendar days, but Length is reported as 445. Every other Length equals calendar days, inclusive. |
| B4 | Table 4, η = 10, MSE | 0.06% daily implies about 14.6% a year, but 15.4% is reported. This is probably rounding of the daily figure. |

We can't resolve these without the authors' data. Comparisons with the paper use its H, L, return and SD
columns, which agree with each other, rather than the cells above.

## C. Claims the tables don't support

| # | Claim | What the tables show |
|---|---|---|
| C1 | Introduction: two-stage MSE long-short Sharpe ratios are "exceeding 9.0 in some cases" | The largest MSE H-L SR in Table 2 is 8.88. Only E2E variants pass 9 (9.05). |
| C2 | Introduction and 4.4: E2E is "consistently dominating both benchmarks across the entire risk spectrum" | On gross returns, E2E is **not** best at η = 2.5, 5, 7.5 or 10 (Table 4). For example, at η = 10 PPP earns 16.7% and E2E 15.7%. E2E does lead on net returns at every η. |
| C3 | Section 4.3: "about 3–4 percentage points higher annualized return" than MSE-Opt | Table 3 gaps are 4.5, 5.1, 2.9 and 3.7 pp. |
| C4 | Section 4.3: "both TO-P and TO-S fall as models become more conservative" | That is true only for E2E. MSE-Top's TO-P and MSE-Opt's TO-S are constant by construction. |
| C5 | "statistically significant at the 5% level or better", for t = 1.74 and 1.83 | These are significant only one-sided (p = 0.041 and 0.034). Two-sided, p = 0.082 and 0.067. In Table A.5, 1.43 is marked significant at 10%, again one-sided. |
| C6 | Section 4.4: E2E "significantly outperforms MSE and PPP in the higher risk tolerance ranges" | This is consistent with the tables for low η, but the gross-return advantage disappears as η rises (see C2). |

We report gross and net separately at every η, and give both one-sided and two-sided p-values. A
difference counts as significant only if the two-sided test agrees.

## D. Fairness of the comparison (controls we add)

| # | Concern | Control |
|---|---|---|
| D1 | E2E starts from the trained MSE weights and then trains further, so it effectively gets more training. | Check that the MSE model is stopped at its best validation epoch. Also report E2E trained from scratch on a subset of windows. |
| D2 | MSE-Opt feeds raw MSE predictions into eq. (19). MSE predictions are shrunk toward zero, which mechanically changes the cost trade-off. E2E can rescale its own predictions, but MSE-Opt has no such adjustment. | Add an **MSE-Opt (tuned)** benchmark that multiplies γ by a factor κ chosen on the validation set. This is what a practitioner would do, and it isolates the value of learning the signal itself. |
| D3 | Section 4.4 compares fixed η values, but E2E can change its prediction scale and so its effective risk aversion. | Compare on realized frontiers (as in Figure 2) and on volatility-matched returns (as in Table 5), as well as at fixed η. |
| D4 | Results for one seed and one data vintage. | Seed variation on NN3 for both methods, and Newey-West tests on daily return differences. |

## E. Typos (no effect on results)

- Eq. (4): r_{i,i+1} should be r_{i,t+1}.
- Section 3.4: "the differentiation approach introduced in Section 3.2" should be Section 3.3.
- Section 3.2: Σ_t should be Σ̂_t.
- Eq. (20): w_{t−1} in the lower level should be w*_{t−1}.
- Section 4.2: "only-only group" should be "long-only group".
- Footnote 19: "value-weighted-average-price" should be volume-weighted.
- Table 1 caption says "portfolio returns", but the table reports realized utility.
- The Table 3 caption calls Ann SR "in percent".
- Table A.3 labels the linear model "MSE-L".
- The Figure 2 caption omits the green PPP lines.
- Section 4.4: "In the tradition approach" should be "traditional".

## Implementation status

| Item | Where it is handled | Tested |
|---|---|---|
| A1, A4–A6, A8, A10 | `mlm/portfolio.py`, `mlm/metrics.py`, `mlm/data.py` | `checks/test_lp.py`, synthetic runs |
| A2, A3 (squared ζ, tuned on validation) | `mlm/e2e.py` (`zeta_rel` grid, chosen on validation objective) | `checks/test_e2e_layer.py`, `checks/test_e2e_runs.py` |
| A7 (raw H−L) | `decile_returns` adds "H-L raw" | `checks/test_extensions.py` |
| A11, A12 (Section 4.4 setup, PPP) | `mlm/meansd.py` (exact mean–SD layer, capped-softmax PPP), `run_e2e(objective=...)` | `checks/test_meansd.py` |
| A13, A14, A15 (tradability) | `mlm/cn.py` build rules | `checks/check_limit_streaks.py`, `checks/smoke_cn.py` |
| C2, C5, C6, D3 (reporting) | `nw_test` (one- and two-sided p), notebook 04 (gross and net per η, frontiers, volatility-matched tests, Holm adjustment) | `checks/test_extensions.py` |
| D1 (warm start) | `E2EConfig(warm_start=False)`, notebook 03 section D | `checks/test_e2e_runs.py` |
| D2 (tuned MSE-Opt) | `tuned_kappa`, validation predictions saved for every model | `checks/test_extensions.py` |

## Checked and consistent

- **Four-asset example (Section 3.1, Table 1):**
  - The OLS fit is [0.652, 0.927], estimated without an intercept (the paper doesn't say this).
  - The predicted returns match to within ±0.001.
  - All six realized utilities match exactly: 31.97, 32.01, 31.90 and 62.84, 62.53, 62.90.
- **Table 2 averages:** the claims of 38.0 → 42.5 and SR 1.86 → 2.08 are correct.
- **Transaction-cost accounting in Table 3:** matches γ × one-sided turnover × 242 to within 0.2 pp.
- **Drawdown lengths:** equal calendar days, inclusive, except B3.
