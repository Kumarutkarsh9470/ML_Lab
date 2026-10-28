"""Cross-check numbers reported in the paper's text and tables."""
import numpy as np

# Table 2 (equal-weighted), per model: (H avg, H SR, L avg, H-L avg, H-L SD, H-L SR), MSE then E2E
t2 = {
    "Linear": [(34.2, 1.73, -59.5, 46.9, 6.2, 7.55), (41.0, 2.00, -54.9, 47.9, 6.6, 7.27)],
    "NN1": [(39.4, 1.93, -68.8, 54.1, 6.1, 8.87), (43.4, 2.12, -61.0, 52.2, 5.9, 8.85)],
    "NN2": [(37.2, 1.81, -66.5, 51.8, 6.0, 8.58), (41.9, 2.06, -62.5, 52.2, 5.8, 8.97)],
    "NN3": [(40.0, 1.94, -66.2, 53.1, 6.0, 8.88), (43.6, 2.14, -63.1, 54.9, 5.9, 9.05)],
    "NN4": [(38.3, 1.84, -68.9, 53.6, 6.2, 8.61), (41.7, 2.04, -58.8, 50.2, 5.6, 8.98)],
    "NN5": [(38.9, 1.90, -67.0, 52.9, 6.1, 8.65), (43.2, 2.10, -59.1, 51.2, 6.0, 8.58)],
}
print("H-L vs (H-L)/2 and H-L:")
for m, rows in t2.items():
    for lab, (h, _, l, hl, sd, sr) in zip(["MSE", "E2E"], rows):
        print(f"  {m:6s} {lab}: reported {hl:5.1f} | (H-L)/2 = {(h - l) / 2:5.2f} | H-L = {h - l:5.1f} | Avg/SD = {hl / sd:5.2f} vs SR {sr}")
mse_h = np.mean([r[0][0] for r in t2.values()]); e2e_h = np.mean([r[1][0] for r in t2.values()])
mse_sr = np.mean([r[0][1] for r in t2.values()]); e2e_sr = np.mean([r[1][1] for r in t2.values()])
print(f"avg H return MSE {mse_h:.2f} E2E {e2e_h:.2f} (+{e2e_h - mse_h:.2f} pp, {e2e_h / mse_h - 1:.1%}); paper: 38.0 -> 42.5, 4.45 pp, ~12%")
print(f"avg H SR MSE {mse_sr:.3f} E2E {e2e_sr:.3f}; paper: 1.86 -> 2.08")
print("max MSE H-L SR:", max(r[0][5] for r in t2.values()), "| paper intro: 'Sharpe ratios exceeding 9.0' for the MSE approach")

# Table A.1 stock counts vs the 5,962,768 observations in Section 4.1
a1 = [1841, 2169, 2393, 2462, 2527, 2736, 2932, 3345, 3612, 3741, 4015, 4485, 4869, 5196]
print(f"Table A.1 sum of avg stocks/day = {sum(a1):,}; x 242 days = {sum(a1) * 242:,.0f} stock-days vs 5,962,768 in text "
      f"(ratio {sum(a1) * 242 / 5962768:.2f})")

# Annualization: Table 4 daily vs annual
for dr, dv, ar, av in [(0.21, 1.52, 51.0, 23.7), (0.06, 1.16, 15.4, 18.1), (0.25, 1.59, 60.4, 24.8)]:
    print(f"Table 4: ann ret / daily ret = {ar / dr:.0f}, (ann vol / daily vol)^2 = {(av / dv) ** 2:.0f}")

# Transaction-cost accounting, Table 3 MSE-Top (TO 60.6% one-sided)
for tcr, ret in [(10, 25.3), (20, 10.6), (30, -4.2), (40, -18.9)]:
    print(f"Table 3 MSE-Top {tcr}bps: drop {40.0 - ret:.1f} pp vs gamma*TO*242 = {tcr / 1e4 * 0.606 * 242 * 100:.1f} pp")

# Table 3 E2E vs MSE-Opt gaps, text says 'about 3-4 pp'
opt = {10: 29.2, 20: 24.0, 30: 20.1, 40: 18.1}; e2e = {10: 33.7, 20: 29.1, 30: 23.0, 40: 21.8}
print("E2E - MSE-Opt (pp):", {k: round(e2e[k] - opt[k], 1) for k in opt})

# Table 3 MDD length is calendar days inclusive
import datetime as dt
for b, e, L in [("2015-06-19", "2015-07-06", 18), ("2015-06-10", "2015-07-06", 27), ("2016-11-18", "2022-04-26", 1986),
                ("2016-11-18", "2023-10-19", 2527), ("2016-06-23", "2023-12-22", 2739), ("2015-06-10", "2018-02-05", 445)]:
    d = (dt.date.fromisoformat(e) - dt.date.fromisoformat(b)).days + 1
    print(f"MDD {b}..{e}: reported length {L}, calendar days incl. {d}")

# One-sided vs two-sided significance
from scipy.stats import norm
for t in [1.74, 1.83, 1.43, 1.52, 1.66, 0.88]:
    print(f"t={t}: one-sided p={1 - norm.cdf(t):.3f}, two-sided p={2 * (1 - norm.cdf(t)):.3f}")

# Table 4: does E2E dominate at every eta? (gross ann ret, gross SR; net ann ret, net SR) for MSE, PPP, E2E
t4 = {0.1: [(51.0, 2.15, 25.7, 1.08), (53.0, 2.18, 26.4, 1.14), (60.4, 2.44, 32.7, 1.34)],
      0.25: [(49.9, 2.21, 25.9, 1.15), (53.6, 2.41, 27.4, 1.29), (56.1, 2.42, 31.3, 1.37)],
      0.5: [(46.4, 2.17, 24.1, 1.13), (45.6, 2.22, 23.6, 1.16), (52.0, 2.40, 29.2, 1.38)],
      0.75: [(44.3, 2.13, 23.3, 1.12), (40.5, 2.02, 21.0, 1.06), (47.2, 2.28, 27.8, 1.36)],
      1.0: [(42.2, 2.08, 22.6, 1.11), (37.1, 1.88, 19.5, 1.00), (44.0, 2.19, 27.5, 1.38)],
      2.5: [(32.3, 1.70, 19.0, 1.00), (27.6, 1.46, 14.8, 0.79), (31.9, 1.69, 20.8, 1.11)],
      5.0: [(23.1, 1.25, 14.9, 0.81), (22.1, 1.19, 13.0, 0.70), (22.9, 1.24, 15.8, 0.86)],
      7.5: [(18.3, 1.00, 12.6, 0.69), (18.9, 1.03, 11.6, 0.63), (18.3, 1.01, 13.3, 0.73)],
      10.0: [(15.4, 0.85, 11.1, 0.61), (16.7, 0.91, 10.5, 0.58), (15.7, 0.87, 11.8, 0.65)]}
names = ["gross ret", "gross SR", "net ret", "net SR"]
for eta, (m, p, e) in t4.items():
    lost = [n for k, n in enumerate(names) if e[k] < max(m[k], p[k])]
    if lost:
        print(f"Table 4 eta={eta}: E2E not best on {lost}")
