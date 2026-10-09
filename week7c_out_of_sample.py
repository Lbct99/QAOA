"""
Week 5c -- Does unequal weighting survive out of sample?

K=2 contains every K=1 allocation as a special case, so its IN-SAMPLE
objective can never be worse than K=1's. A positive in-sample improvement
is therefore guaranteed and says little by itself. The real question is
whether the unequal allocation does better on data the optimizer never saw.

Method: expanding-window splits of the cached daily prices. For each split:
  1. estimate mu and Sigma on the first part (train)
  2. pick the best allocation for K=1 and K=2 by exhaustive search over all
     feasible allocations (the same optimum the QUBO ground state gives once
     the penalty is large enough -- see week7b_continuous_diagnostics.py)
  3. measure the realized annualized return, volatility and return/vol on the
     later part (test), which the optimizer never saw
Baseline: plain 1/N across all 6 stocks.

Caveats, printed again at the end:
  - the splits overlap, so they are not independent samples
  - the 6-stock universe was chosen after looking at full-sample results
  - 2 years of daily data is short; differences of a few percent are noise

Run:
    python week7c_out_of_sample.py
"""

import itertools

import numpy as np
import pandas as pd

PRICE_CACHE = "prices_cache.csv"
TICKERS = ["AAPL", "MSFT", "JPM", "JNJ", "KO", "NVDA"]   # keep in sync with week4_six_stock_pipeline.py
Q_RISK = 0.5            # same RISK_AVERSION_Q as the rest of the pipeline
N_TOTAL = 3
K_LIST = [1, 2]
SPLITS = [0.5, 0.6, 0.7, 0.8]    # fraction of history used for training


def feasible_allocations(n, k_bits, n_total):
    top = 2**k_bits - 1
    for y in itertools.product(range(top + 1), repeat=n):
        if sum(y) == n_total:
            yield np.array(y, dtype=float)


def best_allocation(mu, sigma, k_bits):
    best, best_f = None, np.inf
    for y in feasible_allocations(len(mu), k_bits, N_TOTAL):
        f = y @ sigma @ y - Q_RISK * (mu @ y)
        if f < best_f:
            best, best_f = y, f
    return best


def realized(w, test_returns):
    p = test_returns @ w
    ret = p.mean() * 252
    vol = p.std(ddof=1) * np.sqrt(252)
    return ret, vol, ret / vol


def describe(w):
    return ", ".join(f"{t} {x:.2f}" for t, x in zip(TICKERS, w) if x > 0)


def main():
    prices = pd.read_csv(PRICE_CACHE, index_col=0, parse_dates=True)[TICKERS]
    rets = prices.pct_change().dropna()
    n = len(TICKERS)
    print(f"{len(rets)} daily returns, {rets.index[0].date()} -> {rets.index[-1].date()}\n")

    names = ["1/N (all 6)"] + [f"K={k}" for k in K_LIST]
    ratios = {name: [] for name in names}

    for frac in SPLITS:
        cut = int(len(rets) * frac)
        train, test = rets.iloc[:cut], rets.iloc[cut:]
        mu, sigma = train.mean().values * 252, train.cov().values * 252

        candidates = {"1/N (all 6)": np.ones(n) / n}
        for k in K_LIST:
            candidates[f"K={k}"] = best_allocation(mu, sigma, k) / N_TOTAL

        print(f"Train {train.index[0].date()} -> {train.index[-1].date()} "
              f"| test {test.index[0].date()} -> {test.index[-1].date()} ({len(test)} days)")
        for name, w in candidates.items():
            ret, vol, ratio = realized(w, test.values)
            ratios[name].append(ratio)
            print(f"  {name:12s} test return {ret:+7.1%}  vol {vol:6.1%}  "
                  f"return/vol {ratio:+5.2f}   [{describe(w)}]")
        print()

    print("Mean test return/vol across splits:")
    for name in names:
        print(f"  {name:12s} {np.mean(ratios[name]):+5.2f}")
    k_hi, k_lo = f"K={K_LIST[-1]}", f"K={K_LIST[0]}"
    wins = sum(a > b for a, b in zip(ratios[k_hi], ratios[k_lo]))
    print(f"\n{k_hi} beat {k_lo} out of sample in {wins} of {len(SPLITS)} splits.")
    print("\nCaveats: splits overlap (not independent); the universe was picked after seeing")
    print("full-sample results; ~2 years of data is short. Treat this as a sanity check on")
    print("whether the in-sample gain survives, not as proof either way.")


if __name__ == "__main__":
    main()
