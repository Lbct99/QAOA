"""
Week 1 — Classical foundation for the QAOA portfolio optimization project.

Pipeline:
    1. Pull real historical price data for a stock universe (Yahoo Finance).
    2. Compute annualized expected returns and the covariance matrix.
    3. Formulate Markowitz portfolio *selection* as a QUBO:
           minimize   x^T Sigma x  -  q * (mu^T x)  +  P * (sum(x) - B)^2
       where x in {0,1}^n picks a fixed-size, equal-weight subset of B assets.
    4. Solve it exactly by brute force (fine up to ~20 assets) to get the
       ground-truth optimum you'll benchmark QAOA against later.
    5. Print the QUBO matrix Q so you can hand it straight to the
       QUBO -> Ising mapping in Week 2.

Run:
    python week1_qubo_setup.py
"""

import itertools
import os

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# 1. Config — tweak this
# ---------------------------------------------------------------------------
TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "PG", "V", "KO", "NVDA", "DIS"]
START = "2024-09-28"       # pinned window -> identical data (and identical Q) on every run
END = "2026-09-28"         # yfinance treats END as exclusive
PRICE_CACHE = "prices_cache.csv"   # delete this file to force a fresh download
BUDGET = 3                 # number of stocks to select (B in the QUBO)
RISK_AVERSION_Q = 0.5      # trade-off between return and risk (higher q -> favor return)
PENALTY_P = 0.05           # From week1b_diagnostics.py on your real data: minimum working P was
                            # 0.0261, first OK in the sweep at 0.0409 -> 0.05 gives ~2x margin.
                            # This replaces the auto rule below, which was ~31x larger than needed
                            # and unnecessarily inflates the Ising couplings we build next week.
                            # Set to None to fall back to the conservative auto rule.


def load_prices(tickers, start, end, cache):
    """Adjusted close prices, cached to CSV so every run uses identical data."""
    if os.path.exists(cache):
        prices = pd.read_csv(cache, index_col=0, parse_dates=True)
        if list(prices.columns) == list(tickers):
            print(f"Loaded cached prices from {cache}")
            return prices
        print("Cached tickers differ from TICKERS; downloading again")
    prices = yf.download(tickers, start=start, end=end,
                         auto_adjust=True, progress=False)["Close"][tickers]
    prices.to_csv(cache)
    return prices


def returns_and_cov(prices):
    """Annualized expected returns and covariance from daily price data."""
    daily_returns = prices.pct_change().dropna()   # drops any row with a missing price
    mu = daily_returns.mean().values * 252
    sigma = daily_returns.cov().values * 252
    return mu, sigma


def build_qubo(mu, sigma, budget, q, penalty=None):
    """
    Build the QUBO matrix Q (n x n) for:
        minimize  x^T Sigma x - q * mu^T x + penalty * (sum(x) - budget)^2

    Expanding the penalty term and folding everything into a single
    symmetric matrix Q such that the objective equals x^T Q x (x in {0,1}^n).
    """
    n = len(mu)
    if penalty is None:
        # Rule of thumb: penalty should dominate the scale of the objective
        # terms so infeasible solutions are never competitive.
        penalty = 2.0 * (np.abs(sigma).max() + q * np.abs(mu).max())

    Q = sigma.copy()
    # -q * mu^T x  ->  diagonal contribution
    Q -= q * np.diag(mu)

    # penalty * (sum(x) - budget)^2 = penalty * (sum_i x_i^2 + 2*sum_{i<j} x_i x_j
    #                                             - 2*budget*sum_i x_i + budget^2)
    # x_i^2 = x_i for binary x, so it folds into the diagonal as penalty*(1 - 2*budget).
    # Off-diagonal entries each get +penalty; the constant budget^2 term is dropped
    # since it doesn't affect which x minimizes the objective.
    Q += penalty * (np.ones((n, n)) - 2 * budget * np.eye(n))

    return Q, penalty


def qubo_objective(x, Q):
    return x @ Q @ x


def brute_force_optimum(Q, n):
    """Exhaustive search over all 2^n bitstrings — ground truth for benchmarking."""
    best_x, best_val = None, np.inf
    for bits in itertools.product([0, 1], repeat=n):
        x = np.array(bits)
        val = qubo_objective(x, Q)
        if val < best_val:
            best_val, best_x = val, x
    return best_x, best_val


def main():
    print(f"Loading prices for: {', '.join(TICKERS)}")
    prices = load_prices(TICKERS, START, END, PRICE_CACHE)
    print(f"Prices: {prices.shape[0]} trading days, "
          f"{prices.index[0].date()} -> {prices.index[-1].date()}")
    missing = prices.isna().sum()
    if missing.any():
        print("WARNING: missing prices (rows with gaps are dropped):")
        print(missing[missing > 0])
    mu, sigma = returns_and_cov(prices)

    print("\nAnnualized expected returns:")
    for t, m in zip(TICKERS, mu):
        print(f"  {t:>6s}: {m:+.4f}")

    Q, penalty_used = build_qubo(mu, sigma, BUDGET, RISK_AVERSION_Q, PENALTY_P)
    print(f"\nQUBO built. n = {len(TICKERS)}, budget B = {BUDGET}, "
          f"risk-aversion q = {RISK_AVERSION_Q}, penalty P = {penalty_used:.4f}")

    print("\nQ matrix:")
    np.set_printoptions(precision=3, suppress=True)
    print(Q)

    best_x, best_val = brute_force_optimum(Q, len(TICKERS))
    selected = [t for t, xi in zip(TICKERS, best_x) if xi == 1]
    print(f"\nClassical optimum (brute force):")
    print(f"  Selected portfolio: {selected}")
    print(f"  QUBO value:          {best_val:.6f}  (excludes the constant P*B^2)")
    print(f"  Markowitz objective: {best_val + penalty_used * BUDGET**2:.6f}  (x^T Sigma x - q * mu^T x)")
    print(f"  Feasible (|selected| == budget): {len(selected) == BUDGET}")

    # Save for Week 2 (Ising mapping) and later benchmarking
    np.savez("qubo_week1.npz", Q=Q, mu=mu, sigma=sigma, tickers=TICKERS,
              budget=BUDGET, penalty=penalty_used, best_x=best_x, best_val=best_val,
              q=RISK_AVERSION_Q)
    print("\nSaved Q, mu, Sigma, and the classical optimum to qubo_week1.npz")


if __name__ == "__main__":
    main()
