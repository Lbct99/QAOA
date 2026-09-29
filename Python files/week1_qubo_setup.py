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
import numpy as np
import yfinance as yf

# ---------------------------------------------------------------------------
# 1. Config — tweak this
# ---------------------------------------------------------------------------
TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "PG", "V", "KO", "NVDA", "DIS"]
PERIOD = "2y"              # history window for return/covariance estimation
BUDGET = 3                 # number of stocks to select (B in the QUBO)
RISK_AVERSION_Q = 0.5      # trade-off between return and risk (higher q -> favor return)
PENALTY_P = None           # budget-constraint penalty; auto-set below if None


def fetch_returns_and_cov(tickers, period):
    """Download adjusted close prices and return (mu, Sigma), both annualized."""
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)["Close"]
    data = data[tickers]  # keep column order stable
    daily_returns = data.pct_change().dropna()

    mu = daily_returns.mean().values * 252          # annualized expected return
    sigma = daily_returns.cov().values * 252         # annualized covariance
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
    print(f"Fetching {PERIOD} of price history for: {', '.join(TICKERS)}")
    mu, sigma = fetch_returns_and_cov(TICKERS, PERIOD)

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
    print(f"  Objective value:    {best_val:.6f}")
    print(f"  Feasible (|selected| == budget): {len(selected) == BUDGET}")

    # Save for Week 2 (Ising mapping) and later benchmarking
    np.savez("qubo_week1.npz", Q=Q, mu=mu, sigma=sigma, tickers=TICKERS,
              budget=BUDGET, penalty=penalty_used, best_x=best_x, best_val=best_val)
    print("\nSaved Q, mu, Sigma, and the classical optimum to qubo_week1.npz")


if __name__ == "__main__":
    main()
