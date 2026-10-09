"""
Week 5 -- Continuous-weight portfolio extension.

Generalizes the equal-weight selection model (x_i in {0,1}, exactly B assets
chosen, each worth 1/B of the portfolio) to variable-sized positions:

    y_i in {0, ..., 2^K - 1}   -- "units" of asset i, via K-bit binary expansion
    y_i = sum_{k=0}^{K-1} 2^k * b_{i,k}
    w_i = y_i / N_total         -- asset i's portfolio weight
    constraint: sum_i y_i = N_total  (same budget constraint as before, generalized)

With K=1, y_i in {0,1} and this reduces EXACTLY to the original equal-weight
model -- used below as a correctness check. With K>1, assets can hold
different-sized positions: qubits per asset = K, total qubits = n * K.

Run:
    python week7_continuous_weights.py
"""

import itertools

import numpy as np

from week1_qubo_setup import PRICE_CACHE, returns_and_cov
from week4_six_stock_pipeline import load_subset_prices, HW_TICKERS


K_BITS = 2              # bits per asset; K=1 reproduces the original model exactly
N_TOTAL = 3              # total allocation units across all assets (same semantic as old BUDGET)
RISK_AVERSION_Q = 0.5
TICKERS = HW_TICKERS     # reuse the 6-stock universe


def build_continuous_qubo(mu, sigma, n_total, k_bits, q, penalty=None):
    """
    QUBO over n*k_bits binary variables b_{i,k} (flattened: index = i*k_bits + k).

    IMPORTANT convention, matching week1_qubo_setup.py exactly: the objective
    is built in raw allocation units y_i, NOT normalized weights w_i = y_i/n_total
    -- exactly like the original model used x_i directly, not x_i/budget. This
    is what makes K=1 reproduce the original Q exactly (verified below). Weights
    are computed only when decoding/reporting a solution, same as
    week1b_diagnostics.py did ("w = X[best] / budget"), never inside the QUBO
    itself. (An earlier version of this function normalized inside the
    objective -- that silently changes the effective risk-aversion trade-off
    relative to every result you've already validated on hardware, so it's
    deliberately NOT done here.)

    Objective: y^T Sigma y - q * mu^T y + penalty * (sum_i y_i - n_total)^2
    where y_i = sum_k 2^k b_{i,k}.

    Expand in terms of the flattened binary vector b (length n*k_bits):
        y_i = sum_k c_k b_{i,k},  c_k = 2^k
        y = C b   where C maps b -> y (block-diagonal, c_k per asset)
    So y^T Sigma y = b^T C^T Sigma C b -- still quadratic in b. Same pattern
    for the linear and penalty terms.
    """
    n = len(mu)
    nb = n * k_bits
    weights_per_bit = np.array([2**k for k in range(k_bits)], dtype=float)  # c_k

    # C: (n, nb) matrix mapping b -> y (y_i = row i of C dot b)
    C = np.zeros((n, nb))
    for i in range(n):
        C[i, i * k_bits:(i + 1) * k_bits] = weights_per_bit

    if penalty is None:
        penalty = 2.0 * (np.abs(sigma).max() + q * np.abs(mu).max())

    # Risk term: y^T Sigma y = b^T (C^T Sigma C) b
    Q = C.T @ sigma @ C

    # Return term: -q * mu^T y = -q * (mu^T C) b
    # (linear in b -> folds into the diagonal, since b_i^2 = b_i for binary b)
    linear_return = -q * (mu @ C)
    Q += np.diag(linear_return)

    # Penalty term: P * (sum_i y_i - n_total)^2 = P * (1^T C b - n_total)^2
    a = C.sum(axis=0)  # a_j = c_k for bit j's weight within its asset
    Q += penalty * (np.outer(a, a) - 2 * n_total * np.diag(a))
    # (the + n_total^2 constant term is dropped, as in week1 -- doesn't affect argmin)

    return Q, penalty, C


def brute_force_continuous(Q, nb):
    best_b, best_val = None, np.inf
    for bits in itertools.product([0, 1], repeat=nb):
        b = np.array(bits, dtype=float)
        val = b @ Q @ b
        if val < best_val:
            best_val, best_b = val, b
    return best_b, best_val


def decode_weights(b, C, n_total, tickers, k_bits):
    y = C @ b
    w = y / n_total
    print("  Allocation:")
    for t, yi, wi in zip(tickers, y, w):
        bar = "#" * int(round(wi * 20))
        print(f"    {t:>6s}: {int(yi)}/{n_total} units -> weight {wi:.3f}  {bar}")
    return w


def main():
    prices = load_subset_prices(TICKERS, PRICE_CACHE)
    mu, sigma = returns_and_cov(prices)
    n = len(TICKERS)

    # --- Correctness check: K=1 must reproduce the original model exactly ---
    print("=== Sanity check: K=1 reproduces the original equal-weight QUBO ===")
    from week1_qubo_setup import build_qubo as build_qubo_orig
    Q_orig, p_orig = build_qubo_orig(mu, sigma, N_TOTAL, RISK_AVERSION_Q, None)
    Q_new, p_new, C1 = build_continuous_qubo(mu, sigma, N_TOTAL, 1, RISK_AVERSION_Q, p_orig)
    max_diff = np.abs(Q_orig - Q_new).max()
    print(f"Max |Q_original - Q_continuous(K=1)| = {max_diff:.2e}  "
          f"({'PASS' if max_diff < 1e-9 else 'FAIL -- formulas disagree'})")

    # --- Now the actual extension: K=2 ---
    print(f"\n=== Continuous-weight model: K={K_BITS} bits/asset, "
          f"{n}*{K_BITS}={n*K_BITS} qubits ===")
    Q, penalty, C = build_continuous_qubo(mu, sigma, N_TOTAL, K_BITS, RISK_AVERSION_Q)
    print(f"Penalty used: {penalty:.4f}")

    nb = n * K_BITS
    best_b, best_val = brute_force_continuous(Q, nb)
    y = C @ best_b
    feasible = abs(y.sum() - N_TOTAL) < 1e-6
    print(f"\nClassical optimum (brute force over {2**nb} states):")
    print(f"  QUBO value: {best_val:.6f}  feasible: {feasible}")
    decode_weights(best_b, C, N_TOTAL, TICKERS, K_BITS)

    # Compare against the old equal-weight optimum for context
    print(f"\nFor comparison, the equal-weight (K=1) optimum was: JNJ, KO, NVDA "
          f"(each 1/3 = 0.333)")

    np.savez("continuous_weights_week5.npz", Q=Q, C=C, mu=mu, sigma=sigma,
              tickers=TICKERS, n_total=N_TOTAL, k_bits=K_BITS, penalty=penalty,
              best_b=best_b, best_val=best_val)
    print("\nSaved to continuous_weights_week5.npz")


if __name__ == "__main__":
    main()
