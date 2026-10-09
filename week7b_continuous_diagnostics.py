"""
Week 5b -- Diagnostics for the continuous-weight model.

Two checks, mirroring week1b_diagnostics.py and hw6_diagnostics.py:
    1. Penalty tuning: is the auto-heuristic penalty far larger than needed?
    2. Quantify the improvement: how much better is the continuous-weight
       optimum than the equal-weight optimum, in TRUE Markowitz terms
       (y^T Sigma y - q*mu^T y -- no penalty, since both are feasible,
       this is an apples-to-apples comparison of the actual trade-off).

Run:
    python week7b_continuous_diagnostics.py
"""

import itertools

import numpy as np


def markowitz_value(y, mu, sigma, q):
    return y @ sigma @ y - q * (mu @ y)


def all_allocations(n, k_bits):
    """Every y in {0,...,2^k_bits-1}^n, plus the flattened binary vector for each."""
    nb = n * k_bits
    idx = np.arange(2**nb)[:, None]
    bits = ((idx >> np.arange(nb)) & 1).astype(float)   # (2^nb, nb)
    weights_per_bit = np.array([2**k for k in range(k_bits)], dtype=float)
    C = np.zeros((n, nb))
    for i in range(n):
        C[i, i * k_bits:(i + 1) * k_bits] = weights_per_bit
    Y = bits @ C.T   # (2^nb, n) -- Y[s] is the allocation vector for state s
    return bits, Y


def main():
    d = np.load("continuous_weights_week5.npz")
    mu, sigma, tickers = d["mu"], d["sigma"], [str(t) for t in d["tickers"]]
    n_total, k_bits = int(d["n_total"]), int(d["k_bits"])
    q = 0.5  # RISK_AVERSION_Q from week7_continuous_weights.py
    n = len(tickers)

    bits, Y = all_allocations(n, k_bits)
    size = Y.sum(axis=1)
    feasible = np.isclose(size, n_total)
    f = np.array([markowitz_value(y, mu, sigma, q) for y in Y])

    # --- 1. Penalty tuning ---
    p_grid = np.geomspace(1e-4, 2.0, 200)
    min_p, first_ok = None, None
    for p in p_grid:
        k = np.argmin(f + p * (size - n_total) ** 2)
        if feasible[k]:
            if min_p is None:
                first_ok = p
            min_p = p
            break
    for p in p_grid:
        k = np.argmin(f + p * (size - n_total) ** 2)
        if feasible[k]:
            min_p = p
            break
    print(f"Minimum feasible penalty: {min_p:.4f}  (auto-heuristic from the main "
          f"script used a larger value -- check your run's 'Penalty used' line)")
    print(f"Recommended: ~2x margin -> {2*min_p:.4f}")

    # --- 2. Top allocations and the equal-weight comparison ---
    idx_order = np.flatnonzero(feasible)[np.argsort(f[np.flatnonzero(feasible)])]
    best = f[idx_order[0]]
    print(f"\nTop 5 of {feasible.sum()} feasible allocations "
          f"({n} assets, {k_bits} bits/asset, {n_total} total units):")
    for rank, k in enumerate(idx_order[:5], 1):
        alloc = [(t, int(y)) for t, y in zip(tickers, Y[k]) if y > 0]
        print(f"  {rank}. {alloc}  gap to best = {f[k] - best:.4f}")

    # Equal-weight comparison: find the equal-weight (K=1-equivalent) optimum
    # within this same allocation space -- i.e. the best y with every nonzero
    # entry equal to 1 (one unit each, exactly n_total assets selected).
    equal_weight_mask = feasible & np.all((Y == 0) | (Y == 1), axis=1)
    if equal_weight_mask.any():
        ew_idx = np.flatnonzero(equal_weight_mask)[np.argmin(f[equal_weight_mask])]
        ew_val = f[ew_idx]
        ew_alloc = [t for t, y in zip(tickers, Y[ew_idx]) if y == 1]
        print(f"\nBest EQUAL-WEIGHT allocation within this same space: {ew_alloc}  "
              f"value = {ew_val:.4f}")
        print(f"Best CONTINUOUS (unequal) allocation: value = {best:.4f}")
        improvement = ew_val - best
        rel = improvement / abs(ew_val) * 100 if ew_val != 0 else float("nan")
        print(f"Improvement from unequal weighting: {improvement:.4f} "
              f"({rel:.1f}% of equal-weight's objective magnitude)")
        print("If this improvement is tiny relative to the gaps in the Top-5 table "
              "above, the continuous model isn't meaningfully better here -- the "
              "extra qubits wouldn't be worth it. If it's comparable to or larger "
              "than those gaps, the unequal weighting is a real, worthwhile gain.")
    else:
        print("\nNo valid equal-weight allocation found in this space to compare against.")


if __name__ == "__main__":
    main()
