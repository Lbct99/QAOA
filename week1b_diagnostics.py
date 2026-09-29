"""
Week 1b — Diagnostics on the QUBO saved by week1_qubo_setup.py (no network needed).

Answers three questions before we move to the Ising mapping:
    1. What is the *true* Markowitz objective of the classical optimum?
    2. How close are the runner-up portfolios? (This decides how hard it will be
       for a noisy QAOA run to "find the optimum" rather than a near-optimum.)
    3. How small can the penalty P be while the QUBO ground state is still the
       exact constrained optimum? (A penalty far above this compresses the
       energy differences QAOA has to resolve.)

Run:
    python week1b_diagnostics.py
"""

import numpy as np

from week1_qubo_setup import RISK_AVERSION_Q


def all_bitstrings(n):
    """Row k is the binary expansion of k; column i is asset i."""
    return ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(float)


def main():
    d = np.load("qubo_week1.npz")
    Q, mu, sigma = d["Q"], d["mu"], d["sigma"]
    tickers = [str(t) for t in d["tickers"]]
    budget, p0 = int(d["budget"]), float(d["penalty"])
    q = float(d["q"]) if "q" in d.files else RISK_AVERSION_Q
    n = len(tickers)

    X = all_bitstrings(n)
    size = X.sum(axis=1)
    feasible = size == budget
    f = np.einsum("ki,ij,kj->k", X, sigma, X) - q * (X @ mu)   # Markowitz objective, no penalty

    # Consistency check: the saved Q must reproduce f + penalty term (minus constant P*B^2)
    qval = np.einsum("ki,ij,kj->k", X, Q, X)
    assert np.allclose(qval, f + p0 * (size - budget) ** 2 - p0 * budget**2), "Q inconsistent with sigma/mu"
    assert np.array_equal(X[np.argmin(qval)], d["best_x"]), "saved optimum differs from recomputed one"
    print("Consistency checks passed (Q matches sigma, mu, q, P and the saved optimum).\n")

    # 1 + 2: true objective and runner-ups
    idx = np.flatnonzero(feasible)
    order = idx[np.argsort(f[idx])]
    best = order[0]
    print(f"Top 5 of {len(idx)} feasible portfolios (Markowitz objective, lower is better):")
    for rank, k in enumerate(order[:5], 1):
        names = [t for t, xi in zip(tickers, X[k]) if xi]
        print(f"  {rank}. {names}  f = {f[k]:+.4f}   gap to best = {f[k] - f[best]:.4f}")

    w = X[best] / budget
    print(f"\nBest portfolio at equal weight: expected return {mu @ w:+.1%}, "
          f"volatility {np.sqrt(w @ sigma @ w):.1%}  (annualized, in-sample)")

    # Energy scales
    spread = f[feasible].max() - f[feasible].min()
    print(f"\nObjective spread across feasible portfolios: {spread:.4f}")
    print(f"Current penalty P: {p0:.4f}  (P / spread = {p0 / spread:.1f})")

    # 3: penalty sweep (exact for this instance, found by brute force)
    ps = np.geomspace(p0 / 100, p0 * 2, 60)
    ok = np.array([np.argmin(f + p * (size - budget) ** 2) == best for p in ps])
    start = next((i for i in range(len(ps)) if ok[i:].all()), None)

    print("\nPenalty sweep (does the QUBO ground state equal the constrained optimum?):")
    for i in range(0, len(ps), 6):
        k = np.argmin(f + ps[i] * (size - budget) ** 2)
        print(f"  P = {ps[i]:.4f}  ground state holds {int(size[k])} assets  "
              f"{'OK' if ok[i] else 'WRONG'}")
    if start is None:
        print("No P in the sweep range works — check the inputs.")
    else:
        print(f"\nSmallest P in the sweep that works for all larger P: {ps[start]:.4f}")
        print(f"Current P is {p0 / ps[start]:.1f}x larger than that.")
        print("Heuristic: use roughly 1.5-2x this value as a safety margin, not the exact minimum.")
        print("Note: this minimum is specific to this dataset; it is not a general bound.")


if __name__ == "__main__":
    main()
