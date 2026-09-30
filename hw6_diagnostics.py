"""
Diagnostics for the 6-stock problem (hw6_pipeline.npz from week4_six_stock_pipeline.py).

Checks whether QAOA's low P(optimal) is explained by near-degenerate portfolios
(several feasible options within a hair of the true optimum -- a hard, low-gap
landscape for any optimizer/sampler to resolve sharply) rather than a problem
with the circuit or optimizer themselves.

Run:
    python hw6_diagnostics.py
"""

import numpy as np


def main():
    d = np.load("hw6_pipeline.npz")
    Q, tickers = d["Q"], [str(t) for t in d["tickers"]]
    budget = int(d["budget"])
    n = len(tickers)

    idx = np.arange(2**n)[:, None]
    X = ((idx >> np.arange(n)) & 1).astype(float)
    size = X.sum(axis=1)
    qval = np.einsum("ki,ij,kj->k", X, Q, X)

    feasible = size == budget
    order = np.flatnonzero(feasible)[np.argsort(qval[np.flatnonzero(feasible)])]

    print(f"Top portfolios out of {feasible.sum()} feasible ({n} assets, budget {budget}):")
    best = qval[order[0]]
    for rank, k in enumerate(order[:6], 1):
        names = [t for t, xi in zip(tickers, X[k]) if xi]
        print(f"  {rank}. {names}  gap to best = {qval[k] - best:.4f}")

    spread = qval[feasible].max() - qval[feasible].min()
    gap_to_2nd = qval[order[1]] - qval[order[0]]
    print(f"\nGap from best to 2nd-best: {gap_to_2nd:.4f}")
    print(f"Full feasible spread: {spread:.4f}")
    print(f"Gap / spread ratio: {gap_to_2nd / spread:.1%}  "
          f"(small -> near-degenerate top, genuinely hard to resolve sharply)")


if __name__ == "__main__":
    main()
