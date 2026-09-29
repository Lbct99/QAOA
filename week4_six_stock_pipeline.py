"""
Week 3-4 -- The 6-stock hardware-target problem, full pipeline in one script.

Reuses your already-cached prices (prices_cache.csv from week1_qubo_setup.py) --
the 6 stocks here are a subset of your original 10, so this needs no new
network calls. Runs: QUBO -> classical brute-force optimum -> penalty check
-> Ising mapping -> QAOA REPS sweep, all sized for a 6-qubit problem so it's
fast to iterate on before spending real IBM hardware job quota.

Run:
    python week4_six_stock_pipeline.py
"""

import itertools

import numpy as np
import pandas as pd

from week1_qubo_setup import PRICE_CACHE, returns_and_cov, build_qubo, brute_force_optimum
from week2_ising_mapping import qubo_to_ising, build_hamiltonian, verify_mapping
from week3_qaoa_circuit import cost_hamiltonian
from week3b_reps_sweep import best_over_restarts

HW_TICKERS = ["AAPL", "MSFT", "JPM", "JNJ", "KO", "NVDA"]   # subset of the original 10 -- change freely
HW_BUDGET = 3
RISK_AVERSION_Q = 0.5
REPS_LIST = [1, 2, 3, 4]
N_RESTARTS = 2
SHOTS = 4096
RNG_SEED = 0


def load_subset_prices(tickers, cache):
    prices = pd.read_csv(cache, index_col=0, parse_dates=True)
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise ValueError(f"{missing} not found in {cache} -- these must be a subset of "
                          f"the original TICKERS list in week1_qubo_setup.py")
    return prices[tickers]


def find_min_penalty(mu, sigma, budget, q, n, p_grid):
    """Smallest penalty (from p_grid) whose QUBO ground state is feasible with the right budget."""
    X = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(float)
    size = X.sum(axis=1)
    f = np.einsum("ki,ij,kj->k", X, sigma, X) - q * (X @ mu)
    for p in p_grid:
        k = np.argmin(f + p * (size - budget) ** 2)
        if size[k] == budget:
            return p
    return None


def main():
    prices = load_subset_prices(HW_TICKERS, PRICE_CACHE)
    print(f"Using cached prices, {prices.shape[0]} trading days, "
          f"{prices.index[0].date()} -> {prices.index[-1].date()}")
    mu, sigma = returns_and_cov(prices)
    n = len(HW_TICKERS)

    p_grid = np.geomspace(1e-4, 1.0, 200)
    min_p = find_min_penalty(mu, sigma, HW_BUDGET, RISK_AVERSION_Q, n, p_grid)
    penalty = 2 * min_p if min_p else None
    print(f"Minimum feasible penalty (grid search): {min_p:.4f}" if min_p
          else "Could not find a working penalty in the grid -- check inputs")
    print(f"Using penalty = {penalty:.4f} (2x margin)")

    Q, penalty_used = build_qubo(mu, sigma, HW_BUDGET, RISK_AVERSION_Q, penalty)
    best_x, best_val = brute_force_optimum(Q, n)
    selected = [t for t, xi in zip(HW_TICKERS, best_x) if xi == 1]
    print(f"\nClassical optimum: {selected}  (feasible: {len(selected) == HW_BUDGET})")

    h, J, C0 = qubo_to_ising(Q)
    max_err = verify_mapping(Q, h, J, C0, n)
    print(f"Ising mapping verified, max error: {max_err:.2e}")
    assert max_err < 1e-9

    H = build_hamiltonian(h, J, n)
    print(f"Hamiltonian: {len(H)} Pauli terms on {n} qubits "
          f"({n} single-qubit + {n*(n-1)//2} two-qubit -- fully connected)")

    true_bitstring = "".join(str(int(v)) for v in best_x[::-1])
    rng = np.random.default_rng(RNG_SEED)

    print(f"\nREPS sweep on the {n}-qubit problem (much cheaper than the 10-asset case):")
    for reps in REPS_LIST:
        ansatz, res, label = best_over_restarts(H, n, reps, rng)
        from week3_qaoa_circuit import sample_final_circuit
        counts = sample_final_circuit(ansatz, res.x, n, SHOTS)
        total = sum(counts.values())
        p_true = counts.get(true_bitstring, 0) / total
        p_feasible = sum(c for bs, c in counts.items()
                          if sum(int(b) for b in bs) == HW_BUDGET) / total
        print(f"  reps={reps}  best_start={label:12s}  energy={res.fun:+.4f}  "
              f"P(optimal)={p_true:.3f}  P(any feasible)={p_feasible:.3f}")

    np.savez("hw6_pipeline.npz", Q=Q, h=h, J=J, C0=C0, tickers=HW_TICKERS,
              budget=HW_BUDGET, penalty=penalty_used, best_x=best_x, best_val=best_val)
    print("\nSaved to hw6_pipeline.npz")
    print("If P(optimal) is comfortably high at a modest REPS (say >0.3-0.4 by reps=3-4),")
    print("this problem is a good, honest candidate for the actual IBM hardware run next.")


if __name__ == "__main__":
    main()
