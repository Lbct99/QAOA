"""
Week 3b -- REPS sweep with restarts.

A single QAOA run at one depth tells you little: POWELL is a local optimizer
and can converge to a mediocre local minimum regardless of whether the circuit
itself has enough expressivity. This script separates the two questions by
running several random restarts (plus the linear-ramp start) at each depth
and keeping the best result, then reports, per depth:
    - best energy found, and gap to the true classical optimum
    - probability mass on the true optimal bitstring
    - probability mass on ANY feasible (3-asset) bitstring
If more REPS clearly improves these numbers -> circuit depth was the
bottleneck. If it plateaus -> the optimizer (or the optimization landscape
itself, e.g. barren-plateau-like flatness) is the limiting factor, and
more layers alone won't fix it.

Run:
    python week3b_reps_sweep.py
Produces:
    qaoa_reps_sweep.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer.primitives import EstimatorV2

from week3_qaoa_circuit import load_ising, cost_hamiltonian, sample_final_circuit

REPS_LIST = [1, 2, 4, 6]
N_RESTARTS = 2           # random restarts per depth, in addition to the linear-ramp start
                          # (full sweep takes roughly 3-5 minutes on a laptop; cut REPS_LIST or
                          # N_RESTARTS down if you want faster feedback)
SHOTS = 4096
RNG_SEED = 0


def linear_ramp(reps):
    gammas = np.linspace(1.0 / reps, 1.0, reps) * 0.5
    betas = np.linspace(1.0, 1.0 / reps, reps) * 0.5
    return gammas, betas


def optimize_once(ansatz, H, estimator, x0):
    def objective(params):
        return float(estimator.run([(ansatz, H, params)]).result()[0].data.evs)
    return minimize(objective, x0, method="POWELL")


def best_over_restarts(H, n, reps, rng):
    ansatz = QAOAAnsatz(cost_operator=H, reps=reps).decompose(reps=2)
    estimator = EstimatorV2()

    starts = []
    gammas0, betas0 = linear_ramp(reps)
    x0 = np.zeros(ansatz.num_parameters)
    for k, p in enumerate(ansatz.parameters):
        layer = int("".join(ch for ch in p.name if ch.isdigit()) or 0)
        x0[k] = gammas0[min(layer, reps - 1)] if "gamma" in p.name.lower() or "\u03b3" in p.name \
            else betas0[min(layer, reps - 1)]
    starts.append(("linear-ramp", x0))
    for r in range(N_RESTARTS):
        starts.append((f"random-{r}", rng.uniform(0, np.pi, ansatz.num_parameters)))

    best_res, best_label = None, None
    for label, x0_try in starts:
        res = optimize_once(ansatz, H, estimator, x0_try)
        if best_res is None or res.fun < best_res.fun:
            best_res, best_label = res, label

    return ansatz, best_res, best_label


def main():
    h, J, C0, tickers = load_ising()
    n = len(tickers)
    H = cost_hamiltonian(h, J, n)

    d = np.load("qubo_week1.npz")
    best_x = d["best_x"]
    true_bitstring = "".join(str(int(v)) for v in best_x[::-1])  # qiskit bit order
    print(f"Target (classical optimum) bitstring: {true_bitstring}  "
          f"assets: {[t for t, xi in zip(tickers, best_x) if xi == 1]}")

    d2 = np.load("qubo_week1.npz")
    rng = np.random.default_rng(RNG_SEED)

    rows = []
    for reps in REPS_LIST:
        ansatz, res, label = best_over_restarts(H, n, reps, rng)
        counts = sample_final_circuit(ansatz, res.x, n, SHOTS)
        total = sum(counts.values())
        p_true = counts.get(true_bitstring, 0) / total
        p_feasible = sum(c for bs, c in counts.items()
                          if sum(int(b) for b in bs) == int(d2["budget"])) / total
        print(f"reps={reps:2d}  best_start={label:12s}  energy={res.fun:+.4f}  "
              f"P(optimal bitstring)={p_true:.3f}  P(any feasible)={p_feasible:.3f}")
        rows.append((reps, res.fun, p_true, p_feasible))

    rows = np.array(rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(rows[:, 0], rows[:, 1], "o-")
    ax1.set_xlabel("REPS (p)"); ax1.set_ylabel("best energy found (no C0)")
    ax1.set_title("Energy vs depth")
    ax2.plot(rows[:, 0], rows[:, 2], "o-", label="P(optimal bitstring)")
    ax2.plot(rows[:, 0], rows[:, 3], "s-", label="P(any feasible)")
    ax2.set_xlabel("REPS (p)"); ax2.set_ylabel("probability")
    ax2.set_title("Sample concentration vs depth")
    ax2.legend()
    fig.tight_layout()
    fig.savefig("qaoa_reps_sweep.png", dpi=150)
    print("\nSaved qaoa_reps_sweep.png")

    print("\nHow to read this:")
    print("  - If energy keeps dropping and P(optimal) keeps rising with REPS -> depth was the bottleneck, keep increasing it.")
    print("  - If it plateaus after reps=2 or so -> more layers won't help; look at optimizer restarts, a")
    print("    different classical optimizer (COBYLA, SPSA), or a smarter initialization instead.")


if __name__ == "__main__":
    main()
