"""
Week 4 wrap-up -- simulator vs. raw hardware vs. readout-mitigated, side by side.

Regenerates the exact same warm-start parameters used for the hardware run
(deterministic given the fixed seed -- reproduces the -0.1477 energy you saw),
samples them noise-free on the simulator, and compares against the real
hw6_hardware_result.npz from week5_hardware_run.py.

Run:
    python week6_compare_results.py
Produces:
    sim_vs_hardware.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from week3_qaoa_circuit import cost_hamiltonian, sample_final_circuit
from week3b_reps_sweep import best_over_restarts
import week3b_reps_sweep as sw

FINAL_REPS = 2
N_RESTARTS_FINAL = 3
SHOTS = 4096


def main():
    d = np.load("hw6_pipeline.npz")
    h, J, tickers = d["h"], d["J"], [str(t) for t in d["tickers"]]
    budget = int(d["budget"])
    best_x = d["best_x"]
    n = len(tickers)
    true_bitstring = "".join(str(int(v)) for v in best_x[::-1])

    H = cost_hamiltonian(h, J, n)
    rng = np.random.default_rng(0)
    orig, sw.N_RESTARTS = sw.N_RESTARTS, N_RESTARTS_FINAL
    ansatz, res, label = best_over_restarts(H, n, FINAL_REPS, rng)
    sw.N_RESTARTS = orig
    print(f"Regenerated simulator parameters, energy={res.fun:.4f} "
          f"(should match your hardware run's 'Final simulator energy')")

    sim_counts = sample_final_circuit(ansatz, res.x, n, SHOTS)
    sim_total = sum(sim_counts.values())
    sim_dist = {k: v / sim_total for k, v in sim_counts.items()}

    hw = np.load("hw6_hardware_result.npz", allow_pickle=True)
    raw_counts = hw["raw_counts"].item()
    raw_total = sum(raw_counts.values())
    raw_dist = {k: v / raw_total for k, v in raw_counts.items()}
    mitigated = dict(hw["mitigated"].item())

    def p_optimal(dist):
        return dist.get(true_bitstring, 0.0)

    def p_feasible(dist):
        return sum(v for k, v in dist.items() if sum(int(b) for b in k) == budget)

    print(f"\n{'':20s} {'P(optimal)':>12s} {'P(feasible)':>12s}")
    for name, dist in [("simulator (ideal)", sim_dist),
                        ("raw hardware", raw_dist),
                        ("mitigated", mitigated)]:
        print(f"{name:20s} {p_optimal(dist):>12.3f} {p_feasible(dist):>12.3f}")

    # Union of the top states across all three, for a fair side-by-side chart
    top_states = set()
    for dist in (sim_dist, raw_dist, mitigated):
        top_states.update(sorted(dist, key=dist.get, reverse=True)[:5])
    top_states = sorted(top_states, key=lambda s: -sim_dist.get(s, 0))[:8]

    labels = []
    for s in top_states:
        x = [int(b) for b in s[::-1]]
        sel = [t for t, xi in zip(tickers, x) if xi == 1]
        labels.append(",".join(sel) if sel else s)

    sim_vals = [sim_dist.get(s, 0) for s in top_states]
    raw_vals = [raw_dist.get(s, 0) for s in top_states]
    mit_vals = [mitigated.get(s, 0) for s in top_states]

    x_pos = np.arange(len(top_states))
    width = 0.27
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x_pos - width, sim_vals, width, label="simulator (ideal)")
    ax.bar(x_pos, raw_vals, width, label="raw hardware")
    ax.bar(x_pos + width, mit_vals, width, label="readout-mitigated")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("probability")
    ax.set_title(f"Simulator vs. real IBM hardware (reps={FINAL_REPS}, {n}-stock portfolio)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("sim_vs_hardware.png", dpi=150)
    print("\nSaved sim_vs_hardware.png")


if __name__ == "__main__":
    main()
