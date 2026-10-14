"""
Week 9 -- Check the constant-start rule on the actual hardware instance (hw6_pipeline.npz).

The initializer study measured many random problems. This script takes the one
problem you actually ran on ibm_marrakesh (6 stocks, budget 3) and checks three things:

  1. CONVENTIONS: parameters found by the numpy simulator are bound into the real
     Qiskit circuit (QAOAAnsatz with the pipeline's own cost Hamiltonian from
     week3_qaoa_circuit.cost_hamiltonian) and the energies must agree to 1e-9.
  2. COST AND QUALITY on this instance: the pipeline's multi-start (ramp + 3 random
     restarts, repeated --repeats times because it is random) versus the proposed
     procedure with K = 1..4 constant starts, against a best-known energy from a
     wide search. Also the energy your hardware run's simulator step reported.
  3. WHAT IT MEANS FOR THE CIRCUIT: exact probabilities (statevector, no shot noise)
     of the optimal portfolio and of any feasible portfolio for each parameter set.

If a procedure finds a lower energy than the one your hardware run used (-0.1477),
the parameters sent to the device were not the best available at p=2.

Run (from the project root, next to the other scripts and hw6_pipeline.npz):
    python week9_hardware_instance_check.py
Options: --data FILE, --repeats 20, --budget 25, --starts ... (as in qaoa_pipeline_comparison.py)
"""

import os

# Tiny problems: single-threaded BLAS is fastest. Must be set before numpy is imported.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import warnings

import numpy as np
from scipy.optimize import minimize

from qaoa_instance_generator import QAOASim, ramp_start
from qaoa_pipeline_comparison import run_budgeted, run_full, DEFAULT_STARTS
from week3_qaoa_circuit import cost_hamiltonian


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", default="hw6_pipeline.npz")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=20, help="repetitions of the pipeline's random multi-start")
    ap.add_argument("--budget", type=int, default=25)
    ap.add_argument("--starts", type=float, nargs="+", default=None)
    ap.add_argument("--hardware-energy", type=float, default=-0.1477,
                    help="final simulator energy your hardware run reported")
    ap.add_argument("--hardware-p-opt", type=float, default=0.098, help="simulator P(optimal) from week6 (4096 shots)")
    ap.add_argument("--hardware-p-feas", type=float, default=0.575, help="simulator P(feasible) from week6")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    p, B = args.layers, args.budget

    d = np.load(args.data)
    h, J = d["h"], d["J"]
    tickers = [str(t) for t in d["tickers"]]
    budget = int(d["budget"])
    best_x = np.array(d["best_x"], dtype=int)
    n = len(h)
    starts = np.array(args.starts if args.starts else DEFAULT_STARTS, dtype=float).reshape(-1, 2 * p)
    K = len(starts)

    sim = QAOASim(h, J, p)
    span = float(sim.diag.max() - sim.diag.min())
    scale = float(np.sqrt(np.sum(h**2) + np.sum(J**2)))
    print(f"Instance: {', '.join(tickers)}; budget {budget}; energy span {span:.4f}; scale {scale:.4f}")
    print(f"Classical optimum: {[t for t, x in zip(tickers, best_x) if x == 1]}")

    # --- Qiskit side (the pipeline's own Hamiltonian) ---
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.quantum_info import Statevector
    try:
        from scipy.sparse import SparseEfficiencyWarning
        warnings.simplefilter("ignore", SparseEfficiencyWarning)
    except ImportError:
        pass
    H = cost_hamiltonian(h, J, n)
    ansatz = QAOAAnsatz(cost_operator=H, reps=p)

    def qiskit_state(params):
        binding = {}
        for prm in ansatz.parameters:
            k = int("".join(ch for ch in prm.name if ch.isdigit()))
            is_gamma = "\u03b3" in prm.name or "gamma" in prm.name.lower()
            binding[prm] = params[k] if is_gamma else params[p + k]
        bound = ansatz.assign_parameters(binding)
        try:
            return Statevector(bound)
        except Exception:
            return Statevector(bound.decompose(reps=3))

    opt_idx = int(sum(int(best_x[i]) << i for i in range(n)))            # qubit i = asset i
    feasible = np.array([bin(k).count("1") == budget for k in range(2**n)])

    def describe(params):
        sv = qiskit_state(params)
        e_q = float(sv.expectation_value(H).real)
        e_n = sim.energy(params)
        assert abs(e_q - e_n) < 1e-9, f"numpy simulator and Qiskit disagree: {e_n} vs {e_q}"
        pr = sv.probabilities()
        return e_n, float(pr[opt_idx]), float(pr[feasible].sum())

    # --- best-known: wide search in normalized units, plus every start used below ---
    rng = np.random.default_rng(args.seed)
    ramp = np.concatenate(ramp_start(p))
    cands = [ramp] + [raw for raw in (np.concatenate([s[:p] / scale, s[p:]]) for s in starts)]
    for _ in range(40):
        cands.append(np.concatenate([rng.uniform(-3, 3, p) / scale, rng.uniform(-np.pi / 2, np.pi / 2, p)]))
    best_e, best_params = np.inf, None
    for x0 in cands:
        x, e, _ = run_full(sim, x0)
        if e < best_e:
            best_e, best_params = e, x
    res = minimize(sim.energy, best_params, method="Powell", options={"xtol": 1e-8, "ftol": 1e-12})
    if res.fun < best_e:
        best_e, best_params = float(res.fun), res.x

    # --- pipeline multi-start, repeated (it is random) ---
    pipe_e, pipe_n, pipe_params = [], [], []
    for r in range(args.repeats):
        rr = np.random.default_rng([args.seed, r])
        runs = [run_full(sim, ramp)] + [run_full(sim, rr.uniform(0, np.pi, 2 * p)) for _ in range(3)]
        j = int(np.argmin([t[1] for t in runs]))
        pipe_e.append(runs[j][1]); pipe_n.append(sum(t[2] for t in runs)); pipe_params.append(runs[j][0])

    # --- proposed: K short runs, then polish the best ---
    shorts = [run_budgeted(sim, np.concatenate([s[:p] / scale, s[p:]]), B) for s in starts]
    prop = {}
    for k in range(1, K + 1):
        pick = int(np.argmin([shorts[t][1] for t in range(k)]))
        x, e, n_pol = run_full(sim, shorts[pick][0])
        if shorts[pick][1] < e:
            x, e = shorts[pick][0], shorts[pick][1]
        prop[k] = (x, e, sum(shorts[t][2] for t in range(k)) + n_pol)

    print("\nQiskit cross-check (energies from the numpy simulator vs the real QAOAAnsatz):")
    rows = []
    e_b, po_b, pf_b = describe(best_params)
    rows.append(("best-known (wide search)", e_b, None, po_b, pf_b))
    r0 = int(np.argmin(pipe_e))
    e0, po0, pf0 = describe(pipe_params[int(np.argsort(pipe_e)[len(pipe_e) // 2])])
    rows.append((f"pipeline multi-start (median of {args.repeats})", e0, float(np.median(pipe_n)), po0, pf0))
    for k in range(1, K + 1):
        e, po, pf = describe(prop[k][0])
        rows.append((f"proposed K={k}", e, float(prop[k][2]), po, pf))
    print("  passed for every parameter set below (tolerance 1e-9).\n")

    print(f"{'parameter set':40s} {'energy':>9s} {'gap/span':>9s} {'evals':>7s} {'P(optimal)':>11s} {'P(feasible)':>12s}")
    print(f"{'your hardware run (reported)':40s} {args.hardware_energy:9.4f} "
          f"{(args.hardware_energy - best_e) / span:9.4f} {'':>7s} {args.hardware_p_opt:11.3f} {args.hardware_p_feas:12.3f}"
          f"   <- sampled, 4096 shots")
    for name, e, ev, po, pf in rows:
        evs = f"{ev:7.0f}" if ev is not None else f"{'':>7s}"
        print(f"{name:40s} {e:9.4f} {(e - best_e) / span:9.4f} {evs} {po:11.3f} {pf:12.3f}")

    print(f"\nPipeline multi-start over {args.repeats} repeats: energy median {np.median(pipe_e):.4f} "
          f"[min {np.min(pipe_e):.4f}, max {np.max(pipe_e):.4f}], evaluations median {np.median(pipe_n):.0f}")
    worse = np.mean(np.array(pipe_e) > best_e + 0.01 * span)
    print(f"  share of repeats more than 1% of span above the best-known energy: {worse:.0%}")
    print(f"Proposed procedure is deterministic (fixed starts): K=4 energy {prop[K][1]:.4f} in {prop[K][2]} evaluations.")

    print("\nHow to read this:")
    print("  - 'gap/span' is the energy gap to the best-known energy as a share of this instance's span.")
    print("  - P(optimal) / P(feasible) are exact statevector probabilities (no shot noise); the hardware")
    print("    row was sampled, so differences under about 0.01 are within its noise.")
    print("  - If a row's energy is clearly below your hardware run's, better p=2 parameters existed than the")
    print("    ones sent to the device. Lower energy usually, but not always, means higher P(optimal).")
    print("Caveat: one instance proves nothing about the rule; it checks conventions and shows what the rule")
    print("does on the problem that mattered. The study (qaoa_pipeline_comparison.py) is the evidence.")


if __name__ == "__main__":
    main()
