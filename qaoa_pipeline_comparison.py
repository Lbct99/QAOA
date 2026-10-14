"""
End-to-end comparison: proposed start procedure vs the pipeline's own multi-start.

The pipeline (week3b_reps_sweep.best_over_restarts, as used in week5/week6) runs
POWELL to convergence from the linear ramp plus 3 random restarts drawn
uniformly in [0, pi] for every parameter, and keeps the best: about 4 x 270 =
1000+ circuit evaluations per problem.

A stronger baseline is added for fairness: the same routine with scale-aware random
restarts (gamma/scale uniform in [-3, 3], beta uniform in [-pi/2, pi/2], as used to
build the dataset labels). Uniform restarts in [0, pi] ignore the problem's energy
scale, so part of any gain over them could just be a weak baseline.

Proposed procedure with K constant starts (normalized units, gamma*scale, beta):
    1. run POWELL for at most B evaluations from each of the K starts
    2. keep the best end point
    3. polish it with POWELL to convergence
K = 1 is a single constant start with a budgeted first stage.

For every procedure we record the FINAL energy gap to the best-known energy,
gap = (E - E_best_known) / (E_max - E_min), and the TOTAL number of evaluations
(summed over all runs). Output: share of test instances within 0.5% / 1% / 2%
of the span, and median / mean total evaluations.

Test instances = the later-window test set of qaoa_init_evaluation.py (same
time split). The constant starts are NOT fitted on them: by default they are
the four greedy picks printed by qaoa_start_portfolio.py on your 600-instance
dataset (selected on training + validation windows only). If you regenerate
the dataset or change settings, re-run qaoa_start_portfolio.py and pass its
starts via --starts (4 numbers per start: gamma1*scale gamma2*scale beta1 beta2).

Run (from the project root):
    python qaoa_pipeline_comparison.py
Options: --data FILE, --budget 25, --purge-days 180, --starts a b c d e f g h ..., --seed 0
"""

import os

# Tiny problems: single-threaded BLAS is fastest. Must be set before numpy is imported.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse

import numpy as np
from scipy.optimize import minimize

from qaoa_instance_generator import QAOASim, ramp_start
from qaoa_init_evaluation import load, time_split, raw_start, boot_diff

DEFAULT_STARTS = [
    [-0.742, -1.122, 0.659, 0.396],
    [-1.206, 2.532, 0.614, -0.311],
    [2.113, -2.117, 1.569, 0.785],
    [-3.507, 3.507, 1.571, -0.785],
]
TOLS = (0.005, 0.01, 0.02)


class _Stop(Exception):
    pass


def run_budgeted(sim, x0, budget):
    """POWELL from x0 for at most `budget` evaluations. Returns (best x, best energy, evaluations used)."""
    st = {"n": 0, "e": np.inf, "x": np.array(x0, dtype=float)}

    def f(x):
        if st["n"] >= budget:
            raise _Stop
        v = sim.energy(x)
        st["n"] += 1
        if v < st["e"]:
            st["e"], st["x"] = v, np.array(x, dtype=float)
        return v

    try:
        minimize(f, x0, method="Powell")
    except _Stop:
        pass
    return st["x"], st["e"], st["n"]


def run_full(sim, x0):
    """POWELL from x0 to convergence. Returns (best x, best energy, evaluations used)."""
    st = {"n": 0, "e": np.inf, "x": np.array(x0, dtype=float)}

    def f(x):
        v = sim.energy(x)
        st["n"] += 1
        if v < st["e"]:
            st["e"], st["x"] = v, np.array(x, dtype=float)
        return v

    minimize(f, x0, method="Powell")
    return st["x"], st["e"], st["n"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", default="qaoa_instances.npz")
    ap.add_argument("--budget", type=int, default=25, help="evaluations per start in the first stage")
    ap.add_argument("--purge-days", type=int, default=180)
    ap.add_argument("--starts", type=float, nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    B = args.budget

    D = load(args.data)
    p, n = D["p"], len(D["opt_energy"])
    starts = np.array(args.starts if args.starts else DEFAULT_STARTS, dtype=float).reshape(-1, 2 * p)
    K = len(starts)
    _, _, test, _ = time_split(D, args.purge_days)
    sims = {i: QAOASim(D["h"][i], D["J"][i], p) for i in test}
    print(f"{n} instances, p={p}; {len(test)} later-window test instances; {K} constant starts, "
          f"{B} evaluations each in the first stage.")
    print("Starts (gamma1*scale, gamma2*scale, beta1, beta2):")
    for k, s in enumerate(starts, 1):
        print(f"  {k}: {np.round(s, 3)}")

    ramp = np.concatenate(ramp_start(p))
    names = ["pipeline: ramp only", "pipeline: ramp + 3 random", "pipeline: ramp + 3 scale-aware"] \
        + [f"proposed K={k}" for k in range(1, K + 1)]
    gaps = {m: [] for m in names}
    evals = {m: [] for m in names}

    for j, i in enumerate(test):
        sim, eb, sp, sc = sims[i], D["opt_energy"][i], D["span"][i], D["scale"][i]
        gap = lambda e: max((e - eb) / sp, 0.0)
        rng = np.random.default_rng([args.seed, int(i)])

        # pipeline: ramp only, and ramp + 3 random restarts (uniform in [0, pi] for every parameter)
        _, e_r, n_r = run_full(sim, ramp)
        gaps["pipeline: ramp only"].append(gap(e_r)); evals["pipeline: ramp only"].append(n_r)
        best_e, total = e_r, n_r
        for _ in range(3):
            _, e, nn = run_full(sim, rng.uniform(0, np.pi, 2 * p))
            best_e, total = min(best_e, e), total + nn
        gaps["pipeline: ramp + 3 random"].append(gap(best_e)); evals["pipeline: ramp + 3 random"].append(total)
        # stronger baseline: ramp + 3 scale-aware random restarts (drawn AFTER the uniform ones, so the
        # rows above reproduce exactly)
        best_e2, total2 = e_r, n_r
        for _ in range(3):
            x0 = np.concatenate([rng.uniform(-3, 3, p) / sc, rng.uniform(-np.pi / 2, np.pi / 2, p)])
            _, e, nn = run_full(sim, x0)
            best_e2, total2 = min(best_e2, e), total2 + nn
        gaps["pipeline: ramp + 3 scale-aware"].append(gap(best_e2)); evals["pipeline: ramp + 3 scale-aware"].append(total2)

        # proposed: K short runs, then polish the best
        shorts = [run_budgeted(sim, raw_start(starts[k], sc, p), B) for k in range(K)]
        polished = {}
        for k in range(1, K + 1):
            pick = int(np.argmin([shorts[t][1] for t in range(k)]))
            if pick not in polished:
                polished[pick] = run_full(sim, shorts[pick][0])
            _, e_fin, n_pol = polished[pick]
            gaps[f"proposed K={k}"].append(gap(min(e_fin, shorts[pick][1])))
            evals[f"proposed K={k}"].append(sum(shorts[t][2] for t in range(k)) + n_pol)
        if (j + 1) % 20 == 0 or j + 1 == len(test):
            print(f"  {j + 1}/{len(test)} test instances done")

    print(f"\n=== Final quality and total cost ({len(test)} later-window test instances) ===")
    print(f"{'procedure':31s}" + "".join(f"{'within ' + format(t * 100, 'g') + '%':>12s}" for t in TOLS)
          + f"{'median evals':>14s}{'mean evals':>12s}")
    for m in names:
        g = np.array(gaps[m])
        e = np.array(evals[m])
        print(f"{m:31s}" + "".join(f"{np.mean(g < t):12.0%}" for t in TOLS)
              + f"{np.median(e):14.0f}{np.mean(e):12.0f}")

    rng = np.random.default_rng(args.seed + 7)
    for ref_name in ("pipeline: ramp + 3 random", "pipeline: ramp + 3 scale-aware"):
        ref = np.array(gaps[ref_name]) < 0.01
        print(f"\nShare within 1% of span, proposed minus '{ref_name}' (95% bootstrap CI over test instances):")
        for k in range(1, K + 1):
            cur = np.array(gaps[f"proposed K={k}"]) < 0.01
            d, lo, hi = boot_diff(cur, ref, rng)
            ratio = np.median(evals[ref_name]) / np.median(evals[f"proposed K={k}"])
            verdict = "excludes 0" if (lo > 0 or hi < 0) else "includes 0"
            print(f"  K={k}: {d:+5.0%} [{lo:+5.0%}, {hi:+5.0%}] {verdict:11s} using {1 / ratio:.0%} of its median evaluations "
                  f"({ratio:.1f}x fewer)")

    print("\nHow to read this:")
    print("  - Aim: a share within 1% at least as high as the pipeline's multi-start, at far fewer evaluations.")
    print("  - If the multi-start's share is clearly higher, the proposed procedure trades quality for cost;")
    print("    compare the 'within' columns before quoting any savings.")
    print("Caveats: the random restarts replicate the pipeline's routine, not its exact random draws; test")
    print("instances are correlated (overlapping windows), so intervals are optimistic; labels are best-known,")
    print("not proven optimal; one universe, one price history. Evaluations are counted in the numpy simulator;")
    print("a hardware or Qiskit-estimator evaluation costs far more, which is why the count is what matters.")


if __name__ == "__main__":
    main()
