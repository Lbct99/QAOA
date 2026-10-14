"""
Portfolio of constant QAOA starts (model-free)  -- follow-up to qaoa_init_evaluation.py.

Finding that motivates this: one constant start in normalized units beats the
pipeline's ramp by a wide margin, and neither ridge regression nor kNN beat the
constant. But about 1 in 6 test instances still never reaches within 1% of the
energy span from the constant (the oracle shows 100% is possible): they sit in a
different basin. Instead of PREDICTING the basin, COVER it: pick K constant
starts, run a short POWELL from each, keep the best result.

Procedure:
  1. Selection pool = training + validation instances (all earlier than the test
     windows, same time split and purge as qaoa_init_evaluation.py).
  2. Candidates = a sample of the pool's best-known parameters (normalized units).
     For every (candidate, instance) pair, run POWELL for at most --budget
     evaluations and record the energy gap reached.
  3. Greedy set cover: first the candidate that gets the most instances within
     --tol of the span, then repeatedly the candidate that covers the most
     still-uncovered instances (up to --k-max starts).
  4. Test on the later windows at EQUAL TOTAL evaluations: K starts x budget
     evaluations each, versus the pipeline ramp and versus a single constant
     (the first pick) simply given the same total budget.

gap = (E - E_best_known) / (E_max - E_min).

Run (from the project root, next to the other scripts and qaoa_instances.npz):
    python qaoa_start_portfolio.py
Options: --data FILE, --budget 25, --k-max 4, --tol 0.01, --purge-days 180,
         --candidates 60, --pool-instances 150, --seed 0
"""

import os

# Tiny problems: single-threaded BLAS is fastest. Must be set before numpy is imported.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse

import numpy as np
from scipy.optimize import minimize

from qaoa_instance_generator import QAOASim, ramp_start
from qaoa_init_evaluation import load, time_split, raw_start, gap_trace, boot_diff


class _Stop(Exception):
    pass


def budgeted_gap(sim, x0, e_best, span, budget):
    """Best-so-far gap after at most `budget` evaluations of POWELL started at x0
    (fewer if POWELL converges first, in which case the final value is returned)."""
    hist = []

    def f(x):
        if len(hist) >= budget:
            raise _Stop
        v = sim.energy(x)
        hist.append(min(v, hist[-1]) if hist else v)
        return v

    try:
        minimize(f, x0, method="Powell")
    except _Stop:
        pass
    return max((hist[-1] - e_best) / span, 0.0)


def greedy_cover(M, tol, k_max):
    """M[a, b] = gap reached by candidate a on pool instance b. Returns chosen candidate indices
    and the cumulative covered share after each pick."""
    covered = np.zeros(M.shape[1], dtype=bool)
    chosen, cum = [], []
    for _ in range(k_max):
        newly = (M < tol) & ~covered[None, :]
        gain = newly.sum(axis=1).astype(float)
        rest = ~covered
        if rest.any():
            gain -= 1e-6 * M[:, rest].mean(axis=1)      # tie-break: lower gap on what is still uncovered
        a = int(np.argmax(gain))
        chosen.append(a)
        covered |= M[a] < tol
        cum.append(float(covered.mean()))
    return chosen, cum


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", default="qaoa_instances.npz")
    ap.add_argument("--budget", type=int, default=25, help="evaluations per start")
    ap.add_argument("--k-max", type=int, default=4)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--purge-days", type=int, default=180)
    ap.add_argument("--candidates", type=int, default=60)
    ap.add_argument("--pool-instances", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    B, K, tol = args.budget, args.k_max, args.tol

    D = load(args.data)
    p, n = D["p"], len(D["opt_energy"])
    rng = np.random.default_rng(args.seed)
    train, val, test, _ = time_split(D, args.purge_days)
    pool = np.concatenate([train, val])
    sims = [QAOASim(D["h"][i], D["J"][i], p) for i in range(n)]
    print(f"{n} instances, p={p}. Selection pool {len(pool)} (train+validation), test {len(test)}.")

    cand_idx = rng.choice(pool, min(args.candidates, len(pool)), replace=False)
    inst_idx = rng.choice(pool, min(args.pool_instances, len(pool)), replace=False)
    cand = D["Y"][cand_idx]
    print(f"Selection: {len(cand)} candidates x {len(inst_idx)} instances x up to {B} evaluations "
          f"(about {len(cand) * len(inst_idx) * B / 1e3:.0f}k circuit evaluations)...")
    M = np.zeros((len(cand), len(inst_idx)))
    for a, c in enumerate(cand):
        for b, i in enumerate(inst_idx):
            M[a, b] = budgeted_gap(sims[i], raw_start(c, D["scale"][i], p),
                                   D["opt_energy"][i], D["span"][i], B)
        if (a + 1) % 15 == 0 or a + 1 == len(cand):
            print(f"  candidate {a + 1}/{len(cand)}")

    chosen, cum = greedy_cover(M, tol, K)
    print(f"\nGreedy portfolio on the selection pool (within {tol:.0%} of span after {B} evaluations per start):")
    for k, (a, c) in enumerate(zip(chosen, cum), 1):
        print(f"  start {k}: gamma*scale, beta = {np.round(cand[a], 3)}   cumulative coverage {c:.0%}")
    starts = cand[chosen]

    # --- test at equal total evaluations ---
    ramp = np.concatenate(ramp_start(p))
    m = len(test)
    per_start = np.zeros((m, K))                    # gap after B evaluations from each chosen start
    single = []                                     # the first pick, run longer
    ramp_tr = []
    for j, i in enumerate(test):
        eb, sp, sc = D["opt_energy"][i], D["span"][i], D["scale"][i]
        for k in range(K):
            per_start[j, k] = budgeted_gap(sims[i], raw_start(starts[k], sc, p), eb, sp, B)
        single.append(gap_trace(sims[i], raw_start(starts[0], sc, p), eb, sp))
        ramp_tr.append(gap_trace(sims[i], ramp, eb, sp))
        if (j + 1) % 40 == 0 or j + 1 == m:
            print(f"  tested {j + 1}/{m} instances")

    def at(trace, t):
        return trace[min(t, len(trace)) - 1]

    print(f"\n=== Test: share of {m} later-window instances within {tol:.0%} of the energy span, "
          f"equal TOTAL evaluations ===")
    print(f"{'total evals':>11s} {'ramp':>7s} {'one constant':>13s} {'portfolio':>12s}   "
          f"{'portfolio - one constant':>26s}")
    for k in range(1, K + 1):
        t = k * B
        r = np.array([at(tr, t) < tol for tr in ramp_tr])
        s = np.array([at(tr, t) < tol for tr in single])
        port = per_start[:, :k].min(axis=1) < tol
        if k == 1:
            diff = "(same start)"
        else:
            d, lo, hi = boot_diff(port, s, rng)
            diff = f"{d:+6.0%} [{lo:+5.0%}, {hi:+5.0%}]" + ("  excl. 0" if (lo > 0 or hi < 0) else "  incl. 0")
        print(f"{t:11d} {np.mean(r):7.0%} {np.mean(s):13.0%} {np.mean(port):9.0%} (K={k}) {diff:>26s}")

    print("\nHow to read this:")
    print("  - 'portfolio' = best of K short runs from K constant starts; 'one constant' = the first start")
    print("    alone, given the same total number of evaluations; 'ramp' = the pipeline's start.")
    print("  - If portfolio - one constant has an interval above 0, covering basins beats predicting them.")
    print("  - The pipeline itself used 4 starts x ~250 evaluations; compare that ~1000-evaluation cost with")
    print("    the portfolio's K x budget.")
    print("Caveats: starts were chosen on earlier windows; test instances are correlated (overlapping windows,")
    print("repeated asset subsets), so intervals are optimistic; labels are best-known, not proven optimal.")


if __name__ == "__main__":
    main()
