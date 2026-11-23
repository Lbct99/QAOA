"""
Compare the packaged procedure with the study's code, instance by instance, on a real dataset.

Why this exists: scripts/validate_on_dataset.py reproduced the study's evaluation counts exactly (median 230,
mean 233) but ended within 1% of the energy span for 98% of the test instances instead of the study's 96%. If
both implementations run the same algorithm on the same floating-point numbers, the results must be identical
instance by instance. This script finds out whether they are, and where they diverge if not.

It checks, in order: (1) the constant starts agree, (2) the energy scale of every instance agrees, (3) the two
simulators agree at a fixed parameter vector, (4) the full procedure (k = 4, 25 evaluations per start, then a
polish) agrees instance by instance: final energy, total evaluations, and the first-stage energies of the
instances that differ.

Run from the project root (next to qaoa_pipeline_comparison.py and qaoa_instances.npz), package installed:
    python compare_package_to_study.py --data qaoa_instances.npz
    python compare_package_to_study.py --data qaoa_instances.npz --single-thread      # compare the two runs
"""

import os
import sys

# Must happen BEFORE numpy is imported: pin the BLAS threads if asked to.
if "--single-thread" in sys.argv:
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_var] = "1"

# What the thread settings were when numpy loads (the study modules set them later, which would hide this).
ENV_AT_LOAD = {v: os.environ.get(v, "(unset)") for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}

import argparse

import numpy as np
import scipy

from qaoa_pipeline_comparison import run_budgeted, run_full, DEFAULT_STARTS
from qaoa_init_evaluation import load, time_split, raw_start
from qaoa_instance_generator import QAOASim as StudySim
from qaoa_start_rule import CONSTANT_STARTS, QAOASim as PkgSim, energy_scale, optimize


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", default="qaoa_instances.npz")
    ap.add_argument("--purge-days", type=int, default=180)
    ap.add_argument("--single-thread", action="store_true", help="pin BLAS to one thread (set before numpy loads)")
    ap.add_argument("--show", type=int, default=8, help="how many differing instances to print in detail")
    args = ap.parse_args()

    D = load(args.data)
    p = D["p"]
    _, _, test, _ = time_split(D, args.purge_days)
    print(f"numpy {np.__version__}, scipy {scipy.__version__}; thread settings when numpy loaded: "
          + ", ".join(f"{v}={val}" for v, val in ENV_AT_LOAD.items()))
    print(f"{len(test)} test instances from {args.data}\n")

    # (1) constants
    starts = np.array(DEFAULT_STARTS, float).reshape(-1, 4)
    print(f"(1) constant starts identical in package and study: {np.array_equal(starts, CONSTANT_STARTS)}")

    # (2) scale and (3) simulator
    d_scale, d_diag, d_energy = 0.0, 0.0, 0.0
    x_probe = np.array([-0.4, -0.9, 0.55, 0.3])
    for i in test:
        h, J = D["h"][i], D["J"][i]
        d_scale = max(d_scale, abs(energy_scale(h, J) - D["scale"][i]) / D["scale"][i])
        a, b = StudySim(h, J, p), PkgSim(h, J, p)
        d_diag = max(d_diag, float(np.max(np.abs(a.diag - b.diag))))
        d_energy = max(d_energy, abs(a.energy(x_probe) - b.energy(x_probe)))
    print(f"(2) largest relative difference of the energy scale (package formula vs dataset): {d_scale:.2e}")
    print(f"(3) simulators: largest difference of the diagonal {d_diag:.2e}, of an energy at a fixed point {d_energy:.2e}")

    # (4) the full procedure
    rows = []
    for j, i in enumerate(test):
        h, J = D["h"][i], D["J"][i]
        sc, eb, sp = D["scale"][i], D["opt_energy"][i], D["span"][i]
        sim = StudySim(h, J, p)
        shorts = [run_budgeted(sim, raw_start(starts[k], sc, p), 25) for k in range(4)]
        pick = int(np.argmin([t[1] for t in shorts]))
        x, e, n_pol = run_full(sim, shorts[pick][0])
        if shorts[pick][1] < e:
            x, e = shorts[pick][0], shorts[pick][1]
        study = dict(e=e, n=sum(t[2] for t in shorts) + n_pol, first=[t[1] for t in shorts], pick=pick)
        r = optimize(PkgSim(h, J, p).energy, h, J)
        pkg = dict(e=r.energy, n=r.evaluations, first=r.first_stage_energies, pick=r.chosen_start)
        rows.append((i, study, pkg, eb, sp))
        if (j + 1) % 20 == 0 or j + 1 == len(test):
            print(f"  (4) {j + 1}/{len(test)} instances run")

    diff = [r for r in rows if abs(r[1]["e"] - r[2]["e"]) > 1e-9 or r[1]["n"] != r[2]["n"]]
    gap = lambda e, eb, sp: max((e - eb) / sp, 0.0)
    shares = lambda which: [float(np.mean([gap(r[which]["e"], r[3], r[4]) < t for r in rows])) for t in (0.005, 0.01, 0.02)]
    print(f"\n(4) instances where the final energy or the evaluation count differs: {len(diff)} of {len(rows)}")
    print(f"    study logic : within 0.5/1/2% = " + " / ".join(f"{s:.1%}" for s in shares(1)))
    print(f"    package     : within 0.5/1/2% = " + " / ".join(f"{s:.1%}" for s in shares(2)))
    for i, st, pk, eb, sp in diff[: args.show]:
        print(f"\n  instance {i}: final energy study {st['e']:.10f} vs package {pk['e']:.10f} "
              f"(difference {abs(st['e'] - pk['e']) / sp:.4f} of the span), evaluations {st['n']} vs {pk['n']}, "
              f"chosen start {st['pick']} vs {pk['pick']}")
        print(f"    first-stage energies study  : {np.round(st['first'], 8)}")
        print(f"    first-stage energies package: {np.round(pk['first'], 8)}")

    print("\nHow to read this:")
    if not diff:
        print("  Identical on every instance: the package IS the study's algorithm. Then the earlier mismatch with the")
        print("  study's published 95/96/98% comes from elsewhere (for example the dataset or environment changed since")
        print("  that run): re-run qaoa_pipeline_comparison.py and compare its K=4 row with this output.")
    else:
        print("  The two implementations diverge on some instances even though constants, scale and simulator agree to")
        print("  floating-point precision. POWELL is deterministic but can be sensitive to differences in the last bits")
        print("  of the energy (threading, summation order); compare a run with --single-thread. If the divergence")
        print("  persists with identical floating-point inputs, that points to a real implementation difference.")
        print("  Either way, differences of a few instances change shares by 2-3 points, so treat the study's 96% as")
        print("  carrying that extra numerical uncertainty on top of sampling noise.")


if __name__ == "__main__":
    main()
