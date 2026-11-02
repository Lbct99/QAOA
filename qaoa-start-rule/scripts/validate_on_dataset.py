"""Validate the packaged rule against the 600-instance dataset: it should reproduce the study's results.

Uses the same time split as the study (first 60% of instances by window end date = train, next 20% =
validation, last 20% = test, 180-day purge) and runs the PACKAGED procedure (qaoa_start_rule.optimize) on
the test instances with the exact numpy simulator. Because the procedure is deterministic, the numbers
should match the study's qaoa_pipeline_comparison.py output for the same k almost exactly.

    python scripts/validate_on_dataset.py --data qaoa_instances.npz            # k = 4 (the validated default)
    python scripts/validate_on_dataset.py --data qaoa_instances.npz --k 1      # the single-constant rule
Add --strict to exit with an error code when the reference is not reproduced (for CI).
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # works without `pip install`

from qaoa_start_rule import QAOASim, optimize

# Reference from the study (your real 600-instance dataset, 120 later-window test instances):
# share within 0.5% / 1% / 2% of the energy span, median and mean total evaluations.
REFERENCE = {
    1: (0.78, 0.83, 0.86, 157, 176),
    2: (0.88, 0.88, 0.91, 182, 191),
    3: (0.92, 0.92, 0.94, 206, 211),
    4: (0.95, 0.96, 0.98, 230, 233),
}


def time_split(end_dates, purge_days=180):
    end = np.asarray(end_dates).astype("datetime64[D]")
    order = np.argsort(end)
    n = len(end)
    b1, b2 = end[order[int(0.6 * n)]], end[order[int(0.8 * n)]]
    test = np.flatnonzero(end >= b2)
    return test, (b1, b2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="qaoa_instances.npz")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--purge-days", type=int, default=180)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    D = np.load(args.data)
    p = int(D["layers"])
    if p != 2:
        raise SystemExit("The packaged rule is for p=2; this dataset has p=%d." % p)
    test, (b1, b2) = time_split(D["window_end"], args.purge_days)
    span = D["e_max"] - D["e_min"]
    print(f"{len(D['opt_energy'])} instances; {len(test)} later-window test instances (window end from {b2}); k = {args.k}")

    gaps, evals = [], []
    for j, i in enumerate(test):
        sim = QAOASim(D["h"][i], D["J"][i], p)
        res = optimize(sim.energy, D["h"][i], D["J"][i], k=args.k)
        gaps.append(max((res.energy - D["opt_energy"][i]) / span[i], 0.0))
        evals.append(res.evaluations)
        if (j + 1) % 20 == 0 or j + 1 == len(test):
            print(f"  {j + 1}/{len(test)} done")
    gaps, evals = np.array(gaps), np.array(evals)

    shares = [float(np.mean(gaps < t)) for t in (0.005, 0.01, 0.02)]
    print(f"\n{'':22s} {'within 0.5%':>12s} {'within 1%':>10s} {'within 2%':>10s} {'median evals':>13s} {'mean evals':>11s}")
    print(f"{'packaged rule':22s} " + "".join(f"{s:12.0%}" if i == 0 else f"{s:10.0%}" for i, s in enumerate(shares))
          + f"{np.median(evals):13.0f}{np.mean(evals):11.0f}")
    ref = REFERENCE.get(args.k)
    if ref:
        print(f"{'study reference':22s} {ref[0]:12.0%}{ref[1]:10.0%}{ref[2]:10.0%}{ref[3]:13.0f}{ref[4]:11.0f}")
        ok = (all(abs(s - r) <= 0.015 for s, r in zip(shares, ref[:3])) and abs(np.median(evals) - ref[3]) <= 3)
        if ok:
            print("\nREPRODUCES the study's result for this k.")
        else:
            print("\nDIFFERS from the study's reference. Expected if the dataset, the constants or the split changed; "
                  "otherwise investigate before trusting the package.")
        if args.strict and not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
