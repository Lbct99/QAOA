"""
Evaluate learned QAOA initializers against cheap baselines  (steps 2 and 3 of 3).

Input: qaoa_instances.npz from qaoa_instance_generator.py.

Question: does a LEARNED start (problem features -> gamma, beta) save optimizer
evaluations compared with (a) the pipeline's linear ramp and (b) one single
constant start in normalized units? A learned model is only worth deploying if
it beats (b), which needs no model at all.

Starts compared (all in the same normalized units: gamma * scale, beta):
    pipeline ramp  the pipeline's fixed linear ramp (raw units, scale-blind)
    constant       ONE start for every problem: the best of a sample of training
                   optima, chosen on training instances only
    ridge          linear model on permutation-invariant problem features
    knn            nearest-neighbour label (k chosen on validation)
    oracle         the dataset's best-known parameters (reference ceiling)

Protocol:
  - Split by TIME: first 60% of instances (by window end date) train, next 20%
    validation, last 20% test. Training and validation instances whose window
    ends within --purge-days of a boundary are dropped. Windows are 2-4 years
    long, so test windows still overlap training data heavily; that mirrors
    production (a new problem always overlaps last month's), but it means this
    measures "later windows of the same universe", not unrelated problems.
  - Hyperparameters (ridge alpha, k) are picked on validation by the mean energy
    gap of the predicted start, before any optimization.
  - Every start is then run through the same single-start POWELL as the
    pipeline, and we record the best-so-far energy gap after every evaluation.
    gap = (E - E_best_known) / (E_max - E_min).

Metric: share of test instances whose best-so-far gap is below 1% of the
energy span after k evaluations. It is robust to heavy tails, and "reached
within 1% after k evaluations" is the quantity that matters when each
evaluation is a costly circuit run.

Run (from the project root, next to qaoa_instance_generator.py):
    python qaoa_init_evaluation.py
Options: --data FILE, --purge-days N, --tol 0.01, --seed N
"""

import os

# These problems are tiny (64 amplitudes): single-threaded BLAS is fastest and avoids
# thread-oversubscription slowdowns. Must be set before numpy is imported.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse

import numpy as np
from scipy.optimize import minimize

from qaoa_instance_generator import QAOASim, ramp_start

KS = [1, 10, 25, 50, 100]


# ---------------------------------------------------------------------------
# Data, features, split
# ---------------------------------------------------------------------------
def load(path):
    D = dict(np.load(path))
    p = int(D["layers"])
    D["p"] = p
    D["span"] = D["e_max"] - D["e_min"]
    scale = D["scale"]
    D["Y"] = np.concatenate([D["opt_params"][:, :p] * scale[:, None], D["opt_params"][:, p:]], axis=1)
    return D


def build_features(D):
    """Permutation-invariant features: sorted normalized h and J, spectrum position, risk weight, market stats."""
    s = D["scale"]
    m = D["h"].shape[1]
    iu = np.triu_indices(m, 1)
    hs = -np.sort(-(D["h"] / s[:, None]), axis=1)
    Jv = -np.sort(-(D["J"][:, iu[0], iu[1]] / s[:, None]), axis=1)
    vol = np.sqrt(np.einsum("nii->ni", D["sigma"]))
    corr = D["sigma"] / (vol[:, :, None] * vol[:, None, :])
    extras = np.column_stack([
        np.log(s), D["e_min"] / s, D["e_max"] / s, np.log(D["q"]), np.log(D["penalty"]),
        D["mu"].mean(axis=1), D["mu"].std(axis=1), vol.mean(axis=1), corr[:, iu[0], iu[1]].mean(axis=1),
    ])
    return np.hstack([hs, Jv, extras])


def time_split(D, purge_days):
    end = D["window_end"].astype("datetime64[D]")
    order = np.argsort(end)
    n = len(end)
    b1, b2 = end[order[int(0.6 * n)]], end[order[int(0.8 * n)]]
    purge = np.timedelta64(purge_days, "D")
    train = np.flatnonzero(end < b1 - purge)
    val = np.flatnonzero((end >= b1) & (end < b2 - purge))
    test = np.flatnonzero(end >= b2)
    for name, idx in (("train", train), ("validation", val), ("test", test)):
        if len(idx) < 10:
            raise SystemExit(f"Only {len(idx)} {name} instances after the split; generate more instances "
                             f"or lower --purge-days.")
    return train, val, test, (b1, b2)


# ---------------------------------------------------------------------------
# Models (numpy only)
# ---------------------------------------------------------------------------
def fit_ridge(Xs, Y, alpha):
    ybar = Y.mean(axis=0)
    W = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(Xs.shape[1]), Xs.T @ (Y - ybar))
    return lambda Z: Z @ W + ybar


def fit_knn(Xs, Y, k):
    def predict(Z):
        d = ((Z[:, None, :] - Xs[None, :, :]) ** 2).sum(axis=2)
        nn = np.argsort(d, axis=1)[:, :k]
        return np.median(Y[nn], axis=1)       # k=1 returns a real optimum, never an average of basins
    return predict


# ---------------------------------------------------------------------------
# Start quality and optimizer runs
# ---------------------------------------------------------------------------
def raw_start(ynorm, scale, p):
    """Normalized (gamma*scale, beta) -> raw (gamma, beta) for one instance."""
    return np.concatenate([ynorm[:p] / scale, ynorm[p:]])


def start_gap(sim, x0, e_best, span):
    return (sim.energy(x0) - e_best) / span


def best_constant(D, sims, idx, rng, n_cand=150, n_eval=150):
    p = D["p"]
    cand = D["Y"][rng.choice(idx, min(n_cand, len(idx)), replace=False)]
    ev = rng.choice(idx, min(n_eval, len(idx)), replace=False)
    scores = [np.mean([start_gap(sims[i], raw_start(c, D["scale"][i], p), D["opt_energy"][i], D["span"][i])
                       for i in ev]) for c in cand]
    return cand[int(np.argmin(scores))]


def gap_trace(sim, x0, e_best, span):
    """Best-so-far energy gap after every evaluation of single-start POWELL."""
    hist = []

    def f(x):
        v = sim.energy(x)
        hist.append(min(v, hist[-1]) if hist else v)
        return v

    minimize(f, x0, method="Powell")
    return np.maximum((np.array(hist) - e_best) / span, 0.0)


def reached_at(trace, k, tol):
    return trace[min(k, len(trace)) - 1] < tol


def evals_to(trace, tol):
    hit = np.flatnonzero(trace < tol)
    return float(hit[0] + 1) if len(hit) else np.nan


def boot_diff(a, b, rng, n=2000):
    """Paired bootstrap over test instances for mean(a) - mean(b), booleans or floats."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = len(a)
    idx = rng.integers(0, m, (n, m))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return a.mean() - b.mean(), *np.percentile(d, [2.5, 97.5])


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", default="qaoa_instances.npz")
    ap.add_argument("--purge-days", type=int, default=180)
    ap.add_argument("--tol", type=float, default=0.01, help="gap threshold, share of the energy span")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    tol = args.tol

    D = load(args.data)
    p, n = D["p"], len(D["opt_energy"])
    rng = np.random.default_rng(args.seed)
    train, val, test, (b1, b2) = time_split(D, args.purge_days)
    print(f"{n} instances, p={p}. Time split: train {len(train)} (window end before {b1}), "
          f"validation {len(val)} (before {b2}), test {len(test)} (from {b2}); purge {args.purge_days} days.")
    hits = D["label_hits"]
    print(f"Label reliability in this dataset: >= 2 starts agree for {np.mean(hits >= 2):.0%} of instances "
          f"(best-known labels are not proven global optima).\n")

    sims = [QAOASim(D["h"][i], D["J"][i], p) for i in range(n)]
    X = build_features(D)
    mean, std = X[train].mean(axis=0), X[train].std(axis=0) + 1e-9
    Xs = (X - mean) / std

    def pred_gap(pred_norm, idx):
        return np.mean([start_gap(sims[i], raw_start(pred_norm[j], D["scale"][i], p),
                                  D["opt_energy"][i], D["span"][i]) for j, i in enumerate(idx)])

    # --- model selection on validation (no optimization involved) ---
    best = {}
    ridge_scores = {a: pred_gap(fit_ridge(Xs[train], D["Y"][train], a)(Xs[val]), val) for a in (0.1, 1, 10, 100, 1000)}
    a_best = min(ridge_scores, key=ridge_scores.get)
    best["ridge"] = fit_ridge(Xs[train], D["Y"][train], a_best)
    knn_scores = {k: pred_gap(fit_knn(Xs[train], D["Y"][train], k)(Xs[val]), val) for k in (1, 3, 7)}
    k_best = min(knn_scores, key=knn_scores.get)
    best["knn"] = fit_knn(Xs[train], D["Y"][train], k_best)
    print(f"Validation: ridge alpha={a_best} (mean start gap {ridge_scores[a_best]:.4f}), "
          f"kNN k={k_best} (mean start gap {knn_scores[k_best]:.4f})")

    const = best_constant(D, sims, train, rng)
    print(f"Constant start (normalized gamma*scale, beta): {np.round(const, 3)}\n")

    # --- run POWELL from every start on the test instances ---
    methods = ["pipeline ramp", "constant", "ridge", "knn", "oracle"]
    pred = {"ridge": best["ridge"](Xs[test]), "knn": best["knn"](Xs[test])}
    ramp = np.concatenate(ramp_start(p))
    traces = {m: [] for m in methods}
    for j, i in enumerate(test):
        eb, sp, sc = D["opt_energy"][i], D["span"][i], D["scale"][i]
        starts = {
            "pipeline ramp": ramp,
            "constant": raw_start(const, sc, p),
            "ridge": raw_start(pred["ridge"][j], sc, p),
            "knn": raw_start(pred["knn"][j], sc, p),
            "oracle": D["opt_params"][i],
        }
        for m, x0 in starts.items():
            traces[m].append(gap_trace(sims[i], x0, eb, sp))
        if (j + 1) % 20 == 0 or j + 1 == len(test):
            print(f"  optimized {j + 1}/{len(test)} test instances")

    # --- report ---
    print(f"\n=== Test results ({len(test)} instances). 'within' = best-so-far gap < {tol:.0%} of the energy span ===")
    header = f"{'start':14s} {'median gap@1':>13s}" + "".join(f"{'within@' + str(k):>11s}" for k in KS) \
             + f"{'evals to ' + format(tol, '.0%'):>14s}{'final within':>13s}{'median evals':>13s}"
    print(header)
    reach = {m: {k: np.array([reached_at(t, k, tol) for t in traces[m]]) for k in KS} for m in methods}
    for m in methods:
        g1 = np.median([t[0] for t in traces[m]])
        ev = np.array([evals_to(t, tol) for t in traces[m]])
        med_ev = np.nanmedian(ev) if np.any(~np.isnan(ev)) else np.nan
        fin = np.mean([t[-1] < tol for t in traces[m]])
        nfev = np.median([len(t) for t in traces[m]])
        print(f"{m:14s} {g1:13.4f}" + "".join(f"{np.mean(reach[m][k]):11.0%}" for k in KS)
              + f"{med_ev:14.0f}{fin:13.0%}{nfev:13.0f}")
    print("('evals to' = median over instances that reached the threshold; blank/nan = none did)")

    # --- paired comparisons at k=25 and k=1 ---
    print("\nPaired comparisons, difference in share within threshold (95% bootstrap CI over test instances):")
    learned = max(("ridge", "knn"), key=lambda m: np.mean(reach[m][25]))
    for k in (1, 25):
        for a, b in (("constant", "pipeline ramp"), (learned, "constant")):
            d, lo, hi = boot_diff(reach[a][k], reach[b][k], rng)
            verdict = "interval excludes 0" if (lo > 0 or hi < 0) else "interval includes 0"
            print(f"  after {k:3d} evaluations: {a:13s} - {b:13s} {d:+6.0%}  [{lo:+5.0%}, {hi:+5.0%}]  {verdict}")

    print("\nHow to read this:")
    print("  - 'constant' vs 'pipeline ramp': what a one-line normalization rule buys, with no model at all.")
    print("  - learned vs 'constant': the only number that justifies a model. If its interval includes 0,")
    print("    the constant is the production artifact. 'oracle' is the ceiling for any start.")
    print("Caveats: test instances are correlated (overlapping windows, repeated asset subsets), so the")
    print("intervals are optimistic; labels are best-known, not proven optimal; one universe, one history.")


if __name__ == "__main__":
    main()
