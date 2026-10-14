"""
Instance generator for a learned QAOA (gamma, beta) initializer  (step 1 of 3).

Plan: (1) this script builds a dataset of 6-asset portfolio problems with
their optimized p=2 QAOA parameters and baseline optimizer behaviour;
(2) a training script learns problem features -> parameters; (3) an
evaluation script measures how many optimizer evaluations the learned start
saves on held-out instances, compared with the baselines recorded here.

What each instance is: a random 6-of-10 stock subset, a random trailing
window of daily returns, a random risk weight q, budget B. The QUBO uses the
pipeline's raw-unit convention with penalty = 2 x the minimum feasible
penalty (as in week4_six_stock_pipeline.py), then the same QUBO -> Ising
mapping as week2. Everything is built from prices_long_cache.csv (created by
week8_backtest.py).

Speed: QAOA expectations come from a small exact numpy simulator (6 qubits =
64 amplitudes), thousands of times faster than the Qiskit estimator. Before
generating anything, the script checks (a) the Ising mapping against brute
force, (b) the simulator against Qiskit's QAOAAnsatz + Statevector, and
(c) that the parameter symmetries used for canonicalization leave the energy
unchanged. If any check fails it stops.

Parameter order everywhere in the output: [gamma_1..gamma_p, beta_1..beta_p].
Energies are for H_C WITHOUT the constant C0 (what the Qiskit estimator saw).

Baselines recorded per instance (single-start POWELL, same settings as the
pipeline):
    ramp         the pipeline's linear-ramp start (fixed, scale-blind)
    scaled_ramp  the same ramp with gamma divided by the problem's energy
                 scale s = sqrt(sum h^2 + sum J^2)
For each: evaluations used, final energy, and best-so-far energy after
CHECKPOINTS evaluations (so "evaluations saved" can be measured at any
quality level, not just at convergence). The label ("best-known" parameters)
is the best of those two starts plus N_RANDOM_STARTS random starts, polished
with tight tolerances and canonicalized. It is the best-known optimum, not a
proven global one; `label_hits` counts how many starts ended within 0.2% of
the energy span of it (a reliability indicator).

Why the scaled ramp and the report matter: a learned initializer is only
worth building if it beats cheap heuristics. The report at the end shows
whether a constant start in normalized units already does most of the job.

Splits: each instance stores its window end date, window length and asset
subset, so training can split by TIME (train on early windows, test on late
ones). Neighbouring windows share most of their data; a random split would
leak.

Run (from the project root, next to prices_long_cache.csv):
    python qaoa_instance_generator.py --n 600 --random-starts 20
Produces:
    qaoa_instances.npz      (saved every 50 instances; if a run is interrupted, repeat the
                             same command with --resume to continue where it stopped)

On label quality: with the default 6 random starts, about 1 in 10 best-known
labels was off by more than 1% of the energy span (measured against 30 starts
on synthetic test data). 20 starts is a good compromise; cost grows linearly
with the number of starts.

Saved arrays (N = number of instances, p = layers):
    opt_params (N, 2p)  best-known [gamma_1..gamma_p, beta_1..beta_p], canonical form
    opt_energy, e_min, e_max (N,)   best-known energy and the H_C spectrum range (no C0)
    h (N, 6), J (N, 6, 6), C0, scale (N,)   Ising problem; scale = sqrt(sum h^2 + sum J^2)
    mu (N, 6), sigma (N, 6, 6), q, budget, penalty (N,)   the portfolio problem behind it
    subset (N, 6), window_end (N,), window_len (N,)   for time-based train/test splits
    ramp_* / sramp_*   nfev, final energy and best-so-far energy at `checkpoints` for the two baselines
    label_hits (N,)    how many starts ended within 0.2% of the span of the best-known energy
"""

import os

# These problems are tiny (64 amplitudes): single-threaded BLAS is fastest and avoids
# thread-oversubscription slowdowns. Must be set before numpy is imported.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

PRICE_CACHE = "prices_long_cache.csv"
TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "PG", "V", "KO", "NVDA", "DIS"]
N_ASSETS = 6
BUDGETS = [3]                      # the hardware problem is 3 of 6; add e.g. 2, 4 for diversity
WINDOW_LENGTHS = [504, 756, 1008]  # 2, 3 and 4 years of trading days
Q_RANGE = (0.2, 2.0)               # raw-unit risk weight, sampled log-uniformly (pipeline uses 0.5)
N_RANDOM_STARTS = 6
CHECKPOINTS = [1, 5, 10, 25, 50, 100, 200]
TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Problem construction (same conventions as week1 / week2)
# ---------------------------------------------------------------------------
def all_bits(n):
    return ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(float)


def build_qubo(mu, sigma, budget, q, penalty):
    n = len(mu)
    Q = sigma.copy()
    Q -= q * np.diag(mu)
    Q += penalty * (np.ones((n, n)) - 2 * budget * np.eye(n))
    return Q


def qubo_to_ising(Q):
    h = -0.5 * Q.sum(axis=1)
    J = np.triu(0.5 * Q, k=1)
    C0 = 0.5 * np.trace(Q) + J.sum()
    return h, J, C0


def min_feasible_penalty(mu, sigma, budget, q, X):
    """Smallest penalty on a geometric grid whose QUBO ground state holds exactly `budget` assets."""
    size = X.sum(axis=1)
    f = np.einsum("ki,ij,kj->k", X, sigma, X) - q * (X @ mu)
    grid = np.geomspace(1e-4, 10.0, 400)
    E = f[None, :] + grid[:, None] * (size - budget)[None, :] ** 2
    ok = size[np.argmin(E, axis=1)] == budget
    return float(grid[np.argmax(ok)]) if ok.any() else None


# ---------------------------------------------------------------------------
# Exact QAOA simulator (numpy)
# ---------------------------------------------------------------------------
class QAOASim:
    """<H_C> after p QAOA layers from |+>^n:  cost phase exp(-i*gamma*H_C), then mixer exp(-i*beta*sum X).
    H_C is diagonal; the mixer is applied as W diag(exp(-i*beta*sum z)) W with W = Hadamard on every qubit."""

    def __init__(self, h, J, layers):
        n = len(h)
        self.n, self.p = n, layers
        z = 1.0 - 2.0 * all_bits(n)                  # Z eigenvalues, qubit k = bit k of the basis index
        self.diag = z @ h + 0.5 * np.einsum("ki,ij,kj->k", z, J + J.T, z)
        self.zsum = z.sum(axis=1)
        H1 = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
        W = np.array([[1.0]])
        for _ in range(n):
            W = np.kron(W, H1)
        self.W = W
        self.psi0 = np.full(2**n, 1.0 / np.sqrt(2.0**n), dtype=complex)

    def energy(self, params):
        p = self.p
        psi = self.psi0
        for g, b in zip(params[:p], params[p:]):
            psi = psi * np.exp(-1j * g * self.diag)
            psi = self.W @ (np.exp(-1j * b * self.zsum) * (self.W @ psi))
        return float(np.sum(self.diag * (psi.real**2 + psi.imag**2)))


def canonicalize(params, p):
    """beta has period pi (exp(-i*pi*sum X) is a global phase) and E(-gamma,-beta) = E(gamma,beta)
    for a real H_C. Fold beta into [-pi/2, pi/2) and flip all signs so that beta_1 >= 0."""
    gam, bet = np.array(params[:p], dtype=float), np.array(params[p:], dtype=float)
    bet = (bet + np.pi / 2) % np.pi - np.pi / 2
    if bet[0] < 0:
        gam, bet = -gam, -bet
        bet = (bet + np.pi / 2) % np.pi - np.pi / 2
    return np.concatenate([gam, bet])


def run_powell(sim, x0):
    """Single-start POWELL (scipy defaults, as in the pipeline). Returns x, best energy, evaluations, best-so-far history."""
    hist = []

    def f(x):
        v = sim.energy(x)
        hist.append(min(v, hist[-1]) if hist else v)
        return v

    res = minimize(f, x0, method="Powell")
    return res.x, hist[-1], len(hist), hist


def trace_at_checkpoints(hist):
    return [hist[min(c, len(hist)) - 1] for c in CHECKPOINTS]


def ramp_start(p):
    return np.linspace(1.0 / p, 1.0, p) * 0.5, np.linspace(1.0, 1.0 / p, p) * 0.5


# ---------------------------------------------------------------------------
# Self-checks
# ---------------------------------------------------------------------------
def random_ising(rng, n=6, scale=0.3):
    h = rng.normal(0, scale, n)
    J = np.triu(rng.normal(0, scale / 3, (n, n)), k=1)
    return h, J


def self_checks(layers, seed):
    rng = np.random.default_rng(seed + 12345)
    n = N_ASSETS

    # (a) Ising mapping == QUBO objective on every bitstring
    A = rng.normal(size=(n, n))
    sigma = A @ A.T / n * 0.05
    mu = rng.uniform(-0.05, 0.4, n)
    Q = build_qubo(mu, sigma, 3, 0.5, 0.2)
    h, J, C0 = qubo_to_ising(Q)
    X = all_bits(n)
    z = 1 - 2 * X
    ising = C0 + z @ h + 0.5 * np.einsum("ki,ij,kj->k", z, J + J.T, z)
    assert np.allclose(np.einsum("ki,ij,kj->k", X, Q, X), ising), "QUBO -> Ising mapping mismatch"
    print("  (a) Ising mapping matches the QUBO on all 64 bitstrings.")

    # (c) canonicalization leaves the energy unchanged
    for _ in range(3):
        h, J = random_ising(rng)
        sim = QAOASim(h, J, layers)
        x = rng.uniform(-4, 4, 2 * layers)
        assert abs(sim.energy(x) - sim.energy(canonicalize(x, layers))) < 1e-10, "canonicalization changed the energy"
    print("  (c) canonicalization (beta period, sign flip) leaves the energy unchanged.")

    # (b) numpy simulator == Qiskit QAOAAnsatz + Statevector
    import warnings
    try:
        from scipy.sparse import SparseEfficiencyWarning
        warnings.simplefilter("ignore", SparseEfficiencyWarning)
    except ImportError:
        pass
    try:
        from qiskit.circuit.library import QAOAAnsatz
        from qiskit.quantum_info import SparsePauliOp, Statevector
    except ImportError:
        print("  (b) SKIPPED: qiskit not installed, simulator not cross-checked against Qiskit.")
        return
    for _ in range(3):
        h, J = random_ising(rng)
        sim = QAOASim(h, J, layers)
        terms, coeffs = [], []
        for i in range(n):
            lab = ["I"] * n
            lab[n - 1 - i] = "Z"
            terms.append("".join(lab)); coeffs.append(h[i])
        for i in range(n):
            for j in range(i + 1, n):
                lab = ["I"] * n
                lab[n - 1 - i] = "Z"; lab[n - 1 - j] = "Z"
                terms.append("".join(lab)); coeffs.append(J[i, j])
        H = SparsePauliOp(terms, coeffs)
        ansatz = QAOAAnsatz(cost_operator=H, reps=layers)
        x = rng.uniform(-2, 2, 2 * layers)
        binding = {}
        for prm in ansatz.parameters:
            k = int("".join(ch for ch in prm.name if ch.isdigit()))
            is_gamma = "\u03b3" in prm.name or "gamma" in prm.name.lower()
            binding[prm] = x[k] if is_gamma else x[layers + k]
        bound = ansatz.assign_parameters(binding)
        try:
            sv = Statevector(bound)
        except Exception:
            sv = Statevector(bound.decompose(reps=3))
        e_qiskit = float(sv.expectation_value(H).real)
        assert abs(e_qiskit - sim.energy(x)) < 1e-9, f"simulator disagrees with Qiskit: {e_qiskit} vs {sim.energy(x)}"
    print(f"  (b) numpy simulator matches Qiskit QAOAAnsatz at p={layers} (3 random instances, tol 1e-9).")


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------
def make_instance(R, dates, rng, X, layers):
    T, N = R.shape
    while True:
        L = int(rng.choice(WINDOW_LENGTHS))
        end = int(rng.integers(L, T + 1))
        subset = np.sort(rng.choice(N, N_ASSETS, replace=False))
        budget = int(rng.choice(BUDGETS))
        q = float(np.exp(rng.uniform(np.log(Q_RANGE[0]), np.log(Q_RANGE[1]))))
        win = R[end - L:end][:, subset]
        mu = win.mean(axis=0) * TRADING_DAYS
        sigma = np.cov(win.T) * TRADING_DAYS
        pmin = min_feasible_penalty(mu, sigma, budget, q, X)
        if pmin is None:
            continue
        penalty = 2.0 * pmin
        Q = build_qubo(mu, sigma, budget, q, penalty)
        h, J, C0 = qubo_to_ising(Q)
        s = float(np.sqrt(np.sum(h**2) + np.sum(J**2)))
        if s <= 0:
            continue
        sim = QAOASim(h, J, layers)
        p = layers

        # starts
        ramp_g, ramp_b = ramp_start(p)
        starts = {
            "ramp": np.concatenate([ramp_g, ramp_b]),
            "scaled_ramp": np.concatenate([np.linspace(1.0 / p, 1.0, p) / s, ramp_b]),
        }
        runs = {}
        for name, x0 in starts.items():
            x, e, nfev, hist = run_powell(sim, x0)
            runs[name] = dict(x=x, e=e, nfev=nfev, trace=trace_at_checkpoints(hist))
        best_x, best_e = min(((r["x"], r["e"]) for r in runs.values()), key=lambda t: t[1])
        all_e = [r["e"] for r in runs.values()]
        for _ in range(N_RANDOM_STARTS):
            x0 = np.concatenate([rng.uniform(-3, 3, p) / s, rng.uniform(-np.pi / 2, np.pi / 2, p)])
            x, e, _, _ = run_powell(sim, x0)
            all_e.append(e)
            if e < best_e:
                best_x, best_e = x, e
        # polish the best start with tight tolerances so the label is precise, then count how many
        # of the starts ended within 0.2% of the energy span of it (same basin, loosely converged)
        res = minimize(sim.energy, best_x, method="Powell", options={"xtol": 1e-8, "ftol": 1e-12})
        if res.fun < best_e:
            best_x, best_e = res.x, float(res.fun)
        span_i = float(sim.diag.max() - sim.diag.min())
        label_hits = int(sum(e <= best_e + 2e-3 * span_i for e in all_e))

        return dict(
            subset=subset, window_end=dates[end - 1], window_len=L, budget=budget, q=q, penalty=penalty,
            mu=mu, sigma=sigma, h=h, J=J, C0=C0, scale=s,
            e_min=float(sim.diag.min()), e_max=float(sim.diag.max()),
            opt_params=canonicalize(best_x, p), opt_energy=float(best_e), label_hits=label_hits,
            ramp_nfev=runs["ramp"]["nfev"], ramp_energy=runs["ramp"]["e"], ramp_trace=runs["ramp"]["trace"],
            sramp_nfev=runs["scaled_ramp"]["nfev"], sramp_energy=runs["scaled_ramp"]["e"],
            sramp_trace=runs["scaled_ramp"]["trace"],
        )


def report(D, layers):
    span = D["e_max"] - D["e_min"]
    p = layers
    n = len(span)
    print(f"\n=== Dataset report ({n} instances, p={p}) ===")

    ar_uniform = D["e_max"] / span                       # E = 0 at params = 0, so AR = (E_max - 0)/span
    ar_opt = (D["e_max"] - D["opt_energy"]) / span
    print(f"Approximation ratio (1 = ground state):  uniform superposition {np.median(ar_uniform):.3f}"
          f"   best-known p={p} {np.median(ar_opt):.3f}   (medians)")

    print("\nSingle-start POWELL from each baseline:")
    for name, nf, en in (("ramp (pipeline)", D["ramp_nfev"], D["ramp_energy"]),
                         ("scaled ramp", D["sramp_nfev"], D["sramp_energy"])):
        gap = (en - D["opt_energy"]) / span
        print(f"  {name:16s} median evaluations {np.median(nf):5.0f}   median gap to best-known {np.median(gap):.4f} of span"
              f"   within 1% of span: {np.mean(gap < 0.01):.0%}")

    print("\nAnytime curves: mean gap to best-known (share of energy span) after k evaluations")
    print(f"  {'k':>5s} {'ramp':>9s} {'scaled ramp':>12s}")
    for j, c in enumerate(CHECKPOINTS):
        g1 = np.mean((D["ramp_trace"][:, j] - D["opt_energy"]) / span)
        g2 = np.mean((D["sramp_trace"][:, j] - D["opt_energy"]) / span)
        print(f"  {c:5d} {g1:9.4f} {g2:12.4f}")

    gam_n = D["opt_params"][:, :p] * D["scale"][:, None]       # gamma in units of 1/scale
    bet = D["opt_params"][:, p:]
    print("\nBest-known parameters, canonical form (10th / 50th / 90th percentile):")
    for k in range(p):
        a, b, c = np.percentile(gam_n[:, k], [10, 50, 90])
        print(f"  gamma_{k + 1} x scale   {a:+.2f} / {b:+.2f} / {c:+.2f}")
    for k in range(p):
        a, b, c = np.percentile(bet[:, k], [10, 50, 90])
        print(f"  beta_{k + 1}            {a:+.2f} / {b:+.2f} / {c:+.2f}")
    print(f"  share with gamma_1 > 0: {np.mean(gam_n[:, 0] > 0):.0%}  (a split near 50/50 means the labels are")
    print("  multimodal: a regression model would average between modes, so predicting the basin matters)")

    hits = D["label_hits"]
    print(f"\nLabel reliability: >= 2 of {2 + N_RANDOM_STARTS} starts ended within 0.2% of the energy span of the "
          f"best-known energy in {np.mean(hits >= 2):.0%} of instances (mean {hits.mean():.1f} starts agree)")
    print("  (a low share means the best-known label may not be the global optimum for those instances)")

    # Diagnostic ceiling for any learned model: the best SINGLE constant start in normalized units,
    # picked from the dataset's own optima, with no optimization at all (in-sample).
    sims = [QAOASim(D["h"][i], D["J"][i], p) for i in range(n)]
    cand = np.concatenate([gam_n, bet], axis=1)

    def gap_of(c, i):
        x = np.concatenate([c[:p] / D["scale"][i], c[p:]])
        return (sims[i].energy(x) - D["opt_energy"][i]) / span[i]

    rng = np.random.default_rng(0)
    sub = rng.choice(n, size=min(n, 100), replace=False)
    scores = [np.mean([gap_of(c, i) for i in sub]) for c in cand]
    best_c = cand[int(np.argmin(scores))]
    gaps = np.array([gap_of(best_c, i) for i in range(n)])
    print(f"\nBest single constant start in normalized units, NO optimization (in-sample diagnostic):"
          f"\n  mean gap to best-known {gaps.mean():.4f} of span, median {np.median(gaps):.4f},"
          f" within 1% of span: {np.mean(gaps < 0.01):.0%}")
    print("A learned initializer has to beat this constant, and the baselines above, to be worth building.")
    print("If most instances already land within about 1% of span, there is little headroom left.")


META_KEYS = {"layers", "checkpoints", "tickers", "config"}


def save_rows(path, rows, p, config):
    D = {k: np.array([r[k] for r in rows]) for k in rows[0].keys()}
    np.savez_compressed(path, layers=p, checkpoints=np.array(CHECKPOINTS), tickers=np.array(TICKERS),
                        config=np.array(config), **D)
    return D


def main():
    global N_RANDOM_STARTS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--n", type=int, default=600, help="number of instances")
    ap.add_argument("--layers", type=int, default=2, help="QAOA layers p")
    ap.add_argument("--random-starts", type=int, default=N_RANDOM_STARTS,
                    help="random POWELL starts per instance (more = more reliable labels, slower)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="qaoa_instances.npz")
    ap.add_argument("--skip-checks", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the instances already saved in --out (same seed, layers and random starts)")
    args = ap.parse_args()
    p = args.layers
    N_RANDOM_STARTS = args.random_starts

    if not args.skip_checks:
        print("Self-checks:")
        self_checks(p, args.seed)

    try:
        prices = pd.read_csv(PRICE_CACHE, index_col=0, parse_dates=True)[TICKERS]
    except FileNotFoundError:
        raise SystemExit(f"{PRICE_CACHE} not found. Run week8_backtest.py once (it downloads and caches the prices).")
    returns = (prices / prices.shift(1) - 1).dropna()
    R = returns.values
    dates = returns.index.strftime("%Y-%m-%d").to_numpy()
    X = all_bits(N_ASSETS)
    print(f"\n{len(returns)} daily returns ({dates[0]} -> {dates[-1]}), {len(TICKERS)} assets; "
          f"generating {args.n} instances ({N_ASSETS} assets each, p={p})")

    config = f"seed={args.seed};p={p};random_starts={N_RANDOM_STARTS}"
    rows, start, t0 = [], 0, time.time()
    if args.resume and os.path.exists(args.out):
        old = np.load(args.out)
        if "config" in old.files and str(old["config"]) == config:
            keys_old = [k for k in old.files if k not in META_KEYS]
            m = len(old[keys_old[0]])
            rows = [{k: old[k][i] for k in keys_old} for i in range(min(m, args.n))]
            start = len(rows)
            print(f"Resuming: {start} instances already saved in {args.out}")
        else:
            raise SystemExit(f"{args.out} was made with a different configuration (seed, layers or random "
                             f"starts), or is not a resumable file. Refusing to resume; run without --resume "
                             f"to start over (that overwrites {args.out}).")
    for i in range(start, args.n):
        rows.append(make_instance(R, dates, np.random.default_rng([args.seed, i]), X, p))
        done = i + 1 - start
        if (i + 1) % 50 == 0 or i + 1 == args.n:
            save_rows(args.out, rows, p, config)       # periodic save: an interrupted run can be resumed
            el = time.time() - t0
            print(f"  {i + 1}/{args.n} done and saved, {el:.0f}s elapsed, ~{el / done * (args.n - i - 1):.0f}s left")

    D = save_rows(args.out, rows, p, config)
    print(f"\nSaved {args.out}")
    report(D, p)


if __name__ == "__main__":
    main()
