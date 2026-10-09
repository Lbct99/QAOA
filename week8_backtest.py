"""
Week 8 -- Classical walk-forward backtest harness.
by claude C. 
Question this answers, BEFORE any more quantum work or universe expansion:
    do the pipeline's portfolio models (equal-weight selection K=1, unequal
    weights K=2, and a few variants) actually beat plain 1/N out of sample,
    and is any difference bigger than noise?

Everything here is classical and uses the same objective as the QUBO:
    minimize  w^T Sigma w - q * mu^T w   over feasible allocations,
    w = y / budget,  y_i in {0..2^K - 1},  sum(y) = budget
Exhaustive search over feasible allocations gives the same optimum as the
QUBO ground state once the penalty is large enough (see week1b / week7b), so
whatever wins here is what a perfect QUBO solver, quantum or classical,
would deliver. The quantum pipeline is untouched.

Design:
  - Walk-forward: at each rebalance, estimate mu and Sigma on the trailing
    TRAIN_YEARS of daily returns, choose weights, hold them for the next
    TEST_DAYS trading days (~6 months). Test windows do NOT overlap, and the
    concatenated test returns form one continuous out-of-sample track.
  - Costs: COST_BPS per unit of traded notional at each rebalance (weights
    are drifted by realized returns before measuring turnover). The first
    allocation is free.
  - Uncertainty: circular block bootstrap (BOOT_BLOCK-day blocks) of the
    paired daily returns gives 95% confidence intervals on the difference in
    return/vol between two strategies.
  - q is expressed on the actual portfolio weights. The pipeline's raw-unit
    q=0.5 at budget B=3 equals q = 0.5/3 here (the argmin is identical; a
    self-check below verifies this). This keeps strategies with different
    budgets comparable instead of silently changing their risk aversion.

Run (first run downloads ~12 years of prices into prices_long_cache.csv; if
Yahoo times out, just run it again):
    python week8_backtest.py
"""

import itertools
import os

import numpy as np
import pandas as pd

TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "PG", "V", "KO", "NVDA", "DIS"]
START = "2014-01-01"
END = "2026-09-28"                   # pinned, so every run uses identical data
LONG_CACHE = "prices_long_cache.csv"   # separate from prices_cache.csv: the 2-year data stays untouched

TRAIN_YEARS = 3
ROBUSTNESS_TRAIN_YEARS = [2, 5]
TEST_DAYS = 126                      # ~6 months between rebalances
TRADING_DAYS = 252
COST_BPS = 10
BOOT_BLOCK = 21
BOOT_N = 2000
SEED = 0

PIPE_Q = 0.5 / 3                     # pipeline's raw q=0.5 at B=3, expressed on portfolio weights

# name -> parameters (None = plain 1/N over all assets)
STRATEGIES = {
    "1/N (all assets)":      None,
    "K1 B=3 (pipeline)":     dict(k_bits=1, budget=3, q=PIPE_Q, shrink=0.0),
    "K2 B=3":                dict(k_bits=2, budget=3, q=PIPE_Q, shrink=0.0),
    "K1 B=5":                dict(k_bits=1, budget=5, q=PIPE_Q, shrink=0.0),
    "K1 B=3 min-variance":   dict(k_bits=1, budget=3, q=0.0,    shrink=0.0),
    "K1 B=3 shrunk mu":      dict(k_bits=1, budget=3, q=PIPE_Q, shrink=0.5),
    "K2 B=3 shrunk mu":      dict(k_bits=2, budget=3, q=PIPE_Q, shrink=0.5),
}
BASELINE = "1/N (all assets)"
PAIRS = [("K2 B=3", "K1 B=3 (pipeline)"), ("K2 B=3 shrunk mu", "K1 B=3 shrunk mu")]


# ---------------------------------------------------------------------------
# Allocation search
# ---------------------------------------------------------------------------
_alloc_cache = {}


def allocations(n, k_bits, budget):
    """All feasible weight vectors w = y/budget with y_i in {0..2^k-1}, sum(y) = budget."""
    key = (n, k_bits, budget)
    if key not in _alloc_cache:
        top = 2**k_bits - 1
        rows = []
        for combo in itertools.combinations_with_replacement(range(n), budget):
            y = np.bincount(combo, minlength=n)
            if y.max() <= top:
                rows.append(y)
        _alloc_cache[key] = np.array(rows, dtype=float) / budget
    return _alloc_cache[key]


def choose_weights(mu, sigma, k_bits, budget, q, shrink):
    mu_s = (1 - shrink) * mu + shrink * mu.mean()    # shrink expected returns toward their average
    W = allocations(len(mu), k_bits, budget)
    f = np.einsum("mi,ij,mj->m", W, sigma, W) - q * (W @ mu_s)
    return W[np.argmin(f)]


def self_check():
    """The weight-based search must pick the same allocation as the pipeline's raw-unit objective."""
    rng = np.random.default_rng(123)
    n = 6
    for _ in range(5):
        A = rng.normal(size=(n, n))
        sigma = A @ A.T / n * 0.05
        mu = rng.uniform(-0.05, 0.4, n)
        for k in (1, 2):
            top = 2**k - 1
            best, best_f = None, np.inf
            for y in itertools.product(range(top + 1), repeat=n):
                if sum(y) != 3:
                    continue
                y = np.array(y, dtype=float)
                f = y @ sigma @ y - 0.5 * (mu @ y)
                if f < best_f:
                    best, best_f = y, f
            w = choose_weights(mu, sigma, k, 3, PIPE_Q, 0.0)
            assert np.allclose(w * 3, best), "weight-based search disagrees with the raw-unit objective"
    print("Self-check passed: harness picks the same allocations as the pipeline's QUBO objective.\n")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_long_prices():
    if os.path.exists(LONG_CACHE):
        prices = pd.read_csv(LONG_CACHE, index_col=0, parse_dates=True)
        if list(prices.columns) == TICKERS:
            print(f"Loaded cached prices from {LONG_CACHE}")
            return prices
        print("Cached tickers differ from TICKERS; downloading again")
    import yfinance as yf
    print(f"Downloading {START} -> {END} for {', '.join(TICKERS)} ...")
    prices = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)["Close"][TICKERS]
    coverage = prices.notna().mean()
    if len(prices) < 2500 or (coverage < 0.95).any():
        raise SystemExit(
            "Download looks incomplete (Yahoo timeouts are common) -- NOT cached. Coverage per ticker:\n"
            f"{coverage.round(3).to_string()}\nJust run the script again.")
    prices.to_csv(LONG_CACHE)
    return prices


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------
def walk_forward(returns, train_days):
    R = returns.values
    T, n = R.shape
    starts = list(range(train_days, T - TEST_DAYS + 1, TEST_DAYS))
    out = {}
    for name, spec in STRATEGIES.items():
        daily, traded, weights_hist = [], [], []
        w_prev, growth_prev = None, None
        for t in starts:
            train = R[t - train_days:t]
            if spec is None:
                w = np.ones(n) / n
            else:
                mu = train.mean(axis=0) * TRADING_DAYS
                sigma = np.cov(train.T) * TRADING_DAYS
                w = choose_weights(mu, sigma, **spec)
            test = R[t:t + TEST_DAYS]
            p = test @ w
            if w_prev is not None:
                drifted = w_prev * growth_prev / (w_prev @ growth_prev)
                moved = np.abs(w - drifted).sum()
                traded.append(moved)
                p[0] -= COST_BPS / 1e4 * moved
            daily.append(p)
            weights_hist.append(w)
            w_prev, growth_prev = w, np.prod(1 + test, axis=0)
        out[name] = dict(daily=np.concatenate(daily),
                         traded=float(np.mean(traded)) if traded else 0.0,
                         weights=weights_hist)
    first, last = starts[0], starts[-1] + TEST_DAYS - 1
    span = (returns.index[first].date(), returns.index[last].date(), len(starts))
    return out, span


def summarize(r):
    ret = r.mean() * TRADING_DAYS
    vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    cum = np.concatenate([[1.0], np.cumprod(1 + r)])
    mdd = (cum / np.maximum.accumulate(cum) - 1).min()
    return ret, vol, ret / vol, mdd


def ratio_diff_ci(ra, rb, rng):
    """Point estimate and 95% circular-block-bootstrap CI for ratio(a) - ratio(b)."""
    n = len(ra)
    nblocks = int(np.ceil(n / BOOT_BLOCK))
    offs = np.arange(BOOT_BLOCK)
    diffs = np.empty(BOOT_N)
    for b in range(BOOT_N):
        starts = rng.integers(0, n, nblocks)
        idx = ((starts[:, None] + offs[None, :]) % n).ravel()[:n]
        a, c = ra[idx], rb[idx]
        diffs[b] = (a.mean() / a.std(ddof=1) - c.mean() / c.std(ddof=1)) * np.sqrt(TRADING_DAYS)
    point = (ra.mean() / ra.std(ddof=1) - rb.mean() / rb.std(ddof=1)) * np.sqrt(TRADING_DAYS)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, lo, hi


def changes(weights_hist):
    return sum(not np.allclose(a, b) for a, b in zip(weights_hist[:-1], weights_hist[1:]))


# ---------------------------------------------------------------------------
def main():
    self_check()
    prices = load_long_prices()
    miss = prices.isna().sum()
    if miss.any():
        print("WARNING: missing prices (rows with gaps are dropped):")
        print(miss[miss > 0])
    returns = (prices / prices.shift(1) - 1).dropna()
    print(f"{len(returns)} daily returns, {returns.index[0].date()} -> {returns.index[-1].date()}, "
          f"{len(TICKERS)} assets\n")

    rng = np.random.default_rng(SEED)
    res, (d0, d1, nper) = walk_forward(returns, TRAIN_YEARS * TRADING_DAYS)
    print(f"=== Walk-forward: {TRAIN_YEARS}y training window, rebalance every {TEST_DAYS} days, "
          f"{nper} non-overlapping test periods ===")
    print(f"Out-of-sample track: {d0} -> {d1}   (costs: {COST_BPS} bps per unit traded)\n")

    print(f"{'strategy':22s} {'return':>8s} {'vol':>7s} {'ret/vol':>8s} {'maxDD':>8s} {'traded/reb':>11s}")
    for name, r in res.items():
        ret, vol, ratio, mdd = summarize(r["daily"])
        print(f"{name:22s} {ret:8.1%} {vol:7.1%} {ratio:8.2f} {mdd:8.1%} {r['traded']:11.1%}")

    comps = [(n, BASELINE) for n in STRATEGIES if n != BASELINE] + PAIRS
    print("\nDifference in return/vol, 95% block-bootstrap CI (an interval that excludes 0 = distinguishable from noise):")
    for a, b in comps:
        point, lo, hi = ratio_diff_ci(res[a]["daily"], res[b]["daily"], rng)
        verdict = "CI excludes 0" if (lo > 0 or hi < 0) else "CI includes 0"
        print(f"  {a:22s} - {b:20s} {point:+6.2f}  [{lo:+6.2f}, {hi:+6.2f}]  {verdict}")
    p_any = 1 - 0.95 ** len(comps)
    print(f"\n{len(comps)} comparisons at 95%: roughly a {p_any:.0%} chance that at least one looks "
          f"'significant' by luck alone.")

    print("\nAllocation stability (how often the chosen weights changed between rebalances):")
    for name in ("K1 B=3 (pipeline)", "K2 B=3"):
        wh = res[name]["weights"]
        print(f"  {name:20s} changed at {changes(wh)} of {len(wh) - 1} rebalances")

    print("\n=== Robustness: return/vol by training-window length ===")
    by_years = {TRAIN_YEARS: (res, (d0, d1, nper))}
    for ty in ROBUSTNESS_TRAIN_YEARS:
        by_years[ty] = walk_forward(returns, ty * TRADING_DAYS)
    years = sorted(by_years)
    print(f"{'strategy':22s}" + "".join(f"{str(y) + 'y train':>10s}" for y in years))
    for name in STRATEGIES:
        row = "".join(f"{summarize(by_years[y][0][name]['daily'])[2]:10.2f}" for y in years)
        print(f"{name:22s}{row}")
    for y in years:
        e0, e1, n2 = by_years[y][1]
        print(f"  {y}y train: out-of-sample {e0} -> {e1}, {n2} periods")

    print("\nCaveats:")
    print("  - Universe = today's well-known large caps (survivorship / hindsight): flatters every strategy.")
    print("  - Training windows overlap between rebalances, so period results are correlated.")
    print("  - q, budget and shrinkage were fixed in advance, not tuned on the out-of-sample track.")
    print("  - Costs are a flat per-trade rate; return/vol uses the arithmetic mean and no risk-free rate.")
    print("  - One market regime sample (~10 years): treat results as evidence, not proof.")


if __name__ == "__main__":
    main()
