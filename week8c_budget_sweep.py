"""
Week 8c -- Budget sweep with a random-selection baseline.

Question: the K=1, B=3 pipeline model trails 1/N. Is that because holding only
a few stocks is costly (concentration), or because the optimizer's choice of
WHICH stocks is poor (selection)? A shrinking gap as the budget grows toward
10 cannot answer this alone, because budget 10 IS 1/N by construction.

So each budget B is compared against two things:
  1. 1/N                           -> total gap
  2. random B-stock selections     -> same concentration, no optimizer
Then:
  total gap = concentration effect + selection effect
  concentration effect = (typical random B-stock portfolio) - (1/N)
  selection effect     = (optimizer's B-stock portfolio) - (typical random one)

Everything uses the exact accounting of week8_backtest.py (drift-aware weights,
10 bps costs, 3-year training window, 126-day rebalances); a consistency check
below confirms this script reproduces its K=1, B=3 numbers. Random strategies
re-draw their stocks at every rebalance and pay the same costs. The risk
weight q stays fixed on portfolio weights across budgets, so strategies with
different budgets stay comparable.

Run (from the project root, next to week8_backtest.py and prices_long_cache.csv):
    python week8c_budget_sweep.py
"""

import numpy as np

import week8_backtest as wb

BUDGETS = [2, 3, 4, 5, 6, 7, 8, 9]
N_RANDOM = 500            # random-selection strategies per budget
ROBUSTNESS_YEARS = [2, 3, 5]


def run(R, train_days, chooser):
    """Same drift-aware walk-forward accounting as wb.walk_forward, for one strategy.
    chooser(train_returns) -> weight vector. Returns (daily returns, avg traded per rebalance)."""
    T, n = R.shape
    starts = list(range(train_days, T - wb.TEST_DAYS + 1, wb.TEST_DAYS))
    daily, traded = [], []
    w_prev = growth_prev = None
    for t in starts:
        w = chooser(R[t - train_days:t])
        test = R[t:t + wb.TEST_DAYS]
        p = wb.drift_returns(test, w)
        if w_prev is not None:
            drifted = w_prev * growth_prev / (w_prev @ growth_prev)
            moved = np.abs(w - drifted).sum()
            traded.append(moved)
            p[0] -= wb.COST_BPS / 1e4 * moved
        daily.append(p)
        w_prev, growth_prev = w, np.prod(1 + test, axis=0)
    return np.concatenate(daily), float(np.mean(traded)) if traded else 0.0


def optimized(budget):
    def choose(train):
        mu = train.mean(axis=0) * wb.TRADING_DAYS
        sigma = np.cov(train.T) * wb.TRADING_DAYS
        return wb.choose_weights(mu, sigma, 1, budget, wb.PIPE_Q, 0.0)
    return choose


def random_selection(budget, rng):
    def choose(train):
        n = train.shape[1]
        w = np.zeros(n)
        w[rng.choice(n, budget, replace=False)] = 1.0 / budget
        return w
    return choose


def equal_weight(train):
    return np.ones(train.shape[1]) / train.shape[1]


def ratio(daily):
    return wb.summarize(daily)[2]


def main():
    prices = wb.load_long_prices()
    returns = (prices / prices.shift(1) - 1).dropna()
    R = returns.values
    train_days = wb.TRAIN_YEARS * wb.TRADING_DAYS

    # Consistency check against the verified harness
    ref, _ = wb.walk_forward(returns, train_days)
    mine, _ = run(R, train_days, optimized(3))
    assert np.allclose(mine, ref["K1 B=3 (pipeline)"]["daily"]), "does not reproduce week8_backtest.py"
    print("Consistency check passed: reproduces week8_backtest.py's K=1, B=3 return series.\n")

    base, base_traded = run(R, train_days, equal_weight)
    base_ratio = ratio(base)
    sel_rng = np.random.default_rng(wb.SEED)
    boot_rng = np.random.default_rng(wb.SEED + 1)

    print(f"=== Budget sweep: {wb.TRAIN_YEARS}y training, {N_RANDOM} random-selection strategies per budget ===")
    print(f"1/N (all 10): mean/vol {base_ratio:.2f}\n")
    print(f"{'B':>2s} {'optimized':>10s} {'random median [5%, 95%]':>26s} {'opt. rank':>10s} "
          f"{'opt - 1/N  [95% CI]':>28s} {'traded/reb':>11s}")
    rows = []
    for b in BUDGETS:
        opt, traded = run(R, train_days, optimized(b))
        r_opt = ratio(opt)
        rand = np.array([ratio(run(R, train_days, random_selection(b, sel_rng))[0])
                         for _ in range(N_RANDOM)])
        med, lo5, hi95 = np.median(rand), *np.percentile(rand, [5, 95])
        rank = float(np.mean(rand < r_opt))
        diff, lo, hi = wb.ratio_diff_ci(opt, base, boot_rng)
        rows.append((b, r_opt, med, diff))
        print(f"{b:2d} {r_opt:10.2f} {med:8.2f} [{lo5:5.2f}, {hi95:5.2f}] {rank:9.0%} "
              f"{diff:+9.2f} [{lo:+6.2f}, {hi:+6.2f}] {traded:11.1%}")

    p_any = 1 - 0.95 ** len(BUDGETS)
    print(f"\n'opt. rank' = share of random-selection strategies the optimizer beat "
          f"(50% = no better than random).")
    print(f"{len(BUDGETS)} intervals at 95%: roughly a {p_any:.0%} chance that at least one looks "
          f"'significant' by luck alone.\n")

    print("Decomposition of the gap to 1/N (mean/vol points):")
    print(f"{'B':>2s} {'total':>8s} {'concentration':>14s} {'selection':>10s}")
    for b, r_opt, med, diff in rows:
        print(f"{b:2d} {diff:+8.2f} {med - base_ratio:+14.2f} {r_opt - med:+10.2f}")

    print("\n=== Robustness: optimized mean/vol by training-window length ===")
    print(f"{'B':>5s}" + "".join(f"{str(y) + 'y train':>10s}" for y in ROBUSTNESS_YEARS))
    for b in BUDGETS:
        vals = [ratio(run(R, y * wb.TRADING_DAYS, optimized(b))[0]) for y in ROBUSTNESS_YEARS]
        print(f"{b:5d}" + "".join(f"{v:10.2f}" for v in vals))
    vals = [ratio(run(R, y * wb.TRADING_DAYS, equal_weight)[0]) for y in ROBUSTNESS_YEARS]
    print(f"{'1/N':>5s}" + "".join(f"{v:10.2f}" for v in vals))

    print("\nCaveats:")
    print("  - Ranks and random ranges come from one realized price history; they describe this")
    print("    sample, not a general law. With 500 draws the rank itself is good to about +-2 points.")
    print("  - Same hindsight/survivorship universe as week8_backtest.py (today's large caps).")
    print("  - Budget 10 would equal 1/N; budgets near 10 differ from it only by one or two stocks.")


if __name__ == "__main__":
    main()
