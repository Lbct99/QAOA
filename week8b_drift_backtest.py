"""
Week 8b -- Walk-forward backtest with holdings drift between rebalances.
by Luna Copilot
Unlike week8_backtest.py, this script holds each selected portfolio between
scheduled rebalances: holdings drift with daily asset returns, and transaction
costs are charged only when the portfolio is actually reset. It reuses the
original strategy definitions and adds an equal-weight buy-and-hold reference.

Traded notional is the full L1 change in weights (buys plus sells), matching
week8_backtest.py's interpretation of COST_BPS per unit traded. Initial
investment is free, and the first-day rebalance cost is applied multiplicatively.

Run from the repository root:
    python week8b_drift_backtest.py
"""

import numpy as np
import pandas as pd

import week8_backtest as base


BUY_AND_HOLD = "1/N buy-and-hold"
STRATEGIES = {**base.STRATEGIES, BUY_AND_HOLD: "buy-and-hold"}


def drift_weights(weights, asset_returns):
    """Update portfolio weights after one day without trading."""
    growth = 1 + asset_returns
    portfolio_growth = float(weights @ growth)
    if portfolio_growth <= 0:
        raise ValueError("Portfolio value became non-positive; cannot update weights.")
    return weights * growth / portfolio_growth


def gross_traded_notional(old_weights, new_weights):
    """Fraction of portfolio value traded, counting both buys and sells."""
    return float(np.abs(new_weights - old_weights).sum())


def self_check():
    start = np.array([0.5, 0.5])
    after_returns = drift_weights(start, np.array([0.10, 0.0]))
    assert np.allclose(after_returns, [0.55 / 1.05, 0.50 / 1.05])
    expected_gross_traded = 2 * (0.5 - 0.50 / 1.05)
    assert np.isclose(gross_traded_notional(after_returns, start), expected_gross_traded)
    print("Self-check passed: weights drift between rebalances and traded notional counts buys+sells.\n")


def walk_forward_with_drift(returns, train_days):
    values = returns.to_numpy(dtype=float)
    total_days, n_assets = values.shape
    starts = list(range(train_days, total_days - base.TEST_DAYS + 1, base.TEST_DAYS))
    if not starts:
        raise ValueError("Not enough return history for the requested training and test windows.")

    result = {}
    for name, spec in STRATEGIES.items():
        daily_returns = []
        turnovers = []
        target_history = []
        holdings = None

        for start in starts:
            train = values[start - train_days:start]
            if spec == "buy-and-hold":
                target = np.full(n_assets, 1 / n_assets) if holdings is None else holdings.copy()
            elif spec is None:
                target = np.full(n_assets, 1 / n_assets)
            else:
                mu = train.mean(axis=0) * base.TRADING_DAYS
                sigma = np.cov(train.T) * base.TRADING_DAYS
                target = base.choose_weights(mu, sigma, **spec)

            cost_fraction = 0.0
            if holdings is None:
                holdings = target.copy()
            elif spec != "buy-and-hold":
                turnover = gross_traded_notional(holdings, target)
                turnovers.append(turnover)
                cost_fraction = base.COST_BPS / 1e4 * turnover
                holdings = target.copy()

            target_history.append(target.copy())
            for day_index, asset_returns in enumerate(
                values[start:start + base.TEST_DAYS]
            ):
                gross_return = float(holdings @ asset_returns)
                if day_index == 0 and cost_fraction:
                    net_return = (1 - cost_fraction) * (1 + gross_return) - 1
                else:
                    net_return = gross_return
                daily_returns.append(net_return)
                holdings = drift_weights(holdings, asset_returns)

        result[name] = {
            "daily": np.asarray(daily_returns),
            "traded": float(np.mean(turnovers)) if turnovers else 0.0,
            "weights": target_history,
        }

    first, last = starts[0], starts[-1] + base.TEST_DAYS - 1
    span = (returns.index[first].date(), returns.index[last].date(), len(starts))
    return result, span


def summarize_with_cagr(daily_returns):
    annual_mean, volatility, ratio, drawdown = base.summarize(daily_returns)
    wealth = float(np.prod(1 + daily_returns))
    cagr = wealth ** (base.TRADING_DAYS / len(daily_returns)) - 1
    return annual_mean, cagr, volatility, ratio, drawdown


def main():
    self_check()
    prices = base.load_long_prices()
    returns = (prices / prices.shift(1) - 1).dropna()
    print(f"{len(returns)} daily returns, {returns.index[0].date()} -> "
          f"{returns.index[-1].date()}, {len(base.TICKERS)} assets\n")

    rng = np.random.default_rng(base.SEED)
    results, (first, last, periods) = walk_forward_with_drift(
        returns, base.TRAIN_YEARS * base.TRADING_DAYS
    )
    print(
        f"=== Drift-aware walk-forward: {base.TRAIN_YEARS}y training, "
        f"rebalance every {base.TEST_DAYS} days, {periods} test periods ==="
    )
    print(f"Out-of-sample track: {first} -> {last}; "
          f"{base.COST_BPS} bps per dollar traded; initial investment free\n")
    print(
        f"{'strategy':22s} {'ann.mean':>9s} {'CAGR':>8s} {'vol':>7s} "
        f"{'mean/vol':>9s} {'maxDD':>8s} {'traded/reb':>10s}"
    )
    for name, output in results.items():
        ann_mean, cagr, volatility, ratio, drawdown = summarize_with_cagr(output["daily"])
        print(
            f"{name:22s} {ann_mean:9.1%} {cagr:8.1%} {volatility:7.1%} "
            f"{ratio:9.2f} {drawdown:8.1%} {output['traded']:10.1%}"
        )

    comparisons = [
        (name, base.BASELINE) for name in STRATEGIES if name != base.BASELINE
    ] + base.PAIRS
    print(
        "\nDifference in annualized mean/vol, 95% paired circular-block-bootstrap CI "
        "(interval excluding 0 is evidence of a difference):"
    )
    for left, right in comparisons:
        point, low, high = base.ratio_diff_ci(
            results[left]["daily"], results[right]["daily"], rng
        )
        verdict = "excludes 0" if low > 0 or high < 0 else "includes 0"
        print(
            f"  {left:22s} - {right:20s} {point:+6.2f} "
            f"[{low:+6.2f}, {high:+6.2f}]  CI {verdict}"
        )
    chance_any = 1 - 0.95 ** len(comparisons)
    print(
        f"\n{len(comparisons)} unadjusted 95% comparisons: approximate "
        f"family-wise false-positive chance {chance_any:.0%}."
    )

    print("\nAllocation changes at scheduled rebalances:")
    for name in ("K1 B=3 (pipeline)", "K2 B=3"):
        weights = results[name]["weights"]
        print(f"  {name:20s} {base.changes(weights)} of {len(weights) - 1}")

    print("\n=== Robustness: annualized mean/vol by training-window length ===")
    by_years = {base.TRAIN_YEARS: results}
    for years in base.ROBUSTNESS_TRAIN_YEARS:
        by_years[years], _ = walk_forward_with_drift(
            returns, years * base.TRADING_DAYS
        )
    years_to_show = sorted(by_years)
    print(
        f"{'strategy':22s}"
        + "".join(f"{str(year) + 'y train':>10s}" for year in years_to_show)
    )
    for name in STRATEGIES:
        ratios = "".join(
            f"{summarize_with_cagr(by_years[year][name]['daily'])[3]:10.2f}"
            for year in years_to_show
        )
        print(f"{name:22s}{ratios}")

    print("\nMethod notes:")
    print("  - Selected target weights are held between scheduled rebalances; positions drift daily.")
    print("  - Rebalance costs are applied to gross traded notional (buys plus sells); initial investment is free.")
    print("  - 1/N (all assets) is reset to equal weights each rebalance; buy-and-hold is not.")
    print("  - Universe, overlapping training windows, short history, and flat trading costs")
    print("    retain the original script's survivorship and inference limitations.")


if __name__ == "__main__":
    main()
