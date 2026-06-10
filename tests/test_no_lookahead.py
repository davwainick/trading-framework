"""The tests that keep backtests honest.

1. Prefix consistency: a strategy's signal at bar t must not change when
   future bars are appended — i.e. signals depend only on the past.
2. Next-open execution: the runner must fill a bar-t signal at bar t+1's
   open (with slippage), never on bar t itself.
"""

import numpy as np
import pandas as pd
import pytest

from framework.backtest import run_backtest
from framework.config import RunConfig
from framework.strategies import get_strategy
from framework.strategies.base import Strategy, StrategyParams


def test_signals_are_prefix_consistent(synthetic_ohlcv):
    """generate_signals(data[:t]) must equal the first t rows of generate_signals(data)."""
    strat = get_strategy("sma_crossover")({"fast": 10, "slow": 50})
    full = strat.generate_signals({"X": synthetic_ohlcv})["X"]
    for t in [60, 150, 300, 599]:
        prefix = strat.generate_signals({"X": synthetic_ohlcv.iloc[:t]})["X"]
        pd.testing.assert_series_equal(prefix, full.iloc[:t], check_names=False)


class ImpulseStrategy(Strategy):
    """Goes long on exactly one bar (by integer position), flat otherwise."""

    name = "impulse_test"
    params_model = StrategyParams

    def __init__(self, signal_bar: int, exit_bar: int):
        super().__init__()
        self.signal_bar = signal_bar
        self.exit_bar = exit_bar

    def generate_signals(self, data):
        df = next(iter(data.values()))
        symbol = next(iter(data.keys()))
        weights = pd.Series(0.0, index=df.index)
        weights.iloc[self.signal_bar : self.exit_bar] = 1.0
        return weights.to_frame(symbol)


@pytest.mark.parametrize("engine", ["vectorbt", "backtesting"])
def test_signal_fills_at_next_bars_open(synthetic_ohlcv, engine):
    signal_bar = 100
    config = RunConfig(
        strategy="impulse_test",
        symbols=["X"],
        costs={"commission_pct": 0.0, "slippage_pct": 0.001},
    )
    strat = ImpulseStrategy(signal_bar=signal_bar, exit_bar=200)
    result = run_backtest(
        strat, {"X": synthetic_ohlcv}, config, engine=engine, _with_benchmark=False
    )

    # No position on the signal bar itself: equity unchanged through bar t.
    assert result.equity_curve.iloc[signal_bar] == pytest.approx(
        config.portfolio.initial_cash
    )

    trade = result.trades.iloc[0]
    expected_fill_time = synthetic_ohlcv.index[signal_bar + 1]
    next_open = synthetic_ohlcv["open"].iloc[signal_bar + 1]
    # vectorbt marks slippage into the fill price; backtesting.py reports the
    # raw price and charges slippage/commission against cash instead.
    expected_fill_price = next_open * (1 + 0.001) if engine == "vectorbt" else next_open
    assert trade["entry_time"] == expected_fill_time
    assert trade["entry_price"] == pytest.approx(expected_fill_price, rel=1e-6)


def test_runner_shift_means_last_bar_signal_never_trades(synthetic_ohlcv):
    """A signal on the final bar has no next open — it must produce no trade."""
    n = len(synthetic_ohlcv)
    config = RunConfig(strategy="impulse_test", symbols=["X"])
    strat = ImpulseStrategy(signal_bar=n - 1, exit_bar=n)
    result = run_backtest(
        strat, {"X": synthetic_ohlcv}, config, engine="vectorbt", _with_benchmark=False
    )
    assert len(result.trades) == 0
    assert np.allclose(result.equity_curve, config.portfolio.initial_cash)
