import numpy as np
import pandas as pd
import pytest

from framework.backtest import run_backtest
from framework.backtest.result import TRADE_COLUMNS
from framework.config import RunConfig
from framework.data import load_ohlcv
from framework.strategies import get_strategy

EXPECTED_METRICS = [
    "total_return",
    "cagr",
    "sharpe",
    "sortino",
    "max_drawdown",
    "win_rate",
    "avg_win_loss_ratio",
    "exposure",
    "trade_count",
    "turnover",
]


@pytest.fixture
def strategy():
    return get_strategy("sma_crossover")({"fast": 10, "slow": 50})


def load_test_data(run_config):
    return load_ohlcv(run_config.symbols, data_dir=run_config.data.ohlcv_dir)


@pytest.mark.parametrize("engine", ["vectorbt", "backtesting"])
def test_end_to_end(run_config, strategy, engine):
    result = run_backtest(strategy, load_test_data(run_config), run_config, engine=engine)

    assert result.engine == engine
    assert result.equity_curve.iloc[0] == pytest.approx(
        run_config.portfolio.initial_cash, rel=1e-3
    )
    assert len(result.trades) > 0
    assert list(result.trades.columns) == TRADE_COLUMNS
    for key in EXPECTED_METRICS:
        assert key in result.metrics, f"missing metric {key}"
        assert not np.isnan(result.metrics[key]) or key == "avg_win_loss_ratio"

    # Benchmark ran through the same pipeline.
    assert result.benchmark is not None
    assert result.benchmark.symbol == "BENCH"
    assert "cagr" in result.benchmark.metrics
    assert "alpha" in result.metrics and "beta" in result.metrics

    # Config snapshot is JSON-safe and complete.
    assert result.config["strategy"] == "sma_crossover"
    assert result.config["costs"]["commission_pct"] == 0.0005


def test_engines_agree(run_config, strategy):
    data = load_test_data(run_config)
    r_vbt = run_backtest(strategy, data, run_config, engine="vectorbt", _with_benchmark=False)
    r_bt = run_backtest(strategy, data, run_config, engine="backtesting", _with_benchmark=False)

    assert len(r_vbt.trades) == len(r_bt.trades)
    # Same fill convention -> final equity within 2% (backtesting.py trades
    # whole shares and folds slippage into commission, so exact equality is
    # not expected).
    assert r_vbt.equity_curve.iloc[-1] == pytest.approx(
        r_bt.equity_curve.iloc[-1], rel=0.02
    )
    pd.testing.assert_series_equal(
        r_vbt.trades["entry_time"], r_bt.trades["entry_time"], check_names=False
    )


def test_charter_limits_enforced(run_config):
    """Multi-asset weights get clamped to max_position_pct and gross <= 1."""

    from framework.strategies.base import Strategy, StrategyParams

    class Overweight(Strategy):
        name = "overweight_test"
        params_model = StrategyParams

        def generate_signals(self, data):
            index = next(iter(data.values())).index
            return pd.DataFrame({s: 1.0 for s in data}, index=index)

    data = load_ohlcv(["TEST", "BENCH"], data_dir=run_config.data.ohlcv_dir)
    config = RunConfig(
        strategy="overweight_test",
        symbols=["TEST", "BENCH"],
        data={"ohlcv_dir": str(run_config.data.ohlcv_dir)},
    )
    result = run_backtest(Overweight(), data, config, _with_benchmark=False)

    # Each symbol asked for 100%; charter caps positions at 20% each.
    assert (result.positions.max() <= config.portfolio.max_position_pct + 1e-9).all()
    assert (result.positions.sum(axis=1) <= 1.0 + 1e-9).all()


def test_negative_weights_rejected(run_config):
    from framework.strategies.base import Strategy, StrategyParams

    class Shorter(Strategy):
        name = "shorter_test"
        params_model = StrategyParams

        def generate_signals(self, data):
            df = next(iter(data.values()))
            return pd.DataFrame({"TEST": -0.5}, index=df.index)

    with pytest.raises(ValueError, match="long-only"):
        run_backtest(
            Shorter(), load_test_data(run_config), run_config, _with_benchmark=False
        )


def test_result_save_artifacts(tmp_path, run_config, strategy):
    result = run_backtest(strategy, load_test_data(run_config), run_config)
    out = result.save(tmp_path / "run1")
    for artifact in ["metrics.json", "trades.csv", "equity.parquet", "plot.png"]:
        assert (out / artifact).exists(), f"missing {artifact}"
