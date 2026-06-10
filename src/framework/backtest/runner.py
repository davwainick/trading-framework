"""The backtest runner: one pipeline for any strategy and any engine.

Pipeline: validate weights -> clamp to charter limits -> apply the
look-ahead shift -> dispatch to an engine (fills at bar OPEN with costs) ->
compute engine-independent metrics -> run the buy-and-hold benchmark through
the identical pipeline -> assemble a BacktestResult.

The ``weights.shift(1)`` here is the framework's single look-ahead guard:
a signal computed from bar t's close is executed at bar t+1's open. It lives
in exactly one place so there is one place to test and one place it can break.
"""

from __future__ import annotations

import logging

import pandas as pd

from framework.backtest import metrics as m
from framework.backtest.result import BacktestResult, BenchmarkResult
from framework.config import RunConfig
from framework.data import DataNotFoundError, load_ohlcv
from framework.strategies.base import Strategy

logger = logging.getLogger(__name__)

_ENGINES = {}


def _get_engine(name: str):
    if name not in _ENGINES:
        if name == "vectorbt":
            from framework.backtest.engines import vbt_engine

            _ENGINES[name] = vbt_engine.run
        elif name == "backtesting":
            from framework.backtest.engines import btpy_engine

            _ENGINES[name] = btpy_engine.run
        else:
            raise ValueError(f"Unknown engine {name!r}; use 'vectorbt' or 'backtesting'")
    return _ENGINES[name]


def run_backtest(
    strategy: Strategy,
    data: dict[str, pd.DataFrame],
    config: RunConfig,
    engine: str | None = None,
    _with_benchmark: bool = True,
) -> BacktestResult:
    engine_name = engine or config.backtest.engine
    engine_fn = _get_engine(engine_name)

    raw_weights = strategy.generate_signals(data)
    weights = _prepare_weights(raw_weights, config)

    # THE look-ahead guard: signal at bar t trades at bar t+1.
    shifted = weights.shift(1).fillna(0.0)

    equity, trades = engine_fn(shifted, data, config)
    returns = equity.pct_change().fillna(0.0)

    benchmark = _run_benchmark(config, engine_name) if _with_benchmark else None
    bench_returns = benchmark.returns if benchmark is not None else None

    return BacktestResult(
        strategy_name=getattr(type(strategy), "name", type(strategy).__name__),
        params=strategy.params.model_dump(),
        engine=engine_name,
        start=equity.index[0],
        end=equity.index[-1],
        equity_curve=equity,
        returns=returns,
        positions=shifted,
        trades=trades,
        metrics=m.compute_metrics(equity, returns, trades, shifted, bench_returns),
        benchmark=benchmark,
        config=config.resolved_dict(),
    )


def _prepare_weights(weights: pd.DataFrame, config: RunConfig) -> pd.DataFrame:
    """Validate strategy output and enforce charter limits centrally."""
    weights = weights.astype(float).fillna(0.0)
    if (weights < 0).any().any():
        raise ValueError("Negative weights not supported (long-only framework, Phase 1)")

    limits = config.portfolio

    # Max concurrent positions: keep the largest-weight names per bar.
    active = (weights > 0).sum(axis=1)
    if (active > limits.max_positions).any():
        ranks = weights.rank(axis=1, method="first", ascending=False)
        weights = weights.where(ranks <= limits.max_positions, 0.0)

    # Per-position cap — skipped for single-asset runs, where diversification
    # is impossible and capping would just leave the portfolio 80% idle.
    if len(weights.columns) > 1:
        weights = weights.clip(upper=limits.max_position_pct)

    # No leverage: scale down any bar whose gross weight exceeds 1.
    gross = weights.sum(axis=1)
    if not limits.allow_leverage and (gross > 1.0).any():
        scale = gross.clip(lower=1.0)
        weights = weights.div(scale, axis=0)

    return weights


def _run_benchmark(config: RunConfig, engine_name: str) -> BenchmarkResult | None:
    """Buy-and-hold of the benchmark symbol: same engine, same costs, same metrics."""
    symbol = config.benchmark.symbol
    try:
        data = load_ohlcv([symbol], config.start, config.end, config.data.ohlcv_dir)
    except DataNotFoundError:
        logger.warning("Benchmark data for %s not found; skipping comparison", symbol)
        return None

    index = data[symbol].index
    weights = pd.DataFrame({symbol: 1.0}, index=index)
    # Same shift discipline as the strategy: invested from bar 1's open.
    shifted = weights.shift(1).fillna(0.0)

    engine_fn = _get_engine(engine_name)
    equity, trades = engine_fn(shifted, data, config)
    returns = equity.pct_change().fillna(0.0)
    return BenchmarkResult(
        symbol=symbol,
        equity_curve=equity,
        returns=returns,
        metrics=m.compute_metrics(equity, returns, trades, shifted),
    )
