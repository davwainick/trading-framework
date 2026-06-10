"""Backtesting: runner, engines, result objects, and metric functions."""

from framework.backtest.result import BacktestResult, BenchmarkResult
from framework.backtest.runner import run_backtest

__all__ = ["BacktestResult", "BenchmarkResult", "run_backtest"]
