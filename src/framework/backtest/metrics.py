"""Engine-independent performance metrics.

The engines (vectorbt / backtesting.py) only produce raw outputs — an equity
curve and a trade list. All metrics are computed here from those raw outputs,
with one definition each, so results from different engines are directly
comparable and the same numbers carry through validation (Phase 5), paper
trading (Phase 7), and live monitoring (Phase 10).

All functions assume daily bars (TRADING_DAYS = 252 periods/year) and a 0%
risk-free rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cagr(equity: pd.Series) -> float:
    """Compound annual growth rate of an equity curve."""
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return float("nan")
    years = len(equity) / TRADING_DAYS
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def sharpe(returns: pd.Series) -> float:
    """Annualized Sharpe ratio (rf = 0)."""
    std = returns.std()
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(returns.mean() / std * np.sqrt(TRADING_DAYS))


def sortino(returns: pd.Series) -> float:
    """Annualized Sortino ratio (rf = 0, downside deviation denominator)."""
    downside = returns[returns < 0]
    if len(downside) == 0:
        return float("inf") if returns.mean() > 0 else float("nan")
    dd = np.sqrt((downside**2).mean())
    if dd == 0:
        return float("nan")
    return float(returns.mean() / dd * np.sqrt(TRADING_DAYS))


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown, returned as a negative fraction."""
    running_max = equity.cummax()
    drawdowns = equity / running_max - 1
    return float(drawdowns.min())


def win_rate(trades: pd.DataFrame) -> float:
    """Fraction of closed trades with positive PnL."""
    if len(trades) == 0:
        return float("nan")
    return float((trades["pnl"] > 0).mean())


def avg_win_loss_ratio(trades: pd.DataFrame) -> float:
    """Average winning-trade PnL divided by average losing-trade |PnL|."""
    wins = trades.loc[trades["pnl"] > 0, "pnl"]
    losses = trades.loc[trades["pnl"] < 0, "pnl"]
    if len(wins) == 0 or len(losses) == 0:
        return float("nan")
    return float(wins.mean() / abs(losses.mean()))


def exposure(positions: pd.DataFrame) -> float:
    """Fraction of bars with any non-zero position."""
    if len(positions) == 0:
        return float("nan")
    return float((positions.abs().sum(axis=1) > 0).mean())


def turnover(positions: pd.DataFrame) -> float:
    """Mean annualized one-way turnover: sum |weight changes| / 2, per year."""
    if len(positions) < 2:
        return float("nan")
    daily = positions.diff().abs().sum(axis=1) / 2
    return float(daily.mean() * TRADING_DAYS)


def alpha_beta(returns: pd.Series, benchmark_returns: pd.Series) -> tuple[float, float]:
    """Annualized alpha and beta vs the benchmark, via OLS on daily returns."""
    joined = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(joined) < 2:
        return float("nan"), float("nan")
    y, x = joined.iloc[:, 0].to_numpy(), joined.iloc[:, 1].to_numpy()
    if np.var(x) == 0:
        return float("nan"), float("nan")
    beta, intercept = np.polyfit(x, y, 1)
    return float(intercept * TRADING_DAYS), float(beta)


def compute_metrics(
    equity: pd.Series,
    returns: pd.Series,
    trades: pd.DataFrame,
    positions: pd.DataFrame,
    benchmark_returns: pd.Series | None = None,
) -> dict[str, float]:
    """The standard metric set every BacktestResult carries."""
    out = {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "cagr": cagr(equity),
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "max_drawdown": max_drawdown(equity),
        "win_rate": win_rate(trades),
        "avg_win_loss_ratio": avg_win_loss_ratio(trades),
        "exposure": exposure(positions),
        "trade_count": float(len(trades)),
        "turnover": turnover(positions),
    }
    if benchmark_returns is not None:
        out["alpha"], out["beta"] = alpha_beta(returns, benchmark_returns)
    return out
