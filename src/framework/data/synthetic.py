"""Deterministic synthetic OHLCV generation.

Used by the test suite (hermetic, no network) and by
``scripts/bootstrap_data.py --synthetic`` as an offline fallback.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_ohlcv(
    n_bars: int = 1000,
    seed: int = 42,
    start: str = "2015-01-01",
    initial_price: float = 100.0,
    trend: float = 0.0002,
    vol: float = 0.01,
) -> pd.DataFrame:
    """Generate a seeded geometric-random-walk daily OHLCV DataFrame.

    ``trend`` and ``vol`` are per-bar drift and standard deviation of log
    returns. Output matches the loader contract: lowercase OHLCV columns and a
    UTC DatetimeIndex (business days).
    """
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(trend, vol, n_bars)
    close = initial_price * np.exp(np.cumsum(log_returns))

    # Build plausible open/high/low around the close path.
    open_ = np.empty(n_bars)
    open_[0] = initial_price
    open_[1:] = close[:-1] * np.exp(rng.normal(0, vol / 4, n_bars - 1))
    wiggle = np.abs(rng.normal(0, vol / 2, n_bars))
    high = np.maximum(open_, close) * (1 + wiggle)
    low = np.minimum(open_, close) * (1 - wiggle)
    volume = rng.integers(1_000_000, 10_000_000, n_bars).astype(float)

    index = pd.bdate_range(start=start, periods=n_bars, tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def make_multi_ohlcv(
    trends: dict[str, float],
    n_bars: int = 600,
    seed: int = 42,
    start: str = "2015-01-01",
    vol: float = 0.01,
) -> dict[str, pd.DataFrame]:
    """One seeded OHLCV frame per symbol on a shared business-day index.

    ``trends`` maps symbol -> per-bar drift, so tests can engineer relative
    momentum (winners vs losers, or an all-loser universe) deterministically.
    Each symbol's seed is derived from ``seed`` and its position in the dict,
    keeping the frames independent but the whole dict reproducible.
    """
    return {
        symbol: make_ohlcv(
            n_bars=n_bars, seed=seed + i, start=start, trend=trend, vol=vol
        )
        for i, (symbol, trend) in enumerate(trends.items())
    }


def make_trending_ohlcv(
    n_bars: int = 400,
    seed: int = 7,
    start: str = "2015-01-01",
) -> pd.DataFrame:
    """An up-then-down series engineered to force SMA crossovers at known points.

    The first half trends strongly up, the second half strongly down, so a
    fast/slow SMA pair will cross long somewhere in the first half and cross
    back flat in the second half. Tests compute the exact crossover bars from
    the data itself.
    """
    half = n_bars // 2
    rng = np.random.default_rng(seed)
    up = np.full(half, 0.004) + rng.normal(0, 0.001, half)
    down = np.full(n_bars - half, -0.004) + rng.normal(0, 0.001, n_bars - half)
    log_returns = np.concatenate([up, down])
    close = 100.0 * np.exp(np.cumsum(log_returns))

    open_ = np.empty(n_bars)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    volume = np.full(n_bars, 1_000_000.0)

    index = pd.bdate_range(start=start, periods=n_bars, tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )
