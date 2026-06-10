"""Data source adapters.

Every source implements the :class:`DataSource` protocol and returns data in
the loader contract (lowercase OHLCV columns, UTC DatetimeIndex, split- and
dividend-adjusted prices). Downstream code — the download pipeline, quality
checks, loader, strategies — never knows which vendor produced a file, which
is what makes swapping in a paid survivorship-bias-free source (e.g. Norgate)
a one-adapter change. See docs/data.md for the swap path.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class DataSource(Protocol):
    """A vendor adapter: fetch adjusted daily OHLCV for one symbol."""

    name: str

    def fetch(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Return OHLCV with lowercase columns and a UTC DatetimeIndex.

        Prices must be split- AND dividend-adjusted (total-return prices).
        Raises SourceError if the vendor returns nothing usable.
        """
        ...


class SourceError(RuntimeError):
    """A vendor returned no data or unusable data for a symbol."""


_SOURCES: dict[str, type] = {}


def register_source(cls):
    _SOURCES[cls.name] = cls
    return cls


def get_source(name: str) -> DataSource:
    try:
        return _SOURCES[name]()
    except KeyError:
        raise KeyError(f"Unknown data source {name!r}. Available: {sorted(_SOURCES)}") from None


from framework.data.sources import yfinance_source as _yf  # noqa: E402,F401

__all__ = ["DataSource", "SourceError", "register_source", "get_source"]
