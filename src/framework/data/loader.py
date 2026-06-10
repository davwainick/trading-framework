"""The single entry point for loading market data.

Strategies and the backtest runner only ever receive data through
:func:`load_ohlcv`; nothing else in the framework touches the network or the
filesystem for prices. Phase 2's download layer writes Parquet files to the
same ``{data_dir}/{SYMBOL}.parquet`` contract, so swapping data vendors later
(e.g. yfinance -> Norgate) changes nothing downstream.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from framework.config import REPO_ROOT

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "ohlcv"
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataNotFoundError(FileNotFoundError):
    """Raised when a symbol's Parquet file is missing from the data directory."""


def load_ohlcv(
    symbols: list[str],
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    data_dir: Path | str | None = None,
) -> dict[str, pd.DataFrame]:
    """Load OHLCV history for ``symbols`` from ``{data_dir}/{SYMBOL}.parquet``.

    Returns a dict mapping symbol -> DataFrame with guaranteed shape:
    lowercase columns ``open, high, low, close, volume``, a tz-aware (UTC),
    sorted, de-duplicated ``DatetimeIndex``, sliced to ``[start, end]``.
    """
    directory = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    if not directory.is_absolute():
        directory = REPO_ROOT / directory
    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = directory / f"{symbol.upper()}.parquet"
        if not path.exists():
            raise DataNotFoundError(
                f"No data file for {symbol!r} at {path}. "
                "Run `uv run python scripts/bootstrap_data.py` to fetch sample data."
            )
        out[symbol.upper()] = _normalize(pd.read_parquet(path), symbol, start, end)
    return out


def _normalize(
    df: pd.DataFrame,
    symbol: str,
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None,
) -> pd.DataFrame:
    df = df.rename(columns=str.lower)
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{symbol}: data file is missing columns {missing}")
    df = df[OHLCV_COLUMNS]

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{symbol}: data file must have a DatetimeIndex")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df[~df.index.duplicated(keep="last")].sort_index()

    if start is not None:
        df = df.loc[pd.Timestamp(start, tz="UTC") :]
    if end is not None:
        df = df.loc[: pd.Timestamp(end, tz="UTC")]
    return df
