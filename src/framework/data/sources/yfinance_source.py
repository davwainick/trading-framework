"""Yahoo Finance adapter (free tier).

Known limitations — fine for ETFs and harness development, NOT sufficient for
single-name equity research (see docs/data.md):

- Survivorship bias: delisted tickers simply disappear; no point-in-time
  index membership.
- ``auto_adjust=True`` back-adjusts for splits and dividends (total-return
  prices), but Yahoo's adjustment history occasionally has errors and can be
  silently revised between downloads.
- Occasional gaps/bad prints; an incomplete trailing bar (NaN prices) appears
  when downloading during/just after a session.
- Intraday data only goes back ~730 days (1h) / 60 days (5m–30m) / 30 days (1m).
"""

from __future__ import annotations

import pandas as pd

from framework.data.sources import SourceError, register_source


@register_source
class YFinanceSource:
    name = "yfinance"

    def fetch(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        import yfinance as yf

        df = yf.download(
            symbol,
            start=start,
            end=end,
            period=None if start else "max",
            interval="1d",
            auto_adjust=True,  # split- AND dividend-adjusted (total return)
            progress=False,
        )
        if df is None or len(df) == 0:
            raise SourceError(f"yfinance returned no data for {symbol!r}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df = df[["open", "high", "low", "close", "volume"]].sort_index()
        # Yahoo can return an incomplete trailing bar (NaN prices) for today.
        df = df.dropna(subset=["open", "high", "low", "close"])
        if len(df) == 0:
            raise SourceError(f"yfinance returned only unusable rows for {symbol!r}")
        return df
