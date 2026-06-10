"""One-off bootstrap so the Phase 1 example has data to run on.

Fetches daily, split/dividend-adjusted history via yfinance and writes it
through the loader's Parquet contract. Phase 2 replaces this with the real
data-acquisition layer (quality checks, more sources); nothing downstream
changes because the file contract is the same.

Usage:
    uv run python scripts/bootstrap_data.py [SYMBOL ...] [--synthetic]

yfinance caveats (fine for wiring up the harness, not for research): Yahoo
data has occasional gaps/adjustment quirks and no survivorship-bias control.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from framework.config import REPO_ROOT
from framework.data import make_ohlcv

DATA_DIR = REPO_ROOT / "data" / "ohlcv"


def fetch_yfinance(symbol: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(
        symbol, period="max", interval="1d", auto_adjust=True, progress=False
    )
    if df is None or len(df) == 0:
        raise RuntimeError(f"yfinance returned no data for {symbol!r}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df[["open", "high", "low", "close", "volume"]].sort_index()
    # Yahoo can return an incomplete trailing bar (NaN prices) for today.
    return df.dropna(subset=["open", "high", "low", "close"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbols", nargs="*", default=None, help="Symbols (default: SPY)")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate seeded synthetic data instead of downloading (offline fallback)",
    )
    args = parser.parse_args()
    symbols = [s.upper() for s in (args.symbols or ["SPY"])]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        if args.synthetic:
            df = make_ohlcv(n_bars=2500, seed=hash(symbol) % 2**32, trend=0.0003)
            source = "synthetic"
        else:
            try:
                df = fetch_yfinance(symbol)
                source = "yfinance"
            except Exception as exc:  # noqa: BLE001 - report and continue with other symbols
                print(f"{symbol}: download failed ({exc}). Try --synthetic for offline use.")
                continue
        path = DATA_DIR / f"{symbol}.parquet"
        df.to_parquet(path)
        print(
            f"{symbol}: {len(df)} bars [{df.index[0].date()} -> {df.index[-1].date()}] "
            f"({source}) -> {path.relative_to(REPO_ROOT)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
