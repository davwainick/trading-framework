"""backtesting.py engine adapter — the single-symbol cross-check.

Event-driven second opinion on the vectorbt numbers: it walks the same
pre-shifted weights bar by bar and trades on weight transitions. With
``trade_on_close=False`` (the library default) orders placed in ``next()``
fill at the NEXT bar's open, so the adapter reads the weight one bar AHEAD
when deciding — the fill then lands exactly on that weight's bar, matching
the vectorbt convention.

Limitations (documented, by design): one symbol, long-only, all-in/all-out
(weight > 0 means fully invested). Use it to sanity-check fills, not for
multi-asset portfolios.
"""

from __future__ import annotations

import pandas as pd

from framework.backtest.result import TRADE_COLUMNS
from framework.config import RunConfig


def run(
    weights: pd.DataFrame,
    data: dict[str, pd.DataFrame],
    config: RunConfig,
) -> tuple[pd.Series, pd.DataFrame]:
    """Execute pre-shifted target weights. Returns (equity_curve, trades)."""
    from backtesting import Backtest, Strategy as BtStrategy

    if len(weights.columns) != 1:
        raise ValueError(
            "The backtesting.py engine supports exactly one symbol "
            f"(got {list(weights.columns)}). Use engine='vectorbt' for portfolios."
        )
    symbol = weights.columns[0]
    target = weights[symbol].fillna(0.0)

    ohlcv = data[symbol].reindex(weights.index)
    bt_data = ohlcv.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )

    # Single per-side cost approximation: backtesting.py has one commission
    # knob, so slippage is folded into it.
    commission = config.costs.commission_pct + config.costs.slippage_pct

    class WeightFollower(BtStrategy):
        def init(self):
            pass

        def next(self):
            i = len(self.data) - 1
            if i + 1 >= len(target):
                return
            # Orders fill at bar i+1's open; aim for that bar's target weight.
            want_long = target.iloc[i + 1] > 0
            if want_long and not self.position:
                self.buy()
            elif not want_long and self.position:
                self.position.close()

    bt = Backtest(
        bt_data,
        WeightFollower,
        cash=config.portfolio.initial_cash,
        commission=commission,
        trade_on_close=False,
        exclusive_orders=False,
        finalize_trades=True,
    )
    stats = bt.run()

    equity = stats["_equity_curve"]["Equity"]
    equity.index = weights.index[: len(equity)]
    equity.name = "equity"

    trades = _normalize_trades(stats["_trades"], symbol)
    return equity, trades


def _normalize_trades(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if len(raw) == 0:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    trades = pd.DataFrame(
        {
            "symbol": symbol,
            "entry_time": pd.to_datetime(raw["EntryTime"]),
            "exit_time": pd.to_datetime(raw["ExitTime"]),
            "entry_price": raw["EntryPrice"].astype(float),
            "exit_price": raw["ExitPrice"].astype(float),
            "size": raw["Size"].astype(float),
            "pnl": raw["PnL"].astype(float),
            "return_pct": raw["ReturnPct"].astype(float),
            "bars_held": (raw["ExitBar"] - raw["EntryBar"]).astype(int),
            "fees": float("nan"),  # charged against cash by backtesting.py, not broken out
        }
    )
    return trades[TRADE_COLUMNS].sort_values("entry_time").reset_index(drop=True)
