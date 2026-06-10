"""vectorbt engine adapter — the primary engine.

Vectorized and multi-asset: target weights become ``Portfolio.from_orders``
with ``size_type="targetpercent"``, filled at each bar's OPEN with the
configured fees and slippage.
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
    import vectorbt as vbt  # deferred: numba JIT makes first import/call slow

    symbols = list(weights.columns)
    close = pd.DataFrame({s: data[s]["close"] for s in symbols}).reindex(weights.index)
    open_ = pd.DataFrame({s: data[s]["open"] for s in symbols}).reindex(weights.index)

    pf = vbt.Portfolio.from_orders(
        close=close,
        size=weights,
        size_type="targetpercent",
        price=open_,
        fees=config.costs.commission_pct,
        slippage=config.costs.slippage_pct,
        init_cash=config.portfolio.initial_cash,
        cash_sharing=True,
        group_by=True,
        call_seq="auto",
        freq="1D",
    )

    equity = pf.value()
    if isinstance(equity, pd.DataFrame):
        equity = equity.iloc[:, 0]
    equity.name = "equity"

    trades = _normalize_trades(pf, close, weights.index)
    return equity, trades


def _normalize_trades(pf, close: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    raw = pf.trades.records_readable.copy()
    if len(raw) == 0:
        return pd.DataFrame(columns=TRADE_COLUMNS)

    # vectorbt leaves still-open trades with NaN PnL/Return/exit; close them
    # at the final bar's close (mark-to-market) so metrics see real numbers
    # and both engines treat end-of-data positions the same way.
    open_mask = raw["Status"].astype(str).str.lower() == "open"
    if open_mask.any():
        last_close = raw.loc[open_mask, "Column"].map(
            lambda col: float(close[str(col[-1]) if isinstance(col, tuple) else str(col)].iloc[-1])
        )
        raw.loc[open_mask, "Exit Timestamp"] = index[-1]
        raw.loc[open_mask, "Avg Exit Price"] = last_close
        raw.loc[open_mask, "PnL"] = (
            last_close - raw.loc[open_mask, "Avg Entry Price"]
        ) * raw.loc[open_mask, "Size"] - raw.loc[open_mask, "Entry Fees"]
        raw.loc[open_mask, "Return"] = raw.loc[open_mask, "PnL"] / (
            raw.loc[open_mask, "Avg Entry Price"] * raw.loc[open_mask, "Size"]
        )
        raw.loc[open_mask, "Exit Fees"] = raw.loc[open_mask, "Exit Fees"].fillna(0.0)

    def _symbol(col: object) -> str:
        # vectorbt encodes the column as the symbol name (or a tuple when grouped)
        if isinstance(col, tuple):
            return str(col[-1])
        return str(col)

    def _time(value: object) -> pd.Timestamp:
        # records may carry timestamps directly or integer bar indexes
        if isinstance(value, (int,)) and not isinstance(value, bool):
            return index[value]
        return pd.Timestamp(value)

    trades = pd.DataFrame(
        {
            "symbol": raw["Column"].map(_symbol),
            "entry_time": raw["Entry Timestamp"].map(_time),
            "exit_time": raw["Exit Timestamp"].map(_time),
            "entry_price": raw["Avg Entry Price"].astype(float),
            "exit_price": raw["Avg Exit Price"].astype(float),
            "size": raw["Size"].astype(float),
            "pnl": raw["PnL"].astype(float),
            "return_pct": raw["Return"].astype(float),
            "fees": (raw["Entry Fees"] + raw["Exit Fees"]).astype(float),
        }
    )
    trades["bars_held"] = (
        trades["exit_time"].dt.normalize() - trades["entry_time"].dt.normalize()
    ).dt.days
    return trades[TRADE_COLUMNS].sort_values("entry_time").reset_index(drop=True)
