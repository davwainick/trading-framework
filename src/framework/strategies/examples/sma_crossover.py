"""Simple moving-average crossover — the reference strategy.

Long when the fast SMA is above the slow SMA, flat otherwise. Deliberately
trivial: its job is to demonstrate the plug-in contract (declare params, emit
target weights, no execution logic), not to make money.
"""

from __future__ import annotations

import pandas as pd
from pydantic import Field, model_validator

from framework.strategies import register
from framework.strategies.base import Strategy, StrategyParams


@register
class SmaCrossover(Strategy):
    name = "sma_crossover"

    class Params(StrategyParams):
        fast: int = Field(20, ge=2, description="Fast SMA window (bars)")
        slow: int = Field(100, gt=2, description="Slow SMA window (bars)")

        @model_validator(mode="after")
        def _fast_lt_slow(self) -> "SmaCrossover.Params":
            if self.fast >= self.slow:
                raise ValueError(f"fast ({self.fast}) must be < slow ({self.slow})")
            return self

    params_model = Params

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        weights = {}
        for symbol, df in data.items():
            fast = df["close"].rolling(self.params.fast).mean()
            slow = df["close"].rolling(self.params.slow).mean()
            # (fast > slow) is False during the initial NaN window -> weight 0.
            weights[symbol] = (fast > slow).astype(float)
        return pd.DataFrame(weights)
