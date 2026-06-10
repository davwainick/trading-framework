"""The Strategy interface every strategy plugs into.

A strategy is a pure function from OHLCV history to target portfolio weights.
It never fetches data, never places orders, and never applies its own
execution lag — those responsibilities belong to the data layer, the live
executor, and the backtest runner respectively. Keeping strategies pure is
what lets the identical ``generate_signals`` code drive both backtests and
live trading.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import pandas as pd
from pydantic import BaseModel


class StrategyParams(BaseModel):
    """Base for per-strategy parameter models.

    Frozen and ``extra="forbid"`` so misspelled or out-of-range parameters in
    a config file fail at load time, before any data is touched.
    """

    model_config = {"frozen": True, "extra": "forbid"}


class Strategy(ABC):
    """Contract (enforced by the backtest runner and the test suite):

    - The signal at bar ``t`` may use data up to and including bar ``t``'s
      CLOSE — nothing later.
    - Execution happens at bar ``t+1``'s OPEN. The runner applies that shift
      centrally; strategies must NOT pre-shift their own signals.
    - Output values are target fractions of total equity per symbol
      (long-only for now: ``0.0`` flat to ``1.0`` fully allocated). Charter
      position limits are clamped centrally by the runner.
    """

    #: Registry key used in config files, e.g. ``strategy: sma_crossover``.
    name: ClassVar[str]
    #: Pydantic model that declares and validates this strategy's parameters.
    params_model: ClassVar[type[StrategyParams]] = StrategyParams

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = self.params_model(**(params or {}))

    @abstractmethod
    def generate_signals(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Map OHLCV history to target weights.

        Args:
            data: ``{symbol: OHLCV DataFrame}`` as returned by
                :func:`framework.data.load_ohlcv` (lowercase columns, UTC index).

        Returns:
            DataFrame indexed like the input data (union of indexes), one
            column per symbol, values = desired fraction of equity in
            ``[0, 1]`` as of each bar's close.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.params})"
