"""Tests for the (private, gitignored) dual_momentum strategy.

The strategy module lives in ``src/framework/strategies/private/`` and is not
part of the public repo, so the whole file is skipped when it isn't
registered (e.g. on a fresh public clone).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from framework.data.synthetic import make_multi_ohlcv
from framework.strategies import available_strategies, get_strategy

pytestmark = pytest.mark.skipif(
    "dual_momentum" not in available_strategies(),
    reason="private dual_momentum strategy not present (public clone)",
)

# Short lookbacks so 600 synthetic bars give a long post-warm-up window.
TEST_PARAMS = {"lookback_long": 126, "lookback_short": 21, "top_n": 2}


def make_strategy(**overrides):
    return get_strategy("dual_momentum")({**TEST_PARAMS, **overrides})


@pytest.fixture
def mixed_universe() -> dict[str, pd.DataFrame]:
    """Two winners, two losers, and a flat SHY cash benchmark."""
    return make_multi_ohlcv(
        {
            "UP1": 0.003,
            "UP2": 0.002,
            "DOWN1": -0.002,
            "DOWN2": -0.003,
            "SHY": 0.0001,
        },
        vol=0.005,
    )


def first_bar_of_month(index: pd.DatetimeIndex) -> pd.Series:
    months = pd.Series(index.tz_localize(None).to_period("M"), index=index)
    return months.ne(months.shift(1))


def test_output_shape_matches_input(mixed_universe):
    weights = make_strategy().generate_signals(mixed_universe)
    some_df = next(iter(mixed_universe.values()))
    assert list(weights.columns) == list(mixed_universe)
    assert weights.index.equals(some_df.index)


def test_weights_bounded_and_sum_le_one(mixed_universe):
    weights = make_strategy().generate_signals(mixed_universe)
    assert ((weights >= 0) & (weights <= 1)).all().all()
    assert (weights.sum(axis=1) <= 1.0 + 1e-9).all()
    assert not weights.isnull().any().any()
    assert not np.isinf(weights.to_numpy()).any()


def test_signals_are_prefix_consistent(mixed_universe):
    """generate_signals(data[:t]) must equal the first t rows on full data."""
    strat = make_strategy()
    full = strat.generate_signals(mixed_universe)
    for t in [150, 300, 450, 599]:
        prefix_data = {s: df.iloc[:t] for s, df in mixed_universe.items()}
        prefix = strat.generate_signals(prefix_data)
        pd.testing.assert_frame_equal(
            prefix, full.iloc[:t], check_names=False, check_freq=False
        )


def test_all_negative_momentum_goes_to_cash():
    """Nothing beats the SHY benchmark -> zero weight everywhere (cash)."""
    data = make_multi_ohlcv(
        {"A": -0.003, "B": -0.004, "C": -0.0035, "SHY": 0.0002},
        vol=0.003,
    )
    weights = make_strategy().generate_signals(data)
    assert (weights == 0.0).all().all()


def test_rebalances_only_on_first_bar_of_month(mixed_universe):
    weights = make_strategy().generate_signals(mixed_universe)
    changed = weights.ne(weights.shift(1)).any(axis=1)
    rebalance = first_bar_of_month(weights.index)
    # No weight change outside rebalance bars (row 0 is itself a rebalance).
    assert not (changed & ~rebalance).iloc[1:].any()


def test_missing_shy_falls_back_gracefully():
    data = make_multi_ohlcv({"UP1": 0.003, "UP2": 0.002, "DOWN1": -0.002}, vol=0.005)
    weights = make_strategy().generate_signals(data)  # must not raise
    assert list(weights.columns) == list(data)
    # With a zero benchmark the uptrending symbols pass the filter.
    assert (weights.sum(axis=1) > 0).any()


def test_shy_is_never_held(mixed_universe):
    weights = make_strategy().generate_signals(mixed_universe)
    assert (weights["SHY"] == 0.0).all()
