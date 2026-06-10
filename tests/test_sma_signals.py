from framework.strategies import get_strategy


def make_strategy(fast=10, slow=50):
    return get_strategy("sma_crossover")({"fast": fast, "slow": slow})


def test_weights_match_sma_relationship(trending_ohlcv):
    strat = make_strategy()
    weights = strat.generate_signals({"X": trending_ohlcv})["X"]

    fast = trending_ohlcv["close"].rolling(10).mean()
    slow = trending_ohlcv["close"].rolling(50).mean()
    expected = (fast > slow).astype(float)
    assert (weights == expected).all()


def test_crossover_happens_on_trending_data(trending_ohlcv):
    """The engineered up-then-down series must produce a long entry and a flat exit."""
    weights = make_strategy().generate_signals({"X": trending_ohlcv})["X"]
    assert weights.iloc[0] == 0.0  # NaN SMA window -> flat
    assert (weights == 1.0).any()  # goes long during the uptrend
    assert weights.iloc[-1] == 0.0  # back to flat after the downtrend
    # Exactly one round trip on this clean series: one 0->1 and one 1->0 flip.
    flips = weights.diff().fillna(0.0)
    assert (flips == 1.0).sum() == 1
    assert (flips == -1.0).sum() == 1


def test_weights_bounded(synthetic_ohlcv):
    weights = make_strategy().generate_signals({"X": synthetic_ohlcv})
    assert ((weights >= 0) & (weights <= 1)).all().all()
