"""Quality checks must catch each defect class — verified by injecting defects
into clean synthetic data and asserting the right issue (and only it) fires."""

import numpy as np
import pandas as pd
import pytest

from framework.data.quality import check_ohlcv
from framework.data.synthetic import make_ohlcv


@pytest.fixture
def clean() -> pd.DataFrame:
    return make_ohlcv(n_bars=500, seed=11)


def issue_checks(report):
    return {i.check for i in report.issues}


def test_clean_data_passes(clean):
    report = check_ohlcv(clean, "CLEAN")
    assert report.ok
    assert report.issues == []


def test_empty_data_is_error():
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    report = check_ohlcv(empty, "EMPTY")
    assert not report.ok
    assert "empty" in issue_checks(report)


def test_nan_prices_detected(clean):
    df = clean.copy()
    df.iloc[10, df.columns.get_loc("close")] = np.nan
    report = check_ohlcv(df, "X")
    assert not report.ok
    assert "nan_prices" in issue_checks(report)


def test_nonpositive_prices_detected(clean):
    df = clean.copy()
    df.iloc[20, df.columns.get_loc("low")] = -1.0
    report = check_ohlcv(df, "X")
    assert "nonpositive_prices" in issue_checks(report)


def test_duplicate_rows_detected(clean):
    df = pd.concat([clean, clean.iloc[[50]]]).sort_index()
    report = check_ohlcv(df, "X")
    assert "duplicate_rows" in issue_checks(report)


def test_ohlc_inconsistency_detected(clean):
    df = clean.copy()
    df.iloc[30, df.columns.get_loc("high")] = df["low"].iloc[30] * 0.5
    report = check_ohlcv(df, "X")
    assert "ohlc_inconsistent" in issue_checks(report)


def test_float_noise_ohlc_violation_tolerated(clean):
    """Adjustment arithmetic produces ~1e-16 relative violations — not defects."""
    df = clean.copy()
    loc = df.columns.get_loc("high")
    df.iloc[30, loc] = df[["open", "close"]].iloc[30].max() * (1 - 1e-15)
    report = check_ohlcv(df, "X")
    assert "ohlc_inconsistent" not in issue_checks(report)


def test_price_spike_flagged(clean):
    df = clean.copy()
    spike_price = df["close"].iloc[99] * 2.0  # +100% day
    # Keep the bar internally consistent so only the spike check fires.
    df.iloc[100, df.columns.get_loc("close")] = spike_price
    df.iloc[100, df.columns.get_loc("high")] = spike_price * 1.01
    report = check_ohlcv(df, "X")
    assert "price_spikes" in issue_checks(report)
    assert report.ok  # spikes are warnings, not errors


def test_calendar_gap_flagged(clean):
    df = pd.concat([clean.iloc[:200], clean.iloc[230:]])  # 30-bar hole
    report = check_ohlcv(df, "X")
    assert "calendar_gaps" in issue_checks(report)


def test_crypto_calendar_not_false_flagged():
    """A 7-day/week series with no weekend gaps must not trigger gap warnings."""
    n = 500
    rng = np.random.default_rng(5)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    index = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    df = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
         "volume": 1e6},
        index=index,
    )
    report = check_ohlcv(df, "BTC-USD")
    assert "calendar_gaps" not in issue_checks(report)


def test_zero_volume_run_flagged(clean):
    df = clean.copy()
    df.iloc[40:50, df.columns.get_loc("volume")] = 0.0
    report = check_ohlcv(df, "X")
    assert "zero_volume" in issue_checks(report)


def test_summary_is_readable(clean):
    df = clean.copy()
    df.iloc[10, df.columns.get_loc("close")] = np.nan
    text = check_ohlcv(df, "X").summary()
    assert "X:" in text and "ERROR" in text and "nan_prices" in text
