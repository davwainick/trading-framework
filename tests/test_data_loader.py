import pandas as pd
import pytest

from framework.data import DataNotFoundError, load_ohlcv


def test_round_trip_and_normalization(parquet_data_dir):
    data = load_ohlcv(["TEST"], data_dir=parquet_data_dir)
    df = data["TEST"]
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert not df.index.duplicated().any()


def test_date_slicing(parquet_data_dir):
    full = load_ohlcv(["TEST"], data_dir=parquet_data_dir)["TEST"]
    sliced = load_ohlcv(
        ["TEST"], start=full.index[100], end=full.index[200], data_dir=parquet_data_dir
    )["TEST"]
    assert sliced.index[0] == full.index[100]
    assert sliced.index[-1] == full.index[200]


def test_lowercase_symbol_normalized(parquet_data_dir):
    data = load_ohlcv(["test"], data_dir=parquet_data_dir)
    assert "TEST" in data


def test_missing_symbol_raises_helpful_error(parquet_data_dir):
    with pytest.raises(DataNotFoundError, match="bootstrap_data"):
        load_ohlcv(["NOPE"], data_dir=parquet_data_dir)
