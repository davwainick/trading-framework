"""Shared fixtures. All test data is seeded synthetic — no network, no disk
dependencies beyond pytest's tmp_path."""

import pandas as pd
import pytest

from framework.config import RunConfig
from framework.data.synthetic import make_ohlcv, make_trending_ohlcv


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    return make_ohlcv(n_bars=600, seed=42, trend=0.0004)


@pytest.fixture
def trending_ohlcv() -> pd.DataFrame:
    return make_trending_ohlcv(n_bars=400, seed=7)


@pytest.fixture
def parquet_data_dir(tmp_path, synthetic_ohlcv):
    """A temp data dir holding TEST.parquet plus a BENCH.parquet benchmark."""
    data_dir = tmp_path / "ohlcv"
    data_dir.mkdir()
    synthetic_ohlcv.to_parquet(data_dir / "TEST.parquet")
    make_ohlcv(n_bars=600, seed=99, trend=0.0003).to_parquet(data_dir / "BENCH.parquet")
    return data_dir


@pytest.fixture
def run_config(parquet_data_dir) -> RunConfig:
    return RunConfig(
        strategy="sma_crossover",
        symbols=["TEST"],
        params={"fast": 10, "slow": 50},
        data={"ohlcv_dir": str(parquet_data_dir)},
        benchmark={"symbol": "BENCH"},
    )
