"""Data layer: loading OHLCV from disk and generating synthetic series for tests."""

from framework.data.loader import DataNotFoundError, load_ohlcv
from framework.data.synthetic import make_ohlcv

__all__ = ["DataNotFoundError", "load_ohlcv", "make_ohlcv"]
