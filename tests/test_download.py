"""Download pipeline tests using a fake in-memory source — no network."""

import pytest

from framework.data import load_ohlcv
from framework.data.download import download_symbols, load_universe
from framework.data.sources import SourceError, get_source, register_source
from framework.data.synthetic import make_ohlcv


@register_source
class FakeSource:
    name = "fake_test_source"
    calls: list[str] = []

    def fetch(self, symbol, start=None, end=None):
        type(self).calls.append(symbol)
        if symbol == "BAD":
            raise SourceError("vendor says no")
        return make_ohlcv(n_bars=300, seed=len(symbol))


def test_source_registry():
    assert get_source("fake_test_source").name == "fake_test_source"
    with pytest.raises(KeyError, match="Unknown data source"):
        get_source("nope")


def test_download_writes_loader_compatible_parquet(tmp_path):
    reports = download_symbols(
        ["aaa", "BBB"], source_name="fake_test_source", data_dir=tmp_path
    )
    assert [r.symbol for r in reports] == ["AAA", "BBB"]
    assert all(r.ok for r in reports)
    assert (tmp_path / "AAA.parquet").exists()

    # Round-trips through the standard loader contract.
    data = load_ohlcv(["AAA", "BBB"], data_dir=tmp_path)
    assert list(data["AAA"].columns) == ["open", "high", "low", "close", "volume"]
    assert str(data["AAA"].index.tz) == "UTC"


def test_failed_fetch_reported_not_raised(tmp_path):
    reports = download_symbols(
        ["GOOD", "BAD"], source_name="fake_test_source", data_dir=tmp_path
    )
    by_symbol = {r.symbol: r for r in reports}
    assert by_symbol["GOOD"].ok
    assert not by_symbol["BAD"].ok
    assert by_symbol["BAD"].issues[0].check == "fetch_failed"
    assert not (tmp_path / "BAD.parquet").exists()


def test_load_universe(tmp_path):
    universe_file = tmp_path / "universe.yaml"
    universe_file.write_text("etfs: [SPY, QQQ]\ncrypto: [BTC-USD]\n")
    universe = load_universe(universe_file)
    assert universe == {"etfs": ["SPY", "QQQ"], "crypto": ["BTC-USD"]}


def test_repo_universe_file_is_valid():
    universe = load_universe()
    assert "etfs" in universe and "SPY" in universe["etfs"]
    assert all(isinstance(s, str) for symbols in universe.values() for s in symbols)
