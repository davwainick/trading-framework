"""Download pipeline: fetch -> normalize -> quality-check -> Parquet.

Writes through the loader's ``{data_dir}/{SYMBOL}.parquet`` contract, so the
rest of the framework neither knows nor cares which vendor produced a file.
Files failing quality checks with ERROR severity are still written (so they
can be inspected) but are clearly flagged in the returned reports.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from framework.config import REPO_ROOT
from framework.data.loader import DEFAULT_DATA_DIR
from framework.data.quality import DataQualityReport, Issue, check_ohlcv
from framework.data.sources import SourceError, get_source

logger = logging.getLogger(__name__)

DEFAULT_UNIVERSE_PATH = REPO_ROOT / "config" / "universe.yaml"


def load_universe(path: Path | str | None = None) -> dict[str, list[str]]:
    """Load the universe file: {sleeve_name: [symbols]}."""
    with open(path or DEFAULT_UNIVERSE_PATH) as f:
        universe = yaml.safe_load(f)
    return {sleeve: list(symbols) for sleeve, symbols in universe.items()}


def download_symbols(
    symbols: list[str],
    source_name: str = "yfinance",
    start: str | None = None,
    end: str | None = None,
    data_dir: Path | str | None = None,
) -> list[DataQualityReport]:
    """Download, quality-check, and persist each symbol. Returns one report per symbol."""
    source = get_source(source_name)
    directory = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    directory.mkdir(parents=True, exist_ok=True)

    reports: list[DataQualityReport] = []
    for symbol in [s.upper() for s in symbols]:
        try:
            df = source.fetch(symbol, start=start, end=end)
        except SourceError as exc:
            report = DataQualityReport(symbol=symbol, n_bars=0, start=None, end=None)
            report.issues.append(Issue("fetch_failed", "error", str(exc)))
            reports.append(report)
            logger.error("%s: fetch failed: %s", symbol, exc)
            continue

        report = check_ohlcv(df, symbol)
        df.to_parquet(directory / f"{symbol}.parquet")
        reports.append(report)
        logger.info("%s: wrote %d bars", symbol, len(df))
    return reports
