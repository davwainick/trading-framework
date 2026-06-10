"""Command-line interface.

    tf run config/strategies/sma_crossover.yaml [--engine backtesting] [--plot] [--save]
    tf list-strategies
    tf data download [SYMBOLS...] [--sleeve etfs] [--start 2000-01-01]
    tf data check [SYMBOLS...]
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional

import typer

from framework.config import REPO_ROOT, load_run_config

app = typer.Typer(no_args_is_help=True, add_completion=False)
data_app = typer.Typer(no_args_is_help=True, help="Download and quality-check market data")
app.add_typer(data_app, name="data")


@app.command()
def run(
    config_path: Path = typer.Argument(..., help="Path to a strategy run-config YAML"),
    engine: Optional[str] = typer.Option(
        None, help="Override the engine: 'vectorbt' or 'backtesting'"
    ),
    plot: bool = typer.Option(False, "--plot", help="Show the equity/drawdown plot"),
    save: bool = typer.Option(
        False, "--save", help="Save metrics, trades, equity, and plot to the results dir"
    ),
) -> None:
    """Run a strategy config through the backtest harness and print the summary."""
    from framework.backtest import run_backtest
    from framework.data import load_ohlcv
    from framework.strategies import get_strategy

    config = load_run_config(config_path)
    strategy = get_strategy(config.strategy)(config.params)
    data = load_ohlcv(config.symbols, config.start, config.end, config.data.ohlcv_dir)

    typer.echo(f"Running {config.strategy} on {config.symbols} ...")
    result = run_backtest(strategy, data, config, engine=engine)

    typer.echo("")
    typer.echo(result.summary())

    if save:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = config.results_dir
        if not results_dir.is_absolute():
            results_dir = REPO_ROOT / results_dir
        out = result.save(results_dir / f"{config.strategy}_{stamp}")
        typer.echo(f"\nSaved results to {out}")
    if plot:
        result.plot()


@app.command("list-strategies")
def list_strategies() -> None:
    """List every strategy registered with the framework."""
    from framework.strategies import available_strategies, get_strategy

    for name in available_strategies():
        cls = get_strategy(name)
        fields = ", ".join(
            f"{fname}={finfo.default}" for fname, finfo in cls.params_model.model_fields.items()
        )
        typer.echo(f"{name:<24} params: {fields or '(none)'}")


@data_app.command("download")
def data_download(
    symbols: Optional[list[str]] = typer.Argument(
        None, help="Symbols to fetch (default: the whole universe file)"
    ),
    sleeve: Optional[str] = typer.Option(
        None, help="Restrict to one universe sleeve (e.g. 'etfs', 'crypto')"
    ),
    source: str = typer.Option("yfinance", help="Data source adapter"),
    start: Optional[str] = typer.Option(None, help="Start date (default: max history)"),
    end: Optional[str] = typer.Option(None, help="End date (default: latest)"),
) -> None:
    """Download daily adjusted OHLCV to the data folder and quality-check it."""
    from framework.data.download import download_symbols, load_universe

    if not symbols:
        universe = load_universe()
        if sleeve is not None:
            if sleeve not in universe:
                typer.echo(f"Unknown sleeve {sleeve!r}. Available: {sorted(universe)}")
                raise typer.Exit(1)
            universe = {sleeve: universe[sleeve]}
        symbols = [s for sleeve_symbols in universe.values() for s in sleeve_symbols]

    reports = download_symbols(symbols, source_name=source, start=start, end=end)
    typer.echo("")
    for report in reports:
        typer.echo(report.summary())
    failed = [r.symbol for r in reports if not r.ok]
    typer.echo(
        f"\n{len(reports) - len(failed)}/{len(reports)} symbols OK"
        + (f"; PROBLEMS: {', '.join(failed)}" if failed else "")
    )
    if failed:
        raise typer.Exit(1)


@data_app.command("check")
def data_check(
    symbols: Optional[list[str]] = typer.Argument(
        None, help="Symbols to check (default: every Parquet file in the data dir)"
    ),
) -> None:
    """Re-run quality checks on already-downloaded Parquet files."""
    from framework.data import load_ohlcv
    from framework.data.loader import DEFAULT_DATA_DIR
    from framework.data.quality import check_ohlcv

    if not symbols:
        symbols = sorted(p.stem for p in DEFAULT_DATA_DIR.glob("*.parquet"))
        if not symbols:
            typer.echo("No data files found. Run `tf data download` first.")
            raise typer.Exit(1)

    any_errors = False
    for symbol in symbols:
        df = load_ohlcv([symbol])[symbol.upper()]
        report = check_ohlcv(df, symbol.upper())
        typer.echo(report.summary())
        any_errors |= not report.ok
    if any_errors:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
