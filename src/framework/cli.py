"""Command-line interface.

    tf run config/strategies/sma_crossover.yaml [--engine backtesting] [--plot] [--save]
    tf list-strategies
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional

import typer

from framework.config import REPO_ROOT, load_run_config

app = typer.Typer(no_args_is_help=True, add_completion=False)


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


if __name__ == "__main__":
    app()
