"""The standard results object every backtest produces.

A ``BacktestResult`` is the framework's lingua franca: Phase 4 in-sample
reports, Phase 5 walk-forward windows, and Phase 7 paper-trading
reconciliations all produce or consume this same shape.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

#: Required columns of the normalized trades DataFrame.
TRADE_COLUMNS = [
    "symbol",
    "entry_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "size",
    "pnl",
    "return_pct",
    "bars_held",
    "fees",
]


@dataclass(frozen=True)
class BenchmarkResult:
    """Buy-and-hold benchmark run through the same engine, costs, and metrics."""

    symbol: str
    equity_curve: pd.Series
    returns: pd.Series
    metrics: dict[str, float]


@dataclass(frozen=True)
class BacktestResult:
    strategy_name: str
    params: dict[str, Any]
    engine: str
    start: pd.Timestamp
    end: pd.Timestamp
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float]
    benchmark: BenchmarkResult | None
    config: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable strategy-vs-benchmark metrics table."""
        rows = [
            ("Total return", "total_return", "{:+.2%}"),
            ("CAGR", "cagr", "{:+.2%}"),
            ("Sharpe", "sharpe", "{:.2f}"),
            ("Sortino", "sortino", "{:.2f}"),
            ("Max drawdown", "max_drawdown", "{:.2%}"),
            ("Win rate", "win_rate", "{:.1%}"),
            ("Avg win/loss", "avg_win_loss_ratio", "{:.2f}"),
            ("Exposure", "exposure", "{:.1%}"),
            ("Trades", "trade_count", "{:.0f}"),
            ("Turnover (1-way/yr)", "turnover", "{:.2f}"),
        ]
        bench_label = f"{self.benchmark.symbol} B&H" if self.benchmark else "benchmark"
        lines = [
            f"{self.strategy_name}  [{self.engine}]  "
            f"{self.start.date()} -> {self.end.date()}  params={self.params}",
            "",
            f"{'metric':<22}{'strategy':>12}{bench_label:>14}",
            "-" * 48,
        ]
        for label, key, fmt in rows:
            strat_val = _fmt(self.metrics.get(key), fmt)
            bench_val = _fmt(self.benchmark.metrics.get(key), fmt) if self.benchmark else "-"
            lines.append(f"{label:<22}{strat_val:>12}{bench_val:>14}")
        if "alpha" in self.metrics:
            lines.append(
                f"{'Alpha / Beta':<22}"
                f"{_fmt(self.metrics['alpha'], '{:+.2%}'):>12}"
                f"{_fmt(self.metrics['beta'], '{:.2f}'):>14}"
            )
        return "\n".join(lines)

    def plot(self, save_path: Path | str | None = None):
        """Equity curve vs benchmark plus drawdown panel. Returns the figure."""
        import matplotlib

        if save_path is not None:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax_eq, ax_dd) = plt.subplots(
            2, 1, figsize=(11, 7), sharex=True, height_ratios=[3, 1]
        )
        norm_equity = self.equity_curve / self.equity_curve.iloc[0]
        ax_eq.plot(norm_equity.index, norm_equity, label=self.strategy_name, lw=1.4)
        if self.benchmark is not None:
            norm_bench = self.benchmark.equity_curve / self.benchmark.equity_curve.iloc[0]
            ax_eq.plot(
                norm_bench.index,
                norm_bench,
                label=f"{self.benchmark.symbol} buy & hold",
                lw=1.2,
                alpha=0.75,
            )
        ax_eq.set_ylabel("Growth of $1")
        ax_eq.legend()
        ax_eq.set_title(f"{self.strategy_name} ({self.engine})")

        drawdown = self.equity_curve / self.equity_curve.cummax() - 1
        ax_dd.fill_between(drawdown.index, drawdown, 0, alpha=0.4)
        ax_dd.set_ylabel("Drawdown")

        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=120)
            plt.close(fig)
        else:
            plt.show()
        return fig

    def save(self, out_dir: Path | str) -> Path:
        """Write metrics.json, trades.csv, equity.parquet, and plot.png to ``out_dir``."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "strategy": self.strategy_name,
            "params": self.params,
            "engine": self.engine,
            "start": str(self.start),
            "end": str(self.end),
            "metrics": self.metrics,
            "benchmark_metrics": self.benchmark.metrics if self.benchmark else None,
            "config": self.config,
        }
        (out / "metrics.json").write_text(json.dumps(payload, indent=2, default=str))
        self.trades.to_csv(out / "trades.csv", index=False)
        self.equity_curve.rename("equity").to_frame().to_parquet(out / "equity.parquet")
        self.plot(save_path=out / "plot.png")
        return out


def _fmt(value: float | None, fmt: str) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return fmt.format(value)
