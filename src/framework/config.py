"""Configuration models and YAML loading.

Two-layer configuration: a single ``config/global.yaml`` holds framework-wide
defaults (data paths, costs, charter risk limits), and each strategy run has a
small YAML file that names the strategy, its symbols/dates/params, and may
override any global key. The strategy file wins on conflicts (deep merge).

Everything is validated through pydantic before any data is loaded, so typos
and out-of-range values fail loudly and early.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
GLOBAL_CONFIG_PATH = REPO_ROOT / "config" / "global.yaml"


class DataConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    ohlcv_dir: Path = Path("data/ohlcv")


class CostsConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    commission_pct: float = Field(0.0005, ge=0, lt=0.1)
    slippage_pct: float = Field(0.0005, ge=0, lt=0.1)


class PortfolioConfig(BaseModel):
    """Charter risk limits. Enforced centrally by the backtest runner."""

    model_config = {"frozen": True, "extra": "forbid"}

    initial_cash: float = Field(100_000, gt=0)
    max_position_pct: float = Field(0.20, gt=0, le=1.0)
    max_positions: int = Field(10, ge=1)
    allow_leverage: bool = False


class ExecutionConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    fill: str = Field("next_open", pattern="^next_open$")
    frequency: str = Field("daily", pattern="^daily$")


class BenchmarkConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    symbol: str = "SPY"


class BacktestConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    engine: str = Field("vectorbt", pattern="^(vectorbt|backtesting)$")


class RunConfig(BaseModel):
    """Fully-resolved configuration for one backtest run."""

    model_config = {"frozen": True, "extra": "forbid"}

    strategy: str
    symbols: list[str] = Field(min_length=1)
    start: str | None = None
    end: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)

    data: DataConfig = DataConfig()
    results_dir: Path = Path("results")
    costs: CostsConfig = CostsConfig()
    portfolio: PortfolioConfig = PortfolioConfig()
    execution: ExecutionConfig = ExecutionConfig()
    benchmark: BenchmarkConfig = BenchmarkConfig()
    backtest: BacktestConfig = BacktestConfig()

    def resolved_dict(self) -> dict[str, Any]:
        """JSON-safe snapshot of the full config, stored in results for reproducibility."""
        return self.model_dump(mode="json")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``; override wins on conflicts."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: Path | str) -> dict[str, Any]:
    with open(path) as f:
        loaded = yaml.safe_load(f)
    return loaded or {}


def load_run_config(
    strategy_config_path: Path | str,
    global_config_path: Path | str | None = None,
) -> RunConfig:
    """Load global + strategy YAML, deep-merge (strategy wins), and validate."""
    global_path = Path(global_config_path or GLOBAL_CONFIG_PATH)
    global_cfg = load_yaml(global_path) if global_path.exists() else {}
    strategy_cfg = load_yaml(strategy_config_path)
    return RunConfig(**deep_merge(global_cfg, strategy_cfg))
