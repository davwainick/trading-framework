# Trading Framework

A repeatable harness to formulate, backtest, validate, paper-trade, and
live-deploy algorithmic trading strategies. The framework is the product;
strategies are disposable plug-ins — most of them *should* be rejected by the
validation pipeline, and the harness makes rejecting them cheap.

> **Disclaimer:** engineering project, not investment advice. Backtested
> results do not predict future returns.

## Design

```
config YAML ──► RunConfig (pydantic) ──► Strategy.generate_signals(data)
                                              │ target weights [0..1]
                                              ▼
                              backtest runner (the only place that:
                               • clamps weights to charter risk limits
                               • shifts signals +1 bar  ← look-ahead guard
                               • dispatches to an engine)
                                              │
                          ┌───────────────────┴───────────────────┐
                          ▼                                       ▼
                   vectorbt engine                       backtesting.py engine
                 (primary, vectorized,                  (event-driven, single-
                  multi-asset)                           symbol cross-check)
                          └───────────────────┬───────────────────┘
                                              ▼
                       BacktestResult (equity, trades, engine-independent
                       metrics, SPY buy-and-hold benchmark, config snapshot)
```

Key invariants, enforced centrally and covered by tests:

- **Strategies are pure**: OHLCV in, target weights out. No data fetching, no
  order placement, no execution lag inside a strategy. The same
  `generate_signals` code will drive live trading later.
- **No look-ahead**: a signal computed at bar *t*'s close fills at bar
  *t+1*'s open, applied by a single `weights.shift(1)` in the runner.
  `tests/test_no_lookahead.py` proves both halves (signals are
  prefix-consistent; fills land on the next open).
- **One metric definition**: engines only produce equity curves and trades;
  CAGR/Sharpe/Sortino/drawdown/etc. are computed once in
  `framework.backtest.metrics`, so engines and (later) live results are
  directly comparable.
- **Benchmark always attached**: every result carries buy-and-hold of SPY run
  through the same engine and costs.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 (uv will fetch it).

```bash
git clone <repo> && cd trading-framework
uv sync                                   # create .venv, install pinned deps
uv run pytest -q                          # verify: all tests pass offline
uv run tf data download                   # fetch the universe (config/universe.yaml)
uv run tf data check                      # re-run data-quality checks anytime
```

The download pipeline quality-checks everything it fetches (missing dates,
zero/negative prices, gap/spike detection, OHLC consistency, duplicates) and
refuses to call a symbol OK otherwise. See `docs/data.md` for what the free
data can and cannot honestly support — in particular why single-name equity
backtests are off-limits until a survivorship-bias-free source is wired in.

## Run the example strategy

```bash
uv run tf list-strategies
uv run tf run config/strategies/sma_crossover.yaml --save
uv run tf run config/strategies/sma_crossover.yaml --engine backtesting   # cross-check
```

Sample output (real SPY data, 2018→2026 — note the example strategy honestly
**loses to buy-and-hold**, which is exactly the verdict the harness exists to
deliver):

```
metric                    strategy       SPY B&H
------------------------------------------------
Total return               +84.90%      +211.95%
CAGR                        +7.58%       +14.49%
Sharpe                        0.63          0.80
Max drawdown               -27.04%       -33.72%
Trades                          10             1
```

`--save` writes `metrics.json`, `trades.csv`, `equity.parquet`, and
`plot.png` under `results/`.

Note: the first vectorbt run takes ~10–30 s extra while numba JIT-compiles —
it is not hung.

## Adding a strategy

Drop a module into `src/framework/strategies/private/` (gitignored — see the
README there) or `examples/`:

```python
@register
class MyStrategy(Strategy):
    name = "my_strategy"

    class Params(StrategyParams):
        lookback: int = Field(60, ge=2)

    params_model = Params

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        ...  # weights in [0, 1] per symbol, using ≤ current bar's close
```

Pair it with a YAML in `config/strategies/` and run it with `tf run`. No
harness code changes required.

## Repository layout

```
config/            global.yaml, universe.yaml, per-strategy configs (private/ gitignored)
data/ohlcv/        Parquet OHLCV, one file per symbol (gitignored)
docs/              engines.md (vectorbt vs backtesting.py), data.md (coverage & limits)
src/framework/
  config.py        pydantic config models, YAML deep-merge
  data/            load_ohlcv() — the single data entry point
    sources/       vendor adapters (yfinance now; Norgate drops in later)
    quality.py     data-quality checks behind `tf data check`
    download.py    fetch -> normalize -> check -> Parquet pipeline
  strategies/      Strategy ABC, registry, examples/, private/ (gitignored)
  backtest/        runner, engines/, metrics, BacktestResult
  validation/      (Phase 4–5: walk-forward, Monte Carlo, sensitivity)
  live/            (Phase 6: Alpaca paper/live execution)
tests/             hermetic pytest suite — synthetic data only, no network
```

## Risk limits

Charter limits live in `config/global.yaml` and are enforced by the runner,
not by strategies: max 20% per position (multi-asset), max 10 concurrent
positions, no leverage (gross weight scaled to ≤ 1), long-only. Costs default
to 5 bps commission + 5 bps slippage per side.
