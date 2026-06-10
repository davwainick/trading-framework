# CLAUDE.md

Repeatable algo-trading framework: strategies are plug-ins; the harness
(backtest/validation/paper/live) is shared and must stay strategy-agnostic.

## Commands

```bash
uv sync                                          # env + deps (lockfile pinned)
uv run pytest -q                                 # full suite, hermetic, ~3 s
uv run ruff check .                              # lint (run before committing)
uv run tf list-strategies
uv run tf run config/strategies/<name>.yaml [--engine backtesting] [--plot] [--save]
uv run tf data download [SYMBOLS...] [--sleeve etfs|crypto] [--start YYYY-MM-DD]
uv run tf data check                             # re-validate downloaded data
```

`data/` is gitignored — after a fresh clone, run `tf data download` before any
backtest. The first vectorbt call takes ~10–30 s (numba JIT); not a hang.

## Architecture (data flow)

```
config YAML ─► RunConfig (config.py, pydantic, deep-merge global+strategy)
            ─► Strategy.generate_signals(data) ─► target weights in [0,1]
            ─► runner.py: clamp charter limits ─► weights.shift(1) ─► engine
            ─► BacktestResult (equity, trades, metrics, SPY benchmark, config snapshot)
```

Key files: `src/framework/strategies/base.py` (Strategy ABC),
`backtest/runner.py` (pipeline), `backtest/metrics.py` (single source of
metric definitions), `data/loader.py` (`load_ohlcv()` — the only data entry
point), `data/sources/` (vendor adapters), `data/quality.py` (checks),
`config.py` (all config models).

## Invariants — do not break these

1. **Strategies are pure**: OHLCV in, target weights out. No data fetching,
   no order logic, no self-shifting inside a strategy. The same
   `generate_signals` must be reusable for live trading (Phase 6).
2. **The look-ahead guard lives in one place**: `weights.shift(1)` in
   `runner.run_backtest`. Signals at bar t fill at bar t+1's open. Never
   shift in strategies or engines. `tests/test_no_lookahead.py` enforces
   both halves; it must pass unmodified.
3. **Metrics are engine-independent**: engines return only (equity_curve,
   trades); all metrics come from `backtest/metrics.py`. Never read
   vendor/engine stats objects for reporting.
4. **Loader contract**: `{data_dir}/{SYMBOL}.parquet`, lowercase
   open/high/low/close/volume, tz-aware UTC sorted unique DatetimeIndex,
   split- AND dividend-adjusted prices. Every new data source must emit this.
5. **Charter risk limits are enforced by the runner** (`_prepare_weights`),
   not by strategies: long-only, per-position cap (multi-asset), max
   positions, gross weight ≤ 1.
6. **Strategy interface is frozen**: changing `Strategy`/`StrategyParams`
   signatures breaks every private strategy that isn't in this repo. Extend,
   don't mutate.

## Adding a strategy

1. Module in `src/framework/strategies/private/` (real, gitignored) or
   `examples/` (committed). Subclass `Strategy`, declare a pydantic `Params`
   (frozen, extra=forbid, few parameters), decorate with `@register`.
2. Config YAML in `config/strategies/private/` (gitignored) or
   `config/strategies/`: `strategy:` name, `symbols:`, `start/end`, `params:`.
3. `uv run tf run <config>` — no harness changes should ever be needed.
   See `src/framework/strategies/private/README.md` for a template.

## Adding a data source

Implement the `DataSource` protocol in `src/framework/data/sources/`,
decorate with `@register_source`, emit the loader contract. Then
`tf data download --source <name>`. See `docs/data.md` for the Norgate path.

## Testing conventions

- Tests are hermetic: seeded synthetic data (`data/synthetic.py`), tmp_path
  parquet dirs, zero network. Keep it that way.
- New defect classes in quality checks get an injected-defect test in
  `tests/test_quality.py`.
- Engine changes must keep `test_engines_agree` green (vectorbt vs
  backtesting.py within 2% on identical inputs).

## Gotchas

- The benchmark comparison needs `SPY.parquet` present; missing benchmark
  logs a warning and sets `result.benchmark = None` instead of failing.
- The backtesting.py engine is deliberately single-symbol, long-only,
  all-in/all-out — a cross-check, not a portfolio engine (see
  `docs/engines.md`).
- Crypto trades 7 days/week; equity 5. `quality.py` infers the calendar from
  the data — don't hardcode trading calendars.
- Yahoo silently revises adjusted history between downloads; expected
  quality warnings on current data: 9/11 closure gap, 2007-01-02 mourning
  closure, Mar-2020 crypto crash spikes (all real events, not defects).
- vectorbt 1.0.0 pinned; if its API regresses, fall back to
  `vectorbt~=0.28.5` (same dependency profile).

## Git

- Conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).
- Public portfolio repo. NEVER commit: `data/**`, `.env*`, anything under
  the two `private/` dirs (except their `__init__.py`/README), or values
  derived from paid data sources. Before any push:
  `git ls-files | grep -E "\.parquet$|\.env|/private/"` must show only the
  whitelisted placeholders.
