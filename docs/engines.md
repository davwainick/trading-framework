# vectorbt vs backtesting.py — when to use which

**Default to vectorbt; use backtesting.py as a second opinion.**

vectorbt is fully vectorized (numba-compiled), so it backtests an entire
parameter grid as one array operation — thousands of SMA-pair combinations in
seconds — which is exactly what walk-forward analysis, Monte Carlo runs, and
parameter-sensitivity sweeps (Phases 4–5) need. It also handles multi-asset
target-weight portfolios natively, which is why it is the engine wired into
the runner by default.

backtesting.py is event-driven: it processes one bar at a time through
readable, order-level logic. That makes it slow for sweeps but excellent for
sanity-checking a single configuration's fill behavior and for its
interactive trade-by-trade charts.

In this framework the backtesting.py adapter is deliberately limited —
single symbol, long-only, all-in/all-out — because its job is
**cross-validation, not production**: if the two engines disagree materially
on the same strategy, data, and costs, something is wrong with a fill
assumption, and that needs investigating before trusting either number.
`tests/test_backtest_runner.py::test_engines_agree` automates exactly this
check on synthetic data.

## Known accounting differences (expected, small)

- vectorbt marks slippage into the fill price and reports fees per trade;
  backtesting.py charges commission+slippage against cash and reports raw
  fill prices (per-trade `fees` is NaN).
- backtesting.py trades whole shares; vectorbt allows fractional sizes.
  Final equity typically differs by well under 2% on identical inputs.
- Open positions at the end of data: both engines mark them to the final
  bar's close (the vectorbt adapter does this normalization itself).
