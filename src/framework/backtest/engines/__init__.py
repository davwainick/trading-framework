"""Engine adapters. Each takes pre-shifted weights and raw OHLCV, fills at the
bar's open with the configured costs, and returns (equity_curve, trades).

The look-ahead shift is applied by the runner BEFORE weights reach an engine —
engines execute the weights on the bar they receive them.
"""
