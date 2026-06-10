# Data: coverage, limitations, and the upgrade path

## What we have (free tier, yfinance, downloaded 2026-06)

| Symbol | Bars | History | Notes |
|--------|-----:|---------|-------|
| SPY | 8,396 | 1993 → present | Longest ETF history; also the benchmark |
| DIA | 7,140 | 1998 → present | |
| XLK/XLF/XLE/XLV | 6,906 | 1998-12 → present | Sector SPDRs launched Dec 1998 |
| QQQ | 6,854 | 1999-03 → present | |
| IWM | 6,546 | 2000-05 → present | |
| TLT / IEF | 6,003 | 2002-07 → present | |
| GLD | 5,421 | 2004-11 → present | |
| BTC-USD | 4,285 | 2014-09 → present | Yahoo's BTC series start, not Bitcoin's |
| ETH-USD | 3,136 | 2017-11 → present | |

Quality status: all symbols pass `tf data check`. Remaining warnings are
real market events, not defects — the multi-day calendar gaps are the
post-9/11 closure (2001-09-10→17) and the 2007-01-02 national day of
mourning; the >25% crypto moves are the 2020-03-12 COVID crash.

### Known weaknesses of this dataset (be honest with yourself)

1. **Survivorship bias** — not an issue for index ETFs themselves, but fatal
   for single-name strategies: Yahoo only has tickers that still exist.
   A "buy beaten-down large caps" backtest on today's S&P constituents
   silently excludes every company that went to zero. **Charter rule: no
   single-name equity backtests until a survivorship-bias-free source is in
   place.**
2. **Silent revisions** — Yahoo back-adjusts the whole series on every
   dividend; two downloads weeks apart will not be bit-identical. Keep
   results tied to a download date (the config snapshot in each
   BacktestResult records the data dir, and Parquet files carry mtimes).
3. **No delistings, no point-in-time membership** — we cannot know what the
   S&P 500 contained on 2005-03-14, only what it contains today.
4. **Crypto history is short and regime-concentrated** — BTC has ~11 years
   (one structural bull regime with crashes), ETH ~8. Treat any crypto
   backtest as weakly powered.

## Price adjustment and why it matters

All saved prices are **split- AND dividend-adjusted** (total-return prices,
yfinance `auto_adjust=True`):

- **Unadjusted prices** make every split look like a -50% crash and every
  dividend like a small loss — unusable for signals or returns.
- **Split-only adjustment** (what many free CSV dumps give you) understates
  long-run equity returns badly: SPY's price return since 1993 is roughly
  half its total return. A strategy that holds through ex-dividend dates
  would look artificially worse than buy-and-hold.
- **Total-return adjustment** (ours) is right for signal generation and
  return comparison, with one caveat: adjusted prices are NOT the prices you
  would have traded at. Cost models in this framework are percentage-based
  (bps of fill price), so this does not distort cost modeling; it would only
  matter for per-share commission models or share-count constraints deep in
  history.

## How far back can we honestly test?

- **Index-level US equity**: SPY to 1993 here. Beyond that, S&P 500 *index*
  series exist to 1957 (and reconstructions to 1926 via CRSP) but are
  indexes, not tradable ETFs — fine for regime analysis, wrong for cost-
  realistic backtests.
- **Bonds/gold ETFs**: only to 2002/2004 as ETFs. Longer proxies: constant-
  maturity treasury yield series (FRED, 1962+) and spot gold (1968+ post
  gold-pool, 1975+ for US-legal ownership).
- **Single-name equities**: ~1950s with paid survivorship-bias-free data
  (CRSP-derived, Norgate); realistically the 1990s+ is where intraday-quality
  and microstructure assumptions resemble today.
- **Structural breaks to respect**: decimalization (2001), Reg NMS (2007),
  zero-commission retail (2019). Pre-2001 fills/spreads were materially
  worse than today's — use long history to test whether an edge existed
  across regimes, not to estimate today's expected return from 1995 spreads.

## The Norgate upgrade path (paid, survivorship-bias-free)

The entire framework touches data through two seams, so the swap is small
and nothing downstream changes:

1. **A new adapter** `framework/data/sources/norgate_source.py` implementing
   the same `DataSource.fetch()` protocol via the `norgatedata` package
   (requires their Windows-based updater or API setup, plus a subscription).
   Register it; then `tf data download --source norgate`.
2. **The same Parquet contract** — files land in `data/ohlcv/` exactly as
   before; the loader, strategies, and backtester are unchanged.

What the paid tier actually buys (this is the point, not "more bars"):

- **Delisted securities**: every stock that ever traded, including the ones
  that went to zero — kills survivorship bias.
- **Point-in-time index membership**: "was X in the S&P 500 on date D?" —
  enables honest universe filters. This needs one framework addition (a
  `universe_membership(symbol, date)` helper reading Norgate's watchlists);
  strategies would receive pre-filtered data dicts, so the Strategy
  interface itself is unchanged.
- **Vendor-audited adjustments** instead of Yahoo's best effort.

Data and license keys stay out of git either way: `data/**` and `.env` are
gitignored, and any derived artifact of paid data (e.g. a hardcoded
membership list) must live in `data/` or a gitignored config, never in
committed source.

## Switching to intraday later

The loader contract is frequency-agnostic; what changes is:

1. A source that serves intraday bars (yfinance: only ~60–730 days depending
   on interval — useless for backtesting; realistic options are Alpaca's
   data API (free with account, 2016+) or Polygon (paid)).
2. Store per-frequency directories (e.g. `data/ohlcv_1h/`) and add a
   `frequency` field to `DataConfig`.
3. `metrics.py` annualization (`TRADING_DAYS = 252`) must become
   frequency-aware.
4. The charter currently forbids intraday trading (PDT rule), so this is
   deliberately unbuilt.
