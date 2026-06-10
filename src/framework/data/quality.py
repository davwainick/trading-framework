"""Data-quality checks: garbage data makes every later phase a lie.

Every downloaded file is checked before it is trusted. Checks are heuristics
tuned for daily bars; they REPORT problems rather than silently fixing them,
because the right fix depends on the cause (vendor error vs real market event
— e.g. a -20% day is a defect flag on a bond ETF but real life for crypto).

The calendar check infers the expected bar frequency from the data itself
(equities trade ~5 days/week, crypto 7), so both work without an exchange-
calendar dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: |daily log return| above this is flagged as a suspicious spike.
SPIKE_THRESHOLD = 0.25
#: A calendar gap longer than this many expected bars is flagged.
MAX_MISSING_RUN = 3
#: Relative tolerance for OHLC consistency: split/dividend back-adjustment
#: introduces float rounding ~1e-16; real bad prints are orders larger.
OHLC_RTOL = 1e-9


@dataclass
class Issue:
    check: str
    severity: str  # "error" (do not trust) | "warning" (inspect)
    message: str


@dataclass
class DataQualityReport:
    symbol: str
    n_bars: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        span = f"{self.start.date()} -> {self.end.date()}" if self.n_bars else "no data"
        status = "OK" if self.ok else "PROBLEMS"
        lines = [f"{self.symbol}: {self.n_bars} bars [{span}] {status}"]
        for issue in self.issues:
            lines.append(f"  [{issue.severity.upper():7}] {issue.check}: {issue.message}")
        return "\n".join(lines)


def check_ohlcv(df: pd.DataFrame, symbol: str) -> DataQualityReport:
    """Run all quality checks on one symbol's OHLCV history."""
    report = DataQualityReport(
        symbol=symbol,
        n_bars=len(df),
        start=df.index[0] if len(df) else None,
        end=df.index[-1] if len(df) else None,
    )
    if len(df) == 0:
        report.issues.append(Issue("empty", "error", "no rows"))
        return report

    _check_index(df, report)
    _check_prices(df, report)
    _check_ohlc_consistency(df, report)
    _check_gaps_and_spikes(df, report)
    _check_calendar(df, report)
    _check_volume(df, report)
    return report


def _check_index(df: pd.DataFrame, report: DataQualityReport) -> None:
    dupes = int(df.index.duplicated().sum())
    if dupes:
        report.issues.append(Issue("duplicate_rows", "error", f"{dupes} duplicated timestamps"))
    if not df.index.is_monotonic_increasing:
        report.issues.append(Issue("unsorted_index", "error", "index is not sorted"))


def _check_prices(df: pd.DataFrame, report: DataQualityReport) -> None:
    price_cols = ["open", "high", "low", "close"]
    nan_rows = int(df[price_cols].isna().any(axis=1).sum())
    if nan_rows:
        report.issues.append(Issue("nan_prices", "error", f"{nan_rows} rows with NaN prices"))
    bad = int((df[price_cols] <= 0).any(axis=1).sum())
    if bad:
        report.issues.append(
            Issue("nonpositive_prices", "error", f"{bad} rows with zero/negative prices")
        )


def _check_ohlc_consistency(df: pd.DataFrame, report: DataQualityReport) -> None:
    tol = df["close"].abs() * OHLC_RTOL
    bad_high = (df["high"] < df[["open", "close", "low"]].max(axis=1) - tol) & df["high"].notna()
    bad_low = (df["low"] > df[["open", "close", "high"]].min(axis=1) + tol) & df["low"].notna()
    n = int(bad_high.sum() + bad_low.sum())
    if n:
        report.issues.append(
            Issue("ohlc_inconsistent", "error", f"{n} bars violate low <= open/close <= high")
        )


def _check_gaps_and_spikes(df: pd.DataFrame, report: DataQualityReport) -> None:
    close = df["close"]
    log_ret = np.log(close / close.shift(1)).dropna()
    spikes = log_ret[log_ret.abs() > SPIKE_THRESHOLD]
    if len(spikes):
        worst = spikes.abs().idxmax()
        report.issues.append(
            Issue(
                "price_spikes",
                "warning",
                f"{len(spikes)} daily moves > {SPIKE_THRESHOLD:.0%} "
                f"(worst {np.expm1(log_ret[worst]):+.1%} on {worst.date()}) — "
                "verify vs known events / check for unadjusted splits",
            )
        )


def _check_calendar(df: pd.DataFrame, report: DataQualityReport) -> None:
    if len(df) < 10:
        report.issues.append(Issue("short_history", "warning", "fewer than 10 bars"))
        return
    day_gaps = df.index.to_series().diff().dt.days.dropna()
    # Median gap ~1 day; equities show weekend gaps of 3 days, crypto none.
    typical = float(day_gaps.median()) or 1.0
    threshold = max(MAX_MISSING_RUN * typical, 4.0 if _looks_like_equity(day_gaps) else 1.0)
    gaps = day_gaps[day_gaps > threshold]
    if len(gaps):
        worst_at = gaps.idxmax()
        report.issues.append(
            Issue(
                "calendar_gaps",
                "warning",
                f"{len(gaps)} gaps > {threshold:.0f} days "
                f"(longest {int(gaps.max())} days ending {worst_at.date()})",
            )
        )


def _looks_like_equity(day_gaps: pd.Series) -> bool:
    """Weekend gaps (>=3 days) in >10% of bars implies a 5-day trading week."""
    return float((day_gaps >= 3).mean()) > 0.10


def _check_volume(df: pd.DataFrame, report: DataQualityReport) -> None:
    if "volume" not in df.columns:
        return
    zero_run = (df["volume"] == 0).astype(int)
    if zero_run.sum() == 0:
        return
    # Long runs of zero volume suggest stale/placeholder data.
    runs = zero_run.groupby((zero_run != zero_run.shift()).cumsum()).sum()
    longest = int(runs.max())
    if longest >= 5:
        report.issues.append(
            Issue("zero_volume", "warning", f"longest zero-volume run is {longest} bars")
        )
