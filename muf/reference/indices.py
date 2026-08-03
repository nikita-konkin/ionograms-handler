"""Solar activity indices, fetched and cached.

Empirical ionospheric models are driven by solar activity. Which index depends
on the model: MINIMUF and the CCIR maps want a smoothed sunspot number, IRI
accepts R12, IG12 or F10.7.

Sources, both verified reachable:

* **SILSO** (Royal Observatory of Belgium) -- daily and monthly-smoothed
  international sunspot number, version 2.0.
* **NOAA SWPC** -- monthly observed and smoothed sunspot number and F10.7.

One caveat that shapes the whole module: **a smoothed index does not exist for
recent months.** R12 is a 13-month centred mean, so it needs six months either
side; SILSO and SWPC publish ``-1.0`` until then. For a February 2026 sounding
there is no R12 until about August 2026. :func:`solar_indices` therefore falls
back, in order, to the smoothed value, a partial centred mean, and finally the
daily number, and always reports which was used.
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

SILSO_DAILY = "https://www.sidc.be/SILSO/DATA/SN_d_tot_V2.0.csv"
SILSO_MONTHLY_SMOOTHED = "https://www.sidc.be/SILSO/DATA/SN_ms_tot_V2.0.csv"
SWPC_SOLAR_CYCLE = (
    "https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json"
)

DEFAULT_CACHE = Path.home() / ".cache" / "muf" / "indices"
CACHE_MAX_AGE_DAYS = 7
TIMEOUT_S = 60

#: SILSO uses -1 for "not yet determined".
MISSING = -1.0


class IndexUnavailable(RuntimeError):
    """No index could be obtained, from network or cache."""


@dataclass(frozen=True)
class SolarIndices:
    """Solar activity for one date, with its provenance."""

    date: dt.date
    ssn_daily: float | None = None       # international sunspot number, that day
    ssn_smoothed: float | None = None    # R12, when it exists
    f107: float | None = None            # 10.7 cm flux, monthly
    f107_smoothed: float | None = None
    source: str = ""

    @property
    def r12(self) -> float | None:
        """Best available stand-in for R12, smoothed where possible."""
        return self.ssn_smoothed if self.ssn_smoothed is not None else self.ssn_daily

    @property
    def is_smoothed(self) -> bool:
        return self.ssn_smoothed is not None

    def __str__(self) -> str:
        parts = [f"{self.date}"]
        if self.ssn_daily is not None:
            parts.append(f"SSN {self.ssn_daily:.0f}")
        if self.ssn_smoothed is not None:
            parts.append(f"R12 {self.ssn_smoothed:.1f}")
        else:
            parts.append("R12 unavailable (needs +/-6 months)")
        if self.f107 is not None:
            parts.append(f"F10.7 {self.f107:.1f}")
        return "  ".join(parts)


def _fetch(url: str, cache_dir: Path, name: str, offline: bool = False) -> str:
    """Download ``url``, caching it. Falls back to a stale cache on failure."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / name

    if cached.exists():
        age = dt.datetime.now() - dt.datetime.fromtimestamp(cached.stat().st_mtime)
        if offline or age.days < CACHE_MAX_AGE_DAYS:
            return cached.read_text(encoding="utf-8", errors="replace")

    if offline:
        raise IndexUnavailable(f"offline and nothing cached at {cached}")

    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response:
            text = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if cached.exists():       # stale beats nothing
            return cached.read_text(encoding="utf-8", errors="replace")
        raise IndexUnavailable(f"could not fetch {url}: {exc}") from exc

    cached.write_text(text, encoding="utf-8")
    return text


def _parse_silso_daily(text: str) -> dict[dt.date, float]:
    """``year;month;day;decimal_year;SN;stdev;n_obs;definitive``, semicolon-separated."""
    out: dict[dt.date, float] = {}
    for line in text.splitlines():
        parts = line.split(";")
        if len(parts) < 5:
            continue
        try:
            day = dt.date(int(parts[0]), int(parts[1]), int(parts[2]))
            value = float(parts[4])
        except ValueError:
            continue
        if value != MISSING:
            out[day] = value
    return out


def _parse_silso_monthly(text: str) -> dict[tuple[int, int], float]:
    """``year;month;decimal_year;smoothed;stdev;n_obs;definitive``."""
    out: dict[tuple[int, int], float] = {}
    for line in text.splitlines():
        parts = line.split(";")
        if len(parts) < 4:
            continue
        try:
            key = (int(parts[0]), int(parts[1]))
            value = float(parts[3])
        except ValueError:
            continue
        if value != MISSING:
            out[key] = value
    return out


def _parse_swpc(text: str) -> dict[tuple[int, int], dict]:
    """SWPC monthly indices, keyed by (year, month)."""
    out: dict[tuple[int, int], dict] = {}
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        return out
    for row in rows:
        tag = str(row.get("time-tag", ""))
        if "-" not in tag:
            continue
        try:
            year, month = (int(p) for p in tag.split("-")[:2])
        except ValueError:
            continue
        out[(year, month)] = row
    return out


def _clean(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number == MISSING else number


def _centred_mean(monthly: dict[tuple[int, int], float],
                  year: int, month: int, half_window: int = 6) -> float | None:
    """A partial 13-month centred mean, for dates too recent to have R12.

    Averages whatever months are available within +/-6. Not R12 -- the value
    will drift as later months arrive -- but far better than a single day's
    number, which swings with individual active regions.
    """
    values = []
    for offset in range(-half_window, half_window + 1):
        m = month + offset
        y = year + (m - 1) // 12
        m = (m - 1) % 12 + 1
        if (y, m) in monthly:
            values.append(monthly[(y, m)])
    if len(values) < 3:
        return None
    return sum(values) / len(values)


def solar_indices(
    date: dt.date | dt.datetime,
    cache_dir: Path | None = None,
    offline: bool = False,
) -> SolarIndices:
    """Solar indices for ``date``, from cache or the network.

    Never raises for a merely-missing smoothed index; check
    :attr:`SolarIndices.is_smoothed` to know what you got.
    """
    if isinstance(date, dt.datetime):
        date = date.date()
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE

    daily_text = _fetch(SILSO_DAILY, cache_dir, "silso_daily.csv", offline)
    daily = _parse_silso_daily(daily_text)
    ssn_daily = daily.get(date)

    smoothed = None
    monthly_series: dict[tuple[int, int], float] = {}
    try:
        monthly_text = _fetch(SILSO_MONTHLY_SMOOTHED, cache_dir,
                              "silso_monthly.csv", offline)
        monthly_series = _parse_silso_monthly(monthly_text)
        smoothed = monthly_series.get((date.year, date.month))
    except IndexUnavailable:
        pass

    f107 = f107_smoothed = None
    try:
        swpc_text = _fetch(SWPC_SOLAR_CYCLE, cache_dir, "swpc_cycle.json", offline)
        row = _parse_swpc(swpc_text).get((date.year, date.month))
        if row:
            f107 = _clean(row.get("f10.7"))
            f107_smoothed = _clean(row.get("smoothed_f10.7"))
            if smoothed is None:
                smoothed = _clean(row.get("smoothed_ssn"))
    except IndexUnavailable:
        pass

    source = "SILSO daily"
    if smoothed is not None:
        source += " + smoothed R12"
    else:
        # Too recent for a centred mean to be complete; approximate it and say so.
        if not monthly_series:
            monthly_series = _monthly_from_daily(daily)
        partial = _centred_mean(monthly_series, date.year, date.month)
        if partial is not None:
            smoothed = partial
            source += " + partial centred mean (R12 not yet published)"
        else:
            source += " (no smoothed index available)"

    return SolarIndices(
        date=date,
        ssn_daily=ssn_daily,
        ssn_smoothed=smoothed,
        f107=f107,
        f107_smoothed=f107_smoothed,
        source=source,
    )


def _monthly_from_daily(daily: dict[dt.date, float]) -> dict[tuple[int, int], float]:
    """Monthly means built from the daily series, when no monthly file loaded."""
    buckets: dict[tuple[int, int], list[float]] = {}
    for day, value in daily.items():
        buckets.setdefault((day.year, day.month), []).append(value)
    return {k: sum(v) / len(v) for k, v in buckets.items() if v}
