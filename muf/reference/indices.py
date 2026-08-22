"""Solar activity indices, fetched and cached.

Empirical ionospheric models are driven by solar activity. Which index depends
on the model: MINIMUF and the CCIR maps want a smoothed sunspot number, IRI
accepts R12, IG12 or F10.7.

Sources, every one of them verified reachable and parsed against a known
value:

* **SILSO** (Royal Observatory of Belgium) -- daily and monthly-smoothed
  international sunspot number, version 2.0, plus ``EISN``, the estimated
  number for the *current* month.
* **NOAA SWPC** -- monthly observed and smoothed sunspot number and F10.7,
  and the last 42 days of daily 10.7 cm flux.
* **irimodel.org** -- ``apf107.dat``, the driver file IRI itself reads: daily
  F10.7, its 81-day centred mean, and daily ``ap``, back to 1958-01-01.

The set is deliberately redundant. Any one of them can be down, or blocked by
a firewall, without the model losing its driver -- :func:`solar_indices`
raises only when *nothing* answered and nothing is cached, and always reports
which sources it actually used.

Two caveats shape the whole module.

**A smoothed index does not exist for recent months.** R12 is a 13-month
centred mean, so it needs six months either side; SILSO and SWPC publish
``-1.0`` until then. For a February 2026 sounding there is no R12 until about
August 2026. :func:`solar_indices` therefore falls back, in order, to the
smoothed value, a partial centred mean, and finally the daily number, and
always reports which was used.

**The daily F10.7 is not the model driver.** The CCIR and URSI maps IRI
interpolates were fitted against a *smoothed* index; feeding them a single
day's flux injects rotation-scale variability the maps cannot represent.
:attr:`SolarIndices.f107_driver` is what a model should read -- the 81-day
centred mean where it exists -- while :attr:`SolarIndices.f107` carries the
observed daily number, which is what an operator wants to see.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path

SILSO_DAILY = "https://www.sidc.be/SILSO/DATA/SN_d_tot_V2.0.csv"
SILSO_MONTHLY_SMOOTHED = "https://www.sidc.be/SILSO/DATA/SN_ms_tot_V2.0.csv"
SILSO_EISN = "https://www.sidc.be/SILSO/DATA/EISN/EISN_current.csv"
SWPC_SOLAR_CYCLE = (
    "https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json"
)
SWPC_F107_DAILY = "https://services.swpc.noaa.gov/json/f107_cm_flux.json"
IRI_APF107 = "https://irimodel.org/indices/apf107.dat"

#: Roughly 5 MB of index files land here. Overridable because in a container
#: ``$HOME`` is inside the image layer: every ``docker compose pull`` would
#: throw the cache away and hand the next caller a 5 MB download, and a host
#: with no route out would lose the only copy it had. Point it at a volume --
#: ``deploy/docker-compose.yml`` uses ``/data/indices``, the volume that is
#: already there for the database.
DEFAULT_CACHE = Path(os.environ.get("MUF_INDEX_CACHE")
                     or Path.home() / ".cache" / "muf" / "indices")
CACHE_MAX_AGE_DAYS = 7

#: Shorter than this, a cached copy is not a document -- it is a failed
#: download that got written anyway. The smallest real source here is
#: `silso_monthly.csv` at tens of kilobytes, so the threshold only has to
#: separate "a file" from "nothing"; it is deliberately not tuned per source,
#: because a source that legitimately shrinks to 64 bytes has broken in a way
#: this module should not paper over either.
MIN_USEFUL_BYTES = 64
TIMEOUT_S = 60

#: Identifying the client is good manners -- these are volunteer-run services
#: and a nameless robot is the first thing an operator blocks -- but here it is
#: also load-bearing: irimodel.org runs mod_security, which answers ``406 Not
#: Acceptable`` to urllib's default ``Python-urllib/3.12`` *and* to
#: ``curl/8.7.1``.
#:
#: **Keep this string short and plain.** The rules are opaque and reject on
#: punctuation as much as on content: this exact value passes, and
#: ``ionograms-handler/0.1 (+solar index fetch; python-urllib)`` -- the obvious
#: "more informative" version -- is refused. Re-test against
#: ``https://irimodel.org/indices/ig_rz.dat`` before changing it.
USER_AGENT = "ionograms-handler/0.1"

#: SILSO uses -1 for "not yet determined".
MISSING = -1.0


class IndexUnavailable(RuntimeError):
    """No index could be obtained, from network or cache."""


@dataclass(frozen=True)
class Source:
    """One upstream file, named so that something else can probe it.

    The reachability indicator on the console walks this list rather than
    pinging a public resolver, because "the internet is up" and "this host can
    refresh its solar indices" are different questions and only the second one
    changes what the IRI panel shows.
    """

    key: str
    url: str
    filename: str
    what: str

    @property
    def host(self) -> str:
        from urllib.parse import urlsplit
        return urlsplit(self.url).netloc


#: Every file this module reads, in the order :func:`solar_indices` needs them.
SOURCES: tuple[Source, ...] = (
    Source("silso_daily", SILSO_DAILY, "silso_daily.csv",
           "daily international sunspot number"),
    Source("silso_eisn", SILSO_EISN, "silso_eisn.csv",
           "estimated sunspot number, current month"),
    Source("silso_monthly", SILSO_MONTHLY_SMOOTHED, "silso_monthly.csv",
           "13-month smoothed sunspot number (R12)"),
    Source("swpc_cycle", SWPC_SOLAR_CYCLE, "swpc_cycle.json",
           "monthly sunspot number and F10.7"),
    Source("swpc_f107", SWPC_F107_DAILY, "swpc_f107_daily.json",
           "daily 10.7 cm flux, last 42 days"),
    Source("iri_apf107", IRI_APF107, "apf107.dat",
           "daily F10.7, its 81-day mean, and ap, since 1958"),
)

BY_KEY = {s.key: s for s in SOURCES}


@dataclass(frozen=True)
class SolarIndices:
    """Solar activity for one date, with its provenance."""

    date: dt.date
    ssn_daily: float | None = None       # international sunspot number, that day
    ssn_smoothed: float | None = None    # R12, when it exists
    f107: float | None = None            # 10.7 cm flux, that day
    f107_81: float | None = None         # 81-day centred mean of the above
    f107_monthly: float | None = None    # SWPC monthly observed
    f107_smoothed: float | None = None   # SWPC monthly smoothed
    ap: float | None = None              # daily planetary amplitude
    source: str = ""
    #: Sources that answered, by key. Absence here is not an error -- it is
    #: how a caller tells "the model ran on a full driver set" from "the model
    #: ran on whatever was left when two hosts were unreachable".
    used: tuple[str, ...] = ()

    @property
    def r12(self) -> float | None:
        """Best available stand-in for R12, smoothed where possible."""
        return self.ssn_smoothed if self.ssn_smoothed is not None else self.ssn_daily

    @property
    def f107_driver(self) -> float | None:
        """The flux a *model* should be given, smoothest available first.

        Not :attr:`f107`. A single day's flux moves 50 SFU across one solar
        rotation while the ionosphere's monthly climatology does not, so
        handing the daily number to a map fitted on smoothed indices produces
        swings the map never claimed to predict.
        """
        for value in (self.f107_81, self.f107_smoothed, self.f107_monthly,
                      self.f107):
            if value is not None:
                return value
        return None

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
        if self.f107_81 is not None:
            parts.append(f"F10.7-81 {self.f107_81:.1f}")
        if self.ap is not None:
            parts.append(f"ap {self.ap:.0f}")
        return "  ".join(parts)


def _usable(path: Path) -> bool:
    """Whether a cached file is worth reading.

    An empty file is not a cache hit, it is the wreckage of a failed fetch.
    Treating it as absent is what lets the retry actually retry, and what
    stops `cache_state` reporting a destroyed cache as a fresh one.
    """
    try:
        return path.stat().st_size >= MIN_USEFUL_BYTES
    except OSError:
        return False


def _fetch(url: str, cache_dir: Path, name: str, offline: bool = False) -> str:
    """Download ``url``, caching it. Falls back to a stale cache on failure."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / name

    if _usable(cached):
        age = dt.datetime.now() - dt.datetime.fromtimestamp(cached.stat().st_mtime)
        if offline or age.days < CACHE_MAX_AGE_DAYS:
            return cached.read_text(encoding="utf-8", errors="replace")

    if offline:
        raise IndexUnavailable(f"offline and nothing cached at {cached}")

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            status = getattr(response, "status", None) or response.getcode()
            text = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if _usable(cached):       # stale beats nothing
            return cached.read_text(encoding="utf-8", errors="replace")
        raise IndexUnavailable(f"could not fetch {url}: {exc}") from exc

    # Every source here is a static file, so 200 is the only correct answer and
    # `urlopen` raises for none of the others in the 2xx range. A station met
    # **202 Accepted with an empty body** from `services.swpc.noaa.gov` -- not
    # something NOAA sends, but exactly what an intercepting proxy returns when
    # it swallows a request. Treated as success, that emptied two cached files
    # and left the ionospheric model with no solar driver.
    if status != 200:
        if _usable(cached):
            return cached.read_text(encoding="utf-8", errors="replace")
        raise IndexUnavailable(
            f"{url} answered {status} with {len(text)} byte(s). Only 200 is a "
            f"document here; a 2xx that is not 200 -- especially 202 with an "
            f"empty body -- is usually a proxy or filtering appliance "
            f"intercepting the request rather than the host itself. Check "
            f"whether this host is reachable un-proxied from the container.")

    # An empty 200 is not an exception, so it used to reach the write below and
    # replace a good file with nothing. That is how a station lost 42 days of
    # daily F10.7 while `cache_state` went on reporting the cache as fresh --
    # mtime had been updated by the very write that destroyed it, so nothing
    # anywhere said the model had stopped having a solar driver.
    if len(text.encode("utf-8")) < MIN_USEFUL_BYTES:
        if _usable(cached):
            return cached.read_text(encoding="utf-8", errors="replace")
        raise IndexUnavailable(
            f"{url} answered with {len(text)} byte(s), which is not a "
            f"document; nothing cached to fall back to")

    # Written beside the target and moved into place, so a fetch interrupted
    # half way leaves the previous copy intact rather than a truncated one
    # that would pass every check above.
    #
    # A cache that cannot be written is not a failed fetch: the document is in
    # hand and the caller can have it. Raising here instead cost a station its
    # ionospheric model, because a `docker cp` had left two cache files owned
    # by another uid and every later refresh died on the write rather than on
    # the download it had already completed.
    temporary = cached.with_name(cached.name + ".partial")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(cached)
    except OSError as exc:
        warnings.warn(f"fetched {url} but could not cache it at {cached}: "
                      f"{exc}. The value is correct and will be re-fetched "
                      f"every time until the cache is writable.", stacklevel=2)
        try:
            temporary.unlink()
        except OSError:
            pass
    return text


def cache_state(cache_dir: Path | None = None) -> dict[str, float | None]:
    """Age in seconds of each source's cached copy, ``None`` when absent.

    The indicator shows this beside reachability because they fail
    independently and mean different things: unreachable with a fresh cache is
    a model still answering correctly, while reachable with no cache is a
    model that has never run.
    """
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    now = dt.datetime.now().timestamp()
    out: dict[str, float | None] = {}
    for source in SOURCES:
        path = cache_dir / source.filename
        # An unusable file reports as absent, which is what it is. Reporting
        # its age instead is how an empty cache looked healthy for two days.
        try:
            out[source.key] = now - path.stat().st_mtime if _usable(path) else None
        except OSError:
            out[source.key] = None
    return out


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


def _parse_silso_eisn(text: str) -> dict[dt.date, float]:
    """``year, month, day, decimal_year, EISN, sd, n_calc, n_obs``, comma-separated.

    The definitive daily series runs about two weeks behind -- on 2026-08-13
    ``SN_d_tot`` ended at 2026-07-31 -- so without this a sounding taken today
    has no sunspot number at all. EISN is provisional and gets revised, which
    is why it is only consulted for dates the definitive file does not cover.
    """
    out: dict[dt.date, float] = {}
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
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


#: The Penticton observation that *is* the daily F10.7. SWPC publishes three a
#: day -- 17:00, 20:00 and 22:00 UT -- and only the 20:00 one is taken at local
#: noon, which is the value every index series quotes. Taking the newest
#: instead would silently mix a morning reading into a series of noon ones.
SWPC_NOON_UT = "20:00"


def _parse_swpc_f107_daily(text: str) -> dict[dt.date, float]:
    """SWPC's rolling 42 days of 10.7 cm flux, one value per day."""
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        return {}

    best: dict[dt.date, tuple[int, float]] = {}
    for row in rows:
        tag = str(row.get("time_tag", ""))
        try:
            day = dt.date.fromisoformat(tag[:10])
            flux = float(row["flux"])
        except (ValueError, KeyError, TypeError):
            continue
        if flux <= 0:
            continue
        # Rank: the noon reading wins; otherwise keep the latest of the day so
        # that today, which has only a morning reading, still reports a value.
        rank = 2 if tag[11:16] == SWPC_NOON_UT else 1
        if day not in best or rank > best[day][0]:
            best[day] = (rank, flux)
    return {day: flux for day, (_, flux) in best.items()}


def _apf107_year(two_digit: int) -> int:
    """``58`` is 1958 and ``26`` is 2026; the file starts 1958-01-01."""
    return 1900 + two_digit if two_digit >= 58 else 2000 + two_digit


def _parse_apf107(text: str) -> dict[dt.date, dict[str, float]]:
    """IRI's own driver file: fixed columns, one line per day since 1958.

    ``3I3`` date, ``8I3`` three-hourly ap, ``I3`` daily Ap, ``I3`` a slot that
    reads ``-11`` on every line in the file and is ignored, then ``3F5.1`` for
    F10.7 daily, its 81-day centred mean and its 365-day mean.

    Parsed by column and not by whitespace: ``ap`` reaches 400 in a severe
    storm and three digits in a three-wide field leave no space between
    columns, so a split on spaces merges them.
    """
    out: dict[dt.date, dict[str, float]] = {}
    for line in text.splitlines():
        if len(line) < 49:
            continue
        try:
            day = dt.date(_apf107_year(int(line[0:3])),
                          int(line[3:6]), int(line[6:9]))
            record = {"ap": float(line[33:36]),
                      "f107": float(line[39:44]),
                      "f107_81": float(line[44:49])}
        except ValueError:
            continue
        out[day] = {k: v for k, v in record.items() if v > 0}
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


def _daily_window_mean(daily: dict[dt.date, float], date: dt.date,
                       half_window: int = 40,
                       minimum: int = 30) -> float | None:
    """An 81-day centred mean built from a daily series.

    ``apf107.dat`` carries this ready-made, but it runs about two weeks
    behind, so the newest soundings -- the ones on the console right now -- fall
    off its end. Merging SWPC's rolling 42 days into the same series and
    averaging here covers exactly that gap, at the cost of a mean that is
    trailing rather than centred until the days after the sounding arrive.
    """
    values = [daily[date + dt.timedelta(days=d)]
              for d in range(-half_window, half_window + 1)
              if date + dt.timedelta(days=d) in daily]
    if len(values) < minimum:
        return None
    return sum(values) / len(values)


def solar_indices(
    date: dt.date | dt.datetime,
    cache_dir: Path | None = None,
    offline: bool = False,
) -> SolarIndices:
    """Solar indices for ``date``, from cache or the network.

    Never raises for a merely-missing smoothed index; check
    :attr:`SolarIndices.is_smoothed` to know what you got. Nor for a single
    unreachable host: each source is optional on its own and
    :exc:`IndexUnavailable` is raised only when every one of them failed and
    nothing was cached, because a firewall that blocks one of two providers
    should not stop a model that the other one can still drive.
    """
    if isinstance(date, dt.datetime):
        date = date.date()
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE

    used: list[str] = []
    failures: list[str] = []

    def read(key: str) -> str | None:
        source = BY_KEY[key]
        try:
            text = _fetch(source.url, cache_dir, source.filename, offline)
        except IndexUnavailable as exc:
            failures.append(f"{source.key}: {exc}")
            return None
        used.append(source.key)
        return text

    # --- sunspot number -----------------------------------------------------
    daily: dict[dt.date, float] = {}
    daily_text = read("silso_daily")
    if daily_text:
        daily = _parse_silso_daily(daily_text)
    ssn_daily = daily.get(date)

    provisional = False
    if ssn_daily is None:
        # Only when the definitive series does not reach this far: EISN is
        # revised for weeks afterwards and would otherwise overwrite a
        # settled number with an estimate of it.
        eisn_text = read("silso_eisn")
        if eisn_text:
            ssn_daily = _parse_silso_eisn(eisn_text).get(date)
            provisional = ssn_daily is not None

    smoothed = None
    monthly_series: dict[tuple[int, int], float] = {}
    monthly_text = read("silso_monthly")
    if monthly_text:
        monthly_series = _parse_silso_monthly(monthly_text)
        smoothed = monthly_series.get((date.year, date.month))

    # --- flux ---------------------------------------------------------------
    f107_monthly = f107_smoothed = None
    cycle_text = read("swpc_cycle")
    if cycle_text:
        row = _parse_swpc(cycle_text).get((date.year, date.month))
        if row:
            f107_monthly = _clean(row.get("f10.7"))
            f107_smoothed = _clean(row.get("smoothed_f10.7"))
            if smoothed is None:
                smoothed = _clean(row.get("smoothed_ssn"))

    f107_series: dict[dt.date, float] = {}
    ap = f107_81 = None
    apf107_text = read("iri_apf107")
    if apf107_text:
        apf107 = _parse_apf107(apf107_text)
        f107_series = {d: r["f107"] for d, r in apf107.items() if "f107" in r}
        today = apf107.get(date, {})
        ap = today.get("ap")
        f107_81 = today.get("f107_81")

    swpc_daily_text = read("swpc_f107")
    if swpc_daily_text:
        # Last, so its 42 fresh days win over apf107's older copy of the same
        # dates -- SWPC revises a preliminary flux and apf107 is rebuilt from
        # the revision, so on the overlap the newer file is the corrected one.
        f107_series.update(_parse_swpc_f107_daily(swpc_daily_text))

    f107 = f107_series.get(date)
    if f107_81 is None and f107_series:
        f107_81 = _daily_window_mean(f107_series, date)

    if not used:
        raise IndexUnavailable("; ".join(failures) or "no index source answered")

    # --- provenance ---------------------------------------------------------
    note = "SILSO daily"
    if provisional:
        note = "SILSO EISN (provisional, current month)"
    if smoothed is not None:
        note += " + smoothed R12"
    else:
        # Too recent for a centred mean to be complete; approximate it and say so.
        if not monthly_series:
            monthly_series = _monthly_from_daily(daily)
        partial = _centred_mean(monthly_series, date.year, date.month)
        if partial is not None:
            smoothed = partial
            note += " + partial centred mean (R12 not yet published)"
        else:
            note += " (no smoothed index available)"
    if f107_81 is not None:
        note += " + F10.7-81"
    elif f107 is not None:
        note += " + daily F10.7 only"
    if failures:
        note += f" ({len(failures)} source(s) unavailable)"

    return SolarIndices(
        date=date,
        ssn_daily=ssn_daily,
        ssn_smoothed=smoothed,
        f107=f107,
        f107_81=f107_81,
        f107_monthly=f107_monthly,
        f107_smoothed=f107_smoothed,
        ap=ap,
        source=note,
        used=tuple(used),
    )


def _monthly_from_daily(daily: dict[dt.date, float]) -> dict[tuple[int, int], float]:
    """Monthly means built from the daily series, when no monthly file loaded."""
    buckets: dict[tuple[int, int], list[float]] = {}
    for day, value in daily.items():
        buckets.setdefault((day.year, day.month), []).append(value)
    return {k: sum(v) / len(v) for k, v in buckets.items() if v}
