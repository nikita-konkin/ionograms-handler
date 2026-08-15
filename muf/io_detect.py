"""Reading chirpsounder2 detection files -- which transmitters, on what schedule.

``io_chirp`` reads the ionogram a sounding produced. This module reads the
files that decide *which* soundings exist at all, which is the whole point of
the v2 migration (``architecture.md`` sec. 2.5): v1's schedule table is
hand-maintained, so a transmitter nobody knew about was invisible. v2 searches,
finds, and writes down what it found.

Three files answer the question, in increasing order of processing:

* ``chirp-*.h5`` -- one detection. ``detect_chirps.py`` matched a chirp rate in
  one analysis block and wrote when, at what frequency, how strong.
* ``par-*.h5`` -- one *timing solution*. ``find_timings.py`` clustered
  detections that share a rate and an arrival phase and fitted the sweep they
  belong to. This is the file that says "a transmitter exists here".
* ``cdetections-*.h5`` -- a time-binned dump of raw detections,
  ``[chirp_time, i0/25e6, f0, chirp_rate, snr]`` per row
  (``detections2metadata.py:109``). Redundant with ``chirp-*.h5`` but cheaper
  to sweep, and the only one of the three that is per-station rather than
  per-channel.

**The fractional second is not a propagation delay.** Every range in the chirp
world comes from ``chirp_time``'s fractional part -- transmitters start their
sweep on an integer second, so whatever is left over is the travel time
(``chirp_det.py:261``). That identity holds only if the *receiver* agrees with
the transmitter about where the second begins. Measured at DOB on 2026-08-05,
it did not: the recorder's epoch was 0.956 s out, and every implied range was
nonsense -- 16,700 km for cyprus1, whose path is 3436 km. Nothing in the files
shows it. ``chirp_time`` was stable to 0.5 ms across eleven hours, the schedule
was self-consistent, the ionograms plotted.

So this module reports the fractional second as :attr:`Detection.fraction_s`
and refuses to call it a range. Turning it into one needs an epoch offset, and
:func:`solve_epoch_offset` measures that against a transmitter whose position
*and* published transmit seconds are known independently -- cyprus1 at DOB gave
0.35 ms agreement across four slots. Until then, use the schedule: transmit
seconds are whole numbers, so identification survives an epoch error of up to
half a second even when ranging does not.
"""

from __future__ import annotations

import datetime as _dt
import math
import os
import statistics
import warnings
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .io_chirp import C_M_S, MAX_VIRTUAL_RANGE_KM, _scalar
from .paths import dedupe_paths

#: Default schedule cycle. Chirp transmitters repeat on a human-chosen round
#: number; 300 s covers the ROTHR family, cyprus1 and the Twente chirp list's
#: own notation (``300:235`` = "period 300 s, starts at second 235"). Emitters
#: on a shorter cycle still resolve -- they simply report every slot they
#: occupy within the 300 s window.
DEFAULT_CYCLE_S = 300.0

#: How far apart two arrival phases must be to count as different transmitters.
#: `find_timings.cluster_times` uses 0.1 s to group *detections*; this is the
#: much tighter figure for grouping *solutions*, and is set by what the phase
#: actually does: 0.5 ms of scatter within a slot at DOB, 0.19 ms/hour of
#: drift. 5 ms is ten sigma and still only 1500 km of range.
PHASE_TOLERANCE_S = 5e-3


def _fraction(value: float) -> float:
    """Fractional second, on ``[0, 1)``."""
    return float(value) - math.floor(float(value))


def _unwrap(fractions: np.ndarray) -> np.ndarray:
    """Centre a set of fractional seconds that may straddle the integer.

    An emitter whose phase sits near zero produces fractions of 0.999 and
    0.001 that are 2 ms apart and average to 0.5 -- a 150,000 km range and a
    transmitter that appears to be two transmitters. Every mean, standard
    deviation and cluster boundary in this module goes through here first.
    """
    fractions = np.asarray(fractions, dtype=np.float64)
    if fractions.size and float(np.ptp(fractions)) > 0.5:
        return np.where(fractions > 0.5, fractions - 1.0, fractions)
    return fractions


def _decode(value) -> str:
    raw = _scalar(value)
    return raw.decode() if isinstance(raw, bytes) else str(raw)


@dataclass(frozen=True)
class Detection:
    """One ``chirp-*.h5``: a single chirp, seen once.

    ``snr`` is v2's detection statistic -- a linear power ratio from the
    matched filter, not this pipeline's dB and not comparable to the 43 dB
    threshold. It orders detections; it does not measure them.
    """

    path: Path
    channel: str
    chirp_time: float         # unix seconds, extrapolated 0 Hz crossing
    rate: float               # Hz/s
    f0: float                 # Hz at the moment of detection
    snr: float
    i0: int                   # sample index into the ringbuffer
    n_samples: int
    sample_rate: float

    @property
    def fraction_s(self) -> float:
        """Seconds past the whole second. See the module docstring."""
        return _fraction(self.chirp_time)

    @property
    def datetime(self) -> _dt.datetime:
        return _dt.datetime.fromtimestamp(self.chirp_time, tz=_dt.timezone.utc)


@dataclass(frozen=True)
class TimingSolution:
    """One ``par-*.h5``: ``find_timings.py``'s fit over clustered detections.

    ``t0`` is the fitted sweep start; ``t0s``, ``f0s`` and ``snrs`` are the
    individual detections it was fitted from. The fit is what makes these
    files worth reading separately from ``chirp-*.h5`` -- ``t0`` is stable to
    a few tenths of a millisecond where a single detection scatters by whole
    milliseconds.
    """

    path: Path
    channel: str
    t0: float                 # unix seconds, fitted sweep start
    rate: float               # Hz/s
    t0s: np.ndarray           # per-detection times the fit used
    f0s: np.ndarray           # per-detection frequencies, Hz
    snrs: np.ndarray
    num_detections: int
    software_version: str = ""
    git_commit: str = ""
    git_dirty: bool | None = None

    @property
    def fraction_s(self) -> float:
        return _fraction(self.t0)

    @property
    def datetime(self) -> _dt.datetime:
        return _dt.datetime.fromtimestamp(self.t0, tz=_dt.timezone.utc)

    @property
    def swept_hz(self) -> float:
        """Frequency span the fitted detections cover. Zero for a single one."""
        return float(self.f0s.max() - self.f0s.min()) if self.f0s.size else 0.0


@dataclass(frozen=True)
class Emitter:
    """One transmitter, as inferred from a run of detections or solutions.

    Grouped by chirp rate *and* arrival phase, which is the physical pairing:
    two soundings from the same transmitter share a rate and a path, so they
    share a fractional second. Two transmitters at the same range on the same
    rate would merge, and the schedule is what separates them -- see
    :attr:`transmit_seconds`.
    """

    rate: float                       # Hz/s
    fraction_s: float                 # mean arrival phase
    fraction_sd_s: float              # scatter within the group
    count: int
    observed_seconds: tuple[int, ...]  # whole seconds into the cycle, AS RECEIVED
    cycle_s: float
    first_seen: float                 # unix seconds
    last_seen: float
    snr_median: float

    @property
    def span_hours(self) -> float:
        return (self.last_seen - self.first_seen) / 3600.0

    def transmit_seconds(self, epoch_offset_s: float = 0.0) -> tuple[int, ...]:
        """The seconds the transmitter actually starts on.

        Not the same as :attr:`observed_seconds`, and the difference is not
        cosmetic. cyprus1's fourth slot arrived at DOB during second 299 and
        was transmitted on second 300 -- the receiver's epoch was 0.956 s
        slow, so every slot it reported was one whole second early. Published
        against the Twente list, ``299`` matches nothing and ``300`` matches
        exactly.

        Any epoch error past half a second shifts the whole schedule by a
        whole second while leaving it perfectly self-consistent, so the
        observed seconds cannot be corrected from within the data. Pass what
        :func:`solve_epoch_offset` measured.
        """
        cycle = int(self.cycle_s)
        return tuple(sorted({
            int(round(s - epoch_offset_s)) % cycle for s in self.observed_seconds
        }))

    def delay_s(self, epoch_offset_s: float = 0.0) -> float:
        """Propagation delay, given a measured receiver epoch offset.

        ``epoch_offset_s`` is what :func:`solve_epoch_offset` returns: the
        amount by which this receiver's timestamps run *late*, so it is
        subtracted. The default of zero is not a safe default -- it asserts a
        verified clock, which is exactly the assumption that produced a
        16,700 km cyprus1. It warns when the result is impossible, and it
        cannot detect the far more common case of a wrong-but-plausible one.
        """
        delay = _fraction(self.fraction_s - epoch_offset_s)
        if delay * C_M_S / 1e3 > MAX_VIRTUAL_RANGE_KM:
            warnings.warn(
                f"{self.rate / 1e3:.0f} kHz/s emitter at phase "
                f"{self.fraction_s:.5f}: delay {delay * 1e3:.1f} ms implies "
                f"{delay * C_M_S / 1e3:.0f} km, past the "
                f"{MAX_VIRTUAL_RANGE_KM:.0f} km a one-way terrestrial path can "
                f"reach. The receiver epoch is wrong, not the transmitter.",
                stacklevel=2,
            )
        return delay

    def range_km(self, epoch_offset_s: float = 0.0) -> float:
        """Virtual range in km. Read :meth:`delay_s` before trusting it."""
        return self.delay_s(epoch_offset_s) * C_M_S / 1e3

    def __str__(self) -> str:
        slots = ",".join(str(s) for s in self.observed_seconds)
        return (f"{self.rate / 1e3:.0f} kHz/s  rx {self.cycle_s:.0f}:{slots}  "
                f"n={self.count}  phase {self.fraction_s * 1e3:.2f} ms")


@dataclass(frozen=True)
class EpochOffset:
    """How far this receiver's clock runs from the transmitters' seconds.

    Positive means timestamps are *late*. Subtract it to recover propagation
    delay. ``residual_sd_s`` across independent slots is the figure that says
    whether to believe it: at DOB four cyprus1 slots agreed to 0.35 ms, which
    is 105 km, against an error of 0.956 s that nothing else had caught.
    """

    seconds: float
    residual_sd_s: float
    n_slots: int
    n_samples: int
    reference: str

    @property
    def range_uncertainty_km(self) -> float:
        return self.residual_sd_s * C_M_S / 1e3

    def __str__(self) -> str:
        return (f"epoch offset {self.seconds:+.5f} s from {self.reference} "
                f"({self.n_slots} slots, {self.n_samples} samples, "
                f"+/-{self.residual_sd_s * 1e3:.2f} ms = "
                f"{self.range_uncertainty_km:.0f} km)")


# --------------------------------------------------------------------------
# Finding and reading
# --------------------------------------------------------------------------

def _find(target, pattern: str) -> list[Path]:
    if isinstance(target, (str, Path)):
        targets = [Path(target)]
    else:
        targets = [Path(t) for t in target]
    if not targets:
        raise FileNotFoundError("no target given")

    found: list[Path] = []
    missing: list[Path] = []
    for item in targets:
        if item.is_file():
            # A named file still has to be the kind being looked for. Without
            # this, pointing at one cdetections-*.h5 hands it to the par-*.h5
            # reader and the chirp-*.h5 reader first, and the caller sees two
            # "skipped 1 unreadable file" warnings before the right reader
            # gets it. The three finders share this helper precisely so that
            # a caller can try them in turn.
            if fnmatch(item.name, pattern):
                found.append(item)
        elif item.is_dir():
            found.extend(item.rglob(pattern))
        else:
            missing.append(item)

    if missing and not found:
        raise FileNotFoundError(", ".join(str(m) for m in missing))
    return dedupe_paths(found)


def find_detections(target) -> list[Path]:
    """``chirp-*.h5`` under ``target``. Mirrors :func:`muf.io_chirp.find_h5`."""
    return _find(target, "chirp-*.h5")


def find_timings(target) -> list[Path]:
    """``par-*.h5`` under ``target``."""
    return _find(target, "par-*.h5")


def find_cdetections(target) -> list[Path]:
    """``cdetections-*.h5`` under ``target``."""
    return _find(target, "cdetections-*.h5")


#: The three products' filename prefixes, best first. A chirpsounder2 name is
#: ``<product>-<fields...>.h5``, so the text before the first ``-`` names the
#: product and one pass can sort a directory into all three.
PRODUCT_PREFIXES = ("par", "chirp", "cdetections")


def find_products(target) -> dict[str, list[Path]]:
    """All three detection products under ``target``, in **one** directory pass.

    The three finders above each walk the tree themselves, which is right when
    a caller wants one product and wasteful when it wants the best available --
    which is what a census wants, and what `muf detect` does. On DOB that cost
    three walks of 46,436 entries per day to establish that two of the three
    products are not there: the station writes no ``par-*.h5`` at all, and the
    walk that discovers this visits every one of the 45,602 ``chirp-*.h5`` to
    do it.

    Matching on the prefix rather than the glob also means a `Path` is built
    only for files that are wanted. The 750 ``lfm_ionogram-*.h5`` in that same
    directory are rejected on a string comparison.

    Keys are :data:`PRODUCT_PREFIXES`; every key is present, possibly empty.
    """
    out: dict[str, list[Path]] = {p: [] for p in PRODUCT_PREFIXES}
    target = Path(target)
    if target.is_file():
        # Same rule as `_find`: a named file still has to be the kind asked
        # for, or the wrong reader gets it.
        head = target.name.split("-", 1)[0]
        if head in out and target.name.endswith(".h5"):
            out[head].append(target)
        return out
    for root, _dirs, files in os.walk(target):
        base = Path(root)
        for name in files:
            if not name.endswith(".h5"):
                continue
            head = name.split("-", 1)[0]
            if head in out:
                out[head].append(base / name)
    return out


def read_detection(path: str | Path) -> Detection:
    """Parse one ``chirp-*.h5``."""
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as fh:
        missing = {"chirp_time", "chirp_rate", "f0"} - set(fh.keys())
        if missing:
            raise ValueError(
                f"{path}: not a chirpsounder2 detection file, missing "
                f"{sorted(missing)}; found {sorted(fh.keys())}"
            )
        return Detection(
            path=path,
            channel=_decode(fh["channel"][()]) if "channel" in fh else "",
            chirp_time=float(_scalar(fh["chirp_time"][()])),
            rate=float(_scalar(fh["chirp_rate"][()])),
            f0=float(_scalar(fh["f0"][()])),
            snr=float(_scalar(fh["snr"][()])) if "snr" in fh else float("nan"),
            i0=int(_scalar(fh["i0"][()])) if "i0" in fh else -1,
            n_samples=int(_scalar(fh["n_samples"][()])) if "n_samples" in fh else -1,
            sample_rate=(float(_scalar(fh["sample_rate"][()]))
                         if "sample_rate" in fh else float("nan")),
        )


def read_timing(path: str | Path) -> TimingSolution:
    """Parse one ``par-*.h5``."""
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as fh:
        missing = {"t0", "chirp_rate"} - set(fh.keys())
        if missing:
            raise ValueError(
                f"{path}: not a chirpsounder2 timing file, missing "
                f"{sorted(missing)}; found {sorted(fh.keys())}"
            )
        t0s = np.asarray(fh["t0s"][()], dtype=np.float64) if "t0s" in fh else np.empty(0)
        f0s = np.asarray(fh["f0"][()], dtype=np.float64) if "f0" in fh else np.empty(0)
        snrs = np.asarray(fh["snrs"][()], dtype=np.float64) if "snrs" in fh else np.empty(0)
        return TimingSolution(
            path=path,
            channel=_decode(fh["channel"][()]) if "channel" in fh else "",
            t0=float(_scalar(fh["t0"][()])),
            rate=float(_scalar(fh["chirp_rate"][()])),
            t0s=t0s,
            f0s=f0s,
            snrs=snrs,
            num_detections=(int(_scalar(fh["num_detections"][()]))
                            if "num_detections" in fh else int(t0s.size)),
            software_version=str(fh.attrs.get("chirpsounder2_version", "")),
            git_commit=str(fh.attrs.get("git_commit", "")),
            git_dirty=(bool(fh.attrs["git_dirty"])
                       if "git_dirty" in fh.attrs else None),
        )


#: Column order of ``cdetections-*.h5``'s ``data``, from
#: ``detections2metadata.py:109``. The second column is ``i0 / 25e6`` with the
#: rate hard-coded upstream, so it is a sample offset in seconds *assuming*
#: 25 MS/s and not to be trusted as a time on a station sampling at anything
#: else. Only ``chirp_time`` is a real timestamp.
CDETECTION_COLUMNS = ("chirp_time", "i0_seconds", "f0", "chirp_rate", "snr")


def read_cdetections(path: str | Path) -> np.ndarray:
    """Parse one ``cdetections-*.h5`` into its raw ``(N, 5)`` array.

    Returned as-is rather than as :class:`Detection` records: this file is the
    cheap bulk sweep, and building objects per row defeats the reason to read
    it instead of the individual ``chirp-*.h5``. Columns are
    :data:`CDETECTION_COLUMNS`.
    """
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as fh:
        if "data" not in fh:
            raise ValueError(
                f"{path}: not a chirpsounder2 cdetections file, no 'data'; "
                f"found {sorted(fh.keys())}"
            )
        data = np.asarray(fh["data"][()], dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != len(CDETECTION_COLUMNS):
        raise ValueError(
            f"{path}: data is {data.shape}, expected (N, {len(CDETECTION_COLUMNS)})"
        )
    return data


def load_detections(target) -> list[Detection]:
    """Every ``chirp-*.h5`` under ``target``, unreadable files skipped.

    Skipping rather than raising: these files are written continuously by a
    running detector, so a sweep across a live tree will occasionally catch
    one mid-write. One truncated file must not lose the census.
    """
    out: list[Detection] = []
    bad = 0
    for path in find_detections(target):
        try:
            out.append(read_detection(path))
        except Exception:
            bad += 1
    if bad:
        warnings.warn(f"skipped {bad} unreadable detection file(s) under {target}",
                      stacklevel=2)
    return out


def load_cdetections(target) -> list[Detection]:
    """Every ``cdetections-*.h5`` under ``target``, as :class:`Detection`.

    The consolidated file is one ``(N, 5)`` array rather than a group of named
    datasets, so this is the same information as the individual ``chirp-*.h5``
    minus the fields that array has no columns for -- ``channel``, ``i0``,
    ``n_samples`` and ``sample_rate`` come back empty, ``-1`` and NaN, matching
    :func:`read_detection`'s convention for a field the file does not carry.
    Nothing in :func:`census` reads them.

    Worth having because this is what survives on the archive volume: the
    per-detection ``chirp-*.h5`` are written into the ringbuffer's tree and
    rotate away, while ``cdetections-*.h5`` is the 900 s consolidation that
    gets kept. A census on a synced archive has only these.
    """
    out: list[Detection] = []
    bad = 0
    for path in find_cdetections(target):
        try:
            rows = read_cdetections(path)
        except Exception:
            bad += 1
            continue
        for chirp_time, _i0_seconds, f0, rate, snr in rows:
            out.append(Detection(
                path=path, channel="", chirp_time=float(chirp_time),
                rate=float(rate), f0=float(f0), snr=float(snr),
                i0=-1, n_samples=-1, sample_rate=float("nan"),
            ))
    if bad:
        warnings.warn(f"skipped {bad} unreadable cdetections file(s) under {target}",
                      stacklevel=2)
    return out


def load_timings(target) -> list[TimingSolution]:
    """Every ``par-*.h5`` under ``target``, unreadable files skipped."""
    out: list[TimingSolution] = []
    bad = 0
    for path in find_timings(target):
        try:
            out.append(read_timing(path))
        except Exception:
            bad += 1
    if bad:
        warnings.warn(f"skipped {bad} unreadable timing file(s) under {target}",
                      stacklevel=2)
    return out


# --------------------------------------------------------------------------
# Census
# --------------------------------------------------------------------------

def _times_rates_snrs(items: Sequence) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times, rates, snrs = [], [], []
    for item in items:
        if isinstance(item, Detection):
            times.append(item.chirp_time)
            rates.append(item.rate)
            snrs.append(item.snr)
        elif isinstance(item, TimingSolution):
            times.append(item.t0)
            rates.append(item.rate)
            snrs.append(float(np.median(item.snrs)) if item.snrs.size else float("nan"))
        else:
            raise TypeError(f"census wants Detection or TimingSolution, got {type(item)}")
    return (np.asarray(times, dtype=np.float64),
            np.asarray(rates, dtype=np.float64),
            np.asarray(snrs, dtype=np.float64))


def census(items: Iterable,
           cycle_s: float = DEFAULT_CYCLE_S,
           tolerance_s: float = PHASE_TOLERANCE_S,
           min_count: int = 3) -> list[Emitter]:
    """Group detections or timing solutions into transmitters.

    Grouped by chirp rate, then by arrival phase within ``tolerance_s``. The
    phase does the work: two soundings from one transmitter arrive at the same
    fraction of a second whatever their schedule, and two transmitters at
    different ranges separate even when they share a slot.

    ``min_count`` exists because a search-mode tree is mostly noise. At DOB
    928 raw detections contained six repeating emitters and roughly forty
    one-off false alarms scattered uniformly across the second -- which is what
    a false alarm looks like, since nothing constrains its phase. Anything
    seen once carries no schedule and belongs in the raw detections, not in a
    census. Pass ``min_count=1`` to see them anyway.

    Returns emitters sorted by descending count.
    """
    items = list(items)
    if not items:
        return []
    times, rates, snrs = _times_rates_snrs(items)

    emitters: list[Emitter] = []
    for rate in sorted(set(rates.tolist())):
        on_rate = rates == rate
        t_rate, snr_rate = times[on_rate], snrs[on_rate]

        # Cluster on phase, wrapping the 0/1 boundary so an emitter sitting on
        # it stays one emitter.
        phase = t_rate % 1.0
        order = np.argsort(phase)
        groups: list[list[int]] = []
        current: list[int] = []
        for idx in order:
            if current and phase[idx] - phase[current[-1]] > tolerance_s:
                groups.append(current)
                current = []
            current.append(int(idx))
        if current:
            groups.append(current)
        if (len(groups) > 1
                and (phase[groups[0][0]] + 1.0 - phase[groups[-1][-1]]) <= tolerance_s):
            groups[0] = groups[-1] + groups[0]
            groups.pop()

        for group in groups:
            if len(group) < min_count:
                continue
            member_times = t_rate[group]
            fractions = _unwrap(member_times % 1.0)
            slots = sorted({int(round(float(t) % cycle_s)) % int(cycle_s)
                            for t in member_times})
            emitters.append(Emitter(
                rate=float(rate),
                fraction_s=float(np.mean(fractions)) % 1.0,
                fraction_sd_s=float(np.std(fractions)),
                count=len(group),
                observed_seconds=tuple(slots),
                cycle_s=float(cycle_s),
                first_seen=float(member_times.min()),
                last_seen=float(member_times.max()),
                snr_median=float(np.nanmedian(snr_rate[group])),
            ))

    return sorted(emitters, key=lambda e: (-e.count, e.rate, e.fraction_s))


def solve_epoch_offset(items: Iterable,
                       rate: float,
                       transmit_seconds: Sequence[int],
                       distance_km: float,
                       cycle_s: float = DEFAULT_CYCLE_S,
                       reference: str = "reference transmitter",
                       window_s: float = 1.5) -> EpochOffset:
    """Measure this receiver's clock against a transmitter you already know.

    Needs three things the files cannot supply: the transmitter's chirp rate,
    the whole seconds it starts on, and how far away it is. The published
    schedule is the part that matters -- an epoch error larger than half a
    second cannot be recovered from the data alone, because there is no way to
    tell a late arrival from the next slot. At DOB the error was 0.956 s, so
    every attempt to solve it from the archive alone had it a whole second out
    and looked self-consistent.

    ``t_measured = S + tau + offset`` for each detection, with ``S`` the
    published second nearest in the cycle and ``tau = distance / c``. The
    scatter across independent slots is the check: agreement to well under a
    millisecond on slots hours apart means one clock offset, not a fit.

    Raises when no detection lands within ``window_s`` of any published slot,
    which is the honest outcome of naming the wrong transmitter.
    """
    items = list(items)
    times, rates, _ = _times_rates_snrs(items)
    on_rate = np.isclose(rates, rate)
    if not on_rate.any():
        raise ValueError(
            f"no {rate / 1e3:.0f} kHz/s records among {len(items)}; "
            f"rates present: {sorted(set(rates.tolist()))}"
        )
    times = times[on_rate]
    tau = distance_km * 1e3 / C_M_S

    per_slot: dict[int, list[float]] = {}
    for t in times:
        in_cycle = float(t) % cycle_s
        best, best_gap = None, None
        for slot in transmit_seconds:
            gap = (in_cycle - slot + cycle_s / 2.0) % cycle_s - cycle_s / 2.0
            if best_gap is None or abs(gap) < abs(best_gap):
                best, best_gap = slot, gap
        if best is not None and abs(best_gap) <= window_s:
            per_slot.setdefault(int(best), []).append(best_gap - tau)

    if not per_slot:
        raise ValueError(
            f"no {rate / 1e3:.0f} kHz/s record lands within {window_s} s of "
            f"{list(transmit_seconds)} on a {cycle_s:.0f} s cycle -- either "
            f"{reference} was not transmitting, or this is a different emitter"
        )

    slot_means = [float(np.mean(v)) for v in per_slot.values()]
    weights = [len(v) for v in per_slot.values()]
    offset = float(np.average(slot_means, weights=weights))
    spread = (float(statistics.stdev(slot_means)) if len(slot_means) > 1
              else float(np.std(next(iter(per_slot.values())))))
    return EpochOffset(
        seconds=offset,
        residual_sd_s=spread,
        n_slots=len(per_slot),
        n_samples=int(sum(weights)),
        reference=reference,
    )


def describe(target, cycle_s: float = DEFAULT_CYCLE_S) -> str:
    """A human-readable census of a detection tree, for the CLI and for logs."""
    timings = load_timings(target)
    detections = load_detections(target) if not timings else []
    source = timings or detections
    emitters = census(source, cycle_s=cycle_s)
    kind = "timing solution" if timings else "detection"

    lines = [f"{len(source)} {kind}(s), {len(emitters)} repeating emitter(s)",
             f"{'rate':>9} {'received at':>24} {'n':>5} {'phase ms':>9} "
             f"{'sd ms':>7} {'snr':>8} {'span h':>7}"]
    for e in emitters:
        slots = ",".join(str(s) for s in e.observed_seconds)
        if len(slots) > 22:
            slots = slots[:19] + "..."
        lines.append(
            f"{e.rate / 1e3:8.0f}k {f'{e.cycle_s:.0f}:{slots}':>24} {e.count:5d} "
            f"{e.fraction_s * 1e3:9.2f} {e.fraction_sd_s * 1e3:7.2f} "
            f"{e.snr_median:8.1f} {e.span_hours:7.1f}"
        )
    lines.append("")
    lines.append("Seconds are AS RECEIVED and phase is seconds past the whole "
                 "second -- neither is a transmit time or a range until the")
    lines.append("receiver's epoch offset is known. Measure it with "
                 "solve_epoch_offset() against a transmitter of known position,")
    lines.append("then Emitter.transmit_seconds(offset) and "
                 "Emitter.range_km(offset) mean what they say.")
    return "\n".join(lines)
