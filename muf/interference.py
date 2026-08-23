"""Rejecting interference stripes before the estimators see them.

A burst of interference lands in one spectrogram row -- one moment in time,
which on a chirp sweep is one radio frequency -- and lights up the *whole*
range axis, because a signal that is not a chirp at this rate has no delay and
smears across every beat frequency. An echo does the opposite: it is narrow in
range and continuous in frequency.

That asymmetry is the whole method. This module measures how much *range* each
frequency row has above the detection threshold and rejects the rows where the
answer is physically impossible for an echo.

**Why this exists rather than whitening at the station.** chirpsounder2's
ionogram path does not whiten -- ``calc_ionograms.spectrogram`` combines its
13 oversampled sub-steps by inverse-variance weighting, which weights *time*
sub-steps and does not flatten across frequency. The v1 console does whiten,
and the difference shows: over sampled archives, 9 of 38 v2 soundings had more
than half their 43 dB crossings inside burst rows, against 0 of 9 v1
soundings. Adding whitening to v2 would be an irreversible acquisition-side
change to a station that cannot be A/B tested, because the ringbuffer voltage
lives two minutes. Rejecting the stripes here is reversible, testable against
the whole archive, and needs nothing from the station.

**It does not do anything about the frequency it removes.** A rejected row is
set to the equalized noise floor, which asserts "nothing detectable at this
frequency" -- not "no echo here". The burst may well have been sitting on top
of a real echo, and this cannot recover it. Losing a frequency is the honest
outcome; keeping a MUF picked off a burst is not.

**Measured yield, so nobody expects more of it than it delivers.** Over three
archives:

===============  =====  ==================  ================  =======
archive          n      with burst rows     picks on those    changed
===============  =====  ==================  ================  =======
2026-08-05 v2     54     11                 11 -> 10          1
2026-08-04 v2     48      5                 10 -> 10          0
2026-02-04 v1     10      0                  0 ->  0          0
===============  =====  ==================  ================  =======

So roughly one sounding in five carries a row that cannot be an echo, and
removing it changes a MUF about once in twenty. That is why the flag is off by
default and why ``find`` is the more useful half of this module: it names the
contaminated frequencies, which is a diagnostic worth having even when
suppression changes nothing.

It is emphatically *not* a fix for false picks on a quiet day. On 2026-08-04 --
established elsewhere as a poor day -- the estimators produced 84 picks across
48 soundings and suppression removed none of them. Whatever is generating
those is not burst interference, and this module should not be credited with
addressing it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .extractors.contour import DEFAULT_THRESHOLD_DB
from .spectro import NOISE_COEF

#: Occupied range within one frequency row, in km, above which the row cannot
#: be an echo.
#:
#: Measured rather than chosen. Over the 2026-08-05 v2 archive and the
#: 2026-02-04 v1 archive, the range a single frequency row occupies above
#: 43 dB runs:
#:
#: ===========  ======  ======  =======
#: archive      q50     q90     q99
#: ===========  ======  ======  =======
#: v2 ``.h5``    16 km  146 km   437 km
#: v1 ``.lfs``   44 km   88 km   161 km
#: ===========  ======  ======  =======
#:
#: and then jumps straight to the full axis -- 7998 km and 2666 km. 800 km is
#: 1.8x the worse of the two 99th percentiles, so spread-F and a multi-hop
#: family both survive comfortably.
#:
#: *Occupied* range, not the span from first to last hot bin: three narrow
#: multi-hop echoes 300 km apart occupy ~120 km while spanning 600 km, and it
#: is the occupancy that separates them from a burst.
MAX_ECHO_RANGE_KM = 800.0

#: Hot range bins a row needs before it can be called a burst at all.
#:
#: :data:`MAX_ECHO_RANGE_KM` is in km, so on a coarse range axis a mere handful
#: of bins can exceed it -- at 234 km per bin, four bins is 936 km and a
#: perfectly ordinary narrow echo would be rejected. Real products are nowhere
#: near that (v2 is 2 km per bin, `.lfs` at window 8192 is 14.6 km), but the
#: rule must not depend on that staying true. A burst covers thousands of bins;
#: eight is far below anything real and far above anything an echo produces on
#: an axis fine enough for the km rule to mean something.
MIN_BURST_BINS = 8

#: Linear power of a median-noise cell once `spectro` has equalized it. Each
#: row is divided by ``NOISE_COEF * median(row)``, so the median lands here --
#: 0.3607, which `to_db` renders as 25.571 dB. Suppressed rows are set to it
#: because that is exactly what "this cell was not detected" already reads as:
#: v2 sparsifies below its storage threshold and `io_chirp` fills those cells
#: with the row median, which is the same number.
EQUALIZED_NOISE_POWER = 1.0 / NOISE_COEF


@dataclass(frozen=True)
class Interference:
    """Which frequency rows are burst-contaminated, and by how much."""

    #: Boolean mask over frequency. True means reject.
    rows: np.ndarray
    #: Range occupied above the threshold, km, one entry per frequency.
    occupied_km: np.ndarray
    threshold_db: float
    max_echo_km: float

    @property
    def n_rows(self) -> int:
        return int(self.rows.sum())

    @property
    def any(self) -> bool:
        return bool(self.rows.any())

    @property
    def row_fraction(self) -> float:
        """Share of the sweep lost to rejection. The cost side of the trade."""
        return float(self.rows.mean()) if self.rows.size else 0.0

    def describe(self, freq_mhz=None) -> str:
        if not self.any:
            return "no interference rows"
        parts = [f"{self.n_rows} row(s), {self.row_fraction * 100:.1f}% of the "
                 f"sweep, occupying up to "
                 f"{self.occupied_km[self.rows].max():.0f} km of range"]
        if freq_mhz is not None and len(freq_mhz) == len(self.rows):
            hit = np.asarray(freq_mhz)[self.rows]
            shown = ", ".join(f"{f:.2f}" for f in hit[:6])
            if len(hit) > 6:
                shown += ", ..."
            parts.append(f"at {shown} MHz")
        return "; ".join(parts)


def find(ion, *, threshold_db: float = DEFAULT_THRESHOLD_DB,
         max_echo_km: float = MAX_ECHO_RANGE_KM) -> Interference:
    """Locate burst-contaminated frequency rows in one sounding.

    ``threshold_db`` should match what the estimators use, so this never
    rejects a row on evidence they would have ignored.
    """
    power_db = np.asarray(ion.db)
    vrange = np.asarray(ion.cal.vrange, dtype=np.float64)
    step = (abs(float(vrange[1] - vrange[0])) if vrange.size > 1 else 0.0)

    hot = power_db > threshold_db
    bins = hot.sum(axis=1)
    occupied = bins.astype(np.float64) * step

    # A degenerate axis -- one range bin, or a zero step -- makes the measure
    # meaningless, and a rule that cannot be evaluated must reject nothing.
    if step > 0:
        rows = (occupied > max_echo_km) & (bins >= MIN_BURST_BINS)
    else:
        rows = np.zeros(len(occupied), bool)
    return Interference(rows=rows, occupied_km=occupied,
                        threshold_db=float(threshold_db),
                        max_echo_km=float(max_echo_km))


def suppress(ion, found: Interference | None = None, *,
             threshold_db: float = DEFAULT_THRESHOLD_DB,
             max_echo_km: float = MAX_ECHO_RANGE_KM):
    """Return ``(ionogram, Interference)`` with contaminated rows flattened.

    The sounding is returned unchanged, not copied, when nothing is rejected:
    the common case should cost nothing, and an unmodified array is easier to
    reason about than a defensive copy that is never different.
    """
    import dataclasses

    found = found or find(ion, threshold_db=threshold_db,
                          max_echo_km=max_echo_km)
    if not found.any:
        return ion, found

    power = np.array(ion.power, copy=True)
    power[found.rows, :] = EQUALIZED_NOISE_POWER
    return dataclasses.replace(ion, power=power, _db=None), found


def apply(ion, options):
    """Suppression driven by :class:`~muf.pipeline.Options`, or a pass-through.

    One call site's worth of branching, in one place, so every command that
    runs estimators honours the flag identically instead of three of them
    remembering to.
    """
    if not getattr(options, "reject_interference", False):
        return ion, None
    return suppress(ion)
