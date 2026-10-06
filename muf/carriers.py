"""Rejecting carriers: features narrow in frequency with no trace around them.

An ionospheric echo is a *trace*. Its group range changes smoothly with
frequency, so whatever the trace does at one frequency it was nearly doing a
few hundred kHz either side. A MUF is the top of such a trace.

Some features break that rule. The one that made this module necessary was on
the Rostov circuit on 2026-09-28, 16:11-16:21 UTC: three consecutive sweeps
carrying a bar 0.25 MHz wide (16.85-17.05 MHz) and 500 km tall (1500-2000 km),
peaking at 74 dB, and *nothing else in the ionogram at all*. kmeans and contour
both reported MUF 17.05 MHz from it. The burst rule in
:mod:`muf.interference` does not see it: each row occupies 180-260 km of range,
well inside :data:`muf.interference.MAX_ECHO_RANGE_KM`, because what marks it
out is not how much range it covers but that it is *alone*.

**The rule.** Find rows whose above-threshold energy occupies a lot of range
(:data:`MIN_OCCUPIED_KM`), and split their energy into clusters in range. Grow
each cluster outwards in frequency, one row at a time, for as long as the next
row still has echo inside the cluster's range span. It is a carrier if

- the grown feature is at most :data:`MAX_WIDTH_MHZ` wide,
- within :data:`NEIGHBOURHOOD_MHZ` either side of it, at most
  :data:`MAX_NEIGHBOUR_ROWS` rows have any echo within :data:`RANGE_PAD_KM`
  of its span, and
- the rows *below* it were actually recorded.

The second condition is the one that matters, and the first version of this
rule did without it. v2 traces arrive *chopped*: the transmitters skip bands,
so between 8 and 12 MHz a perfectly good trace is a comb of 0.1-0.3 MHz
columns with empty rows between them. Each column grown on its own is as
narrow as a carrier -- 1,900 of 10,171 archived soundings had one -- and a
contact sheet of them was trace after trace. What separates the two is that a
column has its neighbours at the *same range* a few rows away, across the gap,
and a carrier has nothing there.

**Measured over the local archive** (10,171 v2 soundings, 2026-08-04 to
09-28), counting for each narrow feature the rows within 1 MHz that carry echo
within 50 km of its range -- an early, tighter form of the test:

==============================  ========
rows with echo at that range    features
==============================  ========
0                               702
1                               35
2-12                            ~150
13 or more                      ~2,280
==============================  ========

So the split is clean: the 737 at 0-1 were, on two contact sheets of 48, every
one a stripe (9.9, 27.5 and 29.8 MHz are regulars), a 30 MHz sweep-end
transient, or a compact splash where a transmitter stops at 20 MHz with no
trace below it. None was a piece of trace. The final test is looser on both
axes, which only ever keeps more: 1.5 MHz, because a few compact blobs at the
very bottom of a sweep sit 1.2 MHz from where their trace resumes; and 500 km,
because a blob at a trace's low end 200-400 km above it -- a second mode,
plausibly -- was otherwise removed.

**What the final rule changes**, same archive, with and without it:

===========  ========  =======  ==================  ==================
receiver     products  flagged  contour MUF         contour LOF
===========  ========  =======  ==================  ==================
Yoshkar-Ola  6,939     372      4 (3 gone, 1 down)  4 (3 gone, 1 up)
DOB          3,232     148      35 (29 gone, 6 down) 30 (all gone)
===========  ========  =======  ==================  ==================

Every one of those 39 contour MUFs was looked at, and every one had been read
off the top edge of a lone blob: Rostov's 17.05 three times; SGO -> DOB at
21.395 MHz for fourteen consecutive sweeps and at 14.447 for seven; 16.396 for
six; a 20 MHz sweep-end splash read as 19.997. Most flagged products change
nothing, because a one-row stripe never makes the five-row run a MUF needs.
`algo` never changed. kmeans moved in 52, by a median 0.05 MHz: it clusters
the whole ionogram, so removing a stripe at 30 MHz nudges it anywhere.

The 10 MHz blob that recurs in scheduled NIC products, always near 3000 km,
is most likely RWM (Moscow, 9.996 MHz): its pulses are locked to the UTC
second, as the schedule is, so it lands at one fixed range per slot. It moved
one LOF, from 9.9 MHz to 12.9 where that sounding's trace begins.

**Absence has to be observed.** A v2 product analysed late is missing the
bottom of its sweep (``Calibration.freq_lo``): those rows exist but hold one
constant value. A feature at the first recorded row has no trace below it
because nothing below it was recorded, which is not the same thing -- one such
case on 2026-09-26 was a trace's top 0.3 MHz, and the first version of this
rule removed it. So at least half of the neighbourhood below must be recorded
(:data:`MIN_OBSERVED_BELOW`). The same keeps anything at the very start of a
sweep. Above, the sweep simply ending is fine: a MUF is approached from below,
so it is the rows below that say whether something is a trace.

**What it does not do.** A carrier that lands *on* a trace's range, a few
rows above where the trace really ends, has neighbours at its range by
construction and is kept. That case cannot be told apart from a trace in one
sounding. And like :mod:`muf.interference`, flattening asserts only "nothing
usable here", not "no echo here".

Unlike the burst rule this is on by default: it was written against a MUF that
was wrong, and the measurement above is the evidence that it removes nothing
else. ``Options(reject_carriers=False)`` turns it off.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from .extractors.contour import DEFAULT_THRESHOLD_DB
from .spectro import NOISE_COEF

#: Range a row's above-threshold energy must occupy, km, to seed a feature.
#: An echo row is typically 16 km (the v2 median; see
#: :data:`muf.interference.MAX_ECHO_RANGE_KM`), the Rostov carrier's rows
#: 180-260 km. Set low because the isolation test, not this, does the
#: separating -- this only keeps a lone noise pixel from seeding anything.
MIN_OCCUPIED_KM = 60.0

#: Hot range bins a seed row needs as well, so a coarse range axis cannot
#: pass :data:`MIN_OCCUPIED_KM` on two or three bins. Same reasoning as
#: :data:`muf.interference.MIN_BURST_BINS`.
MIN_SEED_BINS = 8

#: Hot bins inside the feature's range span that count a row as "echo here",
#: both for growing the feature and for looking for a trace beside it. One is
#: a noise pixel; two in the same few hundred km is not, at 43 dB.
TRACE_BINS = 2

#: Widest a feature can be and still be a carrier. The Rostov one is 0.25.
MAX_WIDTH_MHZ = 0.3

#: How far either side of a feature to look for the trace it would belong to.
NEIGHBOURHOOD_MHZ = 1.5

#: Rows in that neighbourhood allowed to show echo at the feature's range
#: before it counts as part of a trace. One, so a single stray row of noise
#: cannot save a carrier; the measured split puts traces at 13 or more.
MAX_NEIGHBOUR_ROWS = 1

#: Share of the neighbourhood *below* a feature that must have been recorded
#: for its emptiness to count. See "Absence has to be observed" above.
MIN_OBSERVED_BELOW = 0.5

#: Padding on the feature's range span, km, for the neighbourhood search: any
#: echo this close in range counts as the trace the feature may belong to.
#: Wide, because an ionogram's modes at one frequency spread over a few
#: hundred km; at 100 km, blobs at a trace's low end 200-400 km above it were
#: removed. No MUF the rule fixes moved when this went from 100 to 500.
RANGE_PAD_KM = 500.0

#: Gap in range, km, that splits the hot cells of a feature's rows into
#: separate clusters, each tested on its own. The Rostov carrier is 57% filled
#: with gaps of a few bins; a trace crossing the same rows at another range is
#: a different cluster, and so are the two or three blobs a transmitter
#: stopping at 20 MHz can leave at different ranges in one search product.
CLUSTER_GAP_KM = 30.0

#: Linear power of an equalized median-noise cell. What a flattened cell is set
#: to; see :data:`muf.interference.EQUALIZED_NOISE_POWER`, which is the same.
EQUALIZED_NOISE_POWER = 1.0 / NOISE_COEF


@dataclass(frozen=True)
class Carrier:
    """One rejected feature: the box of cells that was flattened."""

    freq_mhz: tuple[float, float]
    range_km: tuple[float, float]
    #: Inclusive row and column bounds into the ionogram.
    rows: tuple[int, int]
    cols: tuple[int, int]


@dataclass(frozen=True)
class Carriers:
    found: tuple[Carrier, ...] = ()

    @property
    def any(self) -> bool:
        return bool(self.found)

    @property
    def n(self) -> int:
        return len(self.found)

    def describe(self) -> str:
        if not self.found:
            return "no carriers"
        return "carrier(s) at " + ", ".join(
            f"{c.freq_mhz[0]:.2f}-{c.freq_mhz[1]:.2f} MHz "
            f"{min(c.range_km):.0f}-{max(c.range_km):.0f} km"
            for c in self.found)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive ``(start, stop)`` of each run of True."""
    idx = np.flatnonzero(mask)
    if not idx.size:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = idx[np.r_[0, breaks + 1]]
    stops = idx[np.r_[breaks, idx.size - 1]]
    return [(int(a), int(b)) for a, b in zip(starts, stops)]


def _clusters(mask: np.ndarray, gap: int) -> list[tuple[int, int]]:
    """Inclusive bounds of the runs of True, joining runs split by ``gap`` or
    fewer False."""
    idx = np.flatnonzero(mask)
    if not idx.size:
        return []
    breaks = np.flatnonzero(np.diff(idx) > gap + 1)
    starts = idx[np.r_[0, breaks + 1]]
    stops = idx[np.r_[breaks, idx.size - 1]]
    return [(int(a), int(b)) for a, b in zip(starts, stops)]


def find(ion, *, threshold_db: float = DEFAULT_THRESHOLD_DB) -> Carriers:
    """Locate carriers in one sounding. See the module docstring for the rule."""
    power_db = np.asarray(ion.db)
    freq = np.asarray(ion.cal.freq, dtype=np.float64)
    vrange = np.asarray(ion.cal.vrange, dtype=np.float64)
    n_freq, n_range = power_db.shape
    if n_freq < 2 or n_range < 2:
        return Carriers()
    step_km = abs(float(vrange[1] - vrange[0]))
    step_mhz = abs(float(np.median(np.diff(freq))))
    if step_km <= 0 or step_mhz <= 0:
        return Carriers()

    hot = power_db > threshold_db
    # A row never recorded is one constant value -- see `Calibration.freq_lo`.
    recorded = power_db.max(axis=1) > power_db.min(axis=1)
    bins = hot.sum(axis=1)
    seeds = (bins * step_km >= MIN_OCCUPIED_KM) & (bins >= MIN_SEED_BINS)
    max_rows = max(1, round(MAX_WIDTH_MHZ / step_mhz))
    reach = max(1, round(NEIGHBOURHOOD_MHZ / step_mhz))
    pad = round(RANGE_PAD_KM / step_km)
    gap = max(1, round(CLUSTER_GAP_KM / step_km))

    found: list[Carrier] = []
    for a, b in _runs(seeds):
        rows = hot[a:b + 1]
        # Each cluster of these rows' energy is its own candidate. Taking the
        # whole row's span instead let a trace crossing elsewhere -- or one
        # noise pixel -- stretch it until the trace counted as a neighbour.
        for c0, c1 in _clusters(rows.any(axis=0), gap):
            own = rows[:, c0:c1 + 1].sum(axis=1)
            if not ((own * step_km >= MIN_OCCUPIED_KM)
                    & (own >= MIN_SEED_BINS)).any():
                continue
            carrier = _isolated(hot, recorded, a, b, c0, c1, max_rows=max_rows,
                                reach=reach, pad=pad)
            if carrier is not None:
                lo, hi = carrier
                found.append(Carrier(
                    freq_mhz=(float(freq[lo]), float(freq[hi])),
                    range_km=(float(vrange[c0]), float(vrange[c1])),
                    rows=(lo, hi), cols=(c0, c1)))
    return Carriers(tuple(found))


def _isolated(hot, recorded, a, b, c0, c1, *, max_rows, reach, pad):
    """Grow rows ``a..b`` within columns ``c0..c1`` and test the result.

    Returns the grown ``(lo, hi)`` rows if it is a carrier, else ``None``.
    """
    n_freq, n_range = hot.shape
    in_span = hot[:, c0:c1 + 1].sum(axis=1) >= TRACE_BINS
    lo, hi = a, b
    while lo > 0 and in_span[lo - 1] and hi - lo < max_rows:
        lo -= 1
    while hi < n_freq - 1 and in_span[hi + 1] and hi - lo < max_rows:
        hi += 1
    if hi - lo + 1 > max_rows:
        return None

    below = slice(max(0, lo - reach), lo)
    if recorded[below].sum() < MIN_OBSERVED_BELOW * reach:
        return None
    w0, w1 = max(0, c0 - pad), min(n_range - 1, c1 + pad)
    near = hot[:, w0:w1 + 1].sum(axis=1) >= TRACE_BINS
    beside = np.r_[near[below], near[hi + 1:hi + 1 + reach]]
    if int(beside.sum()) > MAX_NEIGHBOUR_ROWS:
        return None
    return lo, hi


def suppress(ion, found: Carriers | None = None, *,
             threshold_db: float = DEFAULT_THRESHOLD_DB):
    """Return ``(ionogram, Carriers)`` with each carrier's cells flattened.

    Only the carrier's own box is flattened, not its whole rows: an echo at
    another range in the same rows is evidence this has no reason to discard.
    Unchanged, not copied, when nothing is found.
    """
    found = found if found is not None else find(ion, threshold_db=threshold_db)
    if not found.any:
        return ion, found
    power = np.array(ion.power, copy=True)
    for c in found.found:
        power[c.rows[0]:c.rows[1] + 1, c.cols[0]:c.cols[1] + 1] = \
            EQUALIZED_NOISE_POWER
    return dataclasses.replace(ion, power=power, _db=None), found
