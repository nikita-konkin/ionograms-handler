"""The low-frequency end of the trace: LOF, and an observed LUF.

The MUF is where the ionosphere stops returning the signal. The low end is
where *absorption* stops it getting through: below roughly 10 MHz the
non-deviative loss in the D region rises steeply, and the trace fades out. That
edge carries real information -- it tracks solar illumination directly -- but
naming it correctly matters more here than anywhere else in this package.

**LOF, not LUF.** Recommendation ITU-R P.533-13 section 9 defines the lowest
usable frequency as

    the lowest frequency, expressed to the nearest 0.1 MHz, at which a required
    signal-to-noise ratio is achieved by the monthly median signal-to-noise.

Two things there are not measurements of this sounding. A *required* S/N is a
property of a modulation and a service, not of the ionosphere; and a *monthly
median* is a statistic over many soundings. P.533's field-strength chain makes
the dependence explicit -- equation (17) is
``E_w = 136.6 + P_t + G_t + 20 log f - L_b``, carrying transmitter power and
antenna gain, neither of which this instrument knows.

What an oblique ionosonde scales is the **lowest observed frequency**. The URSI
INAG reference on oblique sounding with signal levels (Blagov, UAG-104) names
exactly three parameters: "the maximum observed frequency (MOF), the maximum
useable frequency (MUF), the lowest observed frequency (LOF)". Its
vertical-incidence counterpart is ``fmin``, long used as a qualitative proxy for
D-region absorption. So this module reports LOF, and reports an observed LUF
only in the one form the data supports (see :func:`luf_at_snr`).

**LOF is threshold-relative, and says so.** Every LOF is the lowest frequency
carrying a continuous run of echoes *above some level*, and moving the level
moves the answer -- on 2026-02-04 at 12:00 UTC, 13.84 MHz at 43 dB, 23.10 at 50,
28.83 at 57. This is the century-old criticism of ``fmin`` as an equipment
parameter, and the honest response is not to pick one threshold and hide it but
to carry it in the result and report a ladder of them
(:func:`ladder`). The spread across the ladder is what steep and shallow
absorption look like.

**It works.** Over 45 soundings of 2026-02-04, LOF at 43 dB correlates with the
cosine of the solar zenith angle at the path midpoint at **r = +0.864**: around
16 MHz through the day, pinned at the band floor at night. That is the D-region
signature, and it is the evidence that the number means something.

**The band floor is not the sweep start.** These recordings declare a sweep from
7.5 MHz, but below 8.0 MHz the peak in the range gate is 34-38 dB at every hour,
day and night, with no diurnal variation at all -- a flat noise floor, so a
transmitter or filter edge rather than absorption, which would move with the
sun. Night-time LOF therefore pins at 8.0 and the true value is below the
sounded band. That is censoring, exactly like the MUF band-edge problem in
reverse, and it earns the URSI qualifying letter ``E``. Because the floor is a
property of the circuit and not of the sweep header, it has to be supplied:
``band_floor_mhz`` defaults to the sweep start, and :func:`measure_band_floor`
recovers it from a day of recordings.

**And the MUF band-edge problem itself is here too**, in
:func:`measure_band_ceiling`, because it is the same measurement read off the
other end of :func:`presence_above`. The censoring at the top had the same
defect for longer: ``pipeline.BAND_EDGE_BINS`` was anchored on the sweep's
declared stop rather than on the frequency the circuit actually returns, so on
a path whose usable top is below the sweep's, nothing was ever flagged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import spectro
from .pick import DEFAULT_MIN_RUN, _echo_range, find_runs

#: Detection levels for :func:`ladder`, in dB over the equalized noise floor.
#:
#: 43 dB is the level the ``algo`` and ``contour`` estimators already share
#: (``10*log10(20/1e-3)``), so the first rung is comparable with everything else
#: this package reports. The others are +7 and +14 dB: far enough apart that the
#: rungs separate on real soundings, close enough that the top one still finds a
#: trace through the day.
DEFAULT_LADDER = (43.0, 50.0, 57.0)

#: A LOF within this many frequency bins of the band floor means the trace ran
#: off the *bottom* of the band: the true LOF is at or below the lowest usable
#: sounded frequency. Mirrors ``pipeline.BAND_EDGE_BINS`` and is counted in bins
#: for the same reason.
BAND_FLOOR_BINS = 3


@dataclass(frozen=True)
class LofPick:
    """Outcome of one LOF decision. Deliberately parallel to ``MufPick``."""

    lof_mhz: float          # NaN when nothing qualified
    vrange_km: float        # NaN when nothing qualified
    threshold_db: float     # the level this LOF is relative to; never implicit
    n_detections: int       # frequency bins with trace presence
    run_len: int            # length of the run the pick sits in
    snr_db: float           # peak power at the picked bin, dB over the noise
    freq_index: int         # -1 when nothing qualified
    at_band_floor: bool = False

    @property
    def ok(self) -> bool:
        return self.freq_index >= 0

    def __str__(self) -> str:
        if not self.ok:
            return f"LOF: no detection ({self.n_detections} bins)"
        bound = " (at the band floor: upper bound)" if self.at_band_floor else ""
        return (f"LOF {self.lof_mhz:.2f} MHz at {self.threshold_db:.0f} dB"
                f"{bound}")


NO_PICK = LofPick(np.nan, np.nan, np.nan, 0, 0, np.nan, -1)


def pick_lof(
    presence: np.ndarray,
    freq: np.ndarray,
    power_db: np.ndarray | None = None,
    vrange: np.ndarray | None = None,
    min_run: int = DEFAULT_MIN_RUN,
    threshold_db: float = float("nan"),
    band_floor_mhz: float | None = None,
    parabolic: bool = True,
) -> LofPick:
    """Pick the LOF: the bottom of the lowest qualifying run of echoes.

    The mirror of :func:`muf.pick.pick_muf` -- same continuity rule, same
    sub-bin range interpolation -- reading the first bin of the first run
    instead of the last bin of the last.

    There is no ``percentile`` counterpart. Trimming the top of the MUF
    distribution guards against interference spikes *beyond* the trace; at the
    low end the equivalent bins are inside the trace, so trimming there would
    discard signal rather than noise. The continuity rule is the whole defence.

    Args:
        presence: bool per frequency bin -- trace detected there.
        freq: frequency axis in MHz, ascending.
        power_db: optional ``[n_freq, n_range]`` for the echo's range and level.
        vrange: virtual-range axis matching ``power_db``'s columns.
        min_run: consecutive detected bins required to accept a run.
        threshold_db: the level ``presence`` was derived from, carried into the
            result. A LOF without its threshold is not reproducible.
        band_floor_mhz: lowest frequency the circuit actually radiates. Defaults
            to ``freq[0]``; see the module docstring for why that default is
            usually wrong.
        parabolic: interpolate the echo's range between bins.
    """
    presence = np.asarray(presence, dtype=bool)
    if presence.shape != freq.shape:
        raise ValueError(f"presence {presence.shape} does not match freq {freq.shape}")

    n_detections = int(presence.sum())
    runs = find_runs(presence, min_run)
    if not runs:
        return LofPick(np.nan, np.nan, threshold_db, n_detections, 0, np.nan, -1)

    start, stop = runs[0]
    index = start

    vrange_km, snr_db = np.nan, np.nan
    if power_db is not None and vrange is not None and len(vrange):
        vrange_km, snr_db = _echo_range(power_db[index], vrange, parabolic)

    floor = float(freq[0]) if band_floor_mhz is None else float(band_floor_mhz)
    step = float(freq[1] - freq[0]) if freq.size > 1 else 0.0
    return LofPick(
        lof_mhz=float(freq[index]),
        vrange_km=vrange_km,
        threshold_db=threshold_db,
        n_detections=n_detections,
        run_len=int(stop - start + 1),
        snr_db=snr_db,
        freq_index=index,
        at_band_floor=bool(float(freq[index]) <= floor + BAND_FLOOR_BINS * step),
    )


def presence_above(ion, threshold_db: float) -> np.ndarray:
    """Frequency bins holding any echo above ``threshold_db`` inside the gate.

    Estimator-independent by design: a LOF derived this way is a property of the
    ionogram and its threshold, so it stays comparable when the estimators
    change, which the MUF numbers do not.
    """
    return (ion.db >= threshold_db).any(axis=1)


def lof_at(ion, threshold_db: float, min_run: int = DEFAULT_MIN_RUN,
           band_floor_mhz: float | None = None) -> LofPick:
    """LOF for one detection level, straight off the ionogram."""
    return pick_lof(
        presence_above(ion, threshold_db), ion.freq,
        power_db=ion.db, vrange=ion.vrange, min_run=min_run,
        threshold_db=threshold_db, band_floor_mhz=band_floor_mhz,
    )


def ladder(ion, thresholds=DEFAULT_LADDER, min_run: int = DEFAULT_MIN_RUN,
           band_floor_mhz: float | None = None) -> dict[float, LofPick]:
    """LOF at several detection levels.

    One LOF hides the fact that it is a threshold crossing; three make it
    visible and their spread measures how steeply the signal rolls off into the
    absorption. Cheap: the ionogram is already in memory and each rung is one
    comparison and one run scan.
    """
    return {
        float(level): lof_at(ion, float(level), min_run, band_floor_mhz)
        for level in thresholds
    }


#: Below this true signal-to-noise the metric is saturated: the peak in the
#: range gate of pure noise already reads 4-8 dB (the largest of ~205
#: exponentially distributed samples), so a requirement under it is met
#: everywhere and :func:`luf_at_snr` just returns the bottom of the sweep.
#: Measured on the sub-8 MHz dead band of 2026-02-04, at every hour.
MIN_MEANINGFUL_SNR_DB = 10.0


def luf_at_snr(ion, required_snr_db: float, min_run: int = DEFAULT_MIN_RUN,
               band_floor_mhz: float | None = None) -> LofPick:
    """The lowest frequency whose echo clears a *true* signal-to-noise.

    ITU-R P.533 section 9's definition applied to a measured rather than a
    predicted S/N, and the only form of LUF this instrument can support. Three
    things have to travel with the number:

    * ``required_snr_db`` is a **true power ratio**, not a raw ionogram dB
      value. ``spectro.to_db`` puts the equalized noise floor at
      ``NOISE_FLOOR_DB`` = 30 dB, so this adds the offset for you: asking for
      13 dB tests against the same 43 dB level the estimators use. Passing a
      raw 20 here without the conversion asks for -10 dB and matches noise,
      which is exactly the mistake this signature now prevents.
    * It is **this circuit and this receiver**. Median equalization makes each
      dB value a ratio to the noise floor in that frequency bin, so it is a
      genuine S/N -- of our antenna, our noise environment and our
      transmitter's power. Another link with a different budget has a
      different LUF at the same instant.
    * It is **one sounding**, where P.533 specifies a monthly median. Medians
      over a month remove that objection; they do not touch the previous one.

    Below :data:`MIN_MEANINGFUL_SNR_DB` the answer saturates at the bottom of
    the sweep, because noise itself clears the requirement.
    """
    return lof_at(ion, required_snr_db + spectro.NOISE_FLOOR_DB,
                  min_run, band_floor_mhz)


def measure_band_floor(ions, threshold_db: float = DEFAULT_LADDER[0],
                       quantile: float = 0.02) -> float:
    """Estimate the lowest frequency a circuit actually radiates.

    A hardware edge is constant while absorption swings through the day, so the
    floor is the *minimum* over many soundings of the lowest frequency showing
    any signal. A low quantile rather than the strict minimum, so one sounding
    with an unusual interferer cannot set it.

    Give it a day: on 2026-02-04 this returns 8.0 MHz against a declared sweep
    start of 7.5, and that 0.5 MHz is the difference between night-time LOF
    being a measurement and being an upper bound.
    """
    lowest = []
    for ion in ions:
        bins = np.flatnonzero(presence_above(ion, threshold_db))
        if bins.size:
            lowest.append(float(ion.freq[bins[0]]))
    if not lowest:
        return float("nan")
    return float(np.quantile(lowest, quantile))


def measure_band_ceiling(ions, threshold_db: float = DEFAULT_LADDER[0],
                         quantile: float = 0.98,
                         min_run: int = DEFAULT_MIN_RUN) -> float:
    """Estimate the highest frequency a circuit's echoes actually reach.

    The counterpart of :func:`measure_band_floor`, and it lives here for that
    reason: same threshold, same argument shape, opposite end of the axis.
    (``muf.pick`` is the other candidate home, but this module already imports
    from it and the dependency cannot run both ways.)

    **It is not a mirror image, in two ways, and both were found by running the
    mirror image against real recordings.**

    First, the statistic is a *high* quantile where the floor's is a low one.
    Absorption only ever pushes the LOF up away from the hardware floor, so a
    minimum over a day recovers that floor. The ionosphere only ever pushes the
    MOF down from whatever the circuit could support, so the ceiling is the
    best it ever managed -- a maximum, softened to a quantile so one unusual
    sounding cannot set it.

    Second, and less obvious: this reads the top of the last **run**, not the
    last lit bin. :func:`measure_band_floor` can use a bare
    :func:`presence_above` because the bottom of the band really is dead --
    that is what makes it measurable. The top is not. Narrowband interferers
    sit above the detection level right up to the sweep stop, and a single-bin
    rule returns 24.80 MHz on DOB's Cyprus archive: the interference, not the
    circuit. The continuity rule that defends ``pick_muf`` from exactly this is
    the whole difference between 24.80 and the real answer.

    **Measure it over a span that includes the band's best hours.** Given only
    a night it reads back the ionosphere instead of the circuit, and every
    daytime pick afterwards is flagged as band-limited. The ceiling is a
    property of the path and the equipment, so measure it once and keep it in
    configuration; re-deriving it per run makes the flag mean something
    different every day, which is worse than the fault it fixes.
    """
    highest = []
    for ion in ions:
        runs = find_runs(presence_above(ion, threshold_db), min_run)
        if runs:
            highest.append(float(ion.freq[runs[-1][1]]))
    if not highest:
        return float("nan")
    return float(np.quantile(highest, quantile))
