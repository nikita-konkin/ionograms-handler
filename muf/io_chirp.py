"""Reading chirpsounder2 ``lfm_ionogram-*.h5`` products.

chirpsounder2 has already done the FFT and the range gate by the time it writes
this file, so this module enters the pipeline *below* ``spectro`` and
``calibrate``: it produces an :class:`~muf.spectro.Ionogram` directly, and
everything from ``pick`` onward is unchanged. See ``docs/architecture.md`` sec. 3.

The whole of the difficulty is in one line of ``calc_ionograms.py``::

    noise_floor[i] = n.median(SNR[i,:])
    SNR[i,:] = (SNR[i,:] - noise_floor[i]) / noise_floor[i]

so v2's ``SNR`` is a **linear excess-power ratio referenced to the row median**,
already dimensionless. This pipeline's ``spectro.compute`` normalizes the same
row by ``NOISE_COEF * median`` (``NOISE_COEF = 4*ln2``). Both divide by the
*same statistic*, which makes the conversion exact rather than empirical:

    power_muf = (SNR_v2 + 1) / NOISE_COEF

The ``+1`` undoes v2's mean subtraction, recovering ``P/median``; dividing by
``NOISE_COEF`` re-applies this pipeline's convention. A median-noise cell has
``SNR_v2 = 0`` and lands at ``10*log10(1/NOISE_COEF) + 30 = 25.57 dB`` -- which
is exactly where a median-noise cell lands on the ``.lfs`` path. **One
convention, one threshold**: 43 dB keeps its meaning (``signal-chain.md``
sec. 5.2) without recalibration.

> **This corrects ``architecture.md`` sec. 3.3**, which says to *divide* ``SNR``
> by ``noise_floor``. ``SNR`` is already a ratio to ``noise_floor``; dividing
> again would scale the array by raw spectral power and silently produce the
> plausible-looking wrong numbers that section exists to warn about.
> ``noise_floor`` is read here for provenance only and is deliberately not used
> in the conversion.

Two reasons the offset must still be *measured* at the parallel run
(``architecture.md`` sec. 2.4) rather than declared closed by the algebra above:

* v2 windows and steps its own FFT (``fftlen``, ``fft_step``, scipy's Hann).
  Window gain and overlap change the median-to-peak ratio a coherent trace
  reaches, so equal normalization does not guarantee equal dB on the *echo*.
* ``SNR`` is stored ``float16`` after ``SNR[SNR < storage_snr_threshold] = nan``.

:data:`SNR_OFFSET_DB` exists to carry that measured residual. It is 0.0 until
the parallel run measures it, and it is a knob rather than a constant so the
measurement lands in one place.

What the file does *not* carry: ``cf``, ``dur``, ``dec``, ``rmin``/``rmax``, and
any transmitter or receiver coordinates. :class:`ChirpHeader` reconstructs the
first group from the axes that are present, and takes coordinates from a
registry supplied by the caller -- see :func:`load`.

**The frequency axis starts near 0 MHz, not at 7.5 MHz.** ``freqs`` is
``rate * arange(n) * fft_step / sr_dec``, and v2's ``t0`` is the extrapolated
instant the sweep crosses 0 Hz, so elapsed time times chirp rate *is* absolute
RF frequency. It is genuinely absolute -- MUF read off ``Ionogram.freq`` is
correct -- but where ``.lfs`` covers 7.5-32.5 MHz because v1 records a fixed
band around a fixed LO, v2 covers 0 to ``[lfm] maximum_analysis_frequency``
(default 25 MHz). The bottom megahertz or two are below the antenna's response
and will be noise. That config value is also the only source for the *nominal*
sweep stop, which is what ``sweep_complete`` needs; it is not in the file.

**The stored range axis is relative in two of v2's three branches, and
completing it is mandatory.** ``calc_ionograms.py`` writes ``ranges`` from one
of three places, and only the first has already been made absolute:

* ``serendipitous`` -- stores ``range_gates + range_offset`` and sets
  ``range_offset_applied`` True. Nothing to add.
* ``manual_range_extent`` -- **selects** on ``range_gates + range_offset`` but
  **stores** the raw ``range_gates``, leaving ``range_offset_applied`` False.
  This is the branch DOB runs in production, and the mismatch between the two
  expressions is easy to read past.
* neither -- stores ``range_gates[abs(range_gates) < max_range_extent]``,
  centred on zero, +/-2000 km by default. Also False.

In the two False cases the values are offsets *around the direct-path delay*,
not virtual ranges. The absolute range is recovered from ``t0``, whose
fractional part is the propagation time, exactly as upstream's own
``plot_ionograms.py`` does::

    # assume that t0 is at the start of a standard unix second
    # therefore, the propagation time is anything added to a full second
    dt = (t0 - n.floor(t0))
    dr = dt * c.c / 1e3
    range_gates = dr + ranges / 1e3

Skipping that step leaves an axis centred on 0 km instead of on ~2710 km for
cyprus1 -- an ionogram that plots, and is wrong by the whole path length. Note
the corollary: the default +/-2000 km window is *around the expected echo*, so
it comfortably contains a 2710 km trace. It is not a gate that needs widening.

Observed on the DOB products of 2026-08-04: ``t0`` is an exact integer second
on all 381, so ``range_offset`` is zero and the correction is a no-op there.
That is ``find_timings.py`` clustering detections onto the transmitter's
schedule -- the *detection* files keep the sub-second part
(``chirp_time = 1785856440.05435``), the ionogram's ``t0`` does not. So the
term is right, and unexercised by that particular archive; do not conclude
from a matching plot that it can be dropped.

**And it is exercised, wrongly, by search mode.** The DOB run of 2026-08-05
kept ``t0``'s sub-second part on all 232 products, and every range it implied
was nonsense: 16,700 km for cyprus1, whose path to DOB is 3436 km. The term is
not the problem. It reads the receiver's clock, and DOB's was **0.956 s slow**,
so the fractional second it recovers is the clock error rather than the
propagation delay. A scheduled sounding is immune -- ``t0`` comes from
``sounder_timings``, and a wrong clock moves the whole sounding in time instead
of corrupting its range -- which is why 2026-08-04 showed nothing of this.

Nothing in the file reveals the error. ``t0`` is stable to 0.5 ms across eleven
hours (the sample clock is GPS-disciplined; only its epoch is wrong), the axis
is the right shape, and the ionogram plots. It took an external schedule -- the
Twente chirp list, which publishes cyprus1's transmit seconds -- to see it. So
``absolute_ranges_km`` completes the axis only when something can vouch for the
offset, and returns it relative otherwise; see that function and
``ChirpHeader.range_is_relative``.
"""

from __future__ import annotations

import datetime as _dt
import math
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import calibrate
from .calibrate import Calibration
from .spectro import NOISE_COEF, Ionogram

#: Measured dB offset between v2 products and ``.lfs`` products on the *echo*,
#: added to every dB value this module produces.
#:
#: The normalization algebra in the module docstring is exact, so this covers
#: only the residual from v2's differing FFT window and overlap. It stays 0.0
#: until the parallel run of ``architecture.md`` sec. 2.4 measures it against v1
#: output on the same path. Setting it from a guess would defeat its purpose.
SNR_OFFSET_DB = 0.0

#: Datasets ``calc_ionograms.py`` always writes. Anything missing means the file
#: is not an ionogram product -- a detection file, or a different version.
REQUIRED = ("SNR", "freqs", "ranges", "rate", "t0", "sr")

#: chirpsounder2 builds its range axis with ``scipy.constants.c``, while
#: ``calibrate.C_KM_S`` is exactly 3e5 km/s. Reconstructing v2's axis has to use
#: v2's constant: the 0.07% difference is irrelevant to any range, but
#: ``n_range_full`` is a *rounded ratio* of two lengths and only recovers the
#: true FFT length when both come from the same speed of light.
C_M_S = 299_792_458.0

#: v2's spelling for "no transmitter identified" (``calc_ionograms.py:189``,
#: upstream's typo). Products carrying it come from serendipitous mode and
#: have no station behind them, so nothing can check their range offset.
UNIDENTIFIED_TX = "unkown"

#: Longest one-way terrestrial path, half the Earth's circumference plus room
#: for the ionospheric excess of a multi-hop route. An offset past this is not
#: a distant transmitter, it is a broken timing solution.
MAX_VIRTUAL_RANGE_KM = 22_000.0


def chirp_half_span_km(rate: float, sr: float) -> float:
    """Half-width of v2's ungated range axis, in km.

    ``calc_ionograms.py`` builds the axis as
    ``ds * fftshift(fftfreq(fftlen, 1/sr))`` with
    ``ds = get_m_per_Hz(rate) = c/rate``, commented upstream as "m/Hz one-way
    travel time". Note what is *not* there: ``ds`` depends only on ``rate``, so
    v2 applies no path-type divisor.

    For an **oblique** path that agrees with ``calibrate.range_half_span``
    exactly, because ``div_coef = 2`` and ``C_KM_S * (sr/2) / rate`` is the same
    quantity -- which is why 43 dB, the gate and the 2710 km baseline all carry
    over unchanged on the cyprus1 circuit.

    For a **vertical** sounding they differ by 2x: this pipeline's
    ``div_coef = 4`` halves the axis, v2 does not. The file's ``ranges`` dataset
    is on v2's convention, so that is the one used here -- deriving the span
    from ``div_coef`` instead would contradict the axis actually stored, and
    silently misplace ``gate_idx`` and ``n_range_full``. :func:`load` warns when
    it sees a vertical path for this reason.
    """
    return (C_M_S / rate) * (sr / 2.0) / 1e3

#: ``lfm_ionogram-{txname}-{station_name}-{ch}-{cid:03d}-{t0:.2f}.h5``
_NAME_RE = re.compile(
    r"^lfm_ionogram-(?P<tx>.+?)-(?P<rx>.+?)-(?P<ch>.+?)-(?P<cid>\d+)-(?P<t0>[\d.]+)\.h5$"
)


def _scalar(value):
    """Unwrap an h5py scalar dataset, decoding bytes to ``str``."""
    if isinstance(value, bytes):
        return value.decode("ascii", errors="replace").strip()
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _scalar(value[()])
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class ChirpHeader:
    """Acquisition parameters for one v2 sounding.

    Deliberately mirrors :class:`~muf.io_lfs.LfsHeader`'s attribute surface --
    ``tx_name``/``rx_name``, the four coordinates, ``datetime``, ``is_oblique``,
    ``path_type``, ``div_coef``, and the ``rate``/``sample_rate``/``dec``/
    ``rmin``/``rmax`` that ``calibrate`` reads -- so ``geometry.path_of``,
    ``calibrate.default_gate``, ``render``, ``export.saoxml`` and ``pipeline``
    consume it unchanged. Nothing downstream does an ``isinstance`` check; they
    read attributes, which is what makes one ``Ionogram`` type serve both
    formats (``architecture.md`` sec. 3.1).

    ``cf``, ``dur``, ``sample_rate`` and ``dec`` are *reconstructed* so that
    ``calibrate.sweep_bounds`` and ``calibrate.range_half_span`` reproduce the
    axes actually stored in the file. They are not read from it -- v2 does not
    record them -- so treat them as derived, not as acquisition truth.
    """

    path: Path

    format: str               # discriminator for `sounding.format` (sec. 5.2)
    tx_name: str
    tx_latitude: float
    tx_longitude: float
    rx_name: str
    rx_latitude: float
    rx_longitude: float

    t0: float                 # unix seconds, sweep start
    chirp_id: int
    channel: str

    rate: float               # chirp rate, Hz/s
    sample_rate: float        # decimated rate `sr`; with dec == 1 this is `fd`
    dec: int
    cf: float                 # reconstructed so sweep_bounds matches `freqs`
    dur: float                # reconstructed from the swept span and `rate`
    rmin: int                 # from the stored range axis, km
    rmax: int                 # from the stored range axis, km

    noise_floor_median: float  # provenance only; see the module docstring
    range_offset_applied: bool
    range_start_m: float      # NaN unless v2 gated in serendipitous mode
    range_stop_m: float
    num_detections: int | None

    # Why `absolute_ranges_km` declined to complete the axis, empty when it
    # did. Non-empty means every range on this sounding is relative: the
    # differences are right, the zero is not. Carried on the header rather
    # than left implicit so a consumer that needs absolute range -- geometry,
    # a gate, a stored virtual height -- can refuse rather than quietly
    # publish a number that is off by the direct-path delay.
    range_relative_reason: str

    # Written into every product by `chirpsounder_version.tag_hdf5`. This is the
    # acquisition-side half of the provenance `sounding.config_epoch_id` records
    # (architecture.md sec. 5.2): which code produced the number, alongside
    # which configuration. Empty for a file written before the tagger existed.
    software_version: str
    git_commit: str
    git_dirty: bool | None

    #: True when `save_raw_voltage` was on and the dechirped waveform is in
    #: the file. The only product that can be re-derived at another FFT
    #: window; see `reprocess` and architecture.md sec. 3.4.
    has_raw_voltage: bool = False

    @property
    def range_offset_km(self) -> float:
        """Propagation range implied by ``t0``'s fractional part.

        v2 detects the chirp's extrapolated 0 Hz crossing, and transmitters
        start their sweep on an integer second, so whatever ``t0`` carries past
        the second *is* the one-way travel time. For cyprus1 at 2710 km that is
        9.03 ms. This is the term that turns v2's zero-centred range gates into
        virtual ranges; see the module docstring.
        """
        return (self.t0 - math.floor(self.t0)) * C_M_S / 1e3

    @property
    def range_is_relative(self) -> bool:
        """True when the range axis has no meaningful zero.

        ``vrange`` differences stay correct -- an echo 300 km above another is
        still 300 km above it -- so MUF, which is a frequency, is unaffected.
        Virtual height is not: it would be a differential number near zero.
        """
        return bool(self.range_relative_reason)

    @property
    def datetime(self) -> _dt.datetime:
        return _dt.datetime.fromtimestamp(self.t0, tz=_dt.timezone.utc)

    @property
    def is_oblique(self) -> bool:
        return self.tx_name != self.rx_name

    @property
    def path_type(self) -> str:
        return "oblique" if self.is_oblique else "vertical"

    @property
    def div_coef(self) -> float:
        """Range-scaling divisor: 2 oblique, 4 vertical. As ``LfsHeader``."""
        return 2.0 if self.is_oblique else 4.0

    @property
    def has_coordinates(self) -> bool:
        """False until a station registry supplies lat/lon.

        v2 stores ``txname`` and ``station_name`` as bare strings, so without a
        registry the geometry is unavailable and ``calibrate.default_gate``
        falls back to ``rmin``. Nothing crashes, but the gate is wider and
        ``geometry.path_of`` returns a meaningless distance.
        """
        return all(
            math.isfinite(c)
            for c in (self.tx_latitude, self.tx_longitude,
                      self.rx_latitude, self.rx_longitude)
        )

    def __str__(self) -> str:
        return (
            f"{self.tx_name}->{self.rx_name} ({self.path_type}) "
            f"{self.datetime:%Y-%m-%d %H:%M:%S}Z"
        )


def _absolute_ranges(ranges_m: np.ndarray, t0: float, offset_applied: bool,
                     range_start_m: float, tx_name: str
                     ) -> tuple[np.ndarray, str]:
    """Back end of :func:`absolute_ranges_km`; also reports why.

    Returns the axis and a reason string, empty when the axis came out
    absolute. A non-empty reason means the caller is holding *relative*
    range -- correct differences, meaningless zero -- and has to say so.
    """
    ranges_km = np.asarray(ranges_m, dtype=np.float64) / 1e3
    if offset_applied:
        if (math.isfinite(range_start_m) and ranges_km.size
                and float(np.nanmedian(ranges_km)) < 0.75 * range_start_m / 1e3):
            return ranges_km + range_start_m / 1e3, ""
        return ranges_km, ""

    offset_km = (t0 - math.floor(t0)) * C_M_S / 1e3
    if tx_name == UNIDENTIFIED_TX:
        return ranges_km, (
            f"transmitter is v2's {UNIDENTIFIED_TX!r} marker, so the {offset_km:.0f} km "
            f"implied by t0 rests on a timing solution nothing has cross-checked"
        )
    if offset_km > MAX_VIRTUAL_RANGE_KM:
        return ranges_km, (
            f"t0 implies a {offset_km:.0f} km direct path, past the "
            f"{MAX_VIRTUAL_RANGE_KM:.0f} km a one-way terrestrial path can reach"
        )
    return ranges_km + offset_km, ""


def absolute_ranges_km(ranges_m: np.ndarray, t0: float, offset_applied: bool,
                       range_start_m: float = float("nan"),
                       tx_name: str = "") -> np.ndarray:
    """Turn v2's stored range axis into virtual range in km.

    Mirrors ``plot_ionograms.py``, which is the only statement upstream makes
    about how its own axis is meant to be read:

    * ``range_offset_applied`` False -- the stock path -- means the axis is
      centred on the direct-path delay, so add ``(t0 - floor(t0)) * c``.
    * True means v2 already wrote display ranges, *except* when it stored them
      relative to the window start. Upstream detects that with
      ``nanmedian(ranges) < 0.75 * range_start_m`` and adds ``range_start_m``;
      the same test is applied here, deliberately including its arbitrary
      0.75, because diverging from it would put our ranges on a different axis
      from every plot and detection file v2 produces.

    **Two cases refuse the offset and hand back a relative axis instead.**
    Both come from serendipitous mode, where ``t0`` is *measured* by
    ``find_timings.py`` rather than imposed from ``sounder_timings``, so its
    absolute calibration is only as good as the detector:

    * ``tx_name`` is v2's ``"unkown"`` marker (``calc_ionograms.py:189``) --
      nothing identifies the transmitter, so there is no independent distance
      to check the offset against and no station registry entry it could feed.
    * the offset exceeds ``MAX_VIRTUAL_RANGE_KM``, which no one-way terrestrial
      path can reach whatever the transmitter.

    Measured on the DOB search-mode products of 2026-08-05: cyprus1's four
    published transmit seconds (235, 240, 245, 300 on a 300 s cycle) came back
    as 234.0557, 239.0558, 244.0558 and 299.0560, agreeing on a receiver clock
    0.9557 s slow to within 0.35 ms. The offset those files implied was
    16,700 km on a 3436 km path -- not a scaled range, an unrelated number.

    A relative axis keeps every range *difference* correct, which is what the
    extractors and ``pick_muf`` actually consume, so the cost of refusing is
    the absolute zero and nothing else. Correcting for a measured clock error
    is deliberately **not** done here: the correction belongs at acquisition,
    where the clock can be fixed, not in a reader guessing at what a past
    receiver's clock was doing.
    """
    return _absolute_ranges(ranges_m, t0, offset_applied, range_start_m,
                            tx_name)[0]


def _coords_for(name: str,
                registry: Mapping[str, tuple[float, float]] | None
                ) -> tuple[float, float]:
    """Look a station up in the registry. NaN when absent, never an exception.

    An unknown station must degrade to "no geometry" rather than stopping the
    extraction: a sounding whose transmitter is not yet in the registry still
    has a usable ionogram, and a hard failure here would make adding a station
    to the search (the whole point of v2) break the pipeline until someone
    edited a table.
    """
    if registry is None:
        return float("nan"), float("nan")
    entry = registry.get(name)
    if entry is None:
        return float("nan"), float("nan")
    return float(entry[0]), float(entry[1])


def read_header(path: str | Path,
                stations: Mapping[str, tuple[float, float]] | None = None
                ) -> ChirpHeader:
    """Parse the metadata of an ``lfm_ionogram-*.h5`` without reading ``SNR``.

    Cheap enough to run over a whole archive: it touches the small scalar
    datasets and the two axis vectors, never the compressed ionogram.
    """
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as fh:
        missing = [k for k in REQUIRED if k not in fh]
        if missing:
            present = ", ".join(sorted(fh.keys())) or "<empty>"
            raise ValueError(
                f"{path}: not a chirpsounder2 ionogram product -- missing "
                f"{', '.join(missing)}. Datasets present: {present}"
            )

        freqs = np.asarray(fh["freqs"][()], dtype=np.float64)
        ranges = np.asarray(fh["ranges"][()], dtype=np.float64)
        rate = float(_scalar(fh["rate"][()]))
        sr = float(_scalar(fh["sr"][()]))
        t0 = float(_scalar(fh["t0"][()]))

        tx_name = str(_scalar(fh["txname"][()])) if "txname" in fh else ""
        rx_name = str(_scalar(fh["station_name"][()])) if "station_name" in fh else ""
        channel = str(_scalar(fh["ch"][()])) if "ch" in fh else ""
        chirp_id = int(_scalar(fh["id"][()])) if "id" in fh else -1

        noise_floor = (np.asarray(fh["noise_floor"][()], dtype=np.float64)
                       if "noise_floor" in fh else np.array([np.nan]))
        offset_applied = (bool(_scalar(fh["range_offset_applied"][()]))
                          if "range_offset_applied" in fh else False)
        has_raw_voltage = RAW_VOLTAGE_KEY in fh
        range_start = (float(_scalar(fh["range_gate_start_m"][()]))
                       if "range_gate_start_m" in fh else float("nan"))
        range_stop = (float(_scalar(fh["range_gate_stop_m"][()]))
                      if "range_gate_stop_m" in fh else float("nan"))
        detections = (int(_scalar(fh["num_detections"][()]))
                      if "num_detections" in fh else None)

        attrs = fh.attrs
        software_version = str(_scalar(attrs.get("chirpsounder2_version", "")))
        git_commit = str(_scalar(attrs.get("git_commit", "")))
        git_dirty = (bool(_scalar(attrs["git_dirty"]))
                     if "git_dirty" in attrs else None)

    if freqs.size == 0 or ranges.size == 0:
        raise ValueError(f"{path}: empty frequency or range axis")
    if rate <= 0:
        raise ValueError(f"{path}: non-positive chirp rate {rate}")

    # Names come from the file; the filename is only a fallback, since a station
    # renamed in config would make the two disagree and the file is the record
    # of what was actually written.
    if not tx_name or not rx_name:
        match = _NAME_RE.match(path.name)
        if match:
            tx_name = tx_name or match.group("tx")
            rx_name = rx_name or match.group("rx")

    tx_lat, tx_lon = _coords_for(tx_name, stations)
    rx_lat, rx_lon = _coords_for(rx_name, stations)

    # Reconstruct the acquisition scalars `calibrate` expects. `dec = 1` with
    # `sample_rate = sr` makes `sample_rate / dec` the decimated rate, which is
    # what range_half_span actually wants; `cf` is chosen so that sweep_bounds
    # returns the frequency span the file really covers.
    # rmin/rmax feed calibrate.default_gate, so they have to be absolute
    # virtual range -- the relative axis would put the gate around 0 km.
    ranges_km, relative_reason = _absolute_ranges(ranges, t0, offset_applied,
                                                  range_start, tx_name)
    if relative_reason:
        warnings.warn(
            f"{path.name}: range axis left relative -- {relative_reason}. "
            f"Differences are correct; the zero is not. See "
            f"`ChirpHeader.range_is_relative`.",
            stacklevel=2,
        )
    freq_lo, freq_hi = float(freqs.min()), float(freqs.max())

    return ChirpHeader(
        path=path,
        format="chirp2",
        tx_name=tx_name or "unknown",
        tx_latitude=tx_lat,
        tx_longitude=tx_lon,
        rx_name=rx_name or "unknown",
        rx_latitude=rx_lat,
        rx_longitude=rx_lon,
        t0=t0,
        chirp_id=chirp_id,
        channel=channel,
        rate=rate,
        sample_rate=sr,
        dec=1,
        cf=freq_lo + sr / 2.0,
        dur=(freq_hi - freq_lo) / rate,
        rmin=int(math.floor(float(ranges_km.min()))),
        rmax=int(math.ceil(float(ranges_km.max()))),
        noise_floor_median=float(np.nanmedian(noise_floor)),
        range_offset_applied=offset_applied,
        range_start_m=range_start,
        range_stop_m=range_stop,
        num_detections=detections,
        range_relative_reason=relative_reason,
        has_raw_voltage=has_raw_voltage,
        software_version=software_version,
        git_commit=git_commit,
        git_dirty=git_dirty,
    )


def snr_to_power(snr: np.ndarray) -> np.ndarray:
    """v2 ``SNR`` -> this pipeline's median-equalized linear power.

    ``SNR_v2 = (P - median)/median``, so ``P/median = SNR_v2 + 1``; dividing by
    ``NOISE_COEF`` puts it on the scale ``spectro.compute`` produces. See the
    module docstring for why this is exact and not a fitted offset.

    Cells v2 dropped as noise are stored as NaN
    (``SNR[SNR < storage_snr_threshold] = nan``, threshold 2 by default). They
    are filled with ``SNR_v2 = 0`` -- the row median, the best estimate for a
    cell known only to be below the storage threshold. That places them at
    25.57 dB, well under the 43 dB detection threshold, so the sparsification
    cannot manufacture a detection. It cannot suppress one either: 43 dB is
    ``SNR_v2 = 54.3``, more than an order of magnitude above the storage
    threshold, so nothing a detector would have kept was ever discarded.
    """
    snr = np.asarray(snr, dtype=np.float32)
    # float16 saturates at 65504; a saturated cell is a strong echo, not noise,
    # so clamp to finite rather than dropping it to the floor with the NaNs.
    finite = np.nan_to_num(snr, nan=0.0, posinf=np.float32(65504.0), neginf=-1.0)
    # P >= 0 guarantees SNR >= -1, but float16 rounding can undershoot.
    return np.maximum(finite + 1.0, 0.0) / NOISE_COEF


def _build_calibration(header: ChirpHeader,
                       freqs_mhz: np.ndarray,
                       vrange_km: np.ndarray,
                       gate_km: tuple[float, float],
                       nominal_stop_mhz: float | None) -> Calibration:
    """Axes straight from the file's own ``freqs`` and ``ranges``.

    Not via ``calibrate.build``: that derives the axes from header arithmetic,
    which is right for ``.lfs`` and wrong here. v2 may subsample the frequency
    axis (``max_ionogram_frequency_steps``), so the stored vector is the only
    authority for what each row means -- and it need not be uniformly spaced.
    """
    step = float(np.median(np.abs(np.diff(vrange_km)))) if vrange_km.size > 1 else 1.0
    half_span = chirp_half_span_km(header.rate, header.sample_rate)

    if freqs_mhz.size > 1:
        freq_step = float(np.median(np.diff(freqs_mhz)))
    else:
        freq_step = 0.0
    # Bin *edges*, to match calibrate.build, where freq labels are bin centres
    # and freq_stop is the upper edge of the last bin.
    freq_start = float(freqs_mhz[0]) - freq_step / 2.0
    freq_stop = float(freqs_mhz[-1]) + freq_step / 2.0

    # Where the stored slice sits on the full FFT axis. Exact when v2 did not
    # add its sub-second range offset; approximate when it did, since that
    # shifts the axis off the fftfreq grid.
    n_range_full = max(int(round(2 * half_span / step)), vrange_km.size) if step > 0 else vrange_km.size
    i_lo = max(0, int(round((half_span - float(vrange_km[0])) / step))) if step > 0 else 0
    i_hi = i_lo + vrange_km.size - 1

    return Calibration(
        freq=freqs_mhz,
        vrange=vrange_km,
        freq_start=freq_start,
        freq_stop=freq_stop,
        freq_stop_nominal=(freq_stop if nominal_stop_mhz is None
                           else float(nominal_stop_mhz)),
        half_span=half_span,
        range_step=step,
        # v2 does not zero-pad, so the stored bin spacing is the true
        # resolution. There is no `zero_periods` equivalent to divide out.
        resolution_km=step,
        gate_km=gate_km,
        gate_idx=(i_lo, i_hi),
        n_range_full=n_range_full,
    )


#: Dataset holding the dechirped complex64 voltage, written only when
#: ``save_raw_voltage = true`` in v2's ``[lfm]`` section
#: (``calc_ionograms.py:372``, default false). This is the same *kind* of
#: signal an ``.lfs`` file carries -- stretch-processed, decimated, complex --
#: which is why a product that has it can be re-derived at another FFT window
#: and one that does not cannot.
RAW_VOLTAGE_KEY = "z"

#: v2's incoherent averaging factor (``calc_ionograms.py:134``). Each output
#: row is the mean of 13 sub-windows straddling the nominal position, which is
#: why its ionograms look smoother than a single FFT would give. Reproducing
#: it matters: an ionogram re-derived without it has a different noise
#: distribution, and the 43 dB threshold is defined against the distribution.
V2_OVERSAMPLE = 13


def read_raw_voltage(path: str | Path) -> np.ndarray:
    """The stored dechirped voltage, or raise if the product has none."""
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as fh:
        if RAW_VOLTAGE_KEY not in fh:
            raise ValueError(
                f"{path}: no {RAW_VOLTAGE_KEY!r} dataset. chirpsounder2 writes "
                f"it only with `save_raw_voltage = true` in [lfm]; without it "
                f"the waveform was discarded at acquisition and the sounding "
                f"cannot be re-derived at another window."
            )
        return np.asarray(fh[RAW_VOLTAGE_KEY][()], dtype=np.complex64)


def v2_spectrogram(z: np.ndarray, window: int, step: int,
                   n_oversample: int = V2_OVERSAMPLE) -> np.ndarray:
    """Port of ``calc_ionograms.spectrogram``, deliberately line for line.

    Reimplemented rather than imported: the clone is pinned and disposable
    (``architecture.md`` sec. 2.2), so depending on it at runtime would make
    our ionograms a property of whichever commit is checked out. The cost is
    that this must be kept faithful, and the test that keeps it honest
    re-derives a product at its *original* window and requires the stored
    array back.

    Note ``np.conj(z)`` is applied by the caller, as upstream does, and that
    the row count shrinks as ``window`` grows -- the last partial window is
    dropped, so a coarser reprocessing covers slightly less of the sweep.
    """
    from scipy.signal import windows

    z = np.asarray(z)
    n_spec = int((len(z) - window) / step)
    if n_spec < 1:
        raise ValueError(
            f"window {window} with step {step} leaves no complete row in "
            f"{len(z)} samples; the largest usable window is about "
            f"{len(z) - step}"
        )
    if n_oversample % 2 == 0:
        n_oversample += 1

    n_substep = int(step / n_oversample)
    substep_idx = np.arange(n_oversample, dtype=int) - int((n_oversample - 1) / 2)
    wf = windows.hann(window)

    out = np.zeros((n_spec, window), dtype=np.float64)
    for i in range(n_spec):
        n_avg = 0.0
        for j in range(n_oversample):
            i0 = i * step + substep_idx[j] * n_substep
            i1 = i0 + window
            if i0 >= 0 and i1 < len(z):
                out[i] += np.abs(np.fft.fftshift(np.fft.fft(wf * z[i0:i1]))) ** 2.0
                n_avg += 1.0
        if n_avg > 0:
            out[i] /= n_avg
    return out


def reprocess(path: str | Path,
              window: int,
              gate_km: tuple[float, float] | None = None,
              *,
              stations: Mapping[str, tuple[float, float]] | None = None,
              nominal_stop_mhz: float | None = None) -> Ionogram:
    """Re-derive a v2 product at a different FFT window, from its stored ``z``.

    This is the thing ``architecture.md`` sec. 3.4 says is impossible, made
    possible by one config flag. Range resolution is ``2 * half_span /
    window``, so doubling ``window`` halves the bin size and halves the number
    of frequency rows that fit -- the same trade ``--window`` makes on the
    ``.lfs`` path.

    The frequency axis is recomputed rather than reused: ``freqs = rate *
    arange(n_spec) * step / sr`` (``calc_ionograms.py:260``), and ``n_spec``
    depends on ``window``. Reusing the stored ``freqs`` would silently mislabel
    every row.

    Normalization follows :func:`snr_to_power` exactly as for a stored
    product, so a reprocessed ionogram sits on the same dB scale and the 43 dB
    threshold keeps its meaning.
    """
    header = read_header(path, stations)
    z = read_raw_voltage(path)

    import h5py
    with h5py.File(Path(path), "r") as fh:
        stored_freqs = np.asarray(fh["freqs"][()], dtype=np.float64)
    if stored_freqs.size < 2:
        raise ValueError(f"{path}: cannot recover the frequency step from "
                         f"{stored_freqs.size} stored frequencies")
    sr = header.sample_rate
    step = int(round((float(np.median(np.diff(stored_freqs))) / header.rate) * sr))

    spec = v2_spectrogram(np.conj(z), window=window, step=step)

    # v2's own normalization, applied per row, then this pipeline's convention.
    noise_floor = np.median(spec, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = (spec - noise_floor[:, None]) / noise_floor[:, None]
    power = snr_to_power(snr)

    freqs_mhz = (header.rate * np.arange(spec.shape[0]) * step / sr) / 1e6
    ranges_km = (C_M_S / header.rate) * np.fft.fftshift(
        np.fft.fftfreq(window, d=1.0 / sr)) / 1e3
    ranges_km, relative_reason = _absolute_ranges(
        ranges_km * 1e3, header.t0, header.range_offset_applied,
        header.range_start_m, header.tx_name)
    if relative_reason:
        warnings.warn(
            f"{Path(path).name}: reprocessed range axis left relative -- "
            f"{relative_reason}.", stacklevel=2)

    vrange_km = ranges_km[::-1]
    power = power[:, ::-1]

    lo, hi = float(vrange_km.min()), float(vrange_km.max())
    if gate_km is not None:
        want_lo, want_hi = float(gate_km[0]), float(gate_km[1])
        lo, hi = max(lo, want_lo), min(hi, want_hi)
        keep = (vrange_km >= lo) & (vrange_km <= hi)
        if not keep.any():
            raise ValueError(f"{path}: gate {want_lo:.0f}-{want_hi:.0f} km keeps no bin")
        vrange_km, power = vrange_km[keep], power[:, keep]

    if SNR_OFFSET_DB:
        power = power * 10.0 ** (SNR_OFFSET_DB / 10.0)

    cal = _build_calibration(header, freqs_mhz, vrange_km, (lo, hi),
                             nominal_stop_mhz)
    return Ionogram(power=power.astype(np.float32), cal=cal, header=header,
                    window=window, zero_periods=0)


def load(path: str | Path,
         gate_km: tuple[float, float] | None = None,
         *,
         stations: Mapping[str, tuple[float, float]] | None = None,
         nominal_stop_mhz: float | None = None,
         header: ChirpHeader | None = None) -> Ionogram:
    """Read an ``lfm_ionogram-*.h5`` into an :class:`Ionogram`.

    ``gate_km`` is applied **only where it is narrower than the gate v2 already
    applied** (``architecture.md`` sec. 3.3). v2 wrote only the range window it
    kept, so re-gating to a wider window cannot recover rows that are not in the
    file; asking for one warns and yields the file's own extent. Never
    double-gate.

    ``stations`` maps a station name to ``(lat, lon)``. Without it the geometry
    is unavailable: the gate falls back to the stored range extent and
    ``geometry.path_of`` is meaningless. That registry is the next piece of work
    (``architecture.md`` sec. 2.3.3).

    ``nominal_stop_mhz`` is the frequency the sweep was *intended* to reach.
    v2 records no equivalent of ``sweep_complete``, so without this the
    calibration reports a complete sweep. It is v2's
    ``[lfm] maximum_analysis_frequency`` (default 25e6 Hz, so 25.0 here), which
    lives in the config rather than in the product -- supply it and truncation
    becomes detectable exactly as it is for ``.lfs``.

    ``window`` and ``zero_periods`` are absent by design -- v2 fixed both when it
    computed the product, and an ``.h5`` sounding cannot be re-derived at a
    different FFT window (``architecture.md`` sec. 3.4).
    """
    import h5py

    path = Path(path)
    header = header or read_header(path, stations)

    with h5py.File(path, "r") as fh:
        snr = np.asarray(fh["SNR"][()])
        freqs_hz = np.asarray(fh["freqs"][()], dtype=np.float64)
        ranges_m = np.asarray(fh["ranges"][()], dtype=np.float64)

    if snr.ndim != 2:
        raise ValueError(f"{path}: SNR has shape {snr.shape}, expected 2-D")
    if snr.shape != (freqs_hz.size, ranges_m.size):
        raise ValueError(
            f"{path}: SNR is {snr.shape} but the axes are "
            f"{freqs_hz.size} frequencies x {ranges_m.size} ranges"
        )

    if not header.is_oblique:
        # `get_m_per_Hz` has no path-type divisor, so v2's vertical ranges are
        # 2x this pipeline's. Rescaling them here would contradict the file's
        # own `ranges` dataset, so the axis is left as stored and the
        # discrepancy is surfaced instead of buried.
        warnings.warn(
            f"{path.name}: {header.tx_name} transmits and receives, but v2's "
            f"range axis carries no vertical divisor. These ranges are 2x what "
            f"an .lfs vertical sounding reports for the same echo.",
            stacklevel=2,
        )

    # Complete the axis before anything else touches it: with stock config the
    # stored gates are centred on the direct-path delay, and every range in
    # this module is absolute virtual range.
    # `read_header` already ran this and warned if it came out relative, so the
    # decision is taken from the header rather than made a second time.
    ranges_km = absolute_ranges_km(ranges_m, header.t0,
                                   header.range_offset_applied,
                                   header.range_start_m,
                                   header.tx_name)

    # v2's axis ascends (fftshift of fftfreq); this pipeline's descends, bin 0
    # holding the largest virtual range. See the `calibrate` module docstring
    # for the sign error that convention exists to prevent -- getting this wrong
    # mirrors the ionogram top-to-bottom and still looks plausible.
    vrange_km = ranges_km[::-1]
    power = snr_to_power(snr)[:, ::-1]

    file_lo, file_hi = float(vrange_km.min()), float(vrange_km.max())
    if gate_km is None:
        lo, hi = file_lo, file_hi
    else:
        want_lo, want_hi = float(gate_km[0]), float(gate_km[1])
        if want_lo < file_lo or want_hi > file_hi:
            warnings.warn(
                f"{path.name}: requested gate {want_lo:.0f}-{want_hi:.0f} km is "
                f"wider than the {file_lo:.0f}-{file_hi:.0f} km v2 already "
                f"stored; keeping the stored extent. v2 discarded the rest at "
                f"acquisition and it cannot be recovered here.",
                stacklevel=2,
            )
        lo, hi = max(file_lo, want_lo), min(file_hi, want_hi)
        if lo >= hi:
            raise ValueError(
                f"{path}: gate {want_lo:.0f}-{want_hi:.0f} km does not overlap "
                f"the stored {file_lo:.0f}-{file_hi:.0f} km"
            )
        keep = (vrange_km >= lo) & (vrange_km <= hi)
        if not keep.any():
            raise ValueError(
                f"{path}: gate {lo:.0f}-{hi:.0f} km is narrower than one "
                f"{float(np.median(np.abs(np.diff(vrange_km)))):.1f} km bin"
            )
        vrange_km = vrange_km[keep]
        power = power[:, keep]

    # Applied to `power` before the Ionogram exists, so `.db` -- which caches on
    # first access -- can never be computed from an unoffset array.
    if SNR_OFFSET_DB:
        power = power * 10.0 ** (SNR_OFFSET_DB / 10.0)

    freqs_mhz = freqs_hz / 1e6
    cal = _build_calibration(header, freqs_mhz, vrange_km, (lo, hi),
                             nominal_stop_mhz)

    return Ionogram(
        power=power.astype(np.float32),
        cal=cal,
        header=header,          # ChirpHeader, not LfsHeader -- see the class
        window=cal.n_range_full,
        zero_periods=0,
    )


def find_h5(target) -> list[Path]:
    """Return the ``lfm_ionogram-*.h5`` files under ``target``, sorted by name.

    Mirrors :func:`muf.io_lfs.find_lfs`. The glob is deliberately specific:
    ``detect_chirps.py`` and ``find_timings.py`` write ``chirp-*.h5``,
    ``cdetections-*.h5`` and ``par-*.h5`` into the same tree, and those are
    detection files with an entirely different schema (``architecture.md``
    sec. 2.5).
    """
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
            found.append(item)
        elif item.is_dir():
            found.extend(item.rglob("lfm_ionogram-*.h5"))
        else:
            missing.append(item)

    if missing and not found:
        raise FileNotFoundError(", ".join(str(m) for m in missing))

    unique = {p.resolve(): p for p in found}
    return sorted(unique.values(), key=lambda p: (p.name, str(p)))


def describe(path: str | Path) -> dict[str, str]:
    """Every dataset in the file, with shape and dtype. For schema checking.

    This reader was written against ``calc_ionograms.py`` rather than against a
    real product, because no v2 file existed on the development machine. Run
    this against the first one that arrives to confirm the schema before
    trusting any number that comes out of :func:`load`.
    """
    import h5py

    out: dict[str, str] = {}
    with h5py.File(Path(path), "r") as fh:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                if obj.shape == ():
                    out[name] = f"scalar {obj.dtype}  = {_scalar(obj[()])!r}"
                else:
                    out[name] = f"{obj.shape} {obj.dtype}"

        fh.visititems(visit)
        for key, value in fh.attrs.items():
            out[f"@{key}"] = repr(_scalar(value))
    return out
