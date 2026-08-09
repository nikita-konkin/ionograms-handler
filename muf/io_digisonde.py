"""chirpsounder2 ``digisonde_ionogram-*.h5``: somebody else's sounder, received here.

A Digisonde is a *vertical* incidence pulsed ionosonde -- the UMLCAR
instrument the GIRO network is built from, and the one whose native format
``muf.export.saoxml`` writes. ``receive_digisonde.py`` does not download its
products. It receives the transmissions **off air with this station's own
USRP**, decoding the complementary phase codes the Digisonde transmits, so
what lands here is an *oblique* reception of a vertical sounder several
hundred kilometres away. The operator at the other end never intended it and
does not know.

That makes each one a free oblique circuit with a known transmitter, which is
the interesting part: unlike a serendipitous chirp product, the station is
named, it is in the registry, and its coordinates are exact.

Three things differ from ``io_chirp`` and each is a decision rather than an
accident:

**Two polarizations, and the file does not say which is which.** ``SNR`` is
``(2, n_freq, n_range)``; ``receive_digisonde.py:358`` calls the axis "two
polarizations" and stops there. O and X are physically distinguishable -- they
separate by about half the gyrofrequency -- but nothing in the product records
which channel is which, so this module never claims. They are channel 0 and
channel 1, and :data:`SUM` adds them, which is what upstream's own plot shows
as "Total SNR".

**NaN means "below threshold", not "missing".** ``receive_digisonde.py:535``
writes ``SNR[SNR < snr_threshold] = nan`` before saving, so roughly 90% of a
typical array is NaN by construction. Those cells are noise, not gaps, and
they are read back as an SNR of zero -- the median noise level -- rather than
propagated as NaN into estimators that would then have to special-case them.

**The stored range axis has no offset applied.** ``ranges`` is
``arange(n) * 3 km`` starting at zero; the absolute axis is that plus
``offset_us`` times the speed of light, which is how upstream plots it
(``receive_digisonde.py:514``). That offset is a *configured* per-station
number, not a measured one, so it is applied here and
:attr:`DigisondeHeader.range_is_configured` records that the zero rests on
config rather than on anything cross-checked. Differences are correct
regardless; the zero is only as good as the ini.

The power scale is deliberately ``io_chirp``'s. Both instruments define SNR
the same way -- ``(P - median)/median``, over range, per frequency -- so
:func:`muf.io_chirp.snr_to_power` converts both, and the 43 dB detection level
every estimator shares means the same thing relative to noise in either.
"""

from __future__ import annotations

import datetime as _dt
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from . import calibrate
from .calibrate import Calibration
from .io_chirp import snr_to_power
from .spectro import Ionogram

#: Speed of light in km/s, as ``receive_digisonde.py`` uses it
#: (``scipy.constants.c``) for the ``offset_us`` conversion.
C_KM_S = 299_792.458

#: Datasets every product carries. Anything missing means the file is not a
#: digisonde ionogram -- a chirp product, or a different version.
REQUIRED = ("SNR", "freqs", "ranges", "t0", "transmitter", "receiver")

#: Value of the file's ``type`` dataset. The one positive identification in
#: the schema; the file name is the other.
TYPE_MARKER = "digisonde"

#: Name prefix ``receive_digisonde.py`` writes. Load-bearing for dispatch:
#: chirp ionograms, detection files and these all share ``.h5`` in one tree.
FILE_PREFIX = "digisonde_ionogram-"

#: Combine both polarization channels, as upstream's plot does. The default,
#: because a trace present in only one channel is still a trace, and choosing
#: a channel without knowing whether it is O or X is choosing at random.
SUM = "sum"

#: Decimated receiver sample rate, from ``complementary_code(sr=100e3)``
#: (``receive_digisonde.py:354``). Not stored in the product; the 3 km range
#: bin it implies is, which is the same statement.
SAMPLE_RATE_HZ = 100e3


@dataclass
class DigisondeHeader:
    """Acquisition parameters for one received digisonde sounding.

    Mirrors :class:`~muf.io_chirp.ChirpHeader`'s attribute surface, which
    mirrors :class:`~muf.io_lfs.LfsHeader`'s, so ``geometry.path_of``,
    ``render``, ``export.saoxml`` and ``pipeline`` consume it unchanged.

    ``rate`` is ``nan`` and means it: a Digisonde is **pulsed**, stepping
    through frequencies, and has no chirp rate. Anything reaching for one is
    asking a question this instrument does not answer, and a nan says so
    louder than a plausible number would.
    """

    path: Path

    format: str               # discriminator for `sounding.format` (sec. 5.2)
    tx_name: str
    tx_latitude: float
    tx_longitude: float
    rx_name: str
    rx_latitude: float
    rx_longitude: float

    t0: float                 # unix seconds
    offset_us: float          # configured receiver timing offset
    rate: float               # nan -- pulsed, not chirped
    sample_rate: float
    dec: int
    cf: float
    dur: float
    rmin: int                 # from the stored range axis, km
    rmax: int

    n_pol: int                # channels stored, 2 in every product seen

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
        return not (np.isnan(self.tx_latitude) or np.isnan(self.rx_latitude))

    @property
    def range_offset_km(self) -> float:
        """Absolute-range correction implied by ``offset_us``."""
        return float(self.offset_us) * 1e-6 * C_KM_S

    @property
    def range_is_configured(self) -> bool:
        """True when the range zero rests on a configured offset.

        Always true, and named to be awkward. ``offset_us`` is a per-station
        constant someone typed into the ini; nothing in the product confirms
        it, and 1 ms of error is 300 km. Range *differences* are unaffected.
        The distinction is the same one
        :attr:`muf.io_chirp.ChirpHeader.range_is_relative` draws, arrived at
        from the other direction: there the zero is unknown, here it is
        asserted.
        """
        return True


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def read_header(path: str | Path,
                stations: Mapping[str, tuple[float, float]] | None = None
                ) -> DigisondeHeader:
    """Header only; the ``SNR`` array is not read.

    ``stations`` maps a station name to ``(lat, lon)``. Both ends are bare
    strings in the product -- ``Juliusruh``, ``DOB`` -- so without a registry
    the geometry is unavailable and ``path_of`` is meaningless. Unlike a
    serendipitous chirp product, both names are real and resolvable.
    """
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as fh:
        missing = [k for k in REQUIRED if k not in fh]
        if missing:
            raise ValueError(
                f"{path}: not a digisonde ionogram; missing {', '.join(missing)}")
        kind = _text(fh["type"][()]) if "type" in fh else ""
        if kind and kind != TYPE_MARKER:
            raise ValueError(f"{path}: type is {kind!r}, expected {TYPE_MARKER!r}")

        snr_shape = fh["SNR"].shape
        freqs_hz = np.asarray(fh["freqs"][()], dtype=np.float64)
        ranges_m = np.asarray(fh["ranges"][()], dtype=np.float64)
        t0 = float(fh["t0"][()])
        offset_us = float(fh["offset_us"][()]) if "offset_us" in fh else 0.0
        tx_name = _text(fh["transmitter"][()])
        rx_name = _text(fh["receiver"][()])

    registry = stations or {}
    tx_lat, tx_lon = registry.get(tx_name, (np.nan, np.nan))
    rx_lat, rx_lon = registry.get(rx_name, (np.nan, np.nan))

    absolute_km = ranges_m / 1e3 + float(offset_us) * 1e-6 * C_KM_S
    freq_step_hz = (float(np.median(np.diff(freqs_hz)))
                    if freqs_hz.size > 1 else 0.0)

    return DigisondeHeader(
        path=path,
        format="digisonde",
        tx_name=tx_name, tx_latitude=tx_lat, tx_longitude=tx_lon,
        rx_name=rx_name, rx_latitude=rx_lat, rx_longitude=rx_lon,
        t0=t0,
        offset_us=offset_us,
        rate=float("nan"),
        sample_rate=SAMPLE_RATE_HZ,
        dec=1,
        # Reconstructed so `render` and `saoxml` have the band to report; the
        # product stores the axis, not the settings that produced it.
        cf=float(np.median(freqs_hz)) if freqs_hz.size else float("nan"),
        dur=float(freqs_hz.size) * freq_step_hz / SAMPLE_RATE_HZ,
        rmin=int(round(float(absolute_km.min()))) if absolute_km.size else 0,
        rmax=int(round(float(absolute_km.max()))) if absolute_km.size else 0,
        n_pol=int(snr_shape[0]) if len(snr_shape) == 3 else 1,
    )


def load(path: str | Path,
         gate_km: tuple[float, float] | None = None,
         *,
         stations: Mapping[str, tuple[float, float]] | None = None,
         pol: int | str = SUM,
         header: DigisondeHeader | None = None) -> Ionogram:
    """Read a ``digisonde_ionogram-*.h5`` into an :class:`Ionogram`.

    ``pol`` selects a polarization channel, or :data:`SUM` to add both. The
    file does not identify which channel is O and which is X, so this is a
    channel index and nothing more -- see the module docstring.

    Unlike ``io_chirp``, ``gate_km`` may narrow freely: the product stores the
    whole unambiguous range window (``c`` times the inter-pulse period, 2997 km
    at the usual 10 ms), so nothing was discarded at acquisition and there is
    no wider extent to warn about.
    """
    import h5py

    path = Path(path)
    header = header or read_header(path, stations)

    with h5py.File(path, "r") as fh:
        snr = np.asarray(fh["SNR"][()], dtype=np.float64)
        freqs_hz = np.asarray(fh["freqs"][()], dtype=np.float64)
        ranges_m = np.asarray(fh["ranges"][()], dtype=np.float64)

    if snr.ndim != 3:
        raise ValueError(f"{path}: SNR has shape {snr.shape}, expected 3-D "
                         f"(polarization, frequency, range)")
    if snr.shape[1:] != (freqs_hz.size, ranges_m.size):
        raise ValueError(
            f"{path}: SNR is {snr.shape} but the axes are {freqs_hz.size} "
            f"frequencies x {ranges_m.size} ranges")

    # Below-threshold cells are noise, not gaps. Zero SNR is the median noise
    # level, which is what "nothing above the floor" means on this scale.
    snr = np.nan_to_num(snr, nan=0.0, posinf=0.0, neginf=0.0)

    if pol == SUM:
        # Adding SNR before the +1 in `snr_to_power` keeps the noise floor
        # where it belongs: two channels at noise sum to 0, not to 1.
        combined = snr.sum(axis=0)
    else:
        index = int(pol)
        if not 0 <= index < snr.shape[0]:
            raise ValueError(f"{path}: polarization {index} out of range; the "
                             f"file stores {snr.shape[0]}")
        combined = snr[index]

    if not header.is_oblique:
        warnings.warn(
            f"{path.name}: {header.tx_name} transmits and receives, which is a "
            f"digisonde sounding itself rather than a reception of one. These "
            f"ranges are one-way and will read 2x a vertical .lfs sounding.",
            stacklevel=2,
        )

    absolute_km = ranges_m / 1e3 + header.range_offset_km

    # This pipeline's range axis descends, bin 0 holding the largest virtual
    # range; the stored axis ascends. See the `calibrate` module docstring for
    # the sign error that convention exists to prevent.
    vrange_km = absolute_km[::-1]
    power = snr_to_power(combined)[:, ::-1]

    stored_lo, stored_hi = float(vrange_km.min()), float(vrange_km.max())
    lo, hi = stored_lo, stored_hi

    want = gate_km
    if want is None:
        # The stored extent is `c` times the inter-pulse period -- a property
        # of the transmitter's pulse timing, not of this path. On an 864 km
        # circuit it reaches 3597 km while no echo can arrive before 998, and
        # the near third of that window is where the interference sits. Ask
        # the geometry instead, and fall back to the stored extent only when
        # the geometry is unavailable.
        want = calibrate.geometry_gate(header)

    if want is not None:
        want_lo, want_hi = float(want[0]), float(want[1])
        keep = (vrange_km >= want_lo) & (vrange_km <= want_hi)
        if not keep.any():
            if gate_km is not None:
                raise ValueError(
                    f"{path}: gate {want_lo:.0f}-{want_hi:.0f} km does not "
                    f"overlap the stored {stored_lo:.0f}-{stored_hi:.0f} km")
            # A *derived* gate that misses entirely means the geometry and the
            # range zero disagree -- most likely `offset_us`. Say so and keep
            # the data rather than raising on a file the caller never gated.
            warnings.warn(
                f"{path.name}: no stored range falls in the "
                f"{want_lo:.0f}-{want_hi:.0f} km window this "
                f"{header.tx_name}->{header.rx_name} path allows; keeping the "
                f"stored {stored_lo:.0f}-{stored_hi:.0f} km. Check `offset_us` "
                f"-- the range zero is configured, not measured.",
                stacklevel=2,
            )
        else:
            vrange_km = vrange_km[keep]
            power = power[:, keep]
            lo, hi = float(vrange_km.min()), float(vrange_km.max())

    freqs_mhz = freqs_hz / 1e6
    cal = _build_calibration(freqs_mhz, vrange_km, (lo, hi), absolute_km)

    return Ionogram(
        power=power.astype(np.float32),
        cal=cal,
        header=header,
        window=cal.n_range_full,
        zero_periods=0,
    )


def _build_calibration(freqs_mhz: np.ndarray, vrange_km: np.ndarray,
                       gate_km: tuple[float, float],
                       full_km: np.ndarray) -> Calibration:
    """Axes straight from the file's own ``freqs`` and ``ranges``.

    Not via ``calibrate.build``, for the same reason ``io_chirp`` does not:
    that derives axes from header arithmetic, and here the stored vectors are
    the only authority for what each row means.
    """
    step = (float(np.median(np.abs(np.diff(vrange_km))))
            if vrange_km.size > 1 else 1.0)
    freq_step = float(np.median(np.diff(freqs_mhz))) if freqs_mhz.size > 1 else 0.0

    # Bin edges, matching calibrate.build, where labels are bin centres.
    freq_start = float(freqs_mhz[0]) - freq_step / 2.0
    freq_stop = float(freqs_mhz[-1]) + freq_step / 2.0

    n_range_full = int(full_km.size)
    # Where the kept slice sits on the stored axis, in descending order.
    descending = full_km[::-1]
    i_lo = int(np.searchsorted(-descending, -float(vrange_km[0]), side="left"))
    i_hi = i_lo + vrange_km.size - 1

    return Calibration(
        freq=freqs_mhz,
        vrange=vrange_km,
        freq_start=freq_start,
        freq_stop=freq_stop,
        # The product records no intended stop, so a truncated sounding is
        # indistinguishable from a complete one. Same limitation as chirp2
        # without `nominal_stop_mhz`.
        freq_stop_nominal=freq_stop,
        half_span=float(full_km.max() - full_km.min()) / 2.0,
        range_step=step,
        # Pulse compression, no zero padding: the bin spacing is the true
        # resolution.
        resolution_km=step,
        gate_km=gate_km,
        gate_idx=(i_lo, i_hi),
        n_range_full=n_range_full,
    )


def find_digisonde(target) -> list[Path]:
    """Return the ``digisonde_ionogram-*.h5`` files under ``target``, sorted.

    Mirrors :func:`muf.io_chirp.find_h5`, and is as deliberately specific: the
    same directory carries ``lfm_ionogram-*.h5`` chirp products and
    ``chirp-*.h5`` / ``par-*.h5`` / ``cdetections-*.h5`` detection files, none
    of which share this schema.
    """
    if isinstance(target, (str, Path)):
        targets = [Path(target)]
    else:
        targets = [Path(t) for t in target]
    if not targets:
        raise FileNotFoundError("no target given")

    found: list[Path] = []
    for item in targets:
        if item.is_dir():
            found.extend(item.rglob(f"{FILE_PREFIX}*.h5"))
        elif item.name.startswith(FILE_PREFIX) and item.suffix == ".h5":
            found.append(item)
    if not found:
        raise FileNotFoundError(f"no {FILE_PREFIX}*.h5 under {targets[0]}")
    return sorted(found, key=lambda p: p.name)
