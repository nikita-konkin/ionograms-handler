"""Keeping the echo window of the dechirped voltage.

A power ionogram is ``|FFT|^2``. It keeps where the echoes are and how strong,
and throws away everything an HF *channel* is made of on the fast scale: the
Doppler shift and spread of each mode, its fading, the phase that carries
group-delay dispersion, o/x splitting. A channel simulator built from oblique
soundings needs exactly that part, and today the only place it survives is the
raw ``.lfs`` capture -- 80 MB per sounding, 23 GB per station-day.

Almost none of those 80 MB is echo. After dechirping, an echo at virtual range
``r`` is a tone at a beat frequency proportional to ``r``, and every mode on a
circuit sits within a few hundred km of range: on cyprus1 -> Yoshkar-Ola, 1F
to 3F is 9.0-10.6 ms, a beat band a few hundred Hz wide out of the 40 kHz
stored. Shifting that band to zero, low-passing it and decimating keeps
**everything the echoes contain, losslessly within the band**, in 1.6 MB
(measured on cyprus1, ``docs/echo-window.md`` sec. 6.2). With the power
product beside it the ``.lfs`` is no longer needed for simulator work.

``docs/echo-window.md`` is the specification -- the one source of truth, which
ionozond's ``h5_archive.py`` will implement for ``.lfs`` too. This module is
the reference implementation. It is **additive**: a root-level group,
``echo_window``, that no upstream dataset can collide with (v2's schema is flat
at the root), and nothing in any existing product changes or loads
differently.

Four decisions carry most of the weight:

**The window follows the energy, not the geometry.** ``architecture.md``
records a 45 ms epoch error that displaced an echo by 13,500 km. A window
computed from the path alone would have kept an empty band and discarded the
sounding's only echo, irreversibly once the raw voltage is dropped. So the
window is the *union* of :func:`muf.calibrate.auto_gate` on the product's own
power ionogram and the 1-3 hop geometry, never the geometry alone.

**The filter is alias-free, not merely image-free.** Its stopband begins
inside the output Nyquist, so nothing folds into the stored band at all. That
is what makes :func:`reconstruct` equal to the source passed twice through the
same filter, and so a round trip that can be tested to 1e-3 rather than
argued about.

**Zero-phase, sample-aligned.** The filter is applied with
``scipy.signal.resample_poly``, which compensates its group delay: stored
sample ``k`` *is* source sample ``k * decimation``, in time and in sweep
frequency. Nothing downstream has to know the filter length to put a sample on
the sweep.

**Interference stays.** Broadcast carriers sweep through the beat band after
dechirping; they are what the receiver heard, and a simulator of this channel
has to see them.

The noise the 43 dB threshold is measured against is the *full-band* row
median, which a narrow window cannot reproduce from its own samples. v2
products already store it (``noise_floor``); for ``.lfs`` the group carries it
as ``row_noise``. With it, an ionogram re-derived from the window lands on the
pipeline's own dB scale.
"""

from __future__ import annotations

import contextlib
import functools
import json
import math
import os
import shutil
import subprocess
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

from . import calibrate, io_chirp, io_lfs, spectro
from .spectro import NOISE_COEF, Ionogram

# --- the format, version 1 ---------------------------------------------------

#: The root-level group. v2 writes every dataset at the root, so a group of
#: our own cannot collide with anything upstream adds.
GROUP = io_chirp.ECHO_WINDOW_GROUP
FORMAT = "echo_window"
VERSION = 1

#: Complex64 samples of the window, ``[n]``.
SAMPLES = "z"
#: The filter's taps, float64 ``[numtaps]``, exactly as the cut applied them.
#: Stored rather than regenerated from the design parameters: a reader -- this
#: one, or ionozond's -- then needs no FIR designer that agrees with
#: ``scipy.signal.firwin`` to the last bit, and the ``filter`` JSON's rounded
#: ``kaiser_beta`` cannot drift from what was used. 29 KB at 3595 taps.
TAPS = "taps"
#: ``.lfs`` only: the full-band noise per spectrogram row; see the docstring.
ROW_NOISE = "row_noise"
#: ``.lfs`` only: the recording's 512-byte header, verbatim.
LFS_HEADER = "lfs_header"
#: Reserved for the derived fast-part parameters of a later task. Nothing
#: here writes or reads it; :func:`write` refuses a group that already has it
#: only in the sense that it never touches an existing group.
PARAMS = "params"

SOURCE_V2 = "v2_z"
SOURCE_LFS = "lfs"

#: A standalone window for an ``.lfs`` capture is ``<stem>.echo.h5``.
STANDALONE_SUFFIX = ".echo.h5"

# --- selection ---------------------------------------------------------------

#: Range added to each side of the union of the candidate windows, km. One
#: sporadic-E leg or a second-hop flank just outside the found trace is worth
#: more than the bytes it costs, and the window is irreversible once the raw
#: voltage is gone.
DEFAULT_MARGIN_KM = 150.0

#: Hops the geometry candidate leaves room for, as ``calibrate.geometry_gate``.
DEFAULT_MAX_HOPS = calibrate.DEFAULT_MAX_HOPS

#: The pipeline's detection threshold, ``extractors.contour.DEFAULT_THRESHOLD_DB``
#: -- repeated rather than imported, so cutting a window does not load OpenCV.
#: An edge of the window brighter than this is flagged as a possible truncation.
DETECTION_DB = 43.0

#: Where a median-noise cell lands on the shared dB scale (25.571 dB).
NOISE_DB = float(spectro.to_db(np.array([1.0 / NOISE_COEF]))[0])

#: The ``.lfs`` spectrogram window the selection ionogram and ``row_noise`` are
#: computed at -- the pipeline's default, so ``row_noise`` is exactly the
#: floor ``spectro.compute`` divides by.
LFS_NOISE_WINDOW = spectro.DEFAULT_WINDOW

# --- band and filter ---------------------------------------------------------

#: Kept either side of the beat band so a Doppler-shifted mode stays in it.
DOPPLER_MARGIN_HZ = 20.0

#: Transition width per side, as a fraction of the kept band, with a floor.
#: Narrower costs taps linearly; wider costs output rate. 10 % puts a 1000 km
#: window at 100 kHz/s near 5000 taps at the 40 kHz source rate.
TRANSITION_FRACTION = 0.10
MIN_TRANSITION_HZ = 10.0

#: Output rate over the band including both transitions. Above 1.0 is what
#: keeps the stopband inside the output Nyquist -- see the module docstring.
RATE_HEADROOM = 1.25

#: Required filter performance, *measured* on the designed taps, not assumed.
STOPBAND_DB = 80.0
MAX_RIPPLE_DB = 0.01

#: Kaiser's formula is an estimate; designing a few dB past the requirement
#: lets the measured attenuation clear it on the first try.
DESIGN_MARGIN_DB = 3.0

# --- storage -----------------------------------------------------------------

#: Largest chunk, samples: 512 KB of complex64. The record is split into
#: equal chunks no longer than this, because HDF5 allocates a partial last
#: chunk in full -- 64k-sample chunks put 0.34 MB of nothing into every
#: 1.78 MB cyprus1 window.
CHUNK_SAMPLES = 65536

#: HDF5 filter for ``z``, with byte shuffle. gzip because every HDF5 build
#: and every reader has it. Measured 2026-10-15 on the five 2026-02-04
#: cyprus1 windows: 14.4-14.9 % saved -- the exponent bytes of complex noise,
#: the mantissas being incompressible -- for one standard filter on read.
COMPRESSION: str | None = "gzip"
COMPRESSION_LEVEL = 4


class NoEchoError(ValueError):
    """Nothing in the product says where an echo is, so there is no window."""


# --- sources -----------------------------------------------------------------

@dataclass(frozen=True)
class Source:
    """Where the samples come from, and how their frequency maps to range.

    ``range_km = range_zero_km + km_per_hz * beat_hz``, with ``beat_hz`` the
    frequency *of the samples as stored on disk*. Each format keeps its own
    conventions, because the window's ranges must be on the axis
    ``loader.load`` gives that product:

    * v2 forms its spectrogram on ``conj(z)`` (``calc_ionograms.py``), so an
      echo is a tone at the *negative* of its beat on disk, and ranges use v2's
      exact ``c``. ``range_zero_km`` is the direct-path offset ``t0`` implies,
      or 0 on a relative axis.
    * ``.lfs`` maps beat to range through ``calibrate``'s descending axis,
      ``-f * 2c / (div_coef * rate)`` with ``c = 3e5 km/s``.
    """

    kind: str
    path: Path
    header: object
    sr_hz: float
    rate_hz_s: float
    t0: float
    sweep_hz_at_0: float
    km_per_hz: float
    range_zero_km: float
    relative: bool
    n_samples: int
    #: The source's own spectrogram: v2's ``fftlen`` and ``fft_step``; for
    #: ``.lfs`` the ``row_noise`` window, with no overlap.
    window: int
    step: int

    def beat_hz(self, range_km: float) -> float:
        return (float(range_km) - self.range_zero_km) / self.km_per_hz

    def range_at(self, beat_hz: float) -> float:
        return self.range_zero_km + self.km_per_hz * float(beat_hz)

    @property
    def axis_km(self) -> tuple[float, float]:
        """The range span the samples can represent: beats within +/-sr/2."""
        ends = sorted((self.range_at(-self.sr_hz / 2), self.range_at(self.sr_hz / 2)))
        return ends[0], ends[1]


def _v2_source(path: Path, stations) -> Source:
    header = io_chirp.read_header(path, stations)
    with io_chirp.open_h5(path) as fh:
        freqs = np.asarray(fh["freqs"][()], dtype=np.float64)
        ranges = np.asarray(fh["ranges"][()], dtype=np.float64)
        n = (int(fh[io_chirp.RAW_VOLTAGE_KEY].shape[0])
             if io_chirp.RAW_VOLTAGE_KEY in fh else 0)
    sr, rate = header.sample_rate, header.rate
    # `fftlen` from the stored range spacing, `fft_step` from the frequency
    # spacing -- the same reconstruction `io_chirp.reprocess` makes.
    dr = float(np.median(np.abs(np.diff(ranges)))) if ranges.size > 1 else 0.0
    window = round((io_chirp.C_M_S / rate) * sr / dr) if dr > 0 else 0
    step = (round(float(np.median(np.diff(freqs))) / rate * sr)
            if freqs.size > 1 else 0)
    zero, reason = io_chirp._absolute_ranges(
        np.zeros(1), header.t0, header.range_offset_applied,
        header.range_start_m, header.tx_name)
    return Source(
        kind=SOURCE_V2, path=path, header=header, sr_hz=sr, rate_hz_s=rate,
        t0=header.t0, sweep_hz_at_0=float(freqs[0]),
        km_per_hz=-io_chirp.C_M_S / rate / 1e3,
        range_zero_km=float(zero[0]), relative=bool(reason),
        n_samples=n, window=int(window), step=int(step))


def _lfs_source(path: Path, stations) -> Source:
    header = io_lfs.read_header(path, stations)
    return Source(
        kind=SOURCE_LFS, path=path, header=header,
        sr_hz=header.sample_rate / header.dec, rate_hz_s=float(header.rate),
        t0=float(header.start_epoch),
        sweep_hz_at_0=float(header.cf - header.sample_rate / 2),
        km_per_hz=-calibrate.C_KM_S * 2.0 / (header.div_coef * header.rate),
        range_zero_km=0.0, relative=False,
        n_samples=io_lfs.n_samples(path),
        window=LFS_NOISE_WINDOW, step=LFS_NOISE_WINDOW)


def source_of(path: str | Path,
              stations: Mapping[str, tuple[float, float]] | None = None) -> Source:
    """The :class:`Source` for an ``.lfs`` capture or a v2 product."""
    from . import loader

    path = Path(path)
    fmt = loader.format_of(path)
    if fmt == loader.LFS:
        return _lfs_source(path, stations)
    if fmt == loader.CHIRP2:
        return _v2_source(path, stations)
    raise ValueError(f"{path}: an echo window is cut from an .lfs capture or a "
                     f"v2 lfm_ionogram product, not from {fmt}")


def _source_samples(source: Source) -> np.ndarray:
    if source.kind == SOURCE_V2:
        return io_chirp.read_raw_voltage(source.path)
    return io_lfs.read_iq(source.path)


def _power_ionogram(source: Source) -> Ionogram:
    """The ionogram the window is selected on: the product's own power.

    v2's stored ``SNR``, read as every other consumer reads it. For ``.lfs``,
    ``spectro.compute`` over the *whole* stored axis -- the default gate is
    built from the header geometry, which is exactly what selection must not
    depend on alone.
    """
    if source.kind == SOURCE_V2:
        return io_chirp.load(source.path, header=source.header)
    half = calibrate.range_half_span(source.header)
    return spectro.compute(source.path, window=LFS_NOISE_WINDOW,
                           gate_km=(-half, half), header=source.header)


def _lfs_row_noise(iq: np.ndarray, window: int) -> np.ndarray:
    """``spectro.compute``'s per-row floor, ``NOISE_COEF * median |FFT|^2``.

    Over the full spectrum, before any gate -- that is what the floor of every
    ``.lfs`` ionogram is, and what a window cannot recompute from itself.
    """
    n_rows = len(iq) // window
    win = spectro.hanning(window)
    out = np.empty(n_rows, dtype=np.float64)
    for i in range(n_rows):
        spectrum = np.abs(np.fft.fft(win * iq[i * window:(i + 1) * window])) ** 2
        out[i] = NOISE_COEF * float(np.median(spectrum))
    return out


# --- selection ---------------------------------------------------------------

@dataclass(frozen=True)
class Selection:
    """The range window and how it was arrived at."""

    lo_km: float
    hi_km: float
    method: str
    margin_km: float
    max_hops: int
    auto_km: tuple[float, float] | None
    geometry_km: tuple[float, float] | None
    manual_km: tuple[float, float] | None
    edge_energy_db: tuple[float, float]
    truncated_suspect: bool

    def as_dict(self) -> dict:
        def pair(v):
            return None if v is None else [round(float(v[0]), 3), round(float(v[1]), 3)]

        return {"method": self.method, "margin_km": self.margin_km,
                "max_hops": self.max_hops, "auto_gate_km": pair(self.auto_km),
                "geometry_km": pair(self.geometry_km),
                "manual_km": pair(self.manual_km)}


def _edge_db(ion: Ionogram, lo: float, hi: float) -> float:
    vr = np.asarray(ion.vrange)
    cols = (vr >= lo) & (vr <= hi)
    if not cols.any():
        return float("nan")
    return float(np.max(np.asarray(ion.db)[:, cols]))


def select_window(source: Source, ion: Ionogram | None = None, *,
                  margin_km: float = DEFAULT_MARGIN_KM,
                  max_hops: int = DEFAULT_MAX_HOPS,
                  manual_km: tuple[float, float] | None = None) -> Selection:
    """The range window to keep.

    The union of where the power ionogram's energy is (``auto_gate``) and
    where the 1..``max_hops`` geometry puts the modes, plus ``margin_km``, clipped
    to what the samples can represent. A relative axis -- a search-mode
    ``unkown`` product -- has no geometry to offer, so there it is
    ``auto_gate`` alone. ``manual_km`` replaces both, margin included.

    Either way, energy at the detection threshold within ``margin_km`` of an
    edge marks the window ``truncated_suspect``: something bright the
    candidates did not account for is close enough to be cut.

    Raises :class:`NoEchoError` when neither candidate exists: a sounding of
    pure noise with no known path has no window, and inventing one would be
    the irreversible kind of wrong.
    """
    ion = ion if ion is not None else _power_ionogram(source)
    auto = geometry = None
    if manual_km is not None:
        candidates = [tuple(float(v) for v in manual_km)]
        method = "manual"
    else:
        auto = calibrate.auto_gate(ion.power, ion.vrange, margin_km=0.0)
        if not source.relative:
            geometry = calibrate.geometry_gate(source.header, max_hops=max_hops,
                                               margin_km=0.0)
        candidates = [c for c in (auto, geometry) if c is not None]
        method = "+".join(name for name, c in (("auto_gate", auto),
                                               ("geometry", geometry))
                          if c is not None)
        if not candidates:
            why = ("its axis is relative, so geometry cannot place it"
                   if source.relative else "its path has no usable geometry")
            raise NoEchoError(
                f"{source.path.name}: no range stands out in the power "
                f"ionogram and {why} -- nothing to keep")

    # A manual range is kept as given: whoever typed it chose its edges.
    pad = 0.0 if manual_km is not None else float(margin_km)
    lo = min(c[0] for c in candidates) - pad
    hi = max(c[1] for c in candidates) + pad
    axis_lo, axis_hi = source.axis_km
    lo, hi = max(lo, axis_lo), min(hi, axis_hi)
    if lo >= hi:
        raise NoEchoError(f"{source.path.name}: window {lo:.0f}..{hi:.0f} km "
                          f"lies outside the {axis_lo:.0f}..{axis_hi:.0f} km "
                          f"the samples represent")

    edge = max(float(margin_km), abs(float(ion.cal.range_step)))
    edges = (_edge_db(ion, lo, lo + edge), _edge_db(ion, hi - edge, hi))
    truncated = any(math.isfinite(e) and e >= DETECTION_DB for e in edges)
    return Selection(lo_km=float(lo), hi_km=float(hi), method=method,
                     margin_km=float(margin_km), max_hops=int(max_hops),
                     auto_km=auto, geometry_km=geometry,
                     manual_km=(None if manual_km is None
                                else (float(manual_km[0]), float(manual_km[1]))),
                     edge_energy_db=edges, truncated_suspect=truncated)


# --- band and filter ---------------------------------------------------------

@dataclass(frozen=True)
class Plan:
    """The shift, the decimation and the filter for one window."""

    source_sr_hz: float
    shift_hz: float
    decimation: int
    passband_hz: float          # half-width kept at full gain, Hz
    stopband_hz: float          # where >= STOPBAND_DB attenuation begins, Hz
    numtaps: int
    beta: float
    cutoff_hz: float
    attenuation_db: float       # measured on the taps
    ripple_db: float            # measured on the taps
    #: Taps read from a file. ``None`` while planning, when they are designed.
    stored: np.ndarray | None = field(default=None, compare=False, repr=False)

    @property
    def sr_hz(self) -> float:
        return self.source_sr_hz / self.decimation

    @property
    def lead(self) -> int:
        """Window samples stored *before* source sample 0.

        The filter answers the first source sample half its length early and
        the last one half its length late. Keeping those tails -- some 40
        samples a side at cyprus1's rates -- is what makes the round trip exact
        to the ends of the record rather than only inside it: dropped, the
        first and last 45 ms reconstruct 1e-2 wrong.
        """
        if self.numtaps <= 1:
            return 0
        return -(-((self.numtaps - 1) // 2) // self.decimation)

    def n_out(self, n_source: int) -> int:
        """How many samples :func:`cut` makes of ``n_source``."""
        if self.numtaps <= 1:
            return n_source
        padded = self.lead * self.decimation + n_source + self._tail
        return -(-padded // self.decimation)

    @property
    def _tail(self) -> int:
        return (self.numtaps - 1) // 2 + self.decimation

    def taps(self) -> np.ndarray:
        """The FIR: as stored, for a window read from a file; else designed."""
        if self.stored is not None:
            return self.stored
        return _design(self.numtaps, self.beta, self.cutoff_hz, self.source_sr_hz)

    def filter_json(self) -> str:
        return json.dumps({
            "type": "FIR, Kaiser window, linear phase, applied zero-phase "
                    "polyphase (scipy.signal.resample_poly)",
            "numtaps": self.numtaps,
            "kaiser_beta": round(self.beta, 6),
            "cutoff_hz": self.cutoff_hz,
            "passband_edge_hz": self.passband_hz,
            "stopband_edge_hz": self.stopband_hz,
            "stopband_attenuation_db": round(self.attenuation_db, 3),
            "passband_ripple_db": round(self.ripple_db, 6),
            "gain": "unit DC gain (sum of taps = 1): noise power per Hz preserved",
            "doppler_margin_hz": DOPPLER_MARGIN_HZ,
        }, sort_keys=True)

    @classmethod
    def from_filter_json(cls, text: str, *, source_sr_hz: float, shift_hz: float,
                         decimation: int, taps: np.ndarray | None = None) -> Plan:
        spec = json.loads(text)
        return cls(source_sr_hz=float(source_sr_hz), shift_hz=float(shift_hz),
                   decimation=int(decimation),
                   passband_hz=float(spec["passband_edge_hz"]),
                   stopband_hz=float(spec["stopband_edge_hz"]),
                   numtaps=int(spec["numtaps"]), beta=float(spec["kaiser_beta"]),
                   cutoff_hz=float(spec["cutoff_hz"]),
                   attenuation_db=float(spec["stopband_attenuation_db"]),
                   ripple_db=float(spec["passband_ripple_db"]),
                   stored=taps)


def _design(numtaps: int, beta: float, cutoff_hz: float, sr_hz: float) -> np.ndarray:
    from scipy.signal import firwin

    if numtaps <= 1:
        return np.ones(1)
    return firwin(numtaps, cutoff_hz, window=("kaiser", beta), fs=sr_hz,
                  scale=True)


def _response(taps: np.ndarray, sr_hz: float, passband: float,
              stopband: float) -> tuple[float, float]:
    """Measured ``(stopband attenuation dB, passband ripple dB)``."""
    from scipy.signal import freqz

    nyq = sr_hz / 2.0
    lobe = 20.0 * sr_hz / max(len(taps), 1)     # dense where the worst lobe is
    stop = np.unique(np.concatenate([
        np.linspace(stopband, min(nyq, stopband + lobe), 4096),
        np.linspace(stopband, nyq, 4096)]))
    _, hs = freqz(taps, worN=stop, fs=sr_hz)
    _, hp = freqz(taps, worN=np.linspace(0.0, passband, 2048), fs=sr_hz)
    with np.errstate(divide="ignore"):
        attenuation = -float(np.max(20 * np.log10(np.abs(hs))))
        ripple = float(np.max(np.abs(20 * np.log10(np.abs(hp)))))
    return attenuation, ripple


def plan_for(source: Source, lo_km: float, hi_km: float) -> Plan:
    """Shift, decimation and filter for keeping ``lo_km..hi_km`` of ``source``.

    The kept band is the range window's beat span plus ``DOPPLER_MARGIN_HZ``
    each side; the transition sits outside it; the output rate is about
    ``RATE_HEADROOM`` times the whole, as an integer decimation of the source.
    The taps are measured after design and lengthened until they meet
    ``STOPBAND_DB`` and ``MAX_RIPPLE_DB`` -- the requirement is checked, not
    trusted to Kaiser's formula.
    """
    from scipy.signal import kaiserord

    sr = source.sr_hz
    nyq = sr / 2.0
    f1, f2 = source.beat_hz(lo_km), source.beat_hz(hi_km)
    shift = (f1 + f2) / 2.0
    passband = abs(f2 - f1) / 2.0 + DOPPLER_MARGIN_HZ
    transition = max(MIN_TRANSITION_HZ, TRANSITION_FRACTION * 2.0 * passband)
    stopband = passband + transition
    decimation = int(sr // (RATE_HEADROOM * 2.0 * stopband)) if stopband < nyq else 1

    if decimation <= 1:
        # The window is most of the band: nothing to decimate, and a filter
        # would only cut what the rate can carry anyway.
        return Plan(source_sr_hz=sr, shift_hz=shift, decimation=1,
                    passband_hz=nyq, stopband_hz=nyq, numtaps=1, beta=0.0,
                    cutoff_hz=nyq, attenuation_db=float("inf"), ripple_db=0.0)

    cutoff = (passband + stopband) / 2.0
    extra = 0.0
    for _ in range(8):
        numtaps, beta = kaiserord(STOPBAND_DB + DESIGN_MARGIN_DB + extra,
                                  transition / nyq)
        numtaps |= 1                    # odd: exactly zero-phase when centred
        taps = _design(numtaps, beta, cutoff, sr)
        attenuation, ripple = _response(taps, sr, passband, stopband)
        if attenuation >= STOPBAND_DB and ripple <= MAX_RIPPLE_DB:
            break
        extra += 3.0
    else:
        raise RuntimeError(
            f"filter design did not reach {STOPBAND_DB} dB / {MAX_RIPPLE_DB} dB "
            f"(got {attenuation:.1f} dB, {ripple:.4f} dB)")
    return Plan(source_sr_hz=sr, shift_hz=shift, decimation=decimation,
                passband_hz=passband, stopband_hz=stopband, numtaps=int(numtaps),
                beta=float(beta), cutoff_hz=cutoff, attenuation_db=attenuation,
                ripple_db=ripple)


# --- cutting and reconstructing ----------------------------------------------

def _mix(x: np.ndarray, freq_hz: float, sr_hz: float,
         chunk: int = 1 << 20) -> np.ndarray:
    """``x[m] * exp(j 2 pi freq m / sr)`` in complex128, chunked.

    The phase is reduced to whole cycles *before* the exponential: across ten
    million samples ``freq * m / sr`` reaches 10^5 cycles, and taking the
    exponential of that directly spends float64's precision on the integer
    part.
    """
    x = np.asarray(x)
    out = np.empty(len(x), dtype=np.complex128)
    cycles_per_sample = float(freq_hz) / float(sr_hz)
    for start in range(0, len(x), chunk):
        m = np.arange(start, min(start + chunk, len(x)), dtype=np.float64)
        phase = np.mod(cycles_per_sample * m, 1.0)
        out[start:start + len(m)] = x[start:start + len(m)] * np.exp(2j * np.pi * phase)
    return out


def cut(samples: np.ndarray, plan: Plan) -> np.ndarray:
    """``z_win[k] = decimate(z_src[m] * exp(-j 2 pi shift m / sr_src))``.

    Applied zero-phase, so stored sample ``k`` sits at source sample
    ``(k - plan.lead) * D`` exactly; the source is taken as zero outside the
    record, and the filter's tails either side of it are kept.
    """
    from scipy.signal import resample_poly

    shifted = _mix(samples, -plan.shift_hz, plan.source_sr_hz)
    if plan.numtaps <= 1:
        return shifted.astype(np.complex64)
    d = plan.decimation
    padded = np.concatenate([np.zeros(plan.lead * d, shifted.dtype), shifted,
                             np.zeros(plan._tail, shifted.dtype)])
    return resample_poly(padded, 1, d, window=plan.taps()).astype(np.complex64)


def reconstruct(window: EchoWindow) -> np.ndarray:
    """The window back at the source rate, complex128, ``source_samples`` long.

    Interpolates with the *same* filter and shifts back. Because the design is
    alias-free this equals the source passed through the filter twice -- once
    on the way in, once on the way out -- which is what the round-trip test
    holds it to.
    """
    from scipy.signal import resample_poly

    plan = window.plan
    z = np.asarray(window.samples, dtype=np.complex128)
    if plan.numtaps <= 1:
        up = z
    else:
        up = resample_poly(z, plan.decimation, 1, window=plan.taps())
    # `t_first_s` says where stored sample 0 is; nothing else is needed to
    # put the rest back on the source's sample grid.
    start = round(-window.t_first_s * window.source_sr_hz)
    up = up[start:start + window.source_samples]
    if len(up) < window.source_samples:
        up = np.concatenate([up, np.zeros(window.source_samples - len(up), up.dtype)])
    return _mix(up, plan.shift_hz, plan.source_sr_hz)


# --- reading -----------------------------------------------------------------

def _str(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _str(value[()])
    return str(value)


@dataclass(frozen=True)
class EchoWindow:
    """One stored window: the samples and every attribute, typed."""

    path: Path
    samples: np.ndarray | None
    n: int
    format: str
    version: int
    source: str
    source_name: str
    source_samples: int
    source_sr_hz: float
    sr_hz: float
    decimation: int
    shift_hz: float
    range_lo_km: float
    range_hi_km: float
    range_axis: str
    t0: float
    t_first_s: float
    sweep_hz_at_t_first: float
    rate_hz_s: float
    km_per_hz: float
    range_zero_km: float
    source_window: int
    source_step: int
    filter: dict
    selection: dict
    edge_energy_db: tuple[float, float]
    truncated_suspect: bool
    writer: str
    row_noise: np.ndarray | None = None
    noise_window: int = 0
    lfs_header: bytes | None = None
    taps: np.ndarray | None = None

    @property
    def plan(self) -> Plan:
        return Plan.from_filter_json(json.dumps(self.filter),
                                     source_sr_hz=self.source_sr_hz,
                                     shift_hz=self.shift_hz,
                                     decimation=self.decimation,
                                     taps=self.taps)

    @property
    def nbytes(self) -> int:
        return self.n * np.dtype(np.complex64).itemsize

    @property
    def sweep_hz_at_t0(self) -> float:
        """Sweep frequency at source sample 0, the source's own first row."""
        return self.sweep_hz_at_t_first - self.rate_hz_s * self.t_first_s

    def times_s(self) -> np.ndarray:
        """Each sample's time after ``t0``, s."""
        return self.t_first_s + np.arange(self.n) / self.sr_hz

    def sweep_hz(self) -> np.ndarray:
        """Each sample's sweep frequency: the dechirp reference at that time."""
        return self.sweep_hz_at_t_first + self.rate_hz_s * (np.arange(self.n)
                                                            / self.sr_hz)

    def describe(self) -> str:
        line = (f"echo window {self.range_lo_km:.0f}-{self.range_hi_km:.0f} km "
                f"({self.range_axis}), {self.sr_hz:.1f} Hz (/{self.decimation}), "
                f"{self.nbytes / 1e6:.2f} MB")
        if self.truncated_suspect:
            lo, hi = self.edge_energy_db
            line += (f" -- TRUNCATION SUSPECTED: {lo:.1f} / {hi:.1f} dB at the "
                     f"low / high edge, over the {DETECTION_DB:.0f} dB threshold")
        return line


def has_window(path: str | Path) -> bool:
    import h5py

    try:
        with h5py.File(path, "r") as fh:
            return GROUP in fh
    except OSError:
        return False


def read(path: str | Path, *, samples: bool = True) -> EchoWindow:
    """Read the ``echo_window`` group of ``path``. ``samples=False`` skips ``z``."""
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as fh:
        if GROUP not in fh:
            raise ValueError(f"{path}: no {GROUP!r} group")
        g = fh[GROUP]
        a = g.attrs
        if _str(a.get("format", "")) != FORMAT:
            raise ValueError(f"{path}: {GROUP!r} is not format {FORMAT!r}")
        version = int(a["version"])
        if version != VERSION:
            raise ValueError(f"{path}: echo window version {version}; this "
                             f"reader understands {VERSION}")
        data = np.asarray(g[SAMPLES][()], dtype=np.complex64) if samples else None
        n = int(g[SAMPLES].shape[0])
        row_noise = noise_window = None
        if ROW_NOISE in g:
            row_noise = np.asarray(g[ROW_NOISE][()], dtype=np.float64)
            noise_window = int(g[ROW_NOISE].attrs["window"])
        header = (bytes(np.asarray(g[LFS_HEADER][()], dtype=np.uint8))
                  if LFS_HEADER in g else None)
        if TAPS not in g:
            raise ValueError(f"{path}: echo window has no {TAPS!r}; v1 stores "
                             f"the filter it was cut with")
        taps = np.asarray(g[TAPS][()], dtype=np.float64)
        numtaps = int(json.loads(_str(a["filter"]))["numtaps"])
        if taps.shape != (numtaps,):
            raise ValueError(f"{path}: {len(taps)} stored taps, filter "
                             f"records {numtaps}")
        edge = np.asarray(a["edge_energy_db"], dtype=np.float64)
        return EchoWindow(
            path=path, samples=data, n=n, format=FORMAT, version=version,
            source=_str(a["source"]), source_name=_str(a["source_name"]),
            source_samples=int(a["source_samples"]),
            source_sr_hz=float(a["source_sr_hz"]), sr_hz=float(a["sr_hz"]),
            decimation=int(a["decimation"]), shift_hz=float(a["shift_hz"]),
            range_lo_km=float(a["range_lo_km"]), range_hi_km=float(a["range_hi_km"]),
            range_axis=_str(a["range_axis"]), t0=float(a["t0_unix_s"]),
            t_first_s=float(a["t_first_s"]),
            sweep_hz_at_t_first=float(a["sweep_hz_at_t_first"]),
            rate_hz_s=float(a["rate_hz_s"]), km_per_hz=float(a["km_per_hz"]),
            range_zero_km=float(a["range_zero_km"]),
            source_window=int(a["source_window"]), source_step=int(a["source_step"]),
            filter=json.loads(_str(a["filter"])),
            selection=json.loads(_str(a["selection"])),
            edge_energy_db=(float(edge[0]), float(edge[1])),
            truncated_suspect=bool(a["truncated_suspect"]),
            writer=_str(a["writer"]),
            row_noise=row_noise, noise_window=int(noise_window or 0),
            lfs_header=header, taps=taps)


def read_header(path: str | Path,
                stations: Mapping[str, tuple[float, float]] | None = None):
    """The source ``.lfs`` header of a standalone window, as an ``LfsHeader``."""
    window = read(path, samples=False)
    if window.lfs_header is None:
        raise ValueError(f"{path}: the echo window carries no .lfs header")
    return io_lfs.parse_header(window.lfs_header, path, stations)


# --- writing -----------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _writer() -> str:
    sha = os.environ.get("API_BUILD_SHA", "")
    if not sha or sha == "unknown":
        try:
            done = subprocess.run(
                ["git", "-C", str(Path(__file__).resolve().parent),
                 "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5)
            sha = done.stdout.strip() if done.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):
            sha = ""
    return f"ionograms-handler {sha or 'unknown'}"


def _attrs(source: Source, selection: Selection, plan: Plan) -> dict:
    t_first = -plan.lead * plan.decimation / source.sr_hz
    return {
        "format": FORMAT,
        "version": np.int64(VERSION),
        "source": source.kind,
        "source_name": source.path.name,
        "source_samples": np.int64(source.n_samples),
        "source_sr_hz": float(source.sr_hz),
        "sr_hz": float(plan.sr_hz),
        "decimation": np.int64(plan.decimation),
        "shift_hz": float(plan.shift_hz),
        "range_lo_km": float(selection.lo_km),
        "range_hi_km": float(selection.hi_km),
        "range_axis": "relative" if source.relative else "absolute",
        "t0_unix_s": float(source.t0),
        # The filter is applied zero-phase, so its group delay is already
        # compensated: stored sample 0 is source sample -lead*D exactly, the
        # start of the filter's response to the first sample.
        "t_first_s": t_first,
        "sweep_hz_at_t_first": float(source.sweep_hz_at_0
                                     + source.rate_hz_s * t_first),
        "rate_hz_s": float(source.rate_hz_s),
        "km_per_hz": float(source.km_per_hz),
        "range_zero_km": float(source.range_zero_km),
        "source_window": np.int64(source.window),
        "source_step": np.int64(source.step),
        "filter": plan.filter_json(),
        "selection": json.dumps(selection.as_dict(), sort_keys=True),
        "edge_energy_db": np.asarray(selection.edge_energy_db, dtype=np.float64),
        "truncated_suspect": np.bool_(selection.truncated_suspect),
        "writer": _writer(),
    }


def _write_group(fh, samples: np.ndarray, attrs: dict, taps: np.ndarray, *,
                 row_noise: np.ndarray | None = None,
                 noise_window: int = 0,
                 lfs_header: bytes | None = None) -> None:
    group = fh.create_group(GROUP)
    n = len(samples)
    chunk = max(1, -(-n // max(1, -(-n // CHUNK_SAMPLES))))
    group.create_dataset(SAMPLES, data=np.asarray(samples, dtype=np.complex64),
                         chunks=(chunk,), compression=COMPRESSION,
                         compression_opts=(COMPRESSION_LEVEL
                                           if COMPRESSION == "gzip" else None),
                         shuffle=COMPRESSION is not None)
    for key, value in attrs.items():
        group.attrs[key] = value
    group.create_dataset(TAPS, data=np.asarray(taps, dtype=np.float64))
    if row_noise is not None:
        noise = group.create_dataset(ROW_NOISE,
                                     data=np.asarray(row_noise, dtype=np.float64))
        noise.attrs["window"] = np.int64(noise_window)
    if lfs_header is not None:
        group.create_dataset(LFS_HEADER,
                             data=np.frombuffer(bytes(lfs_header), dtype=np.uint8))


@contextlib.contextmanager
def _staged(target: Path):
    """A temporary sibling of ``target``, renamed over it only on success.

    Same directory, so the rename is atomic on one filesystem. Anything raised
    in between -- a failed verification, an interrupted write -- leaves the
    original exactly as it was and the temporary file gone.
    """
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        yield tmp
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


@dataclass
class Verification:
    ok: bool
    problems: list
    metrics: dict

    def describe(self) -> str:
        if self.ok:
            bits = ", ".join(f"{k} {v}" for k, v in self.metrics.items())
            return f"verified ({bits})" if bits else "verified"
        return "VERIFICATION FAILED: " + "; ".join(self.problems)


@dataclass
class Report:
    source: Path
    target: Path | None
    status: str
    detail: str = ""
    selection: Selection | None = None
    plan: Plan | None = None
    stored_bytes: int = 0
    source_bytes: int = 0
    verification: Verification | None = None

    def line(self) -> str:
        bits = [f"{self.source.name}: {self.status}"]
        if self.selection is not None and self.plan is not None:
            ratio = (f", 1/{self.source_bytes / self.stored_bytes:.0f} of the source"
                     if self.stored_bytes else "")
            bits.append(
                f"{self.selection.lo_km:.0f}-{self.selection.hi_km:.0f} km "
                f"[{self.selection.method}], {self.plan.sr_hz:.1f} Hz "
                f"(/{self.plan.decimation}), {self.stored_bytes / 1e6:.2f} MB{ratio}")
            if self.selection.truncated_suspect:
                bits.append("TRUNCATION SUSPECTED")
        if self.detail:
            bits.append(self.detail)
        if self.verification is not None:
            bits.append(self.verification.describe())
        if self.target is not None and self.target != self.source:
            bits.append(f"-> {self.target}")
        return " -- ".join(bits)


def _standalone_attrs(source: Source) -> dict:
    h = source.header
    attrs = {
        "source_format": "lfs",
        "source_name": source.path.name,
        "rate_hz_s": float(h.rate),
        "sample_rate_hz": float(h.sample_rate),
        "dec": np.int64(h.dec),
        "sr_hz": float(source.sr_hz),
        "t0_unix_s": float(source.t0),
        "sweep_start_hz": float(source.sweep_hz_at_0),
        "tx_name": h.tx_name,
        "rx_name": h.rx_name,
    }
    coords = (h.tx_latitude, h.tx_longitude, h.rx_latitude, h.rx_longitude)
    if all(c is not None and math.isfinite(c) for c in coords):
        attrs.update(tx_latitude=float(coords[0]), tx_longitude=float(coords[1]),
                     rx_latitude=float(coords[2]), rx_longitude=float(coords[3]))
    return attrs


def write(path: str | Path, *,
          margin_km: float = DEFAULT_MARGIN_KM,
          max_hops: int = DEFAULT_MAX_HOPS,
          manual_km: tuple[float, float] | None = None,
          into: str | Path | None = None,
          out_dir: str | Path | None = None,
          drop_raw: bool = False,
          verify: bool = False,
          dry_run: bool = False,
          stations: Mapping[str, tuple[float, float]] | None = None) -> Report:
    """Cut and store the echo window of one source. Idempotent, per file.

    * v2 product with ``z``: the group is appended to the product. With
      ``drop_raw`` the product is rewritten *without* ``z`` -- HDF5 does not
      shrink when a dataset is deleted -- and that is never done unless the
      rewritten file verified.
    * ``.lfs``: the group goes into ``into`` (a chirpsounder2-format product
      of the same capture) or a standalone ``<stem>.echo.h5`` in ``out_dir``
      (default: beside the capture). The ``.lfs`` is never modified.

    A target that already holds a window is left alone, so running twice gives
    the same file. Every write goes through a temporary sibling and an atomic
    rename: an interruption leaves the original intact.
    """
    from . import loader

    path = Path(path)
    stations = loader.resolve_stations(stations)
    source = source_of(path, stations)
    source_bytes = path.stat().st_size

    if source.kind == SOURCE_V2:
        header = source.header
        if header.has_echo_window:
            if drop_raw and header.has_raw_voltage:
                return _drop_raw(source, None, None, None, verify=True,
                                 dry_run=dry_run, source_bytes=source_bytes)
            report = Report(path, path, "present",
                            "already has an echo window"
                            + ("" if header.has_raw_voltage
                               else "; raw z already dropped"),
                            source_bytes=source_bytes)
            if verify and header.has_raw_voltage:
                report.verification = verify_window(path, source=source)
            return report
        if not header.has_raw_voltage:
            return Report(path, None, "skipped",
                          "no `z`: save_raw_voltage was off, nothing to cut",
                          source_bytes=source_bytes)
        target = path
    else:
        if drop_raw:
            warnings.warn(f"{path.name}: --drop-raw does not apply to an .lfs "
                          f"capture, which is never modified", stacklevel=2)
        if into is not None:
            target = Path(into)
            if not target.exists():
                raise FileNotFoundError(
                    f"{target}: --into must name an existing product")
        else:
            target = Path(out_dir or path.parent) / (path.stem + STANDALONE_SUFFIX)
            if not dry_run:
                target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and has_window(target):
            report = Report(path, target, "present", "already has an echo window",
                            source_bytes=source_bytes)
            if verify:
                report.verification = verify_window(target, source=source)
            return report

    ion = _power_ionogram(source)
    selection = select_window(source, ion, margin_km=margin_km,
                              max_hops=max_hops, manual_km=manual_km)
    plan = plan_for(source, selection.lo_km, selection.hi_km)
    stored = plan.n_out(source.n_samples) * np.dtype(np.complex64).itemsize
    if dry_run:
        return Report(path, target, "dry-run", selection=selection, plan=plan,
                      stored_bytes=stored, source_bytes=source_bytes)

    samples = _source_samples(source)
    z_win = cut(samples, plan)
    attrs = _attrs(source, selection, plan)
    extra = {}
    if source.kind == SOURCE_LFS:
        with open(path, "rb") as fh:
            extra = {"row_noise": _lfs_row_noise(samples, LFS_NOISE_WINDOW),
                     "noise_window": LFS_NOISE_WINDOW,
                     "lfs_header": fh.read(io_lfs.HEADER_SIZE)}
    del samples

    if source.kind == SOURCE_V2 and drop_raw:
        return _drop_raw(source, z_win, attrs, (selection, plan), verify=True,
                         dry_run=False, source_bytes=source_bytes)

    import h5py

    report = Report(path, target, "written", selection=selection, plan=plan,
                    stored_bytes=z_win.nbytes, source_bytes=source_bytes)
    with _staged(target) as tmp:
        if target.exists():
            shutil.copyfile(target, tmp)
            mode = "a"
        else:
            mode = "w"
        with h5py.File(tmp, mode) as fh:
            if mode == "w":
                for key, value in _standalone_attrs(source).items():
                    fh.attrs[key] = value
            _write_group(fh, z_win, attrs, plan.taps(), **extra)
        if verify:
            report.verification = verify_window(tmp, source=source)
            if not report.verification.ok:
                raise VerificationError(report.verification.describe())
    return report


class VerificationError(RuntimeError):
    """A written window did not verify; nothing was replaced."""


def _drop_raw(source: Source, z_win, attrs, chosen, *, verify: bool,
              dry_run: bool, source_bytes: int) -> Report:
    """Rewrite a v2 product without its full ``z``, keeping everything else.

    Built as a new file -- HDF5 does not give space back when a dataset is
    deleted -- then verified against the original's ``z`` and renamed into
    place. A product whose rewrite does not verify keeps its ``z``.
    """
    import h5py

    path = source.path
    selection, plan = chosen if chosen is not None else (None, None)
    report = Report(path, path, "raw-dropped", selection=selection, plan=plan,
                    source_bytes=source_bytes)
    if dry_run:
        report.status = "dry-run"
        report.detail = "would rewrite without `z`"
        return report

    with _staged(path) as tmp:
        with h5py.File(path, "r") as src, h5py.File(tmp, "w") as dst:
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            for key in src:
                if key == io_chirp.RAW_VOLTAGE_KEY:
                    continue
                src.copy(src[key], dst, name=key)
            if z_win is not None:
                _write_group(dst, z_win, attrs, plan.taps())
        problems = _same_except_z(path, tmp)
        check = verify_window(tmp, source=source)
        check.problems.extend(problems)
        check.ok = check.ok and not problems
        report.verification = check
        if not check.ok:
            raise VerificationError(f"{path.name}: raw `z` kept -- "
                                    + check.describe())
    report.stored_bytes = read(path, samples=False).nbytes
    report.detail = (f"product {source_bytes / 1e6:.1f} -> "
                     f"{path.stat().st_size / 1e6:.2f} MB")
    return report


def _same_except_z(original: Path, rewritten: Path) -> list:
    """Every dataset and attribute but ``z`` must have come across unchanged."""
    import h5py

    problems: list = []
    with h5py.File(original, "r") as a, h5py.File(rewritten, "r") as b:
        if io_chirp.RAW_VOLTAGE_KEY in b:
            problems.append("rewritten product still holds `z`")
        for key, value in a.attrs.items():
            if key not in b.attrs or not np.array_equal(np.asarray(b.attrs[key]),
                                                       np.asarray(value)):
                problems.append(f"root attribute {key!r} differs")

        def visit(name, obj):
            if name == io_chirp.RAW_VOLTAGE_KEY or not isinstance(obj, h5py.Dataset):
                return
            if name not in b:
                problems.append(f"{name} missing")
                return
            x, y = obj[()], b[name][()]
            if not np.array_equal(np.asarray(x), np.asarray(y), equal_nan=_float(x)):
                problems.append(f"{name} differs")

        a.visititems(visit)
    return problems


def _float(value) -> bool:
    return np.asarray(value).dtype.kind in "fc"


# --- verification ------------------------------------------------------------

#: Cells this far above the median-noise level take part in the power check.
POWER_CHECK_ABOVE_NOISE_DB = 10.0
#: Median absolute dB difference the power check allows.
POWER_CHECK_TOLERANCE_DB = 0.5


def verify_window(path: str | Path, *, source: Source | None = None) -> Verification:
    """Check a stored window against its source.

    Four things: the group reads back with every attribute; the stored taps
    meet ``STOPBAND_DB`` and ``MAX_RIPPLE_DB``, measured afresh; re-cutting
    the source with them gives the stored samples *exactly*; and the power
    re-derived from the window matches the source's own within the window --
    median |difference| at most ``POWER_CHECK_TOLERANCE_DB`` over cells
    ``POWER_CHECK_ABOVE_NOISE_DB`` above noise. Without a source (a v2 product
    whose ``z`` was dropped) only the first two can be checked.
    """
    problems: list = []
    metrics: dict = {}
    try:
        window = read(path)
    except Exception as exc:
        return Verification(False, [f"unreadable: {exc}"], {})
    plan = window.plan
    if plan.numtaps > 1:
        # Measured on the taps as stored, which are the ones every reader uses.
        attenuation, ripple = _response(plan.taps(), plan.source_sr_hz,
                                        plan.passband_hz, plan.stopband_hz)
        metrics["filter"] = f"{attenuation:.1f} dB / {ripple:.4f} dB"
        if attenuation < STOPBAND_DB or ripple > MAX_RIPPLE_DB:
            problems.append(f"stored filter gives {attenuation:.1f} dB stopband, "
                            f"{ripple:.4f} dB ripple")

    if source is None or (source.kind == SOURCE_V2
                          and io_chirp.RAW_VOLTAGE_KEY not in _keys(source.path)):
        metrics["source"] = "not available"
        return Verification(not problems, problems, metrics)

    samples = _source_samples(source)
    if len(samples) != window.source_samples:
        problems.append(f"source has {len(samples)} samples, window records "
                        f"{window.source_samples}")
    again = cut(samples, plan)
    if again.shape != window.samples.shape or not np.array_equal(again, window.samples):
        diff = (float(np.max(np.abs(again - window.samples)))
                if again.shape == window.samples.shape else float("inf"))
        problems.append(f"re-cut differs from the stored samples (max {diff:.3g})")
    else:
        metrics["re-cut"] = "identical"

    full, derived = _power_pair(source, window, samples)
    above = full >= NOISE_DB + POWER_CHECK_ABOVE_NOISE_DB
    if above.any():
        delta = float(np.median(np.abs(full[above] - derived[above])))
        metrics["power"] = f"{delta:.3f} dB median over {int(above.sum())} cells"
        if delta > POWER_CHECK_TOLERANCE_DB:
            problems.append(f"window power differs by {delta:.2f} dB median")
    else:
        metrics["power"] = "no cell 10 dB above noise to compare"
    return Verification(not problems, problems, metrics)


def _keys(path: Path) -> set:
    import h5py

    with h5py.File(path, "r") as fh:
        return set(fh.keys())


def _power_pair(source: Source, window: EchoWindow,
                samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """dB inside the window: from the source, and re-derived from the window.

    One FFT per source row (no oversampling), both normalised by the source's
    full-band row median -- which is what makes this a test of the window and
    not of two different noise estimates.
    """
    rec = reconstruct(window)
    win_len, step = source.window, source.step
    n_rows = max(0, (len(samples) - win_len) // step + 1)
    w = spectro.hanning(win_len)
    beats = np.fft.fftfreq(win_len, d=1.0 / source.sr_hz)
    ranges = np.array([source.range_at(b) for b in beats])
    inside = (ranges >= window.range_lo_km) & (ranges <= window.range_hi_km)
    full_db, derived_db = [], []
    for i in range(n_rows):
        seg = slice(i * step, i * step + win_len)
        p_full = np.abs(np.fft.fft(w * samples[seg])) ** 2
        p_rec = np.abs(np.fft.fft(w * rec[seg])) ** 2
        floor = NOISE_COEF * float(np.median(p_full))
        if floor <= 0:
            continue
        full_db.append(spectro.to_db(p_full[inside] / floor))
        derived_db.append(spectro.to_db(p_rec[inside] / floor))
    if not full_db:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(full_db), np.concatenate(derived_db)


# --- re-deriving an ionogram ---------------------------------------------------

def _hann_power(window: int, *, v2: bool) -> float:
    """Sum of squares of the window each format's spectrogram uses."""
    if v2:
        from scipy.signal import windows

        return float(np.sum(windows.hann(window) ** 2))
    return float(np.sum(spectro.hanning(window) ** 2))


def reprocess_window(path: str | Path,
                     window: int | None = None,
                     gate_km: tuple[float, float] | None = None,
                     *,
                     stations: Mapping[str, tuple[float, float]] | None = None,
                     nominal_stop_mhz: float | None = None) -> Ionogram:
    """An ionogram re-derived from the echo window, restricted to its range.

    The window is reconstructed at the source rate and put through the
    source's own spectrogram -- v2's 13x-oversampled port on the conjugate, or
    ``spectro``'s plain Hann rows -- then divided by the *stored full-band*
    noise, so it lands on the shared 43 dB scale. At the source's own window
    that noise is exact; at another window it is scaled by the window's power
    (exact for white noise) and, for ``.lfs``, interpolated between rows.
    Outside the stored range nothing exists, so the axis stops at the window.
    """
    path = Path(path)
    ew = read(path)
    lo, hi = ew.range_lo_km, ew.range_hi_km
    if gate_km is not None:
        lo, hi = max(lo, float(gate_km[0])), min(hi, float(gate_km[1]))
        if lo >= hi:
            raise ValueError(f"{path}: gate {gate_km[0]:.0f}-{gate_km[1]:.0f} km "
                             f"misses the window {ew.range_lo_km:.0f}-"
                             f"{ew.range_hi_km:.0f} km")
    rec = reconstruct(ew)
    if ew.source == SOURCE_V2:
        return _reprocess_v2(path, ew, rec, window or ew.source_window, (lo, hi),
                             stations, nominal_stop_mhz)
    return _reprocess_lfs(path, ew, rec, window or ew.noise_window, (lo, hi),
                          stations)


def _reprocess_v2(path, ew, rec, window, gate, stations, nominal_stop_mhz) -> Ionogram:
    header = io_chirp.read_header(path, stations)
    with io_chirp.open_h5(path) as fh:
        noise_floor = np.asarray(fh["noise_floor"][()], dtype=np.float64)
    step = ew.source_step
    spec = io_chirp.v2_spectrogram(np.conj(rec), window=window, step=step)
    n_spec = spec.shape[0]
    floor = noise_floor[:n_spec]
    if floor.size < n_spec:
        floor = np.concatenate([floor, np.full(n_spec - floor.size, floor[-1])])
    floor = floor * (_hann_power(window, v2=True)
                     / _hann_power(ew.source_window, v2=True))
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = (spec - floor[:, None]) / floor[:, None]
    power = io_chirp.snr_to_power(snr)

    sr = ew.source_sr_hz
    freqs_mhz = (ew.sweep_hz_at_t0
                 + header.rate * np.arange(n_spec) * step / sr) / 1e6
    ranges_km = (io_chirp.C_M_S / header.rate) * np.fft.fftshift(
        np.fft.fftfreq(window, d=1.0 / sr)) / 1e3
    ranges_km, _ = io_chirp._absolute_ranges(
        ranges_km * 1e3, header.t0, header.range_offset_applied,
        header.range_start_m, header.tx_name)
    vrange = ranges_km[::-1]
    power = power[:, ::-1]

    lo, hi = gate
    if ew.range_axis == "absolute":
        lo = max(lo, io_chirp.MIN_ABSOLUTE_RANGE_KM)
    keep = (vrange >= lo) & (vrange <= hi)
    if not keep.any():
        raise ValueError(f"{path}: window {lo:.0f}-{hi:.0f} km keeps no bin")
    vrange, power = vrange[keep], power[:, keep]
    if io_chirp.SNR_OFFSET_DB:
        power = power * 10.0 ** (io_chirp.SNR_OFFSET_DB / 10.0)
    cal = io_chirp._build_calibration(
        header, freqs_mhz, vrange, (lo, hi), nominal_stop_mhz,
        computed=io_chirp.computed_rows(noise_floor, n_spec))
    return Ionogram(power=power.astype(np.float32), cal=cal, header=header,
                    window=window, zero_periods=0)


def _reprocess_lfs(path, ew, rec, window, gate, stations) -> Ionogram:
    if ew.lfs_header is None or ew.row_noise is None:
        raise ValueError(f"{path}: an .lfs echo window needs its header and "
                         f"row noise, and this one lacks them")
    header = io_lfs.parse_header(ew.lfs_header, path, stations)
    n_freq = len(rec) // window
    if n_freq == 0:
        raise ValueError(f"{path}: {len(rec)} samples is shorter than one "
                         f"{window}-sample window")
    cal = calibrate.build(header, n_freq=n_freq, window=window, gate_km=gate)
    i_lo, i_hi = cal.gate_idx

    # The stored floor is per `noise_window` row; rows of another length are
    # placed by their centre time and scaled by the window's power.
    stored_centre = (np.arange(len(ew.row_noise)) + 0.5) * ew.noise_window
    centre = (np.arange(n_freq) + 0.5) * window
    floor = np.interp(centre, stored_centre, ew.row_noise)
    floor = floor * (_hann_power(window, v2=False)
                     / _hann_power(ew.noise_window, v2=False))

    win = spectro.hanning(window)
    out = np.empty((n_freq, i_hi - i_lo + 1), dtype=np.float32)
    for i in range(n_freq):
        spectrum = np.abs(np.fft.fftshift(
            np.fft.fft(win * rec[i * window:(i + 1) * window]))) ** 2
        out[i] = spectrum[i_lo:i_hi + 1] / floor[i]
    return Ionogram(power=out, cal=cal, header=header, window=window,
                    zero_periods=0)


__all__ = [
    "EchoWindow", "NoEchoError", "Plan", "Report", "Selection", "Source",
    "Verification", "VerificationError", "cut", "plan_for", "read",
    "read_header", "reconstruct", "reprocess_window", "select_window",
    "source_of", "verify_window", "write",
]
