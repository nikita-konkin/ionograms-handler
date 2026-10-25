"""The fast part of the channel, off the phase: Doppler spread, its shape, Rice K.

A power ionogram keeps where each echo is and how strong. The dechirped
voltage -- an ``.lfs``, a chirpsounder2 product with ``z``, or an echo window
cut from either (``docs/echo-window.md``) -- keeps how each echo *moves*.
After dechirping, mode m is a tone that follows its beat trajectory, the
mode's delay curve read along the sweep::

    b(t) = beat(tau_m(f(t))),        f(t) = sweep start + rate * t

Multiplying the record by exp(-j 2 pi integral b dt) -- a second dechirp
along the mode's own curve, as ``lfs_tools/doppler_along_sweep.py`` did in
the prototype -- brings the echo to near zero frequency. Low-passing to a few
tens of hertz then removes every other mode, the interference, and almost all
the noise, and what is left is the mode's fading gain times its slowly
changing level.

**Read off the power, not the phase.** The delay curve the second dechirp
follows is the ionogram's trace, good to a few microseconds -- and 10 us of
error is 1 Hz of beat at 100 kHz/s, a frequency modulation as wide as the
spreads being measured. Its phase is therefore unusable for the spread, but
its power is untouched by it: for a Gaussian-scatter gain the power's
autocovariance is the square of the gain's autocorrelation,
C_P(lag) = |R(lag)|^2 (with a specular part, a quadratic in |R| that
:func:`_pooled_fit` solves). The spread and the spectrum's shape come from
|R| so recovered, the Rice factor from the power's moments.

**Only the first decorrelation.** A mode gives a minute or two of record, a
few tens of fades, and the power's variance over so few is uncertain by
+-40 % -- an estimator that leans on it, fitting |R| down to 0.3, scatters by
as much on a pure Gaussian-scatter gain. The power's structure function
D(L) = <(P(t + L) - P(t))^2> over the first lags is averaged over every
pair of samples that close, and divided by the power's second moment, whose
error moves with its own (both are made by the same strong fades), it is
good to 5-15 % (``tests/test_sim_fast.py`` measures it). So |R| is read only
down to 1/sqrt(2), where the rise of D(L) is still steep and the variance
does not yet matter. A slow change of the level, a dB or two across a band,
scales D and the second moment alike and drops out of their ratio.

**What one sweep cannot give.** A Doppler shift nu moves an up-sweep's beat
exactly as a delay change of nu / rate would, so a single sweep cannot tell
them apart (an up and a down sweep could), and the power, which this reads,
holds no frequency at all. The shift stays a prior here; the slow drift of
delay between soundings is where it can be derived from. A channel that
has not decorrelated to 1/sqrt(2) within the longest lag, MAX_LAG_S or a
third of a band, gives an upper bound -- below ~0.03 Hz -- not a value; one
decorrelated within FIT_MIN_POINTS lags, a lower bound.

**What the sweep adds.** Moving at 100 kHz/s, the sweep crosses the
channel's structure in frequency as well as waiting out its changes in
time: o and x waves 20 us apart beat at rate * 20 us = 2 Hz. Read as one
wave, that beat passes for a spread several times the true one -- 0.5 Hz
read as 3 Hz at a 30 us split. :func:`_fit_pair` fits the two waves
instead, and their beat gives the split and their depth the power ratio:
the state's ``ox_split_us`` and ``x_to_o_db``, which no power ionogram
resolves. Which of the two is x one sweep cannot tell, nor a differential
Doppler shift from the split; the weaker is taken as x, the shift as none.

**Calibrated by the loop.** :func:`calibrate` sounds known states through
:mod:`muf.sim.sounding` and reads them back with the same sweep, sample rate
and processing as a real sounding; ``tests/test_sim_fast.py`` holds the
estimator to what it measured.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .state import MEASURED, ChannelState, Mode, Provenance

#: Pass bands of the low-pass after the second dechirp, Hz, widest first;
#: each stops at STOP_RATIO times its pass. The widest that leaves no second
#: echo inside and MIN_SNR_DB of signal is used, the same for every band of
#: a mode. Wide costs signal-to-noise (40 Hz is 6 dB worse than 10 Hz) and
#: needs the other echoes far off; narrow rounds off a fast spread's
#: autocorrelation -- the fit models the filter, so this costs scatter, not
#: bias. On cyprus1 the low and high rays near the MUF are 20-50 Hz apart.
PASSES_HZ = (40.0, 20.0, 10.0)
STOP_RATIO = 1.25
PASS_HZ = PASSES_HZ[0]
STOP_HZ = STOP_RATIO * PASS_HZ

#: Rate the gain is analysed at, Hz: above twice the widest stop band, and
#: fine enough that a 4 Hz spread decorrelates to 1/sqrt(2) over several lags.
OUT_RATE_HZ = 125.0

#: Another echo must stay this far beyond the stop band, Hz, all through
#: the part of a band used: room for both spreads and the traces' errors.
ISOLATION_MARGIN_HZ = 20.0

#: Shortest stretch of record a band is measured on, s.
MIN_BAND_S = 10.0

#: The trace is re-centred on the echo before the narrow passes: the echo's
#: offset from it is the peak of RECENTRE_S spectra of the widest pass,
#: fitted with a polynomial in time of RECENTRE_ORDER, windows further than
#: RECENTRE_OUTLIER_HZ from the fit dropped (another echo stronger there).
#: A narrow pass is only as good as the trace it follows: 10 Hz of a
#: tracker's error, 100 us, half-empties a 10 Hz pass, and the power it lets
#: through then fades with the trace's wander, not the channel's.
RECENTRE_S = 2.0
RECENTRE_ORDER = 2
RECENTRE_OUTLIER_HZ = 5.0

#: Width of the bands a mode is cut into, MHz (30 s of sweep at 100 kHz/s),
#: and the margin kept from its band's ends -- the low end where it emerges
#: from the noise, the top where the low and high rays meet. A band holds
#: some 30 fades at 0.5 Hz; the autocorrelation of fewer scatters too much.
BAND_MHZ = 3.0
EDGE_LOW_MHZ = 0.2
EDGE_TOP_MHZ = 0.5

#: Seconds trimmed from each end of a filtered band: the FFT filter is
#: circular.
TRIM_S = 1.0

#: Signal-to-noise ratio in the pass band below which a band is skipped.
MIN_SNR_DB = 10.0

#: The scattered part's |R| is fitted from 1 down to this, where the
#: power's autocovariance has fallen to half.
FIT_LOW = 1.0 / math.sqrt(2.0)

#: Fewest lags the fit needs; with fewer the spread is too fast for the
#: analysis rate, and is reported as a lower bound.
FIT_MIN_POINTS = 3

#: A Rice factor is reported only where the scattered part's normalised
#: power variance gamma (1 for Rayleigh) is this many of its standard
#: errors below 1. Over n fades -- duration times spread, as F.1487's test
#: lengths count them -- the standard error is at most GAMMA_SPREAD /
#: sqrt(n): measured on Rayleigh gains of 0.1-2 Hz over 84 s, it is 0.75-1.5
#: for a Gaussian spectrum and 0.6-0.9 for a Lorentzian (tests/test_sim_fast.py).
K_SIGMAS = 2.0
GAMMA_SPREAD = 1.6

#: Another mode whose delay stays this close, us, all through its overlap
#: with a band is the same echo -- the tracker split one trace in two, or
#: named a branch twice -- not one to keep away from.
DUPLICATE_US = 15.0

#: A tracked mode inside the isolation distance cuts a band only if the
#: record shows two echoes: demodulated along each trace in turn and filtered to
#: +-CONFIRM_HZ, where the traces are more than 2 CONFIRM_HZ apart, both
#: carry CONFIRM_DB or more of the mode's power. The tracker makes ghosts --
#: a trace split in two, a fade read as a step, a branch named twice -- and
#: on a synthetic sounding of one path every second mode is one; and where
#: the mode's own trace has wandered, its echo lies on the other's instead.
#: An echo 13 dB down adds 10 % to the power's structure function; a 2 Hz
#: Lorentzian's own tail, 10 Hz out, is 16 dB down.
CONFIRM_HZ = 5.0
CONFIRM_DB = -13.0
#: An overlap shorter than this, s, is too little record to confirm an
#: echo in.
CONFIRM_MIN_S = 2.0

#: Two waves in one mode -- o and x, or two rays a few tens of
#: microseconds apart -- fade independently and beat along the sweep at
#: rate * split: 30 us is 3 Hz. Their sum is still Rayleigh, but its |R|^2
#: is r(L)^2 [1 - 2 q (1 - cos 2 pi nu L)], q = rho / (1 + rho)^2 for a
#: power ratio rho, and read as one wave the beat passes for a spread. The
#: pair is fitted where |R|^2, past its first minimum, climbs back by
#: PAIR_RISE; it is kept if its misfit is under PAIR_GAIN of one wave's over
#: the same lags, and q is at least PAIR_MIN_Q (the weaker 14 dB down).
PAIR_RISE = 0.08
PAIR_GAIN = 0.5
PAIR_MIN_Q = 0.04
#: A beat is told from fading only if one wave is still this correlated,
#: in |R|^2, a beat period on: a beat at least 2.9 times a Gaussian spread,
#: 5.2 times a Lorentzian. Slower, a few fades' worth of noise in |R|^2
#: climbs as a beat would -- 0.1 Hz read as two waves 0.9 us apart.
PAIR_ENVELOPE = 0.3
#: The pair is fitted over this many beat periods.
PAIR_PERIODS = 4.0

#: Where the power varies more than Rayleigh fading can, the level changes
#: too: the bands are divided by their running mean over this many
#: coherence times and fitted again. A level a dB or two from its trend
#: drops out of D / m2 by itself; swings of 10 dB over seconds, as cyprus1's
#: F trace has at noon, empty |R|^2 at long lags, and a pair is not found.
TREND_COHERENCES = 10.0

#: Longest lag of the power's structure function, s.
MAX_LAG_S = 10.0

#: Offsets tried for the noise reference, Hz from the mode's beat.
NOISE_OFFSETS_HZ = (150.0, -150.0, 250.0, -250.0, 100.0, -100.0)


# --- the record ------------------------------------------------------------------

@dataclass
class Record:
    """Dechirped samples and how their frequency maps to delay.

    ``beat_hz`` is the frequency, in the stored samples, of an echo of a
    given delay: the source's range axis (``km_per_hz``, ``range_zero_km``,
    as :class:`muf.echo_window.Source` defines it), less the shift a window
    was mixed down by.
    """

    samples: np.ndarray
    sr_hz: float
    t_first_s: float            # time of samples[0] after the sweep's start
    sweep_hz_at_0: float        # the sweep's frequency at t = 0
    rate_hz_s: float
    km_per_hz: float
    range_zero_km: float
    shift_hz: float
    km_per_us: float            # the state's delay -> range, as observe made it
    source: str

    def beat_hz(self, delay_us) -> np.ndarray:
        range_km = np.asarray(delay_us, dtype=np.float64) * self.km_per_us
        return (range_km - self.range_zero_km) / self.km_per_hz - self.shift_hz

    def time_at(self, f_mhz) -> np.ndarray:
        return (np.asarray(f_mhz, dtype=np.float64) * 1e6 - self.sweep_hz_at_0) \
            / self.rate_hz_s

    def freq_at(self, t_s) -> np.ndarray:
        return (self.sweep_hz_at_0 + self.rate_hz_s * np.asarray(t_s)) / 1e6

    @property
    def duration_s(self) -> float:
        return self.samples.size / self.sr_hz


def record_of(path: str | Path, stations=None) -> Record:
    """The dechirped record of a sounding: ``.lfs``, v2 with ``z``, an echo window."""
    from .. import echo_window, io_chirp, io_lfs
    from . import observe

    path = Path(path)
    if path.suffix == ".h5" and echo_window.has_window(path):
        ew = echo_window.read(path)
        if ew.lfs_header is not None:
            km_per_us = observe.km_per_us(io_lfs.parse_header(ew.lfs_header, path,
                                                              stations))
        else:
            km_per_us = io_chirp.C_M_S * 1e-9
        return Record(samples=np.asarray(ew.samples, dtype=np.complex64),
                      sr_hz=ew.sr_hz, t_first_s=ew.t_first_s,
                      sweep_hz_at_0=ew.sweep_hz_at_t0, rate_hz_s=ew.rate_hz_s,
                      km_per_hz=ew.km_per_hz, range_zero_km=ew.range_zero_km,
                      shift_hz=ew.shift_hz, km_per_us=km_per_us,
                      source=f"{path.name} (echo window)")
    src = echo_window.source_of(path, stations)
    samples = echo_window._source_samples(src)
    if samples is None or len(samples) == 0:
        raise ValueError(f"{path}: no dechirped samples (a v2 product without z)")
    return Record(samples=np.asarray(samples, dtype=np.complex64), sr_hz=src.sr_hz,
                  t_first_s=0.0, sweep_hz_at_0=src.sweep_hz_at_0,
                  rate_hz_s=src.rate_hz_s, km_per_hz=src.km_per_hz,
                  range_zero_km=src.range_zero_km, shift_hz=0.0,
                  km_per_us=observe.km_per_us(src.header), source=path.name)


# --- one band --------------------------------------------------------------------

@dataclass
class BandEstimate:
    """What one band of one mode gave."""

    mode: str
    f_lo_mhz: float
    f_hi_mhz: float
    used: bool
    reason: str = ""
    snr_db: float = float("nan")
    seconds: float = 0.0
    rate_hz: float = 0.0
    pass_hz: float = float("nan")          # the low-pass used
    cut_by: str = ""                       # the echo the band was cut short for
    recentred_hz: float = 0.0              # largest correction to the trace
    coherence_s: float = float("nan")      # where the scattered part's |R| is 1/e
    spread_gaussian_hz: float = float("nan")    # 2 sigma
    spread_lorentzian_hz: float = float("nan")  # full width at half power
    residual_gaussian: float = float("nan")     # rms misfit in ln |R|
    residual_lorentzian: float = float("nan")
    shape: str = ""
    rice_k_db: float | None = None
    bound: str = ""                         # "upper" or "lower" when the spread is one
    autocorrelation: list = field(default_factory=list)
    # What the mode's pooled estimate is made of: the band's power series as
    # received, and the noise reference's power and |R|.
    power: np.ndarray | None = field(default=None, repr=False)
    noise_r: np.ndarray | None = field(default=None, repr=False)
    noise_power: float = 0.0

    @property
    def spread_hz(self) -> float:
        return (self.spread_lorentzian_hz if self.shape == "lorentzian"
                else self.spread_gaussian_hz)


def _lowpass(x: np.ndarray, sr: float, pass_hz: float = PASS_HZ,
             stop_hz: float = STOP_HZ) -> np.ndarray:
    """Zero-phase FFT low-pass: flat to ``pass_hz``, raised-cosine to ``stop_hz``."""
    n = x.size
    f = np.abs(np.fft.fftfreq(n, d=1.0 / sr))
    h = np.clip((stop_hz - f) / (stop_hz - pass_hz), 0.0, 1.0)
    h = 0.5 - 0.5 * np.cos(np.pi * h)
    return np.fft.ifft(np.fft.fft(x) * h)


def _bandwidth(pass_hz: float, stop_hz: float) -> float:
    """Noise bandwidth of :func:`_lowpass`, Hz, two-sided."""
    return 2.0 * (pass_hz + 0.375 * (stop_hz - pass_hz))


def _down(record: Record, i0: int, i1: int, beat: np.ndarray, *,
          pass_hz: float = PASS_HZ, stop_hz: float = STOP_HZ,
          trim_s: float = TRIM_S) -> tuple[np.ndarray, float]:
    """Samples [i0, i1) brought to zero frequency along ``beat``, filtered, thinned."""
    x = np.asarray(record.samples[i0:i1], dtype=np.complex128)
    phase = 2.0 * np.pi * np.cumsum(beat) / record.sr_hz
    y = _lowpass(x * np.exp(-1j * phase), record.sr_hz, pass_hz, stop_hz)
    step = max(1, math.floor(record.sr_hz / OUT_RATE_HZ))
    rate = record.sr_hz / step
    trim = round(trim_s * rate)
    z = y[::step]
    return (z[trim:z.size - trim] if z.size > 2 * trim + 8 else z[:0]), rate


def _beats(record: Record, mode: Mode, t: np.ndarray) -> np.ndarray:
    f = np.clip(record.freq_at(t), mode.f_lo, mode.f_hi)
    return record.beat_hz(mode.delay_at(f))


def _autocorrelation(z: np.ndarray, max_lag: int) -> np.ndarray:
    n = z.size
    return np.array([np.vdot(z[:n - k], z[k:]) / (n - k) for k in range(max_lag + 1)])


def lorentzian_r(lags_s: np.ndarray, gamma_hz: float, pass_hz: float = math.inf,
                 stop_hz: float = math.inf) -> np.ndarray:
    """|R| of a Lorentzian gain of half width ``gamma_hz`` after :func:`_lowpass`.

    exp(-2 pi gamma |L|) unfiltered. Filtered, the spectrum times |H|^2,
    integrated to the stop band: the cusp at zero lag is made of the tails
    the filter takes, and a 10 Hz pass rounds a 1 Hz Lorentzian's enough to
    read it 15 % slow. The Gaussian needs no such care -- its tails are gone
    well inside any of PASSES_HZ.
    """
    lags = np.abs(np.asarray(lags_s, dtype=np.float64))
    # What the filter takes is about 2 gamma / (pi pass) of the power; below
    # 0.1 % it is left out.
    if not math.isfinite(pass_hz) or 2.0 * gamma_hz / (math.pi * pass_hz) < 1e-3:
        return np.exp(-2.0 * np.pi * gamma_hz * lags)
    step = min(gamma_hz / 8.0, 1.0 / (16.0 * max(float(lags.max()), 1e-3)), 0.05)
    positive = lags[lags > 0]
    dt = float(positive.min()) if positive.size else 1.0
    index = lags / dt
    if positive.size and np.allclose(index, np.round(index), atol=1e-6):
        # Lags on a grid of dt, as they always are here: the cosine sum is a
        # real FFT over a frequency grid of 1 / (n dt).
        from scipy.fft import next_fast_len, rfft
        n = next_fast_len(max(math.ceil(1.0 / (dt * step)),
                              2 * int(np.round(index).max()) + 2))
        nu = np.arange(n // 2 + 1) / (n * dt)
        nu = nu[nu < stop_hz]
        h = np.clip((stop_hz - nu) / (stop_hz - pass_hz), 0.0, 1.0)
        h = 0.5 - 0.5 * np.cos(np.pi * h)
        w = h ** 2 / (gamma_hz ** 2 + nu ** 2)
        w[0] *= 0.5                              # the trapezoid's end
        full = np.zeros(n)
        full[:w.size] = w
        r = rfft(full).real[np.round(index).astype(int)]
        return r / (float(np.sum(w)))
    nu = np.arange(0.0, stop_hz, step) + 0.5 * step
    h = np.clip((stop_hz - nu) / (stop_hz - pass_hz), 0.0, 1.0)
    h = 0.5 - 0.5 * np.cos(np.pi * h)
    w = h ** 2 / (gamma_hz ** 2 + nu ** 2)
    return (np.cos(2.0 * np.pi * np.outer(lags, nu)) @ w) / float(np.sum(w))


def _golden(f, lo: float, hi: float, iterations: int = 40) -> float:
    """The minimum of ``f`` on [lo, hi], by golden section."""
    g = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iterations):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return 0.5 * (a + b)


def _fit(lags_s: np.ndarray, r: np.ndarray, pass_hz: float = math.inf,
         stop_hz: float = math.inf) -> dict:
    """Gaussian and Lorentzian fits of the scattered part's |R| down to FIT_LOW.

    Both in -ln|R|, through the origin: the Gaussian's 2 pi^2 sigma^2 L^2
    by least squares, the Lorentzian's 2 pi gamma L likewise and then, with
    the low-pass's effect on its tails (:func:`lorentzian_r`), refined.
    """
    below = np.flatnonzero(r < FIT_LOW)
    stop = below[0] if below.size else r.size
    out = {"points": int(stop - 1), "crossed": bool(below.size)}
    # Not down to FIT_LOW within the lags: the rest of the fall would be
    # extrapolated, and a bound is all the record gives.
    if not below.size or out["points"] < FIT_MIN_POINTS:
        return out
    x, y = lags_s[1:stop], -np.log(r[1:stop])
    a = float(np.sum(y * x ** 2) / np.sum(x ** 4))
    b = max(float(np.sum(y * x) / np.sum(x ** 2)), 1e-9)
    gamma = b / (2.0 * math.pi)
    if math.isfinite(pass_hz):
        def misfit(log_g):
            model = -np.log(np.clip(lorentzian_r(x, math.exp(log_g), pass_hz, stop_hz),
                                    1e-12, None))
            return float(np.sum((y - model) ** 2))
        gamma = math.exp(_golden(misfit, math.log(gamma / 2.0), math.log(gamma * 4.0)))
        model_l = -np.log(np.clip(lorentzian_r(x, gamma, pass_hz, stop_hz), 1e-12,
                                  None))
    else:
        model_l = b * x
    out["sigma"] = math.sqrt(max(a, 0.0) / (2.0 * math.pi ** 2))
    out["gamma"] = gamma
    # Misfits relative to the decorrelation reached, so that bands and modes
    # of any spread compare alike.
    scale = max(float(np.max(y)), 1e-9)
    out["res_g"] = float(np.sqrt(np.mean((y - a * x ** 2) ** 2))) / scale
    out["res_l"] = float(np.sqrt(np.mean((y - model_l) ** 2))) / scale
    return out


def _one_wave(lags: np.ndarray, spread: float, shape: str, pass_hz: float,
              stop_hz: float) -> np.ndarray:
    """|R| of one wave of ``spread``, the filter's effect on a Lorentzian included."""
    half = 0.5 * spread
    if shape == "gaussian":
        return np.exp(-2.0 * np.pi ** 2 * half ** 2 * lags ** 2)
    return lorentzian_r(lags, half, pass_hz, stop_hz)


def _fit_pair(lags: np.ndarray, r: np.ndarray, pass_hz: float, stop_hz: float,
              single: dict) -> dict | None:
    """Two waves beating along the sweep, if |R|^2 shows them; else None."""
    from scipy.optimize import least_squares

    r2 = r ** 2
    smooth = np.convolve(np.pad(r2, 1, mode="edge"), np.ones(3) / 3.0, mode="valid")
    half = np.flatnonzero(smooth < 0.5)
    if half.size == 0 or half[0] < 2:
        return None
    # Where |R|^2 climbs back most above its running minimum, within six
    # times its fall to a half: a beat shows there, a single wave never
    # climbs but by its record's noise, far out.
    span = min(smooth.size, 6 * int(half[0]) + 1)
    low = np.minimum.accumulate(smooth[:span])
    climb = smooth[:span] - low
    j = int(np.argmax(climb))
    if climb[j] < PAIR_RISE:
        return None
    i = int(np.flatnonzero(smooth[:j + 1] == low[j])[-1])   # the minimum before it
    if i < 2:
        return None
    # The minimum is at a half-integer number of beat periods, the climb's
    # top at an integer one: the first that fits both.
    beats = [(n + 0.5) / float(lags[i]) for n in range(4)
             if float(lags[j]) * (n + 0.5) / float(lags[i]) >= n + 0.5]
    if not beats:
        return None
    end = min(r.size, max(j + 4, round(PAIR_PERIODS / min(beats) / float(lags[1]))))
    x, y = lags[1:end], r2[1:end]
    best = None
    for shape in ("gaussian", "lorentzian"):
        def model(v, shape=shape):
            spread, beat, q = math.exp(v[0]), v[1], v[2]
            one = _one_wave(x, spread, shape, pass_hz, stop_hz)
            return one ** 2 * (1.0 - 2.0 * q * (1.0 - np.cos(2.0 * np.pi * beat * x)))
        for beat0 in beats:
            for spread0 in (0.1 * beat0, 0.4 * beat0):
                try:
                    fit = least_squares(lambda v: model(v) - y,
                                        [math.log(max(spread0, 0.01)), beat0, 0.15],
                                        bounds=([math.log(0.005), 0.5 * beat0, 0.0],
                                                [math.log(20.0), 1.5 * beat0, 0.25]))
                except ValueError:
                    continue
                cost = float(np.mean(fit.fun ** 2))
                if best is None or cost < best["cost"]:
                    best = {"shape": shape, "spread": math.exp(fit.x[0]),
                            "beat_hz": fit.x[1], "q": fit.x[2], "cost": cost}
    if best is None or best["q"] < PAIR_MIN_Q:
        return None
    period = np.array([1.0 / best["beat_hz"]])
    if float(_one_wave(period, best["spread"], best["shape"], pass_hz,
                       stop_hz)[0]) ** 2 < PAIR_ENVELOPE:
        return None
    # One wave over the same lags, by the same least squares.
    single_cost = math.inf
    for shape in ("gaussian", "lorentzian"):
        def one(v, shape=shape):
            return _one_wave(x, math.exp(v[0]), shape, pass_hz, stop_hz) ** 2 - y
        guess = 2.0 * (single.get("sigma") or 0.1) if shape == "gaussian" \
            else 2.0 * (single.get("gamma") or 0.1)
        fit = least_squares(one, [math.log(max(guess, 0.01))],
                            bounds=([math.log(0.005)], [math.log(20.0)]))
        single_cost = min(single_cost, float(np.mean(fit.fun ** 2)))
    best["single_cost"] = single_cost
    if best["cost"] > PAIR_GAIN * single_cost:
        return None
    q = best["q"]
    root = math.sqrt(max(1.0 - 4.0 * q, 0.0))
    best["ratio"] = max((1.0 - 2.0 * q - root) / (2.0 * q), 1e-6)   # the weaker's share
    return best


def _structure(power: np.ndarray, max_lag: int) -> np.ndarray:
    """D(L) = <(P(t + L) - P(t))^2>: the power's structure function.

    Unlike the autocovariance it needs no sample mean, whose error over a
    band a few correlation times long would make every slow channel
    decorrelate early -- 0.1 Hz read as 0.2.
    """
    out = np.zeros(max_lag + 1)
    for k in range(1, max_lag + 1):
        out[k] = np.mean((power[k:] - power[:-k]) ** 2)
    return out


def _coherence(lags_s: np.ndarray, r: np.ndarray) -> float:
    below = np.flatnonzero(r < 1.0 / math.e)
    if below.size == 0 or below[0] == 0:
        return float("nan")
    k = below[0]
    r0, r1 = r[k - 1], r[k]
    return float(lags_s[k - 1] + (r0 - 1.0 / math.e) / (r0 - r1)
                 * (lags_s[k] - lags_s[k - 1]))


def _conflicts(record: Record, mode: Mode, others: list[Mode], t: np.ndarray,
               beat: np.ndarray, isolation_hz: float) -> list[Mode]:
    """The tracked modes that come within ``isolation_hz`` of the mode's beat."""
    f = record.freq_at(t)
    out = []
    for o in others:
        inside = o.covers(f)
        if not inside.any():
            continue
        mine = mode.delay_at(np.clip(f[inside], mode.f_lo, mode.f_hi))
        if np.nanmax(np.abs(o.delay_at(f[inside]) - mine)) < DUPLICATE_US:
            continue
        their = record.beat_hz(o.delay_at(f[inside]))
        if np.min(np.abs(their - beat[inside])) < isolation_hz:
            out.append(o)
    return out


def _their_beat(record: Record, other: Mode, f: np.ndarray) -> np.ndarray:
    return record.beat_hz(other.delay_at(np.clip(f, other.f_lo, other.f_hi)))


def _narrow(record: Record, mode: Mode, i0: int, i1: int, keep: np.ndarray,
            offset_hz: float = 0.0) -> float:
    """Mean power along ``mode``'s trace (+ offset) in +-CONFIRM_HZ, kept samples."""
    t = record.t_first_s + np.arange(i0, i1) / record.sr_hz
    coarse = t[::64]
    beat = np.interp(t, coarse, _beats(record, mode, coarse)) + offset_hz
    z, _ = _down(record, i0, i1, beat, pass_hz=CONFIRM_HZ,
                 stop_hz=STOP_RATIO * CONFIRM_HZ, trim_s=0.0)
    k = keep[::max(1, math.floor(record.sr_hz / OUT_RATE_HZ))][:z.size]
    return float(np.mean(np.abs(z[:k.size][k]) ** 2)) if k.any() else 0.0


def _separate(record: Record, mode: Mode, other: Mode, i0: int, i1: int,
              beat: np.ndarray, p_s: float, noise_offset: float) -> bool:
    """Whether the record holds a second echo on ``other``'s trace.

    Where the two traces are more than 2 CONFIRM_HZ apart, each is filtered
    to +-CONFIRM_HZ in turn; two echoes, if both carry CONFIRM_DB of the
    mode's power ``p_s``. With too little such overlap, not.
    """
    t = record.t_first_s + np.arange(i0, i1) / record.sr_hz
    f = record.freq_at(t)
    keep = other.covers(f) & (np.abs(_their_beat(record, other, f) - beat)
                              > 2.0 * CONFIRM_HZ)
    if np.count_nonzero(keep) < CONFIRM_MIN_S * record.sr_hz:
        return False
    floor = _narrow(record, mode, i0, i1, keep, noise_offset)
    threshold = 10.0 ** (CONFIRM_DB / 10.0) * p_s
    return all(_narrow(record, m, i0, i1, keep) - floor >= threshold
               for m in (mode, other))


def _recentre(z: np.ndarray, rate: float, t0_s: float, t: np.ndarray,
              p_n: float) -> np.ndarray:
    """The echo's offset from the trace, Hz, at times ``t``, from ``z``.

    ``z`` is the widest pass, starting at ``t0_s``; ``p_n`` its noise power.
    Zero where too few windows show an echo to fit.
    """
    w = max(16, round(RECENTRE_S * rate))
    n_fft = 4 * w
    freq = np.fft.fftfreq(n_fft, d=1.0 / rate)
    taper = np.hanning(w)
    # Noise in one bin of a window's spectrum, and ten times it as the bar.
    bar = 10.0 * p_n * float(np.sum(taper ** 2))
    tc, nu, weight = [], [], []
    for start in range(0, z.size - w + 1, w // 2):
        spec = np.abs(np.fft.fft(z[start:start + w] * taper, n_fft)) ** 2
        k = int(np.argmax(spec))
        if spec[k] > bar:
            tc.append(t0_s + (start + w / 2) / rate)
            nu.append(freq[k])
            weight.append(spec[k])
    tc, nu, weight = np.asarray(tc), np.asarray(nu), np.sqrt(np.asarray(weight))
    keep = np.ones(tc.size, dtype=bool)
    coef = None
    for _ in range(3):
        if np.count_nonzero(keep) < RECENTRE_ORDER + 3:
            return np.zeros(t.size)
        centre = float(np.mean(tc[keep]))
        coef = np.polyfit(tc[keep] - centre, nu[keep], RECENTRE_ORDER, w=weight[keep])
        keep = np.abs(np.polyval(coef, tc - centre) - nu) <= RECENTRE_OUTLIER_HZ
    lo, hi = float(tc[keep].min()), float(tc[keep].max())
    return np.polyval(coef, np.clip(t, lo, hi) - centre)


def _longest(mask: np.ndarray) -> tuple[int, int]:
    """The longest run of True in ``mask``, as [start, stop)."""
    if not mask.any():
        return 0, 0
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    starts, stops = edges[::2], edges[1::2]
    k = int(np.argmax(stops - starts))
    return int(starts[k]), int(stops[k])


def band_estimate(record: Record, mode: Mode, f_lo: float, f_hi: float,
                  others: list[Mode] = (), *,
                  passes: tuple[float, ...] = PASSES_HZ) -> BandEstimate:
    """One band of one mode, at the widest of ``passes`` it can be measured with.

    The band is cut to its longest stretch clear of every second echo the
    record confirms (:func:`_separate`) by the pass's stop plus
    ISOLATION_MARGIN_HZ; the noise reference is the quietest of the offsets
    -- an echo only adds.
    """
    est = BandEstimate(mode.name, f_lo, f_hi, used=False)
    t0, t1 = record.time_at([f_lo, f_hi])
    i0 = max(0, math.ceil((t0 - record.t_first_s) * record.sr_hz))
    i1 = min(record.samples.size, math.floor((t1 - record.t_first_s) * record.sr_hz))
    if i1 - i0 < MIN_BAND_S * record.sr_hz:
        est.reason = f"under {MIN_BAND_S:g} s of record"
        return est
    t = record.t_first_s + np.arange(i0, i1) / record.sr_hz
    f = record.freq_at(t)
    coarse = t[::64]
    beat = np.interp(t, coarse, _beats(record, mode, coarse))
    widest = STOP_RATIO * max(passes)
    if np.max(np.abs(beat)) > 0.45 * record.sr_hz - widest:
        est.reason = "the mode's beat is outside the record's band"
        return est
    offsets = [o for o in NOISE_OFFSETS_HZ
               if np.max(np.abs(beat + o)) < 0.45 * record.sr_hz - widest]
    if not offsets:
        est.reason = "no offset inside the record's band for the noise reference"
        return est

    # The mode's power, at the widest pass, and the quietest offset for noise.
    z, rate = _down(record, i0, i1, beat, pass_hz=max(passes),
                    stop_hz=widest)
    p_n, offset = min((float(np.mean(np.abs(
        _down(record, i0, i1, beat + o, pass_hz=max(passes), stop_hz=widest)[0]) ** 2)),
        o) for o in offsets)
    p_s = float(np.mean(np.abs(z) ** 2)) - p_n
    if p_s <= 0:
        est.reason = "no signal above the noise"
        return est
    correction = _recentre(z, rate, t[0] + TRIM_S, t, p_n)
    est.recentred_hz = float(np.max(np.abs(correction)))
    beat = beat + correction
    separate = [o for o in _conflicts(record, mode, list(others), coarse, beat[::64],
                                      widest + ISOLATION_MARGIN_HZ)
                if _separate(record, mode, o, i0, i1, beat, p_s, offset)]

    for pass_hz in passes:
        stop = STOP_RATIO * pass_hz
        clear = np.ones(t.size, dtype=bool)
        near = []
        for o in separate:
            close = o.covers(f) & (np.abs(_their_beat(record, o, f) - beat)
                                   < stop + ISOLATION_MARGIN_HZ)
            if close.any():
                near.append(o.name)
            clear &= ~close
        j0, j1 = _longest(clear)
        if j1 - j0 < MIN_BAND_S * record.sr_hz:
            est.reason = (f"a second echo ({', '.join(near)}) within "
                          f"{stop + ISOLATION_MARGIN_HZ:g} Hz")
            continue
        z, rate = _down(record, i0 + j0, i0 + j1, beat[j0:j1], pass_hz=pass_hz,
                        stop_hz=stop)
        noise, _ = _down(record, i0 + j0, i0 + j1, beat[j0:j1] + offset,
                         pass_hz=pass_hz, stop_hz=stop)
        if z.size < (MIN_BAND_S - 2 * TRIM_S) * rate:
            est.reason = "too short after trimming"
            continue
        p_n = float(np.mean(np.abs(noise) ** 2))
        p_s = float(np.mean(np.abs(z) ** 2)) - p_n
        snr = 10.0 * math.log10(p_s / p_n) if p_s > 0 else -math.inf
        if snr < MIN_SNR_DB:
            est.reason = f"SNR {snr:.1f} dB in {2 * pass_hz:g} Hz"
            est.snr_db = snr
            continue
        est.f_lo_mhz, est.f_hi_mhz = float(f[j0]), float(f[j1 - 1])
        est.cut_by = ", ".join(near)
        est.snr_db, est.pass_hz = snr, pass_hz
        est.seconds, est.rate_hz = z.size / rate, rate
        est.power = np.abs(z) ** 2
        max_lag = min(z.size // 3, round(MAX_LAG_S * rate))
        est.noise_r = np.abs(_autocorrelation(noise[:z.size], max_lag))
        est.noise_power = p_n
        est.used, est.reason = True, ""
        _fill(est, _pooled_fit([est]))
        return est
    return est


# --- a mode, a state ---------------------------------------------------------------

def bands_of(mode: Mode, *, band_mhz: float = BAND_MHZ) -> list[tuple[float, float]]:
    lo, hi = mode.f_lo + EDGE_LOW_MHZ, mode.f_hi - EDGE_TOP_MHZ
    if hi - lo < 0.5 * band_mhz:
        return []
    n = max(1, round((hi - lo) / band_mhz))
    edges = np.linspace(lo, hi, n + 1)
    return list(zip(edges[:-1].tolist(), edges[1:].tolist()))


@dataclass
class ModeEstimate:
    """A mode's fast part: the bands pooled into one fit."""

    mode: str
    bands: list[BandEstimate]
    spread_hz: float = float("nan")
    spread_scatter_hz: float = float("nan")   # jackknife over bands
    shape: str = ""
    rice_k_db: float | None = None
    coherence_s: float = float("nan")
    used: int = 0
    seconds: float = 0.0                       # of record pooled
    gamma: float = float("nan")                # scattered power's normalised variance
    gamma_error: float = float("nan")          # its standard error over the fades
    warning: str = ""
    detrended_s: float = float("nan")          # the running mean the level came off by
    notes: list = field(default_factory=list)
    # Rayleigh not rejected, yet gamma below 1: the K the moments give, and
    # the spread that K would make of the same record.
    k_if_rice_db: float | None = None
    spread_if_rice_hz: float = float("nan")
    # Two waves beating along the sweep: their beat, the delay split it
    # makes, the weaker's power against the stronger, and what one wave
    # would have read.
    beat_hz: float = float("nan")
    split_us: float = float("nan")
    weaker_db: float = float("nan")
    spread_one_wave_hz: float = float("nan")
    bound: str = ""                            # "upper" or "lower" when only a bound
    spread_gaussian_hz: float = float("nan")
    spread_lorentzian_hz: float = float("nan")
    residual_gaussian: float = float("nan")
    residual_lorentzian: float = float("nan")
    reason: str = ""

    def as_dict(self) -> dict:
        out = {k: v for k, v in asdict(self).items() if k != "bands"}
        out["bands"] = []
        for b in self.bands:
            row = {k: v for k, v in asdict(b).items()
                   if k not in ("autocorrelation", "power", "noise_r")}
            out["bands"].append(row)
        return out


def gamma_error(seconds: float, spread_hz: float) -> float:
    """Standard error of a Rayleigh gain's measured gamma over ``seconds``."""
    return GAMMA_SPREAD / math.sqrt(max(seconds * spread_hz, 1.0))


def _pooled_fit(bands: list[BandEstimate], *, rice: bool = True) -> dict:
    """One fit to the bands of a mode, pooled.

    The bands' structure functions are averaged over every pair of samples a
    lag apart, and their moments over every sample, as one record of the
    mode: a band holds too few fades for its own.

    The Rice factor comes first, from the moments alone (Greenstein's, with
    the noise reference's power taken off): the scattered part's normalised
    variance gamma is 1 for Rayleigh and (1 + 2K) / (1 + K)^2 below it, and
    a K is kept only where gamma is clear of 1 by K_SIGMAS standard errors.
    Then C(L) = C(0) - D(L)/2 is solved for the scattered part's |R| through
    the specular quadratic, with C(0) from the theory for that K; for
    Rayleigh the scale comes from the second moment, which moves with D.
    """
    rate = bands[0].rate_hz
    pass_hz = bands[0].pass_hz
    stop_hz = STOP_RATIO * pass_hz
    if any(b.pass_hz != pass_hz for b in bands):
        raise ValueError("bands pooled must share their low-pass")
    n_lag = min(b.noise_r.size for b in bands)
    d = np.zeros(n_lag)
    r_n = np.zeros(n_lag)
    weight = np.zeros(n_lag)
    m1 = m2 = p_n = 0.0
    samples = 0
    for b in bands:
        n_b = b.power.size
        w = np.maximum(n_b - np.arange(n_lag), 0).astype(np.float64)
        d += w * _structure(b.power, n_lag - 1)
        r_n += w * b.noise_r[:n_lag]
        weight += w
        m1 += float(np.sum(b.power))
        m2 += float(np.sum(b.power ** 2))
        p_n += b.noise_power * n_b
        samples += n_b
    d /= np.maximum(weight, 1.0)
    r_n /= np.maximum(weight, 1.0)
    m1, m2, p_n = m1 / samples, m2 / samples, p_n / samples
    seconds = samples / rate
    lags = np.arange(n_lag) / rate

    p_s = max(m1 - p_n, 1e-30)
    var_s = (m2 - m1 ** 2) - 2.0 * p_s * p_n - p_n ** 2
    gamma = var_s / p_s ** 2

    def scattered(k):
        if k > 0:
            p_spec, p_sc = p_s * k / (k + 1.0), p_s / (k + 1.0)
            p_w = p_sc + p_n
            c0 = p_w ** 2 + 2.0 * p_spec * p_w
        else:
            # Rayleigh: the received power is exponential, m2 = 2 (P_s + P_n)^2.
            p_tot = math.sqrt(m2 / 2.0)
            p_spec, p_sc = 0.0, max(p_tot - p_n, 1e-30)
            c0 = p_tot ** 2
        c = np.clip(c0 - 0.5 * d, 0.0, None)
        r_sn = np.sqrt(p_spec ** 2 + c) - p_spec
        return np.clip((r_sn - r_n) / p_sc, 1e-6, None)

    r = scattered(0.0)
    fit = _fit(lags, r, pass_hz, stop_hz)
    k = 0.0
    if rice and 0.0 < gamma < 1.0:
        # The fades gamma's error is counted over need a spread that does not
        # itself assume Rayleigh -- under a specular part, Rayleigh's reading
        # is several times slow and would hide the very K being tested. The
        # power's own variance as the scale reads any K within 0.7 to 1 of
        # the truth: 1 - |R|^2 of D(L) / 2 var is 1 - r^2 for Rayleigh and
        # 1 - r for a strong specular part.
        var = max(m2 - m1 ** 2 - p_n ** 2 - 2.0 * p_s * p_n, 1e-30)
        robust = np.sqrt(np.clip(1.0 - 0.5 * d / var, 1e-6, None))
        rough = _fit(lags, robust, pass_hz, stop_hz)
        guess = (2.0 * rough["sigma"] if "sigma" in rough
                 else 2.0 * _slowest_sigma(lags[-1]))
        root = math.sqrt(1.0 - gamma)
        k_moments = root / (1.0 - root)
        alternative = _fit(lags, scattered(k_moments), pass_hz, stop_hz)
        error = gamma_error(seconds, guess)
        if 1.0 - gamma > K_SIGMAS * error:
            k = k_moments
            r = scattered(k)
            fit = alternative
    out = {"k": k, "gamma": gamma, "lags": lags, "scattered": r, "seconds": seconds,
           "coherence": _coherence(lags, r), "last_lag": float(lags[-1])}
    if rice and 0.0 < gamma < 1.0:
        out["gamma_error"] = error
        if k == 0.0 and "sigma" in alternative:
            # Rayleigh not rejected, but not excluded either: what the spread
            # would be if the specular part the moments hint at were real.
            out["k_if_rice"] = k_moments
            out["alternative"] = alternative
    out.update({key: v for key, v in fit.items() if key != "gamma"})
    if "gamma" in fit:
        out["gamma_l"] = fit["gamma"]
    if k == 0.0:
        pair = _fit_pair(lags, r, pass_hz, stop_hz, fit)
        if pair is not None:
            out["pair"] = pair
    return out


def _slowest_sigma(last_lag_s: float) -> float:
    """The Gaussian sigma whose |R| reaches FIT_LOW at the last lag."""
    return math.sqrt(-math.log(FIT_LOW) / (2.0 * math.pi ** 2)) / max(last_lag_s, 1e-9)


def _fill(est, fit: dict) -> None:
    """A band's or a mode's spreads and shape from a fit."""
    est.coherence_s = fit["coherence"]
    if isinstance(est, BandEstimate):
        est.autocorrelation = [round(float(v), 4) for v in
                               fit["scattered"][:round(5 * est.rate_hz) + 1]]
    if "sigma" not in fit:
        if not fit["crossed"]:
            # Still above FIT_LOW at the last lag: slower than that.
            est.spread_gaussian_hz = 2.0 * _slowest_sigma(fit["last_lag"])
            est.shape, est.bound = "gaussian", "upper"
        else:
            # Below it within FIT_MIN_POINTS lags: faster than the rate resolves.
            lag = (FIT_MIN_POINTS + 1) * float(fit["lags"][1])
            est.spread_gaussian_hz = 2.0 * _slowest_sigma(lag)
            est.shape, est.bound = "gaussian", "lower"
        return
    est.spread_gaussian_hz = 2.0 * fit["sigma"]
    est.spread_lorentzian_hz = 2.0 * fit["gamma_l"]
    est.residual_gaussian, est.residual_lorentzian = fit["res_g"], fit["res_l"]
    est.shape = "gaussian" if fit["res_g"] <= fit["res_l"] else "lorentzian"


def _detrended(bands: list[BandEstimate], window_s: float) -> list[BandEstimate]:
    """The bands with their power divided by its running mean over ``window_s``."""
    out = []
    for b in bands:
        n = max(3, round(window_s * b.rate_hz)) | 1
        kernel = np.ones(n) / n
        padded = np.pad(b.power, n // 2, mode="reflect")
        level = np.convolve(padded, kernel, mode="valid")
        level = level / float(np.mean(level))
        c = BandEstimate(b.mode, b.f_lo_mhz, b.f_hi_mhz, b.used, rate_hz=b.rate_hz,
                         pass_hz=b.pass_hz)
        c.seconds = b.seconds
        c.power = b.power / level
        inverse = float(np.mean(1.0 / level))
        c.noise_power = b.noise_power * inverse
        c.noise_r = b.noise_r * inverse
        out.append(c)
    return out


def _reading(fit: dict, shape: str, pair: bool) -> float:
    """The spread a fit gives, as one wave or as a pair."""
    if pair:
        return fit["pair"]["spread"] if "pair" in fit else float("nan")
    if "sigma" not in fit:
        return float("nan")
    return 2.0 * (fit["gamma_l"] if shape == "lorentzian" else fit["sigma"])


def _aggregate(name: str, bands: list[BandEstimate]) -> ModeEstimate:
    out = ModeEstimate(name, bands)
    good = [b for b in bands if b.used and b.power is not None]
    if not good:
        return out
    fit = _pooled_fit(good)
    raw = fit
    if (np.isfinite(fit["coherence"]) and fit["gamma"] - 1.0
            > K_SIGMAS * gamma_error(fit["seconds"],
                                     # A Gaussian's spread from its 1/e time.
                                     math.sqrt(2.0) / (math.pi * max(fit["coherence"],
                                                                     1e-3)))):
        window = TREND_COHERENCES * fit["coherence"]
        if window < 0.5 * min(b.seconds for b in good):
            good = _detrended(good, window)
            fit = _pooled_fit(good)
            out.detrended_s = window
            if "pair" in raw and "pair" not in fit:
                # The record weighted by its level shows two waves that the
                # level-flattened one does not: they are there where the
                # mode is strong, not all along it.
                p = raw["pair"]
                weaker = 10.0 * math.log10(p["ratio"])
                out.notes.append(
                    f"before the level came off, two waves beating at "
                    f"{p['beat_hz']:.2f} Hz (the weaker {weaker:.1f} dB, spread "
                    f"{p['spread']:.3f} Hz {p['shape']}) fitted better: in the "
                    "mode's stronger stretches only")
    _fill(out, fit)
    out.used = len(good)
    out.seconds = fit["seconds"]
    out.gamma = fit["gamma"]
    if not np.isfinite(out.spread_gaussian_hz):
        return out
    out.rice_k_db = 10.0 * math.log10(fit["k"]) if fit["k"] > 0 else None
    out.spread_hz = (out.spread_lorentzian_hz if out.shape == "lorentzian"
                     else out.spread_gaussian_hz)
    pair = "pair" in fit and not out.bound
    if pair:
        p = fit["pair"]
        out.spread_one_wave_hz = out.spread_hz
        out.shape, out.spread_hz = p["shape"], p["spread"]
        nan = float("nan")
        out.spread_gaussian_hz = p["spread"] if p["shape"] == "gaussian" else nan
        out.spread_lorentzian_hz = p["spread"] if p["shape"] == "lorentzian" else nan
        out.beat_hz = p["beat_hz"]
        out.weaker_db = 10.0 * math.log10(p["ratio"])
        # One wave's 1/e time, not the beat's.
        width = math.sqrt(2.0) if p["shape"] == "gaussian" else 1.0
        out.coherence_s = width / (math.pi * p["spread"])
    elif "alternative" in fit:
        alt = fit["alternative"]
        out.k_if_rice_db = 10.0 * math.log10(fit["k_if_rice"])
        out.spread_if_rice_hz = 2.0 * (alt["gamma"] if out.shape == "lorentzian"
                                       else alt["sigma"])
    out.gamma_error = gamma_error(fit["seconds"], out.spread_hz)
    if fit["gamma"] - 1.0 > K_SIGMAS * out.gamma_error:
        # Fading alone cannot make the power vary more than Rayleigh's: the
        # level changes along the bands too. Slowly, it scales D and the
        # second moment alike and the spread stands; but an echo that comes
        # and goes, or a trace the echo strays from, fades on its own.
        out.warning = (f"power varies more than Rayleigh fading (gamma "
                       f"{fit['gamma']:.2f} +- {out.gamma_error:.2f}): the level "
                       "changes along the bands too")
    if len(good) >= 3 and not out.bound:
        # Jackknife over bands: the scatter a band's absence makes.
        loo = [_reading(_pooled_fit(good[:i] + good[i + 1:]), out.shape, pair)
               for i in range(len(good))]
        loo = np.asarray([v for v in loo if np.isfinite(v)])
        if loo.size >= 2:
            n = loo.size
            out.spread_scatter_hz = float(math.sqrt((n - 1) / n
                                                    * np.sum((loo - loo.mean()) ** 2)))
    return out


def estimate(record: Record, state: ChannelState, *,
             band_mhz: float = BAND_MHZ) -> list[ModeEstimate]:
    """Every mode of ``state`` measured on ``record``, band by band."""
    out = []
    for m in state.modes:
        others = [o for o in state.modes if o is not m]
        edges = bands_of(m, band_mhz=band_mhz)
        bands = [band_estimate(record, m, lo, hi, others) for lo, hi in edges]
        # One low-pass for the mode, so that its bands pool: the narrowest
        # any of them needed.
        used = [b.pass_hz for b in bands if b.used]
        if used:
            common = min(used)
            bands = [b if not b.used or b.pass_hz == common
                     else band_estimate(record, m, lo, hi, others, passes=(common,))
                     for b, (lo, hi) in zip(bands, edges)]
        e = _aggregate(m.name, bands)
        if np.isfinite(e.beat_hz):
            # Along an up-sweep a delay split of tau beats at rate * tau.
            e.split_us = e.beat_hz / record.rate_hz_s * 1e6
        out.append(e)
    return out


def apply(state: ChannelState, estimates: list[ModeEstimate], *,
          source: str = "") -> list[str]:
    """Write measured spreads, shapes, Rice factors and o/x splits into ``state``.

    A mode no band could measure keeps its prior; one only bounded gets the
    bound, marked as one. Returns a line per mode saying what was done.
    """
    by_name = {e.mode: e for e in estimates}
    lines = []
    for m in state.modes:
        e = by_name.get(m.name)
        if e is None or not np.isfinite(e.spread_hz):
            lines.append(f"{m.name}: no band measured; the prior stays")
            continue
        note = (f"second dechirp along the trace{', ' + source if source else ''}: "
                f"{e.used} band(s) of {BAND_MHZ:g} MHz")
        if e.bound:
            m.doppler_spread_hz = e.spread_hz
            why = ("slower than the lags resolve" if e.bound == "upper"
                   else "faster than the analysis rate resolves")
            m.provenance["doppler_spread"] = Provenance(
                MEASURED, f"{note}; an {e.bound} bound, the channel being {why}")
        else:
            m.doppler_spread_hz = e.spread_hz
            m.doppler_shape = e.shape
            scatter = (f", band-to-band {e.spread_scatter_hz:.2f} Hz"
                       if np.isfinite(e.spread_scatter_hz) else "")
            m.provenance["doppler_spread"] = Provenance(
                MEASURED, f"{note}; {e.shape} by autocorrelation fit{scatter}")
            m.rician_k_db = e.rice_k_db
            m.provenance["fading"] = Provenance(
                MEASURED, f"{note}; Rice K by moments "
                          + ("Rayleigh" if e.rice_k_db is None
                             else f"{e.rice_k_db:.1f} dB"))
            if np.isfinite(e.split_us):
                m.ox_split_us, m.x_to_o_db = float(e.split_us), float(e.weaker_db)
                m.provenance["magnetoionic"] = Provenance(
                    MEASURED, f"{note}; two waves beating at {e.beat_hz:.2f} Hz along "
                              "the sweep -- the weaker taken as x, no differential "
                              "Doppler shift")
        m.provenance["doppler_shift"] = Provenance(
            m.origin("doppler_shift") or "prior",
            "not measurable from one up-sweep: it moves the beat as a delay change "
            "would")
        lines.append(f"{m.name}: spread {e.spread_hz:.3f} Hz {e.shape}"
                     + (f" ({e.bound} bound)" if e.bound else "")
                     + (f", K {e.rice_k_db:.1f} dB" if e.rice_k_db is not None else "")
                     + (f", o/x split {e.split_us:.1f} us at {e.weaker_db:.1f} dB"
                        if np.isfinite(e.split_us) else "")
                     + f", {e.used} band(s)")
    return lines


def measure(path: str | Path, state: ChannelState | None = None, *,
            method: str = "contour", band_mhz: float = BAND_MHZ,
            stations=None) -> tuple[ChannelState, list[ModeEstimate]]:
    """A sounding's state with its fast part measured from its own phase."""
    from . import observe

    if state is None:
        state = observe.from_sounding(path, method=method, stations=stations)
    record = record_of(path, stations)
    estimates = estimate(record, state, band_mhz=band_mhz)
    for line in apply(state, estimates, source=record.source):
        state.notes.append(f"fast part: {line}")
    return state, estimates


def format_estimates(estimates: list[ModeEstimate]) -> str:
    lines = []
    for e in estimates:
        head = f"{e.mode}: "
        if np.isfinite(e.spread_hz):
            head += (f"spread {e.spread_hz:.3f} Hz {e.shape}"
                     + (f" ({e.bound} bound)" if e.bound else "")
                     + (f" +- {e.spread_scatter_hz:.3f}"
                        if np.isfinite(e.spread_scatter_hz) else "")
                     + f", coherence {e.coherence_s:.2f} s"
                     + (f", K {e.rice_k_db:.1f} dB" if e.rice_k_db is not None
                        else ", Rayleigh")
                     + f"; {e.used} band(s), {e.seconds:.0f} s, gamma {e.gamma:.2f}"
                     + (f" +- {e.gamma_error:.2f}" if np.isfinite(e.gamma_error)
                        else ""))
            if np.isfinite(e.beat_hz):
                head += (f"\n    two waves beating at {e.beat_hz:.2f} Hz: split "
                         f"{e.split_us:.1f} us, the weaker {e.weaker_db:.1f} dB; "
                         f"as one wave the spread would read "
                         f"{e.spread_one_wave_hz:.3f} Hz")
            if np.isfinite(e.detrended_s):
                head += (f"\n    the level varied more than fading can: taken off "
                         f"by a {e.detrended_s:.1f} s running mean")
            if e.warning:
                head += f"\n    warning: {e.warning}"
            for note in e.notes:
                head += f"\n    note: {note}"
            if e.k_if_rice_db is not None:
                head += (f"\n    Rayleigh not rejected; if the moments' K "
                         f"{e.k_if_rice_db:.1f} dB is real, the spread is "
                         f"{e.spread_if_rice_hz:.3f} Hz")
        else:
            head += "not measured"
        lines.append(head)
        for b in e.bands:
            if not b.used:
                lines.append(f"    {b.f_lo_mhz:6.2f}-{b.f_hi_mhz:6.2f} MHz  skipped: "
                             f"{b.reason}")
                continue
            lines.append(
                f"    {b.f_lo_mhz:6.2f}-{b.f_hi_mhz:6.2f} MHz  SNR {b.snr_db:5.1f} dB "
                f"in {2 * b.pass_hz:g} Hz, trace moved {b.recentred_hz:4.1f} Hz  "
                f"gauss {b.spread_gaussian_hz:6.3f} Hz "
                f"(misfit {b.residual_gaussian:.3f})  "
                f"lorentz {b.spread_lorentzian_hz:6.3f} Hz "
                f"(misfit {b.residual_lorentzian:.3f})  -> {b.shape}"
                + (f"  cut for {b.cut_by}" if b.cut_by else ""))
    return "\n".join(lines)


# --- the estimator, calibrated on its own soundings ----------------------------------

def calibrate(spreads=(0.1, 0.25, 0.5, 1.0, 2.0), *,
              shapes=("gaussian", "lorentzian"), k_db: float | None = None,
              ox_split_us: float | None = None, x_to_o_db: float = 0.0,
              realisations: int = 2, seed: int = 0, state: ChannelState | None = None,
              workdir: str | Path | None = None) -> list[dict]:
    """Sound known fast parts through the law preset and read them back.

    The closed loop applied to the fast part: what the estimator returns for
    a spread, a Rice factor, an o/x split that are known, with the same
    sweep, sample rate and processing as a real sounding -- the tracker's
    traces included. Its bias and scatter are what a measured value is read
    against. One row per sounding, of the mode measured on most bands.
    """
    import shutil
    import tempfile

    from . import observe, presets, sounding

    base = state or presets.law_state()
    own = workdir is None
    folder = Path(tempfile.mkdtemp(prefix="muf-fast-")) if own else Path(workdir)
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for shape in shapes:
            for spread in spreads:
                for k in range(realisations):
                    truth = ChannelState.from_dict(base.to_dict())
                    for m in truth.modes:
                        m.doppler_spread_hz, m.doppler_shape = spread, shape
                        m.rician_k_db = k_db
                        m.ox_split_us, m.x_to_o_db = ox_split_us, x_to_o_db
                    path = folder / f"cal_{shape}_{spread:g}_{k}.lfs"
                    sounding.write_lfs(truth, path, seed=seed + k)
                    got = observe.from_sounding(path)
                    best = max(estimate(record_of(path), got), key=lambda e: e.used)
                    rows.append({"shape": shape, "spread_hz": spread, "k_db": k_db,
                                 "split_us": ox_split_us, "x_to_o_db": x_to_o_db,
                                 "seed": seed + k, "mode": best.mode,
                                 "bands": best.used, "bound": best.bound,
                                 "got_hz": best.spread_hz,
                                 "got_scatter_hz": best.spread_scatter_hz,
                                 "got_shape": best.shape, "got_k_db": best.rice_k_db,
                                 "got_split_us": best.split_us,
                                 "got_weaker_db": best.weaker_db,
                                 "one_wave_hz": best.spread_one_wave_hz})
                    path.unlink()
    finally:
        if own:
            shutil.rmtree(folder, ignore_errors=True)
    return rows


def format_calibration(rows: list[dict]) -> str:
    lines = [f"{'truth':>28}  {'read':>28}  bands"]
    for r in rows:
        truth = (f"{r['spread_hz']:g} Hz {r['shape'][:5]}"
                 + (f" K {r['k_db']:g}" if r["k_db"] is not None else "")
                 + (f" split {r['split_us']:g}" if r["split_us"] is not None else ""))
        read = (f"{r['got_hz']:.3f} Hz {r['got_shape'][:5]}"
                + (f" K {r['got_k_db']:.1f}" if r["got_k_db"] is not None else "")
                + (f" split {r['got_split_us']:.1f}"
                   if np.isfinite(r["got_split_us"]) else "")
                + (f" ({r['bound']} bound)" if r["bound"] else ""))
        lines.append(f"{truth:>28}  {read:>28}  {r['bands']}")
    return "\n".join(lines)
