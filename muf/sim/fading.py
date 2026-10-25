"""Tap gains of the Gaussian-scatter HF channel (Watterson; ITU-R F.1487).

F.1487 Annex 1 models each path's gain as a complex Gaussian process --
zero mean, independent real and imaginary parts of equal power, so the
envelope is Rayleigh -- with a Gaussian Doppler power spectrum

    v(nu) = exp(-(nu - nu_s)^2 / (2 sigma^2)) / (sigma sqrt(2 pi))

and Annex 3 quotes its *frequency spread* as 2 sigma. ``spread_hz`` here is
that 2 sigma, so the recommendation's table can be typed in as it stands:
mid-latitude moderate is a spread of 0.5 Hz, sigma 0.25 Hz.

A second shape is offered beside it: a **Lorentzian** spectrum, the
exponential autocorrelation of Vogler and Hoffmeyer's wideband model (NTIA
90-255, 92-284), which the DAMSON measurements also favour for disturbed
paths (Angling et al., Radio Sci. 1998). Its second moment is infinite, so
it has no 2 sigma; its ``spread_hz`` is the full width at half power, the
2 sigma_D the ITS verification reads off its -3 dB contours. For a Gaussian
that width would be 1.18 times F.1487's spread -- the two are not the same
number, and a state says which shape it means.

A gain is drawn once, on a coarse grid a fixed multiple of sigma, and read
at any time by interpolation. That is what lets a long sounding (10 M
samples) and the transfer function of the same realisation agree, and keeps
the cost independent of the signal's sample rate. The Doppler shift is
applied analytically at the read-out time, so a large shift costs nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Coarse-grid rate as a multiple of sigma. At 32 sigma neighbouring samples
#: correlate to 0.981, and linear interpolation loses 0.04 dB of power midway
#: between them -- far inside any statistic this is used for.
OVERSAMPLE = 32.0

#: The same for a Lorentzian, as a multiple of its half width. Its
#: autocorrelation has a cusp at zero lag, which linear interpolation rounds
#: off over a grid step: at 64 the step is a tenth of the coherence time,
#: and a power structure function read over its first third -- as
#: :mod:`muf.sim.fast` reads it -- comes out 15 % slow. At 320 the step is a
#: fiftieth and 0.4 % of the power folds from beyond the grid's Nyquist.
LORENTZ_OVERSAMPLE = 320.0

#: Floor on the coarse rate, Hz, so a nearly frozen channel is still sampled
#: often enough to interpolate smoothly.
MIN_RATE_HZ = 4.0

GAUSSIAN = "gaussian"
LORENTZIAN = "lorentzian"
SHAPES = (GAUSSIAN, LORENTZIAN)


def check_shape(shape: str) -> str:
    if shape not in SHAPES:
        raise ValueError(f"Doppler spectrum shape {shape!r} is not one of {SHAPES}")
    return shape

#: Extra length drawn at each end, in units of the correlation time 1/sigma,
#: and thrown away: the spectrum is shaped by FFT, which is circular, and the
#: margin keeps the two ends of the record from correlating with each other.
WRAP_MARGIN = 4.0


@dataclass
class TapGain:
    """One path's fading gain over ``[start_s, start_s + duration_s]``.

    Call it with times in seconds. Its ensemble mean power is one:
    ``E|g|^2 = 1``, specular and scattered parts together.
    """

    spread_hz: float
    shift_hz: float
    k_db: float | None
    start_s: float
    rate_hz: float
    samples: np.ndarray | None      # the scattered part, on the coarse grid
    phase: float                    # of the specular part
    shape: str = GAUSSIAN

    @property
    def sigma_hz(self) -> float:
        """Gaussian sigma, or the Lorentzian's half width at half power."""
        return self.spread_hz / 2.0

    def __call__(self, t_s) -> np.ndarray:
        t = np.asarray(t_s, dtype=np.float64)
        k = None if self.k_db is None else 10.0 ** (self.k_db / 10.0)
        if self.samples is None:
            # Not fading: unit amplitude, its phase the only thing drawn.
            g = np.full(t.shape, np.exp(1j * self.phase), dtype=np.complex128)
        else:
            pos = (t - self.start_s) * self.rate_hz
            grid = np.arange(self.samples.size, dtype=np.float64)
            scattered = (np.interp(pos, grid, self.samples.real)
                         + 1j * np.interp(pos, grid, self.samples.imag))
            if k is None:
                g = scattered
            else:
                g = (math.sqrt(k / (k + 1.0)) * np.exp(1j * self.phase)
                     + math.sqrt(1.0 / (k + 1.0)) * scattered)
        if self.shift_hz:
            g = g * np.exp(2j * np.pi * self.shift_hz * t)
        return g


def draw(duration_s: float, spread_hz: float, *, shift_hz: float = 0.0,
         k_db: float | None = None, rng: np.random.Generator | None = None,
         start_s: float = 0.0, shape: str = GAUSSIAN) -> TapGain:
    """Draw one fading gain.

    ``spread_hz`` is F.1487's frequency spread, 2 sigma, for a Gaussian
    spectrum and the full width at half power for a Lorentzian; zero gives a
    gain of constant unit amplitude (and random phase), which is a non-fading
    path. ``k_db`` adds a specular part that many dB above the scattered one.
    """
    check_shape(shape)
    rng = rng if rng is not None else np.random.default_rng()
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    if spread_hz < 0:
        raise ValueError(f"negative Doppler spread {spread_hz}")
    if spread_hz == 0:
        return TapGain(0.0, float(shift_hz), k_db, float(start_s), 0.0, None, phase,
                       shape)

    half = spread_hz / 2.0
    oversample = OVERSAMPLE if shape == GAUSSIAN else LORENTZ_OVERSAMPLE
    rate = max(MIN_RATE_HZ, oversample * half)
    margin = math.ceil(WRAP_MARGIN / half * rate)
    n_keep = math.ceil(max(duration_s, 0.0) * rate) + 2
    n = n_keep + 2 * margin
    white = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / math.sqrt(2.0)
    nu = np.fft.fftfreq(n, d=1.0 / rate)
    # Amplitude response: the square root of the power spectrum.
    if shape == GAUSSIAN:
        amp = np.exp(-nu ** 2 / (4.0 * half ** 2))
    else:
        amp = 1.0 / np.sqrt(1.0 + (nu / half) ** 2)
    # Normalised by the filter's power gain, not by the sample's power: one
    # realisation of a slowly fading path is *meant* to stray from unit mean.
    amp = amp / math.sqrt(np.mean(amp ** 2))
    scattered = np.fft.ifft(np.fft.fft(white) * amp)[margin:margin + n_keep]
    return TapGain(float(spread_hz), float(shift_hz), k_db, float(start_s), rate,
                   scattered.astype(np.complex128), phase, shape)


def autocorrelation(lag_s, spread_hz: float, shape: str = GAUSSIAN) -> np.ndarray:
    """The scattered part's normalised autocorrelation |R(lag)| / R(0).

    exp(-2 pi^2 sigma^2 lag^2) for a Gaussian spectrum of sigma = spread/2;
    exp(-2 pi gamma |lag|) for a Lorentzian of half width gamma = spread/2.
    """
    check_shape(shape)
    lag = np.abs(np.asarray(lag_s, dtype=np.float64))
    half = spread_hz / 2.0
    if shape == GAUSSIAN:
        return np.exp(-2.0 * np.pi ** 2 * half ** 2 * lag ** 2)
    return np.exp(-2.0 * np.pi * half * lag)


def doppler_spectrum(gain: np.ndarray, rate_hz: float, segment_s: float = 10.0):
    """Welch power spectrum of a sampled gain: (freq Hz, power, RMS width Hz).

    For tests and for the closed-loop check; the width is sigma, half of
    F.1487's spread, when the input is a Gaussian-scatter gain.
    """
    gain = np.asarray(gain)
    n = int(segment_s * rate_hz)
    if n < 8 or gain.size < n:
        raise ValueError("record shorter than one Welch segment")
    win = np.hanning(n)
    starts = range(0, gain.size - n + 1, n // 2)
    power = np.mean([np.abs(np.fft.fft(win * gain[s:s + n])) ** 2 for s in starts],
                    axis=0)
    freq = np.fft.fftfreq(n, d=1.0 / rate_hz)
    order = np.argsort(freq)
    freq, power = freq[order], power[order]
    centre = np.sum(freq * power) / np.sum(power)
    width = math.sqrt(np.sum((freq - centre) ** 2 * power) / np.sum(power))
    return freq, power, width
