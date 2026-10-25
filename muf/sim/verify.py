"""Level 0 against theory: the generator checked the way simulators are checked.

The unit tests hold each piece to what it was told. This module holds the
whole generator to closed-form results, by the procedures the literature on
HF channel simulators uses, so that a report can say *which* check, by
*whose* procedure, against *what* number:

* **Fading statistics** -- the envelope's distribution against Rayleigh or
  Rice, its autocorrelation against the spectrum's transform, the level
  crossing rate and average fade duration against Rice's formulas, the
  Doppler width against the spread asked for. These are the standard checks
  of a Gaussian-scatter (Watterson) tap generator.
* **The scattering function** S(tau, nu), estimated from the channel's own
  H(t, f) and read at its paths: delay, power share, Doppler shift, spread.
  This is how ITS verified its wideband transfer-function program
  (Sutherland, NTIA Report 98-348, 1998, sec. 4: spectrally averaged
  scattering functions, parameters read at -3 dB). A dispersive mode adds
  one more number: across a band B it smears over |GDD| * B of delay.
* **A reference modem**: DBPSK through one flat Rayleigh path, whose bit
  error rate has a closed form that includes the Doppler spread. It checks
  the fading, the Doppler and the SNR calibration at once -- the calibration
  modem tests depend on (ITU-R F.1487 Annex 2 measures modems by BER against
  S/N for exactly this reason).

Every run lasts as long as ITU-R F.1487 Annex 3 sec. 6 asks of a modem test:
the longer of 3000 over the Doppler spread and 100 over BER times bit rate.
Shorter runs are allowed for tests, and the tolerances widen with them.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from . import channel, fading
from .state import ChannelState, Circuit, Mode

#: F.1487 Annex 3 sec. 6: a test lasts 3000 / Doppler spread seconds ...
F1487_FADES = 3000.0
#: ... or 100 / (BER * bit rate), whichever is longer.
F1487_ERRORS = 100.0


def test_length_s(spread_hz: float, ber: float | None = None,
                  bit_rate: float | None = None) -> float:
    """F.1487 Annex 3 sec. 6's length of test, seconds."""
    out = F1487_FADES / spread_hz if spread_hz > 0 else 0.0
    if ber and bit_rate:
        out = max(out, F1487_ERRORS / (ber * bit_rate))
    return out


@dataclass
class Check:
    """One number against the number theory gives, within a tolerance."""

    group: str
    name: str
    measured: float
    expected: float
    tolerance: float
    relative: bool = True
    unit: str = ""
    basis: str = ""

    @property
    def error(self) -> float:
        diff = self.measured - self.expected
        return diff / self.expected if self.relative and self.expected else diff

    @property
    def passed(self) -> bool:
        return bool(np.isfinite(self.measured) and abs(self.error) <= self.tolerance)

    def as_dict(self) -> dict:
        return {**asdict(self), "error": self.error, "passed": self.passed}


# --- fading statistics --------------------------------------------------------

def rice_cdf(r, k: float = 0.0) -> np.ndarray:
    """CDF of the envelope of unit mean power with Rice factor ``k`` (linear)."""
    from scipy.stats import ncx2

    r = np.asarray(r, dtype=np.float64)
    if k <= 0:
        return 1.0 - np.exp(-r ** 2)
    return ncx2.cdf(2.0 * (k + 1.0) * r ** 2, 2, 2.0 * k)


def crossing_rate(rho, rms_width_hz: float, k: float = 0.0) -> np.ndarray:
    """Upward crossings per second of level ``rho`` (re RMS) by the envelope.

    Rice's formula for a complex Gaussian process whose Doppler spectrum has
    RMS width ``rms_width_hz`` about its centre, with a specular part ``k``
    times stronger at the same Doppler. For Rayleigh it is
    2 sqrt(pi) beta rho exp(-rho^2) -- Jakes's sqrt(2 pi) f_D rho exp(-rho^2)
    with beta = f_D / sqrt 2.
    """
    from scipy.special import i0e

    rho = np.asarray(rho, dtype=np.float64)
    a = 2.0 * rho * math.sqrt(k * (k + 1.0))
    # i0e(a) = exp(-a) I0(a), which keeps large K finite.
    return (2.0 * math.sqrt(math.pi * (k + 1.0)) * rms_width_hz * rho
            * np.exp(-k - (k + 1.0) * rho ** 2 + a) * i0e(a))


def measured_crossings(envelope: np.ndarray, rate_hz: float,
                       level: float) -> tuple[float, float]:
    """(upward crossings per second, mean duration below, s) at ``level``."""
    below = envelope < level
    ups = np.flatnonzero(below[:-1] & ~below[1:])
    duration = envelope.size / rate_hz
    lcr = ups.size / duration
    afd = below.mean() / lcr if lcr > 0 else float("nan")
    return lcr, afd


def rice_k_moments(power: np.ndarray, noise_power: float = 0.0) -> float:
    """Rice factor from the first two moments of |g|^2 (Greenstein et al., 1999).

    gamma = Var(|g|^2) / E(|g|^2)^2 is 1 for Rayleigh and falls towards 0 as
    the specular part grows; K = sqrt(1 - gamma) / (1 - sqrt(1 - gamma)).
    ``noise_power``, when the samples carry additive noise of that power,
    is taken out first: Var|s + n|^2 = Var|s|^2 + 2 P_s P_n + P_n^2.
    """
    p = np.asarray(power, dtype=np.float64)
    p_s = p.mean() - noise_power
    if p_s <= 0:
        return float("nan")
    var = p.var() - 2.0 * p_s * noise_power - noise_power ** 2
    gamma = var / p_s ** 2
    if gamma >= 1.0:
        return 0.0
    if gamma <= 0.0:
        return float("inf")
    root = math.sqrt(1.0 - gamma)
    return root / (1.0 - root)


#: Periodogram segments times smoothing bins a half-power width is read on:
#: below a few hundred, one lumpy bin near the peak sets the threshold.
WIDTH_AVERAGING = 200


def half_power_width(freq: np.ndarray, power: np.ndarray, segments: int = 1) -> float:
    """Full width at half the peak of a Doppler periodogram, Hz.

    Two things bias the obvious reading. The peak bin's own noise lifts the
    threshold -- the highest of fifty bins that scatter by 8 % sits 16 % up --
    so the peak is the vertex of a parabola fitted to log power over the top
    of the spectrum, forty-odd bins whose noise averages out. And a walk out
    from the peak stops at the first noisy dip, so the width is the count of
    bins at or above half that peak. A short record (few ``segments``) is
    smoothed first, across enough bins to make ``WIDTH_AVERAGING`` in all
    but never more than half its RMS width, which widens a Gaussian by 1 %.
    """
    p = np.asarray(power, dtype=np.float64)
    step = float(np.median(np.diff(freq)))
    _centre, rms = _spectrum_moments(freq, p)
    bins = min(math.ceil(WIDTH_AVERAGING / max(segments, 1)), round(0.5 * rms / step))
    if bins > 1:
        p = np.convolve(p, np.ones(bins) / bins, mode="same")
    k = int(np.argmax(p))
    rough = np.count_nonzero(p >= p[k] / 2.0) * step
    top = (np.abs(freq - freq[k]) <= 0.3 * rough) & (p > 0)
    peak = p[k]
    if np.count_nonzero(top) >= 5:
        c = np.polyfit(freq[top] - freq[k], np.log(p[top]), 2)
        if c[0] < 0:
            peak = math.exp(c[2] - c[1] ** 2 / (4.0 * c[0]))
    return float(np.count_nonzero(p >= peak / 2.0) * step)


def _independent(duration_s: float, spread_hz: float) -> float:
    """Roughly how many independent fades a record holds."""
    return max(duration_s * spread_hz, 1.0)


def fading_statistics(spread_hz: float, *, k_db: float | None = None,
                      shape: str = fading.GAUSSIAN, duration_s: float | None = None,
                      seed: int = 0) -> list[Check]:
    """One tap gain of ``spread_hz`` against Rice's statistics."""
    duration = duration_s or test_length_s(spread_hz)
    rng = np.random.default_rng(seed)
    gain = fading.draw(duration, spread_hz, k_db=k_db, rng=rng, shape=shape)
    # Read at four times the gain's own grid: the output is its linear
    # interpolation, and a fade 20 dB down lasts only a few grid steps.
    rate = 4.0 * gain.rate_hz
    t = np.arange(4 * gain.samples.size) / rate
    g = gain(t)
    power = np.abs(g) ** 2
    env = np.sqrt(power / power.mean())
    k = 0.0 if k_db is None else 10.0 ** (k_db / 10.0)
    n_ind = _independent(duration, spread_hz)
    label = f"fading {spread_hz:g} Hz {shape}" + (f" K {k_db:g} dB" if k_db is not None
                                                  else "")
    out = [Check(label, "mean power", float(power.mean()), 1.0,
                 3.0 / math.sqrt(n_ind), basis="E|g|^2 = 1 by construction")]

    # Envelope distribution: Kolmogorov-Smirnov distance to Rice / Rayleigh.
    grid = np.linspace(0.0, 3.0, 301)
    emp = np.searchsorted(np.sort(env), grid, side="right") / env.size
    ks = float(np.max(np.abs(emp - rice_cdf(grid, k))))
    out.append(Check(label, "envelope CDF, KS distance", ks, 0.0,
                     1.63 / math.sqrt(n_ind), relative=False,
                     basis="Rayleigh" if k == 0 else "Rice"))

    # Autocorrelation of the scattered part, where theory gives 0.8, 0.5, 0.2.
    scattered = g - g.mean() if k > 0 else g
    r0 = np.vdot(scattered, scattered).real / scattered.size
    for target in (0.8, 0.5, 0.2):
        if shape == fading.GAUSSIAN:
            lag = math.sqrt(-math.log(target) / 2.0) / (math.pi * spread_hz / 2.0)
        else:
            lag = -math.log(target) / (2.0 * math.pi * spread_hz / 2.0)
        n = max(1, round(lag * rate))
        r = abs(np.vdot(scattered[:-n], scattered[n:])) / (scattered.size - n) / r0
        out.append(Check(label, f"|R| at {n / rate:.3g} s", float(r),
                         float(fading.autocorrelation(n / rate, spread_hz, shape)),
                         3.0 / math.sqrt(n_ind), relative=False,
                         basis="Fourier transform of the Doppler spectrum"))

    # The spectrum: RMS width for a Gaussian (sigma), half-power width for both.
    segment = min(duration / 8.0, 40.0 / spread_hz)
    freq, psd, rms = fading.doppler_spectrum(scattered, rate, segment_s=segment)
    # A periodogram averaged over n segments scatters by 1/sqrt(n) a bin, and
    # a width read off it with it: the tolerance follows the record's length.
    n_segments = max(1, int(2.0 * duration / segment) - 1)
    width_tol = max(0.10, 1.5 / math.sqrt(n_segments))
    if shape == fading.GAUSSIAN:
        out.append(Check(label, "Doppler RMS width", rms, spread_hz / 2.0, 0.05,
                         unit="Hz", basis="F.1487: spread = 2 sigma"))
        fwhm_expected = spread_hz * math.sqrt(2.0 * math.log(2.0))
    else:
        fwhm_expected = spread_hz
    if k == 0:
        out.append(Check(label, "Doppler half-power width",
                         half_power_width(freq, psd, n_segments), fwhm_expected,
                         width_tol,
                         unit="Hz",
                         basis="ITS 98-348: 2 sigma_D read at -3 dB"))

    # Level crossings and fades: Rice's formulas need a finite second moment.
    if shape == fading.GAUSSIAN:
        for level_db in (0.0, -10.0):
            rho = 10.0 ** (level_db / 20.0)
            lcr, afd = measured_crossings(env, rate, rho)
            n_theory = float(crossing_rate(rho, spread_hz / 2.0, k))
            afd_theory = float(rice_cdf(rho, k)) / n_theory
            tol = max(0.05, 3.0 / math.sqrt(max(lcr * duration, 1.0)))
            out.append(Check(label, f"crossings at {level_db:g} dB", lcr, n_theory, tol,
                             unit="1/s", basis="Rice: 2 sqrt(pi(K+1)) beta rho ..."))
            out.append(Check(label, f"fade duration at {level_db:g} dB", afd,
                             afd_theory, tol, unit="s",
                             basis="P(r < rho) / crossing rate"))
    if k > 0:
        k_est = rice_k_moments(power)
        out.append(Check(label, "Rice K by moments", 10 * math.log10(k_est), k_db, 1.0,
                         relative=False, unit="dB", basis="Greenstein et al. 1999"))
    return out


# --- the scattering function --------------------------------------------------

@dataclass
class ScatteringFunction:
    """S(tau, nu): power per delay bin and Doppler bin, linear."""

    delay_us: np.ndarray          # relative to the realisation's reference delay
    doppler_hz: np.ndarray
    power: np.ndarray             # [delay, doppler]
    bandwidth_hz: float
    segments: int

    def delay_profile(self) -> np.ndarray:
        return self.power.sum(axis=1)

    def doppler_at(self, lo_us: float, hi_us: float) -> np.ndarray:
        sel = (self.delay_us >= lo_us) & (self.delay_us <= hi_us)
        return self.power[sel].sum(axis=0)


def scattering_function(real: channel.Realisation, *, bandwidth_hz: float,
                        n_freq: int = 256, rate_hz: float | None = None,
                        segment_s: float | None = None) -> ScatteringFunction:
    """Estimate S(tau, nu) of one realisation from its H(t, f).

    H is read on ``n_freq`` frequencies across ``bandwidth_hz`` (Hann
    weighted, so the delay sidelobes stay 30 dB down) and at ``rate_hz``;
    each segment's h(t, tau) is Fourier transformed along t, and the
    segments' periodograms are averaged -- the spectral averaging of the ITS
    verification.
    """
    spreads = [p.spread_hz for p in real.paths]
    shifts = [abs(p.shift_hz) for p in real.paths]
    widest = max([*spreads, 0.05])
    rate = rate_hz or max(4.0, 8.0 * widest + 4.0 * max(shifts))
    segment = segment_s or 32.0 / widest
    nper = int(2 ** math.ceil(math.log2(segment * rate)))
    df = (np.arange(n_freq) - n_freq // 2) * (bandwidth_hz / n_freq)
    taper = np.hanning(n_freq)
    t_win = np.hanning(nper)
    n_seg = int((real.duration_s * rate) // nper)
    if n_seg < 1:
        raise ValueError("the realisation is shorter than one segment")
    acc = np.zeros((n_freq, nper))
    for s in range(n_seg):
        t = real.start_s + (s * nper + np.arange(nper)) / rate
        h_tf = real.transfer(df, t) * taper[None, :]
        h_ttau = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(h_tf, axes=1), axis=1),
                                 axes=1)
        spec = np.fft.fft(h_ttau * t_win[:, None], axis=0)
        acc += (np.abs(spec) ** 2).T
    acc /= n_seg
    delay = (np.arange(n_freq) - n_freq // 2) / bandwidth_hz * 1e6
    doppler = np.fft.fftfreq(nper, d=1.0 / rate)
    order = np.argsort(doppler)
    return ScatteringFunction(delay_us=delay, doppler_hz=doppler[order],
                              power=acc[:, order], bandwidth_hz=bandwidth_hz,
                              segments=n_seg)


def _spectrum_moments(freq: np.ndarray, power: np.ndarray) -> tuple[float, float]:
    total = power.sum()
    centre = float(np.sum(freq * power) / total)
    width = math.sqrt(float(np.sum((freq - centre) ** 2 * power) / total))
    return centre, width


def scattering_checks(state: ChannelState, fc_mhz: float, *,
                      bandwidth_hz: float = 48e3, duration_s: float | None = None,
                      effects: channel.Effects = channel.Effects(),
                      seed: int = 0, n_freq: int = 256) -> list[Check]:
    """Read each path's delay, power share, shift and spread off S(tau, nu)."""
    paths = channel.components(state, fc_mhz, effects)
    widest = max([p.spread_hz for p in paths] + [0.05])
    duration = duration_s or test_length_s(widest)
    real = channel.Realisation(state, fc_mhz, bandwidth_hz, duration, effects=effects,
                               seed=seed)
    sf = scattering_function(real, bandwidth_hz=bandwidth_hz, n_freq=n_freq)
    resolution_us = 1e6 / bandwidth_hz
    total = sf.power.sum()
    powers = [10.0 ** (p.cn0_dbhz / 10.0) for p in real.paths]
    out = []
    label = f"S(tau, nu) of {state.source or 'state'} at {fc_mhz:g} MHz"
    for p, p_lin in zip(real.paths, powers):
        rel = p.delay_us - real.reference_delay_us
        lo, hi = rel - 3 * resolution_us, rel + 3 * resolution_us
        sel = (sf.delay_us >= lo) & (sf.delay_us <= hi)
        profile = sf.power[sel].sum(axis=1)
        delay = float(np.sum(sf.delay_us[sel] * profile) / profile.sum())
        share = float(profile.sum() / total)
        n_ind = _independent(duration, max(p.spread_hz, 0.05))
        out.append(Check(label, f"{p.label} delay", delay, rel, resolution_us / 4,
                         relative=False, unit="us", basis="ITS 98-348 sec. 4"))
        out.append(Check(label, f"{p.label} power share", share,
                         p_lin / sum(powers), 3.0 / math.sqrt(n_ind),
                         basis="mean powers as set"))
        spec = sf.doppler_at(lo, hi)
        centre, width = _spectrum_moments(sf.doppler_hz, spec)
        if p.spread_hz > 0:
            out.append(Check(label, f"{p.label} Doppler shift", centre, p.shift_hz,
                             max(0.1 * p.spread_hz, 0.01), relative=False, unit="Hz",
                             basis="centroid of the Doppler profile"))
            if p.doppler_shape == fading.GAUSSIAN:
                out.append(Check(label, f"{p.label} spread (2 sigma)", 2.0 * width,
                                 p.spread_hz, max(0.10, 1.0 / math.sqrt(sf.segments)),
                                 unit="Hz",
                                 basis="F.1487: frequency spread = 2 sigma"))
            else:
                width = half_power_width(sf.doppler_hz, spec, sf.segments)
                out.append(Check(label, f"{p.label} spread (-3 dB)", width,
                                 p.spread_hz, max(0.20, 1.5 / math.sqrt(sf.segments)),
                                 unit="Hz",
                                 basis="ITS 98-348: 2 sigma_D at -3 dB"))
    return out


#: Zero-padding of the band before the delay transform, so the extent is
#: read on a grid finer than 1/B.
EXTENT_PAD = 8


def dispersion_extent(mode: Mode, fc_mhz: float, *, bandwidth_hz: float = 1e6,
                      n_freq: int = 4096) -> Check:
    """A dispersive mode's delay smear across a band against |GDD| * B.

    One static (non-fading) realisation with ``dispersion='gdd'``: the
    impulse response of a band B through a linear group delay is a chirp
    spread over |GDD| * B, flat but for Fresnel ripple, and at half its
    amplitude where the band's edges map. So the extent is read where the
    power falls to a quarter of the plateau's median, not of the peak, which
    the ripple overshoots. This is the impulse-response stretching that
    limits wideband HF (Ivanov et al., Radiotekhnika 88(8), 2024).
    """
    state = ChannelState(Circuit("verify", "verify"), 0.0, modes=[mode],
                         source="verify")
    effects = channel.Effects(fading=False, dispersion="gdd")
    real = channel.Realisation(state, fc_mhz, bandwidth_hz, 0.1, effects=effects,
                               seed=0)
    path = real.paths[0]
    df = (np.arange(n_freq) - n_freq // 2) * (bandwidth_hz / n_freq)
    padded = np.zeros(n_freq * EXTENT_PAD, dtype=np.complex128)
    centre = padded.size // 2
    padded[centre - n_freq // 2:centre + n_freq - n_freq // 2] = real.response(path, df)
    h_t = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(padded)))
    p = np.abs(h_t) ** 2
    plateau = float(np.median(p[p > 0.05 * p.max()]))
    above = np.flatnonzero(p >= plateau / 4.0)
    extent = (above.max() - above.min()) / (bandwidth_hz * EXTENT_PAD) * 1e6
    expected = abs(path.gdd_us_per_mhz) * bandwidth_hz / 1e6
    return Check(f"dispersion of {mode.name} at {fc_mhz:g} MHz over "
                 f"{bandwidth_hz / 1e3:g} kHz", "delay extent", float(extent),
                 expected, 0.10, unit="us", basis="|GDD| * B")


# --- a reference modem --------------------------------------------------------

def dpsk_ber_theory(ebn0_db, rho: float) -> np.ndarray:
    """DBPSK in flat Rayleigh fading, the gain correlating ``rho`` bit to bit.

    P_b = (1 + g (1 - rho)) / (2 (1 + g)), g the mean Eb/N0: the slow-fading
    1 / (2 (1 + g)) with the error floor (1 - rho) / 2 the Doppler leaves.
    """
    g = 10.0 ** (np.asarray(ebn0_db, dtype=np.float64) / 10.0)
    return (1.0 + g * (1.0 - rho)) / (2.0 * (1.0 + g))


def dpsk_checks(spread_hz: float, ebn0_db=(0.0, 10.0, 20.0), *,
                bit_rate: float = 2400.0, shape: str = fading.GAUSSIAN,
                duration_s: float | None = None, seed: int = 0) -> list[Check]:
    """BER of DBPSK through one flat fading path against :func:`dpsk_ber_theory`."""
    rho = float(fading.autocorrelation(1.0 / bit_rate, spread_hz, shape))
    theory = dpsk_ber_theory(ebn0_db, rho)
    duration = duration_s or test_length_s(spread_hz, float(theory.min()), bit_rate)
    n = round(duration * bit_rate)
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, n)
    symbols = np.cumprod(np.where(bits == 1, -1.0, 1.0)).astype(np.complex128)
    mode = Mode("flat", [2.0, 30.0], [1000.0, 1000.0], 50.0,
                doppler_spread_hz=spread_hz, doppler_shape=shape)
    state = ChannelState(Circuit("verify", "verify"), 0.0, modes=[mode],
                         source="verify")
    effects = channel.Effects(dispersion="none")
    out = []
    label = f"DBPSK {bit_rate:g} bit/s, {spread_hz:g} Hz {shape}, {duration:.0f} s"
    for e, p_theory in zip(np.atleast_1d(ebn0_db), theory):
        y = channel.simulate(state, symbols, bit_rate, 10.0, effects=effects, seed=seed,
                             snr_db=float(e))
        r = y.samples[y.lead_samples:y.lead_samples + n]
        decided = (np.real(r[1:] * np.conj(r[:-1])) < 0).astype(int)
        ber = float(np.mean(decided != bits[1:]))
        # The estimate's spread is set by the fades, not the bit count: P_b of
        # an exponential Eb/N0 has relative variance g^2 / (1 + 2g) per fade.
        g = 10.0 ** (float(e) / 10.0)
        per_fade = g ** 2 / (1.0 + 2.0 * g)
        tol = max(0.05, 3.0 * math.sqrt(per_fade / _independent(duration, spread_hz)))
        out.append(Check(label, f"BER at Eb/N0 {e:g} dB", ber, float(p_theory), tol,
                         basis="(1 + g(1 - rho)) / (2(1 + g))"))
    return out


# --- everything ----------------------------------------------------------------

# --- delay spread within a mode -------------------------------------------------

def delay_spread_checks(sigma_us: float, *, spread_hz: float = 1.0,
                        fc_mhz: float = 10.0, bandwidth_hz: float = 48e3,
                        duration_s: float = 3000.0, seed: int = 0) -> list[Check]:
    """A Gaussian power-delay profile against its frequency correlation.

    For uncorrelated scattering the transfer function's correlation across
    frequency is the Fourier transform of the power-delay profile (Bello
    1963): |R(df)| = exp(-2 pi^2 sigma^2 df^2) for a Gaussian of RMS sigma.
    Read at the offsets where theory gives 0.8, 0.5 and 0.2, over times one
    correlation time apart and frequencies one coherence bandwidth apart.
    """
    from .state import Circuit

    mode = Mode("spread", [fc_mhz - 1.0, fc_mhz + 1.0], [3000.0, 3000.0], 40.0,
                doppler_spread_hz=spread_hz, delay_spread_us=sigma_us)
    state = ChannelState(Circuit("verify-tx", "verify-rx"), 0.0, modes=[mode],
                         source=f"delay spread {sigma_us:g} us")
    effects = channel.Effects(dispersion="none")
    real = channel.Realisation(state, fc_mhz, bandwidth_hz, duration_s,
                               effects=effects, seed=seed)
    sigma = sigma_us * 1e-6
    coherence = math.sqrt(math.log(2.0) / (2.0 * math.pi ** 2)) / sigma   # |R| = 0.5
    t = np.arange(0.0, duration_s, 1.0 / spread_hz)
    base = np.arange(-bandwidth_hz / 4, bandwidth_hz / 4 - 2 * coherence, coherence)
    label = f"delay spread {sigma_us:g} us at {fc_mhz:g} MHz"
    out = [Check(label, "RMS of the paths' profile",
                 math.sqrt(sum(p_.level_lin * (p_.tap_us ** 2) for p_ in
                               _taps(real))), sigma_us, 0.01, unit="us",
                 basis="the Gaussian, sampled and cut at 4 sigma")]
    n_ind = t.size * base.size
    for rho in (0.8, 0.5, 0.2):
        df = math.sqrt(-math.log(rho) / (2.0 * math.pi ** 2)) / sigma
        h = real.transfer(np.concatenate([base, base + df]), t)
        a, b = h[:, :base.size], h[:, base.size:]
        r = abs(np.mean(a * np.conj(b))) / math.sqrt(np.mean(np.abs(a) ** 2)
                                                     * np.mean(np.abs(b) ** 2))
        out.append(Check(label, f"|R| at {df / 1e3:.2f} kHz", float(r), rho,
                         3.0 / math.sqrt(n_ind), relative=False,
                         basis="Bello 1963: R(df) is the profile's transform"))
    return out


def _taps(real: channel.Realisation):
    """Each path's power share and offset, for the profile's own RMS."""
    from types import SimpleNamespace

    powers = np.array([10.0 ** (p.level_db / 10.0) for p in real.paths])
    return [SimpleNamespace(level_lin=w / powers.sum(), tap_us=p.tap_us)
            for p, w in zip(real.paths, powers)]


def run(*, quick: bool = False, seed: int = 0) -> list[Check]:
    """The full level-0 battery; ``quick`` shortens every run tenfold."""
    from . import presets

    scale = 0.1 if quick else 1.0

    def length(spread, **kw):
        return test_length_s(spread, **kw) * scale

    out = []
    cases = ((0.5, None, fading.GAUSSIAN), (1.0, None, fading.GAUSSIAN),
             (1.0, 6.0, fading.GAUSSIAN), (1.0, None, fading.LORENTZIAN))
    for i, (spread, k_db, shape) in enumerate(cases):
        out += fading_statistics(spread, k_db=k_db, shape=shape,
                                 duration_s=length(spread), seed=seed + i)
    for i, name in enumerate(("mid-moderate", "mid-disturbed")):
        state = presets.f1487(name)
        spread = presets.F1487[name][1]
        out += scattering_checks(state, 10.0, duration_s=length(spread),
                                 seed=seed + 10 + i)
    lorentz = presets.f1487("mid-disturbed")
    for m in lorentz.modes:
        m.doppler_shape = fading.LORENTZIAN
    lorentz.source += ", Lorentzian"
    out += scattering_checks(lorentz, 10.0, duration_s=length(1.0), seed=seed + 12)
    law = presets.law_mode(muf_mhz=25.0)
    out += [dispersion_extent(law, f) for f in (21.0, 23.0, 24.0)]
    out += dpsk_checks(1.0, duration_s=length(1.0, ber=0.005, bit_rate=2400.0),
                       seed=seed + 20)
    for i, sigma in enumerate((20.0, 50.0)):
        out += delay_spread_checks(sigma, duration_s=length(1.0), seed=seed + 30 + i)
    return out


def format_checks(checks: list[Check]) -> str:
    lines, group = [], None
    for c in checks:
        if c.group != group:
            group = c.group
            lines.append(group)
        err = f"{100 * c.error:+.1f} %" if c.relative else f"{c.error:+.3g} {c.unit}"
        tol = (f"{100 * c.tolerance:.0f} %" if c.relative
               else f"{c.tolerance:.3g} {c.unit}")
        lines.append(f"  {'ok  ' if c.passed else 'FAIL'} {c.name:<30} "
                     f"{c.measured:10.4g} vs {c.expected:10.4g} {c.unit:<4} "
                     f"error {err:>10} (tol {tol})  [{c.basis}]")
    n_fail = sum(not c.passed for c in checks)
    lines.append(f"{len(checks) - n_fail}/{len(checks)} passed")
    return "\n".join(lines)
