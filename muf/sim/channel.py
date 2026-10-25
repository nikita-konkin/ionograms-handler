"""The channel for an arbitrary signal: a state, at one carrier.

A complex baseband signal sampled at ``fs`` and centred on ``fc`` goes
through every mode present at ``fc``. Each mode is a dispersive filter
followed by a fading gain::

    y(t) = sum_m  g_m(t) * (h_m conv x)(t)

``h_m`` carries the mode's delay and dispersion, from its group-delay curve
around ``fc``; ``g_m`` its Doppler shift and Watterson fading
(:mod:`muf.sim.fading`). A mode with a delay spread is several such paths
a few microseconds apart, each fading on its own (:meth:`Mode.delay_taps`),
which across the band is frequency-selective fading. Filtering first and
fading second is exact when the fading is slow against the filter's
length -- here a channel that decorrelates in a second against a few
hundred microseconds of dispersion.

How much of the curve the filter uses is the ``dispersion`` effect:

``none``     the delay at ``fc`` only;
``gdd``      the delay and its slope, tau = tau0 + GDD*df;
``gdd+tod``  adding the curvature, + TOD*df^2/2 -- the Taylor series of
             the dissertation's chapter 2, truncated where it truncates it;
``full``     the curve itself, and the mode's level across the band with it:
             above its MUF a mode is simply not there.

The phase follows from the group delay as phi(df) = -2 pi * integral of
tau, so the group delay of the filter is exactly the curve.

Levels. A state's C/N0 is a mode's strength against the receiver's noise
density. By default the output is normalised so the modes' mean powers add
to the input's -- relative levels kept, absolute ones dropped -- and
``snr_db`` sets the noise. ``absolute=True`` keeps the state's C/N0 with a
unit-power input and noise of unit density, so the in-band SNR is C/N0
minus 10 log10(bandwidth).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import fading
from .state import ChannelState, Mode

DISPERSION = ("none", "gdd", "gdd+tod", "full")

#: Width of the cosine roll-off where a mode's band ends inside the signal's,
#: as a fraction of ``fs``. A hard edge rings for the length of the record;
#: the real one is not hard either. Centred on the edge, as in a sounding.
EDGE_TAPER_FRACTION = 0.02

#: Samples of guard either side of everything the filters can move energy to.
GUARD_SAMPLES = 32


@dataclass(frozen=True)
class Effects:
    """Which parts of the channel are switched on."""

    multipath: bool = True      # False keeps the strongest mode at fc only
    dispersion: str = "full"
    fading: bool = True
    doppler: bool = True        # the shifts; spreads are `fading`
    magnetoionic: bool = True   # o/x split, where a mode has one
    delay_spread: bool = True   # the paths within a mode, where it has a spread

    def __post_init__(self) -> None:
        if self.dispersion not in DISPERSION:
            raise ValueError(f"dispersion {self.dispersion!r} is not one of "
                             f"{DISPERSION}")


@dataclass
class Component:
    """One path through the channel: a mode, or one wave of a split mode."""

    mode: Mode
    wave: str                   # "o", "x", or "" for an unsplit mode
    delay_us: float             # at fc
    gdd_us_per_mhz: float
    tod_us_per_mhz2: float
    cn0_dbhz: float             # this component's share of the mode's power
    shift_hz: float
    spread_hz: float
    k_db: float | None
    extra_delay_us: float = 0.0  # the x wave's lag behind o, and the tap's offset
    level_db: float = 0.0        # this path's share of the mode, dB
    doppler_shape: str = "gaussian"
    tap_us: float = 0.0          # offset within the mode's delay spread

    @property
    def label(self) -> str:
        tap = f"{self.tap_us:+.0f}us" if self.tap_us else ""
        return f"{self.mode.name}{'/' + self.wave if self.wave else ''}{tap}"

    def describe(self) -> dict:
        return {"path": self.label, "delay_us": self.delay_us,
                "gdd_us_per_mhz": self.gdd_us_per_mhz,
                "tod_us_per_mhz2": self.tod_us_per_mhz2,
                "cn0_dbhz": self.cn0_dbhz, "doppler_shift_hz": self.shift_hz,
                "doppler_spread_hz": self.spread_hz,
                "doppler_shape": self.doppler_shape}


def components(state: ChannelState, fc_mhz: float, effects: Effects = Effects(), *,
               tap_step_us: float = 10.0) -> list[Component]:
    """The paths a signal at ``fc_mhz`` takes, earliest first.

    A delay spread's paths are ``tap_step_us`` apart, or closer for a narrow
    spread; :class:`Realisation` asks for a quarter of a sample.
    """
    out: list[Component] = []
    present = state.at(fc_mhz)
    if not effects.multipath and present:
        present = [max(present, key=lambda item: item.cn0_dbhz)]
    for item in present:
        m = item.mode
        shift = m.doppler_shift_hz if effects.doppler else 0.0
        spread = m.doppler_spread_hz if effects.fading else 0.0
        k_db = m.rician_k_db if effects.fading else None
        if effects.magnetoionic and m.ox_split_us is not None:
            r = 10.0 ** (m.x_to_o_db / 10.0)
            o_db = 10.0 * math.log10(1.0 / (1.0 + r))
            x_db = 10.0 * math.log10(r / (1.0 + r))
            x_shift = shift + (m.x_shift_hz if effects.doppler else 0.0)
            waves = [("o", 0.0, o_db, shift),
                     ("x", float(m.ox_split_us), x_db, x_shift)]
        else:
            waves = [("", 0.0, 0.0, shift)]
        taps = (m.delay_taps(tap_step_us) if effects.delay_spread
                else [(0.0, 1.0, m.rician_k_db)])
        for wave, extra, level, wshift in waves:
            for offset, share, tap_k in taps:
                tap_db = 10.0 * math.log10(share)
                out.append(Component(
                    mode=m, wave=wave, delay_us=item.delay_us + extra + offset,
                    gdd_us_per_mhz=item.gdd_us_per_mhz,
                    tod_us_per_mhz2=item.tod_us_per_mhz2,
                    cn0_dbhz=item.cn0_dbhz + level + tap_db, shift_hz=wshift,
                    spread_hz=spread, k_db=tap_k if k_db is not None else None,
                    extra_delay_us=extra + offset, level_db=level + tap_db,
                    doppler_shape=m.doppler_shape, tap_us=offset))
    return sorted(out, key=lambda c: c.delay_us)


@dataclass
class Output:
    """What :meth:`Realisation.apply` returns."""

    samples: np.ndarray
    fs_hz: float
    lead_samples: int           # input sample 0 at the reference delay lands here
    reference_delay_us: float   # the earliest path's delay at fc
    noise_variance: float       # per sample; 0 without noise
    signal_scale: float         # amplitude factor applied to every path
    paths: list[dict] = field(default_factory=list)


class Realisation:
    """One draw of the channel: fixed fading, readable as a signal or as H(f, t).

    ``duration_s`` bounds the times the fading gains are drawn for; ``apply``
    needs it to cover the output, input plus the channel's spread.
    """

    def __init__(self, state: ChannelState, fc_mhz: float, fs_hz: float,
                 duration_s: float, *, effects: Effects = Effects(),
                 seed: int | None = None, start_s: float = 0.0) -> None:
        self.state = state
        self.fc_mhz = float(fc_mhz)
        self.fs_hz = float(fs_hz)
        self.effects = effects
        self.start_s = float(start_s)
        self.paths = components(state, fc_mhz, effects,
                                tap_step_us=1e6 / (4.0 * self.fs_hz))
        if not self.paths:
            raise ValueError(f"no mode of {state.source or 'the state'} exists at "
                             f"{fc_mhz:.3f} MHz (MUF {state.muf_mhz:.2f} MHz)")
        self.reference_delay_us = min(p.delay_us for p in self.paths)
        rng = np.random.default_rng(seed)
        self.duration_s = float(duration_s)
        self.gains = [fading.draw(self.duration_s, p.spread_hz, shift_hz=p.shift_hz,
                                  k_db=p.k_db, rng=rng, start_s=self.start_s,
                                  shape=p.doppler_shape)
                      for p in self.paths]

    # -- the filters
    def _relative_delay(self, path: Component, df_hz: np.ndarray) -> np.ndarray:
        """tau(fc + df) - tau_ref, us, by the configured dispersion."""
        df_mhz = np.asarray(df_hz, dtype=np.float64) / 1e6
        mode = self.effects.dispersion
        if mode == "none":
            tau = np.full(df_mhz.shape, path.delay_us)
        elif mode == "gdd":
            tau = path.delay_us + path.gdd_us_per_mhz * df_mhz
        elif mode == "gdd+tod":
            tau = (path.delay_us + path.gdd_us_per_mhz * df_mhz
                   + 0.5 * path.tod_us_per_mhz2 * df_mhz ** 2)
        else:
            f = self.fc_mhz + df_mhz
            m = path.mode
            # Outside the band the level is zero; the delay is held at the
            # edge so the phase stays finite where nothing is transmitted.
            tau = m.delay_at(np.clip(f, m.f_lo, m.f_hi)) + path.extra_delay_us
        return tau - self.reference_delay_us

    def _level(self, path: Component, df_hz: np.ndarray) -> np.ndarray:
        """Amplitude across the band, linear (sqrt of C/N0)."""
        if self.effects.dispersion != "full":
            return np.full(np.shape(df_hz), 10.0 ** (path.cn0_dbhz / 20.0))
        m = path.mode
        f = self.fc_mhz + np.asarray(df_hz, dtype=np.float64) / 1e6
        cn0 = np.interp(f, m.freq_mhz, m.cn0_dbhz) + path.level_db
        amp = 10.0 ** (cn0 / 20.0)
        taper_mhz = EDGE_TAPER_FRACTION * self.fs_hz / 1e6
        inside = np.minimum(f - m.f_lo, m.f_hi - f)
        roll = np.clip((inside + taper_mhz / 2.0) / taper_mhz, 0.0, 1.0)
        return amp * (0.5 - 0.5 * np.cos(np.pi * roll))

    def response(self, path: Component, df_hz) -> np.ndarray:
        """Static transfer function of one path at offsets ``df_hz`` from fc.

        Its group delay, -(1/2pi) dphi/df, is :meth:`_relative_delay` exactly:
        the phase is minus 2 pi times the integral of the delay from fc.
        """
        df = np.asarray(df_hz, dtype=np.float64)
        flat = df.ravel()
        order = np.argsort(flat)
        grid = flat[order]
        if self.effects.dispersion == "full":
            # Integrate on a grid fine enough for the curve, then read off.
            lo, hi = min(grid[0], 0.0), max(grid[-1], 0.0)
            n = max(2049, math.ceil((hi - lo) / 20.0) + 1)
            fine = np.linspace(lo, hi, n)
            tau = self._relative_delay(path, fine) * 1e-6
            steps = 0.5 * (tau[1:] + tau[:-1]) * np.diff(fine)
            cyc = np.concatenate([[0.0], np.cumsum(steps)])
            cyc -= np.interp(0.0, fine, cyc)
            cycles = np.interp(grid, fine, cyc)
        else:
            d0 = (path.delay_us - self.reference_delay_us) * 1e-6
            mode = self.effects.dispersion
            g = path.gdd_us_per_mhz * 1e-12 if mode != "none" else 0.0
            t3 = path.tod_us_per_mhz2 * 1e-18 if mode == "gdd+tod" else 0.0
            cycles = d0 * grid + 0.5 * g * grid ** 2 + t3 * grid ** 3 / 6.0
        h = np.empty(flat.shape, dtype=np.complex128)
        h[order] = self._level(path, grid) * np.exp(-2j * np.pi * cycles)
        return h.reshape(df.shape)

    def gain(self, index: int, t_s) -> np.ndarray:
        return self.gains[index](t_s)

    def transfer(self, df_hz, t_s) -> np.ndarray:
        """H(t, f): rows are times ``t_s``, columns offsets ``df_hz`` from fc.

        Absolute levels -- sqrt(C/N0) -- and the same draw ``apply`` uses.
        """
        df = np.atleast_1d(np.asarray(df_hz, dtype=np.float64))
        t = np.atleast_1d(np.asarray(t_s, dtype=np.float64))
        out = np.zeros((t.size, df.size), dtype=np.complex128)
        for i, path in enumerate(self.paths):
            out += self.gain(i, t)[:, None] * self.response(path, df)[None, :]
        return out

    # -- a signal through it
    def _spread(self) -> tuple[int, int]:
        """Samples the channel can move energy before and after the reference."""
        df = np.linspace(-self.fs_hz / 2, self.fs_hz / 2, 1025)
        lo = hi = 0.0
        for path in self.paths:
            rel = self._relative_delay(path, df)
            lo, hi = min(lo, float(np.min(rel))), max(hi, float(np.max(rel)))
        lead = math.ceil(-lo * 1e-6 * self.fs_hz) + GUARD_SAMPLES
        tail = math.ceil(hi * 1e-6 * self.fs_hz) + GUARD_SAMPLES
        return lead, tail

    def apply(self, x, *, snr_db: float | None = None,
              snr_bandwidth_hz: float | None = None, absolute: bool = False,
              noise_seed: int | None = None) -> Output:
        """``x`` (complex baseband at ``fs``) through this realisation.

        The output is longer than the input by the channel's spread, and
        input sample 0 at the reference delay arrives at ``lead_samples``.
        ``snr_db`` is the mean signal power over the noise in
        ``snr_bandwidth_hz`` (default: the whole simulated band ``fs``).
        """
        from scipy.fft import fft, ifft, next_fast_len

        x = np.asarray(x, dtype=np.complex128).ravel()
        lead, tail = self._spread()
        n_out = x.size + lead + tail
        if n_out / self.fs_hz > self.duration_s + 1e-9:
            raise ValueError(f"the output lasts {n_out / self.fs_hz:.3f} s and the "
                             f"fading was drawn for {self.duration_s:.3f} s")
        n_fft = next_fast_len(n_out)
        df = np.fft.fftfreq(n_fft, d=1.0 / self.fs_hz)
        spectrum = fft(x, n_fft) * np.exp(-2j * np.pi * df * lead / self.fs_hz)
        t = self.start_s + np.arange(n_out) / self.fs_hz

        powers = [10.0 ** (p.cn0_dbhz / 10.0) for p in self.paths]
        scale = 1.0 if absolute else 1.0 / math.sqrt(sum(powers))
        y = np.zeros(n_out, dtype=np.complex128)
        for i, path in enumerate(self.paths):
            through = ifft(spectrum * self.response(path, df))[:n_out]
            y += self.gain(i, t) * through
        y *= scale

        noise_var = 0.0
        if snr_db is not None or absolute:
            if snr_db is not None:
                p_x = float(np.mean(np.abs(x) ** 2)) if x.size else 0.0
                p_sig = p_x * (sum(powers) if absolute else 1.0)
                band = snr_bandwidth_hz or self.fs_hz
                noise_var = p_sig / 10.0 ** (snr_db / 10.0) * (self.fs_hz / band)
            else:
                noise_var = self.fs_hz          # unit noise density
            rng = np.random.default_rng(noise_seed)
            y += math.sqrt(noise_var / 2.0) * (rng.standard_normal(n_out)
                                               + 1j * rng.standard_normal(n_out))
        return Output(samples=y, fs_hz=self.fs_hz, lead_samples=lead,
                      reference_delay_us=self.reference_delay_us,
                      noise_variance=noise_var, signal_scale=scale,
                      paths=[p.describe() for p in self.paths])


def simulate(state: ChannelState, x, fs_hz: float, fc_mhz: float, *,
             effects: Effects = Effects(), seed: int | None = None,
             snr_db: float | None = None, snr_bandwidth_hz: float | None = None,
             absolute: bool = False) -> Output:
    """One realisation, drawn for exactly as long as the output needs."""
    probe = Realisation(state, fc_mhz, fs_hz, 0.0, effects=effects, seed=seed)
    lead, tail = probe._spread()
    duration = (len(x) + lead + tail) / fs_hz
    real = Realisation(state, fc_mhz, fs_hz, duration, effects=effects, seed=seed)
    noise_seed = None if seed is None else seed + 1
    return real.apply(x, snr_db=snr_db, snr_bandwidth_hz=snr_bandwidth_hz,
                      absolute=absolute, noise_seed=noise_seed)
