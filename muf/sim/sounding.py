"""Synthetic chirp soundings: a channel state, sounded as the instrument sounds it.

A chirp sounder dechirps the received sweep against its own, so an echo of
group delay tau arrives as a tone at the beat frequency ``-rate * tau`` --
the convention the ``.lfs`` axis maps to range (``muf.calibrate``). As the
sweep climbs, the beat follows each mode's delay curve::

    x(t) = sum_m  a_m(f(t)) g_m(t) exp(j 2 pi integral(-rate tau_m(f(t)) dt))
           + noise,                f(t) = start + rate * t

``a_m`` comes from the mode's C/N0 against unit-variance noise, so the
synthetic file reads on muf's ionogram scale exactly as a real one would;
``g_m`` is its Watterson fading, which along the sweep is also fading across
frequency. A mode with a delay spread is many such paths about its curve,
each fading on its own (:meth:`Mode.delay_taps`): a beat a little off the
curve's, which the ionogram shows as a wider trace. The result is written
as an ``.lfs`` and goes through muf like any recording:
:func:`muf.loader.load`, the extractors, ``muf.trace``.

The phase is the beat integrated in time, which is the first-order
(stationary-phase) form of a dechirped echo. It is what the instrument's
ionogram resolves; the second-order term, the dispersion *within* one
FFT window, is about GDD * (rate * T)^2 -- 0.3 us for 30 us/MHz over a
20 kHz window, under a hundredth of a range bin.
"""

from __future__ import annotations

import datetime as _dt
import math
from pathlib import Path

import numpy as np

from .. import io_lfs
from . import fading
from .state import ChannelState, Mode

#: Samples synthesised per pass. Bounds the memory to a few times this many
#: complex128s whatever the sweep's length.
CHUNK = 1 << 20

#: The delay curve is evaluated every this many samples and interpolated in
#: between: it changes by under a nanosecond across 64 samples of sweep.
CURVE_STEP = 64

#: Spacing of a delay spread's paths, us: a fifth of the 49 us range bin
#: of the default window, so the profile the ionogram sees is smooth.
DELAY_STEP_US = 10.0

#: Cosine roll-off at each end of a mode's band, MHz -- an echo fades in and
#: out over a few rows rather than switching. Centred on the band's edge, so
#: the mode is at half level exactly where its knots end: a roll-off inside
#: the band would eat the top rows of the trace and put every synthetic MUF
#: low, by 0.03 MHz at this width on cyprus1_20260204_120010.
EDGE_TAPER_MHZ = 0.05


def _require_sweep(state: ChannelState):
    if state.sweep is None:
        raise ValueError("a sounding needs the state's sweep: "
                         "start, rate, duration and sample rate")
    return state.sweep


def _paths(state: ChannelState,
           magnetoionic: bool) -> list[tuple[Mode, float, float, float]]:
    """(mode, extra delay us, level dB, extra shift Hz) per wave."""
    out = []
    for m in state.modes:
        if magnetoionic and m.ox_split_us is not None:
            r = 10.0 ** (m.x_to_o_db / 10.0)
            out.append((m, 0.0, 10 * math.log10(1 / (1 + r)), 0.0))
            out.append((m, float(m.ox_split_us), 10 * math.log10(r / (1 + r)),
                        m.x_shift_hz))
        else:
            out.append((m, 0.0, 0.0, 0.0))
    return out


def _composite(taps, gains, t_s: np.ndarray, rate: float) -> np.ndarray:
    """A spread wave's gain: its paths summed, each a beat ``rate * offset`` off."""
    out = np.zeros(t_s.shape, dtype=np.complex128)
    for (offset_us, share, _k), gain in zip(taps, gains):
        out += math.sqrt(share) * gain(t_s) * np.exp(
            -2j * np.pi * rate * offset_us * 1e-6 * t_s)
    return out


def synthesize(state: ChannelState, *, seed: int | None = None,
               fading_on: bool = True, doppler_on: bool = True,
               magnetoionic: bool = True, noise: bool = True,
               delay_spread: bool = True,
               duration_s: float | None = None) -> np.ndarray:
    """The dechirped record, complex64 at the sweep's sample rate."""
    sweep = _require_sweep(state)
    fs, rate = float(sweep.sample_rate_hz), float(sweep.rate_hz_s)
    duration = float(duration_s if duration_s is not None else sweep.duration_s)
    n = round(duration * fs)
    rng = np.random.default_rng(seed)

    paths = _paths(state, magnetoionic)
    taps, gains = [], []
    for m, _extra, _level, extra_shift in paths:
        shift = (m.doppler_shift_hz + extra_shift) if doppler_on else 0.0
        spread = m.doppler_spread_hz if fading_on else 0.0
        if m.delay_spread_us == 0.0 or not delay_spread:
            taps.append(None)
            gains.append(fading.draw(duration, spread, shift_hz=shift, rng=rng,
                                     k_db=m.rician_k_db if fading_on else None,
                                     shape=m.doppler_shape))
            continue
        # Spread: every path its own draw. Without fading they keep a random
        # phase each, which is a spread frozen in time.
        wave = m.delay_taps(DELAY_STEP_US)
        taps.append(wave)
        gains.append([fading.draw(duration, spread, shift_hz=shift, rng=rng,
                                  k_db=k_db if fading_on else None,
                                  shape=m.doppler_shape) for _d, _w, k_db in wave])
    phase = [0.0] * len(paths)
    out = np.empty(n, dtype=np.complex64)

    for start in range(0, n, CHUNK):
        stop = min(n, start + CHUNK)
        t = np.arange(start, stop, dtype=np.float64) / fs
        f = sweep.freq_at(t)
        coarse = np.arange(start, stop + CURVE_STEP, CURVE_STEP, dtype=np.float64)
        f_coarse = sweep.freq_at(coarse / fs)
        if noise:
            chunk = (rng.standard_normal(stop - start)
                     + 1j * rng.standard_normal(stop - start)) / math.sqrt(2.0)
        else:
            chunk = np.zeros(stop - start, dtype=np.complex128)
        half = EDGE_TAPER_MHZ / 2.0
        for i, (m, extra, level, _shift) in enumerate(paths):
            inside = (f >= m.f_lo - half) & (f <= m.f_hi + half)
            if not inside.any():
                # Nothing to add here. Where its phase starts when it does
                # appear is immaterial: the fading gain's phase is random.
                continue
            tau_coarse = m.delay_at(np.clip(f_coarse, m.f_lo, m.f_hi)) + extra
            tau = np.interp(t * fs, coarse, tau_coarse)
            beat = -rate * tau * 1e-6
            ph = phase[i] + 2.0 * np.pi * np.cumsum(beat) / fs
            phase[i] = float(ph[-1])
            cn0 = np.interp(f, m.freq_mhz, m.cn0_dbhz) + level
            # C/N0 against unit-variance noise: |a|^2 = (C/N0) / fs.
            amp = np.sqrt(10.0 ** (cn0 / 10.0) / fs)
            edge = np.clip((np.minimum(f - m.f_lo, m.f_hi - f) + half) / EDGE_TAPER_MHZ,
                           0.0, 1.0)
            amp = amp * (0.5 - 0.5 * np.cos(np.pi * edge)) * inside
            if taps[i] is None:
                g = gains[i](t)
            else:
                # Summed on the curve's coarse grid and interpolated: the
                # paths' beats are tens of hertz at most, the grid 625 Hz.
                g_coarse = _composite(taps[i], gains[i], coarse / fs, rate)
                g = (np.interp(t * fs, coarse, g_coarse.real)
                     + 1j * np.interp(t * fs, coarse, g_coarse.imag))
            chunk += amp * g * np.exp(1j * ph)
        out[start:stop] = chunk.astype(np.complex64)
    return out


def header_fields(state: ChannelState) -> dict:
    """The ``.lfs`` header a synthetic recording of this state carries."""
    sweep = _require_sweep(state)
    c = state.circuit
    when = _dt.datetime.fromtimestamp(int(state.epoch), tz=_dt.timezone.utc)
    dec = sweep.adc_rate_hz / sweep.sample_rate_hz
    if abs(dec - round(dec)) > 1e-9:
        raise ValueError(f"sample rate {sweep.sample_rate_hz} Hz is not the ADC rate "
                         f"{sweep.adc_rate_hz} Hz divided by an integer")
    return {
        "format": "LFSG", "format_ver": 1.0, "header_id": "fmt ", "header_size": 498,
        "tx_name": c.tx, "tx_latitude": c.tx_lat or 0.0,
        "tx_longitude": c.tx_lon or 0.0,
        "rx_name": c.rx, "rx_latitude": c.rx_lat or 0.0,
        "rx_longitude": c.rx_lon or 0.0,
        "start_year": when.year, "start_daynumber": when.timetuple().tm_yday,
        "start_month": when.month, "start_day": when.day, "start_hour": when.hour,
        "start_minute": when.minute, "start_second": when.second,
        "start_epoch": int(state.epoch), "chirpt": sweep.chirpt,
        "cf": round(sweep.start_mhz * 1e6 + sweep.adc_rate_hz / 2),
        "dur": round(sweep.duration_s), "rate": round(sweep.rate_hz_s),
        "rep": sweep.repetition_s, "rmin": sweep.rmin_km, "rmax": sweep.rmax_km,
        "dec": round(dec), "sample_rate": round(sweep.adc_rate_hz),
        "whiten": 0, "whiten_len": 0, "whiten_n": 0,
    }


def default_name(state: ChannelState, tag: str = "sim") -> str:
    """``<tag>_<tx>_<YYYYmmdd>_<HHMMSS>.lfs``, after the station's own naming."""
    when = _dt.datetime.fromtimestamp(int(state.epoch), tz=_dt.timezone.utc)
    return f"{tag}_{state.circuit.tx}_{when:%Y%m%d_%H%M%S}.lfs"


def write_lfs(state: ChannelState, path: str | Path, *, seed: int | None = None,
              **options) -> Path:
    """Synthesise and write one ``.lfs``. ``options`` go to :func:`synthesize`.

    ``path`` may be a directory, in which case :func:`default_name` is used.
    The duration is the header's whole seconds, as the recorder writes it.
    """
    path = Path(path)
    if path.is_dir():
        path = path / default_name(state)
    head = io_lfs.pack_header(header_fields(state))
    duration = options.pop("duration_s", round(state.sweep.duration_s))
    samples = synthesize(state, seed=seed, duration_s=duration, **options)
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "wb") as fh:
        fh.write(head)
        fh.write(samples.tobytes())
    tmp.replace(path)
    return path
