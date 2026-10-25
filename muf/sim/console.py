"""What the simulator page shows of one sounding: its data, as JSON.

For a sounding it reads the state, runs the closed loop with the next
sounding of the circuit as the natural reference (``muf sim check
--next``), sounds the first synthetic copies again for display, and
gathers what the page draws: the real ionogram against a synthetic one or
the next real one, the state's traces, a row cut across the trace, the
trace's mean profile, C/N0 along each mode, the level-2 verdicts, the
correlations against their ceiling, and the channel a signal would see at
one carrier.

It draws what :mod:`muf.sim.check` computes and computes nothing of its
own beyond cropping and encoding. The hub's simulator page
(``services/api``) and ``docs/ui/console.py`` both read it.
"""

from __future__ import annotations

import base64
import math
import tempfile
from pathlib import Path

import numpy as np

#: The dB range the maps are encoded over, muf's scale (noise at -2.8 dB).
DB_LO, DB_HI = -5.0, 30.0

#: Synthetic soundings sounded again for display (the loop's first seeds).
SHOWN = 3

#: Margins of the crop around the modes.
PAD_MHZ = (0.6, 0.4)
PAD_KM = 230.0

#: Synthetic soundings the closed loop reads; check's level 2 needs 12.
REALISATIONS = 12


def _encode(db: np.ndarray) -> str:
    q = np.clip((db - DB_LO) / (DB_HI - DB_LO) * 255.0, 0, 255).astype(np.uint8)
    return base64.b64encode(q.tobytes()).decode("ascii")


def _when(path: Path) -> tuple[str, float] | None:
    """The circuit's file prefix and a number that sorts its soundings in time.

    ``<tag>_<YYYYmmdd>_<HHMMSS>.lfs`` by the date and time, chirpsounder2's
    ``lfm_ionogram-<tx>-<rx>-ch0-000-<epoch>.h5`` by the epoch. None for a
    name in neither form.
    """
    try:
        if path.suffix == ".h5":
            prefix, epoch = path.stem.rsplit("-", 1)
            return prefix + "-", float(epoch)
        prefix, day, hms = path.stem.rsplit("_", 2)
        return prefix + "_", float(day + hms)
    except ValueError:
        return None


def next_of(path: str | Path) -> Path | None:
    """The circuit's following sounding in the same folder, if any."""
    path = Path(path)
    own = _when(path)
    if own is None:
        return None
    prefix, now = own
    later = []
    for f in path.parent.glob(prefix + "*" + path.suffix):
        when = _when(f)
        if when is not None and when[0] == prefix and when[1] > now:
            later.append((when[1], f))
    return min(later)[1] if later else None


def _profile(ion, mode, others, half: int = 4):
    """The trace's mean profile across range, early to late, peak 1."""
    from . import check, observe

    clean = check.cleaned(ion)[0]
    p = np.asarray(clean.power, dtype=np.float64) - observe.NOISE_MEAN_POWER
    f, vr = np.asarray(ion.freq), np.asarray(ion.vrange)
    scale = observe.km_per_us(ion.header)
    bin_us = abs(vr[1] - vr[0]) / scale
    acc, n = np.zeros(2 * half + 1), 0
    for i in np.flatnonzero((f >= mode.f_lo) & (f <= mode.f_hi)):
        tau = float(mode.delay_at(f[i]))
        reach = (half + 3) * bin_us
        if any(o.covers(f[i]) and abs(float(o.delay_at(f[i])) - tau) < reach
               for o in others):
            continue
        j = int(np.argmin(np.abs(vr - tau * scale)))
        if j - half < 0 or j + half >= vr.size:
            continue
        seg = p[i, j - half:j + half + 1]
        if seg.sum() < 50:
            continue
        acc += seg[::-1]          # vrange falls with the index: reversed, early first
        n += 1
    return (acc / acc.max()).tolist() if n else None


def _channel(state, mode) -> dict:
    from . import channel

    fc = mode.f_lo + 0.75 * (mode.f_hi - mode.f_lo)
    fs, seconds = 48e3, 30.0
    real = channel.Realisation(state, fc, fs, seconds, seed=0)
    df = np.linspace(-fs / 2, fs / 2, 129)[:-1]
    t = np.arange(0.0, seconds, 0.1)
    h = real.transfer(df, t)
    db = 20 * np.log10(np.abs(h) + 1e-12)
    db -= np.percentile(db, 99)
    q = np.clip((db + 30.0) / 30.0 * 255.0, 0, 255).astype(np.uint8)
    paths = [{"label": p.label, "delay_us": p.delay_us - real.reference_delay_us,
              "db": p.cn0_dbhz} for p in real.paths]
    return {"fc_mhz": fc, "fs_hz": fs, "seconds": seconds, "n_f": df.size,
            "n_t": t.size, "h": base64.b64encode(q.tobytes()).decode("ascii"),
            "paths": paths}


def build(path: str | Path, *, realisations: int = REALISATIONS) -> dict:
    """Everything the page draws of the sounding at ``path``, JSON without NaN."""
    from . import check, observe, sounding

    path = Path(path)
    nxt = next_of(path)
    truth, report = check.check_sounding(path, next_sounding=nxt,
                                         realisations=realisations)
    real = observe.load(path)
    scale = observe.km_per_us(real.header)
    freq, vrange = np.asarray(real.freq), np.asarray(real.vrange)
    lo = min(m.f_lo for m in truth.modes) - PAD_MHZ[0]
    hi = max(m.f_hi for m in truth.modes) + PAD_MHZ[1]
    km = np.concatenate([m.delay_us * scale for m in truth.modes])
    rows = np.flatnonzero((freq >= lo) & (freq <= hi))
    cols = np.flatnonzero((vrange >= km.min() - PAD_KM) & (vrange <= km.max() + PAD_KM))

    def crop(ion) -> str:
        clean = check.cleaned(ion)[0]
        p = np.asarray(clean.power, dtype=np.float64)[np.ix_(rows, cols)]
        return _encode(10 * np.log10(np.maximum(p, 1e-3)))

    maps = {"real": crop(real)}
    gate = check.gate_of(real)
    shown = []
    with tempfile.TemporaryDirectory() as tmp:
        for k in range(SHOWN):
            p = Path(tmp) / f"s{k}.lfs"
            sounding.write_lfs(truth, p, seed=report["seed"] + k)
            shown.append(observe.load(p, gate_km=gate))
            p.unlink()
    for k, ion in enumerate(shown):
        maps[f"synthetic{k + 1}"] = crop(ion)
    later = observe.load(nxt, gate_km=gate) if nxt is not None else None
    if later is not None and check.same_grid(real, later):
        maps["next"] = crop(later)

    f_rows = freq[rows]
    modes = []
    for m in truth.modes:
        others = [o for o in truth.modes if o is not m]
        inside = (f_rows >= m.f_lo) & (f_rows <= m.f_hi)
        curve = np.where(inside, m.delay_at(np.clip(f_rows, m.f_lo, m.f_hi)) * scale,
                         np.nan)
        clean = check.cleaned(real)[0]
        f_c, c_real = observe._cn0_along(clean, f_rows, np.nan_to_num(curve),
                                         m.f_lo, m.f_hi)
        _f, c_syn = observe._cn0_along(check.cleaned(shown[0])[0], f_rows,
                                       np.nan_to_num(curve), m.f_lo, m.f_hi)
        smooth = observe._smooth(f_c, c_real, observe.SMOOTH_MHZ)
        smooth_syn = observe._smooth(f_c, c_syn, observe.SMOOTH_MHZ)
        s = report["modes"][m.name]
        modes.append({
            "name": m.name, "f_lo": m.f_lo, "f_hi": m.f_hi,
            "km": [None if not np.isfinite(x) else round(float(x), 2) for x in curve],
            "delay_lo": float(m.delay_us.min()), "delay_hi": float(m.delay_us.max()),
            "cn0_max": float(m.cn0_dbhz.max()),
            "doppler_hz": m.doppler_spread_hz, "doppler_shape": m.doppler_shape,
            "delay_spread_us": m.delay_spread_us, "ox_us": m.ox_split_us,
            "origin": {k: m.origin(k) for k in ("delay", "cn0", "doppler_spread",
                                                "delay_spread", "magnetoionic")},
            "cn0": {"f": f_c.round(4).tolist(),
                    "real": (10 * np.log10(c_real)).round(2).tolist(),
                    "real_smooth": (10 * np.log10(smooth)).round(2).tolist(),
                    "synthetic_smooth": (10 * np.log10(smooth_syn)).round(2).tolist()},
            "profile": {"real": _profile(real, m, others),
                        "synthetic": _profile(shown[0], m, others)},
            "width": truth.metrics.get("cross_trace", {}).get(m.name),
            "summary": {k: s[k] for k in ("found", "delay_bias_us", "delay_rms_us",
                                          "cn0_bias_db", "level", "width")},
        })
    main = max(truth.modes, key=lambda m: m.f_hi - m.f_lo)
    corr = report.get("correlation", {})
    natural = report.get("natural") or {}
    return plain({
        "source": path.name, "next": nxt.name if nxt else None,
        "utc": truth.datetime.strftime("%Y-%m-%d %H:%M UTC"),
        "circuit": f"{truth.circuit.tx} → {truth.circuit.rx}",
        "freq": {"lo": float(f_rows[0]), "step": float(np.median(np.diff(f_rows))),
                 "n": int(rows.size)},
        "range": {"km": vrange[cols].round(2).tolist()},
        "bin_km": float(abs(vrange[1] - vrange[0])), "km_per_us": float(scale),
        "db": [DB_LO, DB_HI], "maps": maps, "modes": modes,
        "muf": {"real": report.get("real_muf_mhz"), "z": report.get("real_muf_z"),
                "inside": report.get("real_muf_inside"),
                "synthetic": report["muf_mhz"]},
        "correlation": {k: corr.get(k) for k in ("rows", "smoothed", "cells",
                                                 "half_bins", "soundings")}
        | {"modes": {n: {k: v[k] for k in ("rows", "smoothed")}
                     for n, v in corr.get("modes", {}).items()},
           "next": natural.get("correlation")},
        "natural": {"dt_s": natural.get("dt_s"),
                    "muf_change_mhz": natural.get("muf_change_mhz"),
                    "modes": {n: {"delay_rms_us": e["delay_rms_us"],
                                  "cn0_bias_db": e["cn0_bias_db"]}
                              for n, e in (natural.get("modes") or {}).items()
                              if e.get("matched")}},
        "realisations": report["realisations"],
        "channel": _channel(truth, main),
    })


def plain(value):
    """JSON without NaN: None in its place."""
    if isinstance(value, float):
        return None if not math.isfinite(value) else round(value, 5)
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return plain(value.item())
    return value
