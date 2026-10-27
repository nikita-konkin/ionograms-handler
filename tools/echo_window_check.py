"""Measure what an echo window keeps, on real ``.lfs`` captures.

The real-data half of ``docs/echo-window.md``. For each capture: cut the
window into ``--out`` (never beside the capture), then compare what the
window gives back against what the full capture gives:

* size -- the window, the standalone file, and both against the ``.lfs``;
* compression -- what gzip-4 with shuffle would save on the samples;
* MUF -- contour and algo through the whole pipeline, full ``.lfs`` (its
  default gate) against the window file, in frequency bins;
* power -- median |dB difference| over cells 10 dB above the noise, on the
  same gate and axes;
* Doppler -- with ``--doppler LO-HI``, the RMS width of the 1F mode along
  that band, by the estimator of ``sw_channel_modeling/lfs_tools/
  doppler_along_sweep.py``, run independently on the capture and on the
  reconstructed window.

The estimator is ported rather than imported: that repository is a separate
project, and this check must keep meaning the same thing if it moves on. What
it does: trace the 1F peak per 0.1 s segment, fit a cubic beat trajectory,
dechirp a second time along it, block-average to 100 Hz, and take a Welch
spectrum (10 s Hann, 50 % overlap); the width is the RMS spread within
+/-3 Hz above the median floor at 10-40 Hz.

    python tools/echo_window_check.py /path/cyprus1_*.lfs --out /tmp/ew \\
        --doppler 120010:22.8-26.8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from muf import echo_window, io_lfs, spectro
from muf.pipeline import Options, process_file

SEG_S = 0.1                 # trace segment, s -> 10 kHz of sweep at 100 kHz/s
TRACE_BEAT_HZ = (-1600.0, -400.0)
TRACE_SNR_DB = 15.0
BLOCK = 400                 # 40 kHz -> 100 Hz
WELCH_S = 10.0
CORE_HZ = 3.0
FLOOR_HZ = (10.0, 40.0)


def _compression(samples: np.ndarray, scratch: Path) -> float:
    """Fraction of the bytes gzip-4 + shuffle saves on these samples."""
    import h5py

    path = scratch / "_compression_probe.h5"
    with h5py.File(path, "w") as fh:
        fh.create_dataset("z", data=samples, chunks=(min(len(samples), 65536),),
                          compression="gzip", compression_opts=4, shuffle=True)
    with h5py.File(path, "r") as fh:
        stored = fh["z"].id.get_storage_size()
    path.unlink()
    return 1.0 - stored / samples.nbytes


def _trace(x, fs, rate, noise_band=None, beat_hz=TRACE_BEAT_HZ):
    """Per-segment 1F beat and SNR, as ``lfs_trace.trace``.

    ``noise_band`` limits the noise median to beats inside it: a
    reconstructed window is zero outside its band, and a median taken over
    the whole spectrum would then report a noise floor that is not there.
    ``beat_hz`` is where the peak is looked for; the default is cyprus1's 1F
    at 100 kHz/s, and a window's own band serves on any other circuit.
    """
    n = round(SEG_S * fs)
    nseg = len(x) // n
    seg = np.asarray(x[:nseg * n]).reshape(nseg, n)
    seg = seg - seg.mean(axis=1, keepdims=True)
    nfft = 8 * n
    spec = np.fft.fftshift(np.fft.fft(seg * np.hanning(n), nfft, axis=1), axes=1)
    fb = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / fs))
    power = np.abs(spec) ** 2
    in_band = np.ones(fb.size, bool) if noise_band is None else (
        (fb >= noise_band[0]) & (fb <= noise_band[1]))
    noise = np.median(power[:, in_band], axis=1)
    sel = (fb >= beat_hz[0]) & (fb <= beat_hz[1])
    k = np.argmax(power[:, sel], axis=1)
    peak = power[:, sel][np.arange(nseg), k]
    snr = 10 * np.log10(peak / noise)
    tseg = (np.arange(nseg) + 0.5) * SEG_S
    return tseg, fb[sel][k], snr


def _width(z, fsd):
    nper = int(WELCH_S * fsd)
    segs = [z[i:i + nper] * np.hanning(nper)
            for i in range(0, len(z) - nper + 1, nper // 2)]
    s = np.mean([np.abs(np.fft.fft(seg)) ** 2 for seg in segs], axis=0)
    fr = np.fft.fftfreq(nper, 1 / fsd)
    floor = np.median(s[(np.abs(fr) > FLOOR_HZ[0]) & (np.abs(fr) < FLOOR_HZ[1])])
    core = np.abs(fr) <= CORE_HZ
    sc = np.clip(s[core] - floor, 0, None)
    shift = float(np.sum(fr[core] * sc) / np.sum(sc))
    rms = float(np.sqrt(np.sum((fr[core] - shift) ** 2 * sc) / np.sum(sc)))
    return rms, shift


def trajectory(x, fs, rate, f0_mhz, band_mhz, noise_band=None,
               beat_hz=TRACE_BEAT_HZ):
    """Cubic fit of the 1F beat along ``band_mhz``, outliers dropped."""
    tseg, fb, snr = _trace(x, fs, rate, noise_band, beat_hz)
    f = f0_mhz + rate * tseg / 1e6
    m = (snr >= TRACE_SNR_DB) & (f >= band_mhz[0]) & (f <= band_mhz[1])
    idx = np.where(m)[0]
    c = np.polyfit(tseg[idx], fb[idx], 3)
    for _ in range(3):
        r = fb[m] - np.polyval(c, tseg[m])
        idx = np.where(m)[0][np.abs(r) < 3 * np.std(r)]
        c = np.polyfit(tseg[idx], fb[idx], 3)
    return c, int(m.sum())


def along(x, fs, rate, f0_mhz, band_mhz, c, block=BLOCK):
    """1F's complex amplitude along ``band_mhz``: ``x`` dechirped a second time
    on the trajectory ``c`` and block-averaged to ``fs / block``."""
    t1 = (band_mhz[0] - f0_mhz) / (rate / 1e6)
    t2 = (band_mhz[1] - f0_mhz) / (rate / 1e6)
    i0, i1 = int(t1 * fs), int(t2 * fs)
    ts = np.arange(i0, i1) / fs
    phase = 2 * np.pi * np.cumsum(np.polyval(c, ts)) / fs
    z = np.asarray(x[i0:i1]) * np.exp(-1j * phase)
    n = len(z) // block * block
    return z[:n].reshape(-1, block).mean(axis=1)


def doppler(x, fs, rate, f0_mhz, band_mhz, noise_band=None, fit=None,
            beat_hz=TRACE_BEAT_HZ, block=BLOCK):
    """RMS Doppler width and residual shift of 1F along ``band_mhz``.

    ``fit`` reuses a trajectory instead of tracing ``x`` -- which separates
    what the window changed in the *signal* from what it changed in which
    segments the tracer accepted. ``block`` is 400 at 40 kHz, to 100 Hz.
    """
    c, segments = fit if fit is not None else trajectory(
        x, fs, rate, f0_mhz, band_mhz, noise_band, beat_hz)
    rms, shift = _width(along(x, fs, rate, f0_mhz, band_mhz, c, block), fs / block)
    return {"rms_hz": rms, "shift_hz": shift, "segments": segments}


def check(lfs: Path, out: Path, doppler_band=None) -> dict:
    t = time.time()
    report = echo_window.write(lfs, out_dir=out, verify=True)
    elapsed = time.time() - t
    window = echo_window.read(report.target)
    row = {
        "file": lfs.name,
        "status": report.status,
        "verification": (report.verification.describe()
                         if report.verification else None),
        "seconds": round(elapsed, 1),
        "range_km": [round(window.range_lo_km), round(window.range_hi_km)],
        "selection": window.selection,
        "edge_energy_db": [round(v, 1) for v in window.edge_energy_db],
        "truncated_suspect": window.truncated_suspect,
        "shift_hz": round(window.shift_hz, 2),
        "decimation": window.decimation,
        "sr_hz": round(window.sr_hz, 2),
        "numtaps": window.plan.numtaps,
        "attenuation_db": window.filter["stopband_attenuation_db"],
        "ripple_db": window.filter["passband_ripple_db"],
        "samples_bytes": window.nbytes,
        "file_bytes": report.target.stat().st_size,
        "lfs_bytes": lfs.stat().st_size,
        "gzip_saves": round(_compression(window.samples, out), 4),
    }

    options = Options(methods=("contour", "algo"))
    full, cut = process_file(lfs, options), process_file(report.target, options)
    ion_full = spectro.compute(lfs, gate_km=(window.range_lo_km, window.range_hi_km))
    step = float(ion_full.cal.freq_step_mhz)
    for name in ("contour", "algo"):
        a, b = full.get(f"muf_{name}"), cut.get(f"muf_{name}")
        row[f"muf_{name}"] = [a, b]
        row[f"muf_{name}_bins"] = (None if a is None or b is None
                                   or not (np.isfinite(a) and np.isfinite(b))
                                   else round(abs(a - b) / step, 2))
    for key in ("error",):
        if full.get(key) or cut.get(key):
            row["pipeline_error"] = [full.get(key), cut.get(key)]

    derived = echo_window.reprocess_window(report.target)
    db_full, db_cut = np.asarray(ion_full.db), np.asarray(derived.db)
    above = db_full >= echo_window.NOISE_DB + echo_window.POWER_CHECK_ABOVE_NOISE_DB
    row["power_cells"] = int(above.sum())
    row["power_median_abs_db"] = round(float(np.median(np.abs(
        db_full[above] - db_cut[above]))), 4) if above.any() else None

    if doppler_band is not None:
        header = io_lfs.read_header(lfs)
        fs = header.sample_rate / header.dec
        f0 = (header.cf - header.sample_rate / 2) / 1e6
        x = io_lfs.read_iq(lfs)
        rebuilt = echo_window.reconstruct(window)
        half = window.plan.passband_hz
        band = (window.shift_hz - half, window.shift_hz + half)
        row["doppler_band_mhz"] = list(doppler_band)
        fit = trajectory(x, fs, header.rate, f0, doppler_band)
        row["doppler_full"] = doppler(x, fs, header.rate, f0, doppler_band, fit=fit)
        row["doppler_window"] = doppler(rebuilt, fs, header.rate, f0, doppler_band,
                                        noise_band=band)
        row["doppler_window_same_fit"] = doppler(rebuilt, fs, header.rate, f0,
                                                 doppler_band, fit=fit)
        a = row["doppler_full"]["rms_hz"]
        for key in ("doppler_window", "doppler_window_same_fit"):
            row[key + "_rel_diff"] = round(abs(row[key]["rms_hz"] - a) / a, 5)
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("lfs", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True,
                    help="where the windows go; never the capture's folder")
    ap.add_argument("--doppler", action="append", default=[],
                    metavar="STEM_PART:LO-HI",
                    help="measure 1F Doppler width on files whose name "
                         "contains STEM_PART, along LO-HI MHz")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    bands = {}
    for spec in args.doppler:
        key, rng = spec.split(":")
        lo, hi = (float(v) for v in rng.split("-"))
        bands[key] = (lo, hi)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for lfs in args.lfs:
        if args.out.resolve() == lfs.parent.resolve():
            ap.error("--out must not be the captures' own folder")
        band = next((b for k, b in bands.items() if k in lfs.name), None)
        row = check(lfs, args.out, band)
        rows.append(row)
        print(json.dumps(row, default=str), flush=True)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
