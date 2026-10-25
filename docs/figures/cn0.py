"""What C/N0 is on an ionogram, drawn from a real sounding.

    python docs/figures/cn0.py SOUNDING.lfs [--mode 1F] [--at 24] [--lang ru]

writes ``cn0.png`` (or ``cn0.ru.png``) next to this file: the ionogram with
a mode's trace, one row cut across it at ``--at`` MHz -- the five range bins
summed, the noise they carry taken off, the rest scaled to C/N0 -- and the
mode's C/N0 along its band, row by row and as the state keeps it.
:mod:`muf.sim.observe` computes it so; this only draws the same numbers.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

TEXT = {
    "en": {
        "title": ("C/N₀: the echo's power over the noise's power in 1 Hz. "
                  "Not tied to a window or a band: SNR in B = C/N₀ − 10 lg B"),
        "a": "a. Ionogram, muf's scale: each row over 4 ln2 × its median, "
             "noise at −2.8 dB",
        "freq": "frequency, MHz", "range": "range, km", "power": "power, dB",
        "trace": "trace of {mode}", "strip": "±2 bins summed",
        "b": "b. Row at {f:.2f} MHz across the trace",
        "bin": "range bin from the trace", "lin": "power, muf's scale",
        "noise": "noise per bin, 1/(4 ln²2)", "sum": "summed: 5 bins",
        "formula": ("tone = (Σ − 5·noise) / 1.5\n"
                    "C/N₀ = tone · 6 ln²2 / T,  T = {t:.3f} s\n"
                    "= {cn0:.1f} dB-Hz"),
        "c": "c. C/N₀ of {mode} along its band",
        "rows": "each row (fading)", "smooth": "±0.2 MHz mean: in the state",
        "below": "{n} rows below the noise, off the scale",
        "cn0": "C/N₀, dB-Hz", "snr": "SNR in 3 kHz, dB",
    },
    "ru": {
        "title": ("C/N₀ — мощность эха к мощности шума в полосе 1 Гц. Не зависит "
                  "от окна и полосы: ОСШ в полосе B = C/N₀ − 10 lg B"),
        "a": "а. Ионограмма в шкале muf: строка делится на 4 ln2 × её медиану, "
             "шум в среднем −2,8 дБ",
        "freq": "частота, МГц", "range": "дальность, км", "power": "мощность, дБ",
        "trace": "след {mode}", "strip": "±2 бина в сумме",
        "b": "б. Строка {f:.2f} МГц поперёк следа",
        "bin": "бин дальности от следа", "lin": "мощность, шкала muf",
        "noise": "шум на бин, 1/(4 ln²2)", "sum": "сумма: 5 бинов",
        "formula": ("тон = (Σ − 5·шум) / 1,5\n"
                    "C/N₀ = тон · 6 ln²2 / T,  T = {t:.3f} с\n"
                    "= {cn0:.1f} дБ·Гц"),
        "c": "в. C/N₀ моды {mode} вдоль её полосы",
        "rows": "каждая строка (замирания)", "smooth": "среднее ±0,2 МГц: в состоянии",
        "below": "{n} строк ниже шума, вне шкалы",
        "cn0": "C/N₀, дБ·Гц", "snr": "ОСШ в 3 кГц, дБ",
    },
}


def main(argv=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from muf import interference, loader
    from muf.pipeline import Options
    from muf.sim import observe

    ap = argparse.ArgumentParser()
    ap.add_argument("sounding", type=Path)
    ap.add_argument("--mode", default="1F")
    ap.add_argument("--at", type=float, default=24.0, help="row to cut, MHz")
    ap.add_argument("--lang", choices=("en", "ru"), default="en")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    t = TEXT[args.lang]

    raw = loader.load(args.sounding)
    state = observe.from_ionogram(raw, source=args.sounding.name)
    mode = next(m for m in state.modes if m.name == args.mode)
    ion, _ = interference.apply(raw, Options())
    scale = observe.km_per_us(ion.header)
    t_w = observe.window_seconds(ion)
    freq, vrange = np.asarray(ion.freq), np.asarray(ion.vrange)
    power = np.asarray(ion.power, dtype=np.float64)

    def trace_km(f):
        return mode.delay_at(np.clip(f, mode.f_lo, mode.f_hi)) * scale

    fig = plt.figure(figsize=(12, 8.6))
    fig.suptitle(t["title"], x=0.08, ha="left", fontsize=12)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.38, wspace=0.28)

    # a. The ionogram around the mode.
    ax = fig.add_subplot(grid[0, :])
    r_mode = trace_km(np.linspace(mode.f_lo, mode.f_hi, 50))
    r_lo, r_hi = r_mode.min() - 250, r_mode.max() + 250
    cols = (vrange >= r_lo) & (vrange <= r_hi)
    rows = (freq >= mode.f_lo - 1.0) & (freq <= mode.f_hi + 0.5)
    img = 10 * np.log10(np.maximum(power[np.ix_(rows, cols)], 1e-3)).T
    mesh = ax.pcolormesh(freq[rows], vrange[cols], img, shading="nearest",
                         cmap="viridis", vmin=-5, vmax=25, rasterized=True)
    fig.colorbar(mesh, ax=ax, pad=0.01, label=t["power"])
    f_line = np.linspace(mode.f_lo, mode.f_hi, 400)
    bin_km = float(np.median(np.abs(np.diff(vrange))))
    half = (observe.SUM_HALF_BINS + 0.5) * bin_km
    ax.plot(f_line, trace_km(f_line), color="w", lw=1.0, label=t["trace"].format(
        mode=mode.name))
    ax.fill_between(f_line, trace_km(f_line) - half, trace_km(f_line) + half,
                    color="w", alpha=0.18, lw=0, label=t["strip"])
    ax.axvline(args.at, color="#F0997B", lw=1.2, ls="--")
    ax.set_xlabel(t["freq"]), ax.set_ylabel(t["range"])
    ax.set_title(t["a"], loc="left")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.6)

    # b. One row across the trace.
    ax = fig.add_subplot(grid[1, 0])
    i = int(np.argmin(np.abs(freq - args.at)))
    j = int(np.argmin(np.abs(vrange - trace_km(freq[i]))))
    k = np.arange(-12, 13)
    cut = power[i, np.clip(j + k, 0, vrange.size - 1)]
    summed = np.abs(k) <= observe.SUM_HALF_BINS
    ax.bar(k[~summed], cut[~summed], color="#B4B2A9", width=0.85)
    ax.bar(k[summed], cut[summed], color="#1D9E75", width=0.85, label=t["sum"])
    ax.axhline(observe.NOISE_MEAN_POWER, color="#D85A30", lw=1.2, label=t["noise"])
    band = power[i, j - observe.SUM_HALF_BINS:j + observe.SUM_HALF_BINS + 1]
    tone = (band.sum() - band.size * observe.NOISE_MEAN_POWER) / observe.HANN_ENBW_BINS
    cn0 = 10 * math.log10(max(tone, 1e-6) / (observe.TONE_PER_CN0_SECOND * t_w))
    ax.text(0.98, 0.97, t["formula"].format(t=t_w, cn0=cn0), transform=ax.transAxes,
            ha="right", va="top", fontsize=9.5,
            bbox={"boxstyle": "round", "fc": "white", "ec": "#B4B2A9"})
    ax.set_yscale("log")
    ax.set_ylim(top=float(cut.max()) * 300)
    ax.set_xlabel(t["bin"]), ax.set_ylabel(t["lin"])
    ax.set_title(t["b"].format(f=freq[i]), loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.74), fontsize=9)

    # c. Along the band.
    ax = fig.add_subplot(grid[1, 1])
    f_rows, cn0_rows = observe._cn0_along(ion, f_line, trace_km(f_line), mode.f_lo,
                                          mode.f_hi)
    smooth = observe._smooth(f_rows, cn0_rows, observe.SMOOTH_MHZ)
    ax.plot(f_rows, 10 * np.log10(cn0_rows), ".", ms=2.5, color="#888780",
            label=t["rows"])
    ax.plot(f_rows, 10 * np.log10(smooth), color="#0F6E56", lw=2, label=t["smooth"])
    ax.axvline(args.at, color="#F0997B", lw=1.2, ls="--")
    ax.set_xlabel(t["freq"]), ax.set_ylabel(t["cn0"])
    lo = 10 * np.log10(np.min(smooth)) - 15
    ax.set_ylim(lo, 10 * np.log10(np.max(smooth)) + 8)
    below = int(np.count_nonzero(10 * np.log10(cn0_rows) < lo))
    if below:
        ax.text(0.98, 0.04, t["below"].format(n=below), transform=ax.transAxes,
                ha="right", fontsize=8.5, color="#5F5E5A")
    right = ax.secondary_yaxis("right", functions=(lambda v: v - 10 * math.log10(3000),
                                                   lambda v: v + 10 * math.log10(3000)))
    right.set_ylabel(t["snr"])
    ax.set_title(t["c"].format(mode=mode.name), loc="left")
    ax.legend(loc="upper left", fontsize=9)

    out = args.out or HERE / ("cn0.png" if args.lang == "en" else "cn0.ru.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
