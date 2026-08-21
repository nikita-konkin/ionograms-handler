"""Drawing ionograms.

Replaces ``MUF.py``'s plotting modes. The y-axis here is labelled from
``calibrate``'s descending virtual-range axis rather than being reversed with
``[::-1]`` at draw time (``MUF.py:116``), so what is plotted and what is
measured are the same array.
"""

from __future__ import annotations

from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd

from .extractors import MufResult
from .spectro import Ionogram

DEFAULT_VMIN_DB = 20.0
DEFAULT_VMAX_DB = 75.0
DEFAULT_DPI = 150

#: The raster's colormap. Named once because two surfaces have to agree on it:
#: the PNG drawn here, and the colour bar the interactive plot puts beside that
#: PNG. A bar built from a different map than the image it explains is worse
#: than no bar, because it reads as authoritative.
DEFAULT_CMAP = "jet"

#: What the colour axis actually is. `spectro.compute` divides every spectrum
#: by its own noise floor before `to_db`, so a cell is already a ratio against
#: the noise -- the same quantity the SAO record spells `MUFSignalToNoise`.
COLOUR_LABEL = "SNR (dB)"

_MARKER_COLOURS = {
    "algo": "#ffffff",
    "kmeans": "#00ff88",
    "contour": "#ff2a6d",
    "cnn": "#ffd166",
}


def colour_scale(cmap: str | None = None, stops: int = 32) -> list[list]:
    """The raster's colormap, sampled as plotly colorscale stops.

    For the interactive plot, which shows the raster as a PNG behind its traces
    and so has nothing carrying a colour axis of its own -- a picture in false
    colour with no key to it. Derived from the same colormap `plot` draws with
    rather than transcribed into the template, because a hand-copied scale goes
    quietly wrong the day someone changes `DEFAULT_CMAP` and stays wrong: the
    bar would keep looking authoritative while describing a different image.

    32 stops rather than the full 256: plotly interpolates between them, jet is
    piecewise linear over about eight anchors, and the difference is invisible
    against a 55 dB span while the frame stays small.
    """
    from matplotlib import colormaps

    table = colormaps[cmap or DEFAULT_CMAP]
    out = []
    for index in range(stops):
        position = index / (stops - 1)
        red, green, blue, _ = table(position)
        out.append([round(position, 4),
                    f"rgb({red * 255:.0f},{green * 255:.0f},{blue * 255:.0f})"])
    return out


def _figure(figsize, **kwargs):
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    return plt, plt.subplots(figsize=figsize, **kwargs)


#: Colours cycled over trace segments. The primary one -- the segment carrying
#: the MUF -- is drawn first and heaviest.
_SEGMENT_COLOURS = ("#ffffff", "#00e5ff", "#ffd166", "#c77dff", "#8ac926")


#: Margin around the detected echoes when the y-axis is auto-focused.
RANGE_MARGIN_KM = 250.0


def _range_limits(ion, segments) -> tuple[float, float]:
    """Y limits: around the echoes when we know where they are, else the gate.

    The gate spans 2,300-5,000 km while the echoes occupy a few hundred of
    those, so plotting the whole gate squashes the trace into a sliver.
    """
    if segments:
        values = np.concatenate([s.vrange for s in segments])
        if values.size:
            return (max(ion.vrange.min(), values.min() - RANGE_MARGIN_KM),
                    min(ion.vrange.max(), values.max() + RANGE_MARGIN_KM))
    return ion.vrange.min(), ion.vrange.max()


def _overlay_trace(ax, segments, reconstruction) -> None:
    """Draw detected trace points per mode, and the reconstructed curve."""
    primary = reconstruction.segment if reconstruction is not None else None

    for i, item in enumerate(segments or []):
        colour = _SEGMENT_COLOURS[i % len(_SEGMENT_COLOURS)]
        is_primary = primary is not None and item is primary
        mode = f"{item.hops}-hop" if item.hops else "mode ?"
        ax.plot(
            item.freq, item.vrange, "o",
            markersize=4.5 if is_primary else 3.0,
            markerfacecolor="none" if not is_primary else colour,
            markeredgecolor=colour,
            markeredgewidth=1.2,
            alpha=0.95 if is_primary else 0.6,
            linestyle="none",
            label=f"{mode}, {item.n_points} pts" + (" (primary)" if is_primary else ""),
        )

    if reconstruction is not None and reconstruction.ok:
        ax.plot(reconstruction.freq, reconstruction.vrange, "-",
                color="#ff2a6d", linewidth=2.4, alpha=0.95,
                label=f"reconstructed ({reconstruction.rms_residual_km:.1f} km rms)")


def _destination(out_path):
    """A path (whose parent is created) or an already-open binary stream.

    Detected by `write`, not by type: BytesIO, a real file and a socket-backed
    stream are all valid targets for `savefig` and share no base class worth
    checking against.
    """
    if hasattr(out_path, "write"):
        return out_path
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def plot(
    ion: Ionogram,
    out_path: str | Path | IO[bytes],
    results: dict[str, MufResult] | None = None,
    vmin_db: float = DEFAULT_VMIN_DB,
    vmax_db: float = DEFAULT_VMAX_DB,
    axes: bool = True,
    dpi: int = DEFAULT_DPI,
    figsize: tuple[float, float] = (16, 7),
    cmap: str | None = None,
    segments=None,
    reconstruction=None,
) -> Path:
    """Render one ionogram, optionally marking MUFs and the reconstructed trace.

    ``out_path`` may be a path or an open binary file. The second is for the
    on-demand renderer (``architecture.md`` sec. 4.2), which serves a PNG per
    request and has no reason to touch the filesystem to do it -- a temporary
    file per request would be an extra failure mode and a cleanup problem in a
    container.
    """
    out_path = _destination(out_path)

    plt, (fig, ax) = _figure(figsize)

    # pcolormesh wants [y, x] = [range, frequency].
    mesh = ax.pcolormesh(
        ion.freq, ion.vrange, ion.db.T,
        shading="nearest", cmap=cmap or DEFAULT_CMAP, vmin=vmin_db, vmax=vmax_db,
    )

    if axes:
        ax.set_xlabel("Frequency (MHz)", fontsize=13)
        ax.set_ylabel("Virtual range (km)", fontsize=13)
        header = ion.header
        title = (
            f"{header.tx_name} - {header.rx_name} ({header.path_type})\n"
            f"{header.datetime:%Y-%m-%d %H:%M:%S}Z"
        )
        if results:
            picked = [
                f"{name} {r.pick.muf_mhz:.2f}"
                for name, r in results.items() if r.ok
            ]
            if picked:
                title += "    MUF: " + "  |  ".join(picked) + " MHz"
        ax.set_title(title, fontsize=14)

        bar = fig.colorbar(mesh, ax=ax, pad=0.01)
        bar.set_label(COLOUR_LABEL)

        if segments or reconstruction is not None:
            _overlay_trace(ax, segments, reconstruction)

        if results:
            for name, result in results.items():
                if not result.ok:
                    continue
                ax.axvline(
                    result.pick.muf_mhz,
                    color=_MARKER_COLOURS.get(name, "#ffffff"),
                    linestyle="--", linewidth=1.4, alpha=0.9, label=name,
                )
        if results or segments or reconstruction is not None:
            ax.legend(loc="upper left", framealpha=0.75, fontsize=9)

        ax.set_xlim(ion.cal.freq_start, ion.cal.freq_stop)
        ax.set_ylim(*_range_limits(ion, segments))
        fig.tight_layout()
    else:
        # Bare raster, for feeding image-based tools.
        ax.set_axis_off()
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.savefig(
        out_path, dpi=dpi,
        bbox_inches="tight" if not axes else None,
        pad_inches=0 if not axes else 0.1,
    )
    plt.close(fig)
    return out_path


#: Trace colours for a SAO plot. A white background, so these are darker than
#: ``_SEGMENT_COLOURS``, which are chosen to sit on top of a jet raster.
#:
#: A Digisonde ionogram colours its traces by polarization, red for O and green
#: for X. This receiver has no polarimetry, so the same visual channel carries
#: the distinction it *can* make -- low ray against high ray -- and the legend
#: says so, rather than borrowing a convention that would be read as O/X.
SAO_BRANCH_COLOURS = {"low": "#1f6feb", "high": "#d1495b", "": "#8d99ae"}

#: The same three over a jet raster, where the white-background palette would
#: vanish into the colormap's own blue and red ends. Drawn as open rings there,
#: so the pixels the labels are claiming stay visible underneath.
SAO_BRANCH_COLOURS_ON_RASTER = {"low": "#ffffff", "high": "#ffd166",
                                "": "#c77dff"}

#: Marker area at the weakest and strongest echo of a record.
SAO_MARKER_AREA = (10.0, 70.0)

#: Padding around the traces, as a fraction of their own extent.
SAO_MARGIN = 0.08


def _sao_sizes(trace, low: float, high: float):
    """Marker areas from echo amplitude, or a constant when there are none."""
    if (trace.amplitude is None or trace.amplitude.size != trace.freq.size
            or high <= low):
        return np.full(trace.freq.shape, float(np.mean(SAO_MARKER_AREA)))
    unit = (trace.amplitude - low) / (high - low)
    lo, hi = SAO_MARKER_AREA
    return lo + np.clip(unit, 0.0, 1.0) * (hi - lo)


def _sao_limits(values, *include) -> tuple[float, float]:
    """Extent of ``values`` and any extra points, padded.

    The extras are folded in *before* padding so that a line drawn at one of
    them -- the MUF, the ground range -- sits inside the frame rather than
    along its edge, where it reads as part of the axis.
    """
    wanted = [float(np.min(values)), float(np.max(values))]
    wanted += [float(v) for v in include if v is not None]
    low, high = min(wanted), max(wanted)
    pad = max((high - low) * SAO_MARGIN, 0.1)
    return low - pad, high + pad


def draw_trace_points(ion, trace: bool | None = None) -> bool:
    """Whether a SAO plot should overlay the scaled trace points.

    Off by default once there is a raster to look at. The points carry the
    branch labels, and those come from segmentation -- the least settled step
    in this pipeline -- while the raster is simply what was measured. Drawing
    the interpretation on top of the evidence by default would put the two on
    equal footing. ``trace`` overrides in either direction.
    """
    return (ion is None) if trace is None else bool(trace)


def _sao_frame(ax, record, ion, drawn, full_band, marks, ground) -> None:
    """Set the axis limits, from whichever of the two sources is present.

    ``drawn`` frames the view even when the points are not being plotted: the
    record still says where the echoes are, and cropping to them is what makes
    a 25 MHz sweep legible.

    With a raster the frequency axis always spans the whole sweep, because that
    is what an ionogram *is* -- cropped to the echoes it stops being one and
    starts being a picture of a few hundred bins. Only the range axis narrows,
    exactly as ``plot`` does, and ``full_band`` opens it back to the gate.
    """
    freq = np.concatenate([t.freq for t in drawn]) if drawn else None
    vrange = np.concatenate([t.vrange for t in drawn]) if drawn else None

    if ion is not None:
        ax.set_xlim(ion.cal.freq_start, ion.cal.freq_stop)
        if full_band or vrange is None:
            ax.set_ylim(float(ion.vrange.min()), float(ion.vrange.max()))
        else:
            ax.set_ylim(
                max(float(ion.vrange.min()), float(vrange.min()) - RANGE_MARGIN_KM),
                min(float(ion.vrange.max()), float(vrange.max()) + RANGE_MARGIN_KM),
            )
        return

    if freq is None:
        return

    if full_band:
        start = record.number("StartFrequency")
        stop = record.number("StopFrequency")
        ax.set_xlim(start if start is not None else float(freq.min()),
                    stop if stop is not None else float(freq.max()))
        ax.set_ylim(*_sao_limits(vrange, ground))
    else:
        ax.set_xlim(*_sao_limits(freq, *marks))
        ax.set_ylim(*_sao_limits(vrange))


def _sao_panel(ax, record) -> None:
    """The left-hand scaled-values panel, in the style of a Digisonde print."""
    ax.set_axis_off()

    distance = record.number("GreatCircleDistance", "path")
    head = [
        f"{record.path.get('TransmitterName', '?')}"
        f" → {record.path.get('ReceiverName', record.station)}",
        f"{record.time:%Y-%m-%d %H:%M:%S} UTC" if record.time else "",
        f"{record.path_type}" + (f", D = {distance:.0f} km" if distance else ""),
        record.scaler,
    ]
    if record.ursi_code:
        head.insert(2, f"URSI {record.ursi_code}")

    # Measured first, modelled after a blank line. The XML keeps them in the
    # order the format wants; the panel keeps them in the order a reader needs,
    # so a modelled value is never mistaken for one this instrument produced.
    measured, modelled = [], []
    for item in record.characteristics:
        name = item.name
        if item.model:
            # Two entries can share a Name -- an IRI foF2 and a secant-law one
            # -- so the model has to be on the row or the panel lies.
            name += f" ({item.model})"
        if item.modelled:
            name += "*"
        # The file's own digits, not our reformatting: see Characteristic.text.
        value = item.text or f"{item.value:g}"
        if item.bound is not None:
            value += f" ± {item.bound:g}"
        row = (f"{name:<20s}{value:>9s} {item.units}"
               + (f" {item.letter}" if item.letter else ""))
        (modelled if item.modelled else measured).append(row)

    rows = measured + ([""] + modelled if modelled and measured else modelled)
    if not rows:
        # An empty CharacteristicList is a real outcome, not a missing panel.
        rows = ["(nothing scaled)"]

    notes = []
    if any(c.modelled for c in record.characteristics):
        notes.append("* modelled, not measured")
    letters = sorted({c.letter for c in record.characteristics if c.letter})
    for letter in letters:
        notes.append(f"{letter} = {_QUALIFYING_LETTERS.get(letter, 'see UAG-23A')}")

    head = [line for line in head if line]
    ax.text(0.0, 1.0, "\n".join(head), transform=ax.transAxes,
            va="top", ha="left", fontsize=10.5, family="monospace",
            linespacing=1.6)
    # Start the values a blank line under the heading, wherever it ended,
    # rather than at a fixed height that leaves a hole when it is short.
    ax.text(0.0, 1.0 - 0.042 * (len(head) + 1.2), "\n".join(rows),
            transform=ax.transAxes, va="top", ha="left", fontsize=9.5,
            family="monospace", linespacing=1.9)
    if notes:
        ax.text(0.0, 0.0, "\n".join(notes), transform=ax.transAxes,
                va="bottom", ha="left", fontsize=8, family="monospace",
                color="#555555", linespacing=1.6)


#: Expansions for the letters ``saoxml.qualifying_letter`` can assign.
_QUALIFYING_LETTERS = {
    "D": "greater than (above the band ceiling)",
    "E": "less than (below the band floor)",
    "U": "uncertain or doubtful",
}


def plot_sao(
    record,
    out_path: str | Path,
    ion: Ionogram | None = None,
    dpi: int = DEFAULT_DPI,
    figsize: tuple[float, float] = (13.0, 6.5),
    full_band: bool = False,
    trace: bool | None = None,
    vmin_db: float = DEFAULT_VMIN_DB,
    vmax_db: float = DEFAULT_VMAX_DB,
    cmap: str | None = None,
) -> Path:
    """Draw one parsed ``<SAORecord>``: scaled values beside the sounding.

    Everything in the panel comes out of the XML, so this renders a record from
    anyone's sounder -- and it is the check that the export is self-describing
    rather than merely well formed.

    Pass ``ion`` to put the ionogram itself behind the annotations. That is the
    honest default when the ``.lfs`` is to hand: the raster is what was
    measured, while the branch labels are an interpretation of it, and
    segmentation is the least settled part of this pipeline. So the trace points
    are drawn only when there is no raster to speak for itself, unless ``trace``
    asks for them.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt, (fig, (panel, ax)) = _figure(figsize, ncols=2,
                                      gridspec_kw={"width_ratios": (1.0, 2.7)})
    _sao_panel(panel, record)

    mesh = None
    if ion is not None:
        mesh = ax.pcolormesh(ion.freq, ion.vrange, ion.db.T, shading="nearest",
                             cmap=cmap or DEFAULT_CMAP, vmin=vmin_db,
                             vmax=vmax_db, zorder=0)

    drawn = [t for t in record.traces
             if t.freq.size and t.freq.size == t.vrange.size]
    plotted = drawn if draw_trace_points(ion, trace) else []

    amplitudes = [t.amplitude for t in plotted
                  if t.amplitude is not None and t.amplitude.size]
    low, high = ((float(np.min(np.concatenate(amplitudes))),
                  float(np.max(np.concatenate(amplitudes))))
                 if amplitudes else (0.0, 0.0))

    colours = SAO_BRANCH_COLOURS_ON_RASTER if mesh is not None \
        else SAO_BRANCH_COLOURS
    for item in plotted:
        colour = colours.get(item.branch, colours[""])
        ax.scatter(item.freq, item.vrange, s=_sao_sizes(item, low, high),
                   facecolors=colour if item.branch and mesh is None else "none",
                   edgecolors=colour, linewidths=1.1, alpha=0.85,
                   label=f"{item.label}, {item.n_points} pts", zorder=3)

    ground = record.number("GreatCircleDistance", "path")
    if ground and full_band:
        ax.axhline(ground, color="#2a9d8f", linestyle=":", linewidth=1.4,
                   zorder=1, label=f"ground range {ground:.0f} km")

    muf = record.muf
    if muf is not None:
        ax.axvline(muf.value, color="#ffffff" if mesh is not None else "#e76f51",
                   linestyle="--", linewidth=1.8, zorder=2,
                   label=f"MUF {muf.value:g} MHz"
                         + (f" ({muf.letter})" if muf.letter else ""))

    # The other end of the propagation window. Drawn in the same family as the
    # MUF line because it is the same kind of quantity -- an edge of the band
    # this circuit can use -- and the two together are the window.
    lof = record.characteristic("LOF", model="")
    if lof is not None:
        ax.axvline(lof.value, color="#00e5ff" if mesh is not None else "#2a9d8f",
                   linestyle="--", linewidth=1.8, zorder=2,
                   label=f"LOF {lof.value:g} MHz"
                         + (f" ({lof.letter})" if lof.letter else ""))

    # The fitted MUF is a second estimate of the same quantity and usually
    # lands beyond the last detected echo. Drawing it keeps the panel and the
    # picture from disagreeing about where the nose is.
    fitted = record.characteristic("MUFNoseFit")
    if fitted is not None:
        shade = "#ffffff" if mesh is not None else "#e76f51"
        ax.axvline(fitted.value, color=shade, linestyle="-", linewidth=1.0,
                   alpha=0.6, zorder=2, label=f"nose fit {fitted.value:g} MHz")
        if fitted.bound:
            ax.axvspan(fitted.value - fitted.bound, fitted.value + fitted.bound,
                       color=shade, alpha=0.15, linewidth=0, zorder=1)

    _sao_frame(ax, record, ion, drawn, full_band,
               [c.value for c in (muf, fitted, lof) if c is not None], ground)

    ax.set_xlabel("Frequency (MHz)", fontsize=12)
    ax.set_ylabel("Group range along the path (km)", fontsize=12)
    # No grid over a raster: gridlines are indistinguishable from the thin
    # bright columns the ionogram itself is made of.
    if mesh is None:
        ax.grid(alpha=0.3, linewidth=0.7)

    if mesh is not None:
        bar = fig.colorbar(mesh, ax=ax, pad=0.09)
        bar.set_label(COLOUR_LABEL, fontsize=9)

    if drawn or muf is not None or mesh is not None:
        # How far the echo travelled beyond the ground path -- the oblique
        # equivalent of reading a virtual height off a vertical ionogram. Only
        # once the y-axis means something: on default limits it reads -2588.
        if ground:
            excess = ax.secondary_yaxis(
                "right", functions=(lambda y: y - ground, lambda y: y + ground))
            excess.set_ylabel(
                f"Excess over the {ground:.0f} km ground path (km)", fontsize=10)

        # Above the axes: with only points and two guide lines, any in-frame
        # legend lands on data. Over a raster the keys are chosen to read on
        # jet, so the legend needs a dark patch of its own or the white ones
        # disappear against the page.
        if ax.get_legend_handles_labels()[0]:
            dark = mesh is not None
            ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01, 1.0, 0.12),
                      mode="expand", ncol=4, fontsize=8.5, borderaxespad=0.0,
                      frameon=dark,
                      **({"facecolor": "#2f3640", "edgecolor": "#2f3640",
                          "labelcolor": "#f5f6fa", "framealpha": 0.95}
                         if dark else {}))
    else:
        # A record whose estimator declined is a valid record. Blank axes read
        # as a broken renderer, so say which it is.
        ax.text(0.5, 0.5, "nothing scaled\n(the estimator found no trace)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=13, color="#8d99ae", family="monospace",
                linespacing=1.8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_track(
    track_frame,
    out_path: str | Path,
    raw_frame=None,
    method: str = "algo",
    dpi: int = DEFAULT_DPI,
    param: str = "muf",
) -> Path:
    """Plot a tracked curve with its uncertainty band.

    ``param`` names the column and every label. A LOF plot titled "Tracked
    MUF" is worse than no plot, because it is the kind of thing that gets put
    in a report.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    label = param.upper()
    plt, (fig, ax) = _figure((14, 6))

    times = pd.to_datetime(track_frame["datetime"])
    muf = track_frame[param].to_numpy(dtype=float)
    sigma = track_frame["sigma"].to_numpy(dtype=float)

    ax.fill_between(times, muf - 2 * sigma, muf + 2 * sigma,
                    color="#d1495b", alpha=0.18, linewidth=0, label="+/-2 sigma")
    ax.plot(times, muf, "-", linewidth=2.0, color="#d1495b", label="tracked")

    if raw_frame is not None and f"{param}_{method}" in raw_frame:
        ax.plot(pd.to_datetime(raw_frame["datetime"]), raw_frame[f"{param}_{method}"],
                ".", markersize=4, alpha=0.5, color="#3d5a80",
                label=f"{method} per sounding")

    if "rejected" in track_frame:
        rejected = track_frame["rejected"].to_numpy(dtype=bool)
        if rejected.any() and raw_frame is not None and f"{param}_{method}" in raw_frame:
            raw = pd.to_numeric(raw_frame[f"{param}_{method}"], errors="coerce")
            ax.plot(times[rejected], raw.to_numpy()[rejected], "x", markersize=7,
                    color="#333333", label="rejected")

    if "measured" in track_frame:
        filled = ~track_frame["measured"].to_numpy(dtype=bool)
        if filled.any():
            ax.plot(times[filled], muf[filled], "o", markersize=3.5,
                    markerfacecolor="none", color="#ee9b00", label="filled")

    ax.set_xlabel("Time (UTC)", fontsize=13)
    ax.set_ylabel(f"{label} (MHz)", fontsize=13)
    ax.set_title(f"Tracked {label}", fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_daily(
    daily_frame,
    out_path: str | Path,
    raw_frame=None,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Plot a day's MUF curve, optionally over the per-sounding values."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt, (fig, ax) = _figure((14, 6))

    if raw_frame is not None:
        for name in [c[4:] for c in raw_frame.columns
                     if isinstance(c, str) and c.startswith("muf_")]:
            ax.plot(
                raw_frame["datetime"], raw_frame[f"muf_{name}"],
                ".", markersize=4, alpha=0.55, label=f"{name} (per sounding)",
            )

    ax.plot(daily_frame["datetime"], daily_frame["muf"],
            "-", linewidth=1.0, alpha=0.8, color="#444444", label="interpolated")
    if "muf_smooth" in daily_frame:
        ax.plot(daily_frame["datetime"], daily_frame["muf_smooth"],
                "-", linewidth=2.2, color="#d1495b", label="smoothed")

    ax.set_xlabel("Time (UTC)", fontsize=13)
    ax.set_ylabel("MUF (MHz)", fontsize=13)
    ax.set_title("MUF over the day", fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
