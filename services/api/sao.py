"""SAO for the server: one scaling of a sounding, shared by three surfaces.

``muf export`` has produced SAO.XML 5.0 since before this module existed;
nothing in ``services/`` knew about it. That was the whole gap. The XML
download, the numbers panel and the interactive plot are three views of one
scaling, so they are built once here rather than three times in three routes
that could disagree about what the sounding says.

**The result is memoised on the file.** Detection products are write-once, so
a path plus its mtime identifies a scaling for good. Building one costs about
a second; a page that redraws on every gate toggle would pay that each time.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

#: Whether a scaling carries IRI's answer beside the measured one. Set
#: ``SAO_MODEL=0`` to turn it off.
#:
#: A knob rather than a constant because the model is the one part of this
#: that can reach the internet: PyIRI needs a solar driver, and an unwarmed
#: index cache means a fetch. The unit suite turns it off, which is also why
#: it is read here and not baked into a default argument.
MODEL = os.environ.get("SAO_MODEL", "1") not in ("0", "", "false")

#: Scalings kept in memory. Each is a few hundred trace points and some
#: strings -- kilobytes, not megabytes -- and the page is stepped through with
#: prev/next, so a handful of neighbours is the access pattern to serve.
CACHE_SIZE = 24

#: Estimators exported, in the order the panel lists them. All three, because
#: the spec's separate-storage rule (sec. 1.3.4) puts each in its own
#: ``<SAORecord>`` and the disagreement between them is information: three
#: methods on one trace is the closest thing this pipeline has to an error bar.
METHODS = ("algo", "kmeans", "contour")

_LOCK = threading.Lock()
_CACHE: OrderedDict[tuple, "Scaling"] = OrderedDict()


@dataclass(frozen=True)
class Scaling:
    """One sounding, scaled: the XML, the parsed records, and the plot frame."""

    #: ``<SAORecordList>``, ready to serialise.
    root: object
    #: Parsed back out of it, so the page draws exactly what the file says
    #: rather than a parallel construction of the same numbers.
    records: tuple = ()
    #: ``(f_lo, f_hi, r_lo, r_hi)`` in MHz and km -- the data rectangle the
    #: bare raster covers, so a browser can place it under the traces.
    extent: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    #: What IRI says for this path at this instant, or why it does not.
    model: dict = field(default_factory=dict)
    cost_s: float = 0.0

    def record(self, method: str):
        """The record for one estimator, or ``None``."""
        return next((r for r in self.records if r.method == method), None)


def raster_extent(ion) -> tuple[float, float, float, float]:
    """The data rectangle a bare ``render.plot`` PNG covers.

    Not the min and max of the axes: ``pcolormesh(shading="nearest")`` draws a
    cell *centred* on each sample, so the image runs half a cell beyond the
    first and last of them. Getting this wrong offsets every overlaid trace by
    half a bin, which on a 3999-cell range axis is invisible until you zoom in
    and the circles sit beside the echo instead of on it.
    """
    import numpy as np

    def bounds(values):
        values = np.asarray(values, dtype=float)
        if values.size == 1:
            return float(values[0] - 0.5), float(values[0] + 0.5)
        lo = values[0] - (values[1] - values[0]) / 2
        hi = values[-1] + (values[-1] - values[-2]) / 2
        return (float(min(lo, hi)), float(max(lo, hi)))

    f_lo, f_hi = bounds(ion.freq)
    r_lo, r_hi = bounds(ion.vrange)
    return f_lo, f_hi, r_lo, r_hi


def load_ion(path: Path, *, gate: str | None = "auto"):
    """Load and gate one sounding the way every surface here must.

    Shared so the PNG route and the plot frame cannot gate differently: the
    image is positioned by the extent this returns, and a mismatch would put
    the raster somewhere the axes do not agree with.
    """
    from muf import calibrate, loader

    ion = loader.load(Path(path))
    if gate == "auto":
        found = calibrate.auto_gate(ion.power, ion.cal.vrange)
        if found is not None:
            ion = ion.regated(*found)
    elif gate and gate not in ("full", "none"):
        # "full" is the word the page's own toggle sends for "do not gate".
        # Without it here the split below reads it as a range pair and raises
        # ValueError on a link the interface itself offers.
        lo, hi = (float(part) for part in gate.split(","))
        ion = ion.regated(lo, hi)
    return ion


def _iri_for(ion) -> dict:
    """IRI at the path's control point, or a stated reason why not.

    Returned as a dict with an ``error`` key rather than raised: a model that
    is not installed, or a host that cannot reach its solar driver, is a
    normal condition here and must degrade to a labelled absence on the page
    -- never to a blank space that reads like a measurement of zero.
    """
    from muf import geometry
    from muf.reference import iri

    if not iri.available():
        return {"error": f"not installed. {iri.INSTALL_HINT}"}

    tx, rx, path_km = geometry.path_of(ion.header)
    if tx is None or rx is None:
        return {"error": "no geometry for this circuit; IRI needs both ends"}

    import pandas as pd

    series = iri.predict(tx, rx, pd.to_datetime([ion.header.datetime]))
    if series.error:
        return {"error": series.error}

    def first(frame, column):
        try:
            return float(frame[column].iloc[0])
        except Exception:                                     # noqa: BLE001
            return None

    return {
        "name": series.name.upper(),
        "muf_mhz": float(series.muf.iloc[0]),
        "fof2_mhz": first(series.detail, "fof2"),
        "hmf2_km": first(series.detail, "hmf2"),
        "path_km": path_km,
        # Both ends' own control point, and two of them on a path that hops
        # twice: `geometry.describe_path` is the one phrasing, shared with
        # `muf export --iri` so the page and the download agree.
        "control": " / ".join(str(point)
                              for point in geometry.control_points(tx, rx)),
        "source": series.source,
        "options": geometry.describe_path(tx, rx),
    }


def build(path: Path, *, gate: str | None = "auto",
          methods: tuple[str, ...] = METHODS,
          iri: bool | None = None) -> Scaling:
    """Scale one sounding and return every view of it. Memoised on the file."""
    import time

    # Resolved before it goes in the key, so turning `MODEL` off does not
    # silently return a scaling built while it was on.
    iri = MODEL if iri is None else iri
    path = Path(path)
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        stamp = 0
    key = (str(path), stamp, gate, methods, iri)

    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            _CACHE.move_to_end(key)
            return hit

    started = time.perf_counter()
    got = _build(path, gate=gate, methods=methods, iri=iri)
    got = Scaling(root=got.root, records=got.records, extent=got.extent,
                  model=got.model, cost_s=time.perf_counter() - started)

    with _LOCK:
        _CACHE[key] = got
        while len(_CACHE) > CACHE_SIZE:
            _CACHE.popitem(last=False)
    return got


def _build(path: Path, *, gate, methods, iri) -> Scaling:
    from muf import interference
    from muf.export import saoxml
    from muf.pipeline import Options

    ion = load_ion(path, gate=gate)

    options = Options(methods=list(methods))
    ion, _ = interference.apply(ion, options)

    # IRI goes *into* the record as `<Modeled>`, not beside it: that is what
    # the element is for, it is what `muf export --iri` writes, and it means
    # the panel reads the modelled numbers out of the same document the
    # download serves rather than out of a parallel calculation that could
    # drift from it.
    model = _iri_for(ion) if iri else {}
    values = None
    if model and not model.get("error"):
        values = saoxml.ModelValues(
            name=model["name"], muf_mhz=model["muf_mhz"],
            fof2_mhz=_nan(model["fof2_mhz"]), hmf2_km=_nan(model["hmf2_km"]),
            options=model["options"],
        )

    elements = saoxml.records_for(ion, options, model=values)
    root = saoxml.build_document(elements)
    records = tuple(saoxml._parse_record(el) for el in elements)

    return Scaling(root=root, records=records, extent=raster_extent(ion),
                   model=model)


def _nan(value) -> float:
    """``None`` as NaN, which is how ``ModelValues`` spells "not available"."""
    return float("nan") if value is None else float(value)


def colour_axis() -> dict | None:
    """The raster's colour key: limits, label and the map itself.

    ``None`` when matplotlib cannot be asked -- the page then draws the plot
    without a bar rather than inventing one. Every other panel on this page
    survives its own source being unavailable and says so; a colour scale
    guessed from a default would instead be a confident wrong answer about
    what the picture means.
    """
    from muf import render

    try:
        stops = render.colour_scale()
    except Exception:
        return None
    return {
        "vmin": render.DEFAULT_VMIN_DB,
        "vmax": render.DEFAULT_VMAX_DB,
        "label": render.COLOUR_LABEL,
        "stops": stops,
    }


def plot_data(scaling: Scaling, method: str) -> dict:
    """One record as the browser needs it: traces, markers and the frame.

    Only the scaled points cross the wire. The raster stays a PNG: at 486 x
    3999 it is 1.94 million cells, about 11 MB as JSON, against 164 KB as an
    image the browser was already going to decode. Everything interactive --
    hover readout, per-trace toggling, zoom -- needs the *traces*, and there
    are about 165 of those points.
    """
    from muf import render

    record = scaling.record(method)
    f_lo, f_hi, r_lo, r_hi = scaling.extent
    out: dict = {
        "extent": {"f_lo": f_lo, "f_hi": f_hi, "r_lo": r_lo, "r_hi": r_hi},
        "traces": [], "marks": [], "relative": False,
        # The key to the raster. It travels with the frame rather than living
        # in the template because it has to be the map the PNG was actually
        # drawn with -- see `render.colour_scale`. Included even when there is
        # no record to scale: the raster is drawn either way, and a picture in
        # false colour with no key is exactly what this fixes.
        "scale": colour_axis(),
    }
    if record is None:
        return out

    out["relative"] = record.range_is_relative
    for index, item in enumerate(record.traces):
        # Branch when the hop identification labelled it, segment colours
        # otherwise -- the same fallback `render._overlay_trace` uses, so the
        # interactive plot and the rendered PNG colour the same trace alike.
        # On this circuit `Branch` comes back empty, and a legend of five
        # entries all reading "unlabelled" in one colour is not a legend.
        span = f"{item.freq.min():.1f}–{item.freq.max():.1f} MHz"
        if item.branch:
            colour = render.SAO_BRANCH_COLOURS_ON_RASTER.get(item.branch,
                                                             "#c77dff")
            # The span even when the branch is known: `contour` labels two of
            # nine traces on this circuit, and two legend entries both reading
            # "low" identify neither of them.
            name = f"{item.label} · {span}"
        else:
            colour = render._SEGMENT_COLOURS[index % len(render._SEGMENT_COLOURS)]
            name = f"segment {index + 1} · {span}"
        amplitude = (item.amplitude.tolist()
                     if item.amplitude is not None
                     and item.amplitude.size == item.freq.size else None)
        out["traces"].append({
            "name": name,
            "freq": item.freq.tolist(),
            "vrange": item.vrange.tolist(),
            "amplitude": amplitude,
            "colour": colour,
            "hops": item.hops,
            "points": item.n_points,
        })

    # MUF and LOF as verticals rather than as trace points: they are scalar
    # characteristics of the whole record, and drawing them as markers would
    # invite reading them off the range axis, which they have no value on.
    for name, colour in (("MUF", "#ffffff"), ("LOF", "#ffd166")):
        found = record.characteristic(name, model="")
        if found is not None:
            out["marks"].append({
                "name": name, "value": found.value, "colour": colour,
                "letter": found.letter, "text": found.text,
            })
    return out


def clear() -> None:
    """Drop the memo. For tests, which must not inherit each other's."""
    with _LOCK:
        _CACHE.clear()
