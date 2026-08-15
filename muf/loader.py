"""Format dispatch: ``.lfs`` recordings and chirpsounder2 ``.h5`` products.

``architecture.md`` sec. 3.2. One entry point, so ``run``, ``plot``, ``export``
and ``lof`` stop being ``.lfs``-only without any of them learning what a format
is. Everything downstream of :class:`~muf.spectro.Ionogram` was already
format-blind -- the extractors, ``pick``, ``render``, ``export`` all take the
gated array and its axes -- so this module is the whole of the change.

Selection is by file extension, with an override for the case the extension
lies. The two paths differ in more than their reader:

**``window`` and ``zero_periods`` do not exist for ``.h5``.** v2 fixed the FFT
window when it wrote the product and the sounding cannot be re-derived at
another one (sec. 3.4) -- the raw IQ is gone. Passing them warns rather than
silently accepting a number that will not be honoured, because a run whose
``--window`` was quietly ignored produces a table that looks like every other
table.

**``.h5`` is not cached.** The cache exists to skip FFTs, and there are none
here: ``io_chirp.load`` is 26 ms against seconds for the ``.lfs`` path, so an
entry would cost 8 MB on disk to save 26 ms. The deeper reason is that
``spectro.load_cached`` rebuilds the axes with ``calibrate.build``, from header
arithmetic -- and ``io_chirp`` deliberately builds them from the file's own
``freqs`` and ``ranges`` datasets instead (sec. 3.3). Restoring a v2 product
through that path would reconstruct a *different* axis and give no sign of it.
:func:`cache_key` still carries a format tag so the two never collide.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Mapping

from . import io_chirp, spectro, stations as _stations
from . import io_digisonde
from .io_lfs import find_lfs
from .io_lfs import read_header as _read_lfs_header
from .paths import dedupe_paths
from .spectro import DEFAULT_WINDOW, DEFAULT_ZERO_PERIODS, Ionogram

#: ``.lfs`` IQ recordings -- this pipeline forms the ionogram itself.
LFS = "lfs"
#: chirpsounder2 ``lfm_ionogram-*.h5`` -- the ionogram arrives already formed.
CHIRP2 = "chirp2"
#: chirpsounder2 ``digisonde_ionogram-*.h5`` -- another station's *vertical*
#: sounder, received obliquely here. Also already formed, and also ``.h5``,
#: which is why extension alone no longer decides. See :func:`format_of`.
DIGISONDE = "digisonde"

FORMATS = (LFS, CHIRP2, DIGISONDE)

#: Extension to format, for the cases an extension settles. ``.h5`` is not one
#: of them: chirpsounder2 writes chirp ionograms, digisonde ionograms and three
#: kinds of detection file into one tree with that suffix, so the ``.h5`` entry
#: is only the fallback :func:`format_of` reaches after the name says nothing.
SUFFIXES = {".lfs": LFS, ".h5": CHIRP2}


class FormatError(ValueError):
    """Raised when a path has no reader, or the requested one does not fit."""


def format_of(path: str | Path, override: str | None = None) -> str:
    """Which reader handles ``path``.

    ``override`` short-circuits the extension, for a recording that was
    renamed or arrived without one. It is checked against :data:`FORMATS`
    rather than trusted, so a typo fails here instead of inside a reader.
    """
    if override is not None:
        if override not in FORMATS:
            raise FormatError(
                f"unknown format {override!r}; choose from {', '.join(FORMATS)}")
        return override
    name = Path(path).name
    suffix = Path(path).suffix.lower()
    # Name before extension, because both product kinds are `.h5`. The prefix
    # is the writer's own (`receive_digisonde.py` builds it), so this is
    # reading the file's identity rather than guessing at it -- and a
    # mis-dispatch is not subtle: the schemas share no dataset but `t0`.
    if suffix == ".h5" and name.startswith(io_digisonde.FILE_PREFIX):
        return DIGISONDE
    fmt = SUFFIXES.get(suffix)
    if fmt is None:
        raise FormatError(
            f"{path}: no reader for {suffix or 'a file with no extension'}; "
            f"expected one of {', '.join(sorted(SUFFIXES))}, or pass an "
            f"explicit format"
        )
    return fmt


def find_soundings(target, *, format: str | None = None) -> list[Path]:
    """Every sounding under ``target``, of either format, sorted by name.

    A tree may hold both -- during the parallel run of ``architecture.md``
    sec. 2.4 it holds both by design -- so both are returned unless ``format``
    narrows it. The ``.h5`` half goes through :func:`muf.io_chirp.find_h5`,
    which matches ``lfm_ionogram-*.h5`` specifically: the same directories
    carry ``chirp-*.h5``, ``par-*.h5`` and ``cdetections-*.h5`` detection
    files, and at DOB ``digisonde_ionogram-*.h5`` from a different instrument
    entirely.
    """
    if format is not None and format not in FORMATS:
        raise FormatError(
            f"unknown format {format!r}; choose from {', '.join(FORMATS)}")

    found: list[Path] = []
    for fmt, finder in ((LFS, find_lfs), (CHIRP2, io_chirp.find_h5),
                        (DIGISONDE, io_digisonde.find_digisonde)):
        if format is not None and fmt != format:
            continue
        try:
            found.extend(finder(target))
        except FileNotFoundError:
            # One format finding nothing is normal in a single-format tree;
            # only both finding nothing is an error, handled below.
            continue

    if not found:
        wanted = format or " or ".join(FORMATS)
        raise FileNotFoundError(f"no {wanted} soundings under {target}")

    return dedupe_paths(found)


def resolve_stations(stations) -> Mapping[str, tuple[float, float]] | None:
    """Registry policy: ``None`` means the built-in table, not "no table".

    ``io_chirp`` stays deliberately dumb -- there, ``None`` means no registry
    and the geometry is unavailable. Choosing a default is policy, so it lives
    here, one level up. Pass an empty mapping to genuinely disable lookup,
    which is what a test wants when it is checking the no-geometry path.
    """
    if stations is None:
        return _stations.default_registry()
    return stations


def read_header(path: str | Path, *, format: str | None = None,
                stations: Mapping[str, tuple[float, float]] | None = None):
    """Header of one sounding: :class:`LfsHeader` or :class:`ChirpHeader`.

    The two are deliberately not a common base class. They share the names
    every consumer uses -- ``tx_name``, ``datetime``, ``path_type``,
    ``div_coef``, ``rmin``/``rmax`` -- and differ in what only one of them can
    answer, such as ``ChirpHeader.range_is_relative``. A caller that needs the
    difference should ask for it, not have it flattened away.
    """
    fmt = format_of(path, format)
    if fmt == CHIRP2:
        return io_chirp.read_header(path, resolve_stations(stations))
    if fmt == DIGISONDE:
        return io_digisonde.read_header(path, resolve_stations(stations))
    return _read_lfs_header(path, resolve_stations(stations))


def cache_key(path: Path, window: int, zero_periods: int,
              gate_km: tuple[float, float] | None,
              format: str | None = None) -> str:
    """Cache key with a format tag, per ``architecture.md`` sec. 3.3.

    ``.lfs`` keys keep their historical shape so existing caches stay valid.
    ``.h5`` gets its own so a ``recording.lfs`` and a ``recording.h5`` under
    one cache directory cannot overwrite each other -- and ``w``/``z`` are
    omitted from it, because for a v2 product they describe nothing.
    """
    fmt = format_of(path, format)
    if fmt == LFS:
        return spectro.cache_key(path, window, zero_periods, gate_km)
    gate = "auto" if gate_km is None else f"{gate_km[0]:.0f}-{gate_km[1]:.0f}"
    return f"{path.stem}_{fmt}_g{gate}"


def load(path: str | Path,
         window: int = DEFAULT_WINDOW,
         zero_periods: int = DEFAULT_ZERO_PERIODS,
         gate_km: tuple[float, float] | None = None,
         cache_dir: str | Path | None = None,
         *,
         format: str | None = None,
         header=None,
         stations: Mapping[str, tuple[float, float]] | None = None,
         nominal_stop_mhz: float | None = None) -> Ionogram:
    """Read one sounding of either format into an :class:`Ionogram`.

    Positional arguments match :func:`muf.spectro.compute_cached` so this is a
    drop-in replacement for it.

    ``window`` and ``zero_periods`` warn for ``.h5``, and only when they differ
    from the module defaults -- a caller who typed ``--window 4096`` needs to
    know it will be ignored, and one who typed nothing does not need to be
    told about a flag they never used.
    """
    path = Path(path)
    fmt = format_of(path, format)

    if fmt == LFS:
        # The registry reaches .lfs too. Its header carries coordinates, so
        # this is a correction rather than a lookup -- one site cannot have
        # two positions depending on which receiver logged it, and `cyprus1`
        # in the archive is `NIC` in the table, 59.9 km apart.
        return spectro.compute_cached(path, window, zero_periods, gate_km,
                                      cache_dir,
                                      stations=resolve_stations(stations))

    if fmt == DIGISONDE:
        # No `window`/`zero_periods` warning: a digisonde product is pulse
        # compressed rather than transformed, so those flags describe nothing
        # here even by analogy -- there is no window that could have been used
        # instead, and warning about one would invent a knob.
        return io_digisonde.load(path, gate_km,
                                 stations=resolve_stations(stations),
                                 header=header)

    header = header or io_chirp.read_header(path, resolve_stations(stations))

    # A product carrying `z` really can be re-derived, so honour --window
    # instead of warning about it. This is the one case where the sec. 3.4
    # limitation does not apply.
    if window != DEFAULT_WINDOW and header.has_raw_voltage:
        if zero_periods != DEFAULT_ZERO_PERIODS:
            warnings.warn(
                f"{path.name}: zero_periods={zero_periods} ignored -- v2 does "
                f"not zero-pad, so its stored bin spacing is the true "
                f"resolution and there is nothing to divide out.",
                stacklevel=2,
            )
        return io_chirp.reprocess(path, window, gate_km,
                                  stations=resolve_stations(stations),
                                  nominal_stop_mhz=nominal_stop_mhz)

    ignored = []
    if window != DEFAULT_WINDOW:
        ignored.append(f"window={window}")
    if zero_periods != DEFAULT_ZERO_PERIODS:
        ignored.append(f"zero_periods={zero_periods}")
    if ignored:
        detail = ("this product carries no `z` dataset, so the waveform was "
                  "discarded at acquisition"
                  if not header.has_raw_voltage else
                  "v2 does not zero-pad")
        warnings.warn(
            f"{path.name}: {' and '.join(ignored)} ignored -- {detail} "
            f"(architecture.md sec. 3.4). Set `save_raw_voltage = true` in "
            f"v2's [lfm] section to make future soundings re-derivable.",
            stacklevel=2,
        )
    return io_chirp.load(path, gate_km, stations=resolve_stations(stations),
                         nominal_stop_mhz=nominal_stop_mhz, header=header)


def describe_formats(paths) -> dict[str, int]:
    """Count soundings by format. For a CLI banner, and for spotting a mixed tree."""
    counts = dict.fromkeys(FORMATS, 0)
    for path in paths:
        try:
            counts[format_of(path)] += 1
        except FormatError:
            continue
    return {fmt: n for fmt, n in counts.items() if n}
