"""Path bookkeeping shared by every finder.

Its own module because the four finders that need it -- ``io_lfs.find_lfs``,
``io_chirp.find_h5``, ``io_detect._find`` and ``loader.find_soundings`` --
already import each other in a chain (``io_chirp`` -> ``calibrate`` ->
``io_lfs``), so there is no existing module all four can import from.

Nothing here opens a file. That is the point: on a network archive the cheap
half of finding a recording is the listing, and the expensive half is every
per-file system call made about it afterwards.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def dedupe_paths(found: Iterable[Path]) -> list[Path]:
    """One entry per file, sorted by name.

    A directory and a file inside it can both be named on one command line, so
    the same recording arrives twice and would otherwise be loaded twice.
    Sorting by name sorts by time, since every product name embeds its
    timestamp.

    **Keyed on the absolute path, not on ``Path.resolve()``.** Those look
    interchangeable and are not: ``resolve`` calls ``realpath``, which is a
    system call per file -- a round trip per file on the network archive the
    api server reads. Deduping 1368 detection files cost 38 ms against 0.4 ms
    for ``abspath`` on a local checkout, and the census was paying it on every
    page load; the directory listing that found those files cost 3 ms. What is
    given up is that two *different* paths reaching one file through a symlink
    no longer collapse to a single entry, which no caller here relies on.
    """
    unique: dict[str, Path] = {}
    for path in found:
        unique.setdefault(os.path.abspath(path), path)
    return sorted(unique.values(), key=lambda p: (p.name, str(p)))
