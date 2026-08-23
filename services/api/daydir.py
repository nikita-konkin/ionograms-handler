"""One definition of "a day folder", for the two modules that must agree.

Its own module for the same reason `muf.paths` is one: discovery
(`services.api.archives`) and the census (`services.api.sources`) both have to
recognise a dated directory, and neither is the natural home for the other's
copy. Until this existed they each carried their own regex, and the copies had
already drifted apart in the part that matters -- see `newest`.

Nothing here touches the filesystem. Deciding whether a name is a date is
free; opening the directory behind it is what costs a round trip on the
archive this serves, and that decision belongs to the caller.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

#: A directory name that is a date: ``2026-08-10``, ``2026.02.04``, ``20260810``.
#:
#: Three spellings because three writers produce them: the receiver, the
#: archive server, and whoever renamed a folder by hand. All three turn up
#: under one root, which is what `newest` exists to survive.
DAY_RE = re.compile(r"^(\d{4})[-._]?(\d{2})[-._]?(\d{2})$")


def day_key(name: str) -> tuple[int, int, int] | None:
    """``(year, month, day)`` for a dated directory name, or ``None``.

    The tuple is the sort key: comparing it compares dates, which comparing
    the name does not.
    """
    match = DAY_RE.match(name)
    return (int(match.group(1)), int(match.group(2)),
            int(match.group(3))) if match else None


def is_day(path: Path) -> bool:
    """Whether this path's *name* is a date. Says nothing about its contents."""
    return day_key(path.name) is not None


def newest(paths: Iterable[Path]) -> list[Path]:
    """The dated paths, newest date first. Undated names are dropped.

    **Not a reverse lexical sort**, which is wrong twice over on a real
    archive and was live here both ways:

    * Two spellings coexist. ``.`` is 0x2E and ``-`` is 0x2D, so *every*
      ``2026.02.*`` sorts above *every* ``2026-08-*``. The census reported
      February as "what is on air today" and never opened an August day; and
      discovery, probing only its first `archives.PROBE_DAYS`, spent the whole
      budget on the oldest folders and called a live dataset empty.
    * Not every subdirectory is a day. ``ionozond_data2`` beats any digit and
      took a slot outright.

    Sorting on `day_key` answers both: a name that is not a date has no key
    and is not ranked at all.
    """
    dated = [(key, path) for path in paths
             for key in (day_key(path.name),) if key is not None]
    dated.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in dated]
