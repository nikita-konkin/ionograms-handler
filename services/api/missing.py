"""Soundings whose file is gone: counted, and removed when asked.

A row outlives its file. The scan indexes what is on disk and never forgets,
and that is right -- a measurement is the record, and `remove_archive` keeps
an unregistered folder's soundings for exactly that reason. But a file that
was *deleted* leaves a row that can no longer be drawn, re-scaled or
re-derived, whose MUF and LOF still feed Series, the forecasts and training,
frozen at whatever the pipeline did on the day it was read. On tesla that was
about 10,000 August soundings of the old ``ionozond_data2`` share, deleted
from disk on purpose (2026-10-16).

Removing them is the operator's call, so this only counts until asked, and
the delete removes exactly what the count found. If any of the files comes
back, the next scan of an enabled archive reads it again: nothing here is
destructive of anything that still exists.

Two things decide the shape:

**One listing per folder, not one stat per file.** On an SMB share every stat
is a round trip (``BACKLOG.md`` sec. 32). 10,000 soundings in seventy day
folders is seventy listings.

**"Gone" means the filesystem said so.** A share that is down *raises* --
EIO, EHOSTDOWN, ESTALE -- and that aborts the count instead of reporting
every file missing. Counting it as missing would turn a NAS reboot into the
deletion of an archive's record. What cannot be told apart from deletion is a
share that was never mounted, whose mountpoint is an empty folder: so the
count also says whether the archive's own folder is absent, empty or present,
and the page puts that next to the button.
"""

from __future__ import annotations

import os
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from . import db

#: Rows per DELETE. Well under SQLite's bound-parameter limit (999 on the
#: oldest builds still about).
CHUNK = 500

#: Paths shown with a count, so "these are the August files" can be checked
#: by eye before anything goes.
EXAMPLES = 3


@dataclass
class Missing:
    ids: list[int] = field(default_factory=list)
    of: int = 0                 # soundings the archive holds
    folder: str = "present"     # the archive's own folder: present | empty | absent
    first: str | None = None    # earliest datetime among the missing
    last: str | None = None
    examples: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.ids)

    def summary(self) -> dict:
        return {"total": self.total, "of": self.of, "folder": self.folder,
                "first": self.first, "last": self.last,
                "examples": self.examples}


def _full(stored: str, archive_root: str | os.PathLike) -> Path:
    """Where a stored path points: the rule `read_routes._sounding_path` uses."""
    path = Path(stored)
    return path if path.is_absolute() else Path(archive_root) / path


def _names(folder: Path) -> set[str] | None:
    """What one folder holds, or None if the folder itself is not there.

    Any other OSError -- the share not answering -- propagates on purpose.
    """
    try:
        with os.scandir(folder) as entries:
            return {entry.name for entry in entries}
    except (FileNotFoundError, NotADirectoryError):
        return None


def _folder_state(folder: Path) -> str:
    try:
        with os.scandir(folder) as entries:
            return "present" if next(entries, None) is not None else "empty"
    except (FileNotFoundError, NotADirectoryError):
        return "absent"


def find(conn: sqlite3.Connection, archive: dict,
         archive_root: str | os.PathLike) -> Missing:
    """The soundings of ``archive`` whose file no longer exists.

    Its soundings are those under its ``relpath`` -- the ``LIKE 'relpath/%'``
    rule `db.archives` counts with, so the denominator here is the number on
    the page. Raises OSError if the storage cannot be read.
    """
    prefix = archive["relpath"].rstrip("/") + "/"
    held = db.rows(conn, "SELECT id, path, datetime FROM sounding"
                         " WHERE path LIKE ? ORDER BY datetime, id",
                   (prefix + "%",))
    by_folder: dict[Path, list[tuple[dict, str]]] = defaultdict(list)
    for row in held:
        full = _full(row["path"], archive_root)
        by_folder[full.parent].append((row, full.name))

    gone = []
    for folder, items in by_folder.items():
        names = _names(folder)
        gone.extend(row for row, name in items
                    if names is None or name not in names)
    gone.sort(key=lambda row: (row["datetime"], row["id"]))

    return Missing(
        ids=[int(row["id"]) for row in gone], of=len(held),
        folder=_folder_state(_full(archive["relpath"], archive_root)),
        first=gone[0]["datetime"] if gone else None,
        last=gone[-1]["datetime"] if gone else None,
        examples=[row["path"] for row in gone[:EXAMPLES]])


def delete(conn: sqlite3.Connection, ids: list[int]) -> int:
    """Delete these soundings. Their `extraction` and `reference` rows cascade.

    Only while foreign keys are enforced on this connection, hence the PRAGMA
    rather than trusting the caller's -- the care `db.delete_archive_orphans`
    takes, for the same reason: an extraction whose sounding is gone is a
    measurement nothing can locate.
    """
    conn.execute("PRAGMA foreign_keys = ON")
    removed = 0
    for start in range(0, len(ids), CHUNK):
        chunk = ids[start:start + CHUNK]
        marks = ",".join("?" * len(chunk))
        removed += conn.execute(f"DELETE FROM sounding WHERE id IN ({marks})",
                                chunk).rowcount
    conn.commit()
    return removed
