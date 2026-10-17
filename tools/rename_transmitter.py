"""Rename an ionozond transmitter: in the schedule, in its products, in the database.

A transmitter's name is not a label here either. The sounder writes the group
name from ``schedule.ini`` into every product, ingest stores it as
``sounding.tx``, and ``muf/stations.py`` resolves it to the coordinates that
give ``path_km`` and so every foF2. A name the registry does not know leaves
its soundings with no position at all.

This exists for one correction, decided on 2026-10-17 (issue #8). The Cyprus
groups on the Yoshkar-Ola laptop were named after slots they do not record.
ionozond's ``chirpt`` is the second the sweep reaches the band's low edge,
``cf`` minus half the sample rate, 75 s after the 0 Hz second chirpsounder2's
verified slots are given in. So ``NIC4_280``, at chirpt 10, records the
verified slot 235, which is ``NIC1``, and ``NIC_900``, at chirpt 315, records
240, which is ``NIC2``. Neither old name resolves, and neither does
``Sodankylä``, which becomes ``SGO``. Its products carry the name intact, but
``muf/io_chirp.py`` decodes ``txname`` as ASCII, so the server stored it as
``Sodankyl`` and two replacement characters; the db step therefore identifies
a row by its file name, and uses ``tx`` only where the file name is no help.

**A rule names one timing.** ``OLD@PHASE=NEW`` renames only the products whose
start time falls PHASE seconds into the 300 s cycle: a group's ``chirpt`` was
retuned over the weeks, and ``NIC_900`` spent its first three weeks at another
slot that matches nothing verified. Those products keep their name, and the
survey counts them. ``OLD=NEW`` renames every start time. A non-ASCII name
also matches ``schedule.ini``'s Qt-escaped spelling of it (``%U00E4``) and the
database's ASCII-garbled one.

Three steps, each a dry run unless ``--apply`` is given:

    schedule  renames the groups in ``schedule.ini`` and removes the ones
              given with --remove. A rule's phase must match the group's own
              chirpt, or the rule does not describe that group. Then
              regenerate ``chirp_config.py``: Regenerate on /ui/live,
              ``python3 -m services.agent ionozond regenerate``, or, while
              the console is still installed, its parameters dialog.
    files     renames every product under the given folders, and rewrites the
              name inside it first: the capture header (.lfs), the sidecar
              header (.lfp), the archive's ``txname`` (.h5), and in an echo
              window its ``tx_name``, ``source_name`` and its copy of the
              capture header. The pruner finds an archive through the capture
              header, so all of them move together. A product modified in the
              last ``--settle`` seconds may still be being written and is left.
              Rerunning is safe: a product rewritten but not yet renamed is
              finished, and one whose new name is taken is left and reported.
    db        renames ``sounding`` rows, their ``file`` and ``path`` with them,
              gives them the registry's coordinates, ``path_km`` and the hop
              count that follows, and renames the live preview. A row ingested
              under the new name meanwhile, from a file already renamed, is
              dropped, keeping the old row and its extractions. The database
              is backed up first. ``config_epoch`` keeps the names the sounder
              read at the time; that is its job.

The order on 2026-10-17, with the NAS timer stopped on the laptop so nothing
is copied half renamed (``sudo systemctl stop ionozond-nas.timer``, and wait
for ``ionozond-nas.service`` to finish):

    RULES="--rule NIC4_280@10=NIC1 --rule NIC_900@15=NIC2 --rule Sodankylä=SGO"
    python3 tools/rename_transmitter.py schedule ~/.config/dsChirp/schedule.ini \\
        $RULES --remove NIC1_235
    python3 tools/rename_transmitter.py files ~/ionozond_data \\
        /mnt/ionozond_5tb/ionozond_data $RULES
    docker compose exec api python -m tools.rename_transmitter db \\
        --db /data/ionograms.sqlite3 $RULES

The first two run on the laptop, the third where the server's database is.
Each prints its plan; add ``--apply`` once it reads right. Then start the NAS
timer again.
"""

from __future__ import annotations

import argparse
import calendar
import collections
import os
import re
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: The verified slots' cycle, in which a rule's phase is counted.
CYCLE = 300
#: ionozond's own rule for a group name, which becomes every product's name.
CODE_RE = re.compile(r"^[A-Za-z0-9_.]{1,24}$")
ARCHIVE = re.compile(r"^lfm_ionogram-(?P<tx>[^-]+)-"
                     r"(?P<rest>.+-(?P<t0>\d+(?:\.\d+)?)\.h5)$")
CAPTURE = re.compile(r"^(?P<tx>.+)_(?P<day>\d{8})_(?P<time>\d{6})"
                     r"(?P<ext>\.lfs|\.lfp|\.echo\.h5)$")
LFS_MAGIC, LFS_NAME = b"LFSG", slice(14, 78)
LFP_MAGIC, LFP_NAME = b"LFPR", slice(0x30, 0x70)


@dataclass(frozen=True)
class Rule:
    """One old name, the start-time phase it applies to, and its new name."""

    old: str
    new: str
    phase: int | None

    @property
    def spellings(self) -> set:
        return spellings(self.old)

    def matches(self, tx: str, t0: float | None) -> bool:
        return tx in self.spellings and (
            self.phase is None or (t0 is not None and round(t0) % CYCLE == self.phase))


def rule(text: str) -> Rule:
    """``OLD=NEW`` or ``OLD@PHASE=NEW``, or why it is not one."""
    found = re.match(r"^(?P<old>[^=@]+)(?:@(?P<phase>[^=]*))?=(?P<new>.*)$", text)
    if not found:
        raise SystemExit("%r is not a rule: write OLD=NEW or OLD@PHASE=NEW" % text)
    phase = found["phase"]
    if phase is not None and not (phase.isdigit() and int(phase) < CYCLE):
        raise SystemExit("%r: the phase must be whole seconds from 0 to %d"
                         % (text, CYCLE - 1))
    if not CODE_RE.match(found["new"]):
        raise SystemExit("%r: the new name must be Latin letters, digits, underscores "
                         "or dots, up to 24, and no dash" % text)
    return Rule(found["old"], found["new"], None if phase is None else int(phase))


def spellings(name: str) -> set:
    """A name, and the one ``muf/io_chirp.py`` stores when it reads it as ASCII."""
    return {name, name.encode("utf-8").decode("ascii", "replace")}


def raw(name: str) -> set:
    """Every way ``schedule.ini`` may hold a group name, as bytes: Qt 5 escapes a
    non-ASCII character as ``%UXXXX``, and a hand edit leaves UTF-8 or Latin-1."""
    out = {name.encode("utf-8"), "".join(c if ord(c) < 0x80 else "%%U%04X" % ord(c)
                                         for c in name).encode("ascii")}
    try:
        out.add(name.encode("latin-1"))
    except UnicodeEncodeError:
        pass
    return out


# --------------------------------------------------------------------------
# schedule.ini
# --------------------------------------------------------------------------

def schedule(data: bytes, rules: list, removed: list) -> tuple:
    """``schedule.ini`` with groups renamed and removed, and what was done."""
    said = []
    for item in rules:
        start, end = _group(data, item.old)
        if any(_header(data, name) is not None for name in [item.new]):
            raise SystemExit("%s cannot become %s: schedule.ini already has a group "
                             "%s" % (item.old, item.new, item.new))
        chirpt = re.search(rb"^\s*chirpt\s*=\s*([^\r\n]*)", data[start:end], re.M)
        if item.phase is not None and not (
                chirpt and _number(chirpt.group(1)) is not None
                and round(_number(chirpt.group(1))) % CYCLE == item.phase):
            raise SystemExit(
                "%s has chirpt %s, not phase %d of the %d s cycle: the rule does not "
                "describe this group" % (item.old, chirpt.group(1).decode(
                    "latin-1") if chirpt else "unset", item.phase, CYCLE))
        line = data.index(b"]", start) + 1
        data = data[:start] + b"[" + item.new.encode("ascii") + b"]" + data[line:]
        said.append("%s -> %s" % (item.old, item.new))
    for name in removed:
        start, end = _group(data, name)
        data = data[:start] + data[end:]
        said.append("removed %s" % name)
    return data, said


def _group(data: bytes, name: str) -> tuple:
    start = _header(data, name)
    if start is None:
        raise SystemExit("schedule.ini has no group %s" % name)
    line = data.find(b"\n", start)
    after = re.compile(rb"^[ \t]*\[", re.M).search(data, len(data) if line < 0
                                                   else line + 1)
    return start, after.start() if after else len(data)


def _header(data: bytes, name: str):
    names = raw(name)
    for found in re.finditer(rb"^[ \t]*\[([^\]\r\n]*)\][ \t]*$", data, re.M):
        if found.group(1).strip() in names:
            return found.start()
    return None


def _number(text: bytes):
    try:
        return float(text.strip().strip(b'"'))
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Products on disk
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Move:
    """One product to rename, and whether its contents still need the new name."""

    path: Path
    target: Path
    rule: Rule
    rewrite: bool


@dataclass
class Survey:
    """Every product a run would move, keep, or leave for a reason."""

    moves: list = field(default_factory=list)
    kept: collections.Counter = field(default_factory=collections.Counter)
    left: list = field(default_factory=list)


def files(roots: list, rules: list, *, apply: bool, settle: float) -> Survey:
    """Find, and with ``apply`` move, every product a rule names."""
    found, now = Survey(), time.time()
    for root in roots:
        for folder, _, names in sorted(os.walk(str(root))):
            for name in sorted(names):
                _surveyed(found, Path(folder) / name, rules, now - settle)
    if apply:
        for move in found.moves:
            _moved(move)
    return found


def _surveyed(found: Survey, path: Path, rules: list, before: float) -> None:
    named = ARCHIVE.match(path.name) or CAPTURE.match(path.name)
    if not named:
        return
    t0 = (float(named["t0"]) if "t0" in named.groupdict() else calendar.timegm(
        time.strptime(named["day"] + named["time"], "%Y%m%d%H%M%S")))
    item = next((r for r in rules if r.matches(named["tx"], t0)), None)
    if item is None:
        if any(named["tx"] in r.spellings for r in rules):
            found.kept[(named["tx"], round(t0) % CYCLE)] += 1
        return
    target = path.with_name(_renamed(path.name, item.new))
    inside = _names(path)
    strange = [n for n in inside if n not in item.spellings and n != item.new]
    if path.stat().st_mtime > before:
        found.left.append((path, "modified too recently; it may still be written"))
    elif target.exists():
        found.left.append((path, "%s is already there" % target.name))
    elif strange:
        found.left.append((path, "it names %s inside"
                           % ", ".join(sorted(set(strange)))))
    else:
        found.moves.append(Move(path, target, item,
                                any(n in item.spellings for n in inside)))


def _renamed(name: str, new: str) -> str:
    named = ARCHIVE.match(name)
    if named:
        return "lfm_ionogram-%s-%s" % (new, named["rest"])
    named = CAPTURE.match(name)
    return "%s_%s_%s%s" % (new, named["day"], named["time"], named["ext"])


def _names(path: Path) -> list:
    """Every transmitter name a product carries inside."""
    if path.suffix == ".lfs":
        return [_field(path, LFS_MAGIC, LFS_NAME).decode("latin-1")]
    if path.suffix == ".lfp":
        return [_field(path, LFP_MAGIC, LFP_NAME).decode("utf-8", "replace")]
    import h5py

    out = []
    with h5py.File(str(path), "r") as fh:
        if "txname" in fh:
            out.append(_text(fh["txname"][()]))
        for node in _nodes(fh):
            if "tx_name" in node.attrs:
                out.append(_text(node.attrs["tx_name"]))
            source = CAPTURE.match(_text(node.attrs.get("source_name", "")))
            if source:
                out.append(source["tx"])
        if "echo_window/lfs_header" in fh:
            out.append(bytes(fh["echo_window/lfs_header"][()])[LFS_NAME].split(
                b"\0")[0].decode("latin-1"))
    return out


def _field(path: Path, magic: bytes, where: slice) -> bytes:
    with open(str(path), "rb") as fh:
        head = fh.read(where.stop)
    if head[:4] != magic:
        raise SystemExit("%s does not start with %r" % (path, magic))
    return head[where].split(b"\0")[0]


def _moved(move: Move) -> None:
    if move.rewrite:
        _rewritten(move.path, move.rule)
    os.rename(str(move.path), str(move.target))


def _rewritten(path: Path, item: Rule) -> None:
    if path.suffix in (".lfs", ".lfp"):
        where, code = ((LFS_NAME, "latin-1") if path.suffix == ".lfs"
                       else (LFP_NAME, "utf-8"))
        with open(str(path), "r+b") as fh:
            fh.seek(where.start)
            fh.write(item.new.encode(code).ljust(where.stop - where.start, b"\0"))
        return
    import h5py
    import numpy as np

    tmp = path.with_name(".%s.rename.tmp" % path.name)
    shutil.copy2(str(path), str(tmp))
    try:
        with h5py.File(str(tmp), "r+") as fh:
            if "txname" in fh and _text(fh["txname"][()]) in item.spellings:
                del fh["txname"]
                fh["txname"] = item.new
            for node in _nodes(fh):
                if _text(node.attrs.get("tx_name", "")) in item.spellings:
                    node.attrs["tx_name"] = item.new
                source = CAPTURE.match(_text(node.attrs.get("source_name", "")))
                if source and source["tx"] in item.spellings:
                    node.attrs["source_name"] = _renamed(source.group(0), item.new)
            if "echo_window/lfs_header" in fh:
                header = bytearray(bytes(fh["echo_window/lfs_header"][()]))
                if header[LFS_NAME].split(b"\0")[0].decode("latin-1") in item.spellings:
                    header[LFS_NAME] = item.new.encode("latin-1").ljust(64, b"\0")
                    fh["echo_window/lfs_header"][...] = np.frombuffer(
                        bytes(header), dtype=np.uint8)
        os.replace(str(tmp), str(path))
    finally:
        if tmp.exists():
            tmp.unlink()


def _nodes(fh) -> list:
    return [fh] + ([fh["echo_window"]] if "echo_window" in fh else [])


def _text(value) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


# --------------------------------------------------------------------------
# The server's database
# --------------------------------------------------------------------------

@dataclass
class Plan:
    """Every row a rename would change, computed before anything is written."""

    rx: str
    renames: list = field(default_factory=list)
    duplicates: list = field(default_factory=list)
    previews: list = field(default_factory=list)
    kept: collections.Counter = field(default_factory=collections.Counter)
    elsewhere: dict = field(default_factory=dict)
    coords: dict = field(default_factory=dict)


def survey(conn: sqlite3.Connection, rules: list, rx: str) -> Plan:
    """The rows each rule renames, and those it leaves. Reads only."""
    from muf.stations import default_registry

    plan = Plan(rx)
    for item in rules:
        station = default_registry().station(item.new)
        if station is None:
            raise SystemExit(
                "%s is not in muf/stations.py, so its rows would get no coordinates "
                "and no path length, which is what this rename repairs. Add it there "
                "first, or choose a name it knows" % item.new)
        plan.coords[item.new] = station.coordinates
        names = sorted(item.spellings)
        marks = ",".join("?" * len(names))
        for row in conn.execute(
                "SELECT id, file, path, tx, datetime FROM sounding WHERE rx = ? AND "
                "tx IN (%s) ORDER BY id" % marks, (rx, *names)):
            filed = ARCHIVE.match(row["file"] or "") or CAPTURE.match(row["file"] or "")
            tx = filed["tx"] if filed else row["tx"]
            if tx not in item.spellings:
                continue
            t0 = _start(row["file"], row["datetime"])
            if not item.matches(tx, t0):
                plan.kept[(tx, None if t0 is None else round(t0) % CYCLE)] += 1
                continue
            file = _renamed(row["file"], item.new) if filed else row["file"]
            path = (row["path"].rsplit("/", 1)[0] + "/" + file
                    if "/" in row["path"] else file)
            plan.renames.append((row["id"], item, file, path))
            plan.duplicates.extend(r[0] for r in conn.execute(
                "SELECT id FROM sounding WHERE file = ? AND id != ?",
                (file, row["id"])))
        plan.previews.extend((r[0], item) for r in conn.execute(
            "SELECT tx FROM station_preview WHERE station = ? AND tx IN (%s)" % marks,
            (rx, *names)))
        for table in ("forecast", "score", "model_registry"):
            count = conn.execute("SELECT COUNT(*) FROM %s WHERE rx = ? AND tx IN (%s)"
                                 % (table, marks), (rx, *names)).fetchone()[0]
            if count:
                plan.elsewhere[table] = plan.elsewhere.get(table, 0) + count
    return plan


def commit(conn: sqlite3.Connection, plan: Plan) -> None:
    """Write the plan. One transaction: it lands whole or not at all."""
    from muf.geometry import Point, great_circle_km, hop_count

    with conn:
        for sid in sorted(set(plan.duplicates)):
            for table in ("extraction", "reference"):
                conn.execute("DELETE FROM %s WHERE sounding_id = ?" % table, (sid,))
            conn.execute("DELETE FROM sounding WHERE id = ?", (sid,))
        for sid, item, file, path in plan.renames:
            lat, lon = plan.coords[item.new]
            row = conn.execute("SELECT rx_lat, rx_lon FROM sounding WHERE id = ?",
                               (sid,)).fetchone()
            km = (great_circle_km(Point(lat, lon), Point(row[0], row[1]))
                  if row[0] is not None and row[1] is not None else None)
            conn.execute("UPDATE sounding SET tx = ?, file = ?, path = ?, tx_lat = ?, "
                         "tx_lon = ?, path_km = ? WHERE id = ?",
                         (item.new, file, path, lat, lon, km, sid))
            if km is not None:
                conn.execute("UPDATE extraction SET hops = ? WHERE sounding_id = ? "
                             "AND (hops IS NULL OR hops != ?)",
                             (hop_count(km), sid, hop_count(km)))
        for tx, item in plan.previews:
            if conn.execute("SELECT 1 FROM station_preview WHERE station = ? "
                            "AND tx = ?", (plan.rx, item.new)).fetchone():
                conn.execute("DELETE FROM station_preview WHERE station = ? AND tx = ?",
                             (plan.rx, tx))
            else:
                conn.execute("UPDATE station_preview SET tx = ? WHERE station = ? AND "
                             "tx = ?", (item.new, plan.rx, tx))


def _start(file: str, when: str):
    named = ARCHIVE.match(file or "")
    if named:
        return float(named["t0"])
    named = CAPTURE.match(file or "")
    stamp = ((named["day"] + named["time"]) if named
             else re.sub(r"\D", "", when or "")[:14])
    try:
        return calendar.timegm(time.strptime(stamp, "%Y%m%d%H%M%S"))
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = parser.add_subparsers(dest="step", required=True)
    for name, text in (("schedule", "rename and remove groups in schedule.ini"),
                       ("files", "rename the products under folders"),
                       ("db", "rename the rows in the server's database")):
        step = steps.add_parser(name, help=text)
        step.add_argument("--rule", dest="rules", action="append", type=rule,
                          default=[], help="OLD=NEW or OLD@PHASE=NEW; repeat")
        step.add_argument("--apply", action="store_true",
                          help="write the changes (default: report and exit)")
    steps.choices["schedule"].add_argument("ini", type=Path)
    steps.choices["schedule"].add_argument("--remove", action="append", default=[],
                                           help="a group to delete; repeat")
    steps.choices["files"].add_argument("roots", nargs="+", type=Path)
    steps.choices["files"].add_argument("--settle", type=float, default=900.0,
                                        help="leave products modified this recently")
    steps.choices["db"].add_argument("--db", type=Path, required=True)
    steps.choices["db"].add_argument("--rx", default="yoshkar-ola",
                                     help="the receiving station (default yoshkar-ola)")
    args = parser.parse_args(argv)
    return {"schedule": _schedule, "files": _files, "db": _db}[args.step](args)


def _schedule(args) -> int:
    before = args.ini.read_bytes()
    after, said = schedule(before, args.rules, args.remove)
    for line in said:
        print(line)
    if not args.apply:
        print("Dry run. Nothing was written. Add --apply to write schedule.ini.")
        return 0
    lock = args.ini.with_name(args.ini.name + ".lock")
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise SystemExit("%s exists: the console or the agent is saving the settings. "
                         "Try again in a moment." % lock) from None
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write("%d\nrename_transmitter\n%s\n" % (os.getpid(), os.uname().nodename
                                                       if hasattr(os, "uname") else ""))
        if args.ini.read_bytes() != before:
            raise SystemExit("schedule.ini changed while this ran; run it again")
        copy = args.ini.with_name("%s.bak-%s" % (args.ini.name, time.strftime(
            "%Y%m%dT%H%M%SZ", time.gmtime())))
        shutil.copy2(str(args.ini), str(copy))
        tmp = args.ini.with_name(".%s.rename.tmp" % args.ini.name)
        tmp.write_bytes(after)
        os.replace(str(tmp), str(args.ini))
    finally:
        lock.unlink()
    print("Written; the old file is %s." % copy)
    print("Now regenerate chirp_config.py: Regenerate on /ui/live, `python3 -m "
          "services.agent ionozond regenerate`, or the console's parameters dialog.")
    return 0


def _files(args) -> int:
    found = files(args.roots, args.rules, apply=args.apply, settle=args.settle)
    by = collections.Counter((m.rule.old, m.rule.new, m.path.suffix)
                             for m in found.moves)
    for (old, new, suffix), count in sorted(by.items()):
        print("%-12s -> %-6s %6d %s" % (old, new, count, suffix))
    for (tx, phase), count in sorted(found.kept.items()):
        print("kept         %-12s phase %3d  %6d (no rule for this timing)"
              % (tx, phase, count))
    for path, why in found.left[:20]:
        print("left         %s: %s" % (path, why))
    if len(found.left) > 20:
        print("left         ... and %d more" % (len(found.left) - 20))
    print("%s %d product(s)." % ("Moved" if args.apply else "Dry run. Would move",
                                 len(found.moves)))
    return 0


def _db(args) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from relabel_station import backup

    from services.api import db

    if not args.db.exists():
        raise SystemExit("no database at %s" % args.db)
    conn = db.connect(args.db)
    try:
        plan = survey(conn, args.rules, args.rx)
        by = collections.Counter((r[1].old, r[1].new) for r in plan.renames)
        for (old, new), count in sorted(by.items()):
            print("sounding     %-12s -> %-6s %6d row(s)" % (old, new, count))
        for (tx, phase), count in sorted(plan.kept.items(), key=str):
            print("kept         %-12s phase %s  %6d (no rule for this timing)"
                  % (tx, phase, count))
        if plan.duplicates:
            print("dropped      %d row(s) already ingested under a new name"
                  % len(set(plan.duplicates)))
        for tx, item in plan.previews:
            print("preview      %s -> %s" % (tx, item.new))
        for table, count in sorted(plan.elsewhere.items()):
            print("NOTE         %s has %d row(s) under an old name; left alone"
                  % (table, count))
        if not args.apply:
            print("Dry run. Nothing was written. Add --apply to commit.")
            return 0
        print("Backup: %s" % backup(args.db))
        commit(conn, plan)
        print("Applied. %d row(s) renamed." % len(plan.renames))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
