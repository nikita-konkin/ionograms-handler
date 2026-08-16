"""Rename a station across the database, and fix what the name determines.

A receiver's name is not a label. `muf/stations.py` resolves it to coordinates,
those coordinates give `path_km`, and `path_km` gives the M-factor that every
foF2 in this archive is divided by. So renaming a receiver is a geometry
change wearing a rename's clothes, and doing it with `UPDATE ... SET rx = ...`
leaves a database whose distances still describe the old site.

This exists for one specific correction. The acquisition laptop wrote
`station_name=DOB` from the day it was installed, and the site is Yoshkar-Ola
(56.38N 47.53E) -- `DOB` is Dombas, 62.073N 9.111E, a real entry in v2's
network table belonging to a different site 2400 km away. Every sounding
ingested up to 2026-08-16 carries it. Confirmed by measurement, not assertion:
with the Nicosia->Dombas distance the reference transmitter solved to
`epoch_offset_s = -0.002108` against a GPSDO-disciplined recorder, where 2.1 ms
of real clock error is not available; the Yoshkar-Ola distance moves it to
+0.000975, and positive is the only sign an unmodelled hop excess can produce.

What it touches, and why each one:

    sounding        rx, and the rx_lat/rx_lon/path_km/path_type that the name
                    decides. path_km is recomputed from the row's own stored
                    tx coordinates, so a transmitter this registry does not
                    know still gets the right answer.
    extraction      hops only. hop_count() is a function of path length, and a
                    path that crosses 4000 km in either direction changes it.
                    Everything else in that table is measurement.
    config_epoch    station
    transmitter     station -- keyed by the RECEIVER, see schema.sql
    health_report   station
    command         station

What it deliberately does not touch:

    gate_lo/gate_hi     They record the window extraction actually ran in, not
                        the window the corrected geometry would have chosen.
                        Rewriting them would describe a run that never
                        happened. Re-extract if you want them right; the MUF
                        and LOF in `extraction` were found inside the old gate
                        whatever this column says.
    extraction.muf      Measurements. A MUF is the highest frequency that came
    extraction.lof      back, and a virtual range is a delay. Neither knows
    extraction.vrange   where the receiver is. The foF2 derived from them does,
                        and it is derived at read time from path_km, so fixing
                        path_km fixes it everywhere at once.
    reference           IRI and GIRO values are evaluated at the path's control
                        point, which moves ~1000 km under this rename, and the
                        GIRO station chosen by proximity moves with it. They
                        are stale, not wrong-by-a-little. Counted and reported;
                        `--drop-reference` deletes them so they get recomputed,
                        which is the only honest option besides leaving them.

**The one thing this cannot fix.** The old name is inside the product files on
disk -- in `station_name` and in the filename -- and `muf/io_chirp.py` prefers
the file's own attribute over everything else, on the grounds that the file is
the record of what was actually written. So re-ingesting a pre-rename product
puts the old name back on that row. This is correct behaviour and the reason
the count of such files is printed: it is the size of the hazard, not a bug to
work around here.

Usage:

    python tools/relabel_station.py --db data/ionograms.sqlite3 \\
        --from DOB --to Yoshkar-Ola

Dry run by default: it prints the plan and writes nothing. Add `--apply` to
commit, which first takes a consistent copy of the database beside itself.

The api can stay up. Measured on a synthetic archive the size of the real one,
6977 soundings with an extraction each: 35 ms to survey, 86 ms to apply,
process start included. That is short enough that the write lock is not worth
scheduling around, and the transaction is what makes it safe either way.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Running a script puts tools/ on sys.path, not the repo root. Same shim, and
# the same reason, as tools/diagnose_reception.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from muf.geometry import Point, great_circle_km, hop_count   # noqa: E402
from muf.stations import default_registry                    # noqa: E402
from services.api import db                                  # noqa: E402

#: Tables with a plain `station` column and nothing derived from it.
STATION_TABLES = ("config_epoch", "transmitter", "health_report", "command")

#: `reference.source` values evaluated at a control point, i.e. everything.
#: Listed rather than assumed so that a source which is genuinely
#: path-independent can be added here and survive a rename.
PATH_DEPENDENT_SOURCES = ("iri", "giro", "chapman", "minimuf")


@dataclass
class Plan:
    """Everything the rename would do, computed before anything is written."""

    old: str
    new: str
    old_coords: tuple[float, float] | None
    new_coords: tuple[float, float]

    soundings: int = 0
    #: Rows whose path length changes, and by how much. Sampled for the report.
    path_changes: list[tuple[str, str, float, float]] = field(default_factory=list)
    hop_changes: int = 0
    path_type_changes: int = 0
    no_tx_coords: int = 0

    reference_rows: int = 0
    station_rows: dict[str, int] = field(default_factory=dict)

    #: (table, code-or-id) that a rename would collide with under a UNIQUE.
    collisions: list[tuple[str, str]] = field(default_factory=list)
    #: Commands queued to the old name and never delivered.
    undelivered: list[tuple[str, str, str]] = field(default_factory=list)
    #: Affected soundings whose file on disk still carries the old name.
    stale_filenames: int = 0

    @property
    def total(self) -> int:
        return self.soundings + sum(self.station_rows.values())


def resolve(name: str) -> tuple[float, float] | None:
    station = default_registry().station(name)
    return None if station is None else station.coordinates


def survey(conn: sqlite3.Connection, old: str, new: str) -> Plan:
    """Compute the plan. Reads only."""
    new_coords = resolve(new)
    if new_coords is None:
        raise SystemExit(
            f"'{new}' is not in muf/stations.py, so every renamed row would get "
            f"NaN coordinates and no path length -- which is the failure this "
            f"tool exists to repair, not to cause. Add the station there first."
        )

    plan = Plan(old=old, new=new, old_coords=resolve(old), new_coords=new_coords)
    rx = Point(*new_coords)

    rows = conn.execute(
        "SELECT id, tx, datetime, file, tx_lat, tx_lon, path_km, path_type "
        "FROM sounding WHERE rx = ?", (old,)
    ).fetchall()
    plan.soundings = len(rows)

    for row in rows:
        if old in (row["file"] or ""):
            plan.stale_filenames += 1

        if row["tx_lat"] is None or row["tx_lon"] is None:
            plan.no_tx_coords += 1
            continue

        was = row["path_km"]
        now = great_circle_km(Point(row["tx_lat"], row["tx_lon"]), rx)
        if was is None or abs(was - now) > 0.05:
            plan.path_changes.append((row["tx"] or "?", row["datetime"],
                                      was if was is not None else float("nan"), now))
        if was is not None and hop_count(was) != hop_count(now):
            plan.hop_changes += 1
        # tx == rx is a vertical sounding; the rename can create or destroy that
        # identity, and div_coef is 2 vs 4 on the strength of it.
        becomes = "oblique" if (row["tx"] or "") != new else "vertical"
        if becomes != (row["path_type"] or ""):
            plan.path_type_changes += 1

    # Subquery, not an IN-list of ids: this archive has thousands of soundings
    # and SQLITE_MAX_VARIABLE_NUMBER is 999 on the builds old enough to matter.
    sources = ",".join("?" * len(PATH_DEPENDENT_SOURCES))
    plan.reference_rows = conn.execute(
        f"SELECT COUNT(*) FROM reference "
        f"WHERE sounding_id IN (SELECT id FROM sounding WHERE rx = ?) "
        f"AND source IN ({sources})",
        (old, *PATH_DEPENDENT_SOURCES),
    ).fetchone()[0]

    for table in STATION_TABLES:
        plan.station_rows[table] = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE station = ?", (old,)
        ).fetchone()[0]

    # `transmitter` is UNIQUE (station, code) and UNIQUE (station, sounder_id).
    # If the new name already has rows -- and it will, if an operator has been
    # identifying transmitters under it -- the rename hits both constraints and
    # the whole transaction rolls back. Find them now rather than at COMMIT.
    for row in conn.execute(
        "SELECT old.code, old.sounder_id FROM transmitter old "
        "JOIN transmitter new ON new.station = ? "
        "AND (new.code = old.code OR new.sounder_id = old.sounder_id) "
        "WHERE old.station = ?", (new, old)
    ):
        plan.collisions.append(("transmitter", f"{row['code']} (id {row['sounder_id']})"))

    # A command queued to the old name has been sitting undeliverable, because
    # no agent reports under that name any more. Renaming it makes it live: the
    # station collects it on its next pull and acts on it. A `set_config` from
    # a week ago landing on a reconfigured station is not a rename's business.
    for row in conn.execute(
        "SELECT id, name, issued_at FROM command "
        "WHERE station = ? AND delivered_at IS NULL "
        "ORDER BY issued_at", (old,)
    ):
        plan.undelivered.append((row["id"], row["name"], row["issued_at"]))

    return plan


def report(plan: Plan) -> None:
    old_where = (f"{plan.old_coords[0]:.3f}N {plan.old_coords[1]:.3f}E"
                 if plan.old_coords else "not in the registry")
    print(f"{plan.old} -> {plan.new}")
    print(f"  {plan.old:<16} {old_where}")
    print(f"  {plan.new:<16} {plan.new_coords[0]:.3f}N {plan.new_coords[1]:.3f}E")
    print()

    print(f"sounding          {plan.soundings} row(s) with rx = {plan.old}")
    if plan.no_tx_coords:
        print(f"                  {plan.no_tx_coords} have no tx coordinates; "
              f"path_km stays NULL")
    if plan.path_changes:
        print(f"                  {len(plan.path_changes)} path length(s) change:")
        by_tx: dict[str, tuple[int, float, float]] = {}
        for tx, _, was, now in plan.path_changes:
            count, _, _ = by_tx.get(tx, (0, was, now))
            by_tx[tx] = (count + 1, was, now)
        for tx, (count, was, now) in sorted(by_tx.items()):
            print(f"                    {tx:<12} {was:8.1f} -> {now:8.1f} km "
                  f"({now - was:+.1f})  x{count}")
    if plan.hop_changes:
        print(f"                  {plan.hop_changes} sounding(s) change hop count "
              f"-- extraction.hops updated to match")
    if plan.path_type_changes:
        print(f"                  {plan.path_type_changes} change path_type "
              f"(oblique/vertical); check this is intended")

    for table, count in plan.station_rows.items():
        print(f"{table:<18}{count} row(s)")

    if plan.reference_rows:
        print()
        print(f"reference         {plan.reference_rows} modelled value(s) are now "
              f"stale -- evaluated at the old path's control point.")
        print(f"                  Left alone unless --drop-reference.")

    if plan.stale_filenames:
        print()
        print(f"NOTE  {plan.stale_filenames} of these soundings are files still "
              f"named '{plan.old}' on disk, and the name is inside the h5 as "
              f"well. Re-ingesting one puts {plan.old} back on that row.")

    if plan.undelivered:
        print()
        print(f"STOP  {len(plan.undelivered)} command(s) queued to {plan.old} were "
              f"never delivered. Renaming them hands them to the live station:")
        for cid, name, issued in plan.undelivered:
            print(f"        {issued}  {name:<12} {cid}")
        print(f"      Pass --release-pending to rename them anyway, or cancel "
              f"them in the console first. Without it they stay under "
              f"{plan.old}, where nothing will collect them.")

    if plan.collisions:
        print()
        print(f"STOP  {len(plan.collisions)} row(s) would violate a UNIQUE "
              f"constraint, because {plan.new} already has them:")
        for table, what in plan.collisions:
            print(f"        {table:<14}{what}")
        print(f"      Pass --on-conflict=keep-new to drop the {plan.old} row, "
              f"or --on-conflict=keep-old to drop the {plan.new} one.")


def apply(conn: sqlite3.Connection, plan: Plan, *,
          on_conflict: str | None, release_pending: bool,
          drop_reference: bool) -> None:
    """Write the plan. One transaction: it lands whole or not at all."""
    rx = Point(*plan.new_coords)

    with conn:  # commits on success, rolls back on any exception
        if plan.collisions and on_conflict == "keep-new":
            conn.execute(
                "DELETE FROM transmitter WHERE station = ? AND (code IN "
                "(SELECT code FROM transmitter WHERE station = ?) OR sounder_id IN "
                "(SELECT sounder_id FROM transmitter WHERE station = ?))",
                (plan.old, plan.new, plan.new))
        elif plan.collisions and on_conflict == "keep-old":
            # Only the rows that actually collide. Dropping every row the new
            # name has would take out identifications it made for transmitters
            # the old name never sounded, which is not what "keep the old one"
            # asks for.
            conn.execute(
                "DELETE FROM transmitter WHERE station = ? AND (code IN "
                "(SELECT code FROM transmitter WHERE station = ?) OR sounder_id IN "
                "(SELECT sounder_id FROM transmitter WHERE station = ?))",
                (plan.new, plan.old, plan.old))

        # Deleted before the rename, while `rx` still identifies the rows.
        if drop_reference:
            sources = ",".join("?" * len(PATH_DEPENDENT_SOURCES))
            conn.execute(
                f"DELETE FROM reference "
                f"WHERE sounding_id IN (SELECT id FROM sounding WHERE rx = ?) "
                f"AND source IN ({sources})",
                (plan.old, *PATH_DEPENDENT_SOURCES))

        rows = conn.execute(
            "SELECT id, tx, tx_lat, tx_lon FROM sounding WHERE rx = ?",
            (plan.old,)).fetchall()

        for row in rows:
            if row["tx_lat"] is None or row["tx_lon"] is None:
                path_km = None
            else:
                path_km = great_circle_km(Point(row["tx_lat"], row["tx_lon"]), rx)

            conn.execute(
                "UPDATE sounding SET rx = ?, rx_lat = ?, rx_lon = ?, path_km = ?, "
                "path_type = ? WHERE id = ?",
                (plan.new, plan.new_coords[0], plan.new_coords[1], path_km,
                 "oblique" if (row["tx"] or "") != plan.new else "vertical",
                 row["id"]))

            if path_km is not None:
                conn.execute(
                    "UPDATE extraction SET hops = ? WHERE sounding_id = ? "
                    "AND hops IS NOT NULL AND hops != ?",
                    (hop_count(path_km), row["id"], hop_count(path_km)))

        for table in STATION_TABLES:
            if table == "command" and not release_pending:
                conn.execute(
                    "UPDATE command SET station = ? "
                    "WHERE station = ? AND delivered_at IS NOT NULL",
                    (plan.new, plan.old))
            else:
                conn.execute(f"UPDATE {table} SET station = ? WHERE station = ?",
                             (plan.new, plan.old))


def backup(path: Path) -> Path:
    """A consistent copy, via SQLite's own backup API.

    Not `cp`: the database runs in WAL mode, so a file copy taken while the api
    is writing can land mid-transaction and is not guaranteed to open.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = path.with_name(f"{path.name}.bak-{stamp}")
    source = sqlite3.connect(str(path))
    try:
        with sqlite3.connect(str(target)) as dest:
            source.backup(dest)
    finally:
        source.close()
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=db.DEFAULT_DB,
                        help=f"SQLite database (default: {db.DEFAULT_DB})")
    parser.add_argument("--from", dest="old", required=True,
                        help="the name as it is stored now")
    parser.add_argument("--to", dest="new", required=True,
                        help="the name to give it; must be in muf/stations.py")
    parser.add_argument("--apply", action="store_true",
                        help="write the changes (default: report and exit)")
    parser.add_argument("--on-conflict", choices=("keep-new", "keep-old"),
                        help="how to resolve a UNIQUE collision in `transmitter`")
    parser.add_argument("--release-pending", action="store_true",
                        help="also rename undelivered commands, handing them to "
                             "the live station on its next pull")
    parser.add_argument("--drop-reference", action="store_true",
                        help="delete modelled values evaluated at the old "
                             "control point so they are recomputed")
    args = parser.parse_args(argv)

    if not args.db.exists():
        raise SystemExit(f"no database at {args.db}")

    conn = db.connect(args.db)
    try:
        plan = survey(conn, args.old, args.new)
        report(plan)

        if plan.total == 0:
            print()
            print(f"Nothing carries the name '{args.old}'. Nothing to do.")
            return 0

        blocked = []
        if plan.collisions and args.on_conflict is None:
            blocked.append("--on-conflict")
        if plan.undelivered and not args.release_pending:
            # Not fatal: undelivered commands are simply left behind, which is
            # the safe half of the decision. Say so rather than implying it.
            print()
            print(f"Undelivered commands will be LEFT under '{args.old}'.")

        if not args.apply:
            print()
            print("Dry run. Nothing was written. Add --apply to commit.")
            return 0

        if blocked:
            print()
            raise SystemExit(f"Refusing to apply: {', '.join(blocked)} needed. "
                             f"See STOP above.")

        copy = backup(args.db)
        print()
        print(f"Backup: {copy}")
        apply(conn, plan, on_conflict=args.on_conflict,
              release_pending=args.release_pending,
              drop_reference=args.drop_reference)
        print(f"Applied. {plan.total} row(s) now read '{args.new}'.")
        if plan.stale_filenames:
            print(f"Re-ingesting any of the {plan.stale_filenames} files still "
                  f"named '{args.old}' will undo it for that row.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
