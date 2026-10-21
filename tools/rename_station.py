"""Rename a station's agent, keeping the receiver name its files carry.

A station has two names. Its agent reports under one -- `station` in
`agent.json` -- and keys health reports, commands, configuration epochs,
identified transmitters, previews and live summaries. Its recorder writes the
other, `rx`, into every product file. Yoshkar-Ola's two receivers wrote
`Yoshkar-Ola` and `yoshkar-ola`, apart only by case, so their agents became
`yoshkar-ola-1` and `yoshkar-ola-2` (issue #27) while the files kept what they
say: a file is the record of what was written.

This moves every row kept under the agent's old name to the new one, and
records that the old `rx` is the new station's own, so the console, the
Sources page and the live archive still find its soundings. It does not touch
`sounding` -- for a receiver whose name was simply wrong, which is a geometry
change, see `tools/relabel_station.py`.

Usage, once the agent reports under the new name or is stopped:

    python tools/rename_station.py --db data/ionograms.sqlite3 \\
        --from Yoshkar-Ola --to yoshkar-ola-1

Dry run by default: it prints the plan and writes nothing. Add `--apply` to
commit, which first takes a consistent copy of the database beside itself.
Running it again is safe, so rows the old agent sent after the first run can
be moved by a second.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Running a script puts tools/ on sys.path, not the repo root. Same shim, and
# the same reason, as tools/relabel_station.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.api import db


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=db.DEFAULT_DB,
                        help=f"SQLite database (default: {db.DEFAULT_DB})")
    parser.add_argument("--from", dest="old", required=True,
                        help="the agent name as it is stored now")
    parser.add_argument("--to", dest="new", required=True,
                        help="the agent name to give it")
    parser.add_argument("--apply", action="store_true",
                        help="write the changes (default: report and exit)")
    args = parser.parse_args(argv)
    if not args.db.exists():
        raise SystemExit(f"no database at {args.db}")
    with db.session(args.db) as conn:
        print(f"{args.old} -> {args.new}")
        for table, n in db.station_rows(conn, args.old).items():
            print(f"  {table:<18}{n} row(s)")
        print(f"  receivers         {', '.join(db.receivers(conn, args.old))}"
              f" become {args.new}'s own")
        if not args.apply:
            print("Dry run. Nothing was written. Add --apply to commit.")
            return 0
        print(f"Backup: {db.backup(args.db)}")
        try:
            db.rename_station(conn, args.old, args.new)
        except ValueError as exc:
            raise SystemExit(f"Refused: {exc}") from None
        print(f"Applied. {args.new} owns {', '.join(db.receivers(conn, args.new))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
