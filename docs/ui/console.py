"""The simulator console: one page that shows the simulator at work.

    python docs/ui/console.py SOUNDING [SOUNDING ...] [--out FILE]
    python docs/ui/console.py --reuse console.html     # new template, old data
    python docs/ui/console.py ... --json console.json  # the data alone as well

Writes what :func:`muf.sim.console.build` gathers for each sounding into
``services/api/sim_console.html`` as one self-contained page
(default ``console.html``). The hub serves the same page at ``/ui/sim``,
building a sounding's data on demand; this script is the offline copy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: The page, shared with the hub, which serves it at /sim/{id}/console.
TEMPLATE = HERE.parents[1] / "services" / "api" / "sim_console.html"

#: Where the template takes its data.
DATA_LEAD = "const DATA = "


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("soundings", type=Path, nargs="*")
    ap.add_argument("--out", type=Path, default=HERE / "console.html")
    ap.add_argument("--reuse", type=Path, metavar="PAGE",
                    help="take the data from a page built before, run nothing")
    ap.add_argument("--json", type=Path, metavar="FILE",
                    help="also write the data alone, for a page that loads it")
    args = ap.parse_args(argv)
    if args.reuse:
        old = args.reuse.read_text(encoding="utf-8")
        start = old.index(DATA_LEAD) + len(DATA_LEAD)
        blob = old[start:old.index(";\n", start)]
    elif args.soundings:
        from muf.sim import console

        data = []
        for path in args.soundings:
            data.append(console.build(path))
            print(f"  {path.name}")
        blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    else:
        ap.error("give soundings or --reuse")
    if args.json:
        args.json.write_text(blob + "\n", encoding="utf-8")
        print(f"-> {args.json} ({args.json.stat().st_size / 1e3:.0f} kB)")
    page = TEMPLATE.read_text(encoding="utf-8")
    args.out.write_text(page.replace(DATA_LEAD + "/*DATA*/null", DATA_LEAD + blob),
                        encoding="utf-8")
    print(f"-> {args.out} ({args.out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
