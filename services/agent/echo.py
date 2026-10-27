"""Cut the echo window into each finished chirpsounder2 product, then drop `z`.

With ``save_raw_voltage = true`` v2 writes the full dechirped voltage `z` into
every ``lfm_ionogram-*.h5``: about 78 MB a sounding against 0.6 MB without it.
Kept whole, that is ~25 GB a day, which neither the 31 GB staging partition
nor the SMB1 link to the NAS can carry. Thrown away, it takes with it the
only record of each echo's Doppler, fading and phase. The echo window
(`docs/echo-window.md`) keeps exactly that part, about 1.6 MB, and
`muf.echo_window.write(drop_raw=True)` already appends it, verifies it, and
rewrites the product without `z` by rename, so an interruption leaves the
original.

This module is the station's half: a sweep, run by
``chirp-echo-window.service`` just before every NAS copy, so that only the cut
product ever crosses the network.

**What it never does is lose the voltage.** A product with nothing to keep --
no echo stands out, and its path has no geometry -- keeps its `z` and is
counted; the prune job removes the local copy after its NAS copy is proven,
as for any product. A cut that does not verify keeps `z` and fails the run, so
the job turns red rather than quietly shipping 78 MB files.

`muf` is imported inside :func:`run`, the way ``preview.py`` imports numpy and
h5py, so the rest of the agent stays stdlib-only. The modules it pulls in are
held to Python 3.8 by ``tests/test_echo_window_py38.py``: the station's venv
is 3.8 while `muf` itself declares 3.10.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

#: Younger products may still be open in calc_ionograms. Two minutes is well
#: past the time v2 takes to write one, and well inside the five between
#: NAS passes, so a product is cut on the pass after it appears.
DEFAULT_MIN_AGE_S = 120.0

#: Products only: the window exists for v2 sounding products, and this walks
#: a staging tree that also holds detection snippets and partial transfers.
PATTERN = "lfm_ionogram-*.h5"


def candidates(root: Path, *, min_age_s: float,
               now: float | None = None) -> list[Path]:
    """Products old enough to be finished, oldest first by name.

    An age below zero reads as zero. On Windows, Python 3.12's clock and NTFS
    share one coarse tick and round an instant to floats differently, so a
    product written in this call's own tick came out one ulp younger than zero
    (#51).
    """
    now = time.time() if now is None else now
    out: list[Path] = []
    for path in sorted(root.rglob(PATTERN)):
        try:
            if max(now - path.stat().st_mtime, 0.0) >= min_age_s:
                out.append(path)
        except OSError:
            continue
    return out


def run(root: Path, *, min_age_s: float, now: float | None = None,
        dry_run: bool = False) -> dict:
    """One pass. Never raises on a single product; a failed one is retried."""
    from muf import echo_window
    result = {"scanned": 0, "cut": 0, "freed": 0, "kept": 0, "failed": 0,
              "errors": []}
    for path in candidates(root, min_age_s=min_age_s, now=now):
        result["scanned"] += 1
        try:
            size = path.stat().st_size
            report = echo_window.write(path, drop_raw=True, dry_run=dry_run)
        except echo_window.NoEchoError:
            result["kept"] += 1
            continue
        except (echo_window.VerificationError, OSError, ValueError,
                KeyError) as exc:
            result["failed"] += 1
            result["errors"].append(f"{path.name}: {exc}")
            continue
        if report.status in ("raw-dropped", "dry-run"):
            result["cut"] += 1
            if not dry_run:
                result["freed"] += size - path.stat().st_size
    return result


def describe(result: dict, dry_run: bool = False) -> str:
    verb = "would cut" if dry_run else "cut"
    line = (f"{verb} {result['cut']} of {result['scanned']} products, "
            f"freed {result['freed'] / 1e9:.2f} GB")
    if result["kept"]:
        line += (f"; {result['kept']} kept whole, with nothing to keep in a "
                 f"window")
    if result["failed"]:
        line += (f"; {result['failed']} failed and kept their z: "
                 + "; ".join(result["errors"]))
    return line


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path,
                        help="staging directory of lfm_ionogram products")
    parser.add_argument("--min-age-s", type=float, default=DEFAULT_MIN_AGE_S,
                        help="leave younger products for the next pass")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be cut, change nothing")
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        print(f"{args.root}: no such directory", file=sys.stderr)
        return 2
    result = run(args.root, min_age_s=args.min_age_s, dry_run=args.dry_run)
    print(describe(result, args.dry_run))
    return 1 if result["failed"] else 0


if __name__ == "__main__":                                    # pragma: no cover
    raise SystemExit(main())
