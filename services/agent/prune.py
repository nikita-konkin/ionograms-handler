"""Delete local products that are provably already on the archive.

The other half of ``chirp-archive-sync``. That job mirrors and never deletes,
deliberately -- a mover removes the only good copy the one time the
destination is silently wrong. On a 17 TB spinning disk that costs nothing. On
a 20 GB slice of the boot SSD it fills in a week, writes begin to fail, and
acquisition stops. So something has to delete, and the question is only how
carefully.

**Three checks before any unlink**, because the failure this guards against is
unrecoverable and the one it causes is merely a full disk:

1. The remote root must look like a mounted archive -- present, a directory,
   and not empty. An unmounted share is an empty directory, and pruning
   against it would delete everything while "confirming" nothing.
2. The file must be older than ``min_age_s``. Recent products may still be
   mid-write, and the sync may not have reached them.
3. The same relative path must exist on the remote **with the same size**.
   Not a timestamp comparison: SMB stamps come from the file server, which on
   this station was 5 h 36 m fast for a day. Size is the property both ends
   agree on.

Only then is the local copy removed. A file that fails any check is left
alone and reconsidered next pass, which is the right default for a job that
runs every hour forever.

Stdlib only, like the rest of this package: it runs under the station's own
Python.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

#: Keep this much history locally regardless. Long enough that a NAS outage
#: over a weekend is invisible, short enough to fit a boot-SSD budget: at the
#: ~3 GB/day this station produces, three days is ~9 GB.
DEFAULT_MIN_AGE_DAYS = 3.0

#: Products only. Never anything else: this walks a directory that may hold
#: logs, partial transfers and `.rsync-partial`, and a prune that deletes by
#: age alone would take them too.
PATTERNS = ("lfm_ionogram-*.h5", "digisonde_ionogram-*.h5", "par-*.h5",
            "chirp-*.h5")


def archive_is_mounted(remote_root: Path) -> bool:
    """Does the remote look like a mounted archive rather than a bare stub.

    An unmounted CIFS share is an empty directory that every `exists()` check
    passes. Requiring content is what stops a prune from confirming each file
    against nothing and deleting the lot.
    """
    try:
        if not remote_root.is_dir():
            return False
        return any(remote_root.iterdir())
    except OSError:
        return False


def is_safe_to_delete(local: Path, local_root: Path, remote_root: Path) -> bool:
    """Is ``local`` present on the remote at the same relative path and size."""
    try:
        relative = local.relative_to(local_root)
    except ValueError:                                        # pragma: no cover
        return False
    remote = remote_root / relative
    try:
        return remote.is_file() and remote.stat().st_size == local.stat().st_size
    except OSError:
        return False


def prunable(local_root: Path, remote_root: Path, *, min_age_s: float,
             now: float | None = None) -> list[Path]:
    """Local products old enough, and verified present on the remote."""
    now = time.time() if now is None else now
    out: list[Path] = []
    for pattern in PATTERNS:
        for path in sorted(local_root.rglob(pattern)):
            try:
                if now - path.stat().st_mtime < min_age_s:
                    continue
            except OSError:
                continue
            if is_safe_to_delete(path, local_root, remote_root):
                out.append(path)
    return out


def run(local_root: Path, remote_root: Path, *, min_age_s: float,
        dry_run: bool = False) -> dict:
    """One pass. Never raises on a single file; a locked one is next time's."""
    result = {"scanned": 0, "removed": 0, "bytes": 0, "skipped": 0,
              "mounted": archive_is_mounted(remote_root)}
    if not result["mounted"]:
        return result

    candidates = prunable(local_root, remote_root, min_age_s=min_age_s)
    result["scanned"] = len(candidates)
    for path in candidates:
        try:
            size = path.stat().st_size
            if not dry_run:
                path.unlink()
            result["removed"] += 1
            result["bytes"] += size
        except OSError:
            result["skipped"] += 1
    return result


def describe(result: dict, dry_run: bool = False) -> str:
    if not result["mounted"]:
        return ("archive is not mounted or is empty; nothing pruned -- this is "
                "the guard working, not a failure")
    verb = "would remove" if dry_run else "removed"
    line = (f"{verb} {result['removed']} of {result['scanned']} verified "
            f"copies, {result['bytes'] / 1e9:.2f} GB")
    if result["skipped"]:
        line += f"; {result['skipped']} could not be removed, retried next pass"
    return line


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("local", type=Path, help="staging directory to prune")
    parser.add_argument("remote", type=Path, help="archive that must hold a copy")
    parser.add_argument("--min-age-days", type=float,
                        default=DEFAULT_MIN_AGE_DAYS)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would go, delete nothing")
    args = parser.parse_args(argv)

    if not args.local.is_dir():
        print(f"{args.local}: no such directory", file=sys.stderr)
        return 1
    result = run(args.local, args.remote,
                 min_age_s=args.min_age_days * 86400.0, dry_run=args.dry_run)
    print(describe(result, args.dry_run))
    return 0


if __name__ == "__main__":                                    # pragma: no cover
    raise SystemExit(main())
