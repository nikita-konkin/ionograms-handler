"""Build the public results page: a folder of static files, nothing live.

The page is served by the host's own web server straight from a folder. It is
deliberately **not** the api behind a proxy. Reads on the api are open in every
deployment so far (no ``READ_TOKEN``), and the read router also answers
``/principals``, ``/models/{id}/artifact``, station health and ``/net``; the
control routes live in the same process. A static snapshot exposes exactly
what this module writes and nothing else, and it keeps serving the last good
picture while the api, the archive mount or this job is down.

**What it writes**, under the site folder::

    index.html app.js style.css strings.json vendor/   the page, copied as is
    status.json                                         circuits, newest pictures
    data/{slug}.json                                    series, forecast, scores
    img/{slug}/{YYYYmmddTHHMMSS}.webp                   one image per sounding

**Allow-listed, twice.** Only circuits named in ``--circuits`` appear, and a
muted one is dropped even then -- muting stops new ingest but leaves the rows
already held, so it has to be checked here. The JSON fields are allow-listed
too: no sounding ids, file paths or model internals leave the server.

**Read-only.** The database is opened with ``mode=ro``. `db.session` and
`db.init` both write (schema, migrations), and a publisher that can write is a
publisher that can corrupt the thing it is publishing.

**Atomic per file.** Everything is written to a temporary name and renamed into
place, so the web server never sends half a file. Images are named for the
sounding's own time and never rewritten, which makes an existing file its own
render cache -- an ``.lfs`` re-runs the whole spectrogram to draw -- and lets
the web server mark ``img/`` immutable. Images nothing references any more are
deleted only after the JSON that stopped referencing them is in place.

Run it once, or with ``--interval`` to stay resident::

    python -m services.publish.publisher --db /data/ionograms.sqlite3 \\
        --out /publish/site --circuits "NIC->DOB" --interval 900
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from services.api import db
from services.api import series as series_mod

#: The page itself, copied into the site folder on every pass.
SITE_SRC = Path(__file__).resolve().parent / "site"

#: Most points one circuit's series carries. Thirty days of a sounding every
#: few minutes is tens of thousands of rows, which is a page that takes seconds
#: to parse on a phone for a picture that cannot show them anyway.
MAX_POINTS = 6000

#: Hours of thumbnails in the strip under the newest ionogram.
STRIP_HOURS = 24

#: The image size: about 640 x 280 px, a WebP of a few tens of kilobytes.
FIGSIZE = (8.0, 3.5)
DPI = 80
WEBP_QUALITY = 70

#: Prefix of every temporary file, so a pass can sweep up after a crash.
TMP_PREFIX = ".tmp-"

#: The database's own spelling of an instant, and the image names'.
DB_SECOND = "%Y-%m-%d %H:%M:%S"
IMG_STAMP = "%Y%m%dT%H%M%S"


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def connect_ro(path: Path) -> sqlite3.Connection:
    """The api's database, opened so that nothing here can write to it.

    A WAL reader still needs the ``-wal`` and ``-shm`` files beside it, which
    is why the compose service mounts the data volume read-write while this
    opens it read-only.
    """
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def parse_circuits(text: str | None) -> list[tuple[str, str]]:
    """``"NIC->DOB, sgo->dob"`` as ``[("nic", "dob"), ("sgo", "dob")]``.

    Folded to lower case: v2 products write ``DOB`` and ``.lfs`` writes
    ``yoshkar-ola``, and an allow-list that silently missed one spelling
    would read as "that circuit has no data".
    """
    out = []
    for part in (text or "").split(","):
        tx, arrow, rx = part.strip().partition("->")
        if arrow and tx.strip() and rx.strip():
            out.append((tx.strip().lower(), rx.strip().lower()))
    return out


def slug(tx: str, rx: str) -> str:
    """A circuit as a file name: ``NIC``, ``DOB`` -> ``nic-dob``."""
    return re.sub(r"[^a-z0-9]+", "-", f"{tx}-{rx}".lower()).strip("-")


def station_name(code: str) -> str:
    """A human name for a station code, or the code when none is known."""
    from muf.stations import default_registry

    station = default_registry().station(code)
    return station.name if station is not None else code


def resolve(path: str, archive_root: Path) -> Path:
    """A ``sounding.path`` as a file: relative to the root, or absolute."""
    found = Path(path)
    return found if found.is_absolute() else Path(archive_root) / found


def _db_time(when: datetime) -> str:
    return when.astimezone(timezone.utc).strftime(DB_SECOND)


def _parse_db_time(text: str) -> datetime:
    """``2026-08-13 04:00:00.123456`` (or with a ``T``) as an aware UTC instant."""
    return datetime.strptime(str(text).replace("T", " ")[:19],
                             DB_SECOND).replace(tzinfo=timezone.utc)


def _iso(text: str | None) -> str | None:
    """A database instant as ISO 8601 UTC with an explicit ``Z``."""
    if not text:
        return None
    return _parse_db_time(text).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def render_webp(path: Path) -> bytes:
    """One sounding as a small WebP with axes, ready to publish.

    Gated by `sao.load_ion` exactly as the console's PNG is, so the two show
    the same window. No estimators run: the picks come from the database, and
    re-deriving them only to draw them would triple the cost.
    """
    import matplotlib
    matplotlib.use("Agg")
    from PIL import Image

    from muf import render
    from services.api import sao as sao_mod

    ion = sao_mod.load_ion(path, gate="auto")
    png = io.BytesIO()
    render.plot(ion, png, None, axes=True, lof=False, dpi=DPI, figsize=FIGSIZE)
    png.seek(0)
    out = io.BytesIO()
    with Image.open(png) as image:
        image.convert("RGB").save(out, format="WEBP", quality=WEBP_QUALITY,
                                  method=6)
    return out.getvalue()


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

def write_atomic(path: Path, data: bytes) -> bool:
    """Write ``data`` to ``path`` unless it already holds exactly that.

    Returns whether anything was written. Skipping an unchanged file keeps its
    mtime, and with it the web server's ETag, so a browser revalidating
    ``status.json`` every visit gets a 304 until something actually changed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.read_bytes() == data:
            return False
    except OSError:
        pass
    temporary = path.with_name(f"{TMP_PREFIX}{path.name}.{os.getpid()}")
    try:
        temporary.write_bytes(data)
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def copy_site(site_dir: Path) -> int:
    """The page's own files into the site folder. Returns how many changed."""
    changed = 0
    for source in sorted(SITE_SRC.rglob("*")):
        if source.is_file():
            target = site_dir / source.relative_to(SITE_SRC)
            changed += write_atomic(target, source.read_bytes())
    return changed


def sweep_temporaries(site_dir: Path) -> None:
    """Remove what a pass that died mid-write left behind."""
    for leftover in site_dir.rglob(f"{TMP_PREFIX}*"):
        leftover.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# One circuit
# --------------------------------------------------------------------------

def _thin(rows: list[dict], limit: int = MAX_POINTS) -> list[dict]:
    """At most ``limit`` rows, evenly spaced, always keeping the newest."""
    if len(rows) <= limit:
        return rows
    step = math.ceil(len(rows) / limit)
    return rows[::-1][::step][::-1]


def _number(value):
    """A finite float, or None -- JSON has no NaN, and a stored one must not
    fail the whole pass."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def series_block(conn, tx: str, rx: str, method: str, since: datetime) -> dict:
    """The measured series and the live forecast, fields allow-listed."""
    rows = _thin(series_mod.picks(conn, method, tx=tx, rx=rx,
                                  start=_db_time(since), limit=None))
    framed = series_mod.frame(rows, model="off",
                              forecasts=series_mod.live_forecasts(conn, rows))
    if not framed["circuits"]:
        return {"path_km": None, "t": [], "muf": [], "lof": [], "limited": [],
                "loflim": [], "forecast": []}
    circuit = framed["circuits"][0]
    return {
        "path_km": _number(circuit["path_km"]),
        "t": circuit["t"],
        "muf": circuit["muf"], "lof": circuit["lof"],
        "limited": circuit["limited"], "loflim": circuit["loflim"],
        "forecast": [{
            "param": entry["param"], "model": entry["model"],
            "trained_from": entry["trained_from"],
            "trained_to": entry["trained_to"],
            "t": entry["t"],
            "value": [_number(v) for v in entry["value"]],
            "sigma": [_number(v) for v in entry["sigma"]],
        } for entry in circuit["forecast"] if not entry.get("comparison")],
    }


def forecast_block(conn, tx: str, rx: str) -> dict:
    """The active model and how it scores against the baselines.

    Only the active model and the baselines: a comparison model is a candidate
    being judged, and putting it on the public page beside the answer would
    present it as one.
    """
    from services.prediction import registry, scoring

    try:
        model = registry.active(conn, "muf", tx, rx)
        board = scoring.leaderboard(conn, "muf", tx, rx)
    except sqlite3.Error:
        return {"model": None, "scores": [], "horizons": []}

    issued = None
    if model is not None:
        row = db.one(conn, "SELECT MAX(issued_at) AS at FROM forecast"
                           " WHERE model_id = ? AND param = 'muf'"
                           " AND tx = ? AND rx = ?", (model["id"], tx, rx))
        issued = row["at"] if row else None
    return {
        "model": None if model is None else {
            "name": model["name"],
            "trained_from": _iso(model.get("trained_from")),
            "trained_to": _iso(model.get("trained_to")),
            "issued_at": _iso(issued),
        },
        "scores": [{
            "name": entry["name"], "kind": entry["kind"],
            # `skill` is absent from score rows written before it existed.
            "mae": entry["mae"], "skill": entry.get("skill") or {},
            "n": entry["n"],
        } for entry in board
            if entry["kind"] == "baseline" or entry.get("state") == "active"],
        "horizons": [str(h) for h in scoring.HORIZONS],
    }


def pictures(conn, tx: str, rx: str, now: datetime) -> list[dict]:
    """The soundings to draw: the newest in each of the last 24 hours."""
    found = db.rows(conn,
                    "SELECT datetime, path FROM sounding"
                    " WHERE tx = ? AND rx = ? AND datetime >= ?"
                    " ORDER BY datetime",
                    (tx, rx, _db_time(now - timedelta(hours=STRIP_HOURS))))
    newest_per_hour: dict[str, dict] = {}
    for row in found:
        newest_per_hour[str(row["datetime"])[:13]] = row
    return list(newest_per_hour.values())


def draw(site_dir: Path, name: str, rows: list[dict], archive_root: Path,
         result: dict) -> list[dict]:
    """Each sounding's image, rendering only those not already on disk.

    A failure skips that one picture -- a sounding whose file is missing or on
    a mount that is not answering must not cost the page every other image.
    """
    drawn = []
    for row in rows:
        when = _parse_db_time(row["datetime"])
        relative = f"img/{name}/{when.strftime(IMG_STAMP)}.webp"
        target = site_dir / relative
        if not target.exists():
            try:
                write_atomic(target, render_webp(resolve(row["path"], archive_root)))
            except Exception as exc:
                result["skipped"] += 1
                result["errors"].append(f"{relative}: {type(exc).__name__}: {exc}")
                continue
            result["rendered"] += 1
        drawn.append({"t": when.strftime("%Y-%m-%dT%H:%M:%SZ"), "img": relative})
    return drawn


# --------------------------------------------------------------------------
# The pass
# --------------------------------------------------------------------------

def choose(conn, wanted: list[tuple[str, str]]) -> list[dict]:
    """Allow-listed circuits that exist and are not muted, in list order."""
    held = db.rows(conn, "SELECT tx, rx, MAX(datetime) AS last, COUNT(*) AS n"
                         " FROM sounding GROUP BY tx, rx")
    muted = db.muted_matcher(conn)
    chosen, seen = [], set()
    for want in wanted:
        for row in held:
            if ((str(row["tx"]).lower(), str(row["rx"]).lower()) == want
                    and not muted(row["tx"], row["rx"])
                    and slug(row["tx"], row["rx"]) not in seen):
                seen.add(slug(row["tx"], row["rx"]))
                chosen.append(row)
    return chosen


def build_site(conn, site_dir: Path, *, circuits: list[tuple[str, str]],
               method: str, days: int, archive_root: Path,
               now: datetime | None = None) -> dict:
    """One full pass. Returns counts for the log line."""
    now = now or datetime.now(timezone.utc)
    site_dir = Path(site_dir)
    site_dir.mkdir(parents=True, exist_ok=True)
    sweep_temporaries(site_dir)

    result = {"circuits": 0, "rendered": 0, "skipped": 0, "pruned": 0,
              "errors": []}
    result["site_files"] = copy_site(site_dir)

    listed = []
    keep_images: set[str] = set()
    keep_data: set[str] = set()
    for row in choose(conn, circuits):
        tx, rx = row["tx"], row["rx"]
        name = slug(tx, rx)
        series = series_block(conn, tx, rx, method,
                              since=now - timedelta(days=days))
        drawn = draw(site_dir, name, pictures(conn, tx, rx, now),
                     archive_root, result)
        keep_images.update(entry["img"] for entry in drawn)

        latest_pick = None
        for i in range(len(series["t"]) - 1, -1, -1):
            if series["muf"][i] is not None or series["lof"][i] is not None:
                latest_pick = {"t": series["t"][i], "muf": series["muf"][i],
                               "lof": series["lof"][i],
                               "limited": series["limited"][i]}
                break

        data_path = f"data/{name}.json"
        keep_data.add(data_path)
        write_atomic(site_dir / data_path, _json_bytes({
            "slug": name, "method": method, "days": days,
            "series": series, "forecast": forecast_block(conn, tx, rx),
        }))
        listed.append({
            "slug": name, "tx": tx, "rx": rx,
            "tx_name": station_name(tx), "rx_name": station_name(rx),
            "last_sounding": _iso(row["last"]),
            "latest_pick": latest_pick,
            "images": drawn,
            "data": data_path,
        })
        result["circuits"] += 1

    write_atomic(site_dir / "status.json", _json_bytes({
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": method, "days": days,
        "build": db.build_id(),
        "circuits": listed,
    }))

    # Only now, with every reference in place, may the unreferenced go.
    for pattern, keep in (("img/**/*.webp", keep_images),
                          ("data/*.json", keep_data)):
        for stale in site_dir.glob(pattern):
            if stale.relative_to(site_dir).as_posix() not in keep:
                stale.unlink(missing_ok=True)
                result["pruned"] += 1
    return result


def describe(result: dict) -> str:
    """One log line per pass."""
    line = (f"published {result['circuits']} circuit"
            f"{'' if result['circuits'] == 1 else 's'}, "
            f"{result['rendered']} new image"
            f"{'' if result['rendered'] == 1 else 's'}")
    if result["skipped"]:
        line += f", {result['skipped']} skipped ({result['errors'][0]})"
    if result["pruned"]:
        line += f", {result['pruned']} pruned"
    return line


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="services.publish.publisher",
        description="Write the public results page as static files.")
    parser.add_argument("--db", type=Path, default=None,
                        help="the api's database (default: $API_DB)")
    parser.add_argument("--out", type=Path,
                        default=Path(os.environ.get("PUBLISH_DIR", "/publish/site")),
                        help="the folder the web server serves "
                             "(default: $PUBLISH_DIR or %(default)s)")
    parser.add_argument("--circuits", default=os.environ.get("PUBLISH_CIRCUITS", ""),
                        help='allow-list, "TX->RX, TX->RX" '
                             "(default: $PUBLISH_CIRCUITS)")
    parser.add_argument("--method", default=os.environ.get("PUBLISH_METHOD", "contour"),
                        help="whose picks to plot (default: $PUBLISH_METHOD or "
                             "%(default)s, the method forecasts are trained on)")
    parser.add_argument("--days", type=int,
                        default=int(os.environ.get("PUBLISH_DAYS", "30")),
                        help="history in the series (default: %(default)s)")
    parser.add_argument("--archive-root", type=Path, default=None,
                        help="sounding.path is stored relative to this "
                             "(default: $ARCHIVE_ROOT)")
    parser.add_argument("--interval", type=float, default=0.0,
                        help="seconds between passes; 0 runs once and exits "
                             "(default: %(default)s)")
    args = parser.parse_args(argv)

    circuits = parse_circuits(args.circuits)
    if not circuits:
        print("no circuits to publish: set PUBLISH_CIRCUITS, e.g. "
              '"NIC->DOB, SGO->DOB"', file=sys.stderr)
        return 2
    archive_root = args.archive_root or db.ARCHIVE_ROOT

    while True:
        started = time.time()
        try:
            conn = connect_ro(args.db or db.DEFAULT_DB)
            try:
                result = build_site(conn, args.out, circuits=circuits,
                                    method=args.method, days=args.days,
                                    archive_root=archive_root)
            finally:
                conn.close()
        except Exception as exc:
            # The usual causes -- a locked database, a mount that went away --
            # clear by themselves, and the page keeps the last good snapshot.
            print(f"{db.utcnow()}  pass failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            if args.interval <= 0:
                return 1
        else:
            print(f"{db.utcnow()}  {describe(result)}", flush=True)

        if args.interval <= 0:
            return 0
        time.sleep(max(1.0, args.interval - (time.time() - started)))


if __name__ == "__main__":
    raise SystemExit(main())
