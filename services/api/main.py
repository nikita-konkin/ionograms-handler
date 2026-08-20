"""The api service: ``uvicorn services.api.main:app``.

One network-facing surface (``architecture.md`` sec. 4.3) with two auth
scopes. Read is soundings, series, health views and rendered ionograms;
control is the station agent's own endpoints plus queueing a command.

**Binds to 127.0.0.1 unless told otherwise.** This process can stop
acquisition on a radio. Reaching it from the sounding laptop means binding to
a LAN address, which is a decision an operator makes on purpose -- so it is an
environment variable with a loud banner, not a default.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import agent_routes, archive_routes, archives, auth, control_routes
from . import db, net, read_routes, sources
from . import web_routes

VERSION = "0.1.0"

#: The commit this image was built from, stamped in by `deploy/Dockerfile.api`.
#:
#: `VERSION` is a hand-edited string and has read 0.1.0 through every deploy so
#: far, which made "is the fix on the server?" a question two four-minute page
#: loads could not answer. This one changes on its own. Empty outside a build,
#: where the answer is whatever the checkout says.
BUILD_SHA = os.environ.get("API_BUILD_SHA", "")
BUILD_TIME = os.environ.get("API_BUILD_TIME", "")

#: Read the archive once at startup instead of making the first visitor do it.
#: Set to 0 where the archive is huge or absent and the census is not wanted.
WARM_CENSUS = os.environ.get("CENSUS_WARM", "1") not in ("0", "", "false")


def _warm(app: FastAPI) -> None:
    """The cold census, off the request path. Runs in a daemon thread."""
    started = time.perf_counter()
    # Announced before it starts, not only when it finishes. The census holds
    # a lock, so while it runs every request for the sources page waits on it;
    # a log that only speaks on success makes "still reading" and "died in the
    # thread" the same silence, which is what happened on DOB.
    print(f"  census warm: reading up to {sources.DEFAULT_MAX_FILES} "
          f"detection file(s) under {app.state.archive_root}", flush=True)
    got = sources.warm(app.state.archive_root)
    if got is None:
        print(f"  census warm: failed after "
              f"{time.perf_counter() - started:.1f}s (see warning above)",
              flush=True)
        return
    cost = got.get("cost", {})
    capped = (f" (newest of {cost['found']})"
              if cost.get("capped") else "")
    print(f"  census warm: {cost.get('files', 0)} file(s){capped}, "
          f"{got.get('count', 0)} emitter(s) in "
          f"{time.perf_counter() - started:.1f}s", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = db.init(db.connect())
    app.state.archive_root = Path(os.environ.get("ARCHIVE_ROOT", "."))
    print(f"api {VERSION}{' ' + BUILD_SHA if BUILD_SHA else ''}  "
          f"db={db.DEFAULT_DB}  archive={app.state.archive_root}")
    print(f"  {auth.describe()}")
    if not auth.READ_TOKEN:
        print("  READ_TOKEN is unset: reads are open. Correct for a rig on "
              "127.0.0.1, wrong anywhere a station can reach.")
    # Whether this host can still refresh the solar indices IRI runs on. Off
    # the request path for the same reason the census is, but on a loop rather
    # than once: a census of write-once files cannot go stale and a route to a
    # third party can.
    app.state.net_check = net.start()

    # Keeps the registered archives current. Off the request path for the
    # same reason the census is, and on a loop rather than once because an
    # archive, unlike a census of write-once files, gains files while the
    # server runs. ARCHIVE_SCAN_INTERVAL_S=0 turns it off.
    app.state.archive_scan = archives.start_periodic(app)

    app.state.census_warm = None
    if WARM_CENSUS:
        # Daemon, so a slow archive cannot hold up a shutdown: this thread
        # owns nothing but a cache, and losing it costs the next reader one
        # cold read rather than leaving anything half-written. Kept on
        # `app.state` so it can be waited on -- a fire-and-forget thread is
        # one a test can only observe by sleeping and hoping.
        app.state.census_warm = threading.Thread(
            target=_warm, args=(app,), daemon=True, name="census-warm")
        app.state.census_warm.start()
    yield
    app.state.db.close()


app = FastAPI(title="ionograms-handler api", version=VERSION,
              lifespan=lifespan)

# Everything this service sends is text over a link that is often the station's
# slow one, and text of a peculiarly compressible kind: tables of numbers and a
# plotting library. Measured on the real archive, gzip takes the vendored
# plotly bundle from 1008 to 348 KB, a 500-sounding JSON listing from 237 to
# 7.4 KB, and the all-circuits series page from 122 to 22 KB. `minimum_size`
# keeps it off the small health responses, where a compressed frame would be
# larger than the answer.
app.add_middleware(GZipMiddleware, minimum_size=1000)

app.include_router(agent_routes.router)
app.include_router(archive_routes.router)
app.include_router(control_routes.router)
app.include_router(read_routes.router)
app.include_router(web_routes.router)

# The one vendored asset: plotly.min.js, served from the image. Not a CDN,
# because the station this serves is on a link that has already been down for
# a week at a time, and a plot that needs the internet to draw a file already
# on disk is a plot that stops working exactly when someone needs it. It is
# also mounted *outside* `require_read`, like any other static asset -- there
# is nothing about this archive in a plotting library.
app.mount("/static",
          StaticFiles(directory=str(Path(__file__).parent / "static")),
          name="static")


@app.get("/healthz", include_in_schema=False)
def healthz() -> dict:
    """Liveness for the container runtime.

    Unauthenticated on purpose, and it deliberately reports nothing about any
    station -- it answers "is this process up", which is the only question a
    restart policy should be asking.

    It does answer one more: *which build* is up. That is not station data and
    it is already visible to anyone who can reach the port, but without it the
    only way to tell a deployed fix from an undeployed one is to look for its
    effects.
    """
    return {"ok": True, "version": VERSION,
            "build": BUILD_SHA or "source", "built_at": BUILD_TIME or None}


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/ui")
