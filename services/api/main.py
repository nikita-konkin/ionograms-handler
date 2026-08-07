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
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from . import agent_routes, auth, control_routes, db, read_routes, web_routes

VERSION = "0.1.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = db.init(db.connect())
    app.state.archive_root = Path(os.environ.get("ARCHIVE_ROOT", "."))
    print(f"api {VERSION}  db={db.DEFAULT_DB}  archive={app.state.archive_root}")
    print(f"  {auth.describe()}")
    if not auth.READ_TOKEN:
        print("  READ_TOKEN is unset: reads are open. Correct for a rig on "
              "127.0.0.1, wrong anywhere a station can reach.")
    yield
    app.state.db.close()


app = FastAPI(title="ionograms-handler api", version=VERSION,
              lifespan=lifespan)

app.include_router(agent_routes.router)
app.include_router(control_routes.router)
app.include_router(read_routes.router)
app.include_router(web_routes.router)


@app.get("/healthz", include_in_schema=False)
def healthz() -> dict:
    """Liveness for the container runtime.

    Unauthenticated on purpose, and it deliberately reports nothing about any
    station -- it answers "is this process up", which is the only question a
    restart policy should be asking.
    """
    return {"ok": True, "version": VERSION}


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/ui")
