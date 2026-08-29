"""The endpoints the station agent speaks.

**The paths here match ``services/agent/client.py``, not
``architecture.md`` sec. 4.3.** The doc proposed ``POST /health/report``; the
agent that exists and is tested posts to ``/stations/health`` and pulls from
``/stations/{id}/commands``. The agent is the deployed half and the awkward one
to change -- it lives on an acquisition laptop reached over AnyDesk -- so the
server serves what the client speaks and sec. 4.3 has been corrected to match.

All of them require the control scope. A health push is not a read: it is a
station identifying itself, and accepting anonymous pushes would let anyone
fabricate a healthy report for a station that is down -- which under sec. 5.4,
where silence is the alert, is precisely how you would hide an outage.
"""

from __future__ import annotations

import base64
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from . import db
from .auth import Capability, Principal, require

router = APIRouter(tags=["agent"])

#: Largest thumbnail this will store. A 128x96 4-bit PNG is about 3 KB and the
#: agent will not send a bigger one, so this is not a tuning knob -- it is the
#: only bound on how much a holder of the control token can write per request,
#: and `station_preview` is the first table here whose rows are not small by
#: construction. 256 KB leaves room for a future agent that sends a larger
#: picture and still keeps a scripted loop from filling the disk quietly.
MAX_PREVIEW_BYTES = 256 * 1024

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@router.post("/stations/health")
async def push_health(request: Request,
                      _: Principal = Depends(require(Capability.AGENT))) -> dict:
    """Receive one health document.

    The station name comes from the document, not from the URL, because that
    is what the agent sends. Rejecting a nameless document rather than filing
    it under "" keeps an unattributable report out of the station list.
    """
    document = await request.json()
    if not isinstance(document, dict):
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "expected a JSON object")
    station = str(document.get("station") or "").strip()
    if not station:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "document has no `station`")

    report_id = db.store_health(request.app.state.db, station, document)
    return {"ok": True, "report_id": report_id}


@router.post("/stations/{station}/preview")
async def push_preview(station: str, request: Request,
                       _: Principal = Depends(require(Capability.AGENT))) -> dict:
    """Receive one thumbnail of the newest product from one transmitter.

    Its own endpoint rather than a field in the health document: that document
    is stored verbatim and forever, and re-read in full on every console
    render. See ``services/agent/preview.py``.

    The station comes from the URL here, unlike the health push, because there
    is no document to carry it -- the picture is *about* a station the caller
    named in the path.
    """
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "expected a JSON object")
    tx = str(payload.get("tx") or "").strip()
    if not tx:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "preview has no `tx`")

    encoded = payload.get("image_b64") or ""
    # Checked before decoding, not after: base64 is 4/3 of the bytes it holds,
    # so this refuses an oversize body without ever materialising it.
    if len(encoded) > MAX_PREVIEW_BYTES * 4 // 3 + 4:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE,
                            f"preview exceeds {MAX_PREVIEW_BYTES} bytes")
    try:
        image = base64.b64decode(encoded, validate=True)
    except Exception:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "`image_b64` is not base64") from None
    if len(image) > MAX_PREVIEW_BYTES:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE,
                            f"preview exceeds {MAX_PREVIEW_BYTES} bytes")
    # Stored bytes are served straight back as `image/png`, so what is stored
    # has to actually be one. Without this the endpoint is a way to have the
    # server hand a browser arbitrary content under a content type of our
    # choosing.
    if not image.startswith(PNG_MAGIC):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "not a PNG")

    db.save_preview(
        request.app.state.db, station, tx, image,
        t0=_number(payload.get("t0")),
        width=_number(payload.get("width")),
        height=_number(payload.get("height")),
        freq_lo_hz=_number(payload.get("freq_lo_hz")),
        freq_hi_hz=_number(payload.get("freq_hi_hz")),
        range_lo_m=_number(payload.get("range_lo_m")),
        range_hi_m=_number(payload.get("range_hi_m")),
        cropped=bool(payload.get("cropped")),
    )
    return {"ok": True, "bytes": len(image)}


def _number(value: Any) -> float | None:
    """A metadata field, or None. A malformed one must not lose the picture."""
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


@router.get("/stations/{station}/commands")
def pull_commands(station: str, request: Request,
                  _: Principal = Depends(require(Capability.AGENT))) -> dict:
    """Hand out pending work. An empty list is the normal answer."""
    pending = db.take_pending(request.app.state.db, station)
    return {"commands": [
        {"id": c["id"], "name": c["name"],
         "params": _json(c["params"])} for c in pending]}


@router.post("/stations/{station}/commands/{command_id}/ack")
async def acknowledge(station: str, command_id: str, request: Request,
                      _: Principal = Depends(require(Capability.AGENT))) -> dict:
    """Record what a command did.

    Accepted even for an unknown id. The agent has already acted by the time it
    acknowledges, so refusing the ack would make it retry an action that
    already happened -- and a retried "restart acquisition" is worse than the
    bookkeeping gap it would fix.
    """
    payload = await request.json()
    results = payload.get("results") if isinstance(payload, dict) else None
    known = db.acknowledge(request.app.state.db, command_id,
                           results if isinstance(results, list) else [])
    return {"ok": True, "known": known}


def _json(text: str | None) -> Any:
    import json

    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}
