"""Two scopes, deliberately separate.

``architecture.md`` sec. 4.3: "Public read of soundings and forecasts must not
share a scope with anything that can stop an acquisition." One surface, two
tokens -- because the read half is the part anyone might reasonably want to
share, and the write half can silence a radio.

``READ_TOKEN`` unset means reads are open. That is the right default for a
rig on ``127.0.0.1`` and the wrong one anywhere else, so it is stated loudly
at startup rather than assumed.

``CONTROL_TOKEN`` unset means control is **refused**, not open. A missing
secret must never be the same as a granted one; the failure mode of getting
that backwards is a stranger stopping acquisition.
"""

from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException, status

READ_TOKEN = os.environ.get("READ_TOKEN", "")
CONTROL_TOKEN = os.environ.get("CONTROL_TOKEN", "")


def _presented(authorization: str | None) -> str:
    if not authorization:
        return ""
    parts = authorization.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return authorization.strip()


def _matches(presented: str, expected: str) -> bool:
    # Constant-time: a token check that returns early on the first wrong byte
    # leaks the token's prefix to anyone willing to time it.
    return bool(expected) and hmac.compare_digest(presented, expected)


def require_read(authorization: str | None = Header(default=None)) -> str:
    if not READ_TOKEN:
        return "anonymous"
    if _matches(_presented(authorization), READ_TOKEN):
        return "read"
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "read token required",
                        headers={"WWW-Authenticate": "Bearer"})


def require_control(authorization: str | None = Header(default=None)) -> str:
    """Control never falls open.

    An unset CONTROL_TOKEN refuses every request rather than admitting them.
    The station is on the other end of these endpoints.
    """
    if not CONTROL_TOKEN:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "no CONTROL_TOKEN configured; control is disabled. Set one in the "
            "environment -- an unset secret is not an open door.")
    if _matches(_presented(authorization), CONTROL_TOKEN):
        return "control"
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "control token required",
                        headers={"WWW-Authenticate": "Bearer"})


def describe() -> str:
    """One line for the startup banner, so the posture is never a surprise."""
    read = "token required" if READ_TOKEN else "OPEN (no READ_TOKEN set)"
    control = "token required" if CONTROL_TOKEN else "DISABLED (no CONTROL_TOKEN set)"
    return f"auth: read {read}; control {control}"
