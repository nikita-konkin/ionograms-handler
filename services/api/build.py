"""Which code is running, and since when.

Stamped into the image by `deploy/Dockerfile.api` and read once here, so
`/healthz`, the startup log and the header on every page say the same thing.

Three dates, because they answer three different questions and the one an
operator means by "last updated" is not the obvious one:

* ``COMMIT_TIME`` -- when the code itself last changed. What "is the fix I
  made yesterday in there?" is really asking, so it is the date the header
  shows.
* ``BUILD_TIME`` -- when the image was made. Can be much later than the
  commit, and on an overlay build it used to be *earlier*: tesla's local
  rebuild inherited the Docker Hub image's stamp and reported a build two
  days older than the code it ran (2026-09-29).
* ``STARTED_AT`` -- when this process came up. Never stamped, so never stale:
  it is the one date that is right even for a checkout run by hand.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

#: Hand-edited, and it has read 0.1.0 through every deploy -- which is why
#: everything below exists. Kept for the OpenAPI document.
VERSION = "0.1.0"

SHA = os.environ.get("API_BUILD_SHA", "")
BUILD_TIME = os.environ.get("API_BUILD_TIME", "")
COMMIT_TIME = os.environ.get("API_COMMIT_TIME", "")
STARTED_AT = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

#: What a stamp reads as when a build did not set it. Treated as absent.
_UNSET = ("", "unknown", "local build")


def _stamp(value: str) -> str | None:
    return None if value.strip().lower() in _UNSET else value.strip()


def short_sha() -> str | None:
    """Seven characters, like `git log --oneline`. CI passes the full SHA."""
    sha = _stamp(SHA)
    return sha[:7] if sha else None


def _day_and_time(value: str | None) -> str | None:
    """``2026-09-29T07:40:12+03:00`` as ``2026-09-29 04:40Z``: UTC, minutes."""
    if not value:
        return None
    try:
        when = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%MZ")


def info() -> dict:
    """Everything known about this build. ``None`` where nothing was stamped."""
    committed = _day_and_time(_stamp(COMMIT_TIME))
    built = _day_and_time(_stamp(BUILD_TIME))
    return {
        "version": VERSION,
        "sha": short_sha(),
        "committed_at": committed,
        "built_at": built,
        "started_at": _day_and_time(STARTED_AT),
        # The one date the header leads with: the code's own, falling back to
        # the build's when the commit was not stamped, and to the start when
        # neither was -- so there is always an honest answer to "how new".
        "updated": committed or built or _day_and_time(STARTED_AT),
        "updated_is": ("commit" if committed else
                       "build" if built else "start"),
    }


def context(request) -> dict:
    """Jinja context processor: ``build`` on every page."""
    return {"build": info()}
