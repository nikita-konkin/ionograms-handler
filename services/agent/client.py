"""Talking to the server: health pushes out, commands pulled in.

``architecture.md`` sec. 5.4, and the direction of each is the whole design:

**Health is pushed.** The agent POSTs on an interval and silence is itself the
alert. No polling to configure, and the station is never a listening service.
That holds on a local network as well as behind NAT -- it costs nothing here
and stays correct if a station is ever remote.

**Commands are pulled.** The agent asks for pending work and acknowledges it.
The station keeps no inbound path, which matters because the thing on the
other end of that path is RF acquisition hardware.

Uses ``urllib`` rather than ``requests``: this runs on the acquisition laptop
alongside the recorder, and a dependency there is a thing that can fail to
install during an incident. The whole transport is small enough not to need
one.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from .config import StationConfig

USER_AGENT = "ionograms-handler-agent/0.1.0"


class TransportError(RuntimeError):
    """The server could not be reached, or refused what was sent."""


@dataclass
class Command:
    """One instruction, as pulled from the server."""

    id: str
    name: str
    params: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict) -> "Command":
        return cls(id=str(data.get("id", "")),
                   name=str(data.get("name", "")),
                   params=dict(data.get("params", {})))


def _request(url: str, *, token: str = "", method: str = "GET",
             payload: dict | None = None, timeout: float = 15.0,
             opener: Callable | None = None) -> dict:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=body, method=method)
    request.add_header("User-Agent", USER_AGENT)
    request.add_header("Accept", "application/json")
    if body is not None:
        request.add_header("Content-Type", "application/json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")

    opener = opener or urllib.request.urlopen
    try:
        with opener(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        raise TransportError(f"{method} {url}: HTTP {exc.code} {exc.reason}") from None
    except urllib.error.URLError as exc:
        raise TransportError(f"{method} {url}: {exc.reason}") from None
    except Exception as exc:
        raise TransportError(f"{method} {url}: {type(exc).__name__}: {exc}") from None

    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TransportError(f"{method} {url}: response was not JSON: {exc}") from None


def push_health(config: StationConfig, report, *, opener=None,
                timeout: float = 15.0) -> dict:
    """POST one health document.

    A push that fails is not an emergency and must not stop the loop: the
    server going away is exactly the situation where the station should keep
    acquiring and keep trying. The caller decides what to do with the raised
    error; :func:`run_once` logs it and carries on.
    """
    if not config.server_url:
        raise TransportError("no server_url configured")
    url = config.server_url.rstrip("/") + "/stations/health"
    return _request(url, token=config.token, method="POST",
                    payload=json.loads(report.to_json()),
                    timeout=timeout, opener=opener)


def pull_commands(config: StationConfig, *, opener=None,
                  timeout: float = 15.0) -> list[Command]:
    """Ask for pending work. An empty list is the normal answer."""
    if not config.server_url:
        raise TransportError("no server_url configured")
    url = (config.server_url.rstrip("/")
           + f"/stations/{config.station}/commands")
    data = _request(url, token=config.token, method="GET",
                    timeout=timeout, opener=opener)
    items = data.get("commands", []) if isinstance(data, dict) else []
    return [Command.from_dict(item) for item in items]


def acknowledge(config: StationConfig, command: Command, results, *,
                opener=None, timeout: float = 15.0) -> dict:
    """Report what a command did, so it is not handed out again.

    Sent whether it succeeded or failed. An unacknowledged failure would be
    redelivered forever, and "restart acquisition" delivered forever is worse
    than the fault it was meant to fix.
    """
    if not config.server_url:
        raise TransportError("no server_url configured")
    url = (config.server_url.rstrip("/")
           + f"/stations/{config.station}/commands/{command.id}/ack")
    payload = {
        "command_id": command.id,
        "name": command.name,
        "results": [r.to_dict() for r in results],
        "acknowledged_at": time.time(),
    }
    return _request(url, token=config.token, method="POST", payload=payload,
                    timeout=timeout, opener=opener)
