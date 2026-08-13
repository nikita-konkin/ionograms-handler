"""Can this host still reach the servers the solar indices come from?

The IRI values on a sounding page are only as current as the indices behind
them, and those come from three public servers. That dependency is invisible
in the output: with no network, :mod:`muf.reference.indices` falls back to its
cache and keeps answering, and a driver from six months ago renders exactly
like a fresh one. Nothing on the page distinguishes them. This module exists
so something does.

Three decisions, each of which is the point rather than an implementation
detail:

**It probes those hosts, not "the internet".** A ping to a public resolver
answers a question nobody asked: it stays green behind a proxy that blocks
HTTPS to ``sidc.be``, and it goes red on a host that reaches every source it
needs through an internal mirror. The list comes from
:data:`muf.reference.indices.SOURCES`, so a source added there is probed here
without anyone remembering to.

**Reachability and freshness are reported separately.** They fail
independently and mean opposite things. Unreachable with a fresh cache is a
model still answering correctly; reachable with no cache is a model that has
never run. Collapsing them into one light would hide whichever failed second.

**Lightweight is a constraint, not an aspiration.** One ``HEAD`` per *host*,
not per file -- reachability is a property of the host, and three requests
answer what six would. They run concurrently, on a short timeout, in a daemon
thread, and no request path ever waits on one: :func:`current` returns the
last result or ``unknown``, and never blocks. At the default interval that is
three HEADs every ten minutes, which is less traffic than one page load.
"""

from __future__ import annotations

import os
import socket
import ssl
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from muf.reference import indices

#: Off by default nowhere, but off by request somewhere: a station on a
#: deliberately isolated network does not need three failing probes every ten
#: minutes to tell it what its operator already knows.
ENABLED = os.environ.get("NET_CHECK", "1") not in ("0", "", "false")

#: How often the background thread re-probes. Reachability changes -- a
#: firewall rule, a dropped uplink -- which is why this loops where the census
#: warm-up runs once: a census of write-once files cannot go stale, and this
#: can.
INTERVAL_S = float(os.environ.get("NET_CHECK_INTERVAL_S", "600"))

#: Short on purpose. This is a liveness probe, not a download: a host that
#: takes longer than this to answer a HEAD is not going to serve a 1.4 MB
#: index file usefully either, and the answer "slow" is the same as the answer
#: "down" for anything the page will say.
TIMEOUT_S = float(os.environ.get("NET_TIMEOUT_S", "4"))

#: Past this, the last result is history rather than a reading. Two intervals
#: plus a margin, so one skipped cycle does not flip the pill.
STALE_AFTER_S = INTERVAL_S * 2 + 60


@dataclass(frozen=True)
class Probe:
    """One host, and what happened when it was asked."""

    host: str
    url: str
    #: ``True`` reachable, ``False`` refused, ``None`` never asked. The same
    #: tri-state the health metrics use, and for the same reason: "we could
    #: not measure it" must not read as "it is fine".
    ok: bool | None
    ms: float | None = None
    detail: str = ""
    #: Sources served by this host, so a failure names what it costs.
    provides: tuple[str, ...] = ()


@dataclass(frozen=True)
class Reachability:
    """The answer the console panel renders."""

    state: str                      # online | degraded | offline | unknown
    detail: str
    probes: tuple[Probe, ...] = ()
    checked_at: float | None = None
    #: Cache age in seconds per source key, ``None`` when never fetched.
    cache: dict[str, float | None] = field(default_factory=dict)

    @property
    def is_online(self) -> bool:
        return self.state == "online"

    @property
    def age_s(self) -> float | None:
        return None if self.checked_at is None else time.time() - self.checked_at

    def as_dict(self) -> dict:
        return {
            "state": self.state,
            "detail": self.detail,
            "checked_at": self.checked_at,
            "age_s": self.age_s,
            "hosts": [{"host": p.host, "ok": p.ok, "ms": p.ms,
                       "detail": p.detail, "provides": list(p.provides)}
                      for p in self.probes],
            "cache": self.cache,
        }


UNKNOWN = Reachability(
    "unknown", "not checked yet; the first probe runs a moment after startup")


def hosts() -> list[tuple[str, str, tuple[str, ...]]]:
    """``(host, representative url, source keys)`` -- one entry per host.

    One URL stands for its host because that is the granularity of the answer.
    A moved file would slip past this; a moved file is a bug in the source
    list, which the next fetch reports with the URL in the message, and not
    the thing a connectivity light is for.
    """
    seen: dict[str, tuple[str, list[str]]] = {}
    for source in indices.SOURCES:
        url, keys = seen.setdefault(source.host, (source.url, []))
        keys.append(source.key)
    return [(host, url, tuple(keys)) for host, (url, keys) in seen.items()]


def probe(url: str, *, timeout: float = TIMEOUT_S) -> tuple[bool, float, str]:
    """HEAD ``url``. Returns ``(ok, milliseconds, detail)`` and never raises.

    The failure kinds are kept apart because they point at different repairs.
    A resolver failure is DNS or ``/etc/resolv.conf``; a timeout is a firewall
    dropping rather than refusing; a certificate error is a TLS-inspecting
    proxy whose CA this container does not trust -- which looks like "no
    internet" to a ping and is fixed by installing a certificate, not by
    calling the network team.
    """
    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": indices.USER_AGENT})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            ms = (time.perf_counter() - started) * 1000
            return True, ms, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        # An answer, from the host we asked. The file may have moved, but the
        # path to the server is demonstrably open, which is the question.
        ms = (time.perf_counter() - started) * 1000
        return True, ms, f"HTTP {exc.code} (reachable; file may have moved)"
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, socket.gaierror):
            detail = f"cannot resolve the name ({reason.strerror or reason})"
        elif isinstance(reason, ssl.SSLError):
            detail = f"TLS refused ({reason.reason or reason}) -- proxy CA?"
        elif isinstance(reason, (TimeoutError, socket.timeout)):
            detail = f"no answer within {timeout:.0f}s"
        else:
            detail = str(reason)
        return False, (time.perf_counter() - started) * 1000, detail
    except (TimeoutError, socket.timeout):
        return False, (time.perf_counter() - started) * 1000, \
            f"no answer within {timeout:.0f}s"
    except Exception as exc:                                  # noqa: BLE001
        # A connectivity indicator that can itself take the page down would be
        # worse than no indicator, so the net is cast wide on purpose.
        return False, (time.perf_counter() - started) * 1000, \
            f"{type(exc).__name__}: {exc}"


def check(*, timeout: float = TIMEOUT_S, cache_dir: Path | None = None,
          probe_fn=None) -> Reachability:
    """Probe every host concurrently and fold the results into one state."""
    run = probe_fn or probe
    targets = hosts()
    results: dict[str, tuple[bool, float, str]] = {}

    def one(host: str, url: str) -> None:
        results[host] = run(url, timeout=timeout)

    threads = [threading.Thread(target=one, args=(host, url), daemon=True,
                                name=f"net-probe-{host}")
               for host, url, _ in targets]
    for thread in threads:
        thread.start()
    # Every probe is already bounded by `timeout`; the margin covers thread
    # start-up so a join cannot outlast the probe it is waiting for.
    for thread in threads:
        thread.join(timeout + 2)

    probes = []
    for host, url, keys in targets:
        ok, ms, detail = results.get(host, (None, None, "probe did not finish"))
        probes.append(Probe(host=host, url=url, ok=ok, ms=ms, detail=detail,
                            provides=keys))

    reached = [p for p in probes if p.ok]
    if not probes:
        state, detail = "unknown", "no sources configured"
    elif len(reached) == len(probes):
        slowest = max(p.ms or 0 for p in probes)
        state = "online"
        detail = (f"all {len(probes)} index hosts answered, slowest "
                  f"{slowest:.0f} ms")
    elif reached:
        down = ", ".join(p.host for p in probes if not p.ok)
        state = "degraded"
        detail = (f"{len(reached)} of {len(probes)} index hosts answered; "
                  f"no route to {down}")
    else:
        state = "offline"
        detail = ("no index host answered -- IRI will run on whatever is "
                  "cached, or not at all")

    return Reachability(state=state, detail=detail, probes=tuple(probes),
                        checked_at=time.time(),
                        cache=indices.cache_state(cache_dir))


# --- the memoised view the pages read ---------------------------------------

_LOCK = threading.Lock()
_LAST: Reachability = UNKNOWN


def current() -> Reachability:
    """The last result, without blocking and without probing.

    A result older than :data:`STALE_AFTER_S` comes back as ``unknown`` rather
    than as its own last state. The refresher is a daemon thread and a daemon
    thread can die; showing its final reading indefinitely would turn a dead
    checker into a permanently green light, which is the exact failure this
    module was written to make visible.
    """
    with _LOCK:
        last = _LAST
    if last.checked_at is None:
        return last
    age = time.time() - last.checked_at
    if age > STALE_AFTER_S:
        return Reachability(
            "unknown",
            f"last checked {age / 60:.0f} min ago and the checker has not run "
            f"since; treat this as no reading, not as a result",
            probes=last.probes, checked_at=last.checked_at, cache=last.cache)
    return last


def refresh(**kwargs) -> Reachability:
    """Probe now and store the result. Never raises."""
    global _LAST
    try:
        got = check(**kwargs)
    except Exception as exc:                                  # noqa: BLE001
        got = Reachability("unknown", f"reachability check failed: {exc!r}",
                           checked_at=time.time())
    with _LOCK:
        _LAST = got
    return got


def reset() -> None:
    """Forget the last result. For tests, which must not inherit each other's."""
    global _LAST
    with _LOCK:
        _LAST = UNKNOWN


def watch(*, interval: float = INTERVAL_S, stop: threading.Event | None = None,
          **kwargs) -> None:
    """Refresh forever. Run this in a daemon thread, never on a request."""
    while True:
        got = refresh(**kwargs)
        print(f"  net: {got.state} -- {got.detail}")
        if stop is not None and stop.wait(interval):
            return
        if stop is None:
            time.sleep(interval)


def start(*, interval: float = INTERVAL_S) -> threading.Thread | None:
    """Start the refresher, or ``None`` when checking is switched off.

    Daemon, for the reason the census warm-up is: this thread owns nothing but
    a cached reading, so losing it at shutdown costs nothing, and a slow
    unreachable host must not be able to hold a container open.
    """
    if not ENABLED:
        return None
    thread = threading.Thread(target=watch, kwargs={"interval": interval},
                              daemon=True, name="net-check")
    thread.start()
    return thread
