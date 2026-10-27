"""Work in a process of its own, stopped if it runs past its time.

The simulator's numerics are long stretches of Python between array calls.
On a thread of the server they would hold the interpreter its requests run
on, and every page would wait; in a process of their own the server goes on
answering, and work that hangs is stopped rather than left holding its slot.

A process of its own cannot answer how far it has got, so it says so in a
file instead (:class:`Progress`), which the server reads (:func:`progress`).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Progress:
    """How far a piece of work has got, kept in ``path`` for the server to read.

    Made where the work is queued, and called where it runs, in its own
    process: ``progress(done, total)`` after each step. ``started`` is the
    epoch the work began, so that the reading says how long it has run.

    On Windows a file being read cannot be replaced. A step written while
    the server reads is lost rather than the work stopped; the next step
    writes it again.
    """

    path: Path
    started: float

    def __call__(self, done: int, total: int) -> None:
        partial = Path(self.path).with_suffix(".partial-progress")
        partial.write_text(json.dumps({"done": int(done), "total": int(total),
                                       "started": self.started}), encoding="utf-8")
        try:
            os.replace(partial, self.path)
        except PermissionError:
            partial.unlink(missing_ok=True)


def progress(path: Path) -> dict | None:
    """The work's ``done`` and ``total`` steps and its ``elapsed_s``, as it last
    wrote them to ``path``; None before it has written any."""
    try:
        got = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return {"done": got["done"], "total": got["total"],
            "elapsed_s": round(time.time() - got["started"], 1)}


def run(fn: Callable[..., Any], *args, timeout_s: float, what: str) -> Any:
    """``fn(*args)`` in a process of its own: its value, or the error it raised.

    ``fn`` and ``args`` must pickle, as the process is spawned. ``what`` names
    the work in the errors: a TimeoutError past ``timeout_s``, and a
    RuntimeError for a failure in ``fn`` or a process that died.
    """
    from muf.pipeline import pool_context

    context = pool_context()
    receive, send = context.Pipe(duplex=False)
    child = context.Process(target=_child, args=(fn, args, what, send), daemon=True)
    child.start()
    send.close()
    try:
        if not receive.poll(timeout_s):
            raise TimeoutError(f"{what} ran past {timeout_s:g} s")
        ok, value = receive.recv()
    except EOFError:
        raise RuntimeError(f"{what}: its process died") from None
    finally:
        if child.is_alive():
            child.terminate()
        child.join(10)
    if not ok:
        raise RuntimeError(value)
    return value


def _child(fn: Callable[..., Any], args: tuple, what: str, send) -> None:
    try:
        send.send((True, fn(*args)))
    except Exception as exc:
        send.send((False, f"{what}: {type(exc).__name__}: {exc}"))
    finally:
        send.close()
