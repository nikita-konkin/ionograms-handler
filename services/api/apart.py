"""Work in a process of its own, stopped if it runs past its time.

The simulator's numerics are long stretches of Python between array calls.
On a thread of the server they would hold the interpreter its requests run
on, and every page would wait; in a process of their own the server goes on
answering, and work that hangs is stopped rather than left holding its slot.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


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
