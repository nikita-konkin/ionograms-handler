"""Station agent.

This module is deliberately written for Python 2.7/3.5 syntax -- no f-strings,
no `from __future__ import annotations`, no dataclasses -- while the rest of
the package needs 3.7. That is not an inconsistency; it is the entire point.

`runpy` imports the package before it compiles `__main__.py`, so this is the
only file that gets to run on a too-old interpreter. Without the check below,
`python3 -m services.agent health` on the acquisition laptop dies with

    File ".../services/agent/__main__.py", line 32
        print(f"{config.station}  {verdict}"
                                           ^
    SyntaxError: invalid syntax

which names neither the requirement nor the interpreter that would satisfy it,
and under `Restart=always` repeats every 30 seconds forever. The station's
/usr/bin/python3 is 3.5 (Ubuntu 16.04); the sounder's own virtualenv is 3.8.
"""

import sys

#: `from __future__ import annotations` (PEP 563) and `dataclasses` both land
#: in 3.7. Syntax alone would pass on 3.6 -- the floor is a runtime one.
MIN_PYTHON = (3, 7)

if sys.version_info < MIN_PYTHON:
    sys.exit(
        "services.agent needs Python %d.%d or newer; this interpreter is %s.\n"
        "  %s\n"
        "On an acquisition laptop the sounder's own virtualenv is normally the\n"
        "right one -- e.g. ~/chirpsounder2/.venv38/bin/python -- and\n"
        "chirp-agent.service must point ExecStart at that same interpreter,\n"
        "not at /usr/bin/python3." % (
            MIN_PYTHON[0], MIN_PYTHON[1],
            ".".join(str(part) for part in sys.version_info[:3]),
            sys.executable))
