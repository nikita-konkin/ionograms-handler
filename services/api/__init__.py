"""The api service: one network-facing surface, two auth scopes.

``architecture.md`` sec. 4.3 and sec. 5.4. Read is soundings, MUF series,
health views and rendered ionograms; control is the station agent's own
endpoints plus queueing a command.

Deliberately small and deliberately temporary: SQLite, stdlib SQL, Jinja
templates with no build step. It exists to prove the agent loop and to look at
the archive, not to be a platform.
"""

from __future__ import annotations

__all__ = ["db", "ingest"]
