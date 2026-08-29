"""Who is asking, and what they may do.

``architecture.md`` sec. 4.3: "Public read of soundings and forecasts must not
share a scope with anything that can stop an acquisition." That began as one
surface with two shared secrets, and this module is what it grew into when a
second person needed access.

**Reads stay open.** ``READ_TOKEN`` unset means anyone may look, which is the
right default for a rig on ``127.0.0.1`` and the wrong one anywhere else, so it
is stated loudly at startup rather than assumed. Accounts gate the *write* half
only.

**Writes are gated by capability, not by role.** An endpoint declares what it
needs -- ``Capability.STATION``, ``Capability.MODEL`` -- and :data:`GRANTS` is
the one place that says which roles hold it. The alternative, testing the role
at each of twenty-nine endpoints, spreads the policy across three modules where
nobody can read it as a whole and where a new endpoint gets whatever its author
was thinking about that afternoon.

**``CONTROL_TOKEN`` still holds everything**, and that is load-bearing in three
separate ways rather than being backward-compatibility slack:

  * The station agent authenticates with it, from an acquisition laptop reached
    over AnyDesk where a redeploy is a manual errand. Its contract is frozen.
  * It is the bootstrap. The first admin account is created with it, so there
    is no seeded default account and no chicken-and-egg.
  * It is the way back in when the accounts table is wrong.

``CONTROL_TOKEN`` unset means control is **refused**, not open. A missing secret
must never be the same as a granted one; the failure mode of getting that
backwards is a stranger stopping acquisition.
"""

from __future__ import annotations

import hmac
import os
from enum import Enum

from fastapi import Header, HTTPException, Request, status

READ_TOKEN = os.environ.get("READ_TOKEN", "")
CONTROL_TOKEN = os.environ.get("CONTROL_TOKEN", "")


class Capability(str, Enum):
    """What an endpoint needs. The vocabulary :data:`GRANTS` is written in."""

    #: Fit, score and manage models. The teaching surface.
    MODEL = "model"
    #: Make a model the live forecast for a circuit, or take it out of service.
    #: Separate from MODEL because promotion is an operational act: it changes
    #: what the station publishes, not what a student is experimenting with.
    PROMOTE = "promote"
    #: Register, scan and delete archives; mute and delete circuits. Destroys
    #: measurements, which is why it is not in with the modelling verbs.
    ARCHIVE = "archive"
    #: Start, stop and reconfigure acquisition. The stop button.
    STATION = "station"
    #: Create and revoke accounts.
    ACCOUNT = "account"
    #: The station agent's own paths -- pushing health and previews, pulling
    #: and acknowledging commands.
    AGENT = "agent"


#: Which roles hold which capabilities. **The whole policy, in one place.**
#:
#: `admin` deliberately does not hold `AGENT`. Those four endpoints are
#: machine-to-machine: a person never needs to post a health report, and
#: withholding it means a leaked admin token cannot forge station telemetry to
#: make a dead receiver look alive. `CONTROL_TOKEN` holds it because that is
#: what the agent actually presents.
GRANTS: dict[str, frozenset[Capability]] = {
    "student": frozenset(),
    "teacher": frozenset({Capability.MODEL}),
    "admin": frozenset({Capability.MODEL, Capability.PROMOTE,
                        Capability.ARCHIVE, Capability.STATION,
                        Capability.ACCOUNT}),
}

#: Every capability, held by `CONTROL_TOKEN` alone. Not `GRANTS["admin"]` --
#: see the note there about `AGENT`.
EVERYTHING = frozenset(Capability)

ROLES = tuple(GRANTS)


class Principal:
    """Who is asking, and what they hold.

    Returned by every capability dependency, so an endpoint that wants to
    record who acted just names the parameter and writes ``who.name``.
    """

    __slots__ = ("name", "role", "capabilities", "source", "id")

    def __init__(self, name: str, role: str,
                 capabilities: frozenset[Capability], source: str,
                 id: int | None = None):
        self.name = name
        self.role = role
        self.capabilities = capabilities
        self.source = source
        self.id = id

    def can(self, capability: Capability) -> bool:
        return capability in self.capabilities

    def __str__(self) -> str:
        # For log lines and error messages only. It is deliberately *not* a
        # safety net for the audit columns: sqlite3 raises on an unrecognised
        # bind type rather than calling str(), so a caller that passes the
        # object where a name belongs fails loudly at the write. That is the
        # behaviour worth having -- the alternative is a column that silently
        # fills with "<Principal object at 0x...>".
        return self.name

    def as_dict(self) -> dict:
        return {"name": self.name, "role": self.role, "source": self.source,
                "capabilities": sorted(c.value for c in self.capabilities)}


#: What an unauthenticated request is. Holds nothing; exists so that `/whoami`
#: and the console have something to describe rather than a null.
ANONYMOUS = Principal("anonymous", "anonymous", frozenset(), "none")

#: What `CONTROL_TOKEN` presents as. Named `control` rather than a person's
#: name because it is a shared secret and claiming otherwise in an audit column
#: would be a lie.
CONTROL = Principal("control", "admin", EVERYTHING, "control")


def _presented(authorization: str | None) -> str:
    if not authorization:
        return ""
    parts = authorization.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return authorization.strip()


def _matches(presented: str, expected: str) -> bool:
    # Constant-time: a token check that returns early on the first wrong byte
    # leaks the token's prefix to anyone willing to time it. Used for the two
    # environment secrets, which are short and human-chosen; an account token
    # is looked up by hash instead and needs no such care.
    return bool(expected) and hmac.compare_digest(presented, expected)


def identify(request: Request, authorization: str | None) -> Principal:
    """Resolve a request to a principal. Never raises.

    Order matters: `CONTROL_TOKEN` is checked first so that the way back in
    keeps working even if the `principal` table is unreadable.
    """
    presented = _presented(authorization)
    if not presented:
        return ANONYMOUS
    if _matches(presented, CONTROL_TOKEN):
        return CONTROL

    from . import db

    conn = getattr(request.app.state, "db", None)
    if conn is None:                           # pragma: no cover - defensive
        return ANONYMOUS
    try:
        row = db.principal_by_token(conn, presented)
    except Exception:                          # pragma: no cover - defensive
        # An accounts table that cannot be read must not take the service
        # down; it degrades to "this token is not recognised".
        return ANONYMOUS
    if row is None:
        return ANONYMOUS
    db.touch_principal(conn, row)
    return Principal(row["name"], row["role"],
                     GRANTS.get(row["role"], frozenset()), "token",
                     id=row["id"])


def current(request: Request,
            authorization: str | None = Header(default=None)) -> Principal:
    """The principal behind this request, whoever it is. For `/whoami`."""
    return identify(request, authorization)


def require(capability: Capability):
    """A dependency that admits only holders of ``capability``.

    Three refusals, and they are deliberately different codes because they call
    for different actions:

      * **503** -- no ``CONTROL_TOKEN`` is configured at all, so nothing can
        ever be granted. The operator has to set one; retrying will not help.
      * **401** -- nothing was presented, or what was presented is not
        recognised. Sign in.
      * **403** -- recognised, but this role does not hold the capability.
        Signing in again changes nothing; someone has to grant it.

    Collapsing 403 into 401 would tell a student their token was wrong when it
    was perfectly good, and send them to re-paste it forever.
    """

    def dependency(request: Request,
                   authorization: str | None = Header(default=None)) -> Principal:
        if not CONTROL_TOKEN:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "no CONTROL_TOKEN configured; control is disabled. Set one in "
                "the environment -- an unset secret is not an open door.")
        who = identify(request, authorization)
        if who is ANONYMOUS:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "a token is required for this. Paste yours on the console, or "
                "ask an admin for an account.",
                headers={"WWW-Authenticate": "Bearer"})
        if not who.can(capability):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"{who.name} is a {who.role}, and this needs "
                f"'{capability.value}'. Your token is fine -- the role is what "
                f"is short. An admin can change it.")
        return who

    return dependency


def require_read(request: Request,
                 authorization: str | None = Header(default=None)) -> str:
    """Reads, when `READ_TOKEN` closes them.

    **A named account reads too.** There is one `Authorization` header and a
    person has one token, so if this compared only against `READ_TOKEN` then
    setting it would lock every account holder out of the whole console --
    including `/whoami`, which the console asks before it can render at all. A
    student who may write nothing may certainly look.

    So the check is "any recognised identity": the read token, an active
    account, or `CONTROL_TOKEN`. Latent while `READ_TOKEN` is unset, which is
    every deployment so far, and the reason it is written down here rather than
    discovered on the day someone closes reads.
    """
    if not READ_TOKEN:
        return "anonymous"
    presented = _presented(authorization)
    if _matches(presented, READ_TOKEN):
        return "read"
    who = identify(request, authorization)
    if who is not ANONYMOUS:
        return who.name
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "read token required",
                        headers={"WWW-Authenticate": "Bearer"})


def describe() -> str:
    """One line for the startup banner, so the posture is never a surprise."""
    read = "token required" if READ_TOKEN else "OPEN (no READ_TOKEN set)"
    control = "token required" if CONTROL_TOKEN else "DISABLED (no CONTROL_TOKEN set)"
    return f"auth: read {read}; control {control}"


def describe_accounts(conn) -> str:
    """A second banner line: how many accounts exist, by role.

    Worth saying at startup because "control is token-required" reads as
    reassuring while meaning nobody but the shared secret can do anything.
    """
    from . import db

    try:
        found = db.principals(conn)
    except Exception:                          # pragma: no cover - defensive
        return "accounts: unreadable"
    live = [p for p in found if not p["disabled_at"]]
    if not live:
        return ("accounts: none yet; CONTROL_TOKEN is the only writer. "
                "POST /principals to add one.")
    counts = {role: sum(1 for p in live if p["role"] == role) for role in ROLES}
    shown = ", ".join(f"{n} {role}" for role, n in counts.items() if n)
    disabled = len(found) - len(live)
    return (f"accounts: {shown}"
            + (f" ({disabled} disabled)" if disabled else ""))
