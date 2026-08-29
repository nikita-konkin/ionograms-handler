"""Accounts, roles and what each one may do.

The load-bearing tests here are the two that would be dangerous to get wrong:

  * **`CONTROL_TOKEN` still holds everything.** The station agent presents it
    from an acquisition laptop reached over AnyDesk, where a redeploy is a
    manual errand. If capabilities ever stop admitting it, acquisition
    telemetry stops arriving and the first symptom is a station that looks
    dead.
  * **A role cannot quietly gain a capability.** The matrix is driven off
    `auth.GRANTS` itself, so adding a capability without deciding who holds it
    fails here rather than defaulting to somebody.

Everything else -- rotation, disabling, the audit trail -- follows from those.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")

from services.api import auth, db                  # noqa: E402

CTL = {"Authorization": "Bearer ctl"}


@pytest.fixture
def api_db(client, tmp_path):
    """A writable handle on the database `client` reads.

    Defined here rather than imported from `test_api`: the `conn` fixture in
    conftest points at a different file, so seeding through it would make every
    assertion in this module pass or fail for the wrong reason.
    """
    with db.session(tmp_path / "api.sqlite3") as conn:
        yield conn


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def account(client, name: str, role: str) -> str:
    """Create an account through the api and return its token, once."""
    r = client.post("/principals", json={"name": name, "role": role},
                    headers=CTL)
    assert r.status_code == 201, r.text
    return r.json()["token"]


# --------------------------------------------------------------------------
# The policy itself
# --------------------------------------------------------------------------

def test_every_capability_is_decided_for_every_role():
    """A new capability must not default into somebody's hands.

    `GRANTS` is the whole policy, so a capability nobody was asked about is a
    capability whose holder is whatever the enum ordering produced. This is the
    test that makes adding one a decision.
    """
    for role, held in auth.GRANTS.items():
        assert held <= auth.EVERYTHING, f"{role} holds something undefined"
    decided = set().union(*auth.GRANTS.values())
    undecided = auth.EVERYTHING - decided - {auth.Capability.AGENT}
    assert not undecided, (
        f"no role holds {sorted(c.value for c in undecided)}; either grant it "
        f"or document it beside AGENT as machine-only")


def test_a_student_holds_nothing_and_an_admin_stops_short_of_the_agent():
    assert auth.GRANTS["student"] == frozenset()
    assert auth.GRANTS["teacher"] == frozenset({auth.Capability.MODEL})
    # Machine-to-machine. A person never posts a health report, and withholding
    # it means a leaked admin token cannot forge telemetry to make a dead
    # receiver look alive.
    assert auth.Capability.AGENT not in auth.GRANTS["admin"]
    assert auth.Capability.AGENT in auth.EVERYTHING


# --------------------------------------------------------------------------
# CONTROL_TOKEN keeps everything
# --------------------------------------------------------------------------

def test_the_control_token_holds_every_capability():
    assert auth.CONTROL.capabilities == auth.EVERYTHING
    for capability in auth.Capability:
        assert auth.CONTROL.can(capability)


def test_the_station_agent_still_reaches_all_four_of_its_paths(client):
    """Its contract is frozen: a redeploy is a manual errand over AnyDesk."""
    from tests.test_api import report

    assert client.post("/stations/health", json=report(),
                       headers=CTL).status_code == 200
    assert client.get("/stations/SIM/commands", headers=CTL).status_code == 200


def test_an_admin_account_cannot_forge_station_telemetry(client):
    """The one thing CONTROL_TOKEN does that an admin deliberately cannot."""
    from tests.test_api import report

    token = account(client, "root", "admin")
    r = client.post("/stations/health", json=report(), headers=bearer(token))
    assert r.status_code == 403
    assert "agent" in r.json()["detail"]


# --------------------------------------------------------------------------
# Roles against the real endpoints
#
# One endpoint per capability group, driven through HTTP rather than by
# inspecting `GRANTS`: the map being right and the decorators using it are
# different claims, and only the second one protects the stop button.
# --------------------------------------------------------------------------

#: (capability, method, path, body) -- one representative per group.
SURFACE = [
    (auth.Capability.MODEL, "post", "/models/train", {}),
    (auth.Capability.PROMOTE, "post", "/models/1/activate", None),
    (auth.Capability.ARCHIVE, "post", "/circuits/mute", {"tx": "unkown"}),
    (auth.Capability.STATION, "post", "/stations/SIM/commands",
     {"name": "stop"}),
    (auth.Capability.ACCOUNT, "post", "/principals",
     {"name": "x", "role": "student"}),
]


@pytest.mark.parametrize("role", ["student", "teacher", "admin"])
def test_each_role_is_admitted_exactly_where_the_policy_says(client, role):
    token = account(client, f"person-{role}", role)
    for capability, method, path, body in SURFACE:
        r = getattr(client, method)(path, json=body, headers=bearer(token))
        allowed = capability in auth.GRANTS[role]
        if allowed:
            # Past the gate. What it does then is another test's business --
            # a 400 or 404 here still means the capability was honoured.
            assert r.status_code != 403, f"{role} was refused {capability.value}"
        else:
            assert r.status_code == 403, (
                f"{role} got {r.status_code} on {capability.value}, "
                f"expected 403")


def test_a_teacher_may_train_but_not_promote_or_touch_the_radio(client):
    """The role boundary, spelled out where a reader will look for it."""
    token = account(client, "teacher-anna", "teacher")

    # Past the capability gate; the spec is then vetted on its own merits.
    assert client.post("/models/train", json={}, headers=bearer(token)
                       ).status_code != 403
    for path, body in (("/models/1/activate", None),
                       ("/archives", {"name": "x", "relpath": "y"}),
                       ("/stations/SIM/commands", {"name": "stop"})):
        r = client.post(path, json=body, headers=bearer(token))
        assert r.status_code == 403, f"{path} admitted a teacher"


def test_a_refusal_says_whether_the_token_or_the_role_is_short(client):
    """403 and 401 call for different actions, so they are different codes.

    Collapsing them tells a student their token is wrong when it is perfectly
    good, and sends them to re-paste it forever.
    """
    token = account(client, "student-boris", "student")

    lacks = client.post("/models/train", json={}, headers=bearer(token))
    assert lacks.status_code == 403
    assert "role is what" in lacks.json()["detail"]

    unknown = client.post("/models/train", json={},
                          headers=bearer("iono_nonsense"))
    assert unknown.status_code == 401


def test_control_stays_refused_when_its_secret_is_missing(client, monkeypatch):
    """An unset secret is not an open door, accounts or no accounts."""
    token = account(client, "root2", "admin")
    monkeypatch.setattr(auth, "CONTROL_TOKEN", "")
    r = client.post("/models/train", json={}, headers=bearer(token))
    assert r.status_code == 503
    assert "not an open door" in r.json()["detail"]


# --------------------------------------------------------------------------
# The token
# --------------------------------------------------------------------------

def test_the_plaintext_token_never_reaches_the_database(client, api_db):
    r = client.post("/principals", json={"name": "vera", "role": "teacher"},
                    headers=CTL)
    token = r.json()["token"]
    assert token.startswith("iono_")

    row = db.one(api_db, "SELECT * FROM principal WHERE name = 'vera'")
    assert token not in str(dict(row))
    assert row["token_sha256"] == db.token_digest(token)


def test_a_token_is_shown_once_and_never_again(client):
    account(client, "vera", "teacher")
    listed = client.get("/principals", headers=CTL).json()["principals"]
    body = str(listed)
    assert "token" not in body and "sha256" not in body


def test_rotating_invalidates_the_old_token_in_the_same_call(client, api_db):
    """No instant in which both work -- that is what rotation has to mean."""
    old = account(client, "vera", "teacher")
    row = db.one(api_db, "SELECT id FROM principal WHERE name = 'vera'")

    new = client.post(f"/principals/{row['id']}/rotate",
                      headers=CTL).json()["token"]
    assert new != old
    assert client.post("/models/train", json={},
                       headers=bearer(old)).status_code == 401
    assert client.post("/models/train", json={},
                       headers=bearer(new)).status_code != 401


def test_a_disabled_account_is_refused_and_can_be_reinstated(client, api_db):
    token = account(client, "vera", "teacher")
    row = db.one(api_db, "SELECT id FROM principal WHERE name = 'vera'")

    client.delete(f"/principals/{row['id']}", headers=CTL)
    assert client.post("/models/train", json={},
                       headers=bearer(token)).status_code == 401

    # Reinstating does not re-issue: disabling is often precautionary, and a
    # forced rotation would make the cautious act expensive enough to skip.
    client.post(f"/principals/{row['id']}/enable", headers=CTL)
    assert client.post("/models/train", json={},
                       headers=bearer(token)).status_code != 401


def test_disabling_keeps_the_row_so_its_history_stays_attributable(client,
                                                                   api_db):
    token = account(client, "vera", "teacher")
    row = db.one(api_db, "SELECT id FROM principal WHERE name = 'vera'")
    client.delete(f"/principals/{row['id']}", headers=CTL)

    still = db.one(api_db, "SELECT * FROM principal WHERE name = 'vera'")
    assert still is not None and still["disabled_at"]


def test_two_accounts_cannot_share_a_name(client):
    account(client, "vera", "teacher")
    r = client.post("/principals", json={"name": "vera", "role": "student"},
                    headers=CTL)
    assert r.status_code == 409


def test_an_unknown_role_is_refused_with_the_ones_that_exist(client):
    r = client.post("/principals", json={"name": "x", "role": "professor"},
                    headers=CTL)
    assert r.status_code == 400
    assert "student" in r.json()["detail"]


# --------------------------------------------------------------------------
# What the audit columns finally record
#
# Eight columns have been writing the constant "control" since they were added.
# This is the whole point of the feature.
# --------------------------------------------------------------------------

def test_a_queued_job_records_the_account_that_asked_for_it(client, api_db):
    token = account(client, "teacher-anna", "teacher")
    client.post("/models/train",
                json={"param": "muf", "tx": "NIC3", "rx": "Yoshkar-Ola",
                      "method": "contour", "estimator": "huber", "lead_h": 24},
                headers=bearer(token))

    row = db.one(api_db, "SELECT requested_by FROM train_job"
                         " ORDER BY id DESC LIMIT 1")
    assert row is not None, "the job was not queued at all"
    assert row["requested_by"] == "teacher-anna"


def test_a_mute_rule_records_who_muted_it(client, api_db):
    token = account(client, "root", "admin")
    client.post("/circuits/mute", json={"tx": "unkown"}, headers=bearer(token))

    row = db.one(api_db, "SELECT muted_by FROM muted_circuit")
    assert row["muted_by"] == "root"


def test_the_shared_secret_signs_its_own_name_not_a_borrowed_one(client,
                                                                 api_db):
    """`CONTROL_TOKEN` is shared, so claiming a person did it would be a lie."""
    client.post("/circuits/mute", json={"tx": "unkown"}, headers=CTL)
    assert db.one(api_db, "SELECT muted_by FROM muted_circuit"
                  )["muted_by"] == "control"


# --------------------------------------------------------------------------
# /whoami
# --------------------------------------------------------------------------

def test_whoami_never_refuses_because_the_console_asks_it_first(client):
    """It renders the signed-out state, so it cannot 401 on a bad token."""
    anon = client.get("/whoami").json()
    assert anon["name"] == "anonymous" and anon["capabilities"] == []

    bad = client.get("/whoami", headers=bearer("iono_nonsense")).json()
    assert bad["name"] == "anonymous"


def test_whoami_reports_the_role_and_what_it_holds(client):
    token = account(client, "teacher-anna", "teacher")
    me = client.get("/whoami", headers=bearer(token)).json()
    assert me == {"name": "teacher-anna", "role": "teacher", "source": "token",
                  "capabilities": ["model"]}

    control = client.get("/whoami", headers=CTL).json()
    assert control["source"] == "control"
    assert set(control["capabilities"]) == {c.value for c in auth.Capability}


def test_the_roster_is_not_an_open_read(client):
    """A list of who has access is a map of what to attack."""
    assert client.get("/principals").status_code == 401
    token = account(client, "student-boris", "student")
    assert client.get("/principals", headers=bearer(token)).status_code == 403
    assert client.get("/principals", headers=CTL).status_code == 200


def test_a_caller_cannot_name_someone_else_in_the_audit_trail(client, api_db):
    """The actor comes from the token, never from the body.

    Before accounts existed the client was the only thing that could say who it
    was, so `issued_by` was read out of the payload with "web" as a default.
    With a real identity behind the request that field is a name anyone can
    claim, and the stop button is the last place to take a caller's word for
    who pressed it.
    """
    token = account(client, "root", "admin")
    client.post("/stations/SIM/commands",
                json={"name": "stop", "issued_by": "somebody-else"},
                headers=bearer(token))

    row = db.one(api_db, "SELECT issued_by FROM command ORDER BY rowid DESC"
                         " LIMIT 1")
    assert row["issued_by"] == "root"


def test_an_account_can_still_read_when_reads_are_closed(client, monkeypatch):
    """One header, one token: `READ_TOKEN` must not lock out account holders.

    There is a single `Authorization` header and a person has a single token.
    If `require_read` compared only against `READ_TOKEN`, setting it would shut
    every account out of the entire console -- `/whoami` included, which the
    console asks before it can render its own signed-in state. A student who
    may write nothing may certainly look.
    """
    token = account(client, "student-boris", "student")
    monkeypatch.setattr(auth, "READ_TOKEN", "rd")

    # Closed to a stranger.
    assert client.get("/stations").status_code == 401
    # Open to the read token, as before.
    assert client.get("/stations", headers=bearer("rd")).status_code == 200
    # And open to a named account, which is the part that was broken.
    assert client.get("/stations", headers=bearer(token)).status_code == 200
    assert client.get("/whoami", headers=bearer(token)).status_code == 200
    assert client.get("/stations", headers=CTL).status_code == 200

    # Still not a way in: reads open, writes unchanged.
    assert client.post("/models/train", json={},
                       headers=bearer(token)).status_code == 403


def test_an_unreadable_accounts_table_denies_rather_than_admits(client,
                                                                monkeypatch):
    """`identify` swallows a database error, and must fail *closed*.

    Raised by a local first-pass review on 2026-08-29 as "authorization bypass
    via unhandled exception". It had the direction backwards -- an unreadable
    table means the token is not recognised, and an unrecognised token is
    refused. But nothing asserted which way it fell, and "the accounts table
    was briefly unreadable" is exactly the moment nobody is watching, so the
    direction is pinned here rather than left to a reading of the code.
    """
    from services.api import db as db_mod

    token = account(client, "root", "admin")

    def broken(*args, **kwargs):
        raise db_mod.sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(db_mod, "principal_by_token", broken)

    # Denied, not admitted -- and 401, because as far as the service can tell
    # the token is simply not one it knows.
    assert client.post("/models/train", json={},
                       headers=bearer(token)).status_code == 401
    # `CONTROL_TOKEN` still works: it is checked before the table is touched,
    # which is the "way back in" the module docstring promises.
    assert client.post("/circuits/mute", json={"tx": "zz"},
                       headers=CTL).status_code == 200


def test_every_page_carries_the_identity_slot_and_a_way_out(client):
    """Signing out has to be reachable from wherever the gates are.

    The gates disable buttons on /ui, /ui/forecast and /ui/archives alike, so a
    teacher seeing greyed controls on the forecast page needs to be told who
    they are and offered a way to stop being that person. Emptying the
    console's token field by hand is not it: `ME` and the gates resolve once
    per page load, so the screen would go on naming a signed-in person with
    buttons enabled until something reloaded it -- and only /ui reloads itself.
    """
    for path in ("/ui", "/ui/forecast", "/ui/archives"):
        page = client.get(path).text
        assert 'id="who"' in page, f"{path} has nowhere to say who you are"
        assert "function signOut()" in page, f"{path} offers no way out"
        # The reload is the mechanism: it re-runs the gates from an empty
        # token. Without it the buttons stay enabled on a signed-out page.
        assert "sessionStorage.removeItem('tok')" in page
        assert "location.reload()" in page
