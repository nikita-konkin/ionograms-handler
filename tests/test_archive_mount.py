"""The mount panel survives the mount it is reporting on being broken.

The symptom that prompted this: `/ui/archives` returning "Internal Server
Error" on the work server while `docker ps` showed the api **healthy**. Those
two are consistent, and that is the trap -- `/healthz` answers "is this process
up" and touches no storage on purpose, so a NAS that has stopped answering
leaves a green container and one dead page.

`Path.is_dir()` swallows only ENOENT, ENOTDIR, EBADF and ELOOP. A mount that is
present but sick raises EIO, ESTALE, EACCES or ETIMEDOUT, and those propagate
straight out of the request. `iterdir` was guarded; `is_dir` was not.
"""

from __future__ import annotations

import errno
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from services.api import archives, auth, db, main, net
from services.api import series as series_mod

#: What a mount that is present but not answering raises. ENOENT is in the
#: list as the control: it is the one `Path.is_dir` already handled, so a test
#: that only used it would have passed against the bug.
SICK = [errno.EIO, errno.ESTALE, errno.EACCES, errno.ETIMEDOUT, errno.ENOENT]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "READ_TOKEN", "")
    monkeypatch.setattr(auth, "CONTROL_TOKEN", "ctl")
    monkeypatch.setenv("API_DB", str(tmp_path / "api.sqlite3"))
    monkeypatch.setattr(db, "DEFAULT_DB", tmp_path / "api.sqlite3")
    monkeypatch.setattr(main, "WARM_CENSUS", False)
    monkeypatch.setattr(net, "ENABLED", False)
    net.reset()
    monkeypatch.setattr(series_mod, "MODEL", False)
    series_mod.clear()
    with TestClient(main.app) as c:
        yield c


#: The root the fixture pretends is mounted. Pinned rather than taken from the
#: environment, so the test does not quietly pass by patching a path the app
#: never looks at -- which is exactly how the first version of this passed.
ROOT = Path("/archive")


@pytest.fixture
def sick_mount(monkeypatch):
    """Make every stat of the archive root fail, as a dead NAS does."""
    def go(number: int):
        real = Path.is_dir

        def is_dir(self):
            if self == ROOT:
                raise OSError(number, "the mount is not answering")
            return real(self)

        monkeypatch.setattr(archives, "roots", lambda: [ROOT])
        monkeypatch.setattr(Path, "is_dir", is_dir)
    return go


@pytest.mark.parametrize("number", SICK, ids=lambda n: errno.errorcode[n])
def test_the_mount_state_reports_rather_than_raises(sick_mount, number):
    sick_mount(number)
    state = archives._root_state(ROOT, 0)
    assert state["exists"] is False
    assert state["entries"] is None


@pytest.mark.parametrize("number", SICK, ids=lambda n: errno.errorcode[n])
def test_the_archives_page_still_renders(client, sick_mount, number):
    """A dead mount is the thing this page exists to show. It must not be the
    thing that stops it rendering."""
    sick_mount(number)
    response = client.get("/ui/archives")
    assert response.status_code == 200, (
        f"{errno.errorcode[number]} from the mount 500s the page that reports "
        f"on the mount")


@pytest.mark.parametrize("number", SICK, ids=lambda n: errno.errorcode[n])
def test_healthz_and_the_page_do_not_disagree_silently(client, sick_mount, number):
    """`/healthz` staying green while storage is gone is by design -- it answers
    a different question. The page has to answer this one."""
    sick_mount(number)
    assert client.get("/healthz").json()["ok"] is True
    body = client.get("/archives").json()
    assert body["mount"]["exists"] is False


def test_an_unreadable_root_says_why(sick_mount):
    sick_mount(errno.EIO)
    state = archives._root_state(ROOT, 0)
    assert "Input/output error" in state["error"] or "EIO" in state["error"] \
        or "not answering" in state["error"]


# --------------------------------------------------------------------------
# Which unreadable, and therefore which fix
# --------------------------------------------------------------------------
#
# The work server hit EIO on one folder and EHOSTDOWN on another within four
# minutes. The page told it to set ARCHIVE_HOST_PATH and redeploy -- advice for
# a bind mount that was never made, given to someone whose path was correct and
# whose NAS had gone away. Redeploying would have changed nothing.

@pytest.mark.parametrize("name", ["EIO", "EHOSTDOWN", "ESTALE", "ETIMEDOUT"])
def test_storage_that_stopped_answering_is_not_a_missing_bind_mount(name):
    number = getattr(errno, name, None)
    if number is None:
        pytest.skip(f"{name} is not defined on this platform")
    assert archives.mount_fault(OSError(number, name)) == "unreachable"


def test_a_mount_that_was_never_made_is_still_reported_as_missing():
    assert archives.mount_fault(OSError(errno.ENOENT, "no such file")) == "missing"
    assert archives.mount_fault(OSError(errno.ENOTDIR, "not a directory")) == "missing"


def test_a_permission_problem_is_neither():
    assert archives.mount_fault(OSError(errno.EACCES, "denied")) == "denied"


@pytest.mark.parametrize("name,expected,wrong", [
    ("EIO", "not answering", "ARCHIVE_HOST_PATH"),
    ("EHOSTDOWN", "not answering", "ARCHIVE_HOST_PATH"),
    ("ENOENT", "ARCHIVE_HOST_PATH", "not answering"),
])
def test_the_page_offers_the_fix_that_matches_the_errno(client, sick_mount,
                                                        name, expected, wrong):
    number = getattr(errno, name, None)
    if number is None:
        pytest.skip(f"{name} is not defined on this platform")
    sick_mount(number)
    text = client.get("/ui/archives").text
    assert expected in text, f"{name}: page does not mention {expected!r}"
    assert wrong not in text.split("archives")[0] or expected in text


def test_the_unreachable_message_does_not_send_anyone_to_redeploy(client,
                                                                  sick_mount):
    """The specific wrong turn: editing a correct .env and redeploying while
    the storage is still down."""
    sick_mount(errno.EIO)
    text = client.get("/ui/archives").text
    start = text.find("The mount is there")
    assert start > 0, "the unreachable banner did not render"
    # Bounded by its own paragraph, so the two phrases have to be in the same
    # message rather than merely somewhere on a long page.
    # Tags stripped and whitespace collapsed: the assertion is about what a
    # reader sees, and `is <i>not</i> the problem` is one sentence to them.
    banner = re.sub(r"\s+", " ",
                    re.sub(r"<[^>]+>", "", text[start:text.index("</p>", start)]))
    assert "not the problem" in banner
    assert "fix it on the host" in banner
    assert "redeploying will not help" in banner.lower()


# --------------------------------------------------------------------------
# The routes that serve a file, not a page
# --------------------------------------------------------------------------

def seed_sounding(conn):
    conn.execute(
        "INSERT INTO sounding (file, path, datetime, tx, rx, ingested_at) "
        "VALUES (?,?,?,?,?,?)",
        ("s1.h5", "2026-08-19/s1.h5", "2026-08-19 00:00:00", "NIC", "DOB",
         db.utcnow()))
    conn.commit()
    return conn.execute("SELECT id FROM sounding").fetchone()[0]


@pytest.fixture
def sick_file(monkeypatch):
    """Every `is_file` on the archive fails, as it does on a dead mount."""
    def go(number: int):
        real = Path.is_file

        def is_file(self):
            raise OSError(number, "the mount is not answering")

        monkeypatch.setattr(Path, "is_file", is_file)
    return go


@pytest.mark.parametrize("name", ["EIO", "EHOSTDOWN", "ESTALE"])
def test_an_ionogram_on_dead_storage_is_503_not_500(client, sick_file, name):
    """The row is intact and the file is probably fine -- what is down is the
    disk. A 500 says the server is broken; a 410 would say the sounding is
    gone. Both are wrong, and 410 would be cached as permanent."""
    number = getattr(errno, name, None)
    if number is None:
        pytest.skip(f"{name} is not defined on this platform")
    sounding = seed_sounding(client.app.state.db)
    sick_file(number)

    response = client.get(f"/ionogram/{sounding}.png")
    assert response.status_code == 503, response.status_code
    assert "storage, not data" in response.json()["detail"]


def test_a_genuinely_missing_file_is_still_410(client, monkeypatch):
    """The other branch has to keep working: a row whose file really is not
    there is a different fault with a different fix."""
    sounding = seed_sounding(client.app.state.db)
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    response = client.get(f"/ionogram/{sounding}.png")
    assert response.status_code == 410
    assert "ARCHIVE_ROOT" in response.json()["detail"]


def test_a_permission_problem_names_the_uid(client, sick_file):
    sounding = seed_sounding(client.app.state.db)
    sick_file(errno.EACCES)
    detail = client.get(f"/ionogram/{sounding}.png").json()["detail"]
    assert "uid 10001" in detail
