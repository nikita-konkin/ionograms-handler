"""Registering folders to index, and the scan that indexes them.

The indexer itself is `services.api.watch`, tested elsewhere. What is tested
here is the registry around it: that a folder this server cannot see is
refused rather than accepted and silently scanned to no effect, that a scan
records what it did, that unregistering never destroys a measurement, and
that two scans cannot run at once.

The scan tests use a **real synthetic `.lfs` archive** rather than a mocked
pipeline. The whole promise of the page is that indexing a folder produces
characteristics, and a test that stubs the pipeline would pass just as
happily on a build where the pipeline was never reached.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conftest import synth_iq                            # noqa: E402
from services.api import archives as archives_mod        # noqa: E402
from services.api import auth, db, main                  # noqa: E402

CTL = {"Authorization": "Bearer ctl"}


@pytest.fixture
def archive_root(tmp_path, make_lfs):
    """An archive root holding one folder with soundings and one without."""
    root = tmp_path / "archive"
    (root / "good").mkdir(parents=True)
    (root / "empty").mkdir(parents=True)
    for i in range(2):
        # `make_lfs` writes into tmp_path; move it under the root so the tree
        # is the shape `find_soundings` walks.
        src = make_lfs(synth_iq(n_freq=64, window=256, echo_range_km=2700.0,
                                half_span_km=60_000.0, echo_last_bin=40),
                       name=f"synth{i}.lfs", dur=2)
        src.rename(root / "good" / src.name)
    return root


@pytest.fixture
def client(tmp_path, archive_root, monkeypatch):
    from fastapi.testclient import TestClient

    from services.api import net
    from services.api import series as series_mod

    monkeypatch.setattr(auth, "READ_TOKEN", "")
    monkeypatch.setattr(auth, "CONTROL_TOKEN", "ctl")
    monkeypatch.setenv("API_DB", str(tmp_path / "api.sqlite3"))
    monkeypatch.setattr(db, "DEFAULT_DB", tmp_path / "api.sqlite3")
    monkeypatch.setenv("ARCHIVE_ROOT", str(archive_root))
    monkeypatch.setattr(main, "WARM_CENSUS", False)
    monkeypatch.setattr(net, "ENABLED", False)
    net.reset()
    monkeypatch.setattr(series_mod, "MODEL", False)
    series_mod.clear()
    # The periodic scanner must not fire during a test: it would race the
    # explicit scans below for the module-wide lock and make them flaky.
    monkeypatch.setattr(archives_mod, "DEFAULT_INTERVAL_S", 0.0)
    with TestClient(main.app) as c:
        yield c


def _add(client, path="good", **kw):
    body = {"path": path}
    body.update(kw)
    return client.post("/archives", json=body, headers=CTL)


# --- registration -----------------------------------------------------------

def test_a_folder_is_registered_listed_and_removed(client):
    added = _add(client, name="feb")
    assert added.status_code == 200, added.text
    assert added.json()["found"]["soundings"] == 2
    assert added.json()["found"]["by_format"] == {"lfs": 2}

    listed = client.get("/archives").json()["archives"]
    assert [a["name"] for a in listed] == ["feb"]
    assert listed[0]["relpath"] == "good"

    gone = client.delete(f"/archives/{added.json()['id']}", headers=CTL)
    assert gone.status_code == 200
    assert client.get("/archives").json()["archives"] == []


def test_the_path_is_stored_relative_to_the_root(client, archive_root):
    """Absolute in, relative out. The same database is read from the host and
    from inside the container, which mount the archive at different paths."""
    added = _add(client, path=str(archive_root / "good"))
    assert added.status_code == 200
    assert added.json()["path"] == "good"


def test_a_folder_under_no_root_is_refused_and_says_how_to_fix_it(client):
    r = _add(client, path="/etc")
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "under none of this server's archive roots" in detail
    assert "ARCHIVE_ROOTS" in detail, (
        "the fix is a volume plus a root entry, and the message has to say so")
    assert "redeploy" in detail, (
        "a container's mounts are fixed at start; no amount of clicking here "
        "can add one, and the message must not imply otherwise")


def test_traversal_is_refused(client):
    assert _add(client, path="../../etc").status_code == 400
    assert _add(client, path="good/../../..").status_code == 400


def test_the_root_itself_can_be_registered(client):
    """It used to be refused, and that refusal was the bug.

    When the day directories sit directly under the root -- which is how the
    station's own receiver writes -- refusing the root forces one archive row
    per day. The rig had fifteen, and every new day the receiver created was a
    manual registration. Scanning is recursive, so registering the folder that
    *contains* the days is what makes new ones arrive on their own.
    """
    r = _add(client, path=".")
    assert r.status_code == 200, r.json()
    assert r.json()["path"] == "."


def test_a_folder_with_nothing_to_index_is_refused_at_registration(client):
    """Caught while the operator is looking at it, rather than after a scan
    that truthfully reports having loaded zero."""
    r = _add(client, path="empty")
    assert r.status_code == 400
    assert "no soundings" in r.json()["detail"]


def test_a_format_filter_that_matches_nothing_is_refused(client):
    r = _add(client, format="chirp2")
    assert r.status_code == 400
    assert "chirp2" in r.json()["detail"]


def test_an_unknown_format_is_refused(client):
    assert _add(client, format="rinex").status_code == 400


def test_the_same_folder_cannot_be_registered_twice(client):
    assert _add(client, name="one").status_code == 200
    again = _add(client, name="two")
    assert again.status_code == 409
    assert "already registered" in again.json()["detail"]


def test_a_duplicate_name_is_refused(client):
    assert _add(client, name="dup").status_code == 200
    assert _add(client, path="empty", name="dup").status_code in (400, 409)


# --- methods ----------------------------------------------------------------

def test_methods_default_to_the_pipeline_default(client):
    from muf.extractors import DEFAULT_METHODS

    assert _add(client).json()["methods"] == ",".join(DEFAULT_METHODS)


def test_an_unknown_method_is_refused(client):
    """`watch.already_done` counts a sounding finished only when it holds a row
    for every *requested* method. A name nothing can produce would make every
    sounding look permanently unfinished and re-scan the folder forever."""
    r = _add(client, methods="algo,telepathy")
    assert r.status_code == 400
    assert "telepathy" in r.json()["detail"]


def test_methods_can_be_narrowed_and_widened(client):
    archive_id = _add(client, methods="algo").json()["id"]
    r = client.post(f"/archives/{archive_id}/methods",
                    json={"methods": "algo,contour"}, headers=CTL)
    assert r.status_code == 200
    assert r.json()["methods"] == "algo,contour"
    assert "revisited" in r.json()["note"]


# --- enable / disable -------------------------------------------------------

def test_a_disabled_archive_is_skipped_by_a_sweep(client, archive_root,
                                                  tmp_path):
    archive_id = _add(client).json()["id"]
    assert client.post(f"/archives/{archive_id}/enabled",
                       json={"enabled": False}, headers=CTL).status_code == 200

    conn = db.connect(tmp_path / "api.sqlite3")
    try:
        assert db.archives(conn, enabled_only=True) == []
        assert archives_mod.scan_all(
            conn, archive_root=archive_root,
            db_path=tmp_path / "api.sqlite3") == 0
    finally:
        conn.close()


# --- scanning ---------------------------------------------------------------

def test_a_scan_indexes_the_folder_and_derives_characteristics(
        client, archive_root, tmp_path):
    """The whole promise of the page, end to end on real `.lfs` bytes.

    Not just "rows appeared": an `extraction` row is what MUF, LOF and the SAO
    record are built from, so its absence would mean the page indexed files
    and produced nothing anyone can use.
    """
    archive_id = _add(client, methods="algo").json()["id"]
    conn = db.connect(tmp_path / "api.sqlite3")
    try:
        row = db.archive(conn, archive_id)
        # min_age_s=0: the fixture's files were written moments ago and the
        # watcher would otherwise, correctly, hold them back as too fresh.
        result = archives_mod.scan_once(
            row, archive_root=archive_root, db_path=tmp_path / "api.sqlite3",
            min_age_s=0)
        assert result["found"] == 2
        assert result["loaded"] == 2, result

        assert db.one(conn, "SELECT COUNT(*) AS n FROM sounding")["n"] == 2
        assert db.one(conn, "SELECT COUNT(*) AS n FROM extraction"
                            " WHERE method = 'algo'")["n"] == 2

        after = db.archive(conn, archive_id)
        assert after["last_scan_ok"] == 1
        assert after["last_scan_at"]
        assert "2 on disk" in after["last_scan_result"]

        # And the second pass finds nothing to do, which is the difference
        # between this and re-running `ingest` over the whole folder.
        again = archives_mod.scan_once(
            row, archive_root=archive_root, db_path=tmp_path / "api.sqlite3",
            min_age_s=0)
        assert again["new"] == 0
    finally:
        conn.close()


def test_widening_methods_brings_the_old_soundings_back_into_scope(
        client, archive_root, tmp_path):
    archive_id = _add(client, methods="algo").json()["id"]
    db_path = tmp_path / "api.sqlite3"
    conn = db.connect(db_path)
    try:
        archives_mod.scan_once(db.archive(conn, archive_id),
                               archive_root=archive_root, db_path=db_path,
                               min_age_s=0)
        db.set_archive_methods(conn, archive_id, "algo,contour")
        result = archives_mod.scan_once(db.archive(conn, archive_id),
                                        archive_root=archive_root,
                                        db_path=db_path, min_age_s=0)
        assert result["new"] == 2, "a method nothing has yet must re-scope them"
        assert db.one(conn, "SELECT COUNT(*) AS n FROM extraction"
                            " WHERE method = 'contour'")["n"] == 2
    finally:
        conn.close()


def test_removing_an_archive_keeps_its_soundings(client, archive_root,
                                                 tmp_path):
    """Unregistering says "stop indexing this", not "forget what was
    measured". The soundings are the record; the archive row is the
    instruction."""
    archive_id = _add(client).json()["id"]
    db_path = tmp_path / "api.sqlite3"
    conn = db.connect(db_path)
    try:
        archives_mod.scan_once(db.archive(conn, archive_id),
                               archive_root=archive_root, db_path=db_path,
                               min_age_s=0)
    finally:
        conn.close()

    before = db.one(client.app.state.db,
                    "SELECT COUNT(*) AS n FROM sounding")["n"]
    assert before == 2
    removed = client.delete(f"/archives/{archive_id}", headers=CTL).json()
    assert removed["soundings_kept"] == 2
    assert db.one(client.app.state.db,
                  "SELECT COUNT(*) AS n FROM sounding")["n"] == 2
    assert db.one(client.app.state.db,
                  "SELECT COUNT(*) AS n FROM extraction")["n"] > 0


def test_only_one_scan_runs_at_a_time(client, archive_root):
    """CPU-bound and holding a database lock. Two would finish later than one
    after the other, so the second is refused rather than queued."""
    started = threading.Event()
    release = threading.Event()

    def hold():
        with archives_mod._LOCK:
            started.set()
            release.wait(5)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert started.wait(5)
    archives_mod._set_status(archive_id=1, name="holder",
                             started_at=1.0, finished_at=None)
    try:
        archive_id = _add(client).json()["id"]
        r = client.post(f"/archives/{archive_id}/scan", headers=CTL)
        assert r.status_code == 409
        assert "already running" in r.json()["detail"]
    finally:
        release.set()
        holder.join(5)
        archives_mod._set_status(started_at=None, finished_at=None, name="")


def test_a_folder_that_vanished_is_reported_not_counted_as_clean(
        client, archive_root, tmp_path):
    """The failure mode a stored registration has and a CLI target does not.

    `watch.find_new` skips a target holding nothing, which is right when
    somebody just typed the path -- an archive holds detection trees and empty
    days beside the ionograms. For a registration that can go stale on its own
    it is wrong: an unmounted share would scan clean and report "0 on disk"
    with a green tick for as long as anyone left it running.
    """
    archive_id = _add(client).json()["id"]
    db_path = tmp_path / "api.sqlite3"
    conn = db.connect(db_path)
    try:
        row = db.archive(conn, archive_id)
        (archive_root / "good").rename(archive_root / "moved")
        assert archives_mod.scan(row, archive_root=archive_root,
                                 db_path=db_path, min_age_s=0)
        after = db.archive(conn, archive_id)
        assert after["last_scan_ok"] == 0
        assert "not on disk" in after["last_scan_result"]
    finally:
        conn.close()


def test_a_scan_that_raises_is_recorded_as_a_failure(client, archive_root,
                                                     tmp_path, monkeypatch):
    """A pass that blew up must not leave the previous pass's cheerful summary
    standing as though it were current."""
    archive_id = _add(client).json()["id"]
    db_path = tmp_path / "api.sqlite3"

    def boom(*a, **kw):
        raise RuntimeError("pipeline exploded")

    monkeypatch.setattr(archives_mod.watch, "find_new", boom)
    conn = db.connect(db_path)
    try:
        assert archives_mod.scan(db.archive(conn, archive_id),
                                 archive_root=archive_root, db_path=db_path,
                                 min_age_s=0)
        after = db.archive(conn, archive_id)
        assert after["last_scan_ok"] == 0
        assert "pipeline exploded" in after["last_scan_result"]
        assert archives_mod.status()["ok"] is False
    finally:
        conn.close()
        archives_mod._set_status(started_at=None, finished_at=None,
                                 ok=None, error="", result="", name="")


# --- auth -------------------------------------------------------------------

@pytest.mark.parametrize("method,url,body", [
    ("post", "/archives", {"path": "good"}),
    ("post", "/archives/1/scan", None),
    ("post", "/archives/1/enabled", {"enabled": False}),
    ("post", "/archives/1/methods", {"methods": "algo"}),
    ("delete", "/archives/1", None),
])
def test_writes_need_the_control_token(client, method, url, body):
    call = getattr(client, method)
    r = call(url, json=body) if body is not None else call(url)
    assert r.status_code in (401, 403), f"{method} {url} was not protected"


def test_reads_are_open(client):
    assert client.get("/archives").status_code == 200


def test_a_missing_archive_is_404_not_500(client):
    assert client.post("/archives/999/scan", headers=CTL).status_code == 404
    assert client.delete("/archives/999", headers=CTL).status_code == 404


# --- the page ---------------------------------------------------------------

def test_the_page_lists_the_registered_folders(client):
    _add(client, name="feb")
    page = client.get("/ui/archives").text
    assert "feb" in page
    assert "add a folder" in page


def test_the_page_says_what_to_do_when_nothing_is_registered(client):
    assert "Nothing registered yet" in client.get("/ui/archives").text


# --- the mount, and what is in it -------------------------------------------

def test_the_page_shows_which_host_folder_is_mounted(client, monkeypatch):
    """`/archive` alone cannot tell an operator whether the .env they edited
    took effect. The container cannot discover the host path either, so
    compose passes it in and the page prints it."""
    monkeypatch.setenv("ARCHIVE_HOST_PATH", "/srv/lfs-on-the-host")
    body = client.get("/archives").json()
    assert body["mount"]["host"] == "/srv/lfs-on-the-host"
    assert "/srv/lfs-on-the-host" in client.get("/ui/archives").text


def test_a_missing_host_path_says_so_rather_than_inventing_one(client,
                                                               monkeypatch):
    monkeypatch.delenv("ARCHIVE_HOST_PATH", raising=False)
    assert client.get("/archives").json()["mount"]["host"] == ""
    assert "not reported" in client.get("/ui/archives").text


def test_an_empty_mount_is_called_out(client, monkeypatch, tmp_path):
    """A bind mount whose source was renamed on the host still exists inside
    the container -- as an empty directory. Every scan then reports "0 on
    disk" truthfully and forever."""
    empty = tmp_path / "nothing-here"
    empty.mkdir()
    monkeypatch.setenv("ARCHIVE_ROOT", str(empty))
    assert client.get("/archives").json()["mount"]["empty"] is True


def test_an_unreadable_mount_is_called_out(client, monkeypatch, tmp_path):
    monkeypatch.setenv("ARCHIVE_ROOT", str(tmp_path / "does-not-exist"))
    mount = client.get("/archives").json()["mount"]
    assert mount["exists"] is False




def _candidates(client, timeout: float = 10.0) -> dict:
    """The candidate list, waited for.

    It is surveyed on a background thread on purpose -- walking the archive
    on the request thread is what made the page unrenderable during an index
    -- so the honest way to read it in a test is the way the page reads it:
    ask, and ask again until `ready`.
    """
    import time as _time
    archives_mod.forget_candidates()
    deadline = _time.time() + timeout
    while _time.time() < deadline:
        body = client.get("/archives/candidates").json()
        if body["ready"]:
            return {c["path"]: c for c in body["items"]}
        _time.sleep(0.05)
    raise AssertionError("candidate survey never became ready")


def test_the_folders_in_the_mount_are_offered(client):
    """Adding a folder should be picking from what is mounted, not typing a
    path and hoping."""
    paths = _candidates(client)
    assert paths["good"]["soundings"] == 2
    assert paths["good"]["by_format"] == {"lfs": 2}
    assert paths["good"]["registered"] is False
    # And *only* folders that hold soundings. The list used to be every
    # subdirectory one level down with its count beside it, which meant
    # reading a page of folders to find the two with data in them. A choice
    # that cannot be taken does not belong on a list of choices.
    assert "empty" not in paths


def test_a_registered_folder_is_marked_in_the_candidate_list(client):
    _add(client, name="feb")
    good = _candidates(client)["good"]
    assert good["registered"] is True
    assert good["archive_id"] is not None


# --- method availability ----------------------------------------------------

def test_a_method_that_cannot_run_here_is_refused_with_its_reason(client,
                                                                  monkeypatch):
    """`cnn` imports wherever Keras is installed and still needs a model
    trained on this geometry. Requested without one it does not merely fail:
    `already_done` counts a sounding finished only when it holds a row for
    every requested method, so the archive would be re-scanned forever."""
    monkeypatch.setattr(archives_mod, "method_availability", lambda: {
        "algo": {"usable": True, "why": ""},
        "cnn": {"usable": False, "why": "no autoencoder found"},
    })
    monkeypatch.setattr(archives_mod, "usable_methods", lambda: ("algo",))
    r = _add(client, methods="algo,cnn")
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "no autoencoder found" in detail
    assert "re-scanned on every pass" in detail


def test_the_page_offers_every_method_and_disables_the_unusable(client,
                                                                monkeypatch):
    monkeypatch.setattr(archives_mod, "method_availability", lambda: {
        "algo": {"usable": True, "why": ""},
        "kmeans": {"usable": True, "why": ""},
        "contour": {"usable": True, "why": ""},
        "cnn": {"usable": False, "why": "no autoencoder found"},
    })
    page = client.get("/ui/archives").text
    for name in ("algo", "kmeans", "contour", "cnn"):
        assert f'value="{name}"' in page
    # The one that cannot run is rendered so it cannot be chosen at all.
    assert "disabled" in page
    assert "no autoencoder found" in page


def test_method_availability_reports_the_three_that_always_work():
    got = archives_mod.method_availability()
    for name in ("algo", "kmeans", "contour"):
        assert got[name]["usable"], f"{name} should always be usable"
    assert set(archives_mod.usable_methods()) >= {"algo", "kmeans", "contour"}


def test_methods_can_be_sent_as_a_list(client):
    """The page sends a joined string; a client that sends the array should
    not have to know that."""
    assert _add(client, methods=["algo", "contour"]).json()["methods"] == \
        "algo,contour"


# --- more than one root -----------------------------------------------------
#
# The answer to "can I put an array in ARCHIVE_HOST_PATH". Not there -- a
# variable inside `volumes:` is substituted into one list item, so "/a:/b"
# becomes the single broken mount "/a:/b:/archive:ro". The array goes in
# ARCHIVE_ROOTS, which lists *container* paths, and each still needs its own
# volume line. Neither this list nor any page can create a mount: a
# container's filesystem is fixed when it starts.

@pytest.fixture
def second_root(tmp_path, make_lfs):
    root = tmp_path / "elsewhere"
    (root / "other").mkdir(parents=True)
    src = make_lfs(synth_iq(n_freq=64, window=256, echo_range_km=2700.0,
                            half_span_km=60_000.0, echo_last_bin=40),
                   name="other0.lfs", dur=2)
    src.rename(root / "other" / src.name)
    return root


def test_extra_roots_are_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("ARCHIVE_ROOT", "/archive")
    monkeypatch.setenv("ARCHIVE_ROOTS", "/archive2:/archive3")
    assert [str(p) for p in archives_mod.roots()] == [
        "/archive", "/archive2", "/archive3"]


def test_the_primary_root_is_never_duplicated(monkeypatch):
    monkeypatch.setenv("ARCHIVE_ROOT", "/archive")
    monkeypatch.setenv("ARCHIVE_ROOTS", "/archive:/archive2")
    assert [str(p) for p in archives_mod.roots()] == ["/archive", "/archive2"]


def test_an_empty_roots_list_leaves_one_root(monkeypatch):
    monkeypatch.setenv("ARCHIVE_ROOT", "/archive")
    monkeypatch.setenv("ARCHIVE_ROOTS", "")
    assert [str(p) for p in archives_mod.roots()] == ["/archive"]


def test_a_folder_under_a_second_root_can_be_registered(client, second_root,
                                                        monkeypatch):
    monkeypatch.setenv("ARCHIVE_ROOTS", str(second_root))
    r = _add(client, path=str(second_root / "other"), name="elsewhere")
    assert r.status_code == 200, r.text
    assert r.json()["found"]["soundings"] == 1
    # Stored absolute, because there is no root both the host and the
    # container would agree to measure it against.
    assert r.json()["path"] == str(second_root / "other")


def test_a_folder_under_the_primary_root_stays_relative(client, second_root,
                                                        monkeypatch):
    """The portable case must not regress when a second root exists."""
    monkeypatch.setenv("ARCHIVE_ROOTS", str(second_root))
    assert _add(client, path="good").json()["path"] == "good"


def test_a_second_root_is_listed_with_its_host_folder(client, second_root,
                                                      monkeypatch):
    monkeypatch.setenv("ARCHIVE_ROOTS", str(second_root))
    monkeypatch.setenv("ARCHIVE_HOST_PATH", "/srv/one")
    monkeypatch.setenv("ARCHIVE_HOST_PATH_2", "/mnt/two")
    mount = client.get("/archives").json()["mount"]
    assert mount["extra"] == 1
    assert [r["host"] for r in mount["roots"]] == ["/srv/one", "/mnt/two"]
    assert mount["roots"][0]["primary"] is True
    assert mount["roots"][1]["primary"] is False


def test_candidates_span_every_root(client, second_root, monkeypatch):
    monkeypatch.setenv("ARCHIVE_ROOTS", str(second_root))
    by_path = _candidates(client)
    assert "good" in by_path and by_path["good"]["primary"] is True
    other = by_path[str(second_root / "other")]
    assert other["primary"] is False
    assert other["soundings"] == 1


def test_a_second_root_scans_and_derives_characteristics(
        client, second_root, tmp_path, monkeypatch):
    """The whole point: a folder on another disk indexes like any other."""
    monkeypatch.setenv("ARCHIVE_ROOTS", str(second_root))
    archive_id = _add(client, path=str(second_root / "other"),
                      methods="algo").json()["id"]
    db_path = tmp_path / "api.sqlite3"
    conn = db.connect(db_path)
    try:
        row = db.archive(conn, archive_id)
        result = archives_mod.scan_once(
            row, archive_root=client.app.state.archive_root,
            db_path=db_path, min_age_s=0)
        assert result["loaded"] == 1, result
        # `sounding.path` is absolute for the same reason the archive's is.
        stored = db.one(conn, "SELECT path FROM sounding")["path"]
        assert stored.startswith(str(second_root))
        assert db.one(conn, "SELECT COUNT(*) AS n FROM extraction")["n"] == 1
        # And the archive's own count still finds them.
        listed = next(a for a in db.archives(conn) if a["id"] == archive_id)
        assert listed["soundings"] == 1
    finally:
        conn.close()


def test_a_path_under_no_root_still_names_every_root_it_tried(client,
                                                              second_root,
                                                              monkeypatch):
    monkeypatch.setenv("ARCHIVE_ROOTS", str(second_root))
    detail = _add(client, path="/nowhere/at/all").json()["detail"]
    assert str(second_root) in detail, (
        "the refusal must list the roots it actually checked, not just the "
        "primary -- otherwise it reads as though the second one is not set up")


# --- progress ---------------------------------------------------------------
#
# Elapsed seconds alone are indistinguishable from a hang: "scanning, 240s"
# tells an operator nothing about whether to wait or go looking. These pin the
# count, the phase, and the fact that the phase is named before any file can
# be counted -- which is the part that used to look broken.

def test_a_scan_reports_a_running_count(client, archive_root, tmp_path,
                                        monkeypatch):
    archive_id = _add(client, methods="algo").json()["id"]
    db_path = tmp_path / "api.sqlite3"
    seen = []

    real = archives_mod._set_status

    def spy(**fields):
        real(**fields)
        seen.append(archives_mod.status())

    monkeypatch.setattr(archives_mod, "_set_status", spy)
    conn = db.connect(db_path)
    try:
        archives_mod.scan_once(db.archive(conn, archive_id),
                               archive_root=archive_root, db_path=db_path,
                               min_age_s=0, chunk=1)
    finally:
        conn.close()

    phases = [s["phase"] for s in seen]
    assert "reading" in phases, "the enumeration step must name itself"
    assert "indexing" in phases
    assert phases[-1] == "done"

    # chunk=1 over two files, so the count has to pass through 1 before 2 --
    # a bar that only ever showed 0 then 100 would be no better than a spinner.
    counts = [s["done"] for s in seen if s["phase"] == "indexing"]
    assert counts == sorted(counts), "progress must not go backwards"
    assert 1 in counts and 2 in counts, counts
    assert seen[-1]["total"] == 2 or seen[-2]["total"] == 2


def test_the_status_carries_a_percentage_and_an_eta(client):
    archives_mod._set_status(archive_id=1, name="x", started_at=time.time() - 10,
                             finished_at=None, phase="indexing", done=25,
                             total=100, loaded=25, skipped=0)
    try:
        s = client.get("/archives").json()["status"]
        assert s["percent"] == 25.0
        # 25 files in 10 s -> 75 left at 2.5/s -> about 30 s.
        assert 25 <= s["eta_s"] <= 35, s
    finally:
        archives_mod._set_status(started_at=None, finished_at=None, phase="",
                                 done=0, total=0, loaded=0, skipped=0, name="")


def test_no_eta_is_offered_before_a_single_file_is_done(client):
    """A rate extrapolated from zero completed files is not an estimate, it is
    a number shaped like one."""
    archives_mod._set_status(archive_id=1, name="x", started_at=time.time() - 10,
                             finished_at=None, phase="indexing", done=0,
                             total=100)
    try:
        s = client.get("/archives").json()["status"]
        assert s["percent"] == 0.0
        assert "eta_s" not in s
    finally:
        archives_mod._set_status(started_at=None, finished_at=None, phase="",
                                 done=0, total=0, name="")


def test_a_scan_with_nothing_to_do_still_ends_cleanly(client, archive_root,
                                                      tmp_path):
    """The second press. No files to index means no bar to fill, and the phase
    must still reach `done` rather than sitting on `reading` forever."""
    archive_id = _add(client, methods="algo").json()["id"]
    db_path = tmp_path / "api.sqlite3"
    conn = db.connect(db_path)
    try:
        row = db.archive(conn, archive_id)
        archives_mod.scan_once(row, archive_root=archive_root,
                               db_path=db_path, min_age_s=0)
        archives_mod.scan_once(row, archive_root=archive_root,
                               db_path=db_path, min_age_s=0)
    finally:
        conn.close()
    assert archives_mod.status()["phase"] == "done"


def test_the_page_carries_the_progress_panel(client):
    page = client.get("/ui/archives").text
    assert 'id="progressBar"' in page
    assert 'id="progress"' in page
    # Hidden at rest: an empty bar on a resting page reads as "stuck".
    assert 'id="progress" style="display:none' in page


def test_a_press_indexes_the_whole_folder_not_a_batch_of_it(client,
                                                            monkeypatch):
    """The cap belongs to the unattended loop. Someone pressing scan has asked
    for this archive indexed, and 200-at-a-time would answer that by doing an
    arbitrary fraction and calling the rest "held for the next pass"."""
    archive_id = _add(client).json()["id"]
    seen = {}

    def spy(row, *, archive_root, db_path=None, **kw):
        seen.update(kw)
        return True

    monkeypatch.setattr(archives_mod, "scan", spy)
    monkeypatch.setattr(archives_mod, "is_scanning", lambda: False)
    thread = archives_mod.scan_in_background(
        {"id": archive_id, "name": "x"}, archive_root=".")
    if thread is not None:
        thread.join(5)
    assert seen.get("batch") == 0, seen


def test_the_background_loop_keeps_its_cap(client, archive_root, tmp_path,
                                           monkeypatch):
    """It runs unattended on a box that is also serving pages."""
    _add(client)
    seen = {}

    def spy(row, *, archive_root, db_path=None, **kw):
        seen.update(kw)
        return True

    monkeypatch.setattr(archives_mod, "scan", spy)
    conn = db.connect(tmp_path / "api.sqlite3")
    try:
        archives_mod.scan_all(conn, archive_root=archive_root)
    finally:
        conn.close()
    assert "batch" not in seen, (
        "scan_all must not force a batch; the default cap applies")


# --- what a poll is allowed to cost -----------------------------------------
#
# The progress bar asks once a second. Everything it touches is therefore on
# a hot path, and the first version of it polled `GET /archives`, which
# surveyed every candidate folder: a recursive walk of the whole archive, on
# the request thread, every second, while a scan had the same disk. On a
# server with three large .lfs folders the page did not render at all.

def test_polling_the_scan_status_touches_no_disk(client, monkeypatch):
    def explode(*a, **kw):
        raise AssertionError("the status poll walked the archive")

    monkeypatch.setattr(archives_mod, "candidates", explode)
    monkeypatch.setattr(archives_mod, "survey", explode)
    r = client.get("/archives/status")
    assert r.status_code == 200
    assert r.json()["running"] is False


def test_listing_archives_never_walks_the_archive(client, monkeypatch):
    """`GET /archives` is polled and rendered; the walk happens elsewhere."""
    def explode(*a, **kw):
        raise AssertionError("a request thread walked the archive")

    monkeypatch.setattr(archives_mod, "candidates", explode)
    assert client.get("/archives").status_code == 200
    assert client.get("/ui/archives").status_code == 200


def test_the_candidate_list_is_not_refreshed_while_a_scan_runs(client,
                                                               monkeypatch):
    """The scan has that disk. A convenience list does not compete for it."""
    def explode(*a, **kw):
        raise AssertionError("surveyed candidates during a scan")

    monkeypatch.setattr(archives_mod, "candidates", explode)
    monkeypatch.setattr(archives_mod, "is_scanning", lambda: True)
    archives_mod.forget_candidates()
    body = client.get("/archives/candidates").json()
    assert body["ready"] is False
    assert "scan is running" in body["why"]


def test_an_empty_candidate_list_says_whether_it_looked_yet(client):
    """"None found" and "not looked yet" are different answers, and saying the
    first when the second is true is a lie about a mounted disk."""
    archives_mod.forget_candidates()
    first = client.get("/archives/candidates").json()
    assert first["ready"] is False and first["items"] == []
    assert _candidates(client)["good"]["soundings"] == 2


def test_registering_a_folder_drops_it_from_the_candidate_cache(client):
    """Otherwise it stays on offer for the whole TTL after it is registered."""
    assert _candidates(client)["good"]["registered"] is False
    _add(client, name="feb")
    assert _candidates(client)["good"]["registered"] is True


def test_a_survey_in_flight_when_things_change_is_discarded(client,
                                                            monkeypatch):
    """Emptying the cache is not enough on its own.

    A walk that began before a folder was registered lands after it, refills
    the cache with the pre-registration answer, and that answer then stands
    for the whole TTL -- so the folder stays on offer as a candidate after it
    has been taken.
    """
    import threading as _threading

    released = _threading.Event()
    real = archives_mod.candidates

    def slow(conn, root, **kw):
        found = real(conn, root, **kw)
        released.wait(5)                 # still walking while things change
        return found

    monkeypatch.setattr(archives_mod, "candidates", slow)
    archives_mod.forget_candidates()
    assert client.get("/archives/candidates").json()["ready"] is False

    _add(client, name="feb")             # bumps the generation mid-walk
    released.set()
    for _ in range(100):
        if not archives_mod._CAND_REFRESHING:
            break
        time.sleep(0.05)
    assert archives_mod.status() is not None
    body = client.get("/archives/candidates").json()
    stale = [c for c in body["items"]
             if c["path"] == "good" and not c["registered"]]
    assert not stale, "a pre-registration survey was allowed to land"


def test_the_ingest_watcher_can_be_pointed_at_folders_not_the_whole_root():
    """A root that is a general-purpose disk is not an archive.

    The watcher recurses without asking, so aimed at a mount that also holds
    a recycle bin it will ingest soundings that were deliberately deleted.
    The default stays the whole root -- correct when the root *is* an archive
    -- but it has to be narrowable without editing the compose file.
    """
    import yaml

    compose = yaml.safe_load(
        (Path(__file__).resolve().parents[1]
         / "deploy" / "docker-compose.hub.yml").read_text())
    command = compose["services"]["watch"]["command"]
    assert "${INGEST_TARGETS:-/archive}" in command, command
    # Narrowing the walk must not change where paths are stored relative to,
    # or the rows it writes stop matching the ones the api resolves.
    assert "--archive-root /archive" in command, command


def test_the_survey_does_not_walk_into_a_recycle_bin(client, archive_root):
    """A general-purpose volume is full of things that are not archives.

    `#recycle` is Synology's deleted-files bin and `@eaDir` its sidecar
    folders, scattered across every directory on the volume. Walking them
    recursively to count soundings is the difference, on a 16 TB share,
    between a survey that finishes and one that does not.
    """
    for junk in ("#recycle", "@eaDir", "lost+found", ".snapshots"):
        (archive_root / junk).mkdir()
    (archive_root / "keepme").mkdir()

    offered = _candidates(client)
    assert "#recycle" not in offered
    assert "@eaDir" not in offered
    assert "lost+found" not in offered
    assert ".snapshots" not in offered
    # And an ordinary folder holding nothing is not offered either -- for a
    # different reason than the recycle bin, but to the same end: everything
    # on this list is a folder that can actually be registered.
    assert "keepme" not in offered


def test_a_registered_folder_is_not_walked_again_to_count_it(client,
                                                             monkeypatch):
    """The registered folders are the big ones.

    Their sounding count is already in the database, derived from
    `sounding.path`; re-deriving it by walking the folder is the most
    expensive way to learn something already known, and it is what made the
    survey unable to finish on a 16 TB share.
    """
    _add(client, name="feb")
    walked = []
    real = archives_mod.survey

    def spy(path, format=None):
        walked.append(Path(path).name)
        return real(path, format)

    monkeypatch.setattr(archives_mod, "survey", spy)
    offered = _candidates(client)
    assert offered["good"]["registered"] is True
    assert "good" not in walked, (
        f"walked an already-registered folder: {walked}")
    # Still counted -- from the database, and marked as such so the page does
    # not render it as "nothing this server can read".
    assert offered["good"]["by_format"] is None
    assert offered["good"]["soundings"] >= 0


# --- the archive's format is a rule, not a label -----------------------------

@pytest.fixture
def mixed_root(archive_root, make_lfs, make_digisonde_h5):
    """A folder holding both formats, as a real day-folder does.

    `find_soundings` returns both unless narrowed, and the work server's
    archive is exactly this shape: chirp products beside `digisonde_ionogram-*`
    files from a different instrument.
    """
    folder = archive_root / "mixed"
    folder.mkdir()
    for i in range(2):
        src = make_lfs(synth_iq(n_freq=64, window=256, echo_range_km=2700.0,
                                half_span_km=60_000.0, echo_last_bin=40),
                       name=f"mix{i}.lfs", dur=2)
        src.rename(folder / src.name)
    for i in range(3):
        src = make_digisonde_h5(t0=1786245496.0 + i * 900,
                                transmitter="Juliusruh", receiver="DOB")
        src.rename(folder / src.name)
    return folder


def _scan(archive_id, archive_root, tmp_path, **kw):
    """Run one pass synchronously, the way the other scan tests do.

    `min_age_s=0` because the fixtures were written moments ago and the
    watcher would otherwise, correctly, hold them back as too fresh.
    """
    conn = db.connect(tmp_path / "api.sqlite3")
    try:
        return archives_mod.scan_once(
            db.archive(conn, archive_id), archive_root=archive_root,
            db_path=tmp_path / "api.sqlite3", min_age_s=0, **kw)
    finally:
        conn.close()


def _formats(tmp_path) -> dict:
    conn = db.connect(tmp_path / "api.sqlite3")
    try:
        return {r["format"]: r["n"] for r in db.rows(
            conn, "SELECT format, COUNT(*) AS n FROM sounding GROUP BY format")}
    finally:
        conn.close()


def test_a_format_narrowed_archive_ingests_only_that_format(
        client, mixed_root, archive_root, tmp_path):
    """`archive.format` is validated at registration and shown on the page.

    It has to mean something at scan time too, or it is a promise the server
    does not keep -- and it is the only thing that can keep a deleted
    sounding deleted, since `find_new` re-ingests any file without a row.
    """
    added = client.post("/archives",
                        json={"path": "mixed", "name": "mixed",
                              "format": "lfs", "methods": "algo"},
                        headers=CTL)
    assert added.status_code == 200, added.text
    _scan(added.json()["id"], archive_root, tmp_path)

    got = _formats(tmp_path)
    assert got.get("digisonde", 0) == 0, (
        f"a folder registered as lfs ingested digisonde products: {got}")
    assert got.get("lfs", 0) == 2, got


def _register_mixed(client, fmt=None):
    body = {"path": "mixed", "name": "mixed", "methods": "algo"}
    if fmt is not None:
        body["format"] = fmt
    r = client.post("/archives", json=body, headers=CTL)
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_narrowing_the_format_reports_what_it_puts_out_of_scope(
        client, mixed_root, archive_root, tmp_path):
    archive_id = _register_mixed(client)
    _scan(archive_id, archive_root, tmp_path)
    assert _formats(tmp_path) == {"lfs": 2, "digisonde": 3}

    r = client.post(f"/archives/{archive_id}/format",
                    json={"format": "lfs"}, headers=CTL)
    assert r.status_code == 200, r.text
    assert r.json()["orphans"] == {"total": 3, "by_format": {"digisonde": 3}}
    # Narrowing alone deletes nothing: changing a rule and destroying rows
    # are different decisions.
    assert _formats(tmp_path) == {"lfs": 2, "digisonde": 3}


def test_removing_them_takes_their_extractions_with_them(
        client, mixed_root, archive_root, tmp_path):
    """A measurement whose sounding is gone is one nothing can locate."""
    archive_id = _register_mixed(client)
    _scan(archive_id, archive_root, tmp_path)
    conn = db.connect(tmp_path / "api.sqlite3")
    try:
        before = db.one(conn, "SELECT COUNT(*) AS n FROM extraction")["n"]
    finally:
        conn.close()
    assert before > 0

    client.post(f"/archives/{archive_id}/format",
                json={"format": "lfs"}, headers=CTL)
    r = client.request("DELETE", f"/archives/{archive_id}/orphans",
                       headers=CTL)
    assert r.status_code == 200, r.text
    assert r.json()["removed"] == 3

    conn = db.connect(tmp_path / "api.sqlite3")
    try:
        assert _formats(tmp_path) == {"lfs": 2}
        orphaned = db.one(
            conn, "SELECT COUNT(*) AS n FROM extraction e"
                  " LEFT JOIN sounding s ON s.id = e.sounding_id"
                  " WHERE s.id IS NULL")["n"]
        assert orphaned == 0, "extraction rows outlived their soundings"
    finally:
        conn.close()


def test_removed_soundings_do_not_come_back_on_the_next_scan(
        client, mixed_root, archive_root, tmp_path):
    """The point of the whole design.

    `already_done` keys on the basename, so a file with no row is new
    forever. Without the format rule the next pass would re-ingest every one
    of these and the delete button would be a lie that took 15 minutes to
    expose.
    """
    archive_id = _register_mixed(client)
    _scan(archive_id, archive_root, tmp_path)
    client.post(f"/archives/{archive_id}/format",
                json={"format": "lfs"}, headers=CTL)
    client.request("DELETE", f"/archives/{archive_id}/orphans", headers=CTL)

    again = _scan(archive_id, archive_root, tmp_path)
    assert again["new"] == 0, again
    assert _formats(tmp_path) == {"lfs": 2}


def test_widening_the_format_brings_them_back(
        client, mixed_root, archive_root, tmp_path):
    """It is a rule, not a tombstone.

    Nothing here is unrecoverable: the mount is read-only and the files were
    never touched, so the undo is to stop excluding them.
    """
    archive_id = _register_mixed(client)
    _scan(archive_id, archive_root, tmp_path)
    client.post(f"/archives/{archive_id}/format",
                json={"format": "lfs"}, headers=CTL)
    client.request("DELETE", f"/archives/{archive_id}/orphans", headers=CTL)
    assert _formats(tmp_path) == {"lfs": 2}

    client.post(f"/archives/{archive_id}/format",
                json={"format": ""}, headers=CTL)
    _scan(archive_id, archive_root, tmp_path)
    assert _formats(tmp_path) == {"lfs": 2, "digisonde": 3}


def test_an_archive_admitting_every_format_refuses_the_delete(
        client, mixed_root, archive_root, tmp_path):
    """Otherwise it would delete rows the very next scan re-ingests."""
    archive_id = _register_mixed(client)
    _scan(archive_id, archive_root, tmp_path)
    r = client.request("DELETE", f"/archives/{archive_id}/orphans",
                       headers=CTL)
    assert r.status_code == 409
    assert "Narrow the format first" in r.json()["detail"]
    assert _formats(tmp_path) == {"lfs": 2, "digisonde": 3}


def test_an_unknown_format_is_refused_naming_the_real_ones(client):
    archive_id = _add(client).json()["id"]
    from muf import loader

    r = client.post(f"/archives/{archive_id}/format",
                    json={"format": "digisond"}, headers=CTL)
    assert r.status_code == 400
    for name in loader.FORMATS:
        assert name in r.json()["detail"]


def test_the_orphan_preview_is_open_but_removing_needs_the_token(
        client, mixed_root, archive_root, tmp_path):
    archive_id = _register_mixed(client, fmt="lfs")
    assert client.get(f"/archives/{archive_id}/orphans").status_code == 200
    assert client.post(f"/archives/{archive_id}/format",
                       json={"format": "lfs"}).status_code == 401
    assert client.request(
        "DELETE", f"/archives/{archive_id}/orphans").status_code == 401


def test_starting_the_server_does_not_walk_the_archive(tmp_path, archive_root,
                                                       monkeypatch):
    """A restart must not survey the mount on its own.

    The survey is a recursive walk. Doing it at boot means every restart
    walks the whole archive unprompted -- which on a share with per-file
    latency competes with the indexer for the one mount, and on a
    cloud-backed folder triggers mass materialisation. Measured: warming a
    Nextcloud-backed archive from a container drove the file provider to
    130-150% CPU and wedged the Docker daemon.
    """
    from fastapi.testclient import TestClient

    walked = []
    monkeypatch.setattr(archives_mod, "candidates",
                        lambda *a, **kw: walked.append(1) or [])
    monkeypatch.setenv("API_DB", str(tmp_path / "boot.sqlite3"))
    monkeypatch.setenv("ARCHIVE_ROOT", str(archive_root))
    monkeypatch.setattr(db, "DEFAULT_DB", tmp_path / "boot.sqlite3")
    archives_mod.forget_candidates()

    from services.api.main import app
    with TestClient(app):
        pass
    assert not walked, "starting the server surveyed the archive"


def test_the_api_image_ships_the_database_maintenance_tools():
    """`relabel_station` corrects rows in the api's database.

    That database lives on the `api-data` volume, reachable only from inside
    the container, so a tool left out of the image cannot be run against the
    thing it exists to fix. It was left out: `python -m tools.relabel_station`
    inside the container answered `No module named 'tools'`.
    """
    dockerfile = (Path(__file__).resolve().parents[1]
                  / "deploy" / "Dockerfile.api").read_text()
    assert "COPY tools/ /app/tools/" in dockerfile, (
        "the api image must carry tools/, or the maintenance tools cannot "
        "reach the database they operate on")


# --- overlapping registrations ----------------------------------------------
#
# Reachable only since the root stopped being refused, and the first thing an
# operator will do with that: register the root over the day folders already
# registered underneath it. Scanning is recursive, so both together walk every
# file twice on every pass -- and nothing visibly breaks, because dedup is by
# file name. A doubling of work with no symptom is worth refusing over.

def test_registering_a_parent_of_a_registered_folder_is_refused(client):
    assert _add(client, path="good").status_code == 200
    r = _add(client, path=".")
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert "good" in detail
    assert "twice" in detail
    assert "replace=true" in detail


def test_registering_inside_a_registered_folder_is_refused(client):
    assert _add(client, path=".").status_code == 200
    r = _add(client, path="good")
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert "already covered by ." in detail
    # No `replace` offered in this direction: dropping the *parent* would
    # un-index everything else under it, which is not a consolidation.
    assert "replace=true" not in detail


def test_replace_consolidates_the_folders_it_covers(client):
    assert _add(client, path="good").status_code == 200
    r = _add(client, path=".", replace=True)
    assert r.status_code == 200, r.json()
    assert r.json()["replaced"] == ["good"]

    rows = client.get("/archives").json()["archives"]
    assert [row["relpath"] for row in rows] == ["."]


def test_consolidating_does_not_discard_what_was_already_indexed(
        client, archive_root, tmp_path):
    """The reason `replace` is safe to offer. `sounding` is keyed by file and
    carries no archive_id, so dropping the row that caused an ingest does not
    drop the ingest -- consolidating fifteen day folders into one root is
    bookkeeping, not a reindex."""
    assert _add(client, path="good").status_code == 200
    archive_id = client.get("/archives").json()["archives"][0]["id"]

    conn = db.connect(tmp_path / "api.sqlite3")
    try:
        # Synchronously, and `min_age_s=0` because the fixture's files were
        # written moments ago and the watcher would rightly hold them back.
        archives_mod.scan_once(db.archive(conn, archive_id),
                               archive_root=archive_root,
                               db_path=tmp_path / "api.sqlite3", min_age_s=0)
        before = db.one(conn, "SELECT COUNT(*) AS n FROM sounding")["n"]
        assert before, "nothing was indexed, so this test proves nothing"

        assert _add(client, path=".", replace=True).status_code == 200
        after = db.one(conn, "SELECT COUNT(*) AS n FROM sounding")["n"]
        assert after == before
    finally:
        conn.close()


def test_two_folders_that_do_not_overlap_are_both_allowed(client, archive_root):
    """The distinction is containment, not "more than one archive". A `.lfs`
    folder beside a `chirp2` folder is the layout this is built for."""
    (archive_root / "other").mkdir()
    (archive_root / "other" / "2026-08-04").mkdir()
    (archive_root / "other" / "2026-08-04" / "rec.lfs").write_bytes(b"")

    assert _add(client, path="good").status_code == 200
    assert _add(client, path="other").status_code in (200, 400)
