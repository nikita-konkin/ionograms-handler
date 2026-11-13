"""Raw deletion requires a verified replacement and a recoverable journal."""

import os
import sqlite3
import time
from pathlib import Path

import numpy as np
import pytest
from conftest import make_header_bytes

from muf import echo_window, loader
from services.api import conversion, db, watch


@pytest.fixture
def archive(tmp_path, monkeypatch):
    root = tmp_path / "archive"
    day = root / "2026.02.04"
    day.mkdir(parents=True)
    monkeypatch.setenv("ARCHIVE_ROOT", str(root))
    monkeypatch.delenv("ARCHIVE_ROOTS", raising=False)
    sr, n = 40000, 200000
    rng = np.random.default_rng(5)
    samples = (
        rng.normal(size=n)
        + 1j * rng.normal(size=n)
        + 30 * np.exp(-2j * np.pi * (2900 / 3) * np.arange(n) / sr)
    )
    source = day / "synth_20260204_000010.lfs"
    source.write_bytes(make_header_bytes() + samples.astype(np.complex64).tobytes())
    old = time.time() - 120
    os.utime(source, (old, old))
    with db.session(tmp_path / "db.sqlite3") as conn:
        archive_id = db.add_archive(
            conn, name="captures", relpath=".", format="lfs", methods="contour"
        )
        yield conn, root, source, archive_id


def reviewed(archive):
    conn, root, source, archive_id = archive
    job = conversion.enqueue(
        conn, archive_id, source.parent.relative_to(root).as_posix(), "admin"
    )
    assert job["mode"] == "survey"
    assert conversion.run_next(conn)["state"] == "ready"
    return conversion.start(conn, job["id"], True)


def index_raw(conn, root, source):
    cur = conn.execute(
        "INSERT INTO sounding(file,path,format,datetime,ingested_at)"
        " VALUES (?,?,'lfs','2026-02-04 00:00:10',?)",
        (source.name, source.relative_to(root).as_posix(), db.utcnow()),
    )
    conn.execute(
        "INSERT INTO extraction(sounding_id,method,muf) VALUES (?,'contour',18.5)",
        (cur.lastrowid,),
    )
    conn.execute(
        "INSERT INTO reference(sounding_id,source,param,value)"
        " VALUES (?,'iri','muf',17.0)",
        (cur.lastrowid,),
    )
    conn.commit()
    return cur.lastrowid


def test_inspection_is_read_only_and_requires_explicit_replacement(archive):
    conn, root, source, archive_id = archive
    checksum = conversion.digest(source)
    job = conversion.enqueue(conn, archive_id, ".", "admin")
    job = conversion.run_next(conn)
    assert job["state"] == "ready" and job["total"] == 1
    assert job["source_bytes"] == source.stat().st_size
    assert conversion.digest(source) == checksum
    assert not list(root.rglob("*.echo.h5"))
    with pytest.raises(conversion.ConversionError, match="confirm"):
        conversion.start(conn, job["id"], False)


def test_verified_replacement_removes_raw_and_preserves_measurements(archive):
    conn, root, source, archive_id = archive
    sounding_id = index_raw(conn, root, source)
    reviewed(archive)
    result = conversion.run_next(conn)
    assert result["state"] == "completed", result
    assert (result["done"], result["failed"]) == (1, 0)
    assert not source.exists()
    target = source.with_suffix(".echo.h5")
    assert target.is_file()
    row = db.one(conn, "SELECT * FROM sounding WHERE id = ?", (sounding_id,))
    assert row["file"] == target.name and row["format"] == "echo_window"
    assert root / row["path"] == target
    assert (
        db.one(
            conn, "SELECT muf FROM extraction WHERE sounding_id = ?", (sounding_id,)
        )["muf"]
        == 18.5
    )
    assert (
        db.one(
            conn, "SELECT value FROM reference WHERE sounding_id = ?", (sounding_id,)
        )["value"]
        == 17.0
    )
    assert loader.load(target).power.size
    assert loader.find_soundings(root) == [target]
    assert db.archive(conn, archive_id)["format"] == "lfs"
    assert db.archive_orphans(conn, archive_id)["total"] == 0
    assert loader.find_soundings(root, format="lfs") == [target]
    assert (
        watch.find_new(
            [root], conn, ("contour",), 0, now=time.time() + 120, archive_root=root
        )[0]
        == []
    )


def test_verification_failure_preserves_raw(archive, monkeypatch):
    conn, _, source, _ = archive
    job = reviewed(archive)
    before = conversion.digest(source)

    def fail(*args, **kwargs):
        raise echo_window.VerificationError("mismatched samples")

    monkeypatch.setattr(echo_window, "write", fail)
    result = conversion.run_next(conn)
    assert result["state"] == "failed" and result["failed"] == 1
    assert conversion.digest(source) == before
    assert (
        "mismatched"
        in db.one(
            conn,
            "SELECT detail FROM archive_conversion_file WHERE job_id = ?",
            (job["id"],),
        )["detail"]
    )


def test_changed_capture_is_not_deleted(archive):
    conn, _, source, _ = archive
    reviewed(archive)
    source.write_bytes(source.read_bytes() + b"more samples")
    result = conversion.run_next(conn)
    assert result["failed"] == 1 and source.exists()
    assert not source.with_suffix(".echo.h5").exists()


def test_foreign_existing_output_is_not_overwritten_or_used_for_deletion(archive):
    conn, _, source, _ = archive
    target = source.with_suffix(".echo.h5")
    target.write_bytes(b"unrelated existing file")
    reviewed(archive)
    result = conversion.run_next(conn)
    assert result["failed"] == 1 and source.exists()
    assert target.read_bytes() == b"unrelated existing file"


def test_possible_cropped_echo_keeps_the_raw(archive, monkeypatch):
    conn, _, source, _ = archive
    reviewed(archive)
    read = echo_window.read

    def flagged(*args, **kwargs):
        window = read(*args, **kwargs)
        window.trace_at_edge = True
        return window

    monkeypatch.setattr(echo_window, "read", flagged)
    result = conversion.run_next(conn)
    assert result["state"] == "failed" and source.exists()


def test_database_error_before_unlink_keeps_raw(archive, monkeypatch):
    conn, _, source, _ = archive
    reviewed(archive)

    def busy(*args):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(conversion, "_relink", busy)
    result = conversion.run_next(conn)
    assert result["state"] == "failed" and source.exists()


def test_restart_recovers_after_unlink_before_database_commit(archive, monkeypatch):
    conn, root, source, _ = archive
    sounding_id = index_raw(conn, root, source)
    job = reviewed(archive)
    unlink = Path.unlink

    def interrupted(path, *args, **kwargs):
        unlink(path, *args, **kwargs)
        if path == source:
            raise OSError("power loss after unlink")

    monkeypatch.setattr(Path, "unlink", interrupted)
    result = conversion.run_next(conn)
    assert result["state"] == "failed" and not source.exists()
    assert (
        db.one(conn, "SELECT file FROM sounding WHERE id = ?", (sounding_id,))["file"]
        == source.name
    )
    monkeypatch.setattr(Path, "unlink", unlink)
    conversion.start(conn, job["id"], True)
    resumed = conversion.run_next(conn)
    assert resumed["state"] == "completed", resumed
    assert (
        db.one(conn, "SELECT file FROM sounding WHERE id = ?", (sounding_id,))["file"]
        == source.with_suffix(".echo.h5").name
    )


def test_cancel_stops_before_the_next_capture(archive, monkeypatch):
    conn, _, source, _ = archive
    second = source.with_name("second.lfs")
    second.write_bytes(source.read_bytes())
    os.utime(second, (time.time() - 120,) * 2)
    job = reviewed(archive)
    replace = conversion.replace_one

    def replace_and_cancel(*args):
        replace(*args)
        conversion.cancel(conn, job["id"])

    monkeypatch.setattr(conversion, "replace_one", replace_and_cancel)
    result = conversion.run_next(conn)
    assert result["state"] == "cancelled" and result["done"] == 1
    assert sum(path.exists() for path in (source, second)) == 1


def test_paths_outside_registered_archive_are_refused(archive):
    conn, root, _, archive_id = archive
    for subpath in ("../elsewhere", str(root.parent)):
        with pytest.raises(conversion.ConversionError):
            conversion.enqueue(conn, archive_id, subpath, "admin")
    assert conversion.listing(conn) == []


def test_only_one_job_can_be_active(archive):
    conn, _, _, archive_id = archive
    conversion.enqueue(conn, archive_id, ".", "admin")
    with pytest.raises(conversion.ConversionError, match="another"):
        conversion.enqueue(conn, archive_id, ".", "admin")


def test_discovery_does_not_double_count_raw_and_derived(archive):
    _, root, source, _ = archive
    target = echo_window.write(source).target
    assert loader.find_soundings(root) == [source]
    assert loader.find_soundings(target) == [target]
    with pytest.raises(FileNotFoundError):
        loader.find_soundings(root, format="echo_window")
    assert not loader.has_soundings(root, format="echo_window")
    source.unlink()
    assert loader.has_soundings(root, format="echo_window")
    assert loader.find_soundings(root) == [target]


def test_lfs_filter_keeps_replacements_without_admitting_other_formats(archive):
    from services.api import archives

    conn, root, source, archive_id = archive
    index_raw(conn, root, source)
    # A non-root registration also exercises the orphan path-prefix check.
    with conn:
        conn.execute(
            "UPDATE archive SET relpath = ? WHERE id = ?",
            (source.parent.relative_to(root).as_posix(), archive_id),
        )
    job = conversion.enqueue(conn, archive_id, ".", "admin")
    conversion.run_next(conn)
    conversion.start(conn, job["id"], True)
    assert conversion.run_next(conn)["state"] == "completed"
    unrelated = source.with_name("lfm_ionogram-other.h5")
    unrelated.touch()
    assert loader.find_soundings(root, format="lfs") == [source.with_suffix(".echo.h5")]
    assert archives.survey(root, format="lfs")["by_format"] == {"echo_window": 1}
    with conn:
        conn.execute(
            "INSERT INTO sounding(file,path,format,datetime,ingested_at)"
            " VALUES (?,?,'chirp2','2026-02-04 00:00:10',?)",
            (unrelated.name, unrelated.relative_to(root).as_posix(), db.utcnow()),
        )
    assert db.archive_orphans(conn, archive_id)["by_format"] == {"chirp2": 1}
    assert db.delete_archive_orphans(conn, archive_id) == 1
    assert db.one(conn, "SELECT COUNT(*) AS n FROM sounding")["n"] == 1


def test_an_unindexed_replacement_can_be_ingested_by_the_archive_pipeline(archive):
    from muf import pipeline
    from services.api import ingest

    conn, root, _, _ = archive
    reviewed(archive)
    assert conversion.run_next(conn)["state"] == "completed"
    result = ingest.ingest(
        [root],
        conn,
        options=pipeline.Options(methods=("contour",)),
        archive_root=root,
        jobs=1,
        progress=False,
    )
    assert result["loaded"] == 1, result
    row = db.one(conn, "SELECT file, format FROM sounding")
    assert row["file"].endswith(".echo.h5") and row["format"] == "echo_window"


def test_existing_hdf5_without_an_echo_group_is_not_modified(archive):
    import h5py

    conn, _, source, _ = archive
    target = source.with_suffix(".echo.h5")
    with h5py.File(target, "w") as handle:
        handle.create_dataset("unrelated", data=[1, 2, 3])
    checksum = conversion.digest(target)
    reviewed(archive)
    assert conversion.run_next(conn)["failed"] == 1
    assert conversion.digest(target) == checksum and source.exists()


def test_manual_existing_window_cannot_authorize_raw_deletion(archive):
    conn, _, source, _ = archive
    target = echo_window.write(source, manual_km=(2600, 3300), verify=True).target
    checksum = conversion.digest(target)
    reviewed(archive)
    assert conversion.run_next(conn)["failed"] == 1
    assert source.exists() and conversion.digest(target) == checksum


def test_echo_with_external_hdf5_samples_cannot_replace_raw(archive, tmp_path):
    import h5py

    conn, _, source, _ = archive
    target = echo_window.write(source).target
    external = tmp_path / "external-samples.h5"
    with h5py.File(target, "r+") as handle, h5py.File(external, "w") as other:
        other.create_dataset("samples", data=handle["echo_window/z"][()])
        del handle["echo_window/z"]
        handle["echo_window/z"] = h5py.ExternalLink(str(external), "/samples")
    checksum = conversion.digest(target)
    reviewed(archive)
    assert conversion.run_next(conn)["failed"] == 1
    assert source.exists() and conversion.digest(target) == checksum


def test_output_symlink_is_never_followed(archive, tmp_path):
    conn, _, source, _ = archive
    other = tmp_path / "unrelated.h5"
    other.write_bytes(b"other data")
    target = source.with_suffix(".echo.h5")
    try:
        target.symlink_to(other)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    reviewed(archive)
    assert conversion.run_next(conn)["failed"] == 1
    assert source.exists() and other.read_bytes() == b"other data"


def test_failed_durable_flush_keeps_the_original(archive, monkeypatch):
    conn, _, source, _ = archive
    reviewed(archive)

    def unavailable(*args):
        raise OSError("flush failed")

    monkeypatch.setattr(conversion, "_durable_target", unavailable)
    assert conversion.run_next(conn)["failed"] == 1
    assert source.exists()


def test_watch_skips_an_incomplete_replacement_journal(archive, monkeypatch):
    conn, root, source, _ = archive
    index_raw(conn, root, source)
    reviewed(archive)
    unlink = Path.unlink

    def interrupted(path, *args, **kwargs):
        unlink(path, *args, **kwargs)
        if path == source:
            raise OSError("interrupted after unlink")

    monkeypatch.setattr(Path, "unlink", interrupted)
    assert conversion.run_next(conn)["state"] == "failed"
    # A concurrent indexer must not create a second sounding during recovery.
    assert watch.find_new([root], conn, ("contour",), 0, archive_root=root)[0] == []


@pytest.fixture
def client(archive, monkeypatch):
    from fastapi.testclient import TestClient

    from services.api import archives, auth, main, net

    conn, _, _, _ = archive
    database_path = conn.execute("PRAGMA database_list").fetchone()[2]
    monkeypatch.setattr(db, "DEFAULT_DB", Path(database_path))
    monkeypatch.setattr(auth, "READ_TOKEN", "")
    monkeypatch.setattr(auth, "CONTROL_TOKEN", "ctl")
    monkeypatch.setattr(archives, "DEFAULT_INTERVAL_S", 0)
    monkeypatch.setattr(main, "WARM_CENSUS", False)
    monkeypatch.setattr(net, "ENABLED", False)
    with TestClient(main.app) as api:
        yield api


AUTH = {"Authorization": "Bearer ctl"}


def test_conversion_tab_is_linked_and_translated(client):
    english = client.get("/ui/conversion?lang=en")
    russian = client.get("/ui/conversion?lang=ru")
    assert english.status_code == russian.status_code == 200
    assert 'href="/ui/conversion"' in english.text
    assert "LFS to echo archive" in english.text
    assert "Преобразование LFS" in russian.text


def test_job_controls_and_file_lists_require_archive_permission(client, archive):
    _, _, _, archive_id = archive
    assert client.get("/archive-conversions").status_code == 401
    assert (
        client.post("/archive-conversions", json={"archive_id": archive_id}).status_code
        == 401
    )
    teacher = client.post(
        "/principals", headers=AUTH, json={"name": "teacher", "role": "teacher"}
    ).json()["token"]
    assert (
        client.get(
            "/archive-conversions", headers={"Authorization": "Bearer " + teacher}
        ).status_code
        == 403
    )


def test_browser_can_inspect_review_start_and_read_the_result(client, archive):
    conn, _, source, archive_id = archive
    response = client.post(
        "/archive-conversions",
        headers=AUTH,
        json={"archive_id": archive_id, "subpath": "."},
    )
    assert response.status_code == 202, response.text
    job_id = response.json()["id"]
    assert conversion.run_next(conn)["state"] == "ready"
    details = client.get(f"/archive-conversions/{job_id}", headers=AUTH).json()
    assert len(details["files"]) == 1 and source.exists()
    assert (
        client.post(
            f"/archive-conversions/{job_id}/start", headers=AUTH, json={}
        ).status_code
        == 409
    )
    assert (
        client.post(
            f"/archive-conversions/{job_id}/start",
            headers=AUTH,
            json={"confirm_delete": True},
        ).status_code
        == 202
    )
    assert conversion.run_next(conn)["state"] == "completed"
    details = client.get(f"/archive-conversions/{job_id}", headers=AUTH).json()
    assert details["files"][0]["state"] == "replaced" and not source.exists()
    assert details["job"]["approved_by"] == "control"


def test_browser_can_cancel_queued_job_without_touching_files(client, archive):
    _, _, source, archive_id = archive
    job = client.post(
        "/archive-conversions", headers=AUTH, json={"archive_id": archive_id}
    ).json()
    response = client.post(f"/archive-conversions/{job['id']}/cancel", headers=AUTH)
    assert response.json()["state"] == "cancelled" and source.exists()


def test_process_lock_excludes_a_second_worker(tmp_path):
    from services.archive_converter import worker_lock

    path = tmp_path / "conversion.lock"
    with worker_lock(path), pytest.raises(OSError), worker_lock(path):
        pytest.fail("second worker acquired the active worker's lock")
    with worker_lock(path):
        pass
