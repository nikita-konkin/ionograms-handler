"""The content-addressed object store.

Small enough to read in one sitting, and load-bearing enough to be worth
pinning: every artifact the console accepts, and every one the trainer
produces, is addressed by these forty lines. The properties that matter are
that an object's name is its content, that placing one is idempotent, and that
a reader never sees a half-written file.
"""

from __future__ import annotations

import os

import pytest

from services.prediction import artifacts, store


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_STORE", str(tmp_path / "models"))
    return tmp_path / "models"


@pytest.fixture
def blob(tmp_path):
    path = tmp_path / "artifact.sav"
    path.write_bytes(b"\x80\x04not really a pickle, but bytes are bytes")
    return path


def test_the_layout_is_two_hex_then_the_whole_digest(root):
    digest = "ab" + "c" * 62
    path = store.path_for(digest)
    assert path == root / "objects" / "ab" / digest
    # The fan-out directory is a prefix of the name, not a truncation of it:
    # `dvc` reads this layout, and so does anyone who has ever opened .git.
    assert path.name.startswith(path.parent.name)


def test_a_digest_that_is_not_one_is_refused_before_it_becomes_a_path(root):
    """The digest is interpolated into a filesystem path.

    A caller-supplied value that is not a digest is therefore a traversal, and
    it is refused here rather than resolved and then regretted.
    """
    for bad in ("", "../../etc/passwd", "abc", "z" * 64, "AB" + "c" * 62 + "d"):
        with pytest.raises(store.StoreError):
            store.path_for(bad)


def test_putting_a_file_addresses_it_by_its_own_hash(root, blob):
    digest = artifacts.sha256(blob)
    stored = store.put(blob, digest)

    assert stored == store.path_for(digest)
    assert stored.read_bytes() == blob.read_bytes()
    assert store.has(digest)
    assert store.verify(digest)


def test_a_stored_object_is_read_only(root, blob):
    """The workers mount the volume read-write because they have to add to it.

    That is not a reason for them to be able to replace something already in
    it -- the same argument `Dockerfile.infer` makes for mounting `models:ro`
    under the process that runs code out of these files.
    """
    digest = artifacts.sha256(blob)
    stored = store.put(blob, digest)
    assert oct(os.stat(stored).st_mode & 0o777) == "0o444"

    with pytest.raises(OSError):
        stored.open("wb")


def test_putting_the_same_bytes_twice_converges_on_one_object(root, blob):
    """Two uploads of the same file, or a training run that reproduces an
    earlier one, must not race to overwrite each other."""
    digest = artifacts.sha256(blob)
    first = store.put(blob, digest)
    before = os.stat(first).st_mtime_ns

    second = store.put(blob, digest)

    assert second == first
    assert os.stat(second).st_mtime_ns == before, "the second put rewrote it"
    assert len(list((root / "objects").glob("*/*"))) == 1


def test_nothing_partial_is_ever_visible_under_the_final_name(root, tmp_path,
                                                              monkeypatch):
    """A failed copy leaves no object, and no temporary file either."""
    source = tmp_path / "big.sav"
    source.write_bytes(b"\x80\x04" + b"x" * 4096)
    digest = artifacts.sha256(source)

    def explode(*args, **kwargs):
        raise OSError("the volume filled up")

    monkeypatch.setattr(store.os, "replace", explode)
    with pytest.raises(store.StoreError):
        store.put(source, digest)

    assert not store.has(digest)
    assert list(store.path_for(digest).parent.glob("*")) == []


def test_verify_catches_an_object_that_is_no_longer_what_it_claims(root, blob):
    digest = artifacts.sha256(blob)
    stored = store.put(blob, digest)
    assert store.verify(digest)

    stored.chmod(0o644)
    stored.write_bytes(b"\x80\x04different bytes entirely")
    assert not store.verify(digest)


def test_unlink_removes_a_refused_import_and_says_so(root, blob):
    digest = artifacts.sha256(blob)
    store.put(blob, digest)

    assert store.unlink(digest) is True
    assert not store.has(digest)
    # Idempotent: the registrar calls it on a path that may already be gone.
    assert store.unlink(digest) is False


def test_resolve_keeps_both_shapes_working(root, blob):
    """A row written by the console names an object; a row written by the
    original shell runbook names a file somebody put on the volume by hand.

    Rewriting the second kind under a running `infer` would be a worse failure
    than the inconsistency, so both resolve and neither is guessed at.
    """
    digest = artifacts.sha256(blob)
    stored = store.put(blob, digest)

    assert store.resolve(str(stored)) == stored
    assert store.resolve(str(blob)) == blob


def test_describe_counts_what_is_there(root, blob, tmp_path):
    assert "empty" in store.describe()

    store.put(blob, artifacts.sha256(blob))
    assert "1 object" in store.describe()

    other = tmp_path / "second.sav"
    other.write_bytes(b"PK\x03\x04and another")
    store.put(other, artifacts.sha256(other))
    assert "2 objects" in store.describe()
