"""The object store: an artifact lives under its own hash, not under a name.

``artifacts.sha256`` already states the problem this solves -- "a models volume
is shared and writable by the training job, so a file at a given path is not
necessarily the file that was registered there" -- and until now the registry
recorded the hash and then went on resolving models by path anyway. Uploads
make that worse, because every operator picks their own filename, and training
makes it worse again, because every run wants one.

So the hash becomes the address::

    /models/objects/<aa>/<the full 64-hex digest>

which is deliberately **DVC's cache layout**. DVC itself is not used here: its
unit of work is a developer's git commit against a configured remote, and what
happens in this service is an operator uploading to, or training on, a running
server. The registry already does the part DVC could not -- input contracts,
golden checks, measured-versus-modelled provenance, and a promotion rule that
is a schema CHECK rather than a convention. Matching the layout costs nothing
and means a ``dvc remote`` could be pointed at this directory later without
moving a byte.

**Objects are written once and then read-only** (mode 0444). The workers mount
the volume read-write because they have to add to it; that is not a reason for
them to be able to replace something already in it. Same argument
``Dockerfile.infer`` gives for mounting ``models:ro`` under the process that
runs code out of these files.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .artifacts import sha256

#: Where the store lives. The volume mount point in every container that has
#: one; overridable so tests and a workstation run do not need `/models`.
ROOT = Path(os.environ.get("MODEL_STORE", "/models"))

#: The subdirectory holding content-addressed objects. Everything *outside* it
#: on the same volume is the legacy flat layout: files an operator dropped
#: there by hand and registered by path. Those keep working -- see `resolve`.
OBJECTS = "objects"

#: A digest is 64 lowercase hex characters. Checked before it is ever used to
#: build a path, because a digest is interpolated into a filesystem path and a
#: caller-supplied one that is not a digest is a path traversal.
DIGEST_LENGTH = 64


class StoreError(RuntimeError):
    """The store could not satisfy the request."""


def root() -> Path:
    """The store root, read at call time.

    Not captured at import: the environment variable is set per container, and
    tests move it between cases.
    """
    return Path(os.environ.get("MODEL_STORE", str(ROOT)))


def _checked(digest: str) -> str:
    digest = (digest or "").strip().lower()
    if len(digest) != DIGEST_LENGTH or not all(c in "0123456789abcdef" for c in digest):
        raise StoreError(
            f"not a sha-256 digest: {digest!r}. The digest becomes a path, so "
            f"anything else is refused here rather than resolved.")
    return digest


def path_for(digest: str) -> Path:
    """Where an object with this digest belongs. Does not mean it is there."""
    digest = _checked(digest)
    return root() / OBJECTS / digest[:2] / digest


def has(digest: str) -> bool:
    return path_for(digest).exists()


def put(source: str | Path, digest: str) -> Path:
    """Place a file in the store under its digest. Returns the object path.

    Idempotent, and that matters more than it looks: two uploads of the same
    bytes, or a training run that reproduces an earlier one, must converge on
    one object rather than racing to overwrite it. An object already present is
    left exactly as it is -- its content is its name, so there is nothing a
    second copy could correct.

    The write is a temporary file in the destination directory followed by
    ``os.replace``, so a reader never sees a partially copied artifact. Same
    directory because ``os.replace`` is only atomic within one filesystem.
    """
    source = Path(source)
    digest = _checked(digest)
    target = path_for(digest)

    if target.exists():
        return target

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise StoreError(
            f"cannot create {target.parent}: {exc}. The model store is "
            f"{root()}; this process needs it mounted read-write."
        ) from exc

    handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=".incoming-")
    temporary_path = Path(temporary)
    try:
        with open(source, "rb") as reader, os.fdopen(handle, "wb") as writer:
            for block in iter(lambda: reader.read(1 << 20), b""):
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
        # Read-only before it is visible, not after: a window in which the
        # object exists and is still writable is the window this prevents.
        os.chmod(temporary_path, 0o444)
        os.replace(temporary_path, target)
    except OSError as exc:
        temporary_path.unlink(missing_ok=True)
        raise StoreError(f"could not store {source} as {digest}: {exc}") from exc

    return target


def verify(digest: str) -> bool:
    """Re-hash a stored object and confirm it is still what its name says."""
    target = path_for(digest)
    if not target.exists():
        return False
    return sha256(target) == _checked(digest)


def unlink(digest: str) -> bool:
    """Remove an object. For an import that was refused, not for retirement.

    Retiring a model keeps its artifact: ``registry.retire`` says so, and
    re-activating it is how a promotion is rolled back. This is only for an
    object that was stored moments ago and whose registration then failed, so
    that a rejected artifact leaves nothing behind.
    """
    target = path_for(digest)
    if not target.exists():
        return False
    # 0444 means the file is not writable; the *directory* is what permits
    # unlinking, so no chmod is needed here.
    target.unlink()
    return True


def resolve(artifact: str | Path) -> Path:
    """The path to load, given whatever a registry row records.

    Both shapes are live and will be for a while. A row written by the console
    path names an object in the store; a row written by the original shell
    runbook names a file somebody put on the volume by hand. Rewriting the
    latter under a running ``infer`` would be a worse failure than the
    inconsistency, so both resolve and neither is guessed at.
    """
    return Path(artifact)


def describe() -> str:
    """One line for a worker's startup banner."""
    objects = root() / OBJECTS
    if not objects.exists():
        return f"model store: {objects} (empty)"
    count = sum(1 for _ in objects.glob("*/*"))
    return f"model store: {objects}, {count} object{'' if count == 1 else 's'}"
