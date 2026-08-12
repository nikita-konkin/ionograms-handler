"""Every patch in patches/ must be a diff `git apply` will accept.

These files are prose first and diff second -- the rationale above the diff is
most of their value, and it gets edited far more often than the diff does. That
is exactly how a hunk breaks: 0004 had its comment grown from five lines to six
during a rewrite, and the `@@` header kept saying eleven. `git apply` rejects
that outright ("corrupt patch at line N"), and nobody notices, because these are
applied by hand on a station in Norway and not by CI.

The test mirrors git's own parser: consume exactly the counts the header
promises, then require the hunk to end. Note that a *bare* empty line is legal
context -- editors strip the trailing space off " " and git tolerates it -- so
it counts on both sides rather than terminating the body.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PATCH_DIR = Path(__file__).resolve().parent.parent / "patches"
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _patches() -> list[Path]:
    return sorted(PATCH_DIR.glob("*.patch"))


def test_patch_directory_is_not_empty() -> None:
    # Guards against the glob silently matching nothing, which would turn every
    # parametrised test below into a pass.
    assert _patches(), f"no *.patch files under {PATCH_DIR}"


@pytest.mark.parametrize("patch", _patches(), ids=lambda p: p.name)
def test_hunk_headers_match_their_bodies(patch: Path) -> None:
    lines = patch.read_text().split("\n")

    starts = [i for i, line in enumerate(lines) if line.startswith("diff --git")]
    assert starts, f"{patch.name}: no `diff --git` line -- patch has no diff at all"

    hunks = 0
    i = starts[0]
    while i < len(lines):
        m = HUNK_RE.match(lines[i])
        if m is None:
            i += 1
            continue

        hunks += 1
        header, at = lines[i], i + 1
        old_want = int(m.group(2)) if m.group(2) is not None else 1
        new_want = int(m.group(4)) if m.group(4) is not None else 1
        old = new = 0

        while (old, new) != (old_want, new_want):
            assert at < len(lines), (
                f"{patch.name}:{i + 1}: {header!r} ran off the end of the file "
                f"with old={old}/{old_want} new={new}/{new_want}"
            )
            line = lines[at]
            kind = line[:1]
            if kind == "+":
                new += 1
            elif kind == "-":
                old += 1
            elif kind in (" ", "") or kind == "\\":
                # "" is a blank context line with its leading space stripped;
                # "\" is git's "\ No newline at end of file" marker.
                if kind != "\\":
                    old += 1
                    new += 1
            else:
                pytest.fail(
                    f"{patch.name}:{at + 1}: line in hunk body starts with "
                    f"{kind!r}, which git reads as the end of the hunk -- "
                    f"{header!r} is short by old={old_want - old} "
                    f"new={new_want - new}. Line: {line!r}"
                )
            assert old <= old_want and new <= new_want, (
                f"{patch.name}:{at + 1}: {header!r} promises "
                f"{old_want}/{new_want} lines but the body has more "
                f"(old={old} new={new}). git rejects this as a corrupt patch."
            )
            at += 1

        # Counts are satisfied: the hunk must end here.
        if at < len(lines):
            nxt = lines[at]
            assert nxt == "" or nxt.startswith(("@@", "diff ")), (
                f"{patch.name}:{at + 1}: {header!r} is satisfied at "
                f"{old_want}/{new_want} lines but the body continues with "
                f"{nxt!r} -- git rejects this as a corrupt patch."
            )
        i = at

    assert hunks, f"{patch.name}: `diff --git` present but no @@ hunk follows it"


@pytest.mark.parametrize("patch", _patches(), ids=lambda p: p.name)
def test_hunk_offsets_accumulate(patch: Path) -> None:
    """A hunk's new-side start must reflect what earlier hunks added or removed.

    git is forgiving here -- it locates by the old side and reports an offset --
    but a wrong number means the patch is describing a file that does not exist,
    and the next person to hand-edit reads it as truth.
    """
    lines = patch.read_text().split("\n")
    drift = 0
    for n, line in enumerate(lines):
        m = HUNK_RE.match(line)
        if m is None:
            continue
        old_start = int(m.group(1))
        old_count = int(m.group(2)) if m.group(2) is not None else 1
        new_start = int(m.group(3))
        new_count = int(m.group(4)) if m.group(4) is not None else 1

        assert new_start == old_start + drift, (
            f"{patch.name}:{n + 1}: {line.split('@@')[1].strip()!r} starts the "
            f"new side at {new_start}, but earlier hunks shift this file by "
            f"{drift:+d}, so it should be {old_start + drift}."
        )
        drift += new_count - old_count
