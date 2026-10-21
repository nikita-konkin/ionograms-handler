"""The public page's pull, kept in step with the vhost that serves it.

The page can be served by a host other than the hub, which pulls the hub's
`PUBLISH_DIR` over the VPN (deploy/README.md, "2d. Public page on another
host"). Three files make that work and none of them can see the others: the
timer, the rsync it starts, and the nginx vhost. Every mistake below leaves
each file valid on its own and the public page broken, stale or unsafe -- and
the host it runs on is the one place in this rig that faces the internet.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"
SERVICE = DEPLOY / "ionograms-public-pull.service"
TIMER = DEPLOY / "ionograms-public-pull.timer"
VHOST = DEPLOY / "public-site.nginx.conf"


def settings(unit: Path) -> dict[str, str]:
    """A unit's directives, continuation lines joined, comments skipped."""
    found, line = {}, ""
    for raw in unit.read_text(encoding="utf-8").splitlines():
        line += raw.strip()
        if line.endswith("\\"):
            line = line[:-1] + " "
            continue
        if line and not line.startswith(("#", ";", "[")) and "=" in line:
            key, _, value = line.partition("=")
            found[key.strip()] = value.strip()
        line = ""
    return found


def pull() -> list[str]:
    """The rsync command line the service runs, as rsync receives it."""
    return shlex.split(settings(SERVICE)["ExecStart"])


def served() -> str:
    """The folder the public vhost serves."""
    return re.search(r"^\s*root\s+([^;\s]+)\s*;", VHOST.read_text(encoding="utf-8"),
                     re.MULTILINE).group(1)


def test_the_pull_lands_where_the_vhost_serves():
    assert pull()[-1].rstrip("/") == served().rstrip("/"), (
        "the pull does not write the folder public-site.nginx.conf serves, so "
        "the public page would never change")


def test_a_pass_reaches_the_public_folder_all_at_once():
    assert "--delay-updates" in pull(), (
        "the pull does not hold a pass's files back until all have arrived, so "
        "a visitor can load a page that names an image not yet copied")


def test_what_the_hub_pruned_goes_only_after_the_new_files_are_in():
    assert {a for a in pull() if a.startswith("--delete")} == {"--delete-delay"}, (
        "the pull does not delay its deletions, so an image can vanish while "
        "the page still on the public host refers to it")


def test_the_pull_skips_what_the_publisher_is_still_writing():
    from services.publish.publisher import TMP_PREFIX

    assert f"--exclude={TMP_PREFIX}*" in pull(), (
        "the pull does not skip the publisher's half-written files, so a pass "
        "can copy one mid-write or fail when it is renamed away")


def test_the_pull_copies_no_symlink():
    linking = [a for a in pull() if a in ("--archive", "--links", "--copy-links",
                                          "--copy-unsafe-links")
               or re.fullmatch(r"-[A-Za-z]*[alL][A-Za-z]*", a)]
    assert not linking, (
        f"the pull keeps or follows symlinks ({linking}), so nginx on the public "
        "host could be made to serve a file outside the published folder")


def test_the_pull_refuses_a_hub_it_has_not_met():
    assert "StrictHostKeyChecking=yes" in " ".join(pull()), (
        "the pull does not insist on the hub's known host key, so anything "
        "answering on the VPN address could feed the public page")


def test_a_stalled_vpn_cannot_hold_the_pull_forever():
    assert settings(SERVICE).get("TimeoutStartSec", "infinity") != "infinity", (
        "the pull has no time limit, and a oneshot left hanging on a dead VPN "
        "stops the timer from starting the next one, so the page goes stale "
        "with nothing failed")


def test_the_timer_starts_the_pull():
    assert settings(TIMER).get("Unit", TIMER.stem + ".service") == SERVICE.name, (
        "the timer does not start ionograms-public-pull.service, so nothing "
        "would ever pull")
