"""Every unit in services/agent/systemd must load on the systemd DOB runs.

DOB is Ubuntu 16.04 with **systemd 229**, and that is not a detail you find out
gently. An unknown *key* is only a warning, so a unit using a newer directive
loads, starts, and fails somewhere downstream that looks like a broken script:
`chirp-drop-watch` died on `mkdir` as an unprivileged user because
`StateDirectory=` (235) was silently ignored and nothing created its directory.
An unknown *value* or prefix is worse -- `ExecStartPre=+/sbin/ethtool` needs 231,
and on 229 `+/sbin/ethtool` is simply not an absolute path, so the unit fails to
load and the recorder never starts. `chirp-rx.service` carried that from the day
it was written and nobody noticed, because nothing had installed it yet.

These units are applied by hand on a station in Norway, so the cost of finding
out in production is a person driving to Dombas.

Raise ``MIN_SYSTEMD`` when the station is upgraded, not before -- the point of
the number is that it matches the oldest host that must run these files.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

UNIT_DIR = Path(__file__).resolve().parent.parent / "services/agent/systemd"

#: systemd on DOB (Ubuntu 16.04). Verified with `systemctl --version`.
MIN_SYSTEMD = 229

#: Directive -> the systemd version that introduced it. Not exhaustive: it
#: covers what these units actually reach for, and grows when something new is
#: used. A directive absent from this map is assumed old enough.
INTRODUCED = {
    "StateDirectory": 235,
    "LogsDirectory": 235,
    "CacheDirectory": 235,
    "ConfigurationDirectory": 235,
    "RuntimeDirectoryPreserve": 235,
    "LockPersonality": 235,
    "RestrictSUIDSGID": 242,
    "ProtectKernelLogs": 244,
    "ProtectClock": 245,
    "ProtectHostname": 242,
    "ProtectKernelTunables": 232,
    "ProtectControlGroups": 232,
    "RestrictNamespaces": 233,
    "MemoryDenyWriteExecute": 231,
    "RestrictRealtime": 231,
    "DynamicUser": 232,
    "OOMPolicy": 243,
    "ExitType": 249,
    "CoredumpFilter": 246,
    # The cgroup-v2 spellings of things that already work on 229 by another
    # name. `AllowedCPUs=` is the trap here: it looks like the modern way to
    # write `CPUAffinity=`, and on 229 it is a warning and no pinning at all --
    # which is the fault patch 0008 exists to fix, silently reintroduced.
    "AllowedCPUs": 244,
    "AllowedMemoryNodes": 244,
    "CPUQuotaPeriodSec": 242,
    "IOReadBandwidthMax": 230,
    "IOWeight": 230,
}

#: Values that arrived later than the directive itself.
VALUES = {
    ("ProtectSystem", "strict"): 232,
}

#: ExecStart-family prefixes and when they became legal. `-` and `@` are
#: ancient; the rest are not, and a prefix systemd does not know is read as
#: part of the path.
PREFIX_INTRODUCED = {"+": 231, "!": 231, "!!": 231, ":": 231}

EXEC_KEYS = ("ExecStart", "ExecStartPre", "ExecStartPost", "ExecStop",
             "ExecStopPost", "ExecReload", "ExecCondition")


def _units() -> list[Path]:
    return sorted(p for p in UNIT_DIR.iterdir()
                  if p.suffix in (".service", ".timer", ".target"))


def _settings(unit: Path):
    """Yield (line number, key, value) for real directives only.

    Comments in these files are long and quote directive names constantly --
    this module's whole subject is which directive is too new -- so a naive
    scan would flag the prose explaining the rule.
    """
    for n, line in enumerate(unit.read_text().splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";", "[")):
            continue
        if "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        yield n, key.strip(), value.strip()


def test_unit_directory_is_not_empty() -> None:
    assert _units(), f"no unit files under {UNIT_DIR}"


@pytest.mark.parametrize("unit", _units(), ids=lambda p: p.name)
def test_no_directive_newer_than_the_station(unit: Path) -> None:
    for n, key, value in _settings(unit):
        since = INTRODUCED.get(key)
        assert since is None or since <= MIN_SYSTEMD, (
            f"{unit.name}:{n}: {key}= needs systemd {since}, station has "
            f"{MIN_SYSTEMD}. An unknown key is only a warning, so this unit "
            f"will load and then fail somewhere that looks unrelated.")

        since = VALUES.get((key, value))
        assert since is None or since <= MIN_SYSTEMD, (
            f"{unit.name}:{n}: {key}={value} needs systemd {since}, station "
            f"has {MIN_SYSTEMD}.")


@pytest.mark.parametrize("unit", _units(), ids=lambda p: p.name)
def test_exec_paths_are_absolute_and_prefixes_are_supported(unit: Path) -> None:
    for n, key, value in _settings(unit):
        if key not in EXEC_KEYS or not value:
            continue
        word = value.split()[0]
        prefixes = ""
        while word and word[0] in "-@+!:":
            prefixes += word[0]
            word = word[1:]

        for prefix in ("!!", "!", "+", ":"):
            if prefix in prefixes:
                since = PREFIX_INTRODUCED[prefix]
                assert since <= MIN_SYSTEMD, (
                    f"{unit.name}:{n}: the {prefix!r} prefix on {key}= needs "
                    f"systemd {since}, station has {MIN_SYSTEMD}. On the older "
                    f"version this is not a prefix -- it is part of the path, "
                    f"which then is not absolute and the unit fails to load. "
                    f"Use PermissionsStartOnly=true instead.")
                break

        assert word.startswith("/"), (
            f"{unit.name}:{n}: {key}= must be an absolute path, got {word!r}")


@pytest.mark.parametrize("unit", _units(), ids=lambda p: p.name)
def test_a_unit_that_drops_privileges_says_how_its_pre_steps_get_them(unit: Path):
    """Root-only ExecStartPre under User= needs one of the two mechanisms.

    Catches the half-fix: dropping the `+` prefix for 229 compatibility and
    forgetting `PermissionsStartOnly`, which leaves the pre-step running as the
    unprivileged user and failing on permissions -- exactly the failure that
    started this.
    """
    settings = list(_settings(unit))
    keys = {key for _, key, _ in settings}
    if "User" not in keys:
        return

    root_only = [(n, value) for n, key, value in settings
                 if key == "ExecStartPre"
                 and value.split()[0].lstrip("-@").startswith(
                     ("/sbin/", "/usr/sbin/", "/bin/mkdir", "/bin/chown"))]
    if not root_only:
        return

    assert "PermissionsStartOnly" in keys, (
        f"{unit.name}: runs as User= and has ExecStartPre needing root "
        f"({root_only[0][1]!r} at line {root_only[0][0]}), but sets neither "
        f"PermissionsStartOnly=true nor a `+` prefix. The pre-step will run "
        f"unprivileged and fail.")


@pytest.mark.parametrize("unit", _units(), ids=lambda p: p.name)
def test_mpirun_under_cpuaffinity_disables_its_own_binding(unit: Path) -> None:
    """CPUAffinity= does not survive mpirun, which re-pins its ranks.

    Open MPI calls sched_setaffinity on every rank after fork; 1.10 (Ubuntu
    16.04) defaults to --bind-to core for -np <= 2 and places rank 0 on physical
    core 0 from the machine topology, ignoring the mask it inherited. On DOB on
    2026-08-13 that put a detect rank and a calc rank on CPUs 0-1 -- the
    recorder's core -- under a launcher already confined to 2-7. The unit looks
    correct and the pinning is not there.
    """
    settings = list(_settings(unit))
    if not any(key == "CPUAffinity" for _, key, _ in settings):
        return

    for n, key, value in settings:
        if key not in EXEC_KEYS or "mpirun" not in value:
            continue
        assert "--bind-to" in value, (
            f"{unit.name}:{n}: {key}= runs mpirun under CPUAffinity= but sets "
            f"no --bind-to. Open MPI will re-pin each rank by topology and put "
            f"rank 0 on physical core 0, outside this mask. Use "
            f"`--bind-to none` so the ranks inherit it.")


@pytest.mark.parametrize("unit", _units(), ids=lambda p: p.name)
def test_a_directory_made_by_root_is_given_to_the_user_that_writes_it(unit: Path):
    """`PermissionsStartOnly` is per-unit, so `mkdir` in a pre-step runs as root.

    The directory it creates is then root-owned and the `User=` process cannot
    write into it. The failure surfaces inside that process -- the recorder
    dies at `mkdir -p /dev/shm/hf25/ch0` with EACCES -- so it reads as a bug in
    the program rather than as ownership, and it stays hidden for as long as
    something else happens to have created the directory first. On the station
    that something else was `dombas.sh`; the migration removed it.
    """
    settings = list(_settings(unit))
    keys = {key for _, key, _ in settings}
    if "User" not in keys:
        return

    chowned = " ".join(value for _, key, value in settings
                       if key in EXEC_KEYS and "/bin/chown" in value)
    for n, key, value in settings:
        if key != "ExecStartPre" or "/bin/mkdir" not in value:
            continue
        made = [word for word in value.split()[1:] if not word.startswith("-")]
        for path in made:
            assert path in chowned, (
                f"{unit.name}:{n}: root creates {path} and User= writes to it, "
                f"but no ExecStartPre chowns it. Add "
                f"`ExecStartPre=/bin/chown <user>:<user> {path}`.")


@pytest.mark.parametrize("unit", _units(), ids=lambda p: p.name)
def test_setting_the_nic_ring_tolerates_it_being_set_already(unit: Path) -> None:
    """`ethtool -G` exits **80** when nothing changed, which fails the start.

    Not a general rule about idempotence -- a specific exit code with a
    specific consequence. The first start of the recorder sets the ring
    256 -> 4096 and succeeds; every start after it asks for a size the ring
    already has, `ethtool` prints `no ring parameters changed, aborting` and
    returns 80, `ExecStartPre` fails, and the recorder never runs. Observed on
    the station 2026-08-16: one clean start, then `status=80` every ten seconds
    indefinitely. The unit works exactly once, which is the worst way for it to
    be wrong -- the restart path is the whole point of running under systemd.
    """
    for n, key, value in _settings(unit):
        if key not in EXEC_KEYS or "ethtool -G" not in value:
            continue
        assert "80" in value, (
            f"{unit.name}:{n}: {key}= runs `ethtool -G` without accepting exit "
            f"80. It succeeds on a ring that is not already the requested size "
            f"and fails on one that is, so this unit would start once and then "
            f"never again.")


def test_the_recorder_is_restarted_after_a_clean_exit() -> None:
    """`rx_uhd_ext_gps` ends its own run at 24 h with exit 0.

    It logs `Channel 0 finished 24h streaming.` and stops -- the timer is in
    the C++ program, not in the launcher, and `dombas.sh`'s `while true` loop
    existed to catch it. `Restart=on-failure` declines to restart a success, so
    with that policy the station records for exactly 24 h per start and then
    stops for good. Nothing reads as broken afterwards: `inactive (dead)` not
    `failed`, `status=0/SUCCESS`, and the first alarm is
    `newest_product_age_s` thirty minutes later. Yoshkar-Ola lost an afternoon
    to it on 2026-08-17.
    """
    unit = UNIT_DIR / "chirp-rx.service"
    policies = [(n, v) for n, k, v in _settings(unit) if k == "Restart"]
    assert policies, f"{unit.name}: no Restart= at all, so a 24 h exit is final."
    n, value = policies[-1]
    assert value == "always", (
        f"{unit.name}:{n}: Restart={value}. The recorder exits 0 on purpose "
        f"every 24 hours, so anything but `always` stops the station for good "
        f"once a day and reports success while doing it.")


@pytest.mark.parametrize("unit", _units(), ids=lambda p: p.name)
def test_a_directory_another_unit_watches_is_emptied_and_not_removed(unit: Path):
    """`drf ringbuffer` watches a path, so removing it disarms the pruner.

    `chirp-ringbuffer.service` is only `After=chirp-rx.service` -- no
    `BindsTo=` -- so it does not follow the recorder through the automatic
    restart that `Restart=always` now performs every 24 h. If the recorder's
    `ExecStopPost` removes `/dev/shm/hf25`, the pruner is left holding a watch
    on an unlinked inode while `ExecStartPre` builds a fresh directory that
    nothing prunes, and the tmpfs fills. That is the failure that cost two days
    of soundings, rearmed to fire daily.
    """
    watched = "/dev/shm/hf25"
    for n, key, value in _settings(unit):
        if key not in EXEC_KEYS or "rm " not in value:
            continue
        if watched not in value:
            continue
        assert f"{watched}/" in value, (
            f"{unit.name}:{n}: {key}= removes {watched} itself. Clear its "
            f"contents instead ({watched}/*) so the pruner's watch survives "
            f"the recorder's 24-hour restart.")
