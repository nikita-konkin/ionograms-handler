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


# --- the recorder's launcher -------------------------------------------------
#
# `chirp-rx.service` no longer execs the binary; it runs a wrapper that reads
# `center_freq` out of my_station.ini and passes `--center-freq`. That removes
# the split between the compiled LO and the configured one, which blinded the
# station twice on 2026-08-19 with empty products and no error in any log
# (BACKLOG sec. 3, patches 0014/0015). The wrapper has two properties that are
# not decoration, and both fail silently or expensively if they regress.

LAUNCHER = Path(__file__).resolve().parent.parent / "tools/chirp-rx-launch.sh"


def test_chirp_rx_runs_the_launcher_that_reads_the_ini():
    unit = (UNIT_DIR / "chirp-rx.service").read_text()
    execs = [l for l in unit.splitlines() if l.startswith("ExecStart=")]
    assert len(execs) == 1, execs
    assert execs[0].endswith("tools/chirp-rx-launch.sh"), (
        f"{execs[0]} -- exec'ing the recorder directly puts the LO back in two "
        f"places with nothing keeping them equal.")
    assert LAUNCHER.exists()


def test_the_launcher_execs_so_the_recorder_stays_mainpid():
    """Without `exec`, the shell is MAINPID and `KillSignal=SIGINT` stops the
    shell instead of the recorder. A USRP that never receives SIGINT keeps
    transmitting UDP to a host that is gone and is recoverable only by removing
    power -- a site visit. This is the most expensive line in the file."""
    body = LAUNCHER.read_text()
    launch = [l for l in body.splitlines() if "rx_uhd_ext_gps" in l
              and not l.lstrip().startswith("#")]
    assert launch, "the launcher no longer starts the recorder at all"
    assert any(l.startswith("exec ") for l in launch), (
        f"the recorder is started without `exec`: {launch}. systemd's MAINPID "
        f"would be the shell, and SIGINT would never reach the USRP.")


def test_the_launcher_passes_the_configured_lo():
    body = LAUNCHER.read_text()
    assert "--center-freq=" in body, (
        "the whole point of the wrapper; without it the recorder falls back to "
        "its compiled default and the ini becomes decoration again.")
    assert "center_freq" in body, "it must read the value from chirp_config"


def test_the_launcher_refuses_to_start_on_an_unreadable_ini():
    """Falling back to the recorder's built-in default is exactly the silent
    12.5-vs-20 MHz split that caused the damage. An empty read must stop the
    start; `Restart=always` retries and the journal says why."""
    body = LAUNCHER.read_text()
    assert re.search(r'if \[ -z "\$CENTER_FREQ" \]', body), (
        "no guard on an empty center_freq read")
    assert "exit 1" in body, "the guard must fail the start, not warn"
    assert "set -eu" in body, (
        "without `set -e` a failing python3 leaves CENTER_FREQ empty and the "
        "guard is the only thing between that and a mistuned radio")


# --- -np against the schedule ------------------------------------------------
#
# The two mpirun units use `rank` for opposite purposes, so a single rule about
# -np would be wrong for one of them:
#
#   calc_ionograms.py:483  `st = conf.sounder_timings[rank]`  -- rank *selects a
#       transmitter*. -np must equal the outer list length exactly.
#   detect_chirps.py       `if block_idx % size == rank`      -- rank *stripes
#       work*. Any -np is valid; more ranks is just more throughput.
#
# The over-count is invisible where you would look for it. The doomed rank
# catches its own IndexError in a `while True` and retries at 1 Hz forever, so
# mpirun stays up, systemd reports active, and rank 0 keeps producing products.
# It ran that way from 2026-08-12 to 2026-08-19.

CLONE = Path(__file__).resolve().parents[2] / "chirpsounder2"


def _np(unit_name: str) -> int:
    text = (UNIT_DIR / unit_name).read_text()
    m = re.search(r"^ExecStart=.*?\s-np\s+(\d+)\b", text, re.M)
    assert m, f"no `-np N` in {unit_name} ExecStart"
    return int(m.group(1))


def _station_timing_groups() -> int:
    ini = CLONE / "my_station.ini"
    if not ini.exists():
        pytest.skip(f"station chirpsounder2 clone not present: {CLONE}")
    import ast
    import configparser

    cp = configparser.ConfigParser()
    cp.read(ini)
    for section in cp.sections():
        if "sounder_timings" in cp[section]:
            return len(ast.literal_eval(cp[section]["sounder_timings"]))
    pytest.skip("no sounder_timings in the station ini")


def test_ionograms_np_matches_the_number_of_transmitters() -> None:
    """-np greater than len(sounder_timings) wedges a rank; smaller drops a
    transmitter. Neither shows up as a failed unit.

    **This compares against a snapshot, and the snapshot goes stale in
    silence.** `my_station.ini` is untracked on purpose -- the agent rewrites
    the station's copy in place on every console change -- so the clone beside
    this repo only matches the station until someone edits the schedule from
    the web. It did not catch the 2026-08-20 mismatch for exactly that reason:
    the schedule went to three transmitters on the station while this copy
    still said one.

    So it is the weaker of the two guards and should be read that way. The one
    that actually holds is `control._validate` on the station, which scans the
    installed unit at apply time -- see `StationConfig.launcher`.
    """
    assert _np("chirp-ionograms.service") == _station_timing_groups()


def test_detect_np_is_not_tied_to_the_schedule() -> None:
    """detect_chirps.py stripes blocks by rank and never indexes
    sounder_timings, so its -np is a throughput choice, not a constraint. This
    test exists so nobody 'fixes' it to match the ionograms unit."""
    text = (UNIT_DIR / "chirp-detect.service").read_text()
    assert "-np" in text
    if (CLONE / "detect_chirps.py").exists():
        src = (CLONE / "detect_chirps.py").read_text()
        assert "sounder_timings[rank]" not in src, (
            "detect_chirps.py now indexes sounder_timings by rank -- its -np "
            "has become a constraint too, and this unit needs the same rule "
            "as chirp-ionograms.service"
        )


# --- a script systemd runs must be runnable ----------------------------------

REPO = Path(__file__).resolve().parents[1]


def _exec_targets(unit: Path) -> list[str]:
    """The program each ``Exec*=`` line runs, prefixes stripped.

    Uses the agent's own line joiner rather than a second copy: ``ExecStart=``
    is routinely wrapped with a trailing backslash, and a scanner that misses
    that reads the program name out of the wrong line.
    """
    from services.agent.control import _logical_lines

    out = []
    for line in _logical_lines(unit.read_text(encoding="utf-8")):
        key, _, value = line.partition("=")
        if not key.strip().startswith("Exec") or not value.strip():
            continue
        word = value.strip().split()[0]
        out.append(word.lstrip("-+@:!"))     # systemd's Exec prefixes
    return out


@pytest.mark.parametrize("unit", _units(), ids=lambda p: p.name)
def test_a_script_this_repo_ships_is_executable_in_git(unit: Path) -> None:
    """``ExecStart`` on a non-executable file is 203/EXEC, and the unit never
    starts.

    Twice now this repo has shipped a launcher git recorded as 100644.
    `tools/chirp-rx-launch.sh` was caught before it ran; `tools/drop-watch.sh`
    was not, and was fixed by hand on the station on some unrecorded day --
    which is worse than the bug, because the working station and the repo then
    disagree and only the station knows why.

    The file mode on disk is not the thing to check: a fresh clone gets its
    mode from git's index, so that is what has to be right.
    """
    import subprocess

    for target in _exec_targets(unit):
        if not target.startswith("/"):
            continue
        # Only paths this repo actually ships. /bin/mkdir is not ours.
        for prefix in ("/home/ionouser/ionograms-handler/",):
            if not target.startswith(prefix):
                continue
            relative = target[len(prefix):]
            if not (REPO / relative).exists():
                continue
            recorded = subprocess.run(
                ["git", "ls-files", "-s", "--", relative],
                cwd=REPO, capture_output=True, text=True).stdout.split()
            assert recorded, f"{relative}: named by {unit.name}, not tracked"
            assert recorded[0] == "100755", (
                f"{relative} is {recorded[0]} in git, but {unit.name} runs it "
                f"as ExecStart. A fresh clone gets 203/EXEC and the unit never "
                f"starts. Fix with: git update-index --chmod=+x {relative}")
