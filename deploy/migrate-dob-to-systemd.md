# Migrating DOB from `dombas.sh` to systemd

Station DOB acquires under a shell script. Everything built around it assumes
systemd, so the console's buttons do not work: on 2026-08-16 five queued
commands — `restart`, `start`, `set_config`, `stop`, `stop` — all failed with
the same sentence,

    no systemd target configured for this station (`target` is empty in the
    agent config), so there is nothing this agent may act on.

That refusal is correct. `target` is empty on purpose, and it is a safety
interlock, not an oversight: `systemctl restart chirp.target` on a script-run
station does not restart what is running, it starts a **second** recorder
against a USRP the first one owns, and two streamers means a drive to Dombås to
pull the power. The way to make the button work is to make the sentence untrue
— to put systemd actually in charge — which is what this document does.

**On code blocks in this file.** A fenced block is a single command, complete,
meant to be run exactly as written at that step. Anything you must edit before
running — a PID, a placeholder, a file's contents — is *indented plain text*
instead, never fenced. If it is not in a fence, do not paste it into a shell.

---

## What this changes, and what it does not

The same programs run, from the same virtualenv, reading the same
`my_station.ini`. Only the supervisor changes. What systemd adds is the set of
things a user-run shell script cannot do:

| gained | why the script cannot |
|---|---|
| `OOMScoreAdjust=-1000` on the recorder | needs `CAP_SYS_RESOURCE`; set by hand it lapses at the next restart |
| NIC ring 4096, `rmem_max` 500 MB | root-only, via `ExecStartPre` with `PermissionsStartOnly=true` |
| `KillSignal=SIGINT` | the script's `while true` loop has no signal policy at all |
| per-unit `CPUAffinity` | `taskset` in a launcher only reaches that script's children |
| the pruner supervised | `drf ringbuffer` dying was invisible for two days |
| restart per process, including the recorder's own 24 h exit | the script revives only the recorder |
| the recorder's output kept | `> logs/thor.log` is truncated on every restart |
| the console's start/stop/restart | the whole point |

What it costs is listed under [Regressions to accept or fix
first](#regressions-to-accept-or-fix-first). Read that section before booking
the window; one of the items is a silent false-green.

Estimated off-air time for the cutover itself: **10–15 minutes**, plus however
long the checks in the next section take — and those are done while the station
is still acquiring, so do them first, on a different day if you like.

---

## 0. Checks to do while the station is still on air

None of these stop anything. All of them are cheaper now than at minute three
of an outage.

**0.1 — systemd version.** The units are written to run on systemd 229 and are
pinned there by `tests/test_systemd_units.py` (`MIN_SYSTEMD = 229`). An older
host is out of scope; a newer one is fine.

```bash
systemctl --version | head -1
```

**0.2 — the NIC name.** `chirp-rx.service` names the interface literally, in
`ExecStartPre=/sbin/ethtool -G enp0s25 rx 4096`. If this laptop's interface is
not `enp0s25`, the unit fails at its first pre-step and the recorder never
starts. Compare the output of:

```bash
ip -o link show | awk -F': ' '{print $2}'
```

**0.3 — the ringbuffer size the station is actually running.** The unit says
`-z 14000MB`. `patches/0003-local-dombas.sh.result` says `12000MB`. Whichever
is running right now is the one that has been measured against this station's
sweeps, so read it off the live process rather than off either file:

```bash
pgrep -a -f 'drf ringbuffer'
```

If it disagrees with the unit, edit the unit — not the station. The size buys
history, not space: at 25 MS/s, 12000MB holds 119 s and 14000MB holds 139 s,
measured. Cyprus's 250 s sweep wrote nothing at the smaller size.

Also confirm `/dev/shm` is big enough to hold it, since tmpfs defaults to half
of RAM:

```bash
df -h /dev/shm
```

**0.4 — `-np` against the schedule.** `chirp-ionograms.service` starts
`calc_ionograms.py` with `-np 2`. In scheduled mode each rank takes its own
transmitter with a bare subscript, so **`-np` must equal the number of rank
groups in `sounder_timings`**. Today the console shows two — NIC1 and NIC2 —
so 2 is right. Neither way of being wrong announces itself: too few and the
transmitters past the cut are never sounded, too many and one rank dies of
`IndexError` while the others carry on and the log looks normal.

```bash
grep -n 'sounder_timings\|serendipitous' /home/ionouser/chirpsounder2/my_station.ini
```

**0.5 — where products are written.** Settled on 2026-08-18: the station
writes to `/home/ionouser/ionozond_data2`, the boot SSD, and all three places
that name it now agree — `ARCHIVE_LOCAL` in `chirp-archive-sync.service` and
`chirp-archive-prune.service`, and `output_dir` in
`deploy/station-dob.json.example`, which had said
`/media/ionouser/DATA3/ionozond_data2` since before the migration.
`test_one_output_dir_is_written_in_three_places` keeps them together.

The authority is still `output_dir` in `my_station.ini` — that is what
chirpsounder2 obeys, it lives on the station, and nothing in this repository
can check it. Read it once here and make sure the three copies match, because
a disagreement is silent in three different ways: the mirror mirrors an empty
directory and reports success, `newest_product_age_s` reports a recorder that
has stopped when it has not, and the station preview shows nothing while the
console blames the agent's version.

**0.6 — the privilege to run `systemctl`.** This is the step most likely to
make the whole migration land and still not work, so do it now and prove it.

The agent runs as `User=ionouser` and calls `systemctl` **bare** —
`services/agent/control.py:125` is `["systemctl", verb, target]`, with no
`sudo`. A non-root `systemctl start` goes to systemd over D-Bus, systemd asks
polkit, and polkit's default answer for a process with no session is no. The
console's button would then fail with `Interactive authentication required`
instead of the message quoted at the top — a different sentence, the same dead
button.

Ubuntu 16.04 ships **polkit 0.105**, which reads `.pkla` files. The JavaScript
`.rules` format — the one every current answer on the internet shows, and the
only one that can restrict the grant to a named unit — needs polkit 0.106. This
is the same class of trap as the `+` prefix in `chirp-rx.service`
(`docs/2026-08-13-systemd-229.md`): the newer syntax does not error, it is
simply never read.

```bash
pkaction --version
```

For 0.105, create `/etc/polkit-1/localauthority/50-local.d/50-chirp.pkla`
containing exactly these four lines:

    [Let ionouser manage the chirp units]
    Identity=unix-user:ionouser
    Action=org.freedesktop.systemd1.manage-units
    ResultAny=yes

`ResultAny`, not `ResultActive`: the agent runs as a system service and has no
login session, so `ResultActive` would match nothing. Be clear about what this
grants — on 0.105 it is `manage-units` for **every** unit on the host, because
per-unit filtering needs `action.lookup("unit")` and that is JavaScript-only.
The narrower alternative is a `sudoers` rule naming the exact commands, but
that also requires teaching `control.py` to invoke `sudo`, which it does not
do; if you want that, it is a code change and a separate commit. On a
single-purpose acquisition laptop whose only user already owns the radio, the
broad grant is the defensible trade.

Then prove it, non-interactively, against the least dangerous unit there is —
restarting the metadata consumer loses nothing:

```bash
sudo -u ionouser -H systemctl restart chirp-metadata.service
```

That will say `Failed to restart chirp-metadata.service: Unit
chirp-metadata.service not found.` until step 1 has installed the units — which
is fine and is *not* the failure you are looking for. The failure you are
looking for is `Interactive authentication required`. Re-run this command after
step 1 and require a clean exit before going anywhere near step 2.

Know what that probe does and does not prove. `sudo -u` runs inside *your*
login session, so polkit sees an active session and a `ResultActive=yes` would
also pass it — while the agent, a system service with no session at all, would
still be denied. The probe catches "polkit denies `ionouser` outright", which
is the common failure; `ResultAny` is what covers the rest, and the definitive
test is queueing a command from the console in section 5.

**0.7 — decide whether `chirp-timings.service` is enabled.**
`deploy/station-dob.json.example` says to drop it outside serendipitous mode,
because `par-*.h5` is what that mode consumes, and `dombas.sh` starts
`find_timings.py` only under `if [ "$SERENDIPITOUS" = "True" ]`. DOB is in
scheduled mode, so it has not run there, and two things follow from that which
argue the other way:

- the archive holds **zero** `par-*.h5`, so `/ui/sources` censuses the worst of
  the three detection products, and `epoch_offset_s` — which needs a timing
  solution against a known transmitter — cannot be computed at all;
- §16's sounding-loss measurement reads its margins out of
  `logs/find_timings.log`, and without the process there are no margins to
  read.

The unit is unconditional: enabled, it runs in either mode. Recommend enabling
it, and watch one cycle after step 3 — if it turns out to misbehave in
scheduled mode, `systemctl disable --now chirp-timings.service` costs nothing
and the rest of the migration is unaffected. Whatever you decide here, the
`units` list in step 4.1 must match it.

---

## 1. Install the units — still on air, still nothing enabled

Copying unit files starts nothing. This whole section is safe with the station
acquiring, and it is where parse errors get caught while there is still margin.

```bash
sudo cp /home/ionouser/ionograms-handler/services/agent/systemd/*.service /home/ionouser/ionograms-handler/services/agent/systemd/*.target /home/ionouser/ionograms-handler/services/agent/systemd/*.timer /etc/systemd/system/
```

```bash
sudo systemctl daemon-reload
```

**Verify every unit parses.** This is the step that would have caught the `+`
prefix bug — a unit that fails to load reports a path error, not a version
error, and the recorder simply never starts:

```bash
systemd-analyze verify /etc/systemd/system/chirp-rx.service /etc/systemd/system/chirp-ringbuffer.service /etc/systemd/system/chirp-detect.service /etc/systemd/system/chirp-timings.service /etc/systemd/system/chirp-ionograms.service /etc/systemd/system/chirp-metadata.service /etc/systemd/system/chirp-sync.service /etc/systemd/system/chirp.target
```

Silence is a pass. Note that on 229 `systemd-analyze verify` does **not**
complain about directives from a later systemd — that is what the repo's own
test is for, so also confirm the suite is green on this checkout before you
copy anything.

Now re-run the privilege probe from 0.6; it must succeed cleanly this time,
because the unit exists:

```bash
sudo -u ionouser -H systemctl restart chirp-metadata.service
```

That command starts a second `detections2metadata.py` beside the one
`dombas.sh` already runs. Harmless — it is a file-to-file converter that
touches no radio — but stop it again before continuing:

```bash
sudo systemctl stop chirp-metadata.service
```

**Enable the services, but do not start the target.** `chirp.target` only
`Wants=chirp-rx.service` directly; every other unit attaches through
`WantedBy=chirp.target` in its `[Install]` section, which is a symlink that
`enable` creates. An un-enabled unit is one the target will not pull, silently.

```bash
sudo systemctl enable chirp-rx.service chirp-ringbuffer.service chirp-detect.service chirp-timings.service chirp-ionograms.service chirp-metadata.service chirp-sync.service
```

```bash
sudo systemctl enable chirp.target
```

Enabling `chirp.target` makes it `WantedBy=multi-user.target`, i.e. acquisition
comes up at boot. That is the intent, and it is also the moment `dombas.sh`
must stop being started by hand or by any `rc.local`/cron/`@reboot` entry —
check for one now, because a reboot that starts both is the two-streamers
failure with nobody at the keyboard:

```bash
crontab -l; ls -la /etc/rc.local 2>/dev/null; grep -rn dombas /etc/cron.d /etc/rc.local 2>/dev/null
```

**Do not enable any `chirp-digisonde@` instance.** The template ships so that a
station which wants them can express them, not because this one should run
them. An earlier revision of this runbook said to enable four, copied out of
`deploy/station-dob.json.example`, and that was wrong in the most expensive
way available: the digisonde receivers are the cause of the 45% sample loss,
and patch 0007 exists to remove them.

They are not downloaders. `receive_digisonde.py` demodulates the sounders **off
air from this station's own ringbuffer** at 25 MS/s, so each instance is a
`detect_chirps`-sized consumer. Measured on 2026-08-12: five of them cost ~969
dropped events/s and ~65,000 `RcvbufErrors`/s at load 10.40 on eight cores;
stopping them gave zero dropped samples and zero dropped datagrams over a full
hour. The pipeline that cannot be avoided already needs 4.4 of the 8 cores, and
the receivers want ~3.4 more.

Nothing on the station will tell you if they come back. `chirp-drop-watch`
counts `D` markers in what `logs/thor.log` grew by, and under systemd that file
stops being written — it reports zero drops for ever (see Regressions). The
console shows a healthy station losing half its samples.

If they are ever wanted back, the first thing to try is `CPUAffinity` pinning
for the recorder and the NIC IRQ — untested, and the budget above says 3.4
cores would have to come from somewhere.

The archive timers are independent of the target and can be started now, on
air, without waiting — they only read products and write to the NAS. Do this
only after 0.5 has settled which directory they should read:

```bash
sudo systemctl enable --now chirp-archive-sync.timer chirp-archive-prune.timer
```

**Do not start `chirp.target`.** Not until `dombas.sh` is dead and verified
dead. That is the next section, and it is the only irreversible part of this
procedure.

---

## 2. Stop `dombas.sh` — the station goes off air here

This is the by-hand sequence from `BACKLOG.md` §23, in the only order that is
safe. Read all four steps before starting the first.

**2.1 — list the supervisors, and read the list.** Two processes match `bash
./dombas.sh`: the outer script and the `while true` subshell it forked. Both
must die before the recorder, or the loop revives it five seconds later.

```bash
pgrep -a -f dombas.sh
```

**Never `pkill -f dombas.sh`.** On both 2026-08-13 and 2026-08-15 that pattern
also matched PID 14016, `git diff examples/marieluise/dombas.sh` — a pager left
open since Aug 9 — and it sorted *first*. `pgrep -a` prints full argv precisely
so you can see which lines are shells and which are somebody's pager. Take the
PIDs of the two whose argv is `bash ./dombas.sh` and no others.

**2.2 — kill those two, by explicit PID.** Substitute the two numbers you just
read; a plain `kill` (TERM) is right for shells:

    kill <outer-PID> <subshell-PID>

**2.3 — confirm both are gone before touching the radio.** If either survives,
go back to 2.2. Do not proceed on a partial result:

```bash
pgrep -a -f dombas.sh
```

**2.4 — stop the recorder, with SIGINT and nothing else.** UHD sends the
stop-streaming command from its `SIGINT` handler and from no other signal. TERM
or KILL leaves the USRP transmitting UDP to a host that is gone: it stops
answering ARP and discovery, and nothing on the host recovers it — only
removing power, in Dombås.

```bash
pkill -INT -f rx_uhd_ext_gps
```

**2.5 — wait past the revive window and confirm.** The loop's `sleep 5` is why
this is a separate step. Ten seconds, then:

```bash
pgrep -a -f rx_uhd_ext_gps
```

Empty output is the pass. Anything here means a supervisor is still alive —
return to 2.1.

**2.6 — kill the orphans, by PID.** Nothing else in the tree touches the radio,
so these are safe to kill in any order, but they are **not** safe to leave: the
target is about to start its own copy of each, and two `drf ringbuffer`
processes pruning one directory, or two `mpirun` groups consuming one stream,
is a worse state than either. List them, then kill by PID:

```bash
pgrep -a -f 'drf ringbuffer|detect_chirps|calc_ionograms|find_timings|detections2metadata|sync_iono_data|iono_housekeeping|station_monitor|receive_digisonde|plot_'
```

    kill <each PID from that list>

Then confirm the list is empty by re-running the same `pgrep`.

**2.7 — clear the stale ringbuffer.** The old recorder's ~14 GB is still in
`/dev/shm` and nothing will reclaim it: the unit's `ExecStopPost=` clears
`/dev/shm/hf25/*` and that unit never ran. Left in place, the new
recorder dies at sample 0 with ENOSPC and prints
`dataset_samples_written = 0` — which is exactly what happened on 2026-08-06.
`dombas.sh` cleared it at every start for this reason.

Only run this once 2.5 and 2.6 are both empty. Nothing must be reading it:

```bash
rm -rf /dev/shm/hf25
```

```bash
df -h /dev/shm
```

---

## 3. Start acquisition under systemd

One command. The target pulls the enabled units and systemd resolves the
ordering — the recorder first because everything else `Requires` and `After`s
it, the pruner after the directory exists, the consumers after the stream.

```bash
sudo systemctl start chirp.target
```

**3.1 — every unit active.**

```bash
systemctl list-units 'chirp*' --all --no-pager
```

**3.2 — the recorder is actually streaming.** Its stdout is now in the journal
rather than in a file that the next restart truncates, which is one of the
things this migration buys. Watch for a growing sample count, and for the
absence of `dataset_samples_written = 0`:

```bash
journalctl -u chirp-rx.service -n 50 --no-pager
```

**3.3 — the epoch is this year.** The recorder copies the host clock into the
USRP epoch verbatim, so a start before NTP has stepped the clock stamps every
sample with whatever the RTC held — on 2026-08-06 that was 2021-04-02. The unit
orders itself after `time-sync.target`, which is the fix on our side, but
ordering cannot help if NTP never converges:

```bash
timedatectl status
```

**3.4 — the ringbuffer is being pruned.** Watch it rise and then hold. If it
climbs past 90% and keeps going, the pruner is not doing its job and that is
the fault that cost two days of soundings:

```bash
df -h /dev/shm
```

**3.5 — products appear.** New files under `output_dir` within a sounding
cycle. If nothing appears within two cycles, stop and read
`journalctl -u chirp-ionograms.service`.

**3.6 — restart it once, and watch it come back.** Not a formality. Two of the
three faults found on the real cutover only appear on the *second* start, and
the second start is the console button:

```bash
sudo systemctl restart chirp.target
```

### If the recorder restart-loops

Both of these were found on 2026-08-16, on the first real cutover, and both are
fixed in the repo's units — so if you are seeing them, the copy in
`/etc/systemd/system` predates the fix. Re-run the `cp` in step 1 and
`daemon-reload`.

The symptom of the first is a journal that repeats, every ten seconds:

    chirp-rx.service: Control process exited, code=exited status=80

`Control process` means `ExecStartPre`, and **80 is `ethtool`'s exit code for
"no ring parameters changed, aborting"**. The first start sets the ring
256 → 4096 and succeeds; every start after it asks for a size the NIC already
has, so `ethtool` refuses, the pre-step fails, and the recorder never runs. The
unit works exactly once. `systemctl status chirp-rx.service -l` names the
failing line if you want it confirmed before touching anything.

To get back on air immediately, without editing anything, put the ring back to
its boot default so the next `ethtool -G` has something to change:

```bash
sudo ethtool -G enp0s25 rx 256
```

That is a one-start reprieve, not a fix — the unit is right again on the next
restart. The fix is the `case $? in 0|80)` wrapper the unit now carries.

The second fault has no exit code at all: the recorder starts, then dies at
`mkdir -p /dev/shm/hf25/ch0`. `PermissionsStartOnly` is per-unit, so
`ExecStartPre=/bin/mkdir -p /dev/shm/hf25` runs as **root** while the recorder
runs as `ionouser`, and a root-owned ringbuffer directory is one the recorder
cannot write into. It stayed hidden for as long as `dombas.sh` had already
created that directory as `ionouser`; step 2.7 removes it, so the first start
after the cutover is the one that finds out. The unit now chowns it. Ahead of
that fix, creating it yourself as `ionouser` is enough, because `mkdir -p` on
an existing directory leaves its ownership alone:

```bash
mkdir -p /dev/shm/hf25
```

---

## 4. Hand control to the agent

Until this step the console still says the same sentence, because nothing has
told the agent that the station changed.

Edit `/home/ionouser/agent.json`. Three changes; the third is the one that is
easy to miss.

**4.1 — `units`.** Replace the empty list with every unit whose death should be
visible. A process missing from this list cannot be reported unhealthy.
`deploy/station-dob.json.example` already carries the list under
`_units_when_migrated`:

    "units": [
      "chirp-rx.service",
      "chirp-ringbuffer.service",
      "chirp-detect.service",
      "chirp-timings.service",
      "chirp-ionograms.service",
      "chirp-metadata.service",
      "chirp-sync.service"
    ],

Include only units you actually enabled in step 1 — and no `chirp-digisonde@`
instance, for the reason given there. `systemctl is-active` on a unit that does
not exist answers `inactive`, `health.py` reads that as a definite failure, and
the station reports UNHEALTHY while acquiring perfectly well. Eleven false reds
is how a status column stops being read.

The same sentence is why this list and the enabled set have to move together:
stopping a unit that is still named here turns it red, so a digisonde being
disabled after the fact means editing both, in either order, before the next
push.

**4.2 — `target`.** This is the line that makes the console's buttons work, and
it must not be set before step 3 has succeeded:

    "target": "chirp.target",

**4.3 — `launcher`.** Repoint it at the unit that now starts
`calc_ionograms.py`:

    "launcher": "/etc/systemd/system/chirp-ionograms.service",

Not cosmetic. `set_config` refuses a schedule whose rank-group count disagrees
with the `-np` the launcher passes, and `control.py:_launcher_ranks` finds that
number by text-scanning the launcher for a line mentioning `calc_ionograms.py`.
The unit's `ExecStart` is a single line of exactly that shape, so the guard
keeps working — pointed at it. Left pointing at `dombas.sh`, the guard would
validate tomorrow's schedule against a script that no longer starts anything.

**Do not change `station` while you are in here.** It is tempting, because the
migration is when everything else about the station's identity gets touched,
and it is the one field that does not mean what it looks like. `station` in
`agent.json` labels *health and commands*; the receiver name on a **sounding**
comes from the product file's own attributes, written by chirpsounder2 out of
`my_station.ini`, with the filename as a fallback (`muf/io_chirp.py:484`).
Changing one and not the other splits the station in two on the console: a new
name with live metrics and no data, an old name with every sounding ever
ingested and no agent. Future soundings keep arriving under the old name,
because nothing about `agent.json` reaches the ingest path.

If the receiver really is misnamed, that is a separate change with its own
consequences — `muf/stations.py` supplies the coordinates each name resolves
to, so the name decides `rx_lat`, `rx_lon` and therefore `path_km` and every
MUF derived from it. Make it deliberately, on its own, not as a side effect of
this one.

Then restart the agent:

```bash
sudo systemctl restart chirp-agent.service
```

```bash
systemctl status chirp-agent.service --no-pager
```

---

## 5. Verify from the console

The point of the whole exercise. All of this is done from `/ui` on the work
server, not on the laptop.

| check | expected |
|---|---|
| the station's health panel | unit rows **green**, where they were grey before |
| queue `restart` | `acked`, `ok` — and the units restart in the journal |
| queue `stop`, then `start` | both `acked`; the radio stops and comes back |
| `set_config` with an unchanged schedule | reaches `apply_config` instead of "not attempted: the stop failed" |
| the result column on a failure | the agent's reason, not the parameters you sent |

If a command comes back failed, the reason is now printed in the console's
result column — that was the fix in `49f0f76`. `Interactive authentication
required` there means step 0.6 did not take.

Then the one that only time can answer, and the one this migration got wrong
the first time: **24 hours and 10 seconds after the recorder started, it must
be running again.** `rx_uhd_ext_gps` ends its own run at 24 h with exit 0 and
`Channel 0 finished 24h streaming.`, so this is the check that the restart
policy is `always` and not `on-failure` — with `on-failure` the station simply
stops here, reporting `inactive (dead)` and `status=0/SUCCESS`. Its earlier
output is still readable across that restart, too; under `dombas.sh` it was
not.

```bash
journalctl -u chirp-rx.service --since '2 hours ago' --no-pager | tail -40
```

---

## Regressions to accept or fix first

Three things that worked under `dombas.sh` and do not survive the cutover
unchanged. The first is the dangerous kind — it fails to green rather than to
red.

**`chirp-drop-watch` stops counting, and reports zero.** `tools/drop-watch.sh`
measures packet loss by counting `D` characters in whatever `thor.log` grew by
since the last sample (`DROP_WATCH_THOR`, line 27). Under systemd the
recorder's stdout goes to the journal and `logs/thor.log` stops being written,
so `[ -f "$THOR" ]` is false, `drops` stays 0, and the drop counter reads as a
perfectly clean stream forever.

There is no unit-file fix on 229: `StandardOutput=file:` needs systemd 236 and
`append:` needs 240. Wrapping `ExecStart` in a shell to redirect is worse than
the problem — `KillMode=mixed` would send the `SIGINT` to the wrapper shell
rather than to the recorder, which is the signal that protects the USRP.

The real fix is to teach `drop-watch.sh` to read
`journalctl -u chirp-rx.service --since <last sample>` instead of a file, using
a cursor rather than a byte offset. Until that lands, **do not read the
drop-watch numbers as evidence** — mask the unit so its silence is honest:

```bash
sudo systemctl disable --now chirp-drop-watch.timer
```

**`logs/find_timings.log` moves to the journal too.** §16's 4.28% sounding-loss
measurement greps that file for `-?[0-9.]+ s left`. Same substitution:
`journalctl -u chirp-timings.service`. Mind the minus sign either way — a
pattern that drops it turns every failure into a comfortable pass.

**Three programs have no unit at all.** They stop at step 2.6 and do not come
back:

- `iono_housekeeping.py` — overlaps `chirp-archive-prune.service`, and which of
  the two should own pruning is an open question in the backlog. Decide before
  the cutover, or the local disk fills.
- `station_monitor.py` — uploads to `http://4.235.86.214/upload.php`, an
  arrangement nobody in this repo has decided to keep.
- the bare `receive_digisonde.py` with no `--sounder` — the template unit only
  covers named instances.

The three plotters (`plot_rtf.py`, `plot_detectionfiles.py`,
`plot_ionograms.py`) also stop, and that is deliberate: patch 0007 dropped
them.

---

## Rollback

If acquisition does not come up and the cause is not obvious within the window
you booked, go back. Order matters here as much as it did going forward.

**First blank `target` in `/home/ionouser/agent.json`** and restart the agent —
before anything else. While the units are stopped and `dombas.sh` is running,
a queued `start` from the console would start a second recorder against the
USRP `dombas.sh` owns.

    "target": "",
    "units": [],

```bash
sudo systemctl restart chirp-agent.service
```

Then stop and disable the target so a reboot does not undo the rollback:

```bash
sudo systemctl stop chirp.target
```

```bash
sudo systemctl disable chirp.target chirp-rx.service chirp-ringbuffer.service chirp-detect.service chirp-timings.service chirp-ionograms.service chirp-metadata.service chirp-sync.service
```

Confirm the recorder is really gone — the same 2.5 check, for the same
five-second reason — and that `/dev/shm/hf25` was emptied by `ExecStopPost`
(the directory stays; it is the ~14 GB inside it that must be back):

```bash
pgrep -a -f rx_uhd_ext_gps; df -h /dev/shm
```

Only then relaunch the script the way it has always been launched, from
`/home/ionouser/chirpsounder2` with `.venv38` active, its stdout and stderr
going to `/home/ionouser/dombas-launch.log`.

---

## Follow-ups this migration creates

- **`-np` now lives in two places.** Patch 0009 made `dombas.sh` derive the
  rank count from `sounder_timings`; `chirp-ionograms.service` hardcodes `-np
  2`. Changing the schedule now means editing the unit and `daemon-reload`
  in the same change. A `chirp-ionograms` wrapper that derives it would restore
  what 0009 gave, and would let `_launcher_ranks` return `None` — "no answer,
  do not check" — which is the good case that flag was written for.
- **`drop-watch.sh` against the journal**, per the regression above.
- **Decide `iono_housekeeping.py` vs `chirp-archive-prune.service`**, and
  `sync_iono_data.py` vs `chirp-archive-sync.service`. Two of those four
  overlap and the migration is the moment it stops being theoretical.
- **`BACKLOG.md` §23 points at §17** for this migration; §17 is the schedule
  results and tracks nothing of the sort. This file is the tracker now.
