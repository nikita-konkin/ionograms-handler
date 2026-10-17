# Phase 4: retiring the ionozond Qt console

Design, 2026-10-17, for phase 4 of the live console. It follows phase 3
([`2026-10-16-ionozond-parameters-and-schedule.md`](2026-10-16-ionozond-parameters-and-schedule.md),
issue #4), whose section 11 left out four things: retiring the Qt console,
`/etc/default/ionozond`, the prune and NAS timers, and seeding transmitters
from `/ui/sources`. This design covers all four, plus two of phase 3's
follow-ups. Section 11 records the decisions, settled on 2026-10-17.

**Phase 4 starts only after two phase 3 checks pass on the station.** Both
are still open:

- a real schedule change from `/ui/live`, made while signed in, which proves
  `try-restart` and the polkit rule;
- the QSettings round trip and the lock files with the Qt console running.

The laptop has also not yet pulled `ae01f00` (PR #7, the `pending` fix). A
`git pull` and a `chirp-agent` restart come first (section 7, step 0).

## 1. What changes for the operator

Today the Yoshkar-Ola laptop has two ways to change the sounder: the Qt
console, reached over AnyDesk, and `/ui/live`, since phase 3. After phase 4,
`/ui/live` is the only one anyone needs:

- the schedule, as in phase 3;
- the parameters an operator actually changes, now including the `.h5`
  archive switch and how long the NAS keeps local `.h5` files;
- the site settings in `/etc/default/ionozond`, all shown and a few editable;
- the prune and NAS timers, shown with their last run, each with Run now;
- new transmitters seeded from the ones verified on `/ui/sources`.

The console is no longer launched, but it stays installed for a rollback. A
setting the web does not offer is changed by hand in the ini file on the
laptop, and the agent regenerates `chirp_config.py` from it (section 3).

## 2. What the console does today, and where each part goes

From `src/frmmain.cpp`, `src/parametersdialog.cpp` and `tools/sounder.sh` in
the ionozond repository:

| The console's part | After phase 4 |
|---|---|
| Schedule dialog | `/ui/live` schedule panel (phase 3) |
| Parameters dialog: `fft_count_index`, `keep_lfs`, `keep_lfs_hours`, `save_raw_voltage` | `/ui/live` (phase 3) |
| Parameters dialog: `h5_archive`, `nas_keep_h5_days` | `/ui/live`, new in phase 4 |
| Parameters dialog: `nas_dest` | shown on `/ui/live`; changed on the laptop (section 4) |
| Parameters dialog: `sample_rate_index`, `dec`, `whiten*`, `obj_*`, `iono_*`, `tb`, `lfsr_polynome_degree`, `direct_signal_cutting`, `data_dir` | edited by hand in `config.ini`, then regenerated (section 3). Regenerate refuses a `data_dir` that names a different folder from `IONOZOND_ARCHIVE` |
| Parameters dialog: colour maps, `ig_*`, the `[Variations]` group, `live_ionogram` | retired with the console; the web has its own display |
| `sound_app`, `config_file` | retired. The agent still checks that `config_file` names the file the unit reads |
| Regenerating `chirp_config.py` on every dialog OK and on START | the agent, on every web change and on Regenerate |
| START and STOP of its own child sounder | the unit, through Start and Stop on `/ui/live` (phase 2). The console's START already refuses while the unit runs |
| Live ionogram, SNR and PDP panels | `/ui/live` (phases 1 and 2) |
| Archive of recent captures | `/ui/soundings` and the standalone viewer (`viewer.pro`) on the laptop |
| Disk pie chart | the agent's `disk_free` metric |
| CPU widget | dropped. Nobody has acted on it, and an overloaded sounder shows up as `sounder_overflows` |
| Log pane | the unit's journal, the event stream on `/ui/live`, and the `logs` command |
| The warning to restart the service after a schedule change | the `pending` banner (phase 3, as fixed in #7) |

## 3. The files after retirement

| File | Written by | Read by |
|---|---|---|
| `~/.config/dsChirp/schedule.ini` | the agent | the agent, to generate the next file |
| `~/.config/dsChirp/config.ini` | the agent, and the operator by hand for settings the web does not offer | the same |
| `$IONOZOND_CONFIG`, `~/chirp_config.py` | the agent only | `rx_dechirp.py`, once, at start; `nas_sync.py`, every hour |
| `/etc/default/ionozond` | root | the units (`EnvironmentFile=`) |
| `~/.config/ionozond/web.env` (new) | the agent | the prune and NAS units, after `/etc/default/ionozond` |

**The ini files stay the store, and the agent becomes their only writer**
(decision 1). Phase 3's section 11 said the ini files "may stop being the
source of truth". They stay, for three reasons:

- **The rollback costs little.** The console reads the same two files, and
  the generator is byte-equal to the console's. Its first dialog OK
  regenerates a `chirp_config.py` identical to the agent's when the ini files
  already hold every key the dialog writes; otherwise the OK adds the missing
  keys first (section 8).
- **Phase 3 carries over unchanged.** The line editor, the generator, the
  golden tests, `base`, `sounder_hash` and drift all keep working.
- **A new store would buy little.** A station document that the agent
  owns, say `~/.config/ionozond/station.json`, would drop the QSettings
  dialect. But a rollback would then have to write the ini files back from
  it, so the line editor would still be needed.

Generating `chirp_config.py` alone, with no ini files and no document, is not
an option. That file lists only the transmitters that are switched on, so it
has nowhere to keep one that is switched off. It would need a change in
`rx_dechirp.py` first.

**What drift means afterwards.** With no console, drift means someone edited
an ini file by hand and has not regenerated yet. The phase 3 refusal text,
"open its parameters dialog on the laptop and press OK", no longer applies.
It becomes "regenerate", and there are two ways to regenerate (decision 2):

- `python3 -m services.agent ionozond regenerate` on the laptop. It prints
  what the sounder would read differently, asks for confirmation, writes
  `chirp_config.py` and asks for `try-restart`.
- a **Regenerate** button on `/ui/live` with the `station` capability. It
  queues `regenerate_ionozond` carrying the `sounder_hash` and `base` it was
  shown. The agent refuses when either has changed since, as `set_ionozond`
  does. The panel shows the diff before Apply, from the drifted report, which
  then carries the regenerated literals alongside the current ones.

Both go through the same `apply` path as a schedule change: lock, back up,
write, `try-restart`, and roll back if the restart is refused. A regenerated
file the sounder could not read is refused before it is written.

**Regenerate refuses a folder that disagrees.** Section 5 keeps
`IONOZOND_ARCHIVE` read-only because the sounder, the live feed, the prune
unit and the NAS unit must all name the same folder. A hand edit of
`data_dir` could break that rule from the other side, and Regenerate is the
only way such an edit reaches `chirp_config.py`, since `set_ionozond` refuses
a drifted station. So Regenerate refuses when `data_dir` in `config.ini`, or
the agent's own `live_dir`, names a different folder from
`IONOZOND_ARCHIVE`, with the paths compared after normalising. The refusal
names all three values.

## 4. The NAS and prune settings

**Where each setting is read today**, highest precedence first:

| Setting | Read from | Today on the laptop |
|---|---|---|
| NAS destination | `--dest`, else `nas_dest` in `chirp_config.py`, else `NAS_DEST` | `/mnt/ionozond_5tb/ionozond_data`, from the console; `NAS_DEST` is empty |
| `.h5` kept locally | `--keep-h5-days`, else `nas_keep_h5_days`, else `NAS_KEEP_H5_DAYS` | 21, from the console; `NAS_KEEP_H5_DAYS=30` |
| NAS bandwidth, SSH key | `NAS_BWLIMIT`, `NAS_SSH_KEY`, passed as flags by the unit | `/etc/default/ionozond` |
| Prune floor and ceiling | `PRUNE_KEEP_DAYS`, `PRUNE_FREE_GB`, passed as flags by the unit | `/etc/default/ionozond` (defaults 7 and 50) |

`nas_sync.py` uploads `.h5` and `.lfp`. It deletes a local `.h5` only past
the retention window, and only after `rsync --checksum` shows the NAS copy
matches. With no destination anywhere, it exits 0 and says so.

**`nas_dest` and `nas_keep_h5_days` keep their home in `config.ini`**
(decision 3). They reach `nas_sync.py` through `chirp_config.py` exactly as
now, so retiring the console moves nothing and the 21 days stand.

- `nas_keep_h5_days` becomes editable on `/ui/live`: −1 for never delete,
  otherwise 1 to 3650 days, as the dialog allowed.
- `nas_dest` is shown, not editable. Phase 3's reason holds: a web field
  that can send the station's archive anywhere is a way to take the data, not
  a setting. It is changed on the laptop by hand, then regenerated.
- **The fallbacks are brought into line.** `/etc/default/ionozond` says 30
  days and no destination. Today they are inert, but a `config.ini` that ever
  loses the keys, for example after a rollback to an old backup, would
  silently switch retention to 30 days and stop the NAS copy. The migration
  sets `NAS_DEST=/mnt/ionozond_5tb/ionozond_data` and `NAS_KEEP_H5_DAYS=21` as
  root, so both sources agree (section 7, step 3).

**The timers.** `ionozond-nas.timer` is enabled and runs hourly. Whether
`ionozond-prune.timer` is enabled is not known; check it in step 0.

- **Health reports both**, by listing `ionozond-nas.service` and
  `ionozond-prune.service` in the agent's `job_units`. `health.job_states`
  already reports a failed run in red and a timer that has stopped firing as
  a definite failure. The example config left `job_units` empty only because
  both timers ship disabled.
- **The server already sees the NAS side.** It ingests the NAS folder as
  archive 5 (`ionozond_5tb/ionozond_data`). The archive's newest sounding is
  therefore an end-to-end check that the copy runs, and costs nothing.
- **Enabling a timer stays a step on the laptop** (decision 5). Enabling
  writes unit symlinks, which takes polkit's `manage-unit-files` action, a
  far wider grant than the one unit and four verbs the rule allows today.
  Stopping a timer from the web would pause it only until the next boot,
  which is worse than not offering it.
- **Run now** on `/ui/live` starts `ionozond-nas.service` or
  `ionozond-prune.service` once. The polkit rule gains those two units, for
  the verb `start` only.
- **Run now always goes out as `systemctl --no-block start`**, whatever the
  station's `no_block` says; that setting defaults to false
  (`services/agent/config.py`). For a oneshot, a blocking `start` waits for
  the whole run, and `ionozond-nas.service` allows a run an hour
  (`TimeoutStartSec=3600`). `control.systemctl` gives up after 120 s, so a
  blocking start would be acknowledged as a failure while the copy went on,
  and would hold up every command queued behind it. The run's outcome shows
  in the job metric afterwards.
- **The prune timer is a backstop.** With `keep_lfs` off, the sounder itself
  deletes each capture once its `.h5` is written, and the pruner finds
  nothing to do. With `keep_lfs` on, the pruner is what stops the disk from
  filling. It is enabled after the README's dry run, whatever `keep_lfs`
  says, since a full disk costs a night of soundings (section 7, step 6).

## 5. `/etc/default/ionozond` from the web

Every key is **shown** on `/ui/live`, read from the file the agent already
reads for `IONOZOND_CONFIG`. `NAS_SSH_KEY` is a path, not a key, so nothing
secret leaves the laptop.

| Key | Editable from the web | Why |
|---|---|---|
| `RADIO_ADDR`, `IFACE` | no | They change with the hardware. A wrong address leaves `ExecStartPre` pinging until the start times out |
| `JUMBO` | no | The only proof of a frame size is a capture that completes (`17-probe-frame-size.py`). 8000 killed every sounding on this link. The web could revert a bad value, but nothing would be gained over the probe |
| `IONOZOND_CONFIG`, `IONOZOND_ARCHIVE` | no | `live_dir`, the prune unit and the NAS unit must all name the same folder |
| `NAS_DEST`, `NAS_SSH_KEY` | no | where the data goes, and the credential that takes it there |
| `PRUNE_KEEP_DAYS`, `PRUNE_FREE_GB` | yes | storage policy, checked: the floor 1 to 365 days, the ceiling 0 to the disk size |
| `NAS_KEEP_H5_DAYS` | no | a fallback; the setting is `nas_keep_h5_days` (section 4) |
| `NAS_BWLIMIT` | yes | 0 or more KB/s |

**The file belongs to root, and stays so** (decision 4). The editable keys
go into an override file that the agent owns, `~/.config/ionozond/web.env`.
A drop-in, installed once by root during the migration, loads it after the
main file in the prune and NAS units:

```ini
# /etc/systemd/system/ionozond-prune.service.d/web.conf (and the same for nas)
# web.env is writable by samsung. Load it only in a unit that runs as samsung
# with no "+" or "!" step: never in ionozond-sounder.service.
[Service]
EnvironmentFile=-/home/samsung/.config/ionozond/web.env
```

- systemd reads environment files in order, and a later one wins, so a key
  in `web.env` overrides the same key in `/etc/default/ionozond`.
- The leading `-` means a missing file is not an error. Deleting `web.env`
  is therefore a complete rollback of every web edit.
- The agent writes only the allow-listed keys, with `_atomic_write_bytes`,
  and refuses any other key. That limits what a web edit can change. It does
  not protect the file: any process running as samsung can edit `web.env`,
  as it can edit `config.ini`. The file grants nothing beyond samsung's own
  rights, because every unit that loads it runs as samsung.
- The prune and NAS units are oneshots started by their timers. A change
  applies at the next run, with no restart and no `pending`.

**`web.env` is loaded only by units that run as samsung and have no
root-run step.** An environment file can set `PATH` or `LD_PRELOAD`. Loaded
by a unit with a step that runs as root, a file samsung can write would let
samsung make root run a program of its choosing. So the rule is:

- a unit that loads `web.env` runs as `User=samsung`, and has no step
  prefixed with `+` or `!` and no `PermissionsStartOnly=`;
- `ionozond-sounder.service` never loads it. Its line 40,
  `ExecStartPre=+/bin/sh -c 'ethtool -G "$IFACE" rx 4096 2>/dev/null || true'`,
  runs as root. Making `JUMBO` editable from the web would therefore need a
  different mechanism, not this file;
- `ionozond-prune.service` and `ionozond-nas.service` qualify today: both run
  as samsung, with no root-run step. The drop-in's own comment states the
  rule, so whoever changes either unit later meets it, and the migration
  checks it before installing (section 7, step 4).

A root helper behind a one-line sudoers rule could edit
`/etc/default/ionozond` itself and keep one file. It is not used: the agent
would then run a root command whose input comes from the web. The override
file keeps root out of the path entirely.

## 6. Seeding transmitters from `/ui/sources`

`/ui/sources` records a verified transmitter per receiving station, as a code
and a list of `{chirp-rate, rep, chirpt}` slots with the census row that
justified them (`transmitter` table, `control_routes.save_transmitter`).
DOB's scheduled mode already builds its schedule from these.

**The schedule panel gains "Add from verified"** (decision 7). It lists the
verified transmitters of every station, each labelled with the station that
verified it. Choosing one adds a group to the edit in progress, which then goes through
the ordinary check and Apply:

| ionozond field | Seeded from |
|---|---|
| name | the code. Both sides use the same `CODE_RE` |
| `rate` | `chirp-rate` / 1000 (Hz/s to kHz/s) |
| `rep` | `rep` |
| `lat`, `lon` | `muf/stations.py`'s registry for the code. If it cannot resolve the code, the panel says so, as `save_transmitter` does |
| `cf` | the band's low edge, which the operator enters, plus half the full sample rate. The census has no frequency to seed it from: an emitter carries a rate, a cycle and slot seconds, nothing more |
| `chirpt` | the verified `chirpt`, moved to the moment the sweep reaches the band's low edge: `chirpt + low_edge / rate`, modulo `rep` |
| `dur` | the full sample rate over `rate`, the whole receive band (25 MHz at 500 kHz/s is 50 s), capped so that `dur` and the product time fit the unit's `TimeoutStopSec` |
| `active` | false. The operator switches it on after the checker has shown any collision |

**The `chirpt` conversion must be checked before it ships.** It assumes a
verified `chirpt` is the moment the sweep is at 0 Hz, chirpsounder2's
convention. ionozond's `chirpt` is the moment the sweep is at `cf` minus half
the full sample rate; this is the difference that once put Magadan 8 s late.
`cyprus1` is heard by both receivers at the Yoshkar-Ola site, so a seeded
`cyprus1` can be compared with the hand-made group already in the schedule.
The census seconds are also as received, not as transmitted. At these
distances that is milliseconds, inside a slot either way.

**One slot per group.** ionozond's schedule has one `rep` and one `chirpt`
per group, and the group's name becomes the archive's `txname`. A transmitter
verified with two slots is offered one slot at a time. A second group for the
same transmitter would need another name, and a name that `muf/stations.py`
does not resolve leaves its products without coordinates. Recording several
slots under one name is not in phase 4.

## 7. Migration on a running station

None of these steps restarts the sounder: `chirp_config.py` does not change.

0. **Preconditions.**
   - The laptop runs `ae01f00` or later: `git pull`, then restart
     `chirp-agent`.
   - The two open phase 3 checks pass.
   - The health loop reports `sounder_config` under `yoshkar-ola`, with no
     drift and no problems.
   - `systemctl is-enabled ionozond-prune.timer` is recorded.
   - Whether the chirpsounder2 receiver posting as `Yoshkar-Ola` runs on this
     laptop is recorded. It is still writing (section 10); which machine it
     runs on is not known.
1. **Back up** `~/.config/dsChirp`, `chirp_config.py` and
   `/etc/default/ionozond` into one dated tarball in the home directory.
2. **Deploy** the phase 4 server (watchtower, on merge) and agent
   (`git pull`, then restart `chirp-agent` and `chirp-live`).
3. **Align the NAS fallbacks**, as root: `NAS_DEST` and
   `NAS_KEEP_H5_DAYS=21` in `/etc/default/ionozond` (section 4).
4. **Install the drop-ins** for the prune and NAS units, then
   `daemon-reload` (decision 4). First check each unit as installed:
   `systemctl cat ionozond-prune.service ionozond-nas.service` must show
   `User=samsung`, no `ExecStart*=` line starting with `+` or `!`, and no
   `PermissionsStartOnly=`. If either fails, stop; the drop-in must not go
   in.
5. **Extend the polkit rule** with `start` on `ionozond-nas.service` and
   `ionozond-prune.service`. Prove it with `pkcheck` and one Run now.
6. **Enable the prune timer** after the README's dry run
   (`python3 tools/prune_lfs.py $IONOZOND_ARCHIVE`, which deletes nothing),
   then list both job units in `agent.json`: `ionozond-nas.service` and
   `ionozond-prune.service`.
7. **Retire the console.** Remove `~/.local/share/applications/ionozond.desktop`
   and the copy on the desktop, if any, so it is no longer started from
   habit. Leave the binary and `~/.cache/ionozond-build` in place.
8. **Verify:** a web schedule change, a Regenerate after a deliberate hand
   edit of an unused key, an edit of `PRUNE_FREE_GB`, and a green NAS run.

## 8. Rollback

- **To the console:** run `19-install-desktop.sh` again, which restores the
  launcher. The console reads the ini files the agent kept. Its first dialog
  OK writes every key the dialog holds into the ini files, so it also adds
  the keys they lack. On 2026-10-17 the phase 3 round trip showed it: the OK
  that set `keep_lfs_hours` added `live_ionogram=true` and
  `save_raw_voltage=false`, which `config.ini` had not held. So the OK
  rewrites `chirp_config.py` byte for byte as the agent left it only when the
  ini files already hold every key the dialog writes. Otherwise check the
  next health report: drift false and an unchanged `sounder_hash` mean
  nothing restarts, and a new hash opens an epoch as any station change
  does.
- **Web edits of the site settings:** delete `~/.config/ionozond/web.env`.
  The units fall back to `/etc/default/ionozond` at their next run.
- **The drop-ins:** delete them and run `daemon-reload`.
- **The agent:** the previous release answers `regenerate_ionozond` with
  "unknown command" and ignores `web.env`. The server keeps queueing only what
  the panel offers.

The lock-file check left open in phase 3 matters again here, because a
rollback runs the console beside the agent.

## 9. Server and agent changes

- **Agent, `services/agent/ionozond.py`:**
  - `regenerate`, shared by the CLI and the new `regenerate_ionozond`
    command. It refuses when `data_dir` or `live_dir` names a different
    folder from `IONOZOND_ARCHIVE` (section 3);
  - `h5_archive` and `nas_keep_h5_days` join `GENERAL`;
  - the drift text names Regenerate, not the console's dialog;
  - `sounder_config` gains `site`, the keys of `/etc/default/ionozond` and
    `web.env` with where each came from, and `nas_dest` from `config.ini`;
  - a drifted report also carries `regenerated`, the literals a regeneration
    would give, so the panel can show Regenerate's diff before Apply.
- **Agent, new `site` handling:** reading and writing `web.env` for
  `set_ionozond_site`, a new command with the allow list of section 5. Only
  the prune and NAS units load the file (section 5).
- **Agent, `runner._dispatch`:** `regenerate_ionozond`,
  `set_ionozond_site`, and `start` on the two job units for Run now. The
  agent's `ALLOWED_VERBS` already has `start`; the unit allow list is new.
  Run now always passes `no_block=True`, whatever `config.no_block` says
  (section 4).
- **Server:** `QUEUEABLE` gains the two commands, each vetted against the
  last report as `set_ionozond` is. Run now is queued as `start` with a
  `unit` parameter limited to the two job units.
- **`/ui/live`:** the new parameters, the site table, the timers with Run
  now, Regenerate, and Add from verified. Strings in English and Russian.
- **The ionozond repository**, a companion issue there: the unit comments,
  `18-install-service.sh` and `deploy.md` name the console's dialog as the
  place for the NAS settings. They should name `/ui/live` and `config.ini`.

## 10. Phase 3's follow-ups

| Follow-up | In phase 4? |
|---|---|
| An unacknowledged `set_ionozond` opens no epoch | **Yes.** Once the console is gone, every change comes through a command, and this gap becomes the main way left to lose one. The fix is small: `observe` compares a report's `sounder_hash` with the open epoch's `config_hash`, where the epoch has one, instead of with the previous report. The first report after the wait for the acknowledgement then opens the epoch the missing acknowledgement never did |
| `sounder_config` adds about 2.5 MB a day | **Yes.** Phase 4 grows the block with `site`. Send it whole only when `base`, `sounder_hash`, the site's hash, `pending` or `drift` changes, and once an hour regardless. Other reports carry just those five fields. The server keeps the last whole block per station, so the panel and the vetting no longer depend on the latest report |
| Disabled schedule inputs look editable to anonymous viewers | **No.** It is one CSS rule in `base.html`, beside `button:disabled`, for every page's disabled inputs and selects. It ships now on its own, and phase 4's new inputs inherit it |
| The `Yoshkar-Ola` / `yoshkar-ola` split | **No**, but seeding must not depend on it. `Yoshkar-Ola` is the chirpsounder2 receiver's health name (`deploy/station-dob.json.example`), and `yoshkar-ola` is the ionozond receiver's. They are two receivers at one site, and both still record: the server's NAS folder `ionozond_5tb/ionozond_data2` has DOB day folders up to 2026-10-16, holding files such as `cdetections-Yoshkar-Ola-1792110600.h5`, and its archive scan loaded 87 new files at 20:40 UTC that day. Seeding reads every station's verified transmitters, so it works with the split in place. Merging or renaming is a data migration of its own: products carry the receiver name from the receiver's own config, and the DOB example warns that changing one name without the other splits a station in two. Which machine runs the chirpsounder2 receiver is still an open question |

## 11. Decisions

**Settled 2026-10-17: every recommendation of the draft was accepted.**

1. **The store.** `schedule.ini` and `config.ini` stay the store, and the
   agent becomes their only writer. An agent-owned station document is not
   introduced. Generating `chirp_config.py` alone stays out until
   `rx_dechirp.py` can hold a switched-off transmitter.
2. **Regenerating after a hand edit.** Both the CLI and a Regenerate button
   on `/ui/live`.
3. **`nas_dest` and `nas_keep_h5_days`.** Both stay in `config.ini`. The
   retention is editable on `/ui/live` and stays at 21 days. The destination
   is shown on `/ui/live` and changed only on the laptop. The
   `/etc/default/ionozond` fallbacks are set to the same values.
4. **`/etc/default/ionozond` from the web.** Every key is shown.
   `PRUNE_KEEP_DAYS`, `PRUNE_FREE_GB` and `NAS_BWLIMIT` are edited through the
   agent-owned `web.env`, which drop-ins load in the prune and NAS units
   only: never in a unit with a root-run step, so never in the sounder unit.
   The radio address, interface and frame size stay read-only, and no
   sudoers helper is added.
5. **The timers.** Enabling stays a step on the laptop. The web shows both
   timers through `job_units` and offers Run now, always without blocking.
   `ionozond-prune.timer` is enabled after its dry run, as a backstop.
6. **Parameters beyond phase 3's four.** `h5_archive` and `nas_keep_h5_days`
   are added. The receive chain and the processing internals are hand edits
   plus Regenerate. The display settings retire with the console.
7. **Seeding.** Add from verified, across every station, seeded switched off,
   one slot per group. It ships only after the `chirpt` conversion is checked
   on `cyprus1`.
8. **The station-name split** stays out of phase 4. Both receivers still
   record (section 10), so they remain two stations.
9. **Phase 3's follow-ups**, as section 10 sets out:
   - the epoch for an unacknowledged change is fixed in phase 4;
   - the report is sent whole only on change, in phase 4;
   - the disabled inputs get a separate small fix, outside phase 4;
   - the name split stays outside phase 4.

**Open questions**, none of which blocks the design:

- Which machine runs the chirpsounder2 receiver posting as `Yoshkar-Ola`?
- Is `ionozond-prune.timer` enabled on the laptop today?
- Is a verified `chirpt` the moment the sweep is at 0 Hz? Checked on
  `cyprus1` before seeding ships (section 6).

## 12. Tests first

- `regenerate` writes exactly what `build` produces from the ini files, and
  refuses when `base` or `sounder_hash` has moved since the report it was
  queued against.
- `regenerate` refuses a `config.ini` whose `data_dir`, or an agent whose
  `live_dir`, names a different folder from `IONOZOND_ARCHIVE`, and accepts
  the same folder written differently (a trailing slash, a `~`).
- A hand edit to `config.ini` reports drift. After `regenerate`, drift is
  false and `sounder_hash` is the new one.
- `set_ionozond_site` writes only allow-listed keys to `web.env`, refuses
  every other key, and leaves the old file on a failed write.
- `site` in the report names, for each key, whether it came from
  `/etc/default/ionozond` or from `web.env`.
- `nas_keep_h5_days` accepts −1 and 1 to 3650, and nothing else.
- Run now goes out as `systemctl --no-block start` on one of the two job
  units, even when the station's `no_block` is false, and is refused for any
  other unit.
- `observe` opens an epoch when an unacknowledged change's hash differs from
  the open epoch's `config_hash`, and opens none for a hash the epoch already
  holds.
- A report without the whole block keeps the server's last whole block, and
  the panel and the vetting read that one.
- Seeding converts a verified slot into the group the hand-made `cyprus1`
  already has, once the convention is confirmed.

## 13. Not in phase 4

- Replacing the ini files with an agent-owned document (decision 1).
- Editing `RADIO_ADDR`, `IFACE`, `JUMBO`, the archive paths, `NAS_DEST` or
  `NAS_SSH_KEY` from the web.
- Enabling or disabling timers from the web.
- Several slots for one transmitter.
- Merging or renaming the `Yoshkar-Ola` and `yoshkar-ola` stations.
- Uninstalling the console's binary. That waits until a rollback is no longer
  wanted.
