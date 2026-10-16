# Phase 3: the ionozond sounder's parameters and schedule from the web console

Design, 2026-10-16, for issue #4. Issue #1 named this phase; phases 1 and 2
are in `architecture.md` §5.5. Section 9 records the decisions.

## 1. What changes for the operator

Today the operator reaches the Yoshkar-Ola laptop over AnyDesk, edits the Qt
console's schedule or parameters dialog, and presses OK. The console
regenerates `chirp_config.py`, and the sounder takes it up the next time it
starts. After phase 3 the operator does the same on `/ui/live`. The page shows
the schedule the sounder is running. An edit is checked while the operator
watches, travels through the command queue, is checked again on the laptop,
and is written there. A restart then lets the sounding in progress finish.

## 2. The files on the station

| File | Written by | Read by |
|---|---|---|
| `~/.config/dsChirp/schedule.ini` | the console's schedule dialog | the console, to generate the next file |
| `~/.config/dsChirp/config.ini` | the console's parameters dialog | the same |
| `$IONOZOND_CONFIG`, `~/chirp_config.py` | the console (`configwriter.cpp`), on every dialog OK and on START | `rx_dechirp.py`, once, at start |
| `/etc/default/ionozond` | `18-install-service.sh`, as root | the unit (`EnvironmentFile=`) and `sounder.sh` |

Three things follow from this table.

- **The sounder reads its config once.** A change takes effect only on
  restart. The console's own `ApplyConfigChange` records the evening this cost:
  every file agreed, and only the running process disagreed.
- **`chirp_config.py` is derived.** If the agent edited only that file, the
  next console dialog closed on the laptop would regenerate it from the two
  ini files and silently undo the web change. So the agent edits the ini files
  and regenerates `chirp_config.py` exactly as the console does. The ini files
  stay the source of truth until phase 4 retires the console.
- **`/etc/default/ionozond` belongs to root and stays out of reach.** It holds
  the radio address, the interface, the frame size and the prune and NAS
  fallbacks. The agent reads it (mode 0644) to learn `IONOZOND_CONFIG`, the
  file the unit actually hands the sounder, and never writes it. This is the
  same lesson as `StationConfig.launcher`: check the file that is in use, not
  the one you believe is. The agent also refuses a change when the console's
  `config_file` setting names a different file, because then the console and
  the service would each be configuring a sounder the other cannot see.

### What the laptop does not run yet

Sections 7 and 8 rely on health reports that carry `sounder_config`. The
Yoshkar-Ola laptop sends no health reports today. It runs only
`chirp-live.service`, which posts live batches as `yoshkar-ola`, and that is
the station Start and Stop are queued for. The only health under a
Yoshkar-Ola name comes from a chirpsounder2 agent posting as `Yoshkar-Ola`,
with capital letters. As things stand, the server would refuse every edit,
because no report under `yoshkar-ola` carries a `sounder_config`. There are
two ways to close the gap.

- **Run the health loop on the laptop** (recommended). This means
  `chirp-agent.service` with `ExecStart=/usr/bin/python3 -m services.agent run`,
  `User=samsung` and the same `agent.json` as the live feed, so the same
  `station`. `deploy/README.md` already describes this unit for an ionozond
  station.
  - For: health is where §5.4 puts station state, and every report is kept.
    So `pending`, the epochs for console edits and the server's check all read
    what already exists (`db.latest_health`).
  - For: it is a separate process from the feed, so a capture the feed cannot
    parse does not silence it (§5.5).
  - For: it still pulls commands once a minute when the feed is down.
  - For: it finally runs the ionozond health metrics that shipped in #2 but
    have never run on the station.
  - Against: one more unit to install and watch on the laptop.
- **Carry `sounder_config` in the live feed's batches.**
  - For: no new unit.
  - Against: the server keeps live data in memory only, by design, so it
    would need a new place to keep the last config for the check and for the
    epochs.
  - Against: the feed is the process built to be expendable.
  - Against: the health metrics from #2 still would not run.

The `Yoshkar-Ola` station the chirpsounder2 agent reports as stays a separate
question, and phase 3 does not depend on it. If that agent runs on this same
laptop, its unit and the new one need different unit names or one shared
config; check this before installing.

## 3. What to expose

**The schedule**, one `[group]` per transmitter in `schedule.ini`:

| Field | Unit | Why it is here |
|---|---|---|
| `active` | bool | switching a transmitter on or off, the console's own way to drop one |
| `cf` | kHz | the band's centre; the sweep starts at `cf − sample_rate/2` |
| `rate` | kHz/s | the sweep rate; it also fixes the range resolution |
| `dur` | s | the sweep length; `dur × rate` must fit the receive band |
| `rep`, `chirpt` | s | the slot: the sweep starts where `(unix time) % rep == chirpt` |
| `lat`, `lon` | degrees | written into each capture's header; the geometry gate is built from them |
| `fft_count` | points | optional per-transmitter FFT length, one of `FFT_COUNT_LIST` |

The operator may add a transmitter. Its name must match ionograms-handler's
`CODE_RE`, `^[A-Za-z0-9_.]{1,24}$` (`control_routes.py:270`): one to 24 Latin
letters, digits, underscores or dots. That is stricter than every one of the
console's rules (Latin letters, no quote, slash or control character, at most
64 characters), so it is the rule that applies. It forbids a dash because the
name becomes the dash-delimited `txname` in the `.h5` archive. Existing groups
keep their names, since the rule applies only to names the web adds; the
receiver's group, `yoshkar-ola`, would fail it. Deleting and renaming are not
offered: deactivating covers deletion, and a rename is a new transmitter whose
old archive keeps the old name.

**The parameters**, from `[General]` in `config.ini`:

| Key | What it does |
|---|---|
| `fft_count_index` | the FFT length for transmitters without their own `fft_count` |
| `keep_lfs`, `keep_lfs_hours` | how long the 80 MB captures are kept |
| `save_raw_voltage` | keeps the full voltage in the `.h5`, about 69 MB a capture |

**Deliberately left out:**

- **The receiver**, the group with `rx=true`. It is the site, not a setting.
- **`sample_rate`, `sample_rate_index` and `dec`.** They define the receive
  chain, and every check below depends on them. DOB's sample rate is left out
  for the same reason.
- **`data_dir`.** The unit overrides it with `--outdir $IONOZOND_ARCHIVE`, and
  `live_dir`, the prune unit and the NAS unit must all name the same folder.
- **`nas_dest`.** A web field that can send the station's archive to any
  address typed into it is a way to take the data, not a setting.
- **Processing internals**: `whiten*`, `obj_*`, `iono_*`, `tb`,
  `lfsr_polynome_degree` and `direct_signal_cutting`. Nobody has changed them
  since the port; add one when someone needs it.
- **The console's display settings**: `colormap*`, `ig_*`, `snr_*`, `pdp_*`
  and `live_ionogram`.
- **Everything in `/etc/default/ionozond`.**

## 4. How a change travels

```
 /ui/live  Check ──► POST /stations/{id}/ionozond/check    errors, warnings, next captures; queues nothing
           Apply ──► POST /stations/{id}/commands           set_ionozond; needs `station`
                                   │
 agent     live reply {"queued": 1} ──► GET /stations/{id}/commands       within 2 s, or within the minute
           check · lock · write · systemctl --no-block try-restart
           POST .../commands/{cmd}/ack  {journal, config_hash} ──► config_epoch
 health    sounder_config: the schedule in force, `pending` until the sounder restarts
           (needs the health loop of section 2)
```

The station still only makes outbound connections. The command arrives by the
same pull as Start and Stop (§5.5).

**A new command, `set_ionozond`, not `set_config`.** The vocabulary differs
(`cf`/`rate`/`dur` per named transmitter, against chirpsounder2's
`sounder_timings` entries), and so do the file, the checker and the restart.
An agent too old to know the new command answers "unknown command" honestly.
It cannot misread the command as a chirpsounder2 change. `set_config` sent to
an ionozond station is refused by name; today it fails on DOB's default ini
path.

**The command carries the whole document and the hash it was edited from.**

```json
{"name": "set_ionozond",
 "params": {"base": "<hash of the document as the page loaded it>",
            "schedule": {"cyprus1": {"active": true, "cf": 20000, "rate": 100, "dur": 250,
                                     "rep": 300, "chirpt": 10, "lat": 35, "lon": 34}},
            "general": {"fft_count_index": 5, "keep_lfs": false, "keep_lfs_hours": 0,
                        "save_raw_voltage": false}}}
```

The whole document, because collisions and the band can only be judged over
the whole schedule. The base hash, because a command can wait in the queue
for minutes while someone edits the schedule in the console. The agent
refuses a change made against a schedule that is no longer there; it does not
overwrite the newer one. The hash covers only the fields above, so a colour
map changed in the console does not invalidate a queued edit. Every existing
transmitter must appear in the document; a missing one is refused, not
deleted.

## 5. How the agent applies it

A new module, `services/agent/ionozond.py`: stdlib only and Python 3.8
compatible. It runs inside `runner.obey`, which means in the live feed's
process or the health loop's, whichever pulls the command first. So it must
return in well under a second, and it never waits on the sounder.

1. **Read.** Read `IONOZOND_CONFIG` from `/etc/default/ionozond`, then the two
   ini files and `chirp_config.py`.
2. **Compare.** The document parsed from `chirp_config.py` must hash to
   `base`. Also, the ini files regenerated *without* the change must give that
   same document. If they do not, someone edited the ini files without the
   console regenerating, and the agent refuses rather than ship that edit
   along with this one.
3. **Validate** the new document with a port of ionozond's
   `src/schedulecheck.cpp`, keeping its two levels.
   - **Its errors refuse the whole change**: a malformed name; a number Python
     cannot read (a decimal comma, a leading zero); a latitude or longitude
     out of range; `rep` not above zero; an FFT longer than the whole capture;
     and a delay window wider than the band after decimation can hold. That
     last one is the case the phase 2 hand-over flagged, where the live feed's
     geometry gate falls past the unambiguous range and its captures are
     skipped silently.
   - **Its warnings come back to the page**, and the operator confirms them
     before Apply: a sweep wider than the band or starting below zero;
     `chirpt` outside `0…rep`; `dur` plus the sounder's 3 s lead longer than
     `rep`; sessions lost to a collision between transmitters (from a port of
     `simulateSounder`); and a day that `rep` does not divide. The console
     only warns on these because an operator at the laptop may press START
     anyway. The web keeps that choice and puts the confirmation in front of
     the person making it.
   - **One check is new, and refuses**: `dur` plus product time must fit the
     unit's `TimeoutStopSec` (300 s, read with `systemctl show`). Otherwise a
     stop during that sounding ends in SIGKILL and a truncated capture. At
     100 kHz/s the band caps `dur` at 250 s, but at 50 kHz/s it allows 500.
4. **Generate** `chirp_config.py` with a port of `buildChirpConfig`. Then parse
   the result with `rx_dechirp.load_config`'s rules: literal assignments only,
   and `sounders` present and not empty. An empty schedule makes the sounder
   exit at start, and systemd would retry it every minute. A file the sounder
   could not read is refused before it is written.
5. **Write.** Hold the lock files QSettings itself takes (`<ini>.lock`, a
   QLockFile), so a console saving at the same moment cannot interleave.
   Back up all three files. Replace `schedule.ini`, then `config.ini`, then
   `chirp_config.py` last. The sounder reads only the last file, so it never
   sees half a change. If any replace fails, the backups are restored.
   - **The settings files are edited as bytes, not through configparser.** Qt
     writes them with `QSettings::IniFormat` (`frmmain.cpp:85-86`). That
     dialect differs from configparser's in escaping, list values, `@Variant`
     values, the `%General` group and key handling, so a configparser round
     trip could change what the console reads back. Instead, a small line
     editor in `services/agent/ionozond.py` changes only the lines of the
     keys being edited. It adds a new key at the end of its group and a new
     group at the end of the file, and leaves every other byte as the console
     wrote it. It writes only values that need no QSettings escaping, which
     covers everything the web sends: numbers, `true` and `false`, and names
     that match `CODE_RE`. The generator does need every `[General]` value,
     so the editor also reads QSettings' quoted strings, such as
     `data_dir="…"`. It refuses, rather than guesses, when a value is a list
     or an `@`-typed value, since the console itself would write such a value
     into `chirp_config.py` as an empty, unreadable assignment.
   - **One atomic write serves both kinds of file.** `control._atomic_write`
     takes a `configparser.ConfigParser` and calls `parser.write`
     (`control.py:783`), so it fits neither. Add a sibling,
     `control._atomic_write_bytes(path, data)`, which owns the sequence: a
     temporary file in the same directory, fsync, then `os.replace`. Then make
     `_atomic_write` render the parser to text and delegate to it, so the
     sequence exists once and DOB's path behaves as before. The ini files go
     through it with their edited bytes, and `chirp_config.py` goes through it
     as UTF-8 text, the encoding `rx_dechirp.load_config` reads.
6. **Restart**: `systemctl --no-block try-restart ionozond-sounder.service`.
7. **Acknowledge** with one result for the write and one for the restart. The
   write's result carries `journal`: the changes as `{key: {from, to}}`, with
   keys such as `cyprus1.chirpt` and `keep_lfs`, plus the new `config_hash` and
   the warnings. The existing `_journal_epoch` then opens the epoch.

**`try-restart`, not `restart`.** `restart` starts a stopped unit. An operator
who stopped the sounder, for example to use the console's START, which needs
the radio, would have it started behind their back by a schedule edit. And a
restart queued behind a Stop still in progress replaces the stop job, undoing
the Stop. `try-restart` restarts only an active unit, and does nothing to one
that is stopped or stopping. The agent reads `ActiveState` beforehand only to
say which of the two happened. For a stopped sounder, the acknowledgement says
the change takes effect at the next Start.

**The agent does not wait; systemd does.** On SIGTERM the sounder finishes
the capture in progress, writes its products and exits. systemd then starts it
on the new file, within `TimeoutStopSec=300`, exactly as for Stop in phase 2.
The cost: the new process opens the radio and disciplines its clock before its
first capture, so a session due within that time is lost. That is at most one
session per change, and the page says so.

**A refused restart rolls the write back.** polkit refuses before systemd
queues anything, so the agent knows at once. The old process is still
running the old file, and the file must describe what runs. So the agent
restores the backups and reports the write as failed, with no journal, and no
epoch opens.

This refines §5.4's rule for this station. For chirpsounder2 the epoch opens
on a successful write even when the start fails, because the stop came first.
Here the write comes before the restart, so it stands only once systemd has
accepted the restart, or once the sounder is stopped and will read the file at
its next start. A capture in progress at the acknowledgement began before
`valid_from`, so it stays in the old epoch, the configuration it was recorded
under.

## 6. What polkit and the agent must allow

Two allowlists are involved, and they are not the same thing.

**polkit, on the laptop.** Today the rule allows `start`, `stop` and `restart`
on `ionozond-sounder.service`, for `samsung`. Phase 3 needs **one more verb,
`try-restart`**, and nothing else:

- The three files are in `samsung`'s home, so writing them needs no privilege.
- `systemctl show` and `is-active` only read properties, and polkit does not
  check them.
- `/etc/default/ionozond` is read, never written.
- There is no `reload`, since the sounder has none, no `enable` or
  `daemon-reload`, and no other unit. The prune and NAS timers stay out.

The deploy README's rule gains `verb == "try-restart"`. Prove it with
`pkcheck ... --detail verb try-restart` (exit 0), with the `ssh.service` probe
still refused. Still to verify: that systemd on the laptop reports this verb
as `try-restart`, which its source does by naming the verb after the job type.
The definitive test is a schedule change acknowledged with the restart
accepted.

**The agent's own allowlist**, `control.ALLOWED_VERBS` (`control.py:74`),
is today `start`, `stop`, `restart`, `status` and `is-active`. It gains
`try-restart`. The property reads, `TimeoutStopSec`, `ActiveState` and
`ActiveEnterTimestamp`, need `systemctl show`, which the list lacks. There are
two ways to issue those reads:

- Through `health._show` (recommended). Health already reads unit properties
  this way, and it already copes with old systemd lacking `--value`. It calls
  `systemctl` directly and never passes the allowlist, so the list needs no
  `show`.
- Through `control.systemctl`. Then `ALLOWED_VERBS` gains `show` as well.

The alternative that keeps the rule unchanged: read `ActiveState`, then issue
`restart` only on an active unit with no job pending. That leaves a race. If
someone runs `systemctl stop` on the laptop between the two calls, the agent
overrides it. I recommend the extra verb.

## 7. Showing the schedule in force

**Health carries it**, once the laptop runs a health loop under the live
feed's station name (section 2, "What the laptop does not run yet"). Each of
its reports then gains `sounder_config`:

```json
"sounder_config": {"hash": "…", "written_at": 1760630000, "loaded_at": 1760620000,
                   "pending": false, "schedule": {…}, "general": {…},
                   "receiver": {"name": "yoshkar-ola", "lat": 56.38, "lon": 47.53},
                   "sample_rate_hz": 25000000, "dec": 625, "timeout_stop_s": 300}
```

- It is read from `chirp_config.py`, the file the sounder reads. It is not
  read from the ini files or from the journal. DOB's panel shows the schedule
  from the open epoch (`acquisition.current`), and that view cannot see a
  change made in the Qt console, which is still the ordinary way to edit
  ionozond.
- `pending` is true while `chirp_config.py` is newer than the unit's
  `ActiveEnterTimestamp`: the change is written but not yet taken up. This is
  the console's "every file agreed, the process did not" trap, made visible.
- The block is under a kilobyte. Every report is kept verbatim, so at one a
  minute that is about 1.3 MB a day. If that is too much, send the block only
  when the hash changes.

**A change made on the station still opens an epoch.** When the reported hash
changes and no acknowledged `set_ionozond` produced it, the server opens a
`config_epoch` with `changed_by` set to `station` and the diff as `changes`.
Without that, a schedule edited in the console leaves later soundings
attributed to a configuration that no longer exists, which is the failure
`config_epoch` exists to prevent.

**`/ui/live` gains a Schedule panel**, for anyone who can read:

- the transmitter table, with each transmitter's band, from `cf − sr/2` to
  that plus `dur × rate`;
- the next six captures, from the same `simulateSounder` port;
- who changed the schedule and when, from the open epoch;
- while `pending`, the banner "written; takes effect after the sounding in
  progress".

With the `station` capability the panel becomes editable. Each edit is
checked with `/ionozond/check`, Apply queues the command, and Apply asks once
more when the check warns of lost sessions. All strings come in English and
Russian.

## 8. Server side

- `control_routes`: `QUEUEABLE` gains `set_ionozond`. `_vet_ionozond` imports
  `services.agent.ionozond`, as `_vet_config` already imports
  `services.agent.control`, so the server and the station run one checker. The
  check runs against the station's last `sounder_config` (sample rate, `dec`,
  and the transmitters the edit leaves alone), so collisions and the band are
  judged while the operator is looking. It refuses a station whose last report
  has no `sounder_config`: a station that is not ionozond, an agent too old
  for this, or a laptop with no health loop. The refusal names which one, so
  the last case does not read as a bad edit.
- `POST /stations/{id}/ionozond/check` does the same vetting and returns the
  errors, the warnings and the next captures, queuing nothing.

## 9. Decisions

**Decided 2026-10-16: every recommendation below was accepted** ([issue #4,
decisions comment](https://github.com/nikita-konkin/ionograms-handler/issues/4#issuecomment-6223534228)).
The laptop runs a health loop under the live feed's station. A web edit
changes the two ini files and regenerates `chirp_config.py`. The generator and
the checker live in the agent. The restart is `try-restart`, and polkit gains
that one verb. The parameter list stands, with `nas_dest` and `data_dir` left
out. The web offers no deletion or renaming. A schedule changed on the station
opens a `config_epoch` too.

1. **How `sounder_config` reaches the server.** Either a health loop runs on
   the laptop (`chirp-agent.service`, `services.agent run`) under the live
   feed's `station` (recommended), or the live feed carries the block in its
   batches. Section 2 weighs the two.
2. **Edit the ini files and regenerate** (recommended), or edit only
   `chirp_config.py` and declare the console's dialogs read-only on this
   station from now on.
3. **Where the generator and checker live.** In the agent (recommended): the
   server and the station then share one checker, and the C++ originals go
   with the console in phase 4. The alternative is the ionozond repository,
   which would keep one owner for the file format, tested beside the C++.
4. **`try-restart` plus one polkit verb** (recommended), or read
   `ActiveState` and then `restart`, leaving the rule unchanged but racy.
5. **The parameter list in section 3**, in particular leaving `nas_dest` and
   `data_dir` out.
6. **No deletion or renaming** of transmitters from the web in phase 3.
7. **Epochs for changes made on the station** (recommended). This amends
   §5.4: an epoch can then open without a command.

## 10. Tests first

- The Python generator is byte-equal to `tests/golden/chirp_config.py` from
  ionozond's ini fixtures, the file produced by the original binary.
- The Python checker agrees with `test_schedulecheck.cpp` on every case, at
  the same level and for the same reason.
- A real `schedule.ini` and `config.ini`, copied from the laptop, come back
  byte-identical after a read and a write with no edit. An edit of one key
  changes only that key's line.
- The line editor refuses to write a value that would need QSettings
  escaping, and refuses to read a list or an `@`-typed value.
- `_atomic_write_bytes` leaves either the old file or the new one, never a
  partial one. `_atomic_write`'s existing tests pass unchanged once it
  delegates.
- A failure between two replaces restores all three files.
- A stale `base` is refused, and so are ini files that drifted from
  `chirp_config.py`.
- The restart goes out as `--no-block try-restart`. A refused restart
  restores the files and journals nothing.
- `pending` follows the file's mtime against `ActiveEnterTimestamp`.
- The server opens an epoch on a hash change no command explains, and not on
  one a command does.
- `obey` with a fake `systemctl` returns within one live interval.

## 11. Not in phase 3

- `/etc/default/ionozond`: radio, frame size, prune and NAS settings.
- Enabling the prune and NAS timers.
- Seeding new transmitters from the verified ones on `/ui/sources`.
- Retiring the Qt console (phase 4). At that point the ini files may stop
  being the source of truth, and the agent would write `chirp_config.py`
  alone.
