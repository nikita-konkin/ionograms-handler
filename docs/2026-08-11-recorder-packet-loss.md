# DOB is losing samples: two faults, one cause

**2026-08-11, station DOB.** The recorder was discarding roughly **43% of its
samples**, and separately the receive socket had been dropping **6% of every
recording for five days**. Neither was reported by anything. Both were found
by reading `logs/thor.log` for an unrelated reason.

They looked like two independent problems and were treated as such for most of
a night. They are not. **Both are the same host failing to keep up with its own
radio** — and specifically, failing on *latency*, not throughput. DOB has eight
cores, the pipeline needs about six, and the recorder was still losing 45% of
its stream, because it needs its 0.8 of a core **the instant a packet arrives**
and a run queue of 6–12 does not give it that.

Two changes on 2026-08-12, in this order:

1. **Stop receiving the digisondes** (patch 0007) — ~2.7 of eight cores, none
   of it touching the radio. Both counters went to zero for an hour at load
   7.5. This was believed to be the fix. It was not: at load 9.4 the next
   evening the same six processes lost 358,691 samples in 900 s.
2. **Give the recorder a core nothing else may use** (patch 0008) — same six
   processes, recorder pinned to CPU 0:

```
drops: 0                                           over 900 s    (was ~399/s)
was RcvbufErrors: 3432084131                                     (was ~12,000/s)
now RcvbufErrors: 3432084131        — unmoved 74 min later
load average: 8.04
```

Removing load was necessary and not sufficient. **A machine with no headroom
does not fail when you average it; it fails when it is busy** — which is why
the first fix looked complete on a quiet evening and came apart on a busy one.
The isolation result still needs confirming at load 9.4; see §5.

This is the working record: what was measured, what was established, what was
guessed and turned out wrong, and what is still open. The wrong guesses are
here on purpose — several of them are the obvious first thought, and each cost
an hour.

---

## 1. Fault A — socket receive queue overflow (fixed, but not by the obvious patch)

```
netstat -su
    1859745252 packet receive errors
    RcvbufErrors: 1859745252
```

1.86 billion UDP datagrams discarded because the socket queue was full. At
1472 bytes that is **2.7 TB against 41 TB received, ~6% of every recording**,
continuously since boot.

**Why nothing saw it.** Socket-queue overflows appear in `netstat -su` and
`/proc/net/snmp` and *nowhere else*. Every other counter was spotless:

| checked | result |
|---|---|
| `ethtool enp0s25` | 1000Mb/s, full duplex, link detected |
| `ethtool -S enp0s25` | 27 billion packets, **zero** CRC / align / frame / fifo errors |
| `ip -s link` | 0 errors, 0 dropped, 0 overrun |
| `rx_missed_errors` | 68,666 over 5 days ≈ **1 second** of samples |

**Cause.** The station had `net.core.rmem_max = 500000000` and an RX ring of
4096 — the two settings usually blamed — and neither does anything alone.
`rmem_max` is a *ceiling* on what a process may request; UHD has to ask, via
`recv_buff_size` in `--usrp_args`. `dombas.sh` never passed it.
`chirp-rx.service` always has, and has never been used.

**Partial fix.** `patches/0005-dombas-set-usrp-recv-buff-size.patch`. Keep it —
a socket queue UHD never asked to enlarge is a real defect, and the patch is
correct on its own terms.

**But it does not fix Fault A**, and this document said it did for a day. That
claim rested on a single 20-second window in which `RcvbufErrors` did not move.
Overnight the counter grew by **1.17 billion**. A 20-second sample of a
bursty fault is not a result — the same mistake as the buffering trap in §3,
made a second time on a different counter.

What a bigger buffer actually buys is tolerance for a *transient* stall. It
cannot help when the host is behind continuously, which is what it was. Fault A
and Fault B are one root cause reported by two different counters: the socket
queue overflows when the host is too slow to drain it, and the device
overflows when the host is too slow to ask.

**What actually stopped it** was taking 2.7 cores of unrelated work off the
machine (§2). Measured over a full hour on 2026-08-12, with the digisonde
receivers and plotters gone:

```
was RcvbufErrors: 3214702405
now RcvbufErrors: 3214702405
```

Identical. The counter had grown by ~1.17 billion overnight *with* patch 0005
applied, and by zero in an hour without the load. Keep 0005 — a socket buffer
UHD never asked to enlarge is a real defect and the fix is correct — but it is
a guardrail, not the cure.

---

## 2. Fault B — device-side overflow from host CPU contention (cause found, fix chosen)

`thor.log` fills with `D` markers and `samp_diff` gaps:

```
1,075,876 events   mean 0.33 ms   largest 0.39 s   ~1500-2000 events/s
427 s of gaps in 957 s of uptime  =  ~45%
```

**`D` is `ERROR_CODE_OVERFLOW`.** The *device* discarded samples because the
host did not call `recv()` fast enough. Those packets were never transmitted,
which is why no network counter can see them — and why the pristine NIC
statistics above are consistent with the fault rather than evidence against it.

**Cause, established by subtraction.** Kill everything except the recorder and
`drf ringbuffer`:

```bash
pkill -f receive_digisonde; pkill -f plot_; pkill -f detect_chirps
pkill -f calc_ionograms;    pkill -f find_timings
A=$(grep -c samp_diff logs/thor.log); sleep 60; B=$(grep -c samp_diff logs/thor.log)
```

```
2049398 -> 2049398  =  0 events/s
```

**Exactly zero over 60 seconds.** With the full pipeline running it is 1500/s.
This is CPU contention and nothing else.

### Why the machine loses

Eight cores, and at the time of measurement a load average of **8–11**:

| | |
|---|---|
| `detect_chirps.py` | 2 MPI ranks, ~90% each |
| `calc_ionograms.py` | 2–4 MPI ranks, 40–70% each |
| `receive_digisonde.py` | **five** instances |
| `plot_rtf`, `plot_detectionfiles`, `plot_ionograms` | three more |
| `find_timings.py`, `drf ringbuffer`, `station_monitor` | |
| Xorg 25%, compiz, nautilus, three AnyDesk instances | a full desktop session |
| apache2, cups, evolution ×4, zeitgeist, snapd, update-notifier | disabled: apache2, cups |
| `mount.ntfs` | 8.2%, the FUSE daemon for the products disk |

The recorder's working thread runs at ~79% of one core under `SCHED_OTHER`,
competing with all of it. Only an idle helper thread had real-time priority:

```
3147  3158  RR  50   0.0 rx_uhd_ext_gps   <- real-time, idle
3147  3161  TS   -  79.5 rx_uhd_ext_gps   <- does the work, ordinary priority
```

All NIC interrupts land on **CPU7** alone (single queue, 1.7 billion on that
core), so whichever process shares CPU7 competes with 68,000 packets/second of
softirq work.

### What fixes it: run less on the box (2026-08-12)

The machine has eight cores. The pipeline that cannot be avoided needs about
half of them:

| must run here | ~cores |
|---|---|
| `rx_uhd_ext_gps` | 0.8 |
| `drf ringbuffer` | 0.5 |
| `detect_chirps` ×2 ranks | 1.9 |
| `calc_ionograms` ×2 ranks | 1.2 |
| **total** | **~4.4 of 8** |

Everything else was optional, and two groups came off: five
`receive_digisonde.py` instances, and three plotters (`plot_rtf`,
`plot_detectionfiles`, `plot_ionograms`) that write PNGs the web UI re-renders
anyway.

**With both groups off:**

```
load average: 9.15 -> 6.98
0 samp_diff events in 600 s          (was ~1500/s)
```

confirmed over a full hour, both counters together:

```
drops: 0
was RcvbufErrors: 3214702405   now RcvbufErrors: 3214702405
load average: 7.53
```

**Then with only the digisonde receivers restored**, plotters still off, to
find out which group actually mattered:

```
drops: 872081 in 900 s                 (~969/s)
RcvbufErrors 3294919800 -> 3353488759  (~65,000/s)
load average: 10.40
```

It is the digisonde receivers, essentially entirely. The plotters were never
the problem — they are dropped because nothing reads their output, not because
they cost anything much.

### Why they cannot simply be moved elsewhere

This was got badly wrong for several hours and the reasoning is worth keeping.
`receive_digisonde.py` sounds like a downloader. It is not. It **receives the
transmissions off air with this station's own USRP**, decoding the
complementary phase codes a Digisonde emits — `import digital_rf as drf`, and
the `decimation`, `n_ipp`, `ipp_us`, `freq_start/stop` and `use_c_downconvert`
keys in each `[digisonde-*]` ini section are its demodulator settings. It is a
ringbuffer consumer exactly like `detect_chirps`, bound to `/dev/shm` at
25 MS/s, and relocating it would mean shipping 100 MB/s — 8.6 TB/day — across
the network.

`README.md` and `muf/io_digisonde.py` both say so in their opening lines. An
afternoon went into designing a server-side container for these processes
before either was read. **A process named `receive_*` in a radio codebase
receives on the radio.**

### What DOB decided

Not fewer receivers: **no digisonde reception at all.** (This was decided on the
products' merits, below, not as the CPU fix — at the time it was believed to be
the CPU fix as well, and the next two sections are how that turned out.)

The products do not serve what this station is for. Each `offset_us` is a
*configured* constant — 7300 for Dourbes, 3900 for Chilton, 2000 for Juliusruh
— not a measured delay, so the range zero rests on a hand-set number.
`muf/io_digisonde.py` documents this at the decision, and the test suite still
carries a warning about a Juliusruh→DOB product whose stored range falls
outside the window that path allows. For estimating radio-channel parameters a
named transmitter with an untrustworthy range zero is worth less than a
serendipitous chirp with a solved timing solution. DOB works the chirp
circuits.

### Removing the receivers was necessary and not sufficient (2026-08-12 19:47)

The hour at zero was measured on a quiet evening. With the receivers gone for
good and nothing else changed, a busier one put the drops straight back:

    drops: 358694 in 900 s        (~399/s)
    RcvbufErrors +10,886,218
    load average: 9.17 → 9.40

Six processes, eight cores, and still losing samples. The digisonde receivers
were the largest single load on the machine but they were never the mechanism —
they were the load that pushed a marginal host past its margin, and the host is
still marginal without them. `vmstat` says why, and says it is not what it
looks like:

    r  b   swpd    free    us sy id wa   si so
    6  0  7987916  3832140 53  8 38  0    0  0
    8  0  7987912  3688344 77 14  8  0    0  0
    12 0  7987912  3863152 75 14 10  0    0  0

`b` zero, `wa` zero, `si`/`so` zero: nothing is blocked on disk and nothing is
paging, despite 7.6 GB sitting in swap and RAM reported at 90%. A run queue of
6–12 on eight cores is the entire fault. **The recorder needs 0.8 of a core,
but it needs it the instant a packet arrives** — scheduled late, it does not
call `recv()` in time and the USRP discards what it cannot hand over. This is a
latency failure on a box with CPU to spare, which is why every throughput
remedy (bigger buffer, `-O2`, fewer ranks) moved the number a little and none
of them fixed it.

### Core isolation: the actual fix (2026-08-12 20:25)

Recorder pinned to CPU 0, everything else confined to CPU 1–7:

| condition | drops / 900 s | `RcvbufErrors` | load |
|---|---|---|---|
| six processes, no pinning | 358,691 | +10,886,218 | 9.40 |
| six processes, recorder on CPU 0 | **0** | **+0** | 8.04 |
| same, 74 min later | — | **still +0** | 6–8 |

Zero, and the socket counter frozen at 3432084131 across 74 minutes. **The
900 s window is not a clean comparison** — activity fell during it and the load
average with it, so on its own it proves less than it appears. The frozen
counter over 74 minutes spanning load 6 to 8 is the stronger evidence. What it
has not yet seen is load 9.4, the condition that produced 358,691 events, and
that is the reading that settles it.

CPU 7 is deliberately left in the general pool: every NIC interrupt lands there
— one queue, 1.7 billion interrupts on that core over five days — so the
softirq work is already concentrated away from the recorder's core. Fencing it
off as well would cost the pipeline a core to guard against a load it does not
carry.

`taskset` applied by hand does not survive: affinity dies with the process, and
`dombas.sh` relaunches the recorder every 24 hours, so a live pinning lapses at
an unpredictable hour of the morning. **Patch 0008** puts both lines in the
script — the shell pinned once so every child inherits it, and `taskset -c 0`
on the recorder, which works because a process may always widen its own mask.
The systemd units express the same thing properly, with `CPUAffinity=0` on
`chirp-rx.service` and `AllowedCPUs=1-7` on the rest.

**The first attempt at the 600 s measurement was invalid** and is worth
recording. The window was three seconds, not six hundred, because the setup
line was run twice and the second truncate landed just before the read. The
tell was not in `thor.log` at all — it was the load average, which sat at
9.15 → 8.84 → 9.18 across ten minutes in which ~3 cores of work were supposedly
gone. `dombas.sh` supervises the recorder in a `while true` loop, so `pkill` on
a supervised child is a five-second pause. **Always read `uptime` beside a drop
measurement**: if the load did not move, the experiment did not happen.

---

## 3. What was guessed wrong

Five hypotheses, each plausible, each acted on before the facts were in. They
are recorded so nobody spends the evening on them again.

| guess | why it looked right | what killed it |
|---|---|---|
| **MPI ranks (`-np 4`)** | drops appeared after the change; load hit 11.4 | 63% → "6%" was a **buffering artifact** — see below. `-np 2` still lost 45% |
| **Build flags (`-O0`)** | the rebuild command carried no `-O`, and `-O0` is genuinely slow | `-O2` rebuild: 52% → 42%. No real change |
| **Socket buffer** | correct diagnosis of Fault A | a real defect, but only a mitigation — `RcvbufErrors` grew 1.17 billion overnight *after* it. Did nothing for Fault B |
| **Link speed** | 100Mb/s would explain everything | `1000Mb/s Full` |
| **Wrong interface** | two NICs are up, one is USB | `ip route get 192.168.10.2` → `enp0s25`, the one already tuned |

### The measurement trap

`rx_uhd_ext_gps` writes stdout to a file, so it is **block-buffered, not
line-buffered**. Readings taken in the first few minutes of a run show an
unflushed buffer, not a healthy recorder:

```
t= 63 s   0.00 s lost     <- looked like a fix
t=156 s   9.13 s (5.9%)   <- looked like a 10x improvement
t=571 s  298    s (52%)   <- the truth
```

Three separate conclusions were drawn from that first column. **Wait at least
ten minutes before believing any drop measurement**, and prefer the rate
between two late samples over any cumulative figure.

---

## 4. Station state after this session

Changed in the chirpsounder2 clone:

| file | change | backup |
|---|---|---|
| `rx_uhd_ext_gps.cpp` | patch 0001, rebuilt with `-O2` + `setcap cap_sys_nice+ep` | `~/rx_uhd_ext_gps.cpp.old-patch0001.bak` (superseded revision) |
| `examples/marieluise/dombas.sh` | patch 0004 (`-np 2`), patch 0005 (`recv_buff_size`) | `~/dombas.sh.bak`, `~/dombas.sh.pre-recvbuf.bak` |
| `Makefile` | modified — confirm whether `-O2` was made permanent | — |

Outside the clone:

- `apache2`, `cups`, `cups-browsed` **disabled** — they have no business on an
  acquisition machine
- a live `chrt -r -p 20` on the recorder's working thread, which **does not
  survive a restart**
- products moved: `my_station.ini` `output_dir` → local disk; `agent.json`,
  `chirp-archive-sync` and `chirp-archive-prune` must all name the same path
- `chirp-archive-sync.timer` and `chirp-archive-prune.timer` enabled

### The fix as it will actually run (2026-08-12 17:21)

Everything before this was a hand-held state: the receivers and plotters were
gone because of a manual `pkill`, and a reboot would have undone it. With
`patches/0007` applied to `dombas.sh`, `required_processes` trimmed to match,
and the station relaunched from scratch:

```
drops: 0
was RcvbufErrors: 3214729559   now RcvbufErrors: 3214729559
load average: 8.23, 7.71, 7.54
```

**`find_timings` marker: 15 entries at 17:06.** Sounding loss from here is

```bash
grep -oE '\-?[0-9.]+ s left' logs/find_timings.log \
  | awk 'NR>15 {n++; if($1<=0) z++} END {printf "n=%d lost=%d (%.2f%%)\n", n, z+0, 100*z/n}'
```

against the 4.28% baseline. Give it a day — `find_timings` produced 592 entries
over the previous multi-day run, so a short window has no power. Do not use a
cumulative figure over the whole log: it still contains the damaged period, and
reads 17.40% for that reason alone.

**The restart itself is a hazard worth writing down.** `pkill -f dombas.sh`
kills the supervisor and *orphans every child* -- the ringbuffer, both MPI
jobs, `find_timings`, `station_monitor` and the three utility scripts all keep
running. Relaunching without clearing them gives you two of everything: four
`calc_ionograms` ranks and four `detect_chirps` ranks on one ringbuffer, which
is the `-np 4` configuration that cost 63% of the stream the day before, plus
two `drf ringbuffer` processes pruning the same tmpfs. It happened here, and
`RcvbufErrors` moved by 27,154 during the eight minutes it lasted -- the only
movement in that counter all afternoon, which is a neat confirmation of the
mechanism.

Order for a clean restart: kill `dombas.sh` first (otherwise its `while true`
loop revives the recorder five seconds later), then `pkill -INT -f
rx_uhd_ext_gps` -- **SIGINT only**, TERM or KILL leaves the USRP transmitting
UDP and needs a physical power cycle -- then kill the orphans explicitly, then
verify before relaunching:

```bash
echo "detect: $(pgrep -c -f detect_chirps.py)  calc: $(pgrep -c -f calc_ionograms.py) \
 drf: $(pgrep -c -f 'drf ringbuffer')  rx: $(pgrep -c -f rx_uhd_ext_gps)"
```

`detect: 3  calc: 3  drf: 1  rx: 1` is one healthy set -- three because
`pgrep -f` matches the `mpirun` wrapper alongside its two ranks.

### The one unambiguous win

The epoch. Patch 0001's current revision is running and correct:

```
* mboard 0 gps_locked: true
GPSDO gps_time: 1786491857 (sensor read took 0.813828 s; host clock is 0.0179174 s into that GPS second)
Setting USRP time to: 1786491858 at next PPS [source: GPSDO gps_time]
Epoch check inconclusive: sampled 0.0183029 s into the second ... (USRP 1786491860, GPS 1786491860)
```

"Inconclusive" is the check declining to claim certainty too near a PPS edge —
the two clocks read identically. The previous binary carried a **superseded
revision** of patch 0001, with the `time_last_pps` comparison and the automatic
re-set that the current patch removed as unsafe; its signature is the line
`USRP last pps is -2 s from GPSDO gps_time; setting again`, which no current
build can print.

`epoch_offset_s` still reported −2.2 ms (659 km) before the rebuild. Whether
that survived has not been measured — it needs several hours of fresh
`par-*.h5`.

---

## 5. Open questions

1. **Does core isolation hold at load 9+?** Answered in part on 2026-08-12:
   pinning the recorder to CPU 0 took 358,691 drops per 900 s to zero and froze
   `RcvbufErrors` for 74 minutes (§2). But the validating window ran at load
   6–8, and the fault was measured at 9.4. **Repeat the 900 s measurement
   during tomorrow's activity peak**, reading `uptime` beside it — a zero at
   load 6 is not evidence about a failure that happens at load 9.4. Note the
   history on this question: isolation was written off twice in this document as
   untested and unnecessary, on the strength of an hour at zero that had simply
   been measured on a quiet evening.
2. **Did the epoch rebuild move `epoch_offset_s`?** −2.2 ms / 659 km was
   stable across 75 samples beforehand. It cannot change without a recorder
   restart, and one has now happened. **Blocked** — see below.
3. **How much of the 4.28% sounding loss was really Fault A or B?**
   `find_timings.log` margins were measured on a stream that was missing 6%
   at the socket and up to 45% at the device. Every conclusion about ringbuffer
   size, consumer throughput and schedule capacity rests on that stream and
   should be re-derived once the recorder is clean. **Blocked** — see below.
4. **Was Fault B present before 2026-08-11?** No historical data —
   `dombas.sh` truncates `thor.log` on every launch. Fault A demonstrably was.

**2 and 3 are blocked by the move to scheduled mode** on 2026-08-12. Both are
computed from files only search mode produces — `find_timings.log` and
`par-*.h5` — and `dombas.sh` does not start `find_timings.py` when
`serendipitous` is false. Reopening them costs one day back in search mode;
`BACKLOG.md` §16 carries the detail and the commands.

That was a decision taken with the cost known, not an oversight. It does mean
this stands: nothing about slot counts or ringbuffer sizing should be decided
from the numbers now on file. They were measured on a recording with holes in
it, and the measurement that would replace them is not currently running.

Question 1 is **not** blocked — `~/drop-watch.sh` samples both counters against
the load average every 15 minutes regardless of mode, and its truncation check
survives the `thor.log` reset at each `dombas.sh` launch.
