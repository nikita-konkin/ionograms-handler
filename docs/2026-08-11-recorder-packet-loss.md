# DOB is losing samples: two faults, one fixed

**2026-08-11, station DOB.** The recorder was discarding roughly **43% of its
samples**, and separately the receive socket had been dropping **6% of every
recording for five days**. Neither was reported by anything. Both were found
by reading `logs/thor.log` for an unrelated reason.

This is the working record: what was measured, what was established, what was
guessed and turned out wrong, and what is still open. The wrong guesses are
here on purpose — several of them are the obvious first thought, and each cost
an hour.

---

## 1. Fault A — socket receive queue overflow (fixed)

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

**Fix.** `patches/0005-dombas-set-usrp-recv-buff-size.patch`. The counter
stopped moving immediately and completely.

---

## 2. Fault B — device-side overflow from host CPU contention (open)

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

### What would fix it

Core isolation, not another tuning flag. The recorder and the NIC IRQ need
cores that nothing else may use — `CPUAffinity` / `AllowedCPUs` in the systemd
units, which is one more thing `dombas.sh` cannot express. Reducing the
station's process count would also help: five digisonde receivers and three
plotters on an eight-core laptop is the underlying problem.

**Not yet attempted.** Everything above is measurement.

---

## 3. What was guessed wrong

Five hypotheses, each plausible, each acted on before the facts were in. They
are recorded so nobody spends the evening on them again.

| guess | why it looked right | what killed it |
|---|---|---|
| **MPI ranks (`-np 4`)** | drops appeared after the change; load hit 11.4 | 63% → "6%" was a **buffering artifact** — see below. `-np 2` still lost 45% |
| **Build flags (`-O0`)** | the rebuild command carried no `-O`, and `-O0` is genuinely slow | `-O2` rebuild: 52% → 42%. No real change |
| **Socket buffer** | correct diagnosis of Fault A | fixed Fault A entirely, did nothing for Fault B |
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

1. **Does core isolation fix Fault B?** The zero-drop measurement says CPU
   contention is sufficient to explain it; nothing has tested whether pinning
   is sufficient to cure it.
2. **Did the epoch rebuild move `epoch_offset_s`?** −2.2 ms / 659 km was
   stable across 75 samples beforehand. It cannot change without a recorder
   restart, and one has now happened.
3. **How much of the 4.28% sounding loss was really Fault A or B?**
   `find_timings.log` margins were measured on a stream that was missing 6%
   at the socket and up to 45% at the device. Every conclusion about ringbuffer
   size, consumer throughput and schedule capacity rests on that stream and
   should be re-derived once the recorder is clean.
4. **Was Fault B present before 2026-08-11?** No historical data —
   `dombas.sh` truncates `thor.log` on every launch. Fault A demonstrably was.

Nothing about scheduled mode, slot counts or ringbuffer sizing should be
decided until 1 and 3 are answered. Those measurements were taken on a
recording with holes in it.
