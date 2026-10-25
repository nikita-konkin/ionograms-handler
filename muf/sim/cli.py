"""``muf sim``: the channel simulator's commands. See docs/simulator.md."""

from __future__ import annotations

import json
import sys
from pathlib import Path

TEST_SIGNALS = ("impulse", "tone", "noise", "chirp")


def _gate(text):
    if text is None:
        return None
    lo, hi = (float(v) for v in str(text).split(","))
    return lo, hi


def _carriers(text):
    if text is None:
        return None
    from . import link

    lo, hi, step = (float(v) for v in str(text).split(":"))
    return link.carrier_grid(lo, hi, step)


def _doppler(text: str):
    try:
        return float(text)
    except ValueError:
        return text


def cmd_state(args) -> int:
    from ..cli import _load_registry
    from . import observe

    stations = _load_registry(args.stations)
    state = observe.from_sounding(
        args.sounding, method=args.method, window=args.window, gate_km=_gate(args.gate),
        stations=stations, doppler=_doppler(args.doppler), knot_mhz=args.knot_mhz)
    if args.fast:
        from . import fast

        fast.measure(args.sounding, state, stations=stations)
    print(state.describe(args.fc))
    for note in state.notes:
        print(f"  note: {note}")
    if args.out:
        print(f"-> {state.save(args.out)}")
    return 0 if state.modes else 2


def cmd_fast(args) -> int:
    from ..cli import _load_registry
    from . import fast

    if args.calibrate:
        rows = fast.calibrate(
            [float(v) for v in args.spreads.split(",")],
            shapes=tuple(args.shapes.split(",")), k_db=args.k_db,
            ox_split_us=args.split, x_to_o_db=args.x_to_o,
            realisations=args.realisations, seed=args.seed)
        print(fast.format_calibration(rows))
        if args.json:
            Path(args.json).write_text(json.dumps(rows, indent=1) + "\n",
                                       encoding="utf-8")
            print(f"-> {args.json}")
        return 0
    if args.sounding is None:
        print("muf sim fast: a sounding, or --calibrate", file=sys.stderr)
        return 2
    state, estimates = fast.measure(args.sounding, method=args.method,
                                    stations=_load_registry(args.stations))
    print(fast.format_estimates(estimates))
    if args.json:
        rows = [e.as_dict() for e in estimates]
        Path(args.json).write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
        print(f"-> {args.json}")
    if args.out:
        print(f"-> {state.save(args.out)}")
    return 0 if any(e.used for e in estimates) else 2


def cmd_preset(args) -> int:
    from . import presets

    if args.list or args.name is None:
        print("F.1487 conditions: differential delay ms, frequency spread Hz")
        for name, (delay, spread) in presets.F1487.items():
            print(f"  {name:<20} {delay:4.1f} ms  {spread:5.1f} Hz")
        print("  law                  one dispersive 1F mode on cyprus1's sweep "
              "(--muf, --tau0, --b, --cn0, --spread)")
        return 0
    if args.name == "law":
        state = presets.law_state(muf_mhz=args.muf, tau0_us=args.tau0,
                                  b_us_per_mhz=args.b, cn0_dbhz=args.cn0,
                                  spread_hz=args.spread)
    else:
        state = presets.f1487(args.name)
    print(state.describe(args.fc))
    if args.out:
        print(f"-> {state.save(args.out)}")
    return 0


def cmd_show(args) -> int:
    from .state import ChannelState

    state = ChannelState.load(args.state)
    print(state.describe(args.fc))
    for name, item in sorted(state.provenance.items()):
        print(f"  {name}: {item}")
    if args.provenance:
        for m in state.modes:
            print(f"  {m.name}:")
            for key, item in sorted(m.provenance.items()):
                print(f"    {key:<15} {item}")
    return 0


def cmd_sounding(args) -> int:
    from . import sounding
    from .state import ChannelState

    state = ChannelState.load(args.state)
    extra = {"duration_s": args.duration} if args.duration else {}
    path = sounding.write_lfs(state, args.out, seed=args.seed,
                              fading_on=not args.no_fading,
                              doppler_on=not args.no_doppler,
                              magnetoionic=not args.no_ox, noise=not args.no_noise,
                              delay_spread=not args.no_delay_spread, **extra)
    print(f"{state.source or args.state}: {state.sweep.start_mhz:.2f}-"
          f"{state.sweep.stop_mhz:.2f} MHz, {len(state.modes)} mode(s), "
          f"seed {args.seed} -> {path}")
    return 0


def _test_signal(kind: str, fs: float, duration: float, seed: int | None):
    import numpy as np

    n = round(duration * fs)
    if kind == "impulse":
        x = np.zeros(n, dtype=np.complex128)
        x[0] = 1.0
    elif kind == "tone":
        x = np.ones(n, dtype=np.complex128)
    elif kind == "noise":
        rng = np.random.default_rng(seed)
        x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2.0)
    else:
        t = np.arange(n) / fs
        span = 0.8 * fs
        x = np.exp(1j * np.pi * span * (t ** 2 / duration - t))
    return x


def cmd_channel(args) -> int:
    import numpy as np

    from . import channel
    from .state import ChannelState

    state = ChannelState.load(args.state)
    if args.input:
        x = np.load(args.input)
        what = args.input
    else:
        x = _test_signal(args.test, args.fs, args.duration, args.seed)
        what = f"test {args.test}, {args.duration:g} s"
    effects = channel.Effects(multipath=not args.no_multipath,
                              dispersion=args.dispersion, fading=not args.no_fading,
                              doppler=not args.no_doppler, magnetoionic=not args.no_ox,
                              delay_spread=not args.no_delay_spread)
    out = channel.simulate(state, x, args.fs, args.fc, effects=effects, seed=args.seed,
                           snr_db=args.snr_db, snr_bandwidth_hz=args.snr_bandwidth,
                           absolute=args.absolute)
    print(f"{what} at {args.fc:.3f} MHz, fs {args.fs:g} Hz, "
          f"dispersion {args.dispersion}:")
    for p in out.paths:
        relative = p["delay_us"] - out.reference_delay_us
        print(f"  {p['path']:<12} delay {relative:9.2f} us "
              f"(abs {p['delay_us']:.1f})  GDD {p['gdd_us_per_mhz']:8.2f} us/MHz  "
              f"C/N0 {p['cn0_dbhz']:5.1f} dB-Hz  "
              f"shift {p['doppler_shift_hz']:+.2f} Hz  "
              f"spread {p['doppler_spread_hz']:.2f} Hz")
    np.save(args.out, out.samples.astype(np.complex64))
    meta = {"fs_hz": out.fs_hz, "fc_mhz": args.fc, "lead_samples": out.lead_samples,
            "reference_delay_us": out.reference_delay_us,
            "noise_variance": out.noise_variance, "signal_scale": out.signal_scale,
            "paths": out.paths, "state": str(args.state), "input": what,
            "seed": args.seed, "effects": effects.__dict__}
    meta_path = Path(args.out).with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(f"-> {args.out} ({out.samples.size} samples, input sample 0 at "
          f"{out.lead_samples}) and {meta_path}")
    return 0


def cmd_check(args) -> int:
    from . import check

    truth, report = check.check_sounding(args.sounding, realisations=args.realisations,
                                         seed=args.seed, method=args.method,
                                         workdir=args.keep, keep=args.keep is not None,
                                         next_sounding=args.next, fast=args.fast,
                                         fast_realisations=args.fast_realisations)
    print(truth.describe())
    print(check.format_report(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1) + "\n",
                                   encoding="utf-8")
        print(f"-> {args.json}")
    return 0


def cmd_forecast(args) -> int:
    from ..cli import _load_registry
    from . import forecast, observe

    stations = _load_registry(args.stations)
    options = {"method": args.method, "doppler": _doppler(args.doppler),
               "knot_mhz": args.knot_mhz}
    ion = observe.load(args.sounding, window=args.window, stations=stations)
    state = observe.from_ionogram(ion, source=Path(args.sounding).name, **options)
    predicted = forecast.persist(state, args.lead)
    print(predicted.describe())
    if args.out:
        print(f"-> {predicted.save(args.out)}")
    if args.next is None:
        return 0 if predicted.modes else 2
    score = forecast.evaluate(predicted, args.next, origin=args.sounding,
                              copies=args.copies, window=args.window, stations=stations,
                              carriers=_carriers(args.carriers),
                              bandwidth_hz=args.bandwidth, snr_min_db=args.snr_min,
                              **options)
    print(forecast.format_score(score))
    if args.json:
        Path(args.json).write_text(json.dumps(score.to_dict(), indent=1) + "\n",
                                   encoding="utf-8")
        print(f"-> {args.json}")
    return 0


def cmd_series(args) -> int:
    from ..cli import _load_registry
    from . import series

    def progress(done, total):
        print(f"\r{done}/{total} pairs", end="", file=sys.stderr, flush=True)

    out = series.run_series(args.paths, args.out, circuit=args.circuit,
                            lead_s=args.lead, tolerance_s=args.tolerance,
                            copies=args.copies, recovery_every=args.recovery_every,
                            recovery_copies=args.recovery_copies,
                            carriers=_carriers(args.carriers),
                            bandwidth_hz=args.bandwidth, snr_min_db=args.snr_min,
                            method=args.method, window=args.window,
                            knot_mhz=args.knot_mhz, doppler=_doppler(args.doppler),
                            stations=_load_registry(args.stations),
                            stations_file=args.stations and str(args.stations),
                            jobs=args.jobs, resume=args.resume, progress=progress)
    print(file=sys.stderr)
    print(f"-> {out}")
    return 0


def cmd_summary(args) -> int:
    from . import summary

    found = summary.summarise(args.run, by=[c for c in args.by.split(",") if c],
                              block=args.block, reps=args.reps)
    print(summary.format_summary(found))
    path = Path(args.run) / "summary.json"
    path.write_text(json.dumps(found, indent=1) + "\n", encoding="utf-8")
    print(f"-> {path}")
    return 0


def cmd_live(args) -> int:
    from ..cli import _load_registry
    from . import live

    observing = {"method": args.method, "window": args.window,
                 "knot_mhz": args.knot_mhz, "doppler": _doppler(args.doppler)}
    leads = (None if args.lead is None
             else [float(v) for v in args.lead.split(",") if v.strip()])
    live.replay(args.paths, circuit=args.circuit, leads=leads,
                tolerance_s=args.tolerance, pace=args.pace, span_s=args.span,
                out=args.out, params=observing, stations=_load_registry(args.stations),
                stations_id=args.stations and str(args.stations),
                emit=lambda line: print(line, flush=True))
    if args.out:
        print(f"-> {args.out}")
    return 0


def cmd_verify(args) -> int:
    from . import verify

    checks = verify.run(quick=args.quick, seed=args.seed)
    print(verify.format_checks(checks))
    if args.json:
        rows = [c.as_dict() for c in checks]
        Path(args.json).write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
        print(f"-> {args.json}")
    return 0 if all(c.passed for c in checks) else 1


def register(sub) -> None:
    """Add ``sim`` and its subcommands to muf's parser."""
    from .. import spectro

    sim = sub.add_parser("sim", help="HF channel simulator (docs/simulator.md)",
                         description="Channel states from soundings or scenarios; "
                                     "synthetic soundings; a signal through the "
                                     "channel.")
    ss = sim.add_subparsers(dest="sim_command", required=True)

    p = ss.add_parser("state", help="describe a sounding's channel as a state (JSON)")
    p.add_argument("sounding", help=".lfs, chirpsounder2 .h5 or .echo.h5")
    p.add_argument("--out", type=Path, help="write the state here")
    p.add_argument("--method", default="contour", help="extractor that finds the trace")
    p.add_argument("--doppler", default="mid-moderate",
                   help="prior for the spread a power ionogram cannot measure: an "
                        "F.1487 condition or a spread in Hz (default mid-moderate)")
    p.add_argument("--window", type=int, default=spectro.DEFAULT_WINDOW)
    p.add_argument("--gate", metavar="LO,HI", help="range gate, km")
    p.add_argument("--knot-mhz", type=float, default=0.1)
    p.add_argument("--fc", type=float, help="also print the expansion at this MHz")
    p.add_argument("--stations", type=Path, metavar="FILE")
    p.add_argument("--fast", action="store_true",
                   help="measure the fast part from the sounding's phase (.lfs, v2 "
                        "with z, echo window) in place of the Doppler prior")
    p.set_defaults(func=cmd_state)

    p = ss.add_parser("fast", help="the fast part from a sounding's phase: Doppler "
                                   "spread and shape, Rice K, o/x split")
    p.add_argument("sounding", nargs="?", help=".lfs, v2 .h5 with z, or .echo.h5")
    p.add_argument("--out", type=Path, help="write the state, fast part measured")
    p.add_argument("--json", type=Path, help="write the estimates, band by band")
    p.add_argument("--method", default="contour", help="extractor that finds the trace")
    p.add_argument("--stations", type=Path, metavar="FILE")
    p.add_argument("--calibrate", action="store_true",
                   help="instead: sound known fast parts through the law preset and "
                        "read them back")
    p.add_argument("--spreads", default="0.1,0.25,0.5,1,2",
                   help="calibrate: spreads, Hz, comma-separated")
    p.add_argument("--shapes", default="gaussian,lorentzian",
                   help="calibrate: Doppler spectrum shapes")
    p.add_argument("--k-db", type=float, help="calibrate: Rice K, dB")
    p.add_argument("--split", type=float, help="calibrate: o/x split, us")
    p.add_argument("--x-to-o", type=float, default=0.0,
                   help="calibrate: x power against o, dB")
    p.add_argument("--realisations", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_fast)

    p = ss.add_parser("preset", help="an ITU-R F.1487 test channel, or 'law'")
    p.add_argument("name", nargs="?", help="F.1487 condition (see --list) or 'law'")
    p.add_argument("--list", action="store_true")
    p.add_argument("--out", type=Path)
    p.add_argument("--fc", type=float, help="also print the expansion at this MHz")
    p.add_argument("--muf", type=float, default=25.0, help="law: MUF, MHz")
    p.add_argument("--tau0", type=float, default=8900.0,
                   help="law: low-frequency delay, us")
    p.add_argument("--b", type=float, default=7.52, help="law: GDD coefficient, us/MHz")
    p.add_argument("--cn0", type=float, default=45.0, help="law: C/N0, dB-Hz")
    p.add_argument("--spread", type=float, default=0.5,
                   help="law: frequency spread, Hz")
    p.set_defaults(func=cmd_preset)

    p = ss.add_parser("show", help="print a state, optionally expanded at a carrier")
    p.add_argument("state", type=Path)
    p.add_argument("--fc", type=float)
    p.add_argument("--provenance", action="store_true", help="every field's origin")
    p.set_defaults(func=cmd_show)

    p = ss.add_parser("sounding", help="synthesise an .lfs from a state")
    p.add_argument("state", type=Path)
    p.add_argument("--out", type=Path, required=True, help=".lfs file or a directory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--duration", type=float, help="seconds, default the sweep's")
    p.add_argument("--no-fading", action="store_true")
    p.add_argument("--no-doppler", action="store_true")
    p.add_argument("--no-ox", action="store_true")
    p.add_argument("--no-delay-spread", action="store_true")
    p.add_argument("--no-noise", action="store_true")
    p.set_defaults(func=cmd_sounding)

    p = ss.add_parser("channel", help="a signal through the channel at one carrier")
    p.add_argument("state", type=Path)
    p.add_argument("--fc", type=float, required=True, help="carrier, MHz")
    p.add_argument("--fs", type=float, required=True, help="sample rate, Hz")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, help="complex baseband samples, .npy")
    src.add_argument("--test", choices=TEST_SIGNALS, help="a built-in test signal")
    p.add_argument("--duration", type=float, default=1.0, help="test signal length, s")
    p.add_argument("--out", type=Path, required=True,
                   help="output .npy (+ .json beside)")
    p.add_argument("--dispersion", choices=("none", "gdd", "gdd+tod", "full"),
                   default="full")
    p.add_argument("--snr-db", type=float)
    p.add_argument("--snr-bandwidth", type=float, metavar="HZ",
                   help="bandwidth --snr-db is measured in (default fs)")
    p.add_argument("--absolute", action="store_true",
                   help="keep the state's C/N0 levels, noise of unit density")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-fading", action="store_true")
    p.add_argument("--no-doppler", action="store_true")
    p.add_argument("--no-multipath", action="store_true")
    p.add_argument("--no-ox", action="store_true")
    p.add_argument("--no-delay-spread", action="store_true")
    p.set_defaults(func=cmd_channel)

    p = ss.add_parser("check", help="closed loop: sounding -> state -> synthetic "
                                    "soundings -> state")
    p.add_argument("sounding")
    p.add_argument("--realisations", type=int, default=12,
                   help="synthetic soundings; level 2 draws its verdicts from 12 "
                        "or more")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", default="contour")
    p.add_argument("--keep", type=Path, metavar="DIR",
                   help="write the synthetic soundings here and keep them")
    p.add_argument("--json", type=Path, help="write the full report here")
    p.add_argument("--next", type=Path, metavar="SOUNDING",
                   help="the circuit's following sounding: report the natural change "
                        "between the two beside the recovery errors")
    p.add_argument("--fast", action="store_true",
                   help="measure the fast part from the sounding's phase, and read it "
                        "back from the first synthetic soundings")
    p.add_argument("--fast-realisations", type=int, default=3,
                   help="synthetic soundings the fast part is read back from, about "
                        "13 s each")
    p.set_defaults(func=cmd_check)

    p = ss.add_parser("forecast", help="persistence: a sounding's state as the "
                                       "forecast for a lead later, scored against "
                                       "the next sounding")
    p.add_argument("sounding")
    p.add_argument("--next", type=Path, metavar="SOUNDING",
                   help="the sounding the forecast is for: score it against its state")
    p.add_argument("--lead", type=float, default=300.0, help="seconds ahead")
    p.add_argument("--out", type=Path, help="write the forecast state here")
    p.add_argument("--json", type=Path, help="write the pair's score here")
    p.add_argument("--carriers", metavar="LO:HI:STEP",
                   help="working frequencies, MHz (default every 0.25 MHz across "
                        "the sweep)")
    p.add_argument("--bandwidth", type=float, default=3000.0,
                   help="band a link's SNR is read in, Hz")
    p.add_argument("--snr-min", type=float, default=10.0,
                   help="SNR in that band a link is usable from, dB")
    p.add_argument("--copies", type=int, default=0,
                   help="synthetic soundings of the forecast to compare the "
                        "ionograms with (80 MB each, deleted once read); 12 or "
                        "more for a verdict")
    p.add_argument("--method", default="contour", help="extractor that finds the trace")
    p.add_argument("--doppler", default="mid-moderate",
                   help="prior for the spread a power ionogram cannot measure")
    p.add_argument("--window", type=int, default=spectro.DEFAULT_WINDOW)
    p.add_argument("--knot-mhz", type=float, default=0.1)
    p.add_argument("--stations", type=Path, metavar="FILE")
    p.set_defaults(func=cmd_forecast)

    p = ss.add_parser("series", help="persistence scored over a run of soundings; "
                                     "rows to a folder, one pair at a time")
    p.add_argument("paths", nargs="+", type=Path, help="soundings or folders of them")
    p.add_argument("--out", type=Path, required=True, help="folder for the rows")
    p.add_argument("--circuit", metavar="TX:RX",
                   help="the circuit, when the soundings hold more than one")
    p.add_argument("--lead", type=float, default=300.0, help="seconds ahead")
    p.add_argument("--tolerance", type=float, default=30.0,
                   help="seconds the next sounding may miss the lead by")
    p.add_argument("--copies", type=int, default=4,
                   help="synthetic soundings of each forecast for the ionograms")
    p.add_argument("--recovery-every", type=int, default=12,
                   help="run the closed loop on every k-th sounding (0: never)")
    p.add_argument("--recovery-copies", type=int, default=4,
                   help="copies the closed loop reads, shared with the ionograms")
    p.add_argument("--carriers", metavar="LO:HI:STEP",
                   help="working frequencies, MHz (default every 0.25 MHz across "
                        "the sweep)")
    p.add_argument("--bandwidth", type=float, default=3000.0,
                   help="band a link's SNR is read in, Hz")
    p.add_argument("--snr-min", type=float, default=10.0,
                   help="SNR in that band a link is usable from, dB")
    p.add_argument("--jobs", type=int, default=1, help="pairs scored at once")
    p.add_argument("--resume", action="store_true",
                   help="carry on a run in --out, scoring only the pairs it lacks")
    p.add_argument("--method", default="contour", help="extractor that finds the trace")
    p.add_argument("--doppler", default="mid-moderate",
                   help="prior for the spread a power ionogram cannot measure")
    p.add_argument("--window", type=int, default=spectro.DEFAULT_WINDOW)
    p.add_argument("--knot-mhz", type=float, default=0.1)
    p.add_argument("--stations", type=Path, metavar="FILE")
    p.set_defaults(func=cmd_series)

    p = ss.add_parser("summary", help="a series' rows summed up, with block-bootstrap "
                                      "intervals, into its summary.json")
    p.add_argument("run", type=Path, help="the folder muf sim series wrote")
    p.add_argument("--by", default="hour,day,sun,lead,fmuf",
                   help="cuts, of hour, day, sun, lead, fmuf (mode classes are "
                        "always summed apart)")
    p.add_argument("--block", type=int, default=12, help="pairs to a bootstrap block")
    p.add_argument("--reps", type=int, default=4000, help="bootstrap resamples")
    p.set_defaults(func=cmd_summary)

    p = ss.add_parser("live", help="persistence as it would run live: an archive "
                                   "replayed, each forecast scored when its "
                                   "sounding comes")
    p.add_argument("paths", nargs="+", type=Path, help="soundings or folders of them")
    p.add_argument("--circuit", metavar="TX:RX",
                   help="the circuit, when the soundings hold more than one")
    p.add_argument("--lead", metavar="S[,S...]",
                   help="seconds ahead (default the next two soundings)")
    p.add_argument("--tolerance", type=float, default=30.0,
                   help="seconds a sounding may miss a forecast's time by")
    p.add_argument("--pace", type=float, default=0.0,
                   help="times faster than real time (0: as fast as it goes)")
    p.add_argument("--span", type=float, default=3600.0,
                   help="seconds of pairs the running figures cover")
    p.add_argument("--out", type=Path, metavar="FILE",
                   help="the events as JSON Lines")
    p.add_argument("--method", default="contour", help="extractor that finds the trace")
    p.add_argument("--doppler", default="mid-moderate",
                   help="prior for the spread a power ionogram cannot measure")
    p.add_argument("--window", type=int, default=spectro.DEFAULT_WINDOW)
    p.add_argument("--knot-mhz", type=float, default=0.1)
    p.add_argument("--stations", type=Path, metavar="FILE")
    p.set_defaults(func=cmd_live)

    p = ss.add_parser("verify", help="level 0: the generator against closed-form "
                                     "theory (fading, S(tau, nu), DBPSK)")
    p.add_argument("--quick", action="store_true",
                   help="runs a tenth of F.1487's test length; looser tolerances")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", type=Path, help="write every check here")
    p.set_defaults(func=cmd_verify)


if __name__ == "__main__":  # pragma: no cover
    sys.exit("run as: muf sim ...")
