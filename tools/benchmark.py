"""What is this box actually spending its time on, and has that changed?

Run it on the station, keep the JSON, run it again when something feels slow
and pass the old file as ``--baseline``. The interesting output is not any
single number -- boxes differ and a figure from the dev Mac says nothing about
a station -- it is the *difference* between two runs of the same machine, and a
handful of ratios that mean the same thing everywhere.

Three questions it answers that a stopwatch cannot:

1. **Is the parallelism real?** Workers are processes, and every one of them
   opens its own BLAS and OpenMP thread pools. Unpinned, eight workers on ten
   cores fight over the machine with eighty threads and the speed-up collapses
   to under 3x. `muf.pipeline.PIN_THREADS` holds them to one thread each; this
   tool measures whether that is still working, because the failure mode is
   silent -- nothing errors, the run is merely half the speed it should be.

2. **Is the archive mount keeping up?** Reading an 80 MB sounding should be a
   rounding error next to processing it: on a local disk it is 6-12 ms against
   308 ms of work, about 3 %. On a network mount that ratio is the first thing
   to move, and it moves long before anyone notices a page is slow.

3. **Did the answers change?** Every performance change has to be checked
   against the picks it produces, so ``--picks`` writes the MUF/LOF column out
   and ``--baseline`` diffs it. A speed-up that moves a MUF is not a speed-up.

Nothing here writes to the archive or the database. The extraction benchmark
reads a bounded sample -- ``--files``, forty by default, drawn with a fixed
seed so two runs measure the same soundings -- which costs a minute or so on a
station rather than the half hour a full day would. It runs that sample twice,
once serially and once across workers, because the ratio between the two is
the portable number.

    python tools/benchmark.py --archive /data/lfs --json today.json
    python tools/benchmark.py --archive /data/lfs --baseline today.json
    python tools/benchmark.py --url http://127.0.0.1:8000 --token "$READ_TOKEN"

Reference figures from the machine this was written on -- 10 cores, local SSD,
one day of the real archive -- so a wildly different box is recognisable as
such rather than mistaken for a regression: 15.3 files/s at ``jobs=8`` pinned,
308 ms of CPU per sounding, 303 MB resident per worker, warm pages under 6 ms.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import resource
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: Ratios that mean the same thing on any machine, and the point at which each
#: stops being a fact about the hardware and starts being a problem.
#:
#: Absolute timings are deliberately absent: the only honest comparison for
#: those is this same box on an earlier day, which is what ``--baseline`` is.
LIMITS = {
    "speedup": 3.0,       # x over one worker, at >= 4 cores. Below: unpinned?
    "io_share": 0.25,     # of per-sounding time spent reading. Above: the mount
    "rss_growth": 1.15,   # peak RSS after N files over after one. Above: a leak
    "error_rate": 0.0,    # soundings that raised. Above: not a speed problem
    "regression": 0.20,   # slower than --baseline by this much: report it
}

#: Below this much serial work, the speed-up figure is measuring the pool
#: rather than the pipeline and no verdict is drawn from it.
#:
#: Spawning eight workers costs about 0.8 s and each one then imports
#: scikit-learn before its first task. Against a 2 s serial run that overhead
#: *is* the result: a healthy machine benchmarked over eight files reports
#: 1.2x and looks broken. Ten seconds of serial work puts the fixed cost under
#: a tenth of the total, which is roughly 40 soundings.
MIN_SERIAL_S = 10.0

#: Pages worth timing, and whether the answer is expected to be cheap. The
#: archive-backed ones are the only places real work happens on a request.
PAGES = [
    ("/healthz", "liveness"),
    ("/ui", "console"),
    ("/ui/soundings", "soundings table"),
    ("/ui/series?method=kmeans&circuit=all&model=off", "series, no model"),
    ("/ui/series?method=kmeans&circuit=all", "series + IRI"),
    ("/ui/sources", "sources census"),
    ("/soundings?limit=500", "json soundings"),
]


# --- helpers -----------------------------------------------------------------

def rss_mb() -> float:
    """Peak resident set size. macOS reports bytes, Linux kilobytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def sample(archive: Path, count: int, seed: int = 0) -> list[Path]:
    """A stable random sample, so two runs benchmark the same files."""
    found = sorted(archive.rglob("*.lfs")) or sorted(archive.rglob("*.h5"))
    if not found:
        raise FileNotFoundError(f"no soundings under {archive}")
    if len(found) <= count:
        return found
    return sorted(random.Random(seed).sample(found, count))


def pct(part: float, whole: float) -> float:
    return round(part / whole, 4) if whole else 0.0


# --- extraction --------------------------------------------------------------

def measure_stages(path: Path) -> dict:
    """Where one sounding's time goes, timed directly rather than by profiler.

    Run twice, reporting the second: the first pays for importing scikit-learn
    and warming the file cache, which is a real cost but a once-per-process one
    and not what this number is for.
    """
    from muf import extractors, spectro
    from muf.io_lfs import read_iq

    out: dict = {}
    for attempt in (0, 1):
        t0 = time.perf_counter()
        read_iq(path)
        t1 = time.perf_counter()
        ion = spectro.compute(path)
        t2 = time.perf_counter()

        methods: dict[str, float] = {}
        for name in extractors.DEFAULT_METHODS:
            t3 = time.perf_counter()
            try:
                extractors.get(name)(ion)
            except Exception:            # an estimator that cannot run is not
                methods[name] = float("nan")   # a timing result; say so
                continue
            methods[name] = round((time.perf_counter() - t3) * 1000, 1)

        if attempt:
            total = (time.perf_counter() - t0) * 1000
            out = {
                "read_ms": round((t1 - t0) * 1000, 1),
                "spectrogram_ms": round((t2 - t1) * 1000, 1),
                "methods_ms": methods,
                "total_ms": round(total, 1),
                "io_share": pct((t1 - t0) * 1000, total),
                "n_freq": int(ion.power.shape[0]),
                "n_range": int(ion.power.shape[1]),
            }
    return out


def measure_extraction(paths: list[Path], jobs: int) -> dict:
    """Throughput at one worker and at ``jobs``, and what scaling was won."""
    from muf import pipeline

    result: dict = {"n_files": len(paths), "jobs": jobs,
                    "pinned": pipeline.PIN_THREADS}
    targets = [str(p) for p in paths]

    before = rss_mb()
    for label, workers in (("serial", 1), ("parallel", jobs)):
        started = time.perf_counter()
        frame = pipeline.process_many(
            targets, pipeline.Options(), jobs=workers, progress=False)
        elapsed = time.perf_counter() - started
        errors = int(frame["error"].notna().sum()) if "error" in frame else 0
        result[label] = {
            "workers": workers,
            "seconds": round(elapsed, 2),
            "files_per_s": round(len(paths) / elapsed, 2),
            "ms_per_file": round(elapsed / len(paths) * 1000, 1),
            "errors": errors,
        }
        result["error_rate"] = pct(errors, len(paths))

    serial = result["serial"]["seconds"]
    result["speedup"] = round(serial / result["parallel"]["seconds"], 2)
    result["efficiency"] = round(result["speedup"] / jobs, 3)
    result["rss_mb"] = round(rss_mb(), 1)
    result["rss_growth"] = round(rss_mb() / before, 3) if before else 1.0
    return result


def measure_picks(paths: list[Path], jobs: int) -> list[dict]:
    """The picks themselves, so a later run can prove nothing moved."""
    from muf import pipeline

    frame = pipeline.process_many(
        [str(p) for p in paths], pipeline.Options(), jobs=jobs, progress=False)
    keep = [c for c in frame.columns
            if c == "datetime" or c.startswith(("muf", "lof"))]
    rows = frame[keep].sort_values("datetime").to_dict("records")
    return [{k: _plain(v) for k, v in row.items()} for row in rows]


def _plain(value):
    """JSON, and comparable across runs: timestamps as text, NaN as null."""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, float) and value != value:
        return None
    return value.item() if hasattr(value, "item") else value


# --- the served pages --------------------------------------------------------

def fetch(url: str, token: str | None) -> tuple[int, int, float]:
    request = urllib.request.Request(url)
    request.add_header("Accept-Encoding", "gzip")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
            code = response.status
    except urllib.error.HTTPError as exc:
        body, code = exc.read(), exc.code
    except OSError as exc:
        raise SystemExit(f"cannot reach {url}: {exc}")
    return code, len(body), (time.perf_counter() - started) * 1000


def measure_pages(base: str, token: str | None, repeats: int = 6) -> list[dict]:
    """Cold and warm latency per page, and what each costs on the wire.

    Cold is the first call and warm the median of the rest, because that gap is
    the whole story on this service: nearly every page is already faster than
    anyone can perceive once its imports and caches are warm, and the cost is
    concentrated in whoever asks first.
    """
    out = []
    for path, label in PAGES:
        code, size, first = fetch(base.rstrip("/") + path, token)
        rest = [fetch(base.rstrip("/") + path, token)[2]
                for _ in range(repeats - 1)]
        out.append({
            "page": label,
            "path": path,
            "status": code,
            "cold_ms": round(first, 1),
            "warm_ms": round(statistics.median(rest), 1),
            "p99_ms": round(max([first] + rest), 1),
            "kb": round(size / 1024, 1),
        })
    return out


# --- verdicts ----------------------------------------------------------------

def verdicts(run: dict, baseline: dict | None) -> list[tuple[str, str, str]]:
    """Turn the numbers into the short list of things worth looking at."""
    found: list[tuple[str, str, str]] = []

    extraction = run.get("extraction")
    if extraction:
        cores = run["host"]["cpus"]
        serial_s = extraction["serial"]["seconds"]
        if serial_s < MIN_SERIAL_S:
            found.append((
                "INFO", "sample too small to judge scaling",
                f"{serial_s:.1f} s of serial work is not enough to see past "
                f"pool start-up; the {extraction['speedup']}x above is mostly "
                f"the cost of spawning workers. Re-run with --files "
                f"{max(40, extraction['n_files'] * 4)} to get a real figure."))
        elif cores >= 4 and extraction["speedup"] < LIMITS["speedup"]:
            found.append((
                "BAD", "parallel speed-up",
                f"{extraction['speedup']}x over one worker on {cores} cores. "
                f"Expected above {LIMITS['speedup']}x -- check that "
                f"MUF_PIN_THREADS is not set to 0 "
                f"(currently pinned={extraction['pinned']}), and that nothing "
                f"else is using the machine."))
        if extraction["error_rate"] > LIMITS["error_rate"]:
            found.append((
                "BAD", "soundings that failed",
                f"{extraction['error_rate']:.0%} of the sample raised. That is "
                f"a correctness problem, not a speed one -- run "
                f"tools/diagnose_reception.py."))
        if extraction["rss_growth"] > LIMITS["rss_growth"]:
            found.append((
                "WARN", "memory growth",
                f"peak RSS grew {extraction['rss_growth']}x across the sample. "
                f"A worker should settle after its first file."))

    stages = run.get("stages")
    if stages and stages["io_share"] > LIMITS["io_share"]:
        found.append((
            "WARN", "archive reads",
            f"{stages['io_share']:.0%} of a sounding is spent reading it "
            f"({stages['read_ms']} ms of {stages['total_ms']} ms). On a local "
            f"disk this is about 3 %. Suspect the mount."))

    for page in run.get("pages", []):
        if page["status"] != 200:
            found.append(("BAD", f"page {page['page']}",
                          f"HTTP {page['status']}"))

    if baseline:
        found += regressions(run, baseline)

    return found


def regressions(run: dict, baseline: dict) -> list[tuple[str, str, str]]:
    """Only ever this box against itself, which is the comparison that holds."""
    found = []
    if run["host"]["node"] != baseline["host"].get("node"):
        found.append((
            "WARN", "different machine",
            f"baseline was taken on {baseline['host'].get('node')!r}, this is "
            f"{run['host']['node']!r}. Timings are not comparable across "
            f"boxes; only the ratios above are."))
        return found

    was = baseline.get("extraction")
    now = run.get("extraction")
    if was and now:
        ratio = now["parallel"]["ms_per_file"] / was["parallel"]["ms_per_file"]
        if ratio > 1 + LIMITS["regression"]:
            found.append((
                "BAD", "extraction slower",
                f"{now['parallel']['ms_per_file']} ms/file against "
                f"{was['parallel']['ms_per_file']} ms/file in the baseline "
                f"({ratio:.2f}x)."))

    old_pages = {p["path"]: p for p in baseline.get("pages", [])}
    for page in run.get("pages", []):
        prior = old_pages.get(page["path"])
        if not prior or prior["warm_ms"] <= 0:
            continue
        ratio = page["warm_ms"] / prior["warm_ms"]
        if ratio > 1 + LIMITS["regression"] and page["warm_ms"] > 5:
            found.append((
                "WARN", f"page {page['page']}",
                f"{page['warm_ms']} ms warm against {prior['warm_ms']} ms "
                f"({ratio:.2f}x)."))

    if baseline.get("picks") and run.get("picks"):
        if baseline["picks"] != run["picks"]:
            moved = sum(1 for a, b in zip(baseline["picks"], run["picks"])
                        if a != b)
            found.append((
                "BAD", "the picks moved",
                f"{moved} of {len(run['picks'])} rows differ from the "
                f"baseline. Whatever changed, changed the measurements."))
    return found


# --- reporting ---------------------------------------------------------------

def report(run: dict, found: list[tuple[str, str, str]]) -> None:
    host = run["host"]
    print(f"\n{host['node']}  {host['cpus']} cpus  {host['platform']}  "
          f"python {host['python']}")
    print(f"archive {run['archive']}\n")

    stages = run.get("stages")
    if stages:
        print("one sounding, warm")
        print(f"  read            {stages['read_ms']:8.1f} ms"
              f"   {stages['io_share']:.0%}")
        print(f"  spectrogram     {stages['spectrogram_ms']:8.1f} ms")
        for name, ms in stages["methods_ms"].items():
            print(f"  {name:<15} {ms:8.1f} ms")
        print(f"  {'total':<15} {stages['total_ms']:8.1f} ms\n")

    extraction = run.get("extraction")
    if extraction:
        print(f"extraction over {extraction['n_files']} files"
              f"   (threads pinned: {extraction['pinned']})")
        for label in ("serial", "parallel"):
            block = extraction[label]
            print(f"  {block['workers']:>2} worker(s)  {block['seconds']:7.2f} s"
                  f"   {block['files_per_s']:6.2f} files/s"
                  f"   {block['ms_per_file']:7.1f} ms/file")
        print(f"  speed-up {extraction['speedup']}x"
              f"   efficiency {extraction['efficiency']:.0%}"
              f"   peak rss {extraction['rss_mb']:.0f} MB\n")

    pages = run.get("pages")
    if pages:
        # KB is what crossed the wire, compressed, not the size of the page.
        print(f"{'page':<20}{'cold ms':>9}{'warm ms':>9}{'p99 ms':>9}"
              f"{'wire KB':>9}")
        for page in pages:
            print(f"  {page['page']:<18}{page['cold_ms']:9.1f}"
                  f"{page['warm_ms']:9.1f}{page['p99_ms']:9.1f}"
                  f"{page['kb']:9.1f}")
        print()

    if not found:
        print("nothing to look at.\n")
        return
    print("worth looking at")
    for level, what, detail in found:
        print(f"  [{level:<4}] {what}: {detail}")
    print()


# --- entry point -------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path,
                        default=Path(os.environ.get("ARCHIVE_ROOT", "data")),
                        help="where the soundings are")
    parser.add_argument("--files", type=int, default=40,
                        help="how many to benchmark (default 40: fewer than "
                             "that and the scaling figure is measuring pool "
                             "start-up rather than the pipeline)")
    parser.add_argument("--jobs", type=int, default=0,
                        help="workers; 0 means cores minus one")
    parser.add_argument("--url", default=None,
                        help="a running api to time, e.g. http://127.0.0.1:8000")
    parser.add_argument("--token", default=os.environ.get("READ_TOKEN"),
                        help="read token, if that api needs one")
    parser.add_argument("--picks", action="store_true",
                        help="record the picks too, so a later run can diff them")
    parser.add_argument("--json", type=Path, default=None,
                        help="write the run here, to use as a later --baseline")
    parser.add_argument("--baseline", type=Path, default=None,
                        help="an earlier run of this box to compare against")
    parser.add_argument("--no-extract", action="store_true",
                        help="skip extraction and time only the served pages")
    args = parser.parse_args(argv)

    jobs = args.jobs or max(1, (os.cpu_count() or 2) - 1)
    run: dict = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "archive": str(args.archive),
        "host": {
            "node": platform.node(),
            "cpus": os.cpu_count() or 0,
            "platform": platform.platform(terse=True),
            "python": platform.python_version(),
        },
    }

    if not args.no_extract:
        paths = sample(args.archive, args.files)
        run["stages"] = measure_stages(paths[0])
        run["extraction"] = measure_extraction(paths, jobs)
        if args.picks:
            run["picks"] = measure_picks(paths, jobs)

    if args.url:
        run["pages"] = measure_pages(args.url, args.token)

    baseline = None
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())

    found = verdicts(run, baseline)
    report(run, found)

    if args.json:
        args.json.write_text(json.dumps(run, indent=2) + "\n")
        print(f"written to {args.json}")

    return 1 if any(level == "BAD" for level, _, _ in found) else 0


if __name__ == "__main__":
    raise SystemExit(main())
