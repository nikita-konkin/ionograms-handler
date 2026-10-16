"""The ionozond console's schedule checks, ported so the web asks the same questions.

A port of ionozond's ``src/schedulecheck.cpp``, held to its own test cases
(``tests/test_schedulecheck.cpp``) in ``tests/test_agent_schedulecheck.py``.
It finds the schedules that record less than they look as if they would:
stations whose soundings collide on the one radio, replayed from the
sounder's own choice of what to record; values ``chirp_config.py`` cannot
carry; and settings that are legal but throw data away.

The levels are the console's. Errors refuse a change. Warnings are what the
console lets an operator at the laptop override with START, so the web shows
them and asks before it applies the change. The console's file checks --
its START program, ``data_dir`` and free space -- are left out: under the
service the unit names the archive and the program, not the console.

The messages are in English, like every other message the agent returns.
They keep the console's facts and numbers, which is what the tests compare.
"""

from __future__ import annotations

import bisect
import math
import re
from dataclasses import dataclass, replace

ERROR, WARNING, INFO = 0, 1, 2

DAY_S = 86400.0
EPS = 1e-6
SOUNDER_LEAD_S = 3.0
EARTH_RADIUS_KM = 6378.137
LAY_E_SQR = 40000.0
LIGHT_SPEED_KM_S = 300e3
VIRT_HEIGHT_WINDOW_KM = 1500.0
VIRT_HEIGHT_WINDOW_MARGIN_KM = 180.0
FFT_COUNT_LIST = [512 << i for i in range(8)]
SAMPLE_RATE_LIST = [3125.0, 6250.0, 12500.0, 25000.0]

NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
LEADING_ZERO = re.compile(r"^[+-]?0+[1-9][0-9]*$")
LABELS = ("repetition period", "chirptime", "sweep rate", "duration",
          "oscillator frequency")


@dataclass(frozen=True)
class Entry:
    """One station of the schedule, its numbers as typed."""

    name: str
    active: bool
    rx: bool
    rep: str
    chirpt: str
    rate: str
    dur: str
    cf: str
    lat: str
    lon: str
    fft: int = 0


@dataclass(frozen=True)
class Settings:
    """The parameters the checks need, in the console's units."""

    sample_rate_khz: float = 25000.0
    sample_rate_index: int = -1
    dec: float = 625.0
    fft: int = 16384
    obj_w: int = 9
    obj_h: int = 3
    obj_level: float = 11.0

    @property
    def if_rate_hz(self) -> float:
        return self.sample_rate_khz * 1e3 / self.dec if self.dec > 0 else 0.0


@dataclass(frozen=True)
class Issue:
    """One finding: its level, the station it is about, and why."""

    level: int
    station: str
    text: str

    def to_dict(self) -> dict:
        return {"level": ("error", "warning", "info")[self.level],
                "station": self.station, "text": self.text}


@dataclass(frozen=True)
class SimStation:
    """A station as the sounder schedules it: period, offset, capture, processing."""

    rep: float
    chirpt: float
    dur: float
    post: float


@dataclass(frozen=True)
class Capture:
    """A capture the sounder would start: which station, and when."""

    station: int
    start: float


@dataclass(frozen=True)
class Station:
    name: str
    rep: float
    chirpt: float
    rate_khz: float
    dur: float
    cf_khz: float


def processing_estimate_s(dur: float) -> float:
    """The sounder's sidecar and archive time after a capture, generously."""
    return 2.0 + 0.06 * dur


def check(entries: list[Entry], settings: Settings) -> list[Issue]:
    """Every issue with this schedule, errors first, as ``checkSchedule`` has it."""
    out: list[Issue] = []
    _general(settings, out)
    seen, twice = set(), []
    for e in entries:
        _name(e, out)
        if e.name in seen and e.name not in twice:
            twice.append(e.name)
        seen.add(e.name)
    for name in twice:
        out.append(Issue(ERROR, name, "two columns with one name: only one will be "
                                      "kept"))
    receivers = [e for e in entries if e.rx]
    rx_ok, rx_lat, rx_lon = False, 0.0, 0.0
    if receivers:
        rx_ok, rx_lat, rx_lon = _coords(receivers[0], out)
    if not receivers:
        out.append(Issue(WARNING, "", "no receiving station is chosen: the "
                                      "ionograms' delay window is built from "
                                      "coordinates 0, 0"))
    elif len(receivers) > 1:
        out.append(Issue(WARNING, "", "%d stations are marked as receivers; "
                                      "chirp_config.py gets the last"
                         % len(receivers)))
    active = []
    for e in entries:
        if not e.active:
            continue
        values, ok = [], True
        for k, text in enumerate((e.rep, e.chirpt, e.rate, e.dur, e.cf)):
            good, value, problem = _number(text)
            if not good:
                out.append(Issue(ERROR, e.name, "%s: %s" % (LABELS[k], problem)))
                ok = False
            elif k != 1 and value <= 0.0:
                out.append(Issue(ERROR, e.name, (
                    "repetition period %s: must be above zero, or the sounder hangs "
                    "choosing its next session" % _val(value)) if k == 0 else
                    "%s %s: must be above zero" % (LABELS[k], _val(value))))
                ok = False
            values.append(value)
        if e.rx:
            coords_ok, lat, lon = rx_ok, rx_lat, rx_lon
        else:
            coords_ok, lat, lon = _coords(e, out)
        if not ok:
            continue
        s = Station(e.name, *values)
        _station(s, e.fft if e.fft > 0 else settings.fft, settings,
                 coords_ok and rx_ok, lat, lon, rx_lat, rx_lon, out)
        active.append(s)
    if not [e for e in entries if e.active]:
        out.append(Issue(WARNING, "", "no station is marked for the active schedule: "
                                      "the sounder has nothing to record"))
    active.sort(key=lambda s: s.name)
    if active:
        _collisions(active, out)
    return sorted(out, key=lambda i: i.level)


def simulate(stations: list[SimStation], start: float, stop: float) -> list[Capture]:
    """``simulateSounder``: the captures in ``[start, stop)``, from midnight 0."""
    return _replayed(stations, 0.0, start, stop)


def upcoming(stations: list[SimStation], now: float, count: int) -> list[Capture]:
    """The next ``count`` captures the sounder would start after ``now``, unix time."""
    out: list[Capture] = []
    if not stations or not all(s.rep > 0.0 for s in stations):
        return out
    while len(out) < count:
        best, at = _picked(stations, now)
        out.append(Capture(best, at))
        now = at + max(0.0, stations[best].dur) + max(0.0, stations[best].post)
    return out


def _replayed(stations, now, start, stop) -> list[Capture]:
    out: list[Capture] = []
    if not stations or not all(s.rep > 0.0 for s in stations):
        return out
    while now < stop:
        best, at = _picked(stations, now)
        if at >= stop:
            break
        if at >= start:
            out.append(Capture(best, at))
        now = at + max(0.0, stations[best].dur) + max(0.0, stations[best].post)
    return out


def _picked(stations, now) -> tuple[int, float]:
    """``pick_next``: the earliest, the first listed on a tie."""
    best, at = -1, 0.0
    for i, s in enumerate(stations):
        t = _next_start(s, now)
        if best < 0 or t < at:
            best, at = i, t
    return best, at


def _next_start(s: SimStation, now: float) -> float:
    """``determine_next``: midnight + chirpt + k*rep, at least 3 s away."""
    first = math.floor(now / DAY_S) * DAY_S + s.chirpt
    t0 = first
    if t0 - now < SOUNDER_LEAD_S:
        k = math.ceil((now + SOUNDER_LEAD_S - first) / s.rep)
        t0 = first + max(0.0, k) * s.rep
        while t0 - now < SOUNDER_LEAD_S:
            t0 += s.rep
        while t0 - s.rep >= first and t0 - s.rep - now >= SOUNDER_LEAD_S:
            t0 -= s.rep
    return t0


def _number(text: str) -> tuple[bool, float, str]:
    t = text.strip()
    if not t:
        return False, 0.0, "not set"
    if not NUMBER.match(t) or not math.isfinite(float(t)):
        if "," in t and NUMBER.match(t.replace(",", ".")):
            return False, 0.0, '"%s": the fraction is separated by a point, not a ' \
                               'comma' % t
        return False, 0.0, '"%s" is not a number' % t
    if LEADING_ZERO.match(t):
        return False, 0.0, '"%s": Python does not accept an integer with a leading ' \
                           'zero' % t
    return True, float(t), ""


def _val(v: float) -> str:
    """``QString::number(v, 'g', 10)``."""
    return format(v, ".10g")


def _fixed(v: float, digits: int) -> str:
    return format(v, ".%df" % digits)


def _name(e: Entry, out: list[Issue]) -> None:
    n = e.name
    used = ERROR if e.active or e.rx else WARNING
    if not n.strip():
        out.append(Issue(ERROR, n, "the station has no name"))
        return
    if "/" in n or "\\" in n:
        out.append(Issue(ERROR, n, '"/" and "\\" in a name break schedule.ini and the '
                                   "capture paths"))
    if "'" in n or '"' in n:
        out.append(Issue(used, n, "a quote in the name: chirp_config.py would not be "
                                  "readable"))
    if any(ord(c) < 0x20 or ord(c) == 0x7F for c in n):
        out.append(Issue(ERROR, n, "a control character in the name"))
    if any(ord(c) > 0xFF for c in n):
        out.append(Issue(used, n, "the name is not in Latin letters: the sounder "
                                  "writes it into the capture header in Latin-1, and "
                                  "the capture fails. Use Latin letters, for example "
                                  "Norilsk."))
    elif len(n) > 64:
        out.append(Issue(WARNING, n, "the name is longer than 64 characters and is cut "
                                     "short in the capture header"))


def _coords(e: Entry, out: list[Issue]) -> tuple[bool, float, float]:
    good_lat, lat, lat_problem = _number(e.lat)
    good_lon, lon, lon_problem = _number(e.lon)
    ok = True
    if not good_lat:
        out.append(Issue(ERROR, e.name, "latitude: %s" % lat_problem))
        ok = False
    elif abs(lat) > 90.0:
        out.append(Issue(ERROR, e.name, "latitude %s outside -90..90" % _val(lat)))
        ok = False
    if not good_lon:
        out.append(Issue(ERROR, e.name, "longitude: %s" % lon_problem))
        ok = False
    elif abs(lon) > 180.0:
        out.append(Issue(ERROR, e.name, "longitude %s outside -180..180" % _val(lon)))
        ok = False
    if ok and lat == 0.0 and lon == 0.0:
        out.append(Issue(WARNING, e.name, "coordinates 0, 0: not set? The ionogram's "
                                          "delay window is built from them."))
    return ok, lat, lon


def _general(cfg: Settings, out: list[Issue]) -> None:
    if not cfg.sample_rate_khz > 0.0:
        out.append(Issue(ERROR, "", "the sample rate is not set"))
    if not cfg.dec > 0.0:
        out.append(Issue(ERROR, "", "the decimation must be above zero"))
    elif abs(cfg.dec - math.floor(cfg.dec)) > 1e-9:
        out.append(Issue(WARNING, "", "decimation %s is not whole: the sounder will "
                                      "use %s" % (_val(cfg.dec),
                                                  _fixed(math.floor(cfg.dec), 0))))
    index = cfg.sample_rate_index
    if (0 <= index < len(SAMPLE_RATE_LIST)
            and abs(SAMPLE_RATE_LIST[index] - cfg.sample_rate_khz) > 1e-6):
        out.append(Issue(WARNING, "", "the sample rate is written twice in config.ini, "
                                      "and differently (%s and %s kHz). Open the "
                                      "parameters dialog and save it."
                         % (_val(cfg.sample_rate_khz),
                            _val(SAMPLE_RATE_LIST[index]))))
    w = 2 * (max(cfg.obj_w, 0) // 2) + 1
    h = 2 * (max(cfg.obj_h, 0) // 2) + 1
    if cfg.obj_level > w * h:
        out.append(Issue(ERROR, "", "the small-object filter's threshold %s exceeds "
                                    "the %d x %d = %d cells in its window: "
                                    "everything would be removed from the ionogram"
                         % (_val(cfg.obj_level), w, h, w * h)))


def _distance_km(lat1, lon1, lat2, lon2) -> float:
    la1, la2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2) - math.radians(lon1)
    x = (math.sin(la1) * math.sin(la2)
         + math.cos(la1) * math.cos(la2) * math.cos(dlon))
    y = math.hypot(math.cos(la2) * math.sin(dlon),
                   math.cos(la1) * math.sin(la2)
                   - math.sin(la1) * math.cos(la2) * math.cos(dlon))
    return EARTH_RADIUS_KM * math.atan2(y, x)


def _station(s: Station, fft: int, cfg: Settings, path_ok: bool, tx_lat: float,
             tx_lon: float, rx_lat: float, rx_lon: float, out: list[Issue]) -> None:
    n, if_rate = s.name, cfg.if_rate_hz
    sr_hz, rate_hz = cfg.sample_rate_khz * 1e3, s.rate_khz * 1e3
    if s.cf_khz < 1000.0:
        out.append(Issue(WARNING, n, "the oscillator frequency %s kHz is below 1 MHz. "
                                     "It is given in kHz: 12.5 MHz is 12500."
                         % _val(s.cf_khz)))
    if s.rate_khz * s.dur < 1000.0:
        out.append(Issue(WARNING, n, "the sweep covers only %s kHz (rate x duration). "
                                     "The rate is given in kHz/s: 100 kHz/s is 100."
                         % _val(s.rate_khz * s.dur)))
    if sr_hz > 0.0 and rate_hz > 0.0 and s.dur - sr_hz / rate_hz >= 0.5:
        fits = sr_hz / rate_hz
        out.append(Issue(WARNING, n, "the sweep of %s s x %s kHz/s = %s MHz is wider "
                                     "than the %s MHz receive band: the last %s s are "
                                     "recorded outside the band, as noise. %s s is "
                                     "enough."
                         % (_val(s.dur), _val(s.rate_khz),
                            _fixed(rate_hz * s.dur / 1e6, 1), _fixed(sr_hz / 1e6, 1),
                            _fixed(s.dur - fits, 1),
                            _fixed(math.floor(fits + 1e-9), 0))))
    if sr_hz > 0.0 and s.cf_khz * 1e3 - sr_hz / 2.0 < 0.0:
        out.append(Issue(WARNING, n, "the sweep would start at %s MHz, below zero: the "
                                     "oscillator is set to the transmitter's start "
                                     "frequency plus half the band (%s MHz)."
                         % (_fixed((s.cf_khz * 1e3 - sr_hz / 2.0) / 1e6, 2),
                            _fixed(sr_hz / 2e6, 3))))
    if s.chirpt < 0.0 or s.chirpt >= s.rep:
        fold = s.chirpt - math.floor(s.chirpt / s.rep) * s.rep
        out.append(Issue(WARNING, n, "chirptime %s is outside 0..%s s: the sounder "
                                     "misses sessions at the start of the day, and the "
                                     "console shows another time. It is the same "
                                     "session as chirptime %s."
                         % (_val(s.chirpt), _val(s.rep), _val(fold))))
    if abs(DAY_S / s.rep - math.floor(DAY_S / s.rep + 0.5)) > 1e-9:
        out.append(Issue(WARNING, n, "a day does not divide into periods of %s s: the "
                                     "sounder restarts its session grid at UTC "
                                     "midnight, the console keeps it running, and over "
                                     "a day the two drift apart." % _val(s.rep)))
    if path_ok and if_rate > 0.0 and rate_hz > 0.0:
        d_km = _distance_km(tx_lat, tx_lon, rx_lat, rx_lon)
        tau = (math.sqrt(LAY_E_SQR + d_km * d_km) + VIRT_HEIGHT_WINDOW_KM) \
            / LIGHT_SPEED_KM_S
        reach = if_rate / 2.0 / rate_hz
        if tau > reach:
            out.append(Issue(ERROR, n, "the delay window reaches %s ms (a %s km path), "
                                       "but at %s kHz/s and a band of +-%s kHz after "
                                       "decimation no more than %s ms can be told "
                                       "apart: echoes would land on other delays. "
                                       "Lower the rate or the decimation."
                             % (_fixed(tau * 1e3, 1), _fixed(d_km, 0),
                                _val(s.rate_khz), _fixed(if_rate / 2e3, 1),
                                _fixed(reach * 1e3, 1))))
    if if_rate > 0.0 and fft > 0 and rate_hz > 0.0:
        _fft(s, fft, cfg, if_rate, rate_hz, out)


def _fft(s: Station, fft: int, cfg: Settings, if_rate: float, rate_hz: float,
         out: list[Issue]) -> None:
    n = s.name
    samples = if_rate * s.dur
    spectra = math.floor(samples / fft)
    rows = rate_hz * ((VIRT_HEIGHT_WINDOW_KM + VIRT_HEIGHT_WINDOW_MARGIN_KM)
                      / LIGHT_SPEED_KM_S) * fft / if_rate
    if spectra < 1.0:
        out.append(Issue(ERROR, n, "an FFT of %d points is longer than the whole "
                                   "capture (%s samples): there would be no ionogram"
                         % (fft, _fixed(samples, 0))))
    elif spectra <= 2 * (max(cfg.obj_w, 0) // 2):
        out.append(Issue(ERROR, n, "only %s spectra in frequency, and the small-object "
                                   "filter's window is %d: it would remove everything. "
                                   "Use a shorter FFT." % (_fixed(spectra, 0),
                                                           cfg.obj_w)))
    elif spectra < 60.0:
        out.append(Issue(WARNING, n, "only %s spectra in frequency (the original had "
                                     "610): the ionogram will be coarse. Use a shorter "
                                     "FFT." % _fixed(spectra, 0)))
    if rows < 30.0:
        out.append(Issue(WARNING, n, "the delay window spans only %s FFT rows: the "
                                     "ionogram will be coarse. Use a longer FFT."
                         % _fixed(rows, 1)))
    ideal = 16384.0 * (100e3 / rate_hz) * (if_rate / 40000.0)
    rec = max(512, min(65536, 1 << math.floor(math.log(ideal) / math.log(2.0) + 0.5)))
    if fft >= 4 * rec or rec >= 4 * fft:
        sweep, rec_sweep = rate_hz * fft / if_rate, rate_hz * rec / if_rate
        out.append(Issue(INFO, n, "an ionogram cell of %s kHz x %s km (the original's "
                                  "was 41 x 7.3). At %s kHz/s an FFT of %d gives the "
                                  "original's shape: %s x %s."
                         % (_fixed(sweep / 1e3, 0),
                            _fixed(LIGHT_SPEED_KM_S / sweep, 1), _val(s.rate_khz), rec,
                            _fixed(rec_sweep / 1e3, 0),
                            _fixed(LIGHT_SPEED_KM_S / rec_sweep, 1))))


@dataclass(frozen=True)
class Loss:
    count: int = 0
    max_dur: float = 1e300
    min_gap: float = 1e300


def _key(start: float, station: int) -> int:
    return math.floor(start * 1000.0 + 0.5) * 64 + station


def _losses(s: list[SimStation], together: list[int] | None = None) -> dict:
    """``lossesOf``: sessions each station records alone and not with the others."""
    out: dict = {}
    captured = simulate(s, 0.0, 2.0 * DAY_S)
    starts = [c.start for c in captured]
    recorded = {_key(c.start, c.station) for c in captured}
    if together is not None:
        for c in captured:
            if c.start >= DAY_S:
                together[c.station] += 1
    for i in range(len(s)):
        for own in simulate([s[i]], DAY_S, 2.0 * DAY_S):
            t = own.start
            if _key(t, i) in recorded:
                continue
            at = bisect.bisect_right(starts, t + EPS) - 1
            if at < 0:
                continue
            c = captured[at]
            cs = s[c.station]
            if c.station == i or c.start + cs.dur + cs.post <= t - SOUNDER_LEAD_S - EPS:
                continue
            was = out.get((i, c.station), Loss())
            out[(i, c.station)] = Loss(
                was.count + 1,
                min(was.max_dur, t - SOUNDER_LEAD_S - cs.post - c.start),
                min(was.min_gap, t - (c.start + cs.dur)))
    return out


def _shorter(s: list[SimStation], victim: int, culprit: int, max_dur: float) -> float:
    d = math.floor(max_dur + EPS)
    if d < 1.0 or d >= s[culprit].dur:
        return 0.0
    tried = list(s)
    tried[culprit] = replace(s[culprit], dur=d, post=processing_estimate_s(d))
    after = _losses(tried)
    if (victim, culprit) in after or (culprit, victim) in after:
        return 0.0
    return d


def _sparser(s: list[SimStation], victim: int, culprit: int) -> float:
    for k in range(2):
        tried = list(s)
        tried[culprit] = replace(s[culprit], rep=2.0 * s[culprit].rep,
                                 chirpt=s[culprit].chirpt + k * s[culprit].rep)
        after = _losses(tried)
        if (victim, culprit) not in after and (culprit, victim) not in after:
            return tried[culprit].chirpt
    return -1.0


def _fix(est: list[SimStation], st: list[Station], victim: int, culprit: int,
         max_dur: float) -> str:
    """What would let ``victim`` be recorded too, as the end of a sentence, or empty."""
    c = st[culprit]
    d = _shorter(est, victim, culprit, max_dur)
    grid = _sparser(est, victim, culprit)
    ways = []
    if d > 0.0 and d * c.rate_khz >= 5000.0:
        ways.append("shorten %s to %s s" % (c.name, _fixed(d, 0)))
    if grid >= 0.0:
        ways.append("record %s every other session (period %s s, chirptime %s)"
                    % (c.name, _val(2.0 * c.rep), _val(grid)))
    return " or ".join(ways)


def _collisions(st: list[Station], out: list[Issue]) -> None:
    n = len(st)
    zero = [SimStation(s.rep, s.chirpt, s.dur, 0.0) for s in st]
    est = [SimStation(s.rep, s.chirpt, s.dur, processing_estimate_s(s.dur)) for s in st]
    reported = False
    alone0, alone_e = [0] * n, [0] * n
    for i, s in enumerate(st):
        nominal = len(simulate([replace(zero[i], dur=0.0)], DAY_S, 2.0 * DAY_S))
        alone0[i] = len(simulate([zero[i]], DAY_S, 2.0 * DAY_S))
        alone_e[i] = len(simulate([est[i]], DAY_S, 2.0 * DAY_S))
        if alone0[i] < nominal:
            out.append(Issue(WARNING, s.name, "a %s s session plus 3 s to prepare does "
                                              "not fit a %s s period: %d of %d "
                                              "sessions a day are recorded."
                             % (_val(s.dur), _val(s.rep), alone0[i], nominal)))
            reported = True
        elif alone_e[i] < nominal:
            out.append(Issue(WARNING, s.name, "after a session %s s remain before the "
                                              "next, but processing the capture and "
                                              "preparing take about %s s: %d of %d "
                                              "sessions a day may be recorded."
                             % (_fixed(s.rep - s.dur, 0),
                                _fixed(est[i].post + SOUNDER_LEAD_S, 0), alone_e[i],
                                nominal)))
            reported = True
    definite = _losses(zero)
    risky = _losses(est, [0] * n)
    for (v, c), loss in sorted(definite.items()):
        max_dur = risky[(v, c)].max_dur if (v, c) in risky else loss.max_dur
        advice = _fix(est, st, v, c, max_dur)
        if advice:
            advice = " To record both, %s." % advice
        elif loss.max_dur < 1.0:
            advice = (" Sessions start almost together: one receiver cannot take "
                      "both; keep one of the stations active.")
        else:
            advice = " Keep one of the stations active."
        out.append(Issue(WARNING, st[v].name, "loses %d of %d sessions a day: %s is "
                                              "being recorded then.%s"
                         % (loss.count, alone0[v], st[c].name, advice)))
        reported = True
    for (v, c), loss in sorted(risky.items()):
        if (v, c) in definite and definite[(v, c)].count >= loss.count:
            continue
        advice = _fix(est, st, v, c, loss.max_dur)
        out.append(Issue(WARNING, st[v].name, "may lose up to %d of %d sessions a day: "
                                              "%s's capture ends %s s before it "
                                              "starts, and processing the capture and "
                                              "preparing take about %s s.%s"
                         % (loss.count, alone_e[v], st[c].name,
                            _fixed(max(0.0, loss.min_gap), 0),
                            _fixed(est[c].post + SOUNDER_LEAD_S, 0),
                            " More reliably, %s." % advice if advice else "")))
        reported = True
    if n >= 2 and not reported:
        out.append(Issue(INFO, "", "the sessions of %d stations do not overlap: a day "
                                   "was checked, with 3 s to prepare and time to "
                                   "process each capture" % n))
