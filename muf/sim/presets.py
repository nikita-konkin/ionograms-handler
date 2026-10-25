"""Ready-made states: the ITU-R F.1487 test channels, and a mode from a law.

Both are scenarios rather than observations, and say so in their
provenance: the F.1487 channels are priors from the recommendation, the law
mode a model whose one coefficient is a prior from February's data.
"""

from __future__ import annotations

import numpy as np

from .state import MODEL, PRIOR, ChannelState, Circuit, Mode, Provenance, Sweep

#: ITU-R F.1487 (05/2000), Annex 3 sections 2-4: two independently fading
#: paths of equal mean power, no frequency shift, magneto-ionic splitting
#: neglected. Values are (differential time delay ms, frequency spread Hz),
#: the spread being 2 sigma of the Gaussian Doppler spectrum.
F1487 = {
    "low-quiet": (0.5, 0.5),
    "low-moderate": (2.0, 1.5),
    "low-disturbed": (6.0, 10.0),
    "mid-quiet": (0.5, 0.1),
    "mid-moderate": (1.0, 0.5),
    "mid-disturbed": (2.0, 1.0),
    "mid-disturbed-nvis": (7.0, 1.0),
    "high-quiet": (1.0, 0.5),
    "high-moderate": (3.0, 10.0),
    "high-disturbed": (7.0, 30.0),
}

_SECTION = {"low": "2", "mid": "3", "high": "4"}

#: The one-parameter dispersion law fitted to the 1F low ray, GDD(f) =
#: b (1/sqrt(1 - f/MUF) - 1), us/MHz: 583 sweeps of cyprus1 -> Yoshkar-Ola,
#: 4-9 February 2026 (sw_channel_modeling/lfs_tools/dispersion_stats.py).
#: One circuit and one winter week; elsewhere it is a prior to be checked.
FEBRUARY_B_US_PER_MHZ = 7.52

#: cyprus1's sweep, as its .lfs headers record it.
CYPRUS1_SWEEP = Sweep(start_mhz=7.5, rate_hz_s=100e3, duration_s=250.0,
                      sample_rate_hz=40e3, adc_rate_hz=25e6, rmin_km=0,
                      rmax_km=5000, repetition_s=300, chirpt=10)


def f1487_provenance(name: str) -> Provenance:
    region, _, condition = name.partition("-")
    return Provenance(PRIOR, f"ITU-R F.1487 Annex 3 sec. {_SECTION[region]}, "
                             f"{region}-latitude {condition.replace('-', ' ')}")


def doppler_prior(name: str) -> tuple[float, Provenance]:
    """A preset's frequency spread and where it is from, for other states."""
    if name not in F1487:
        raise KeyError(f"unknown F.1487 condition {name!r}; one of {', '.join(F1487)}")
    return F1487[name][1], f1487_provenance(name)


def f1487(name: str, *, base_delay_us: float = 0.0, cn0_dbhz: float = 60.0,
          band_mhz: tuple[float, float] = (2.0, 30.0),
          epoch: float = 0.0) -> ChannelState:
    """One of F.1487's test channels as a state.

    Two modes of equal C/N0, flat in delay over ``band_mhz`` -- no
    dispersion, which the recommendation's model does not have -- the second
    later by the differential delay. The absolute C/N0 is arbitrary: the
    channel normalises it away unless asked not to.
    """
    if name not in F1487:
        raise KeyError(f"unknown F.1487 condition {name!r}; one of {', '.join(F1487)}")
    delay_ms, spread = F1487[name]
    origin = f1487_provenance(name)
    knots = np.asarray(band_mhz, dtype=np.float64)
    modes = []
    for i, delay in enumerate((base_delay_us, base_delay_us + 1e3 * delay_ms), start=1):
        modes.append(Mode(
            name=f"path{i}", freq_mhz=knots, delay_us=np.full(2, delay),
            cn0_dbhz=np.full(2, cn0_dbhz), doppler_spread_hz=spread,
            provenance={
                "delay": origin, "doppler_spread": origin, "doppler_shift": origin,
                "fading": Provenance(PRIOR, "Rayleigh, Gaussian Doppler spectrum "
                                            "(ITU-R F.1487 Annex 1)"),
                "cn0": Provenance(MODEL, "equal mean power, absolute level arbitrary"),
                "band": Provenance(MODEL, "flat over the band, no dispersion"),
                "magnetoionic": Provenance(PRIOR, "neglected, as Annex 3 prescribes"),
                "delay_spread": Provenance(PRIOR, "none: one tap per path, as "
                                                  "Annex 1's model has it"),
            }))
    return ChannelState(circuit=Circuit("f1487-tx", "f1487-rx"), epoch=epoch,
                        modes=modes, source=f"ITU-R F.1487 {name}",
                        provenance={"circuit": Provenance(MODEL,
                                                          "none: a test channel")})


def law_delay_us(f_mhz, muf_mhz: float, tau0_us: float,
                 b_us_per_mhz: float = FEBRUARY_B_US_PER_MHZ) -> np.ndarray:
    """tau(f) = tau0 + b MUF (2 (1 - sqrt(1 - x)) - x), x = f/MUF.

    The integral of the GDD law; ``tau0`` is the delay it tends to at low
    frequency.
    """
    x = np.asarray(f_mhz, dtype=np.float64) / muf_mhz
    return tau0_us + b_us_per_mhz * muf_mhz * (2.0 * (1.0 - np.sqrt(1.0 - x)) - x)


def law_gdd(f_mhz, muf_mhz: float,
            b_us_per_mhz: float = FEBRUARY_B_US_PER_MHZ) -> np.ndarray:
    x = np.asarray(f_mhz, dtype=np.float64) / muf_mhz
    return b_us_per_mhz * (1.0 / np.sqrt(1.0 - x) - 1.0)


def law_mode(name: str = "1F", *, muf_mhz: float = 25.0, tau0_us: float = 8900.0,
             b_us_per_mhz: float = FEBRUARY_B_US_PER_MHZ,
             f_lo_mhz: float | None = None, top_fraction: float = 0.995,
             cn0_dbhz: float = 45.0, spread_hz: float = 0.5,
             step_mhz: float = 0.05, hops: int | None = 1) -> Mode:
    """A low-ray mode whose delay follows the dispersion law up to near its MUF.

    The law's GDD diverges at the MUF itself, so the knots stop at
    ``top_fraction`` of it: 0.995 is 99 us/MHz at b = 7.52.
    """
    lo = f_lo_mhz if f_lo_mhz is not None else 0.6 * muf_mhz
    hi = top_fraction * muf_mhz
    knots = np.arange(lo, hi, step_mhz)
    knots = np.append(knots[knots < hi - 1e-9], hi)
    law = Provenance(MODEL, f"dispersion law, MUF {muf_mhz:g} MHz, tau0 {tau0_us:g} us")
    b_origin = (Provenance(PRIOR, "b from 583 sweeps, cyprus1 -> Yoshkar-Ola, "
                                  "4-9 Feb 2026, 1F low ray")
                if b_us_per_mhz == FEBRUARY_B_US_PER_MHZ else
                Provenance(MODEL, f"b = {b_us_per_mhz:g} us/MHz"))
    spread_origin = Provenance(MODEL, f"frequency spread {spread_hz:g} Hz")
    return Mode(name=name, freq_mhz=knots,
                delay_us=law_delay_us(knots, muf_mhz, tau0_us, b_us_per_mhz),
                cn0_dbhz=np.full(knots.size, cn0_dbhz), doppler_spread_hz=spread_hz,
                hops=hops,
                provenance={"delay": law, "band": law, "dispersion_b": b_origin,
                            "cn0": Provenance(MODEL, f"flat {cn0_dbhz:g} dB-Hz"),
                            "doppler_spread": spread_origin,
                            "doppler_shift": Provenance(MODEL, "none"),
                            "fading": Provenance(PRIOR,
                                                 "Rayleigh (ITU-R F.1487 Annex 1)"),
                            "magnetoionic": Provenance(MODEL, "not split"),
                            "delay_spread": Provenance(MODEL, "none: one path")})


def cyprus1_circuit() -> Circuit:
    """cyprus1 -> Yoshkar-Ola at the registry's coordinates."""
    from .. import loader

    stations = loader.resolve_stations(None)
    tx, rx = stations.get("cyprus1"), stations.get("yoshkar-ola")
    return Circuit("cyprus1", "yoshkar-ola",
                   *(tx if tx else (35.0, 34.0)), *(rx if rx else (56.38, 47.53)))


def law_state(*, muf_mhz: float = 25.0, tau0_us: float = 8900.0,
              b_us_per_mhz: float = FEBRUARY_B_US_PER_MHZ, cn0_dbhz: float = 45.0,
              spread_hz: float = 0.5, epoch: float = 1770206410.0,
              sweep: Sweep | None = CYPRUS1_SWEEP) -> ChannelState:
    """One dispersive 1F mode on cyprus1's circuit and sweep: a scenario to sound."""
    mode = law_mode(muf_mhz=muf_mhz, tau0_us=tau0_us, b_us_per_mhz=b_us_per_mhz,
                    cn0_dbhz=cn0_dbhz, spread_hz=spread_hz)
    return ChannelState(circuit=cyprus1_circuit(), epoch=epoch, modes=[mode],
                        sweep=sweep, source=f"law: MUF {muf_mhz:g} MHz",
                        provenance={"circuit": Provenance(MODEL,
                                                          "cyprus1 -> Yoshkar-Ola"),
                                    "sweep": Provenance(MODEL, "cyprus1's sweep")})


def names() -> list[str]:
    return list(F1487)


__all__ = ["CYPRUS1_SWEEP", "F1487", "FEBRUARY_B_US_PER_MHZ", "cyprus1_circuit",
           "doppler_prior", "f1487", "law_delay_us", "law_gdd", "law_mode",
           "law_state", "names"]
