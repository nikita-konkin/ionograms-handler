"""A quasi-parabolic layer, the rays it returns, and a layer fitted to echoes."""

from __future__ import annotations

import math
import random

import numpy as np
from scipy import integrate, optimize

from muf.layer import Layer, fit

EARTH_KM = 6371.0


def _layer() -> Layer:
    return Layer(random.uniform(6.0, 12.0), random.uniform(240.0, 400.0),
                 random.uniform(50.0, 150.0))


def _integral(layer: Layer, f: float, elevation: float) -> tuple[float, float]:
    """Ground range and group path by quadrature along the ray, from the profile."""
    rm = EARTH_KM + layer.hm_km
    rb = rm - layer.ym_km
    a = EARTH_KM * math.cos(elevation)

    def x(r):
        fn2 = 0.0 if r < rb else layer.fc_mhz ** 2 * (
            1.0 - ((r - rm) / layer.ym_km) ** 2 * (rb / r) ** 2)
        return r * r * (1.0 - fn2 / f ** 2) - a * a

    grid = np.linspace(rb + 1e-9, rm, 4001)
    first = int(np.argmax([x(r) <= 0.0 for r in grid]))
    top = optimize.brentq(x, grid[first - 1], grid[first])

    def ground(u):
        r = top - u * u
        return 2.0 * u * a / (r * math.sqrt(max(x(r), 1e-300)))

    def group(u):
        r = top - u * u
        return 2.0 * u * r / math.sqrt(max(x(r), 1e-300))

    span = math.sqrt(top - EARTH_KM)
    edge = math.sqrt(top - rb)
    d = sum(integrate.quad(ground, lo, hi, limit=400)[0]
            for lo, hi in ((0.0, edge), (edge, span)))
    p = sum(integrate.quad(group, lo, hi, limit=400)[0]
            for lo, hi in ((0.0, edge), (edge, span)))
    return 2.0 * EARTH_KM * d, 2.0 * p


def _returning(layer: Layer) -> tuple[float, float]:
    while True:
        f = layer.fc_mhz * random.uniform(1.2, 3.5)
        elevation = math.radians(random.uniform(2.0, 30.0))
        ground, _ = layer.ray(f, elevation)
        if np.isfinite(ground):
            return f, elevation


def test_a_ray_reaches_as_far_as_the_integral_along_it_says():
    layer = _layer()
    f, elevation = _returning(layer)
    closed = float(layer.ray(f, elevation)[0])
    assert abs(closed - _integral(layer, f, elevation)[0]) < 1e-3, \
        "a ray's closed-form ground range did not match the integral along it"


def test_a_low_ray_turning_below_the_peak_reaches_as_far_as_the_integral_says():
    layer = Layer(10.229974426112786, 288.310065061253, 143.92214396962638)
    f, elevation = 35.472714260844796, math.radians(2.0293613509078345)
    closed = float(layer.ray(f, elevation)[0])
    assert abs(closed - _integral(layer, f, elevation)[0]) < 1e-3, \
        "a low ray's ground range did not match the integral along it"


def test_a_ray_is_as_long_as_the_integral_along_it_says():
    layer = _layer()
    f, elevation = _returning(layer)
    closed = float(layer.ray(f, elevation)[1])
    assert abs(closed - _integral(layer, f, elevation)[1]) < 1e-3, \
        "a ray's closed-form group path did not match the integral along it"


def test_a_vertical_ray_above_the_critical_frequency_does_not_return():
    layer = _layer()
    f = layer.fc_mhz * random.uniform(1.01, 2.0)
    assert np.isnan(layer.ray(f, math.pi / 2)[0]), \
        "a vertical ray above the critical frequency came back"


def test_every_ray_of_a_trace_lands_at_its_ground_range():
    layer = _layer()
    ground = random.uniform(1800.0, 3200.0)
    f, _, elevation = layer.trace(ground)
    assert np.max(np.abs(layer.ray(f, elevation)[0] - ground)) < 0.1, \
        "a trace's ray did not land at the trace's ground range"


def test_a_trace_high_ray_arrives_later_than_its_low_ray():
    layer = _layer()
    f, group, _ = layer.trace(random.uniform(1800.0, 3200.0))
    nose = int(np.argmax(f))
    low = np.interp(f[nose + 1:], f[:nose + 1], group[:nose + 1])
    assert np.all(group[nose + 1:] > low), \
        "a trace's high ray arrived before its low ray at the same frequency"


def _points(layer: Layer, ground: float, offset: float):
    f, group, _ = layer.trace(ground)
    nose = int(np.argmax(f))
    muf = float(f[nose])
    low = np.arange(0.5 * muf, muf, 0.04)
    high = np.arange(0.8 * muf, muf, 0.04)
    rays = [(low, np.interp(low, f[:nose + 1], group[:nose + 1])),
            (high, np.interp(high, f[nose:][::-1], group[nose:][::-1]))]
    start = random.uniform(0.3, 0.4) * muf
    for side in (-1.0, 1.0):
        band = np.arange(start, 0.5 * muf, 0.04)
        level = rays[0][1][0] + side * random.uniform(25.0, 60.0)
        rays.append((band, np.full(band.size, level)))
    freq = np.concatenate([r[0] for r in rays])
    vrange = (np.concatenate([r[1] for r in rays]) + offset
              + np.random.normal(0, 4.0, freq.size))
    return freq, vrange, np.random.uniform(15.0, 40.0, freq.size), muf


def test_a_fitted_layer_lays_its_low_ray_on_the_echo_of_the_layer_that_made_it():
    layer = _layer()
    ground = random.uniform(2200.0, 3000.0)
    offset = random.uniform(-100.0, 200.0)
    freq, vrange, weight, muf = _points(layer, ground, offset)
    found = fit(freq, vrange, weight, ground, muf + random.uniform(-0.2, 0.2))
    f, group, _ = layer.trace(ground)
    nose = int(np.argmax(f))
    probe = np.linspace(0.5, 0.95, 20) * muf
    truth = np.interp(probe, f[:nose + 1], group[:nose + 1]) + offset
    assert np.max(np.abs(np.interp(probe, *found.low()) - truth)) < 10.0, \
        "a fitted layer's low ray strayed off the echo it was fitted to"
