"""Derived cycling metrics computed from raw FIT records.

This module relies on the structured output from :mod:`ride_explorer.fit_parser`
and uses NumPy's vectorized finite-difference helpers to compute:

* ``climbing_rate`` – vertical velocity in meters per second based on altitude.
* ``acceleration`` – change in speed (m/s²) derived from the speed stream.

No smoothing is applied here so consumers can decide how to post-process the
signals (e.g., moving averages) as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .fit_parser import RecordPoint

GRAVITY_M_PER_S2 = 9.80665
# Standard sea-level air density used for steady-state speed calculations.
STANDARD_AIR_DENSITY_KG_PER_M3 = 1.225


@dataclass(frozen=True)
class DerivedSeries:
    """Container for a derived time series aligned to FIT record timestamps."""

    # Seconds since the first valid timestamp in the series.
    times: np.ndarray
    # Derived values for each timestamp, matching the ``times`` array.
    values: np.ndarray


def _extract_numeric_series(
    records: Sequence[RecordPoint], attribute: str, *, time_origin: float | None = None
) -> DerivedSeries:
    """Convert a ``RecordPoint`` attribute stream into aligned NumPy arrays.

    The parser aligns all streams ahead of time, so this helper expects every
    record to contain a timestamp and the requested attribute. If any values are
    missing, an empty series is returned to signal that the derivative cannot be
    computed reliably.
    """

    if not records:
        return DerivedSeries(times=np.array([]), values=np.array([]))

    times = np.array(
        [record.timestamp.timestamp() for record in records if record.timestamp],
        dtype=float,
    )
    values = np.array(
        [
            np.nan if getattr(record, attribute) is None else float(getattr(record, attribute))
            for record in records
        ],
        dtype=float,
    )

    if times.size != len(records) or np.any(np.isnan(values)):
        return DerivedSeries(times=np.array([]), values=np.array([]))

    start = time_origin if time_origin is not None else times[0]
    return DerivedSeries(times=times - start, values=values)


def _time_shift_series(series: DerivedSeries, shift: float) -> DerivedSeries:
    """Shift a numeric series in time while preserving its sampling grid."""

    if shift == 0 or series.times.size == 0:
        return series

    shifted_times = series.times + float(shift)
    values = np.interp(
        series.times,
        shifted_times,
        series.values,
        left=series.values[0],
        right=series.values[-1],
    )
    return DerivedSeries(times=series.times, values=values)


def steady_state_speed(
    power_w: float,
    system_mass_kg: float,
    gradient_radians: float,
    *,
    eta: float,
    crr: float,
    cda: float,
    air_density: float = STANDARD_AIR_DENSITY_KG_PER_M3,
) -> float:
    """Compute steady-state speed for a given power and road gradient.

    The calculation assumes a balance between rider power (scaled by drivetrain
    efficiency) and resistive forces from rolling resistance, gravity, and
    aerodynamic drag. The resulting cubic equation in speed ``v`` is::

        0.5 * rho * CdA * v^3 + m * g * (Crr + sin(theta)) * v - eta * P = 0

    Parameters
    ----------
    power_w:
        Crank power in watts. Non-positive power yields a speed of ``0``.
    system_mass_kg:
        Combined rider + bike mass in kilograms; must be positive.
    gradient_radians:
        Road gradient in radians (positive for uphill, negative for downhill).
    eta:
        Drivetrain efficiency (0–1]; must be positive.
    crr:
        Rolling resistance coefficient (unitless); must be non-negative.
    cda:
        Aerodynamic drag area in m²; must be non-negative.
    air_density:
        Air density in kg/m³ used for the aerodynamic term. Defaults to ISA sea
        level density (1.225 kg/m³).

    Returns
    -------
    float
        Steady-state speed in meters per second. If the inputs do not permit a
        positive solution, ``0.0`` is returned.
    """

    if system_mass_kg <= 0:
        raise ValueError("system_mass_kg must be positive")
    if eta <= 0:
        raise ValueError("eta must be positive")
    if crr < 0:
        raise ValueError("crr must be non-negative")
    if cda < 0:
        raise ValueError("cda must be non-negative")
    if air_density <= 0:
        raise ValueError("air_density must be positive")

    available_power = eta * power_w
    if available_power <= 0:
        return 0.0

    linear_term = system_mass_kg * GRAVITY_M_PER_S2 * (crr + np.sin(float(gradient_radians)))
    cubic_term = 0.5 * air_density * cda

    if cubic_term == 0:
        return available_power / linear_term if linear_term > 0 else 0.0

    coefficients = [cubic_term, 0.0, linear_term, -available_power]
    roots = np.roots(coefficients)
    positive_real = [float(root.real) for root in roots if abs(root.imag) < 1e-9 and root.real > 0]

    return max(positive_real) if positive_real else 0.0


def compute_climbing_rate(
    records: Sequence[RecordPoint],
    *,
    time_origin: float | None = None,
    elevation_lag_s: float = 0.0,
) -> DerivedSeries:
    """Compute climbing rate (vertical speed) in m/s using finite differences.

    The calculation applies NumPy's ``gradient`` to the altitude stream using the
    recorded timestamps as spacing. ``np.gradient`` performs the following:

    * Central differences for interior points to improve accuracy.
    * Forward and backward differences at the boundaries to avoid shrinking the
      output series.

    Parameters
    ----------
    records:
        Ordered sequence of :class:`ride_explorer.fit_parser.RecordPoint`
        instances as returned by :func:`ride_explorer.fit_parser.parse_cycling_fit`.

    Parameters
    ----------
    records:
        Ordered sequence of :class:`ride_explorer.fit_parser.RecordPoint`
        instances as returned by :func:`ride_explorer.fit_parser.parse_cycling_fit`.
    time_origin:
        Optional absolute timestamp (seconds since epoch) used as the zero
        reference for the returned ``times`` array. When omitted, the first valid
        timestamp in the altitude series is used.
    elevation_lag_s:
        Time shift (seconds) applied to the altitude stream to compensate for
        delays in elevation readings. Positive values move elevation samples
        earlier in time before differencing.

    Returns
    -------
    DerivedSeries
        ``times``: seconds since the first valid altitude sample.
        ``values``: climbing rate in meters per second for each timestamp.
        If fewer than two valid samples exist, both arrays are empty.
    """

    series = _extract_numeric_series(records, "altitude", time_origin=time_origin)
    if elevation_lag_s != 0:
        series = _time_shift_series(series, elevation_lag_s)
    if series.times.size < 2:
        # Derivatives require at least two samples; return empty arrays otherwise.
        return DerivedSeries(times=np.array([]), values=np.array([]))

    climbing_rate = np.gradient(series.values, series.times)
    return DerivedSeries(times=series.times, values=climbing_rate)


def compute_acceleration(
    records: Sequence[RecordPoint], *, time_origin: float | None = None
) -> DerivedSeries:
    """Compute acceleration (m/s²) from speed and time using finite differences.

    This mirrors :func:`compute_climbing_rate` but operates on the ``speed``
    attribute. FIT speed values are already in meters per second, so the
    resulting derivative directly represents acceleration without unit
    conversion.

    Parameters
    ----------
    records:
        Ordered sequence of :class:`ride_explorer.fit_parser.RecordPoint`
        instances as returned by :func:`ride_explorer.fit_parser.parse_cycling_fit`.
    time_origin:
        Optional absolute timestamp (seconds since epoch) used as the zero
        reference for the returned ``times`` array. When omitted, the first valid
        timestamp in the speed series is used.
    """

    series = _extract_numeric_series(records, "speed", time_origin=time_origin)
    if series.times.size < 2:
        return DerivedSeries(times=np.array([]), values=np.array([]))

    acceleration = np.gradient(series.values, series.times)
    return DerivedSeries(times=series.times, values=acceleration)


def compute_all_derived_metrics(
    records: Sequence[RecordPoint],
    *,
    time_origin: float | None = None,
    elevation_lag_s: float = 0.0,
) -> dict[str, DerivedSeries]:
    """Compute all available derived metrics for a given record stream.

    The helper makes it easy for callers to fetch multiple derived quantities in
    one pass without recomputing the base series extraction logic.

    Parameters
    ----------
    records:
        Ordered sequence of :class:`ride_explorer.fit_parser.RecordPoint`.
    time_origin:
        Optional absolute timestamp (seconds since epoch) used as the zero
        reference for the returned ``times`` arrays. When omitted, each series
        uses its own first valid timestamp.
    elevation_lag_s:
        Optional elevation lag (seconds) applied before computing climbing rate.
    """

    return {
        "climbing_rate": compute_climbing_rate(
            records, time_origin=time_origin, elevation_lag_s=elevation_lag_s
        ),
        "acceleration": compute_acceleration(records, time_origin=time_origin),
    }


def compute_mechanical_power(
    records: Sequence[RecordPoint],
    system_mass_kg: float,
    *,
    time_origin: float | None = None,
    elevation_lag_s: float = 0.0,
) -> dict[str, DerivedSeries]:
    """Estimate mechanical power components for acceleration and climbing.

    Power is split into two series instead of a single summed signal:

    * ``climbing_power``: ``m * g * vertical_speed`` to overcome gravity.
    * ``acceleration_power``: ``m * acceleration * speed`` to change kinetic
      energy (speed already in m/s from FIT data).

    Streams are assumed to be time-aligned by :mod:`ride_explorer.fit_parser`, so
    both components are evaluated on the shared record timeline.

    Parameters
    ----------
    records:
        Ordered sequence of :class:`ride_explorer.fit_parser.RecordPoint`
        instances as returned by :func:`ride_explorer.fit_parser.parse_cycling_fit`.
    system_mass_kg:
        Combined rider + bike mass in kilograms.
    time_origin:
        Optional absolute timestamp (seconds since epoch) used as the zero
        reference for the returned ``times`` arrays. When omitted, the first
        valid timestamp in each series is used.
    elevation_lag_s:
        Time shift (seconds) applied to the elevation stream before computing
        climbing rate, useful for compensating delayed barometric sensors.

    Returns
    -------
    dict[str, DerivedSeries]
        ``climbing_power`` and ``acceleration_power`` series, each aligned to
        the acceleration time grid. Empty arrays are returned for both entries
        if the required derivatives cannot be computed.
    """

    if system_mass_kg <= 0:
        raise ValueError("system_mass_kg must be positive")

    speed_series = _extract_numeric_series(
        records, "speed", time_origin=time_origin
    )
    if speed_series.times.size == 0:
        empty = DerivedSeries(times=np.array([]), values=np.array([]))
        return {"climbing_power": empty, "acceleration_power": empty}

    # Acceleration uses the speed stream; the speed samples define the time grid
    # for the final power calculation.
    accel_series = compute_acceleration(records, time_origin=time_origin)
    climb_series = compute_climbing_rate(
        records, time_origin=time_origin, elevation_lag_s=elevation_lag_s
    )

    if accel_series.times.size == 0 or climb_series.times.size == 0:
        empty = DerivedSeries(times=np.array([]), values=np.array([]))
        return {"climbing_power": empty, "acceleration_power": empty}

    if not np.array_equal(accel_series.times, climb_series.times):
        empty = DerivedSeries(times=np.array([]), values=np.array([]))
        return {"climbing_power": empty, "acceleration_power": empty}

    aligned_times = accel_series.times
    aligned_speed = speed_series.values
    aligned_accel = accel_series.values
    aligned_climb_rate = climb_series.values

    climbing_power = system_mass_kg * GRAVITY_M_PER_S2 * aligned_climb_rate
    acceleration_power = system_mass_kg * aligned_accel * aligned_speed

    return {
        "climbing_power": DerivedSeries(times=aligned_times, values=climbing_power),
        "acceleration_power": DerivedSeries(
            times=aligned_times, values=acceleration_power
        ),
    }
