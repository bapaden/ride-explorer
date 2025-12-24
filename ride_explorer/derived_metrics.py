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


@dataclass(frozen=True)
class DerivedSeries:
    """Container for a derived time series aligned to FIT record timestamps."""

    # Seconds since the first valid timestamp in the series.
    times: np.ndarray
    # Derived values for each timestamp, matching the ``times`` array.
    values: np.ndarray


def _extract_numeric_series(
    records: Sequence[RecordPoint], attribute: str
) -> DerivedSeries:
    """Convert a ``RecordPoint`` attribute stream into aligned NumPy arrays.

    Only records that contain both a timestamp and the requested attribute are
    kept; missing entries are skipped entirely. The timestamps are shifted to
    start at zero seconds to improve numerical stability for finite differences.
    """

    times: list[float] = []
    values: list[float] = []

    first_timestamp: float | None = None
    for record in records:
        ts = record.timestamp
        value = getattr(record, attribute)
        if ts is None or value is None:
            # Drop records that cannot contribute to the derivative calculation.
            continue

        first_timestamp = first_timestamp or ts.timestamp()
        times.append(ts.timestamp() - first_timestamp)
        values.append(float(value))

    return DerivedSeries(times=np.asarray(times), values=np.asarray(values))


def compute_climbing_rate(records: Sequence[RecordPoint]) -> DerivedSeries:
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

    Returns
    -------
    DerivedSeries
        ``times``: seconds since the first valid altitude sample.
        ``values``: climbing rate in meters per second for each timestamp.
        If fewer than two valid samples exist, both arrays are empty.
    """

    series = _extract_numeric_series(records, "altitude")
    if series.times.size < 2:
        # Derivatives require at least two samples; return empty arrays otherwise.
        return DerivedSeries(times=np.array([]), values=np.array([]))

    climbing_rate = np.gradient(series.values, series.times)
    return DerivedSeries(times=series.times, values=climbing_rate)


def compute_acceleration(records: Sequence[RecordPoint]) -> DerivedSeries:
    """Compute acceleration (m/s²) from speed and time using finite differences.

    This mirrors :func:`compute_climbing_rate` but operates on the ``speed``
    attribute. FIT speed values are already in meters per second, so the
    resulting derivative directly represents acceleration without unit
    conversion.
    """

    series = _extract_numeric_series(records, "speed")
    if series.times.size < 2:
        return DerivedSeries(times=np.array([]), values=np.array([]))

    acceleration = np.gradient(series.values, series.times)
    return DerivedSeries(times=series.times, values=acceleration)


def compute_all_derived_metrics(
    records: Sequence[RecordPoint],
) -> dict[str, DerivedSeries]:
    """Compute all available derived metrics for a given record stream.

    The helper makes it easy for callers to fetch multiple derived quantities in
    one pass without recomputing the base series extraction logic.
    """

    return {
        "climbing_rate": compute_climbing_rate(records),
        "acceleration": compute_acceleration(records),
    }


def compute_mechanical_power(
    records: Sequence[RecordPoint], system_mass_kg: float
) -> DerivedSeries:
    """Estimate mechanical power needed for acceleration and climbing.

    Power is calculated as the sum of:

    * ``m * g * vertical_speed`` for overcoming gravity while climbing.
    * ``m * acceleration * speed`` for changing the system's kinetic energy.
      The speed term is taken directly from the FIT data so no additional
      interpolation is required for that part of the calculation.

    The climbing component is interpolated onto the acceleration time grid so
    the resulting series stays aligned with the speed samples used for the
    kinetic term. Values outside the altitude coverage window are discarded to
    avoid extrapolating derivatives.

    Parameters
    ----------
    records:
        Ordered sequence of :class:`ride_explorer.fit_parser.RecordPoint`
        instances as returned by :func:`ride_explorer.fit_parser.parse_cycling_fit`.
    system_mass_kg:
        Combined rider + bike mass in kilograms.

    Returns
    -------
    DerivedSeries
        ``times``: seconds since the first valid speed/altitude samples where
        both climbing rate and acceleration are defined.
        ``values``: estimated mechanical power in watts for each timestamp.
        If derivatives cannot be computed (e.g., too few samples), both arrays
        are empty.
    """

    if system_mass_kg <= 0:
        raise ValueError("system_mass_kg must be positive")

    # Acceleration uses the speed stream; the speed samples define the time grid
    # for the final power calculation.
    speed_series = _extract_numeric_series(records, "speed")
    accel_series = compute_acceleration(records)
    climb_series = compute_climbing_rate(records)

    if accel_series.times.size == 0 or climb_series.times.size == 0:
        return DerivedSeries(times=np.array([]), values=np.array([]))

    accel_times = accel_series.times

    # Keep only acceleration timestamps that lie within the altitude coverage
    # window to avoid extrapolating the climbing derivative.
    in_altitude_range = (accel_times >= climb_series.times[0]) & (
        accel_times <= climb_series.times[-1]
    )
    if not np.any(in_altitude_range):
        return DerivedSeries(times=np.array([]), values=np.array([]))

    aligned_times = accel_times[in_altitude_range]
    aligned_speed = speed_series.values[in_altitude_range]
    aligned_accel = accel_series.values[in_altitude_range]

    # Interpolate the climbing rate onto the acceleration timeline so both
    # components can be summed sample-by-sample.
    aligned_climb_rate = np.interp(
        aligned_times, climb_series.times, climb_series.values
    )

    power_from_grade = system_mass_kg * GRAVITY_M_PER_S2 * aligned_climb_rate
    power_from_accel = system_mass_kg * aligned_accel * aligned_speed

    return DerivedSeries(times=aligned_times, values=power_from_grade + power_from_accel)
