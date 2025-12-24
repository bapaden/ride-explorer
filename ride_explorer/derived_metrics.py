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
    records: Sequence[RecordPoint], attribute: str, *, time_origin: float | None = None
) -> DerivedSeries:
    """Convert a ``RecordPoint`` attribute stream into aligned NumPy arrays.

    Only records that contain both a timestamp and the requested attribute are
    kept; missing entries are skipped entirely. The timestamps are shifted to
    start at zero seconds to improve numerical stability for finite differences,
    using ``time_origin`` when provided or the first valid timestamp otherwise.
    """

    times: list[float] = []
    values: list[float] = []

    first_timestamp: float | None = time_origin
    for record in records:
        ts = record.timestamp
        value = getattr(record, attribute)
        if ts is None or value is None:
            # Drop records that cannot contribute to the derivative calculation.
            continue

        if first_timestamp is None:
            first_timestamp = ts.timestamp()

        times.append(ts.timestamp() - first_timestamp)
        values.append(float(value))

    return DerivedSeries(times=np.asarray(times), values=np.asarray(values))


def compute_climbing_rate(
    records: Sequence[RecordPoint], *, time_origin: float | None = None
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

    Returns
    -------
    DerivedSeries
        ``times``: seconds since the first valid altitude sample.
        ``values``: climbing rate in meters per second for each timestamp.
        If fewer than two valid samples exist, both arrays are empty.
    """

    series = _extract_numeric_series(records, "altitude", time_origin=time_origin)
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
    records: Sequence[RecordPoint], *, time_origin: float | None = None
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
    """

    return {
        "climbing_rate": compute_climbing_rate(records, time_origin=time_origin),
        "acceleration": compute_acceleration(records, time_origin=time_origin),
    }


def compute_mechanical_power(
    records: Sequence[RecordPoint],
    system_mass_kg: float,
    *,
    time_origin: float | None = None,
) -> dict[str, DerivedSeries]:
    """Estimate mechanical power components for acceleration and climbing.

    Power is split into two series instead of a single summed signal:

    * ``climbing_power``: ``m * g * vertical_speed`` to overcome gravity.
    * ``acceleration_power``: ``m * acceleration * speed`` to change kinetic
      energy (speed already in m/s from FIT data).

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
    time_origin:
        Optional absolute timestamp (seconds since epoch) used as the zero
        reference for the returned ``times`` arrays. When omitted, the first
        valid timestamp in each series is used.

    Returns
    -------
    dict[str, DerivedSeries]
        ``climbing_power`` and ``acceleration_power`` series, each aligned to
        the acceleration time grid. Empty arrays are returned for both entries
        if the required derivatives cannot be computed.
    """

    if system_mass_kg <= 0:
        raise ValueError("system_mass_kg must be positive")

    # Acceleration uses the speed stream; the speed samples define the time grid
    # for the final power calculation.
    speed_series = _extract_numeric_series(
        records, "speed", time_origin=time_origin
    )
    accel_series = compute_acceleration(records, time_origin=time_origin)
    climb_series = compute_climbing_rate(records, time_origin=time_origin)

    if accel_series.times.size == 0 or climb_series.times.size == 0:
        empty = DerivedSeries(times=np.array([]), values=np.array([]))
        return {"climbing_power": empty, "acceleration_power": empty}

    accel_times = accel_series.times

    # Keep only acceleration timestamps that lie within the altitude coverage
    # window to avoid extrapolating the climbing derivative.
    in_altitude_range = (accel_times >= climb_series.times[0]) & (
        accel_times <= climb_series.times[-1]
    )
    if not np.any(in_altitude_range):
        empty = DerivedSeries(times=np.array([]), values=np.array([]))
        return {"climbing_power": empty, "acceleration_power": empty}

    aligned_times = accel_times[in_altitude_range]
    aligned_speed = speed_series.values[in_altitude_range]
    aligned_accel = accel_series.values[in_altitude_range]

    # Interpolate the climbing rate onto the acceleration timeline so both
    # components can be evaluated sample-by-sample.
    aligned_climb_rate = np.interp(
        aligned_times, climb_series.times, climb_series.values
    )

    climbing_power = system_mass_kg * GRAVITY_M_PER_S2 * aligned_climb_rate
    acceleration_power = system_mass_kg * aligned_accel * aligned_speed

    return {
        "climbing_power": DerivedSeries(times=aligned_times, values=climbing_power),
        "acceleration_power": DerivedSeries(
            times=aligned_times, values=acceleration_power
        ),
    }
