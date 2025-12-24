"""FIT file parsing utilities for cycling activities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from fitparse import FitFile


Semicircle = Optional[int]
Degrees = Optional[float]


def _semicircles_to_degrees(value: Semicircle) -> Degrees:
    if value is None:
        return None
    return value * 180 / 2**31


@dataclass(frozen=True)
class DeviceInfo:
    manufacturer: Optional[str]
    product: Optional[str]
    software_version: Optional[float]
    serial_number: Optional[int]
    battery_level: Optional[float]
    battery_status: Optional[int]
    descriptor: Optional[str]
    timestamp: Optional[datetime]


@dataclass(frozen=True)
class RecordPoint:
    timestamp: Optional[datetime]
    position: Optional[Tuple[Degrees, Degrees]]
    altitude: Optional[float]
    distance: Optional[float]
    speed: Optional[float]
    heart_rate: Optional[int]
    cadence: Optional[int]
    temperature: Optional[float]
    power: Optional[int]


@dataclass(frozen=True)
class Lap:
    start_time: Optional[datetime]
    total_elapsed_time: Optional[float]
    total_timer_time: Optional[float]
    total_distance: Optional[float]
    total_calories: Optional[int]
    avg_speed: Optional[float]
    max_speed: Optional[float]
    avg_cadence: Optional[int]
    max_cadence: Optional[int]
    avg_heart_rate: Optional[int]
    max_heart_rate: Optional[int]
    avg_power: Optional[int]
    max_power: Optional[int]


@dataclass(frozen=True)
class Session:
    start_time: Optional[datetime]
    total_elapsed_time: Optional[float]
    total_timer_time: Optional[float]
    total_distance: Optional[float]
    total_calories: Optional[int]
    total_ascent: Optional[int]
    total_descent: Optional[int]
    avg_speed: Optional[float]
    max_speed: Optional[float]
    avg_cadence: Optional[int]
    max_cadence: Optional[int]
    avg_heart_rate: Optional[int]
    max_heart_rate: Optional[int]
    avg_power: Optional[int]
    max_power: Optional[int]
    training_load: Optional[int]


@dataclass(frozen=True)
class CyclingFitData:
    """Structured representation of a cycling FIT activity."""

    source: Path
    devices: Sequence[DeviceInfo]
    sessions: Sequence[Session]
    laps: Sequence[Lap]
    records: Sequence[RecordPoint]


def _safe_get(record, key):
    field = record.get(key)
    if field:
        return field.value
    return None


def _parse_devices(fit: FitFile) -> List[DeviceInfo]:
    devices: List[DeviceInfo] = []
    for message in fit.get_messages("device_info"):
        fields = {field.name: field for field in message}
        devices.append(
            DeviceInfo(
                manufacturer=_safe_get(fields, "manufacturer"),
                product=_safe_get(fields, "product"),
                software_version=_safe_get(fields, "software_version"),
                serial_number=_safe_get(fields, "serial_number"),
                battery_level=_safe_get(fields, "battery_level"),
                battery_status=_safe_get(fields, "battery_status"),
                descriptor=_safe_get(fields, "descriptor"),
                timestamp=_safe_get(fields, "timestamp"),
            )
        )
    return devices


def _parse_records(fit: FitFile) -> List[RecordPoint]:
    records: List[RecordPoint] = []
    for message in fit.get_messages("record"):
        fields = {field.name: field for field in message}
        lat = _safe_get(fields, "position_lat")
        lon = _safe_get(fields, "position_long")
        records.append(
            RecordPoint(
                timestamp=_safe_get(fields, "timestamp"),
                position=(
                    _semicircles_to_degrees(lat),
                    _semicircles_to_degrees(lon),
                )
                if lat is not None and lon is not None
                else None,
                altitude=_safe_get(fields, "altitude"),
                distance=_safe_get(fields, "distance"),
                speed=_safe_get(fields, "speed"),
                heart_rate=_safe_get(fields, "heart_rate"),
                cadence=_safe_get(fields, "cadence"),
                temperature=_safe_get(fields, "temperature"),
                power=_safe_get(fields, "power"),
            )
        )
    return records


def _build_time_grid(records: Sequence[RecordPoint]) -> list[datetime]:
    """Return sorted, de-duplicated timestamps from a record stream."""

    seen: set[float] = set()
    grid: list[datetime] = []

    for record in records:
        ts = record.timestamp
        if ts is None:
            continue
        ts_seconds = ts.timestamp()
        if ts_seconds in seen:
            continue
        seen.add(ts_seconds)
        grid.append(ts)

    return sorted(grid, key=lambda ts: ts.timestamp())


def _interpolate_numeric_stream(
    records: Sequence[RecordPoint],
    attribute: str,
    target_seconds: np.ndarray,
    *,
    round_to_int: bool = False,
) -> np.ndarray:
    """Interpolate a numeric record attribute onto ``target_seconds``."""

    source_times: list[float] = []
    source_values: list[float] = []

    for record in records:
        ts = record.timestamp
        value = getattr(record, attribute)
        if ts is None or value is None:
            continue
        source_times.append(ts.timestamp())
        source_values.append(float(value))

    if not source_times:
        return np.full_like(target_seconds, np.nan, dtype=float)

    times = np.asarray(source_times, dtype=float)
    values = np.asarray(source_values, dtype=float)
    order = np.argsort(times)
    times = times[order]
    values = values[order]
    times, unique_indices = np.unique(times, return_index=True)
    values = values[unique_indices]

    if times.size == 1:
        filled = np.full_like(target_seconds, values[0], dtype=float)
    else:
        filled = np.interp(
            target_seconds,
            times,
            values,
            left=values[0],
            right=values[-1],
        )

    if round_to_int:
        return np.rint(filled)
    return filled


def _interpolate_positions(
    records: Sequence[RecordPoint], target_seconds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate latitude and longitude streams onto ``target_seconds``."""

    source_times: list[float] = []
    lats: list[float] = []
    lons: list[float] = []

    for record in records:
        ts = record.timestamp
        position = record.position
        if ts is None or position is None:
            continue
        lat, lon = position
        if lat is None or lon is None:
            continue

        source_times.append(ts.timestamp())
        lats.append(lat)
        lons.append(lon)

    if not source_times:
        empty = np.full_like(target_seconds, np.nan, dtype=float)
        return empty, empty

    times = np.asarray(source_times, dtype=float)
    lat_values = np.asarray(lats, dtype=float)
    lon_values = np.asarray(lons, dtype=float)
    order = np.argsort(times)
    times = times[order]
    lat_values = lat_values[order]
    lon_values = lon_values[order]
    times, unique_indices = np.unique(times, return_index=True)
    lat_values = lat_values[unique_indices]
    lon_values = lon_values[unique_indices]

    if times.size == 1:
        lat_interp = np.full_like(target_seconds, lat_values[0], dtype=float)
        lon_interp = np.full_like(target_seconds, lon_values[0], dtype=float)
    else:
        lat_interp = np.interp(
            target_seconds, times, lat_values, left=lat_values[0], right=lat_values[-1]
        )
        lon_interp = np.interp(
            target_seconds, times, lon_values, left=lon_values[0], right=lon_values[-1]
        )

    return lat_interp, lon_interp


def _cast_value(value: float, *, to_int: bool = False):
    if np.isnan(value):
        return None
    if to_int:
        return int(round(value))
    return float(value)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Return a centered moving average with edge-value padding."""

    if window <= 1:
        return values

    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    kernel = np.full(window, 1 / window, dtype=float)
    return np.convolve(padded, kernel, mode="valid")


def _align_and_smooth_records(
    records: Sequence[RecordPoint], smoothing_window: int
) -> list[RecordPoint]:
    """Resample all record fields onto a common timestamp grid and smooth them.

    The returned records contain a value for every field at each timestamp,
    using interpolation (with edge values held constant) for attributes that do
    not report on every FIT record. Streams that are entirely missing remain
    unset (``None``) across all records.
    """

    time_grid = _build_time_grid(records)
    if not time_grid:
        return []

    target_seconds = np.asarray(
        [ts.timestamp() for ts in time_grid], dtype=float
    )

    altitude = _moving_average(
        _interpolate_numeric_stream(records, "altitude", target_seconds),
        smoothing_window,
    )
    distance = _moving_average(
        _interpolate_numeric_stream(records, "distance", target_seconds),
        smoothing_window,
    )
    speed = _moving_average(
        _interpolate_numeric_stream(records, "speed", target_seconds),
        smoothing_window,
    )
    heart_rate = _moving_average(
        _interpolate_numeric_stream(
            records, "heart_rate", target_seconds, round_to_int=True
        ),
        smoothing_window,
    )
    cadence = _moving_average(
        _interpolate_numeric_stream(
            records, "cadence", target_seconds, round_to_int=True
        ),
        smoothing_window,
    )
    temperature = _moving_average(
        _interpolate_numeric_stream(records, "temperature", target_seconds),
        smoothing_window,
    )
    power = _moving_average(
        _interpolate_numeric_stream(
            records, "power", target_seconds, round_to_int=True
        ),
        smoothing_window,
    )
    latitudes, longitudes = _interpolate_positions(records, target_seconds)

    aligned: list[RecordPoint] = []
    for idx, timestamp in enumerate(time_grid):
        lat = latitudes[idx]
        lon = longitudes[idx]
        position = (
            (lat, lon) if not np.isnan(lat) and not np.isnan(lon) else None
        )

        aligned.append(
            RecordPoint(
                timestamp=timestamp,
                position=position,
                altitude=_cast_value(altitude[idx]),
                distance=_cast_value(distance[idx]),
                speed=_cast_value(speed[idx]),
                heart_rate=_cast_value(heart_rate[idx], to_int=True),
                cadence=_cast_value(cadence[idx], to_int=True),
                temperature=_cast_value(temperature[idx]),
                power=_cast_value(power[idx], to_int=True),
            )
        )

    return aligned


def _parse_laps(fit: FitFile) -> List[Lap]:
    laps: List[Lap] = []
    for message in fit.get_messages("lap"):
        fields = {field.name: field for field in message}
        laps.append(
            Lap(
                start_time=_safe_get(fields, "start_time"),
                total_elapsed_time=_safe_get(fields, "total_elapsed_time"),
                total_timer_time=_safe_get(fields, "total_timer_time"),
                total_distance=_safe_get(fields, "total_distance"),
                total_calories=_safe_get(fields, "total_calories"),
                avg_speed=_safe_get(fields, "avg_speed"),
                max_speed=_safe_get(fields, "max_speed"),
                avg_cadence=_safe_get(fields, "avg_cadence"),
                max_cadence=_safe_get(fields, "max_cadence"),
                avg_heart_rate=_safe_get(fields, "avg_heart_rate"),
                max_heart_rate=_safe_get(fields, "max_heart_rate"),
                avg_power=_safe_get(fields, "avg_power"),
                max_power=_safe_get(fields, "max_power"),
            )
        )
    return laps


def _parse_sessions(fit: FitFile) -> List[Session]:
    sessions: List[Session] = []
    for message in fit.get_messages("session"):
        fields = {field.name: field for field in message}
        sessions.append(
            Session(
                start_time=_safe_get(fields, "start_time"),
                total_elapsed_time=_safe_get(fields, "total_elapsed_time"),
                total_timer_time=_safe_get(fields, "total_timer_time"),
                total_distance=_safe_get(fields, "total_distance"),
                total_calories=_safe_get(fields, "total_calories"),
                total_ascent=_safe_get(fields, "total_ascent"),
                total_descent=_safe_get(fields, "total_descent"),
                avg_speed=_safe_get(fields, "avg_speed"),
                max_speed=_safe_get(fields, "max_speed"),
                avg_cadence=_safe_get(fields, "avg_cadence"),
                max_cadence=_safe_get(fields, "max_cadence"),
                avg_heart_rate=_safe_get(fields, "avg_heart_rate"),
                max_heart_rate=_safe_get(fields, "max_heart_rate"),
                avg_power=_safe_get(fields, "avg_power"),
                max_power=_safe_get(fields, "max_power"),
                training_load=_safe_get(fields, "training_load"),
            )
        )
    return sessions


def parse_cycling_fit(
    path: Path | str, *, smoothing_window: int = 1
) -> CyclingFitData:
    """Parse a Garmin cycling FIT file into a structured, aligned object.

    All numeric record streams are resampled onto the shared timestamp grid
    emitted by the FIT ``record`` messages, then smoothed with a centered
    moving average. Missing samples are filled via linear interpolation with
    edge values held constant so downstream consumers can assume fields are
    present and time aligned.
    """

    fit_file = FitFile(str(path))
    fit_file.parse()

    if smoothing_window <= 0:
        raise ValueError("smoothing_window must be positive")

    devices = _parse_devices(fit_file)
    records = _parse_records(fit_file)
    laps = _parse_laps(fit_file)
    sessions = _parse_sessions(fit_file)

    return CyclingFitData(
        source=Path(path),
        devices=devices,
        sessions=sessions,
        laps=laps,
        records=_align_and_smooth_records(records, smoothing_window),
    )
