"""FIT file parsing utilities for cycling activities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

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


def parse_cycling_fit(path: Path | str) -> CyclingFitData:
    """Parse a Garmin cycling FIT file into a structured object."""

    fit_file = FitFile(str(path))
    fit_file.parse()

    devices = _parse_devices(fit_file)
    records = _parse_records(fit_file)
    laps = _parse_laps(fit_file)
    sessions = _parse_sessions(fit_file)

    return CyclingFitData(
        source=Path(path),
        devices=devices,
        sessions=sessions,
        laps=laps,
        records=records,
    )
