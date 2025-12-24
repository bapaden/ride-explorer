"""Helpers for fetching elevation data from online APIs.

The functions here intentionally avoid using the altitude stored in the FIT
file. Instead, they sample coordinates along the ride at fixed spatial
intervals and query Open-Meteo's elevation API. The samples are then ready to
be interpolated back onto the FIT timestamp grid by callers in
``derived_metrics``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import requests

from .fit_parser import RecordPoint

EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class ElevationSample:
    """Coordinate and timestamp for an elevation lookup."""

    timestamp: float
    latitude: float
    longitude: float


def latlon_to_local_meters(
    reference_lat: float, reference_lon: float, target_lat: float, target_lon: float
) -> tuple[float, float]:
    """Return north/east displacements in meters between two coordinates.

    Garmin FIT messages store latitude and longitude in degrees using the
    (latitude, longitude) ordering. Open-Meteo expects the same convention, so
    we keep that ordering throughout the displacement calculation.
    """

    ref_lat_rad = math.radians(reference_lat)
    ref_lon_rad = math.radians(reference_lon)
    tgt_lat_rad = math.radians(target_lat)
    tgt_lon_rad = math.radians(target_lon)

    delta_lat = tgt_lat_rad - ref_lat_rad
    delta_lon = tgt_lon_rad - ref_lon_rad

    north_m = EARTH_RADIUS_M * delta_lat
    east_m = EARTH_RADIUS_M * math.cos((ref_lat_rad + tgt_lat_rad) / 2) * delta_lon
    return north_m, east_m


def _great_circle_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    north, east = latlon_to_local_meters(lat1, lon1, lat2, lon2)
    return math.hypot(north, east)


def greedy_sample_records(
    records: Sequence[RecordPoint], spacing_meters: float = 25.0
) -> list[ElevationSample]:
    """Return ride coordinates spaced approximately ``spacing_meters`` apart."""

    if spacing_meters <= 0:
        raise ValueError("spacing_meters must be positive")

    samples: list[ElevationSample] = []
    accumulated_distance = 0.0

    last_position: tuple[float, float] | None = None
    last_timestamp: float | None = None

    for record in records:
        if record.timestamp is None or record.position is None:
            continue

        lat, lon = record.position
        if lat is None or lon is None:
            continue

        timestamp = record.timestamp.timestamp()
        if last_position is None:
            samples.append(ElevationSample(timestamp=timestamp, latitude=lat, longitude=lon))
            last_position = (lat, lon)
            last_timestamp = timestamp
            continue

        segment = _great_circle_distance(last_position[0], last_position[1], lat, lon)
        accumulated_distance += segment
        last_position = (lat, lon)
        last_timestamp = timestamp

        if accumulated_distance >= spacing_meters:
            samples.append(ElevationSample(timestamp=timestamp, latitude=lat, longitude=lon))
            accumulated_distance = 0.0

    if last_position and samples and last_timestamp != samples[-1].timestamp:
        samples.append(
            ElevationSample(timestamp=last_timestamp, latitude=last_position[0], longitude=last_position[1])
        )

    return samples


def fetch_open_meteo_elevations(
    coordinates: Iterable[ElevationSample],
    *,
    session: requests.Session | None = None,
    timeout: float = 10.0,
) -> list[float]:
    """Fetch elevations (meters above sea level) from Open-Meteo."""

    coords = list(coordinates)
    if not coords:
        return []

    latitudes = ",".join(f"{coord.latitude:.6f}" for coord in coords)
    longitudes = ",".join(f"{coord.longitude:.6f}" for coord in coords)
    params = {"latitude": latitudes, "longitude": longitudes}

    http_get = session.get if session is not None else requests.get
    response = http_get("https://api.open-meteo.com/v1/elevation", params=params, timeout=timeout)
    response.raise_for_status()

    payload = response.json()
    elevations = payload.get("elevation")
    if elevations is None or len(elevations) != len(coords):
        raise ValueError("Open-Meteo elevation response did not match requested coordinates")

    return [float(value) for value in elevations]


def sample_elevations_along_track(
    records: Sequence[RecordPoint],
    *,
    spacing_meters: float = 25.0,
    session: requests.Session | None = None,
    timeout: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return timestamped elevation samples along a ride."""

    samples = greedy_sample_records(records, spacing_meters=spacing_meters)
    if not samples:
        empty = np.array([], dtype=float)
        return empty, empty

    elevations = fetch_open_meteo_elevations(samples, session=session, timeout=timeout)
    sample_times = np.fromiter((sample.timestamp for sample in samples), dtype=float)
    sample_elevations = np.asarray(elevations, dtype=float)
    return sample_times, sample_elevations
