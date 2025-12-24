"""Utilities for exploring cycling FIT files.

This package currently exposes a single convenience function for turning a
Garmin cycling FIT file into a structured Python object.
"""

from .fit_parser import (
    CyclingFitData,
    DeviceInfo,
    Lap,
    RecordPoint,
    Session,
    parse_cycling_fit,
)
from .derived_metrics import (
    DerivedSeries,
    compute_acceleration,
    compute_all_derived_metrics,
    compute_climbing_rate,
    compute_mechanical_power,
)

__all__ = [
    "CyclingFitData",
    "DeviceInfo",
    "DerivedSeries",
    "Lap",
    "RecordPoint",
    "Session",
    "compute_acceleration",
    "compute_all_derived_metrics",
    "compute_climbing_rate",
    "compute_mechanical_power",
    "parse_cycling_fit",
]
