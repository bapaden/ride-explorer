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

__all__ = [
    "CyclingFitData",
    "DeviceInfo",
    "Lap",
    "RecordPoint",
    "Session",
    "parse_cycling_fit",
]
