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
from .coefficient_estimator import (
    AIR_DENSITY_KG_PER_M3,
    PowerBalanceData,
    estimate_coefficients_from_records,
    estimate_air_density_from_records,
    fit_power_balance_parameters,
    prepare_power_balance_data,
)

__all__ = [
    "CyclingFitData",
    "DeviceInfo",
    "PowerBalanceData",
    "DerivedSeries",
    "Lap",
    "RecordPoint",
    "Session",
    "AIR_DENSITY_KG_PER_M3",
    "compute_acceleration",
    "compute_all_derived_metrics",
    "compute_climbing_rate",
    "compute_mechanical_power",
    "estimate_coefficients_from_records",
    "estimate_air_density_from_records",
    "fit_power_balance_parameters",
    "prepare_power_balance_data",
    "parse_cycling_fit",
]
