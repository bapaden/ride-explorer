"""Air density estimation utilities.

This module centralizes the standard-atmosphere calculation used to estimate
air density from ride records. The helper mirrors the previous implementation
embedded in :mod:`ride_explorer.coefficient_estimator` to keep parameter
estimation focused on fitting logic.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .derived_metrics import GRAVITY_M_PER_S2
from .fit_parser import RecordPoint

AIR_DENSITY_KG_PER_M3 = 1.225
STANDARD_PRESSURE_PA = 101_325.0
STANDARD_TEMPERATURE_K = 288.15
LAPSE_RATE_K_PER_M = 0.0065
SPECIFIC_GAS_CONSTANT_AIR = 287.05


def estimate_air_density_from_records(
    records: Sequence[RecordPoint],
    *,
    fallback_air_density: float = AIR_DENSITY_KG_PER_M3,
) -> float:
    """Estimate air density using available temperature and elevation samples.

    The calculation applies the ICAO standard atmosphere barometric formula to
    derive pressure from altitude, then combines that pressure with observed
    temperature (if present) to compute density via the ideal gas law. When
    temperature samples are missing, the standard lapse-adjusted temperature is
    used instead.
    """

    altitudes: list[float] = []
    temperatures_c: list[float] = []
    for record in records:
        if record.altitude is not None:
            altitudes.append(float(record.altitude))
        if record.temperature is not None:
            temperatures_c.append(float(record.temperature))

    if not altitudes:
        return fallback_air_density

    altitude_m = float(np.nanmedian(altitudes))
    temperature_k: float | None = (
        float(np.nanmedian(temperatures_c)) + 273.15 if temperatures_c else None
    )

    temperature_term = 1 - LAPSE_RATE_K_PER_M * altitude_m / STANDARD_TEMPERATURE_K
    if temperature_term <= 0:
        return fallback_air_density

    pressure = STANDARD_PRESSURE_PA * temperature_term ** (
        GRAVITY_M_PER_S2 / (SPECIFIC_GAS_CONSTANT_AIR * LAPSE_RATE_K_PER_M)
    )
    if temperature_k is None:
        temperature_k = STANDARD_TEMPERATURE_K - LAPSE_RATE_K_PER_M * altitude_m

    density = pressure / (SPECIFIC_GAS_CONSTANT_AIR * temperature_k)
    if not np.isfinite(density) or density <= 0:
        return fallback_air_density

    return float(density)

