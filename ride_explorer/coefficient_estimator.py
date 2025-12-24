"""Estimate drivetrain efficiency and resistive coefficients from ride data.

The module exposes two public entry points:

* :func:`prepare_power_balance_data` converts FIT record streams into aligned
  NumPy arrays for power-balance fitting (crank power, gravitational power,
  rolling and aerodynamic terms, optional cadence for weighting).
* :func:`estimate_coefficients_from_records` performs an iteratively reweighted
  fit of drivetrain efficiency, rolling resistance coefficient (Crr), and CdA
  using configurable loss functions (L2, Huber, Cauchy, Tukey).

The residual model is::

    eta * crank_power + gravity + Crr * m * g * speed + 0.5 * rho * CdA * speed^3 = 0

Weights supplied by callers are applied to each residual block prior to the
robust loss. Huber, Cauchy, and Tukey losses are implemented via IRLS so no
extra dependencies beyond NumPy are required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .air_density import AIR_DENSITY_KG_PER_M3, estimate_air_density_from_records
from .derived_metrics import (
    GRAVITY_M_PER_S2,
    compute_acceleration,
    compute_climbing_rate,
)
from .fit_parser import RecordPoint

LossName = Literal["l2", "huber", "cauchy", "tukey"]


@dataclass(frozen=True)
class PowerBalanceData:
    """Aligned data required to evaluate the power-balance residuals."""

    timestamps: np.ndarray
    crank_power: np.ndarray
    rolling_term: np.ndarray
    aero_term: np.ndarray
    gravity_power: np.ndarray
    acceleration_power: np.ndarray
    cadence: np.ndarray | None

    @property
    def sample_count(self) -> int:
        return int(self.crank_power.size)


def _first_timestamp(records: Sequence[RecordPoint]) -> float:
    for record in records:
        if record.timestamp is not None:
            return record.timestamp.timestamp()
    raise ValueError("No timestamps available in records")


def _extract_numeric_series(
    records: Sequence[RecordPoint], attribute: str, time_origin: float
) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    values: list[float] = []

    for record in records:
        ts = record.timestamp
        value = getattr(record, attribute)
        if ts is None or value is None:
            continue

        times.append(ts.timestamp() - time_origin)
        values.append(float(value))

    return np.asarray(times), np.asarray(values)


def _interpolate_series(
    source_times: np.ndarray, source_values: np.ndarray, target_times: np.ndarray
) -> np.ndarray:
    if source_times.size == 0:
        return np.full_like(target_times, np.nan, dtype=float)

    in_range = (target_times >= source_times[0]) & (target_times <= source_times[-1])
    if not np.any(in_range):
        return np.full_like(target_times, np.nan, dtype=float)

    resampled = np.full_like(target_times, np.nan, dtype=float)
    resampled[in_range] = np.interp(target_times[in_range], source_times, source_values)
    return resampled


def prepare_power_balance_data(
    records: Sequence[RecordPoint],
    system_mass_kg: float,
    air_density: float = AIR_DENSITY_KG_PER_M3,
) -> PowerBalanceData:
    """Convert raw FIT records into aligned power-balance inputs.

    Parameters
    ----------
    records:
        Sequence of FIT record points parsed by :mod:`ride_explorer.fit_parser`.
    system_mass_kg:
        Combined mass of rider and bike in kilograms; must be positive.
    air_density:
        Air density in kg/m³ used for aerodynamic drag. Defaults to ISA sea
        level density (1.225).

    Returns
    -------
    PowerBalanceData
        Arrays for crank power, gravitational power, rolling and aerodynamic
        terms aligned to the speed samples. Cadence is included when available
        for caller-provided weighting strategies.
    """

    if system_mass_kg <= 0:
        raise ValueError("system_mass_kg must be positive")

    time_origin = _first_timestamp(records)

    speed_times, speed_values = _extract_numeric_series(
        records, "speed", time_origin
    )
    power_times, power_values = _extract_numeric_series(
        records, "power", time_origin
    )
    cadence_times, cadence_values = _extract_numeric_series(
        records, "cadence", time_origin
    )

    if speed_times.size == 0 or power_times.size == 0:
        raise ValueError("Speed and power streams are required for coefficient fitting")

    climb_series = compute_climbing_rate(records, time_origin=time_origin)
    acceleration_series = compute_acceleration(records, time_origin=time_origin)

    power_resampled = _interpolate_series(power_times, power_values, speed_times)
    cadence_resampled = _interpolate_series(cadence_times, cadence_values, speed_times)
    climb_rate_resampled = _interpolate_series(
        climb_series.times, climb_series.values, speed_times
    )
    acceleration_resampled = _interpolate_series(
        acceleration_series.times, acceleration_series.values, speed_times
    )

    rolling_term = system_mass_kg * GRAVITY_M_PER_S2 * speed_values
    aero_term = 0.5 * air_density * np.power(speed_values, 3)
    gravity_power = system_mass_kg * GRAVITY_M_PER_S2 * np.nan_to_num(
        climb_rate_resampled, nan=0.0
    )
    acceleration_power = system_mass_kg * np.nan_to_num(
        acceleration_resampled, nan=0.0
    ) * speed_values

    mask = np.isfinite(power_resampled) & np.isfinite(rolling_term) & np.isfinite(
        aero_term
    )
    if not np.any(mask):
        raise ValueError("Insufficient overlapping data to build power-balance samples")

    cadence_aligned = None
    if np.any(np.isfinite(cadence_resampled)):
        cadence_aligned = cadence_resampled
        mask &= np.isfinite(cadence_resampled) | np.isnan(cadence_resampled)

    timestamps = speed_times[mask]
    crank_power = power_resampled[mask]
    rolling = rolling_term[mask]
    aero = aero_term[mask]
    gravity = gravity_power[mask]
    acceleration_p = acceleration_power[mask]
    cadence = cadence_aligned[mask] if cadence_aligned is not None else None

    return PowerBalanceData(
        timestamps=timestamps,
        crank_power=crank_power,
        rolling_term=rolling,
        aero_term=aero,
        gravity_power=gravity,
        acceleration_power=acceleration_p,
        cadence=cadence,
    )


def _robust_weights(residuals: np.ndarray, loss: LossName) -> np.ndarray:
    scaled = np.abs(residuals)

    if loss == "l2":
        return np.ones_like(residuals)

    if loss == "huber":
        weights = np.ones_like(residuals)
        mask = scaled > 1
        weights[mask] = 1 / scaled[mask]
        return weights

    if loss == "cauchy":
        return 1 / (1 + np.square(scaled))

    if loss == "tukey":
        weights = np.zeros_like(residuals)
        mask = scaled < 1
        weights[mask] = np.square(1 - np.square(scaled[mask]))
        return weights

    raise ValueError(f"Unsupported loss '{loss}'")


def _weighted_least_squares(
    design: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    sqrt_w = np.sqrt(weights)
    design_w = design * sqrt_w[:, None]
    target_w = target * sqrt_w
    solution, _, _, _ = np.linalg.lstsq(design_w, target_w, rcond=None)
    return solution


def _iteratively_reweighted_fit(
    design: np.ndarray,
    target: np.ndarray,
    base_weights: np.ndarray,
    loss: LossName,
    loss_scale: float,
    max_iterations: int,
    tolerance: float,
) -> np.ndarray:
    weights = base_weights
    params = _weighted_least_squares(design, target, weights)

    for _ in range(max_iterations):
        residuals = design @ params - target
        scaled_residuals = residuals / max(loss_scale, np.finfo(float).eps)
        robust = _robust_weights(scaled_residuals, loss)
        new_weights = base_weights * robust

        if np.allclose(new_weights, weights, rtol=tolerance, atol=tolerance):
            break

        weights = new_weights
        params = _weighted_least_squares(design, target, weights)

    return params


def fit_power_balance_parameters(
    data: PowerBalanceData,
    weights: np.ndarray | None = None,
    *,
    loss: LossName = "tukey",
    loss_scale: float = 1.0,
    include_drivetrain_efficiency: bool = True,
    fixed_efficiency: float = 1.0,
    max_iterations: int = 25,
    tolerance: float = 1e-6,
) -> tuple[float, float, float]:
    """Estimate drivetrain efficiency, rolling resistance, and CdA.

    Parameters
    ----------
    data:
        Aligned power-balance inputs from :func:`prepare_power_balance_data`.
    weights:
        Optional per-sample weights. If provided, must match the number of
        samples in ``data``.
    loss:
        Robust loss function to apply (``"l2"``, ``"huber"``, ``"cauchy"``,
        or ``"tukey"``).
    loss_scale:
        Scaling factor applied to residuals before computing the robust weight.
    include_drivetrain_efficiency:
        Whether to fit drivetrain efficiency (``eta``). If ``False``,
        ``fixed_efficiency`` is used instead.
    fixed_efficiency:
        Drivetrain efficiency to hold constant when ``include_drivetrain_efficiency``
        is ``False``.
    max_iterations:
        Maximum IRLS iterations for robust losses.
    tolerance:
        Convergence tolerance for weight updates.

    Returns
    -------
    tuple[float, float, float]
        Estimated ``(eta, Crr, CdA)``.
    """

    if data.sample_count == 0:
        raise ValueError("PowerBalanceData must contain at least one sample")

    base_weights = np.ones(data.sample_count, dtype=float)
    if weights is not None:
        if weights.shape != base_weights.shape:
            raise ValueError("weights must match the number of samples in data")
        base_weights = np.asarray(weights, dtype=float)

    if include_drivetrain_efficiency:
        design = np.column_stack(
            [data.crank_power, data.rolling_term, data.aero_term]
        )
        target = -(data.gravity_power + data.acceleration_power)
    else:
        design = np.column_stack([data.rolling_term, data.aero_term])
        target = -(fixed_efficiency * data.crank_power + data.gravity_power + data.acceleration_power)

    params = _iteratively_reweighted_fit(
        design=design,
        target=target,
        base_weights=base_weights,
        loss=loss,
        loss_scale=loss_scale,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )

    if include_drivetrain_efficiency:
        eta, crr, cda = params.tolist()
    else:
        eta = fixed_efficiency
        crr, cda = params.tolist()

    return float(eta), float(crr), float(cda)


def estimate_coefficients_from_records(
    records: Sequence[RecordPoint],
    system_mass_kg: float,
    *,
    air_density: float = AIR_DENSITY_KG_PER_M3,
    use_record_air_density: bool = False,
    cadence_weight_threshold: int | None = 30,
    weights: np.ndarray | None = None,
    loss: LossName = "tukey",
    loss_scale: float = 1.0,
    include_drivetrain_efficiency: bool = True,
    fixed_efficiency: float = 1.0,
    max_iterations: int = 25,
    tolerance: float = 1e-6,
) -> tuple[float, float, float]:
    """Driver helper to fit coefficients directly from FIT records.

    Cadence-weighting support is built in: when cadence is available, samples
    with cadence below ``cadence_weight_threshold`` receive zero weight. Caller
    weights (``weights``) are applied multiplicatively after cadence gating.
    Set ``cadence_weight_threshold`` to ``None`` to disable cadence-based
    weighting. Setting ``use_record_air_density`` to ``True`` automatically
    estimates air density from temperature and altitude streams when available.
    """

    if use_record_air_density:
        air_density = estimate_air_density_from_records(
            records, fallback_air_density=air_density
        )

    data = prepare_power_balance_data(
        records=records,
        system_mass_kg=system_mass_kg,
        air_density=air_density,
    )

    base_weights = np.ones(data.sample_count, dtype=float)
    if data.cadence is not None and cadence_weight_threshold is not None:
        cadence_mask = data.cadence >= cadence_weight_threshold
        base_weights = base_weights * cadence_mask.astype(float)

    if weights is not None:
        if weights.shape != base_weights.shape:
            raise ValueError("weights must match the number of samples in data")
        base_weights *= weights

    return fit_power_balance_parameters(
        data=data,
        weights=base_weights,
        loss=loss,
        loss_scale=loss_scale,
        include_drivetrain_efficiency=include_drivetrain_efficiency,
        fixed_efficiency=fixed_efficiency,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
