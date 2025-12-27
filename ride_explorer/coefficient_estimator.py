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

# Retained for backward compatibility of type hints; only L2 is used currently.
LossName = Literal["l2"]


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
    lap_index: np.ndarray | None

    @property
    def sample_count(self) -> int:
        return int(self.crank_power.size)


@dataclass(frozen=True)
class PowerBalanceEstimation:
    """Full estimation result including weights and residuals."""

    data: PowerBalanceData
    weights: np.ndarray
    residuals: np.ndarray
    eta: float
    crr: float
    cda: float
    elevation_lag_s: float
    stats: dict[str, float]


def _timestamp_offsets(records: Sequence[RecordPoint]) -> np.ndarray:
    timestamps = np.array(
        [record.timestamp.timestamp() for record in records if record.timestamp],
        dtype=float,
    )
    if timestamps.size != len(records):
        raise ValueError("All records must contain timestamps after parsing")
    return timestamps - timestamps[0]


def _aligned_attribute(records: Sequence[RecordPoint], attribute: str) -> np.ndarray:
    return np.asarray(
        [
            np.nan if getattr(record, attribute) is None else float(getattr(record, attribute))
            for record in records
        ],
        dtype=float,
    )


def _time_shift_attribute(
    records: Sequence[RecordPoint], attribute: str, shift: float
) -> np.ndarray:
    """Shift a record attribute in time using linear interpolation.

    The helper treats the input ``records`` as samples of ``attribute`` taken at
    their associated timestamps. The samples are shifted by ``shift`` seconds
    (positive values move the attribute earlier in time) and interpolated back
    onto the original record timeline. When no finite samples are available, the
    returned array is filled with ``NaN``.
    """

    if not records:
        return np.array([], dtype=float)

    if shift == 0:
        return _aligned_attribute(records, attribute)

    timestamps = _timestamp_offsets(records)
    values = _aligned_attribute(records, attribute)

    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return values

    shifted_times = timestamps + float(shift)
    finite_times = shifted_times[finite_mask]
    finite_values = values[finite_mask]
    order = np.argsort(finite_times)
    ordered_times = finite_times[order]
    ordered_values = finite_values[order]

    return np.interp(
        timestamps,
        ordered_times,
        ordered_values,
        left=ordered_values[0],
        right=ordered_values[-1],
    )


def prepare_power_balance_data(
    records: Sequence[RecordPoint],
    system_mass_kg: float,
    air_density: float = AIR_DENSITY_KG_PER_M3,
    *,
    elevation_lag_s: float = 0.0,
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
    elevation_lag_s:
        Time shift (seconds) applied to the elevation stream to compensate for
        delayed altitude readings. Positive values move elevation samples
        earlier in time.

    Returns
    -------
    PowerBalanceData
        Arrays for crank power, gravitational power, rolling and aerodynamic
        terms aligned to the speed samples. Cadence is included when available
        for caller-provided weighting strategies.
    """

    if system_mass_kg <= 0:
        raise ValueError("system_mass_kg must be positive")
    if not records:
        raise ValueError("records must contain at least one entry")

    first_timestamp = records[0].timestamp
    if first_timestamp is None:
        raise ValueError("All records must contain timestamps after parsing")

    time_offsets = _timestamp_offsets(records)
    speed_values = _aligned_attribute(records, "speed")
    power_values = _aligned_attribute(records, "power")
    cadence_values = _aligned_attribute(records, "cadence")
    lap_values = _aligned_attribute(records, "lap")

    if np.any(np.isnan(speed_values)) or np.any(np.isnan(power_values)):
        raise ValueError("Speed and power streams are required for coefficient fitting")

    time_origin = first_timestamp.timestamp()
    climb_series = compute_climbing_rate(
        records, time_origin=time_origin, elevation_lag_s=elevation_lag_s
    )
    climb_rates = climb_series.values
    if climb_rates.size == 0:
        climb_rates = np.zeros_like(speed_values)

    acceleration_series = compute_acceleration(records, time_origin=time_origin)
    acceleration_rates = acceleration_series.values
    if acceleration_rates.size == 0:
        acceleration_rates = np.zeros_like(speed_values)

    if (
        climb_rates.shape != speed_values.shape
        or acceleration_rates.shape != speed_values.shape
    ):
        raise ValueError("Aligned derived metrics must match the record timeline")

    rolling_term = -system_mass_kg * GRAVITY_M_PER_S2 * speed_values
    aero_term = -0.5 * air_density * np.power(speed_values, 3)
    gravity_power = -system_mass_kg * GRAVITY_M_PER_S2 * climb_rates
    acceleration_power = -system_mass_kg * acceleration_rates * speed_values

    mask = np.isfinite(power_values) & np.isfinite(rolling_term) & np.isfinite(
        aero_term
    )
    if not np.any(mask):
        raise ValueError("Insufficient overlapping data to build power-balance samples")

    cadence_aligned = None
    if np.any(np.isfinite(cadence_values)):
        cadence_aligned = cadence_values
        mask &= np.isfinite(cadence_values) | np.isnan(cadence_values)

    timestamps = time_offsets[mask]
    crank_power = power_values[mask]
    rolling = rolling_term[mask]
    aero = aero_term[mask]
    gravity = gravity_power[mask]
    acceleration_p = acceleration_power[mask]
    cadence = cadence_aligned[mask] if cadence_aligned is not None else None
    lap_index = lap_values[mask] if np.any(np.isfinite(lap_values)) else None

    return PowerBalanceData(
        timestamps=timestamps,
        crank_power=crank_power,
        rolling_term=rolling,
        aero_term=aero,
        gravity_power=gravity,
        acceleration_power=acceleration_p,
        cadence=cadence,
        lap_index=lap_index,
    )


def _power_balance_residual(
    data: PowerBalanceData, *, eta: float, crr: float, cda: float
) -> np.ndarray:
    return (
        eta * data.crank_power
        + crr * data.rolling_term
        + cda * data.aero_term
        + data.gravity_power
        + data.acceleration_power
    )


def _weighted_least_squares(
    design: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    sqrt_w = np.sqrt(weights)
    design_w = design * sqrt_w[:, None]
    target_w = target * sqrt_w
    solution, residuals, rank, s = np.linalg.lstsq(design_w, target_w, rcond=None)
    print("Singular Values: ", s)
    return solution


def fit_power_balance_parameters(
    data: PowerBalanceData,
    weights: np.ndarray | None = None,
    *,
    include_drivetrain_efficiency: bool = True,
    include_rolling_resistance: bool = True,
    fixed_efficiency: float = 0.98,
    fixed_crr: float = 0.004,
) -> tuple[float, float, float]:
    """Estimate drivetrain efficiency, rolling resistance, and CdA.

    Parameters
    ----------
    data:
        Aligned power-balance inputs from :func:`prepare_power_balance_data`.
    weights:
        Optional per-sample weights. If provided, must match the number of
        samples in ``data``.
    include_drivetrain_efficiency:
        Whether to fit drivetrain efficiency (``eta``). If ``False``,
        ``fixed_efficiency`` is used instead.
    include_rolling_resistance:
        Whether to fit rolling resistance (``Crr``). If ``False``,
        ``fixed_crr`` is used instead.
    fixed_efficiency:
        Drivetrain efficiency to hold constant when ``include_drivetrain_efficiency``
        is ``False``.
    fixed_crr:
        Rolling resistance coefficient to hold constant when
        ``include_rolling_resistance`` is ``False``.

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

    design_terms = []
    target = -(data.gravity_power + data.acceleration_power)

    if include_drivetrain_efficiency:
        design_terms.append(data.crank_power)
    else:
        target -= fixed_efficiency * data.crank_power

    if include_rolling_resistance:
        design_terms.append(data.rolling_term)
    else:
        target -= fixed_crr * data.rolling_term

    design_terms.append(data.aero_term)
    design = np.column_stack(design_terms)

    params = _weighted_least_squares(design=design, target=target, weights=base_weights)

    eta = fixed_efficiency
    crr = fixed_crr
    param_index = 0

    if include_drivetrain_efficiency:
        eta = float(params[param_index])
        param_index += 1
    if include_rolling_resistance:
        crr = float(params[param_index])
        param_index += 1

    cda = float(params[param_index])

    return eta, crr, cda


def estimate_coefficients_from_records(
    records: Sequence[RecordPoint],
    system_mass_kg: float,
    *,
    air_density: float = AIR_DENSITY_KG_PER_M3,
    use_record_air_density: bool = False,
    cadence_weight_threshold: int | None = 30,
    power_weight_threshold: float | None = 50.0,
    weights: np.ndarray | None = None,
    even_lap_weighting: bool = False,
    include_drivetrain_efficiency: bool = True,
    include_rolling_resistance: bool = True,
    fixed_efficiency: float = 1.0,
    fixed_crr: float = 0.004,
    initial_cda: float = 0.32,
    residual_outlier_min: float | None = None,
    residual_outlier_max: float | None = None,
    elevation_lag_s: float = 0.0,
    estimate_elevation_lag: bool = False,
    elevation_lag_bound: float = 0.0,
) -> tuple[float, float, float]:
    """Driver helper to fit coefficients directly from FIT records.

    Cadence- and power-based weighting support is built in: when cadence is
    available, samples with cadence below ``cadence_weight_threshold`` receive
    zero weight; samples with crank power below ``power_weight_threshold`` are
    similarly zeroed. Caller weights (``weights``) are applied multiplicatively
    after gating. Set the thresholds to ``None`` to disable each weighting
    strategy. Enabling ``even_lap_weighting`` zeroes samples from odd-numbered
    laps (0-indexed) when lap annotations are present. Setting
    ``use_record_air_density`` to ``True`` automatically
    estimates air density from temperature and altitude streams when available.
    ``elevation_lag_s`` shifts the altitude stream before computing climbing
    rate to compensate for delayed elevation sensors. When ``estimate_elevation_lag``
    is ``True``, a golden-section search between zero and ``elevation_lag_bound`` is
    used to select the lag that minimizes weighted RMS residual.
    """

    result = estimate_power_balance(
        records,
        system_mass_kg,
        air_density=air_density,
        use_record_air_density=use_record_air_density,
        cadence_weight_threshold=cadence_weight_threshold,
        power_weight_threshold=power_weight_threshold,
        weights=weights,
        even_lap_weighting=even_lap_weighting,
        include_drivetrain_efficiency=include_drivetrain_efficiency,
        include_rolling_resistance=include_rolling_resistance,
        fixed_efficiency=fixed_efficiency,
        fixed_crr=fixed_crr,
        initial_cda=initial_cda,
        residual_outlier_min=residual_outlier_min,
        residual_outlier_max=residual_outlier_max,
        elevation_lag_s=elevation_lag_s,
        estimate_elevation_lag=estimate_elevation_lag,
        elevation_lag_bound=elevation_lag_bound,
    )
    return result.eta, result.crr, result.cda


def _weighted_rms(residuals: np.ndarray, weights: np.ndarray) -> float:
    clipped = np.clip(weights, 0, None)
    if np.count_nonzero(clipped) == 0:
        return float("inf")
    return float(np.sqrt(np.average(np.square(residuals), weights=clipped)))


def _estimation_for_lag(
    records: Sequence[RecordPoint],
    system_mass_kg: float,
    *,
    air_density: float,
    cadence_weight_threshold: int | None,
    power_weight_threshold: float | None,
    weights: np.ndarray | None,
    even_lap_weighting: bool,
    include_drivetrain_efficiency: bool,
    include_rolling_resistance: bool,
    fixed_efficiency: float,
    fixed_crr: float,
    initial_cda: float,
    residual_outlier_min: float | None,
    residual_outlier_max: float | None,
    elevation_lag_s: float,
) -> PowerBalanceEstimation:
    data = prepare_power_balance_data(
        records=records,
        system_mass_kg=system_mass_kg,
        air_density=air_density,
        elevation_lag_s=elevation_lag_s,
    )

    base_weights = np.ones(data.sample_count, dtype=float)
    if power_weight_threshold is not None:
        base_weights *= (data.crank_power >= power_weight_threshold).astype(float)
    if data.cadence is not None and cadence_weight_threshold is not None:
        base_weights *= (data.cadence >= cadence_weight_threshold).astype(float)

    if even_lap_weighting and data.lap_index is not None:
        lap_mask = np.isfinite(data.lap_index)
        odd_lap_mask = lap_mask & (np.remainder(data.lap_index, 2) == 1)
        base_weights[odd_lap_mask] = 0.0

    if weights is not None:
        if weights.shape != base_weights.shape:
            raise ValueError("weights must match the number of samples in data")
        base_weights *= weights

    if (
        residual_outlier_min is not None
        or residual_outlier_max is not None
    ):
        initial_residuals = _power_balance_residual(
            data, eta=fixed_efficiency, crr=fixed_crr, cda=initial_cda
        )
        if residual_outlier_min is not None:
            base_weights[initial_residuals < residual_outlier_min] = 0
        if residual_outlier_max is not None:
            base_weights[initial_residuals > residual_outlier_max] = 0

    eta, crr, cda = fit_power_balance_parameters(
        data=data,
        weights=base_weights,
        include_drivetrain_efficiency=include_drivetrain_efficiency,
        include_rolling_resistance=include_rolling_resistance,
        fixed_efficiency=fixed_efficiency,
        fixed_crr=fixed_crr,
    )

    residuals = _power_balance_residual(data, eta=eta, crr=crr, cda=cda)
    stats = {
        "samples": data.sample_count,
        "weighted_samples": int(np.count_nonzero(base_weights)),
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "weighted_rms": _weighted_rms(residuals, base_weights),
    }

    return PowerBalanceEstimation(
        data=data,
        weights=base_weights,
        residuals=residuals,
        eta=float(eta),
        crr=float(crr),
        cda=float(cda),
        elevation_lag_s=float(elevation_lag_s),
        stats=stats,
    )


def estimate_power_balance(
    records: Sequence[RecordPoint],
    system_mass_kg: float,
    *,
    air_density: float = AIR_DENSITY_KG_PER_M3,
    use_record_air_density: bool = False,
    cadence_weight_threshold: int | None = 30,
    power_weight_threshold: float | None = 50.0,
    weights: np.ndarray | None = None,
    even_lap_weighting: bool = False,
    include_drivetrain_efficiency: bool = True,
    include_rolling_resistance: bool = True,
    fixed_efficiency: float = 1.0,
    fixed_crr: float = 0.004,
    initial_cda: float = 0.32,
    residual_outlier_min: float | None = None,
    residual_outlier_max: float | None = None,
    elevation_lag_s: float = 0.0,
    estimate_elevation_lag: bool = False,
    elevation_lag_bound: float = 0.0,
) -> PowerBalanceEstimation:
    """Estimate coefficients and residuals with optional elevation lag search.

    The helper performs coefficient estimation for a fixed elevation lag or runs
    a golden-section search between zero and ``elevation_lag_bound`` (negative
    bounds allowed) to select the lag that minimizes the weighted RMS residual.
    Residual outlier gating uses an absolute wattage limit instead of
    standard-deviation scaling and supports asymmetric lower/upper bounds.
    """

    if (
        residual_outlier_min is not None
        and residual_outlier_max is not None
        and residual_outlier_min >= residual_outlier_max
    ):
        raise ValueError("residual_outlier_min must be less than residual_outlier_max")

    if use_record_air_density:
        air_density = estimate_air_density_from_records(
            records, fallback_air_density=air_density
        )

    best_result = _estimation_for_lag(
        records,
        system_mass_kg,
        air_density=air_density,
        cadence_weight_threshold=cadence_weight_threshold,
        power_weight_threshold=power_weight_threshold,
        weights=weights,
        even_lap_weighting=even_lap_weighting,
        include_drivetrain_efficiency=include_drivetrain_efficiency,
        include_rolling_resistance=include_rolling_resistance,
        fixed_efficiency=fixed_efficiency,
        fixed_crr=fixed_crr,
        initial_cda=initial_cda,
        residual_outlier_min=residual_outlier_min,
        residual_outlier_max=residual_outlier_max,
        elevation_lag_s=elevation_lag_s,
    )

    if not estimate_elevation_lag or elevation_lag_bound == 0:
        return best_result

    lower = min(0.0, float(elevation_lag_bound))
    upper = max(0.0, float(elevation_lag_bound))
    h = upper - lower

    if h <= 0:
        return best_result

    golden_ratio = (1 + np.sqrt(5)) / 2
    inv_phi = 1 / golden_ratio
    inv_phi2 = inv_phi * inv_phi
    a, b = lower, upper

    n = int(np.ceil(np.log((1e-2) / h) / np.log(inv_phi)))
    c = a + inv_phi2 * h
    d = a + inv_phi * h

    def objective(lag: float) -> float:
        nonlocal best_result
        result = _estimation_for_lag(
            records,
            system_mass_kg,
            air_density=air_density,
            cadence_weight_threshold=cadence_weight_threshold,
            power_weight_threshold=power_weight_threshold,
            weights=weights,
            even_lap_weighting=even_lap_weighting,
            include_drivetrain_efficiency=include_drivetrain_efficiency,
            include_rolling_resistance=include_rolling_resistance,
            fixed_efficiency=fixed_efficiency,
            fixed_crr=fixed_crr,
            initial_cda=initial_cda,
            residual_outlier_min=residual_outlier_min,
            residual_outlier_max=residual_outlier_max,
            elevation_lag_s=lag,
        )
        if result.stats["weighted_rms"] < best_result.stats["weighted_rms"]:
            best_result = result
        return result.stats["weighted_rms"]

    fc = objective(c)
    fd = objective(d)
    for _ in range(max(n - 1, 0)):
        if fc < fd:
            b = d
            d = c
            fd = fc
            h = inv_phi * h
            c = a + inv_phi2 * h
            fc = objective(c)
        else:
            a = c
            c = d
            fc = fd
            h = inv_phi * h
            d = a + inv_phi * h
            fd = objective(d)

    return best_result
