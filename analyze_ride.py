"""Command line utility for visualizing Garmin cycling FIT files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from ride_explorer.coefficient_estimator import (
    PowerBalanceData,
    fit_power_balance_parameters,
    prepare_power_balance_data,
)
from ride_explorer.derived_metrics import compute_mechanical_power
from ride_explorer.fit_parser import CyclingFitData, RecordPoint, parse_cycling_fit


def _time_series(
    records: Sequence[RecordPoint],
    attribute: str,
    transform: Callable[[float], float] | None = None,
) -> Tuple[List[float], List[float]]:
    """Return time offsets (seconds) and values for a numeric record attribute.

    Records without timestamps or the target attribute are skipped.
    """

    transform = transform or (lambda value: value)
    timestamps: List[float] = []
    values: List[float] = []

    first_timestamp: Optional[float] = None
    for record in records:
        ts = record.timestamp
        value = getattr(record, attribute)
        if ts is None or value is None:
            continue

        first_timestamp = first_timestamp or ts.timestamp()
        timestamps.append(ts.timestamp() - first_timestamp)
        values.append(transform(float(value)))

    return timestamps, values


def _plot_route(ax, positions: Iterable[Tuple[float, float, Optional[float]]]) -> None:
    lons: List[float] = []
    lats: List[float] = []
    alts: List[float] = []
    for lon, lat, alt in positions:
        lons.append(lon)
        lats.append(lat)
        alts.append(alt if alt is not None else np.nan)

    if not lons or not lats:
        ax.text(0.5, 0.5, "No GPS data", ha="center", va="center")
        ax.set_axis_off()
        return

    if np.all(np.isnan(alts)) or len(lons) < 2:
        ax.plot(lons, lats, color="tab:blue", linewidth=2)
        colorbar = None
    else:
        coords = np.column_stack((lons, lats))
        segments = np.stack([coords[:-1], coords[1:]], axis=1)
        alt_series = np.asarray(alts, dtype=float)
        alt_min = np.nanmin(alt_series)
        alt_max = np.nanmax(alt_series)
        if np.isclose(alt_min, alt_max):
            alt_max = alt_min + 1.0
        norm = Normalize(vmin=alt_min, vmax=alt_max)
        cmap = plt.get_cmap("plasma")
        lc = LineCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidths=2.5,
        )
        lc.set_array((alt_series[:-1] + alt_series[1:]) / 2)
        line = ax.add_collection(lc)
        colorbar = ax.figure.colorbar(line, ax=ax, pad=0.02)
        colorbar.set_label("Elevation (m)")

    ax.plot(lons, lats, color="black", linewidth=0.8, alpha=0.3)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("Route (colored by elevation)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)




def _plot_metrics(ax, records: Sequence[RecordPoint]) -> None:
    metrics = [
        ("Power (W)", "power", None),
        ("Cadence (rpm)", "cadence", None),
        ("Speed (km/h)", "speed", lambda value: value * 3.6),
        ("Heart rate (bpm)", "heart_rate", None),
        ("Altitude (m)", "altitude", None),
        ("Temperature (°C)", "temperature", None),
    ]

    has_data = False
    for label, attr, transform in metrics:
        times, values = _time_series(records, attr, transform=transform)
        if not values:
            continue
        has_data = True
        ax.plot(times, values, label=label)

    if not has_data:
        ax.text(0.5, 0.5, "No ride metrics available", ha="center", va="center")
        ax.set_axis_off()
        return

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")
    ax.set_title("Ride metrics")
    ax.legend(loc="upper right")
    ax.grid(True)


def _plot_power_components(ax, power_series) -> None:
    has_data = False
    for label, key in [
        ("Acceleration power (W)", "acceleration_power"),
        ("Climbing power (W)", "climbing_power"),
    ]:
        series = power_series[key]
        if series.times.size == 0 or series.values.size == 0:
            continue
        has_data = True
        ax.plot(series.times, series.values, label=label)

    if not has_data:
        ax.text(0.5, 0.5, "No derived power data", ha="center", va="center")
        ax.set_axis_off()
        return

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (W)")
    ax.set_title("Mechanical power components")
    ax.legend(loc="upper right")
    ax.grid(True)


def _plot_power_balance_terms(
    ax,
    power_data: PowerBalanceData | None,
    eta: float,
    crr: float,
    cda: float,
    error_message: str | None = None,
) -> None:
    if error_message:
        ax.text(0.5, 0.5, error_message, ha="center", va="center")
        ax.set_axis_off()
        return

    if power_data is None or power_data.sample_count == 0:
        ax.text(0.5, 0.5, "No power-balance data available", ha="center", va="center")
        ax.set_axis_off()
        return

    times = power_data.timestamps
    components = [
        ("η × crank power (W)", eta * power_data.crank_power),
        ("Rolling losses (W)", crr * power_data.rolling_term),
        ("Aerodynamic losses (W)", cda * power_data.aero_term),
        ("Gravitational power (W)", power_data.gravity_power),
        ("Acceleration power (W)", power_data.acceleration_power),
    ]

    has_data = False
    for label, values in components:
        if values.size == 0:
            continue
        has_data = True
        ax.plot(times, values, label=label)

    residual = (
        eta * power_data.crank_power
        + crr * power_data.rolling_term
        + cda * power_data.aero_term
        + power_data.gravity_power
        + power_data.acceleration_power
    )
    if residual.size:
        ax.plot(times, residual, label="Power balance residual (W)", linestyle="--")

    if not has_data:
        ax.text(0.5, 0.5, "No derived power data", ha="center", va="center")
        ax.set_axis_off()
        return

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (W)")
    ax.set_title("Power balance components")
    ax.legend(loc="upper right")
    ax.grid(True)


def _plot_residuals(
    ax,
    power_data: PowerBalanceData | None,
    residuals: np.ndarray | None,
    weights: np.ndarray | None,
    *,
    eta: float,
    crr: float,
    cda: float,
    error_message: str | None = None,
) -> None:
    if error_message:
        ax.text(0.5, 0.5, error_message, ha="center", va="center")
        ax.set_axis_off()
        return

    if power_data is None or power_data.sample_count == 0 or residuals is None:
        ax.text(0.5, 0.5, "No residuals to display", ha="center", va="center")
        ax.set_axis_off()
        return

    times = power_data.timestamps
    marker_style = {"marker": "x", "s": 25, "alpha": 0.9}
    if weights is None:
        ax.scatter(
            times,
            residuals,
            label="Power balance residual (W)",
            color="tab:blue",
            **marker_style,
        )
    else:
        zero_weight_mask = weights <= 0
        inlier_mask = ~zero_weight_mask
        if np.any(inlier_mask):
            ax.scatter(
                times[inlier_mask],
                residuals[inlier_mask],
                label="Weighted samples",
                color="tab:blue",
                **marker_style,
            )
        if np.any(zero_weight_mask):
            ax.scatter(
                times[zero_weight_mask],
                residuals[zero_weight_mask],
                label="Zero-weighted samples",
                color="tab:red",
                **marker_style,
            )

    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Residual (W)")
    ax.set_title(
        (
            "Residuals " f"(η={eta:.3f}, Crr={crr:.4f}, CdA={cda:.3f})"
        )
    )
    ax.legend(loc="upper right")
    ax.grid(True)


def _plot_residual_histogram(
    ax,
    residuals: np.ndarray | None,
    weights: np.ndarray | None,
    *,
    eta: float,
    crr: float,
    cda: float,
    error_message: str | None = None,
) -> None:
    if error_message:
        ax.text(0.5, 0.5, error_message, ha="center", va="center")
        ax.set_axis_off()
        return

    if residuals is None or residuals.size == 0:
        ax.text(0.5, 0.5, "No residuals to display", ha="center", va="center")
        ax.set_axis_off()
        return

    hist_weights = None
    if weights is not None:
        hist_weights = np.clip(weights, 0, None)

    ax.hist(residuals, bins=30, weights=hist_weights, color="tab:blue", alpha=0.8)
    ax.set_xlabel("Residual (W)")
    ax.set_ylabel("Count")
    ax.set_title(
        (
            "Residual distribution "
            f"(η={eta:.3f}, Crr={crr:.4f}, CdA={cda:.3f})"
        )
    )
    ax.grid(True)


def _build_route_positions(
    records: Sequence[RecordPoint],
) -> List[Tuple[float, float, Optional[float]]]:
    positions: List[Tuple[float, float, Optional[float]]] = []
    for record in records:
        if record.position is None:
            continue
        lat, lon = record.position
        if lat is None or lon is None:
            continue
        positions.append((lon, lat, record.altitude))
    return positions


def _figure_output_path(base: Path, suffix: str) -> Path:
    if base.suffix:
        return base.with_name(f"{base.stem}_{suffix}{base.suffix}")
    return base.with_name(f"{base.name}_{suffix}.png")


def _power_balance_residual(
    power_data: PowerBalanceData, eta: float, crr: float, cda: float
) -> np.ndarray:
    return (
        eta * power_data.crank_power
        + crr * power_data.rolling_term
        + cda * power_data.aero_term
        + power_data.gravity_power
        + power_data.acceleration_power
    )


def _plot_activity(
    data: CyclingFitData,
    system_mass_kg: float,
    show: bool,
    output: Path | None,
    *,
    eta: float,
    crr: float,
    cda: float,
    estimate_parameters: bool,
    estimate_efficiency: bool,
    residual_std_multiplier: float,
    elevation_lag_s: float,
) -> None:
    plt.style.use("ggplot")

    figures: list[tuple[str, plt.Figure]] = []

    route_positions = _build_route_positions(data.records)

    route_fig, route_ax = plt.subplots(figsize=(10, 8))
    route_fig.suptitle(f"GPS Route: {data.source.name}", fontsize=14)
    _plot_route(route_ax, route_positions)
    route_fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    figures.append(("route", route_fig))

    metrics_fig, metrics_ax = plt.subplots(figsize=(12, 6))
    metrics_fig.suptitle(f"Ride Metrics: {data.source.name}", fontsize=14)
    _plot_metrics(metrics_ax, data.records)
    metrics_fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    figures.append(("metrics", metrics_fig))

    power_fig, power_ax = plt.subplots(figsize=(12, 6))
    power_fig.suptitle(
        f"Derived Power (system mass: {system_mass_kg} kg): {data.source.name}",
        fontsize=14,
    )
    power_series = compute_mechanical_power(
        data.records, system_mass_kg, elevation_lag_s=elevation_lag_s
    )
    _plot_power_components(power_ax, power_series)
    power_fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    figures.append(("power", power_fig))

    power_balance_data: PowerBalanceData | None = None
    power_balance_error: str | None = None
    estimated_params: tuple[float, float, float] | None = None
    estimation_weights: np.ndarray | None = None
    residuals: np.ndarray | None = None
    plot_eta = eta
    plot_crr = crr
    plot_cda = cda
    estimation_stats: dict[str, float] | None = None
    try:
        power_balance_data = prepare_power_balance_data(
            data.records,
            system_mass_kg=system_mass_kg,
            elevation_lag_s=elevation_lag_s,
        )
    except ValueError as exc:
        power_balance_error = str(exc)

    if power_balance_data is not None and power_balance_error is None:
        estimation_weights = np.ones(power_balance_data.sample_count, dtype=float)
        estimation_weights[power_balance_data.crank_power < 50] = 0
        if power_balance_data.cadence is not None:
            estimation_weights[power_balance_data.cadence < 25] = 0

        initial_residuals = _power_balance_residual(
            power_balance_data, eta, crr, cda
        )
        residual_mean = float(np.mean(initial_residuals))
        residual_std = float(np.std(initial_residuals))
        if residual_std > 0 and np.isfinite(residual_std):
            threshold = residual_std_multiplier * residual_std
            outlier_mask = np.abs(initial_residuals - residual_mean) > threshold
            estimation_weights[outlier_mask] = 0

        try:
            if estimate_parameters:
                estimated_params = fit_power_balance_parameters(
                    data=power_balance_data,
                    weights=estimation_weights,
                    include_drivetrain_efficiency=estimate_efficiency,
                    fixed_efficiency=eta,
                )
                plot_eta, plot_crr, plot_cda = estimated_params
                print(
                    "Estimated parameters "
                    f"η={plot_eta:.3f}, Crr={plot_crr:.4f}, CdA={plot_cda:.3f}"
                )
            else:
                print(
                    "Skipping estimation; using provided parameters "
                    f"η={plot_eta:.3f}, Crr={plot_crr:.4f}, CdA={plot_cda:.3f}"
                )
            residuals = _power_balance_residual(
                power_balance_data, plot_eta, plot_crr, plot_cda
            )

            nonzero_weights = (
                estimation_weights is not None
                and np.count_nonzero(estimation_weights) > 0
            )
            if nonzero_weights:
                weighted_rms = np.sqrt(
                    np.average(
                        np.square(residuals),
                        weights=np.clip(estimation_weights, 0, None),
                    )
                )
            else:
                weighted_rms = float("nan")

            estimation_stats = {
                "samples": power_balance_data.sample_count,
                "weighted_samples": int(np.count_nonzero(estimation_weights)),
                "residual_mean": float(np.mean(residuals)),
                "residual_std": float(np.std(residuals)),
                "weighted_rms": weighted_rms,
            }

            print(
                "Estimation stats: "
                f"samples={estimation_stats['samples']}, "
                f"weighted_samples={estimation_stats['weighted_samples']}, "
                f"residual_mean={estimation_stats['residual_mean']:.2f} W, "
                f"residual_std={estimation_stats['residual_std']:.2f} W, "
                f"weighted_rms={estimation_stats['weighted_rms']:.2f} W"
            )
        except Exception as exc:  # pragma: no cover - defensive user-facing handler
            power_balance_error = f"Coefficient estimation failed: {exc}"

    power_balance_fig, power_balance_ax = plt.subplots(figsize=(12, 6))
    power_balance_fig.suptitle(
        (
            f"Power balance (CdA={plot_cda:.3f}, Crr={plot_crr:.4f}, "
            f"η={plot_eta:.2f}): {data.source.name}"
        ),
        fontsize=14,
    )
    _plot_power_balance_terms(
        power_balance_ax,
        power_balance_data,
        eta=plot_eta,
        crr=plot_crr,
        cda=plot_cda,
        error_message=power_balance_error,
    )
    power_balance_fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    figures.append(("power_balance", power_balance_fig))

    residual_fig, residual_ax = plt.subplots(figsize=(12, 4))
    _plot_residuals(
        residual_ax,
        power_balance_data,
        residuals,
        estimation_weights,
        eta=plot_eta,
        crr=plot_crr,
        cda=plot_cda,
        error_message=power_balance_error,
    )
    residual_fig.tight_layout()
    figures.append(("residuals", residual_fig))

    residual_hist_fig, residual_hist_ax = plt.subplots(figsize=(10, 4))
    _plot_residual_histogram(
        residual_hist_ax,
        residuals,
        estimation_weights,
        eta=plot_eta,
        crr=plot_crr,
        cda=plot_cda,
        error_message=power_balance_error,
    )
    residual_hist_fig.tight_layout()
    figures.append(("residual_histogram", residual_hist_fig))

    if output:
        for suffix, fig in figures:
            target = _figure_output_path(output, suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target, dpi=150)
            print(f"Saved {suffix} visualization to {target}")

    if show:
        plt.show()
    else:
        for _, fig in figures:
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize GPS route and ride metrics from a Garmin FIT file.",
    )
    parser.add_argument(
        "--fit_file",
        type=Path,
        required=True,
        help="Path to the .fit file to analyze",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the generated figure instead of displaying it.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the matplotlib window (useful for headless environments).",
    )
    parser.add_argument(
        "--system_mass",
        type=float,
        required=True,
        help=(
            "Total system mass in kilograms (rider + bike) used for derived power "
            "calculations."
        ),
    )
    parser.add_argument(
        "--residual_std_multiplier",
        type=float,
        default=2.0,
        help=(
            "Multiplier applied to residual standard deviation for zero-weighting "
            "outlier samples (default: 2.0)."
        ),
    )
    parser.add_argument(
        "--cda",
        type=float,
        default=0.32,
        help="Aerodynamic drag area (m^2) used for power balance calculations.",
    )
    parser.add_argument(
        "--crr",
        type=float,
        default=0.004,
        help="Rolling resistance coefficient (unitless) for power balance calculations.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.97,
        help="Drivetrain efficiency (0–1) used for power balance calculations.",
    )
    parser.add_argument(
        "--smoothing_window",
        type=int,
        default=1,
        help=(
            "Window size for moving-average smoothing applied to interpolated "
            "record streams. A value of 1 disables smoothing."
        ),
    )
    parser.add_argument(
        "--elevation-lag",
        type=float,
        default=0.0,
        help=(
            "Lag to apply to elevation data (seconds). Positive values shift "
            "elevation earlier to compensate delayed sensors."
        ),
    )
    parser.add_argument(
        "--estimate_parameters",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Whether to run parameter estimation. When true, supplied "
            "eta/Crr/CdA values are used for weighting gates and plots; "
            "when false, supplied values are used without optimization."
        ),
    )
    parser.add_argument(
        "--estimate_efficiency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether to estimate drivetrain efficiency (eta). When false, "
            "eta is fixed to the provided value and only Crr/CdA are estimated."
        ),
    )

    args = parser.parse_args()

    fit_path = args.fit_file
    if not fit_path.exists():
        raise FileNotFoundError(f"FIT file not found: {fit_path}")

    if args.system_mass <= 0:
        raise ValueError("--system_mass must be positive (in kilograms)")
    if args.cda <= 0:
        raise ValueError("--cda must be positive")
    if args.crr < 0:
        raise ValueError("--crr must be non-negative")
    if not 0 < args.eta <= 1:
        raise ValueError("--eta must be in the range (0, 1]")
    if args.smoothing_window <= 0:
        raise ValueError("--smoothing_window must be a positive integer")

    data = parse_cycling_fit(fit_path, smoothing_window=args.smoothing_window)

    show_plot = not args.no_show and args.output is None
    _plot_activity(
        data,
        system_mass_kg=args.system_mass,
        show=show_plot,
        output=args.output,
        eta=args.eta,
        crr=args.crr,
        cda=args.cda,
        estimate_parameters=args.estimate_parameters,
        estimate_efficiency=args.estimate_efficiency,
        residual_std_multiplier=args.residual_std_multiplier,
        elevation_lag_s=args.elevation_lag,
    )


if __name__ == "__main__":
    main()
