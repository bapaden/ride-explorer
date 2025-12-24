"""Command line utility for visualizing Garmin cycling FIT files."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from ride_explorer.coefficient_estimator import (
    PowerBalanceData,
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
    for lon, lat, *_ in positions:
        lons.append(lon)
        lats.append(lat)

    if not lons or not lats:
        ax.text(0.5, 0.5, "No GPS data", ha="center", va="center")
        ax.set_axis_off()
        return

    ax.plot(lons, lats, color="tab:blue", linewidth=2)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("Route (top-down)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)


def _plot_elevation_ribbon(
    ax, positions: Iterable[Tuple[float, float, Optional[float]]]
) -> None:
    lons: list[float] = []
    lats: list[float] = []
    alts: list[float] = []
    for lon, lat, alt in positions:
        if alt is None:
            continue
        lons.append(lon)
        lats.append(lat)
        alts.append(alt)

    if not lons or not alts:
        ax.text(
            0.5,
            0.5,
            "No elevation data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return

    points_llh = np.column_stack((lons, lats, alts))
    if len(points_llh) < 2:
        ax.text(
            0.5,
            0.5,
            "Insufficient elevation samples",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return

    lat0 = points_llh[0, 1]
    lon0 = points_llh[0, 0]
    cos_lat0 = math.cos(math.radians(lat0))
    east_m = (points_llh[:, 0] - lon0) * 111_320 * cos_lat0
    north_m = (points_llh[:, 1] - lat0) * 110_540
    east_km = east_m / 1000.0
    north_km = north_m / 1000.0

    points = np.column_stack((east_km, north_km, alts))

    path_extent = max(
        max(east_km) - min(east_km),
        max(north_km) - min(north_km),
    )
    ribbon_width = 0.04 * path_extent if path_extent > 0 else 0.01

    xy = points[:, :2]
    vectors = np.diff(xy, axis=0)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals = np.column_stack((-vectors[:, 1], vectors[:, 0])) / norms

    point_normals = np.zeros_like(xy)
    point_normals[0] = normals[0]
    point_normals[-1] = normals[-1]
    if len(points) > 2:
        point_normals[1:-1] = normals[:-1] + normals[1:]
        lengths = np.linalg.norm(point_normals[1:-1], axis=1, keepdims=True)
        lengths[lengths == 0] = 1.0
        point_normals[1:-1] /= lengths

    left_edge = xy + ribbon_width * point_normals
    right_edge = xy - ribbon_width * point_normals

    ribbon_faces = []
    face_colors = []
    for idx in range(len(points) - 1):
        face = [
            (left_edge[idx, 0], left_edge[idx, 1], points[idx, 2]),
            (right_edge[idx, 0], right_edge[idx, 1], points[idx, 2]),
            (right_edge[idx + 1, 0], right_edge[idx + 1, 1], points[idx + 1, 2]),
            (left_edge[idx + 1, 0], left_edge[idx + 1, 1], points[idx + 1, 2]),
        ]
        ribbon_faces.append(face)
        face_colors.append((points[idx, 2] + points[idx + 1, 2]) / 2)

    alt_min = min(alts)
    alt_max = max(alts)
    if math.isclose(alt_min, alt_max):
        alt_max = alt_min + 1.0

    norm = Normalize(vmin=alt_min, vmax=alt_max)
    cmap = plt.get_cmap("plasma")

    poly = Poly3DCollection(
        ribbon_faces,
        facecolors=cmap(norm(face_colors)),
        edgecolors="k",
        linewidths=0.25,
        alpha=0.9,
    )
    ax.add_collection3d(poly)

    segments = np.stack([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(
        segments,
        colors="k",
        linewidths=0.5,
        alpha=0.4,
    )
    ax.add_collection3d(lc)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(alts)
    ax.figure.colorbar(sm, ax=ax, pad=0.1, label="Elevation (m)")

    ax.set_xlabel("East offset (km)")
    ax.set_ylabel("North offset (km)")
    ax.set_zlabel("Elevation (m)")
    ax.set_title("Elevation ribbon")
    ax.view_init(elev=25, azim=-70)
    ax.grid(True, alpha=0.25)
    ax.set_facecolor("white")
    east_range = max(east_km) - min(east_km)
    north_range = max(north_km) - min(north_km)
    alt_range = alt_max - alt_min
    ax.set_box_aspect((east_range or 1.0, north_range or 1.0, alt_range or 1.0))


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


def _plot_activity(
    data: CyclingFitData,
    system_mass_kg: float,
    show: bool,
    output: Path | None,
    *,
    eta: float,
    crr: float,
    cda: float,
) -> None:
    plt.style.use("ggplot")

    figures: list[tuple[str, plt.Figure]] = []

    route_positions = _build_route_positions(data.records)

    route_fig = plt.figure(figsize=(14, 6))
    route_fig.suptitle(f"GPS Route: {data.source.name}", fontsize=14)
    route_grid = route_fig.add_gridspec(1, 2, width_ratios=[1, 1.2])
    route_ax = route_fig.add_subplot(route_grid[0, 0])
    route_ax_3d = route_fig.add_subplot(route_grid[0, 1], projection="3d")
    _plot_route(route_ax, route_positions)
    _plot_elevation_ribbon(route_ax_3d, route_positions)
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
    power_series = compute_mechanical_power(data.records, system_mass_kg)
    _plot_power_components(power_ax, power_series)
    power_fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    figures.append(("power", power_fig))

    power_balance_data: PowerBalanceData | None = None
    power_balance_error: str | None = None
    try:
        power_balance_data = prepare_power_balance_data(
            data.records, system_mass_kg=system_mass_kg
        )
    except ValueError as exc:
        power_balance_error = str(exc)

    power_balance_fig, power_balance_ax = plt.subplots(figsize=(12, 6))
    power_balance_fig.suptitle(
        (
            f"Power balance (CdA={cda:.3f}, Crr={crr:.4f}, η={eta:.2f}): "
            f"{data.source.name}"
        ),
        fontsize=14,
    )
    _plot_power_balance_terms(
        power_balance_ax,
        power_balance_data,
        eta=eta,
        crr=crr,
        cda=cda,
        error_message=power_balance_error,
    )
    power_balance_fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    figures.append(("power_balance", power_balance_fig))

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

    data = parse_cycling_fit(fit_path)

    show_plot = not args.no_show and args.output is None
    _plot_activity(
        data,
        system_mass_kg=args.system_mass,
        show=show_plot,
        output=args.output,
        eta=args.eta,
        crr=args.crr,
        cda=args.cda,
    )


if __name__ == "__main__":
    main()
