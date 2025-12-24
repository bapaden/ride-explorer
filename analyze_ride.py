"""Command line utility for visualizing Garmin cycling FIT files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

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


def _plot_route(ax, positions: Iterable[Tuple[float, float]]) -> None:
    lons: List[float] = []
    lats: List[float] = []
    for lon, lat in positions:
        lons.append(lon)
        lats.append(lat)

    if not lons or not lats:
        ax.text(0.5, 0.5, "No GPS data", ha="center", va="center")
        ax.set_axis_off()
        return

    ax.plot(lons, lats, color="tab:blue", linewidth=2)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("Route")
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


def _build_route_positions(records: Sequence[RecordPoint]) -> List[Tuple[float, float]]:
    positions: List[Tuple[float, float]] = []
    for record in records:
        if record.position is None:
            continue
        lat, lon = record.position
        if lat is None or lon is None:
            continue
        positions.append((lon, lat))
    return positions


def _figure_output_path(base: Path, suffix: str) -> Path:
    if base.suffix:
        return base.with_name(f"{base.stem}_{suffix}{base.suffix}")
    return base.with_name(f"{base.name}_{suffix}.png")


def _plot_activity(
    data: CyclingFitData, system_mass_kg: float, show: bool, output: Path | None
) -> None:
    plt.style.use("ggplot")

    figures: list[tuple[str, plt.Figure]] = []

    route_fig, route_ax = plt.subplots(figsize=(8, 8))
    route_fig.suptitle(f"GPS Route: {data.source.name}", fontsize=14)
    _plot_route(route_ax, _build_route_positions(data.records))
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

    args = parser.parse_args()

    fit_path = args.fit_file
    if not fit_path.exists():
        raise FileNotFoundError(f"FIT file not found: {fit_path}")

    if args.system_mass <= 0:
        raise ValueError("--system_mass must be positive (in kilograms)")

    data = parse_cycling_fit(fit_path)

    show_plot = not args.no_show and args.output is None
    _plot_activity(
        data, system_mass_kg=args.system_mass, show=show_plot, output=args.output
    )


if __name__ == "__main__":
    main()
