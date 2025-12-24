"""Command line utility for visualizing Garmin cycling FIT files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

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


def _plot_activity(data: CyclingFitData, show: bool, output: Path | None) -> None:
    plt.style.use("ggplot")

    fig, (route_ax, metrics_ax) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f"Ride summary: {data.source.name}", fontsize=14)

    _plot_route(route_ax, _build_route_positions(data.records))
    _plot_metrics(metrics_ax, data.records)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
        print(f"Saved visualization to {output}")

    if show:
        plt.show()
    else:
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

    args = parser.parse_args()

    fit_path = args.fit_file
    if not fit_path.exists():
        raise FileNotFoundError(f"FIT file not found: {fit_path}")

    data = parse_cycling_fit(fit_path)

    show_plot = not args.no_show and args.output is None
    _plot_activity(data, show=show_plot, output=args.output)


if __name__ == "__main__":
    main()
