"""Command-line argument definitions for ride analysis utilities.

The helpers centralize argument construction so flags stay consistent across
scripts and can be documented in one place.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_argument_parser() -> argparse.ArgumentParser:
    """Construct the argument parser used by ``analyze_ride.py``.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser ready for ``parse_args``.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Visualize GPS route and ride metrics from a Garmin FIT file and "
            "optionally estimate aerodynamic and rolling resistance coefficients."
        ),
    )

    parser.add_argument(
        "--fit_file",
        type=Path,
        required=True,
        help="Path to the .fit file to analyze.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional path to save generated figures instead of displaying them. "
            "The base path is suffixed for each plot."
        ),
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
    parser.add_argument(
        "--estimate_crr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether to estimate rolling resistance (Crr). When false, "
            "Crr is fixed to the provided value and only eta/CdA are estimated."
        ),
    )
    parser.add_argument(
        "--min-power-weighting",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether to zero-weight samples with crank power below the minimum "
            "power threshold during coefficient estimation."
        ),
    )
    parser.add_argument(
        "--min-power-threshold",
        type=float,
        default=50.0,
        help=(
            "Minimum crank power (W) required for a nonzero sample weight when "
            "minimum power weighting is enabled."
        ),
    )
    parser.add_argument(
        "--min-cadence-weighting",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether to zero-weight samples with cadence below the minimum cadence "
            "threshold during coefficient estimation when cadence data is available."
        ),
    )
    parser.add_argument(
        "--min-cadence-threshold",
        type=float,
        default=25.0,
        help=(
            "Minimum cadence (rpm) required for a nonzero sample weight when "
            "minimum cadence weighting is enabled."
        ),
    )

    return parser

