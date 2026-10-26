"""Generate the analytical local-mu(I) Bagnold chute-flow reference.

The solution assumes steady, uniform, fully developed flow of constant solid
fraction and depth over a no-slip base.  It is deliberately independent of the
HydraxMPM solver so it can serve as a validation oracle.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class BagnoldParameters:
    height: float = 0.10
    grain_diameter: float = 0.002
    grain_density: float = 2650.0
    solid_fraction: float = 0.60
    gravity: float = 9.81
    chute_angle_deg: float = 24.0
    static_friction_angle_deg: float = 20.0
    dynamic_friction_angle_deg: float = 30.0
    inertial_number_scale: float = 0.35

    @property
    def bulk_density(self) -> float:
        return self.grain_density * self.solid_fraction

    @property
    def slope_friction(self) -> float:
        return float(np.tan(np.deg2rad(self.chute_angle_deg)))

    @property
    def mu_s(self) -> float:
        return float(np.tan(np.deg2rad(self.static_friction_angle_deg)))

    @property
    def mu_d(self) -> float:
        return float(np.tan(np.deg2rad(self.dynamic_friction_angle_deg)))

    @property
    def inertial_number(self) -> float:
        if not self.mu_s < self.slope_friction < self.mu_d:
            raise ValueError(
                "A steady local-mu(I) Bagnold solution requires "
                "mu_s < tan(theta) < mu_d"
            )
        return self.inertial_number_scale * (
            (self.slope_friction - self.mu_s)
            / (self.mu_d - self.slope_friction)
        )

    @property
    def velocity_prefactor(self) -> float:
        theta = np.deg2rad(self.chute_angle_deg)
        return (
            2.0
            * self.inertial_number
            / (3.0 * self.grain_diameter)
            * np.sqrt(self.solid_fraction * self.gravity * np.cos(theta))
        )

    @property
    def mean_velocity(self) -> float:
        theta = np.deg2rad(self.chute_angle_deg)
        return (
            2.0
            / 5.0
            * self.inertial_number
            / self.grain_diameter
            * np.sqrt(self.solid_fraction * self.gravity * np.cos(theta))
            * self.height**1.5
        )

    @property
    def surface_velocity(self) -> float:
        return self.velocity_prefactor * self.height**1.5


def evaluate_bagnold(y: np.ndarray, params: BagnoldParameters) -> dict[str, np.ndarray]:
    """Evaluate velocity, shear rate, and lithostatic pressure at heights ``y``."""
    y = np.asarray(y, dtype=float)
    y_clipped = np.clip(y, 0.0, params.height)
    depth = params.height - y_clipped
    theta = np.deg2rad(params.chute_angle_deg)
    shear_rate = (
        params.inertial_number
        / params.grain_diameter
        * np.sqrt(params.solid_fraction * params.gravity * np.cos(theta) * depth)
    )
    velocity = params.velocity_prefactor * (
        params.height**1.5 - depth**1.5
    )
    pressure = params.bulk_density * params.gravity * np.cos(theta) * depth
    return {
        "y": y,
        "velocity": velocity,
        "shear_rate": shear_rate,
        "pressure": pressure,
        "inertial_number": np.full_like(y, params.inertial_number),
        "stress_ratio": np.full_like(y, params.slope_friction),
    }


def compare_profile(path: Path, params: BagnoldParameters):
    numerical = np.genfromtxt(path, delimiter=",", names=True)
    if "y" not in numerical.dtype.names or "mean_vx" not in numerical.dtype.names:
        raise ValueError("numerical profile must contain y and mean_vx columns")

    y = np.atleast_1d(numerical["y"]).astype(float)
    velocity = np.atleast_1d(numerical["mean_vx"]).astype(float)
    valid = np.isfinite(y) & np.isfinite(velocity) & (y >= 0.0) & (y <= params.height)
    y = y[valid]
    velocity = velocity[valid]
    reference = evaluate_bagnold(y, params)["velocity"]

    if "particle_count" in numerical.dtype.names:
        weights = np.atleast_1d(numerical["particle_count"]).astype(float)[valid]
    else:
        weights = np.ones_like(y)
    weights = np.maximum(weights, 0.0)

    residual = velocity - reference
    relative_l2 = np.sqrt(np.sum(weights * residual**2) / np.sum(weights * reference**2))
    numerical_mean = np.sum(weights * velocity) / np.sum(weights)
    mean_error = (numerical_mean - params.mean_velocity) / params.mean_velocity
    relative_linf = np.max(np.abs(residual)) / params.surface_velocity
    return {
        "y": y,
        "numerical_velocity": velocity,
        "reference_velocity": reference,
        "relative_l2_profile_error": relative_l2,
        "relative_linf_profile_error": relative_linf,
        "numerical_weighted_mean_velocity": numerical_mean,
        "analytical_mean_velocity": params.mean_velocity,
        "relative_mean_velocity_error": mean_error,
    }


def write_reference(
    params: BagnoldParameters,
    output_dir: Path,
    num_samples: int = 201,
    numerical_profile: Path | None = None,
    write_plot: bool = True,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    y = np.linspace(0.0, params.height, num_samples)
    reference = evaluate_bagnold(y, params)
    np.savetxt(
        output_dir / "bagnold_reference.csv",
        np.column_stack(
            [
                reference["y"],
                reference["velocity"],
                reference["shear_rate"],
                reference["pressure"],
                reference["inertial_number"],
                reference["stress_ratio"],
            ]
        ),
        delimiter=",",
        header="y,velocity,shear_rate,pressure,inertial_number,stress_ratio",
        comments="",
    )

    summary = {
        "inertial_number": params.inertial_number,
        "stress_ratio": params.slope_friction,
        "mean_velocity": params.mean_velocity,
        "surface_velocity": params.surface_velocity,
    }
    comparison = None
    if numerical_profile is not None:
        comparison = compare_profile(numerical_profile, params)
        for name in (
            "relative_l2_profile_error",
            "relative_linf_profile_error",
            "numerical_weighted_mean_velocity",
            "analytical_mean_velocity",
            "relative_mean_velocity_error",
        ):
            summary[name] = comparison[name]
        np.savetxt(
            output_dir / "profile_comparison.csv",
            np.column_stack(
                [
                    comparison["y"],
                    comparison["numerical_velocity"],
                    comparison["reference_velocity"],
                    comparison["numerical_velocity"]
                    - comparison["reference_velocity"],
                ]
            ),
            delimiter=",",
            header="y,numerical_velocity,reference_velocity,error",
            comments="",
        )

    with (output_dir / "bagnold_summary.csv").open("w", encoding="utf-8") as stream:
        stream.write("metric,value\n")
        for name, value in summary.items():
            stream.write(f"{name},{value:.16e}\n")

    if write_plot:
        plt.figure(figsize=(6, 4))
        plt.plot(reference["velocity"], reference["y"], label="Bagnold", linewidth=2)
        if comparison is not None:
            plt.plot(
                comparison["numerical_velocity"],
                comparison["y"],
                "o-",
                label="HydraxMPM",
            )
        plt.xlabel("streamwise velocity [m/s]")
        plt.ylabel("height above base [m]")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "bagnold_profile.png", dpi=180)
        plt.close()

    print(f"I(theta) = {params.inertial_number:.8f}")
    print(f"tan(theta) = {params.slope_friction:.8f}")
    print(f"Analytical mean velocity = {params.mean_velocity:.8f} m/s")
    print(f"Analytical surface velocity = {params.surface_velocity:.8f} m/s")
    if comparison is not None:
        print(
            "Numerical comparison: "
            f"mean error={comparison['relative_mean_velocity_error']:.3%}, "
            f"L2 profile error={comparison['relative_l2_profile_error']:.3%}"
        )
    print(f"Saved reference to {output_dir}")


def _parse_args():
    defaults = BagnoldParameters()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=float, default=defaults.height)
    parser.add_argument("--grain-diameter", type=float, default=defaults.grain_diameter)
    parser.add_argument("--grain-density", type=float, default=defaults.grain_density)
    parser.add_argument("--solid-fraction", type=float, default=defaults.solid_fraction)
    parser.add_argument("--gravity", type=float, default=defaults.gravity)
    parser.add_argument("--angle", type=float, default=defaults.chute_angle_deg)
    parser.add_argument(
        "--static-friction-angle",
        type=float,
        default=defaults.static_friction_angle_deg,
    )
    parser.add_argument(
        "--dynamic-friction-angle",
        type=float,
        default=defaults.dynamic_friction_angle_deg,
    )
    parser.add_argument("--I0", type=float, default=defaults.inertial_number_scale)
    parser.add_argument("--num-samples", type=int, default=201)
    parser.add_argument("--numerical-profile", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "bagnold_reference",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    parameters = BagnoldParameters(
        height=args.height,
        grain_diameter=args.grain_diameter,
        grain_density=args.grain_density,
        solid_fraction=args.solid_fraction,
        gravity=args.gravity,
        chute_angle_deg=args.angle,
        static_friction_angle_deg=args.static_friction_angle,
        dynamic_friction_angle_deg=args.dynamic_friction_angle,
        inertial_number_scale=args.I0,
    )
    write_reference(
        parameters,
        args.output_dir,
        num_samples=args.num_samples,
        numerical_profile=args.numerical_profile,
        write_plot=not args.no_plot,
    )
