"""Postprocess saved mu(I) chute-flow results against the Bagnold solution.

No simulation is executed here. The steady profiles are time averages of the
saved velocity snapshots over the quasi-steady plateaus identified from each
case's mean-velocity history. Later output is excluded because all three
histories subsequently develop systematic drift.
"""

# ruff: noqa: I001 -- matplotlib backend must be selected before pyplot import.

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from projects.granular_chute_flow.bagnold_reference import (
    BagnoldParameters,
    evaluate_bagnold,
)


ROOT = Path(__file__).resolve().parent
COMPARISON_DIR = ROOT / "mu_i_variant_comparison"
CASES = {
    "mu_i": {
        "label": "MuI_LC",
        "directory": ROOT / "mu_i",
        "steady_window": (3.0, 5.0),
    },
    "mu_i_regularized": {
        "label": "MuI_IC_regularized",
        "directory": ROOT / "mu_i_regularized",
        "steady_window": (8.0, 12.0),
    },
    "mu_i_incompressible": {
        "label": "MuI_IC",
        "directory": ROOT / "mu_i_incompressible",
        "steady_window": (8.0, 12.0),
    },
}


def load_history(case_dir: Path) -> np.ndarray:
    path = case_dir / "mean_velocity.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing velocity history: {path}")
    history = np.genfromtxt(path, delimiter=",", names=True)
    required = {"time", "mean_vx"}
    if history.dtype.names is None or not required.issubset(history.dtype.names):
        raise ValueError(f"{path} must contain the columns {sorted(required)}")
    return np.atleast_1d(history)


def ordered_profile_files(case_dir: Path) -> list[Path]:
    profiles = []
    for path in case_dir.glob("velocity_profile_*.csv"):
        step = path.stem.rsplit("_", maxsplit=1)[-1]
        if step.isdigit():
            profiles.append((int(step), path))
    return [path for _, path in sorted(profiles)]


def average_steady_profile(
    case_dir: Path,
    history: np.ndarray,
    steady_window: tuple[float, float],
) -> dict[str, np.ndarray | float | int]:
    profiles = ordered_profile_files(case_dir)
    if len(profiles) != len(history):
        raise ValueError(
            f"{case_dir} has {len(profiles)} profiles but {len(history)} "
            "mean-velocity samples; snapshots cannot be paired with times"
        )

    time = np.asarray(history["time"], dtype=float)
    output_interval = float(np.median(np.diff(time)))
    tolerance = 0.5 * output_interval + 1.0e-12
    start, end = steady_window
    indices = np.flatnonzero(
        (time >= start - tolerance) & (time <= end + tolerance)
    )
    if indices.size < 2:
        raise ValueError(
            f"steady window {steady_window} contains fewer than two snapshots"
        )

    velocity_profiles = []
    particle_counts = []
    y_reference = None
    for index in indices:
        snapshot = np.genfromtxt(
            profiles[index],
            delimiter=",",
            names=True,
        )
        names = set(snapshot.dtype.names or ())
        if not {"y", "mean_vx", "particle_count"}.issubset(names):
            raise ValueError(
                f"{profiles[index]} must contain y, mean_vx, and particle_count"
            )
        y = np.atleast_1d(snapshot["y"]).astype(float)
        if y_reference is None:
            y_reference = y
        elif not np.allclose(y, y_reference):
            raise ValueError(f"height bins changed in {profiles[index]}")
        velocity_profiles.append(
            np.atleast_1d(snapshot["mean_vx"]).astype(float)
        )
        particle_counts.append(
            np.atleast_1d(snapshot["particle_count"]).astype(float)
        )

    velocities = np.stack(velocity_profiles)
    counts = np.stack(particle_counts)
    finite = np.isfinite(velocities)
    samples_per_bin = np.sum(finite, axis=0)
    mean_velocity = np.divide(
        np.nansum(velocities, axis=0),
        samples_per_bin,
        out=np.full(samples_per_bin.shape, np.nan, dtype=float),
        where=samples_per_bin > 0,
    )
    mean_particle_count = np.mean(counts, axis=0)

    steady_history = np.asarray(history["mean_vx"][indices], dtype=float)
    steady_time = time[indices]
    slope = float(np.polyfit(steady_time, steady_history, 1)[0])
    history_mean = float(np.mean(steady_history))
    history_std = float(np.std(steady_history))
    relative_drift = (
        slope * (steady_time[-1] - steady_time[0]) / abs(history_mean)
    )

    return {
        "y": y_reference,
        "mean_velocity": mean_velocity,
        "mean_particle_count": mean_particle_count,
        "samples_per_bin": samples_per_bin,
        "actual_start_time": float(steady_time[0]),
        "actual_end_time": float(steady_time[-1]),
        "num_snapshots": int(indices.size),
        "history_mean_velocity": history_mean,
        "history_velocity_std": history_std,
        "history_slope": slope,
        "relative_linear_drift": float(relative_drift),
    }


def compare_with_bagnold(
    profile: dict[str, np.ndarray | float | int],
    reference: BagnoldParameters,
) -> dict[str, np.ndarray | float]:
    y = np.asarray(profile["y"])
    numerical = np.asarray(profile["mean_velocity"])
    weights = np.asarray(profile["mean_particle_count"])
    valid = (
        np.isfinite(y)
        & np.isfinite(numerical)
        & np.isfinite(weights)
        & (weights > 0.0)
        & (y >= 0.0)
        & (y <= reference.height)
    )
    y = y[valid]
    numerical = numerical[valid]
    weights = weights[valid]
    analytical = evaluate_bagnold(y, reference)["velocity"]
    residual = numerical - analytical

    relative_l2 = np.sqrt(
        np.sum(weights * residual**2) / np.sum(weights * analytical**2)
    )
    relative_linf = np.max(np.abs(residual)) / reference.surface_velocity
    return {
        "y": y,
        "numerical_velocity": numerical,
        "bagnold_velocity": analytical,
        "error": residual,
        "mean_particle_count": weights,
        "relative_l2_profile_error": float(relative_l2),
        "relative_linf_profile_error": float(relative_linf),
    }


def write_profile_csv(
    model: str,
    comparison: dict[str, np.ndarray | float],
) -> None:
    np.savetxt(
        COMPARISON_DIR / f"{model}_steady_profile.csv",
        np.column_stack(
            [
                comparison["y"],
                comparison["numerical_velocity"],
                comparison["bagnold_velocity"],
                comparison["error"],
                comparison["mean_particle_count"],
            ]
        ),
        delimiter=",",
        header=(
            "y,steady_mean_vx,bagnold_vx,error,mean_particle_count"
        ),
        comments="",
    )


def postprocess() -> list[dict[str, float | int | str]]:
    reference = BagnoldParameters()
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    histories = {}
    comparisons = {}

    for model, case in CASES.items():
        history = load_history(case["directory"])
        profile = average_steady_profile(
            case["directory"],
            history,
            case["steady_window"],
        )
        comparison = compare_with_bagnold(profile, reference)
        write_profile_csv(model, comparison)

        mean_error = (
            profile["history_mean_velocity"] - reference.mean_velocity
        ) / reference.mean_velocity
        summaries.append(
            {
                "model": model,
                "label": case["label"],
                "requested_start_time": case["steady_window"][0],
                "requested_end_time": case["steady_window"][1],
                "actual_start_time": profile["actual_start_time"],
                "actual_end_time": profile["actual_end_time"],
                "num_snapshots": profile["num_snapshots"],
                "steady_mean_velocity": profile["history_mean_velocity"],
                "bagnold_mean_velocity": reference.mean_velocity,
                "relative_mean_velocity_error": mean_error,
                "mean_velocity_std": profile["history_velocity_std"],
                "relative_mean_velocity_std": (
                    profile["history_velocity_std"]
                    / abs(profile["history_mean_velocity"])
                ),
                "mean_velocity_slope": profile["history_slope"],
                "relative_linear_drift": profile["relative_linear_drift"],
                "relative_l2_profile_error": comparison[
                    "relative_l2_profile_error"
                ],
                "relative_linf_profile_error": comparison[
                    "relative_linf_profile_error"
                ],
            }
        )
        histories[model] = history
        comparisons[model] = comparison

    with (COMPARISON_DIR / "comparison_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    plt.figure(figsize=(7, 4.5))
    for model, case in CASES.items():
        history = histories[model]
        line = plt.plot(
            history["time"],
            history["mean_vx"],
            linewidth=1.5,
            label=case["label"],
        )[0]
        start, end = case["steady_window"]
        plt.axvspan(start, end, color=line.get_color(), alpha=0.15)
    plt.axhline(
        reference.mean_velocity,
        color="k",
        linestyle="--",
        label="Bagnold mean",
    )
    plt.xlabel("time [s]")
    plt.ylabel("mean streamwise velocity [m/s]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        COMPARISON_DIR / "mean_velocity_history_with_steady_windows.png",
        dpi=180,
    )
    plt.close()

    plt.figure(figsize=(6.5, 4.5))
    y_reference = np.linspace(0.0, reference.height, 301)
    plt.plot(
        evaluate_bagnold(y_reference, reference)["velocity"],
        y_reference,
        "k--",
        linewidth=2,
        label="Bagnold",
    )
    for model, case in CASES.items():
        comparison = comparisons[model]
        start, end = case["steady_window"]
        plt.plot(
            comparison["numerical_velocity"],
            comparison["y"],
            "o-",
            label=f"{case['label']} ({start:g}-{end:g} s)",
        )
    plt.xlabel("streamwise velocity [m/s]")
    plt.ylabel("height above base [m]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / "steady_velocity_profile_comparison.png", dpi=180)
    plt.close()

    for summary in summaries:
        print(
            f"{summary['label']}: steady window "
            f"{summary['requested_start_time']:g}-"
            f"{summary['requested_end_time']:g} s, mean velocity="
            f"{summary['steady_mean_velocity']:.6f} m/s, "
            f"Bagnold mean error={summary['relative_mean_velocity_error']:.2%}, "
            f"profile L2 error={summary['relative_l2_profile_error']:.2%}, "
            f"linear drift={summary['relative_linear_drift']:.2%}"
        )
    print(f"Saved postprocessing results to {COMPARISON_DIR}")
    return summaries


if __name__ == "__main__":
    postprocess()
