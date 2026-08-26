"""Analyze saved production chute-flow results and generate manuscript figures.

This module is deliberately postprocessing-only. Production simulations are run
from ``chute_flow.py``; this script reads their saved CSV/NPZ and parameter files.
"""

# ruff: noqa: I001 -- matplotlib backend must be selected before pyplot import.

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from matplotlib import rcParams
from matplotlib.ticker import FormatStrFormatter, LinearLocator, MaxNLocator

import hydraxmpm as hdx

from projects.granular_chute_flow.bagnold_reference import (
    BagnoldParameters,
    evaluate_bagnold,
)
from projects.granular_chute_flow.chute_flow import (
    ChuteParameters,
)

PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parents[1]
RESULTS_DIR = PROJECT_DIR / "manuscript_results"
FIGURES_DIR = REPOSITORY_DIR / "CPC_manuscript" / "figures"
LOCAL_FIGURES_DIR = RESULTS_DIR / "figures"
AVERAGING_START = 8.0
AVERAGING_END = 12.0
PROFILE_FIGURE_WIDTH_IN = 7.0
MODEL_DIRECTORIES = {
    "mu_i_regularized": PROJECT_DIR / "mu_i_regularized",
    "drucker_prager": PROJECT_DIR / "drucker_prager",
}


@dataclass(frozen=True)
class CaseMetrics:
    surface_velocity: float
    mean_velocity: float
    relative_drift: float
    relative_surface_error: float


def load_csv(path: Path) -> np.ndarray:
    """Load a named-column CSV and always return a one-dimensional array."""
    if not path.exists():
        raise FileNotFoundError(f"missing postprocessing input: {path}")
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.dtype.names is None:
        raise ValueError(f"CSV has no named columns: {path}")
    return np.atleast_1d(data)


def load_particle_snapshot(path: Path) -> dict[str, np.ndarray]:
    """Load a raw particle snapshot written by ``chute_flow.py``."""
    required = {"step", "time", "position", "velocity", "mass", "density", "stress"}
    with np.load(path) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"particle snapshot {path} is missing {sorted(missing)}")
        return {name: np.asarray(archive[name]) for name in required}


def compute_velocity_profile(
    position: np.ndarray,
    velocity: np.ndarray,
    parameters: ChuteParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin particle streamwise velocity in the chute-normal direction."""
    n_bins = int(round(parameters.domain_height / parameters.particle_spacing))
    edges = np.linspace(
        parameters.base_y,
        parameters.base_y + parameters.domain_height,
        n_bins + 1,
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(position[:, 1], bins=edges)
    weighted, _ = np.histogram(position[:, 1], bins=edges, weights=velocity[:, 0])
    mean_vx = np.full(centers.shape, np.nan)
    np.divide(weighted, counts, out=mean_vx, where=counts > 0)
    return centers, mean_vx, counts


def compute_grid_fields(
    snapshot: dict[str, np.ndarray],
    parameters: ChuteParameters,
) -> dict[str, np.ndarray]:
    """Mass-weight a raw particle snapshot onto the background grid."""
    domain = hdx.GridDomain.create(
        parameters.origin,
        parameters.end,
        parameters.cell_size,
        padding=3,
        periodic_axes=(True, False),
    )
    mapping = hdx.ShapeFunctionMapping(
        "quadratic",
        dim=2,
        periodic_axes=domain.periodic_axes,
    )
    position_stack = jnp.asarray(snapshot["position"])
    mass_stack = jnp.asarray(snapshot["mass"])
    cache = mapping.compute(
        position_stack,
        domain.origin,
        domain.grid_size,
        domain._inv_cell_size,
    )
    nodal_mass = mapping.scatter_to_grid(
        cache,
        jnp.ones_like(mass_stack),
        mass_stack,
        domain.num_cells,
        normalize=False,
    )
    volume_fraction = mapping.scatter_to_grid(
        cache,
        jnp.asarray(snapshot["density"]) / parameters.grain_density,
        mass_stack,
        domain.num_cells,
    )
    velocity = mapping.scatter_to_grid(
        cache,
        jnp.asarray(snapshot["velocity"]),
        mass_stack,
        domain.num_cells,
    )
    stress = mapping.scatter_to_grid(
        cache,
        jnp.asarray(snapshot["stress"]),
        mass_stack,
        domain.num_cells,
    )
    grid_position = np.asarray(domain.position_stack)
    physical_node = (
        (grid_position[:, 0] >= parameters.periodic_x_min)
        & (grid_position[:, 0] < parameters.periodic_x_max)
        & (grid_position[:, 1] >= parameters.base_y)
        & (grid_position[:, 1] < parameters.base_y + parameters.domain_height)
    )
    return {
        "position": grid_position,
        "physical_node": physical_node,
        "mass": np.asarray(nodal_mass),
        "volume_fraction": np.asarray(volume_fraction),
        "velocity": np.asarray(velocity),
        "stress": np.asarray(stress),
    }


def compute_time_averaged_profile(
    profiles: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Average supported velocity bins over saved steady snapshots."""
    if not profiles:
        return np.array([]), np.array([])
    y = np.asarray(profiles[0][0], dtype=float)
    stacked_vx = np.stack(
        [np.asarray(profile[1], dtype=float) for profile in profiles], axis=0
    )
    valid_counts = np.sum(np.isfinite(stacked_vx), axis=0)
    mean_vx = np.full(y.shape, np.nan)
    np.divide(
        np.nansum(stacked_vx, axis=0),
        valid_counts,
        out=mean_vx,
        where=valid_counts > 0,
    )
    return y, mean_vx


def compute_time_averaged_grid_fields(
    samples: list[dict[str, np.ndarray]],
    small_mass_cutoff: float = 1.0e-9,
) -> dict[str, np.ndarray] | None:
    """Average nodal fields only while a physical node has particle support."""
    if not samples:
        return None
    position = np.asarray(samples[0]["position"], dtype=float)
    physical_node = np.asarray(samples[0]["physical_node"], dtype=bool)
    mass_stack = np.stack([sample["mass"] for sample in samples])
    supported = (mass_stack > small_mass_cutoff) & physical_node[None, :]
    support_count = np.sum(supported, axis=0)

    def supported_mean(name: str) -> np.ndarray:
        values = np.stack([sample[name] for sample in samples])
        expanded_support = supported.reshape(
            supported.shape + (1,) * (values.ndim - supported.ndim)
        )
        summed = np.sum(np.where(expanded_support, values, 0.0), axis=0)
        denominator = support_count.reshape(
            support_count.shape + (1,) * (summed.ndim - support_count.ndim)
        )
        result = np.full(summed.shape, np.nan, dtype=float)
        np.divide(summed, denominator, out=result, where=denominator > 0)
        return result

    return {
        "position": position,
        "physical_node": physical_node,
        "support_fraction": support_count / len(samples),
        "mass": np.where(physical_node, np.mean(mass_stack, axis=0), np.nan),
        "volume_fraction": supported_mean("volume_fraction"),
        "velocity": supported_mean("velocity"),
        "stress": supported_mean("stress"),
    }


def save_velocity_profile(
    path: Path,
    y: np.ndarray,
    velocity: np.ndarray,
    counts: np.ndarray | None = None,
) -> None:
    """Save a derived velocity profile with optional particle support counts."""
    columns = [y, velocity]
    header = "y,mean_vx"
    if counts is not None:
        columns.append(counts)
        header += ",particle_count"
    np.savetxt(
        path,
        np.column_stack(columns),
        delimiter=",",
        header=header,
        comments="",
    )


def save_grid_fields(output_dir: Path, fields: dict[str, np.ndarray]) -> None:
    """Save reusable and human-readable steady grid-field products."""
    position = fields["position"]
    velocity = fields["velocity"]
    stress = fields["stress"]
    columns = np.column_stack(
        [
            position[:, 0],
            position[:, 1],
            fields["physical_node"].astype(int),
            fields["support_fraction"],
            fields["mass"],
            fields["volume_fraction"],
            velocity[:, 0],
            velocity[:, 1],
            np.linalg.norm(velocity, axis=1),
            stress[:, 1, 1],
            stress[:, 1, 0],
        ]
    )
    np.savetxt(
        output_dir / "steady_grid_fields.csv",
        columns,
        delimiter=",",
        header=(
            "x,y,physical_node,support_fraction,nodal_mass,volume_fraction,velocity_x,"
            "velocity_y,speed,sigma_yy,sigma_yx"
        ),
        comments="",
    )
    np.savez_compressed(output_dir / "steady_grid_fields.npz", **fields)


def postprocess_case(
    output_dir: Path,
    parameters: ChuteParameters,
    averaging_start: float = AVERAGING_START,
    averaging_end: float = AVERAGING_END,
) -> bool:
    """Generate histories and steady-window averages from raw particle snapshots.

    Returns ``False`` for legacy result folders that contain only the already
    derived products, and ``True`` when raw snapshots were processed.
    """
    snapshot_paths = sorted((output_dir / "particle_states").glob("step_*.npz"))
    if not snapshot_paths:
        legacy_products = (
            output_dir / "mean_velocity.csv",
            output_dir / "steady_velocity_profile.csv",
            output_dir / "steady_grid_fields.csv",
            output_dir / "steady_grid_fields.npz",
        )
        if all(path.exists() for path in legacy_products):
            return False
        raise FileNotFoundError(f"no raw particle snapshots found in {output_dir}")

    history = []
    steady_profiles = []
    steady_grid_samples = []
    final_profile = None

    for path in snapshot_paths:
        snapshot = load_particle_snapshot(path)
        step = int(snapshot["step"].item())
        time = step * parameters.dt
        velocity = snapshot["velocity"]
        history.append(
            (
                time,
                float(np.mean(np.linalg.norm(velocity, axis=1))),
                float(np.mean(velocity[:, 0])),
            )
        )
        profile = compute_velocity_profile(
            snapshot["position"], snapshot["velocity"], parameters
        )
        save_velocity_profile(output_dir / f"velocity_profile_{step:05d}.csv", *profile)
        final_profile = profile
        if averaging_start - 1.0e-6 <= time <= averaging_end + 1.0e-6:
            steady_profiles.append(profile)
            steady_grid_samples.append(compute_grid_fields(snapshot, parameters))

    history.sort(key=lambda row: row[0])
    np.savetxt(
        output_dir / "mean_velocity.csv",
        np.asarray(history),
        delimiter=",",
        header="time,mean_speed,mean_vx",
        comments="",
    )
    if final_profile is not None:
        save_velocity_profile(output_dir / "final_velocity_profile.csv", *final_profile)

    average_y, average_vx = compute_time_averaged_profile(steady_profiles)
    if not average_y.size:
        raise ValueError(
            f"no snapshots fall in the steady window "
            f"{averaging_start:g}--{averaging_end:g} s"
        )
    save_velocity_profile(
        output_dir / "steady_velocity_profile.csv", average_y, average_vx
    )
    averaged_fields = compute_time_averaged_grid_fields(steady_grid_samples)
    if averaged_fields is None:
        raise ValueError(f"no grid fields available in the steady window: {output_dir}")
    save_grid_fields(output_dir, averaged_fields)
    return True


def load_and_validate_parameters(
    output_dir: Path,
    model: str,
) -> ChuteParameters:
    """Require saved parameters to match the fixed production configuration."""
    path = output_dir / "parameters.json"
    if not path.exists():
        raise FileNotFoundError(f"missing production parameters: {path}")
    saved = json.loads(path.read_text(encoding="utf-8"))
    legacy_steady_fraction = saved.pop("steady_start_fraction", None)
    if legacy_steady_fraction is not None and not np.isclose(
        legacy_steady_fraction * saved["total_time"], AVERAGING_START
    ):
        raise ValueError(
            f"{model} results do not use the {AVERAGING_START:g} s averaging start"
        )
    if saved.get("total_time", 0.0) < AVERAGING_END:
        raise ValueError(
            f"{model} ends before the {AVERAGING_END:g} s averaging-window limit"
        )
    parameters = ChuteParameters(**saved)
    expected_payload = asdict(ChuteParameters(constitutive_model=model))
    compared_names = set(expected_payload) - {"total_time"}
    if any(saved.get(name) != expected_payload[name] for name in compared_names):
        differences = {
            name: (saved.get(name, "<missing>"), expected_value)
            for name, expected_value in expected_payload.items()
            if name in compared_names and saved.get(name, "<missing>") != expected_value
        }
        extras = sorted(set(saved) - set(expected_payload))
        raise ValueError(
            f"{model} results do not use the production parameters; "
            f"differences={differences}, extra_keys={extras}"
        )
    return parameters


def characterize_case(
    output_dir: Path,
    parameters: ChuteParameters,
    target_surface_velocity: float,
) -> CaseMetrics:
    """Measure surface speed and linear drift over the saved steady window."""
    profile = load_csv(output_dir / "steady_velocity_profile.csv")
    history = load_csv(output_dir / "mean_velocity.csv")
    profile_names = set(profile.dtype.names or ())
    history_names = set(history.dtype.names or ())
    if not {"y", "mean_vx"}.issubset(profile_names):
        raise ValueError(f"invalid steady profile in {output_dir}")
    if not {"time", "mean_vx"}.issubset(history_names):
        raise ValueError(f"invalid velocity history in {output_dir}")

    y = np.asarray(profile["y"], dtype=float)
    velocity = np.asarray(profile["mean_vx"], dtype=float)
    profile_mask = (
        np.isfinite(y)
        & np.isfinite(velocity)
        & (y >= parameters.base_y)
        & (y <= parameters.base_y + parameters.fill_depth)
    )
    time = np.asarray(history["time"], dtype=float)
    mean_vx = np.asarray(history["mean_vx"], dtype=float)
    history_mask = (
        np.isfinite(time)
        & np.isfinite(mean_vx)
        & (time >= AVERAGING_START - 1.0e-6)
        & (time <= AVERAGING_END + 1.0e-6)
    )
    if not np.any(profile_mask) or np.count_nonzero(history_mask) < 2:
        raise ValueError(f"insufficient finite steady-state data in {output_dir}")

    surface_velocity = float(np.max(velocity[profile_mask]))
    window_time = time[history_mask]
    window_velocity = mean_vx[history_mask]
    mean_velocity = float(np.mean(window_velocity))
    slope = float(np.polyfit(window_time, window_velocity, 1)[0])
    relative_drift = abs(slope * (AVERAGING_END - AVERAGING_START)) / max(
        abs(mean_velocity), 1.0e-12
    )
    relative_surface_error = (
        surface_velocity - target_surface_velocity
    ) / target_surface_velocity
    metrics = np.asarray(
        [surface_velocity, mean_velocity, relative_drift, relative_surface_error]
    )
    if not np.all(np.isfinite(metrics)):
        raise ValueError(f"non-finite steady-state metrics in {output_dir}")
    return CaseMetrics(*metrics)


def analyze_grid_fields(output_dir: Path, model: str) -> dict[str, float]:
    """Validate supported nodal fields and return compact field diagnostics."""
    fields = load_csv(output_dir / "steady_grid_fields.csv")
    required = {
        "physical_node",
        "support_fraction",
        "volume_fraction",
        "velocity_x",
        "velocity_y",
        "sigma_yy",
        "sigma_yx",
    }
    if not required.issubset(fields.dtype.names or ()):
        raise ValueError(f"{model} grid fields are missing {sorted(required)}")
    supported = np.asarray(fields["physical_node"], dtype=bool) & (
        fields["support_fraction"] > 0.0
    )
    if not np.any(supported):
        raise ValueError(f"{model} has no supported physical grid nodes")
    field_names = required - {"physical_node", "support_fraction"}
    if not all(np.all(np.isfinite(fields[name][supported])) for name in field_names):
        raise ValueError(f"{model} has non-finite supported grid fields")
    npz_path = output_dir / "steady_grid_fields.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"missing reusable grid archive: {npz_path}")
    return {
        "mean_supported_volume_fraction": float(
            np.mean(fields["volume_fraction"][supported])
        ),
        "nonnegative_sigma_yy_fraction": float(
            np.mean(fields["sigma_yy"][supported] >= -1.0e-5)
        ),
    }


def make_bagnold_parameters(parameters: ChuteParameters) -> BagnoldParameters:
    """Build the analytical reference from the production material parameters."""
    return BagnoldParameters(
        height=parameters.fill_depth,
        grain_diameter=0.002,
        grain_density=parameters.grain_density,
        solid_fraction=parameters.initial_solid_fraction,
        chute_angle_deg=parameters.chute_angle_deg,
        static_friction_angle_deg=parameters.friction_angle_deg,
        dynamic_friction_angle_deg=parameters.dynamic_friction_angle_deg,
        inertial_number_scale=0.35,
    )


def analyze_results(
    parameters: dict[str, ChuteParameters],
) -> dict[str, float]:
    """Validate both production cases and write manuscript acceptance metrics."""
    reference = make_bagnold_parameters(parameters["mu_i_regularized"])
    case_metrics = {
        model: characterize_case(
            output_dir,
            parameters[model],
            reference.surface_velocity,
        )
        for model, output_dir in MODEL_DIRECTORIES.items()
    }
    grid_metrics = {
        model: analyze_grid_fields(output_dir, model)
        for model, output_dir in MODEL_DIRECTORIES.items()
    }
    checks = {
        "mu_relative_drift": case_metrics["mu_i_regularized"].relative_drift,
        "mu_relative_surface_error": case_metrics[
            "mu_i_regularized"
        ].relative_surface_error,
        "dp_relative_drift": case_metrics["drucker_prager"].relative_drift,
        "dp_relative_surface_error": case_metrics[
            "drucker_prager"
        ].relative_surface_error,
        "mu_mean_supported_volume_fraction": grid_metrics["mu_i_regularized"][
            "mean_supported_volume_fraction"
        ],
        "mu_nonnegative_sigma_yy_fraction": grid_metrics["mu_i_regularized"][
            "nonnegative_sigma_yy_fraction"
        ],
        "dp_mean_supported_volume_fraction": grid_metrics["drucker_prager"][
            "mean_supported_volume_fraction"
        ],
        "dp_nonnegative_sigma_yy_fraction": grid_metrics["drucker_prager"][
            "nonnegative_sigma_yy_fraction"
        ],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "acceptance_metrics.csv").open("w", encoding="utf-8") as stream:
        stream.write("metric,value\n")
        for name, value in checks.items():
            stream.write(f"{name},{value:.16e}\n")

    if checks["mu_relative_drift"] > 0.01:
        raise AssertionError("MuI_regularized drifts by more than 1% over 8--12 s")
    if checks["dp_relative_drift"] > 0.01:
        raise AssertionError("Drucker-Prager drifts by more than 1% over 8--12 s")
    if abs(checks["mu_relative_surface_error"]) > 0.01:
        raise AssertionError(
            "MuI_regularized surface speed differs from Bagnold by >1%"
        )
    if abs(checks["mu_mean_supported_volume_fraction"] - 0.60) > 0.01:
        raise AssertionError("MuI_regularized volume fraction is not near 0.60")
    if (
        min(
            checks["mu_nonnegative_sigma_yy_fraction"],
            checks["dp_nonnegative_sigma_yy_fraction"],
        )
        < 0.99
    ):
        raise AssertionError("compression-positive sigma_yy has inconsistent signs")
    return checks


def configure_plotting() -> None:
    rcParams["mathtext.fontset"] = "cm"
    rcParams["font.family"] = "serif"
    rcParams["font.size"] = 14
    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "axes.labelsize": 11,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def save_figure(fig: plt.Figure, filename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    fig.savefig(LOCAL_FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def grid_depth_profile(
    fields: np.ndarray,
    field_name: str,
    parameters: ChuteParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Average one projected nodal field across periodic x nodes at each depth."""
    surface_y = parameters.base_y + parameters.fill_depth
    supported = (
        np.asarray(fields["physical_node"], dtype=bool)
        & (fields["support_fraction"] > 0.0)
        & (fields["y"] >= parameters.base_y - 1.0e-9)
        & (fields["y"] <= surface_y + 1.0e-9)
        & np.isfinite(fields[field_name])
    )
    if not np.any(supported):
        raise ValueError(f"no supported values available for {field_name}")
    supported_y = np.asarray(fields["y"], dtype=float)[supported]
    supported_values = np.asarray(fields[field_name], dtype=float)[supported]
    y = np.unique(supported_y)
    values = np.asarray(
        [np.mean(supported_values[np.isclose(supported_y, level)]) for level in y]
    )
    return y, values


def plot_depth_profiles(
    output_dir: Path,
    parameters: ChuteParameters,
    model: str,
    filename: str,
    volume_fraction_xlim: tuple[float, float],
) -> None:
    """Plot depth-resolved stress, packing, and velocity in a one-row panel."""
    fields = load_csv(output_dir / "steady_grid_fields.csv")
    surface_y = parameters.base_y + parameters.fill_depth
    y, sigma_yy = grid_depth_profile(fields, "sigma_yy", parameters)
    _, sigma_yx = grid_depth_profile(fields, "sigma_yx", parameters)
    _, volume_fraction = grid_depth_profile(fields, "volume_fraction", parameters)
    _, velocity = grid_depth_profile(fields, "velocity_x", parameters)
    sigma_yy /= 1.0e3
    sigma_yx /= 1.0e3

    if model == "mu_i_regularized":
        reference = make_bagnold_parameters(parameters)
        reference_depth = np.linspace(0.0, reference.height, 401)
        y_reference = parameters.base_y + reference_depth
        v_reference = evaluate_bagnold(reference_depth, reference)["velocity"]
        axes_3_ref = (v_reference, y_reference)
    else:
        axes_3_ref = None

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(PROFILE_FIGURE_WIDTH_IN, 2.6),
        sharey=True,
    )
    axes[0].plot(sigma_yy, y, color="k", linewidth=1.0)
    axes[0].set_xlabel(r"$\sigma_{yy}$ [kPa]")
    axes[0].set_ylabel(r"$y$ [m]")
    axes[0].grid(True, linestyle="--", linewidth=0.7)

    axes[1].plot(sigma_yx, y, color="k", linewidth=1.0)
    axes[1].set_xlabel(r"$\tau_{yx}$ [kPa]")
    axes[1].set_ylabel("")
    axes[1].grid(True, linestyle="--", linewidth=0.7)

    axes[2].plot(volume_fraction, y, color="k", linewidth=1.0)
    axes[2].set_xlabel(r"$\phi$ [-]")
    axes[2].set_ylabel("")
    axes[2].set_xlim(volume_fraction_xlim)
    axes[2].grid(True, linestyle="--", linewidth=0.7)

    model_label = r"$\mu(I)$" if model == "mu_i_regularized" else None
    axes[3].plot(
        velocity,
        y,
        color="k",
        linewidth=1.0,
        label=model_label,
    )
    if axes_3_ref is not None:
        axes[3].plot(
            axes_3_ref[0],
            axes_3_ref[1],
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Bagnold",
        )
    axes[3].set_xlabel(r"$v_x$ [m s$^{-1}$]")
    axes[3].set_ylabel("")
    axes[3].grid(True, linestyle="--", linewidth=0.7)
    if axes_3_ref is not None:
        axes[3].legend(
            frameon=False,
            loc="upper left",
            fontsize=8,
            handlelength=1.5,
        )

    for ax in axes:
        ax.set_ylim(parameters.base_y, surface_y)
        ax.set_frame_on(True)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.tick_params(axis="x", labelsize=9)
    axes[2].xaxis.set_major_locator(LinearLocator(2))
    axes[2].xaxis.set_major_formatter(FormatStrFormatter("%.4f"))

    fig.tight_layout()
    save_figure(fig, filename)


def shared_volume_fraction_limits(
    parameters: dict[str, ChuteParameters],
) -> tuple[float, float]:
    """Return one padded packing-fraction range for both depth-profile panels."""
    values = []
    for model, output_dir in MODEL_DIRECTORIES.items():
        fields = load_csv(output_dir / "steady_grid_fields.csv")
        surface_y = parameters[model].base_y + parameters[model].fill_depth
        supported = (
            np.asarray(fields["physical_node"], dtype=bool)
            & (fields["support_fraction"] > 0.0)
            & (fields["y"] <= surface_y + 1.0e-9)
            & np.isfinite(fields["volume_fraction"])
        )
        values.append(np.asarray(fields["volume_fraction"], dtype=float)[supported])
    combined = np.concatenate(values)
    lower = float(np.min(combined))
    upper = float(np.max(combined))
    padding = max(0.05 * (upper - lower), 5.0e-6)
    return lower - padding, upper + padding


def plot_layout(parameters: ChuteParameters) -> None:
    """Plot the initial material points and physical background grid."""
    particles_per_axis = int(round(np.sqrt(parameters.ppc)))
    spacing = parameters.cell_size / particles_per_axis
    stream = np.arange(
        parameters.periodic_x_min + 0.5 * spacing,
        parameters.periodic_x_max,
        spacing,
    )
    normal = np.arange(
        parameters.base_y + 0.5 * spacing,
        parameters.base_y + parameters.fill_depth,
        spacing,
    )
    xx, yy = np.meshgrid(stream, normal, indexing="xy")
    particles = np.column_stack([xx.ravel(), yy.ravel()])
    grid_x = np.arange(
        parameters.periodic_x_min,
        parameters.periodic_x_max + 0.5 * parameters.cell_size,
        parameters.cell_size,
    )
    grid_y = np.arange(
        parameters.base_y,
        parameters.base_y + parameters.domain_height + 0.5 * parameters.cell_size,
        parameters.cell_size,
    )
    np.savetxt(
        RESULTS_DIR / "initial_material_points.csv",
        particles,
        delimiter=",",
        header="x,y",
        comments="",
    )
    np.savetxt(
        RESULTS_DIR / "physical_background_grid.csv",
        np.asarray([(x, y) for x in grid_x for y in grid_y]),
        delimiter=",",
        header="x,y",
        comments="",
    )
    fig, ax = plt.subplots(figsize=(3.35, 3.0))
    ax.vlines(
        grid_x,
        parameters.base_y,
        parameters.base_y + parameters.domain_height,
        color="0.82",
        linewidth=0.55,
    )
    ax.hlines(
        grid_y,
        parameters.periodic_x_min,
        parameters.periodic_x_max,
        color="0.82",
        linewidth=0.55,
    )
    ax.scatter(
        particles[:, 0],
        particles[:, 1],
        s=5,
        color="#1f77b4",
        label="Material\npoints",
        zorder=3,
    )
    ax.plot([], [], color="0.65", linewidth=0.8, label="Grid")
    ax.axhline(parameters.base_y, color="black", linewidth=1.4)
    ax.set(
        xlabel=r"$x$ [m]",
        ylabel=r"$y$ [m]",
        xlim=(parameters.periodic_x_min, parameters.periodic_x_max),
        ylim=(parameters.base_y - 0.003, parameters.base_y + parameters.domain_height),
    )
    ax.set_aspect("equal")
    ax.legend(loc="upper right", frameon=False)
    save_figure(fig, "chute_layout.png")


def structured_field(
    data: np.ndarray,
    name: str,
    periodic_x_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reshape a supported nodal column and close the periodic x boundary."""
    physical = np.asarray(data["physical_node"], dtype=bool)
    x = np.unique(np.asarray(data["x"])[physical])
    y = np.unique(np.asarray(data["y"])[physical])
    field = np.full((y.size, x.size), np.nan)
    for row in np.atleast_1d(data):
        if not bool(row["physical_node"]) or row["support_fraction"] <= 0.0:
            continue
        ix = int(np.argmin(abs(x - row["x"])))
        iy = int(np.argmin(abs(y - row["y"])))
        field[iy, ix] = row[name]
    x = np.append(x, periodic_x_max)
    field = np.column_stack([field, field[:, 0]])
    return x, y, field


def plot_velocity_field(
    data: np.ndarray,
    parameters: ChuteParameters,
    filename: str,
) -> None:
    x, y, speed = structured_field(data, "speed", parameters.periodic_x_max)
    _, _, velocity_x = structured_field(data, "velocity_x", parameters.periodic_x_max)
    _, _, velocity_y = structured_field(data, "velocity_y", parameters.periodic_x_max)
    xx, yy = np.meshgrid(x, y)
    fig, ax = plt.subplots(figsize=(3.35, 3.0))
    image = ax.pcolormesh(x, y, speed, shading="nearest", cmap="gray")
    vectors = (
        np.isfinite(velocity_x)
        & np.isfinite(velocity_y)
        & (yy <= parameters.base_y + parameters.fill_depth + 1.0e-9)
    )
    vector_speed = np.hypot(velocity_x[vectors], velocity_y[vectors])
    maximum_arrow_length = 0.015
    arrow_scale = float(np.max(vector_speed)) / maximum_arrow_length
    ax.quiver(
        xx[vectors],
        yy[vectors],
        velocity_x[vectors],
        velocity_y[vectors],
        color="#d62728",
        angles="xy",
        scale_units="xy",
        scale=arrow_scale,
        width=0.008,
        headwidth=3.0,
    )
    fig.colorbar(image, ax=ax, label=r"speed $|\mathbf{v}|$ [m s$^{-1}$]", pad=0.03)
    ax.set(
        xlabel=r"$x$ [m]",
        ylabel=r"$y$ [m]",
        xlim=(parameters.periodic_x_min, parameters.periodic_x_max),
        ylim=(parameters.base_y, parameters.base_y + parameters.domain_height),
    )
    ax.set_aspect("equal")
    ax.grid(False)
    save_figure(fig, filename)


def generate_figures(parameters: dict[str, ChuteParameters]) -> None:
    """Generate all manuscript figures from existing production outputs."""
    configure_plotting()
    plot_layout(parameters["mu_i_regularized"])
    for model in ("mu_i_regularized", "drucker_prager"):
        fields = load_csv(MODEL_DIRECTORIES[model] / "steady_grid_fields.csv")
        plot_velocity_field(
            fields,
            parameters[model],
            f"chute_{model}_velocity_field.png",
        )
    volume_fraction_xlim = shared_volume_fraction_limits(parameters)
    for model in ("mu_i_regularized", "drucker_prager"):
        plot_depth_profiles(
            MODEL_DIRECTORIES[model],
            parameters[model],
            model,
            f"chute_{model}_depth_profiles.png",
            volume_fraction_xlim,
        )


def main() -> None:
    parameters = {
        model: load_and_validate_parameters(output_dir, model)
        for model, output_dir in MODEL_DIRECTORIES.items()
    }
    for model, output_dir in MODEL_DIRECTORIES.items():
        postprocess_case(output_dir, parameters[model])
    metrics = analyze_results(parameters)
    generate_figures(parameters)
    print(
        "Postprocessed fixed production results: "
        f"mu drift={metrics['mu_relative_drift']:.3%}, "
        f"DP drift={metrics['dp_relative_drift']:.3%}"
    )


if __name__ == "__main__":
    main()
