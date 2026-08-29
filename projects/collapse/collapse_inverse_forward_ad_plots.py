"""Plots collapse forward-AD inverse analysis."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

import hydraxmpm as hdx
from projects.collapse.collapse import (
    CELL_SIZE,
    END,
    ORIGIN,
    simulate_collapse,
)

FIGSIZE = (11.69 / 3, 8.27 / 3)
FIGURES_DIR = Path(__file__).resolve().parent / "output" / "figures"
FIELD_ERROR_FLOOR = 1.0e-12


def configure_plotting() -> None:
    """Apply the common project plotting style."""
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


def save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    """Save one 300-dpi figure and close its Matplotlib resources."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_loss_history(
    loss_history: Sequence[float],
    *,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Plot normalized inverse loss against optimizer iteration."""
    loss = np.asarray(loss_history, dtype=float)
    if loss.ndim != 1 or loss.size == 0:
        raise ValueError("loss_history must be a nonempty one-dimensional sequence")

    iterations = np.arange(loss.size)
    positive_loss = np.maximum(loss, np.finfo(float).tiny)
    fig, axis = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    axis.semilogy(
        iterations,
        positive_loss,
        color="black",
        linewidth=1.2,
        marker="o",
        markersize=3.5,
        markerfacecolor="white",
    )
    axis.set(
        xlabel="Iteration [-]",
        ylabel=r"Normalized loss $L$ [-]",
        xlim=(0, max(loss.size - 1, 1)),
    )
    return save_figure(fig, output_dir, "collapse_inverse_loss.png")


def plot_parameter_history(
    friction_angle_history: Sequence[float],
    *,
    reference_friction_angle: float,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Plot the identified friction-angle trajectory."""
    friction_angle = np.asarray(friction_angle_history, dtype=float)
    if friction_angle.ndim != 1 or friction_angle.size < 2:
        raise ValueError(
            "friction_angle_history must contain the initial value and one update"
        )

    iterations = np.arange(friction_angle.size)
    fig, axis = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    axis.plot(
        iterations,
        friction_angle,
        color="#0072B2",
        linewidth=1.2,
        marker="o",
        markersize=3.5,
        markerfacecolor="white",
        label="Inferred",
    )
    axis.axhline(
        reference_friction_angle,
        color="darkred",
        linestyle="--",
        linewidth=1.0,
        label="Reference",
    )
    axis.set(
        xlabel="Iteration [-]",
        ylabel=r"Friction angle $\phi$ [$^\circ$]",
        xlim=(0, friction_angle.size - 1),
    )
    axis.legend(frameon=False)
    return save_figure(fig, output_dir, "collapse_inverse_parameter.png")


def project_volume_fraction(
    final_state: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project particle volumes with HydraxMPM's quadratic P2G mapping."""
    material_points = final_state.world.material_points[0]
    domain = hdx.GridDomain.create(ORIGIN, END, CELL_SIZE, padding=0)
    mapping = hdx.ShapeFunctionMapping("quadratic", dim=2)
    cache = mapping.compute(
        material_points.position_stack[:, :2],
        domain.origin,
        domain.grid_size,
        domain._inv_cell_size,
    )
    nodal_volume = mapping.scatter_to_grid(
        cache,
        jnp.ones_like(material_points.volume_stack),
        material_points.volume_stack,
        domain.num_cells,
        normalize=False,
    )
    volume_fraction = nodal_volume / CELL_SIZE**2
    volume_fraction = volume_fraction / jnp.maximum(
        jnp.max(volume_fraction),
        FIELD_ERROR_FLOOR,
    )
    position_mesh = domain.position_mesh
    return (
        np.asarray(jax.device_get(position_mesh[..., 0])),
        np.asarray(jax.device_get(position_mesh[..., 1])),
        np.asarray(jax.device_get(volume_fraction.reshape(domain.grid_size))),
    )


def simulate_volume_fraction(
    friction_angle: float,
    *,
    cohesion: float,
    num_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run one collapse and return its field and runout extent."""
    result = simulate_collapse(
        fric_angle=friction_angle,
        c0=cohesion,
        num_steps=num_steps,
        compute_local=False,
        return_final_state=True,
    )
    final_state = result["final_state"]
    runout_distance = result["global"]["final_runout_distance"]
    right_extent = float(jax.device_get(2.0 * CELL_SIZE + runout_distance))
    return (*project_volume_fraction(final_state), right_extent)


def plot_volume_fraction_error(
    identified_friction_angle: float,
    *,
    reference_friction_angle: float,
    cohesion: float,
    num_steps: int,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Plot ground-truth, identified, and signed-error volume-fraction fields."""
    x_ref, y_ref, reference, reference_extent = simulate_volume_fraction(
        reference_friction_angle,
        cohesion=cohesion,
        num_steps=num_steps,
    )
    x_identified, y_identified, identified, identified_extent = simulate_volume_fraction(
        identified_friction_angle,
        cohesion=cohesion,
        num_steps=num_steps,
    )
    if reference.shape != identified.shape:
        raise ValueError("reference and identified volume-fraction grids differ")
    if not (
        np.array_equal(x_ref, x_identified)
        and np.array_equal(y_ref, y_identified)
    ):
        raise ValueError("reference and identified grid coordinates differ")

    crop_extent = min(END[0], max(reference_extent, identified_extent))
    crop_mask = x_ref <= crop_extent
    error = identified - reference
    cropped_error = error[crop_mask]
    cropped_reference = reference[crop_mask]
    cropped_identified = identified[crop_mask]
    relative_l2_error = np.linalg.norm(cropped_error) / max(
        np.linalg.norm(cropped_reference),
        FIELD_ERROR_FLOOR,
    )
    field_limit = max(
        float(np.max(cropped_reference)),
        float(np.max(cropped_identified)),
        FIELD_ERROR_FLOOR,
    )
    error_limit = max(float(np.max(np.abs(cropped_error))), FIELD_ERROR_FLOOR)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=[FIGSIZE[0] * 3, FIGSIZE[1] * 2],
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    reference_map = axes[0].pcolormesh(
        x_ref,
        y_ref,
        reference,
        shading="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=field_limit,
    )
    axes[1].pcolormesh(
        x_ref,
        y_ref,
        identified,
        shading="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=field_limit,
    )
    error_map = axes[2].pcolormesh(
        x_ref,
        y_ref,
        error,
        shading="nearest",
        cmap="RdBu_r",
        vmin=-error_limit,
        vmax=error_limit,
    )
    axes[0].set_title("Ground truth", fontsize=11)
    axes[1].set_title("Inferred", fontsize=11)
    axes[2].set_title(
        rf"Error ($L_2={relative_l2_error:.2e}$)",
        fontsize=11,
    )
    for axis in axes:
        axis.set(
            xlabel=r"$x$ [m]",
            xlim=(ORIGIN[0], crop_extent),
            ylim=(ORIGIN[1], END[1]),
        )
        axis.set_aspect("equal")
        axis.grid(False)
    axes[0].set_ylabel(r"$y$ [m]")

    field_colorbar = fig.colorbar(
        reference_map,
        ax=axes[:2],
        orientation="horizontal",
        pad=0.08,
        fraction=0.12,
        aspect=30,
    )
    field_colorbar.set_label(r"Volume fraction $\varphi$ [-]", fontsize=9)
    field_colorbar.ax.tick_params(labelsize=8)
    error_colorbar = fig.colorbar(
        error_map,
        ax=axes[2],
        orientation="horizontal",
        pad=0.08,
        fraction=0.12,
        aspect=15,
    )
    error_colorbar.set_label(r"$\varphi_{id}-\varphi_{ref}$ [-]", fontsize=9)
    error_colorbar.ax.tick_params(labelsize=8)
    return save_figure(
        fig,
        output_dir,
        "collapse_inverse_volume_fraction_error.png",
    )


def create_inverse_plots(
    *,
    loss_history: Sequence[float],
    friction_angle_history: Sequence[float],
    identified_friction_angle: float,
    reference_friction_angle: float,
    cohesion: float,
    num_steps: int,
    output_dir: str | Path = FIGURES_DIR,
) -> dict[str, Path]:
    """Create all three inverse-analysis figures."""
    configure_plotting()
    destination = Path(output_dir)
    paths = {
        "loss": plot_loss_history(loss_history, output_dir=destination),
        "parameter": plot_parameter_history(
            friction_angle_history,
            reference_friction_angle=reference_friction_angle,
            output_dir=destination,
        ),
        "volume_fraction_error": plot_volume_fraction_error(
            identified_friction_angle,
            reference_friction_angle=reference_friction_angle,
            cohesion=cohesion,
            num_steps=num_steps,
            output_dir=destination,
        ),
    }
    for name, path in paths.items():
        print(f"saved {name} figure: {path}")
    return paths
