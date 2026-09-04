"""Forward collapse benchmark used by the AD inverse example.

This module provides:
1) A forward solver entrypoint: ``simulate_collapse``.
2) Post-processing utilities for global/local measures.
3) Optional Rerun visualization for interactive runs.
4) Optional saving of a reference bundle for inverse analysis.
"""

import json
import multiprocessing
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Benchmark geometry / discretization defaults.
ORIGIN = (0.0, 0.0)
END = (2.0, 0.4)
CELL_SIZE = 0.005
BULK_HEIGHT_CUTOFF = 2.0 * CELL_SIZE
COLUMN_WIDTH = 0.5
COLUMN_HEIGHT = 0.25
PPC = 4
DT = 5e-5
TOTAL_TIME = 0.5
OUTPUT_TIME = 0.05

# Initial granular-state parameters. ``density_stack`` is the bulk density of
# the porous continuum, not the intrinsic density of the solid grains.
GRAIN_DENSITY = 2650.0
INITIAL_SOLID_VOLUME_FRACTION = 0.60
BULK_DENSITY = GRAIN_DENSITY * INITIAL_SOLID_VOLUME_FRACTION
LATERAL_STRESS_RATIO = 0.5
SEPARATION_DENSITY_RATIO = 0.90
GRAVITY_MAGNITUDE = 9.81
BASE_FRICTION = 0.9


def initialize_lithostatic_stress(
    position_stack: jnp.ndarray,
    density_stack: jnp.ndarray,
    *,
    surface_elevation: float,
    lateral_stress_ratio: float = LATERAL_STRESS_RATIO,
) -> jnp.ndarray:
    """Initialize gravity-equilibrated, compression-positive stress.

    This is the horizontal-bed specialization of the initialization used by
    ``projects/granular_chute_flow.chute_flow``. The vertical stress is
    hydrostatic/lithostatic, the lateral stresses follow a constant K0 ratio,
    and there is no initial shear stress on the level bed.
    """
    if position_stack.ndim != 2 or position_stack.shape[1] < 2:
        raise ValueError("position_stack must have shape (num_particles, >=2)")
    if density_stack.shape != (position_stack.shape[0],):
        raise ValueError("density_stack must have one value per particle")

    depth_below_surface = jnp.maximum(
        jnp.asarray(surface_elevation, dtype=position_stack.dtype)
        - position_stack[:, 1],
        0.0,
    )
    sigma_yy = density_stack * GRAVITY_MAGNITUDE * depth_below_surface
    sigma_lateral = lateral_stress_ratio * sigma_yy
    stress_stack = jnp.zeros(
        (position_stack.shape[0], 3, 3), dtype=position_stack.dtype
    )
    stress_stack = stress_stack.at[:, 0, 0].set(sigma_lateral)
    stress_stack = stress_stack.at[:, 1, 1].set(sigma_yy)
    stress_stack = stress_stack.at[:, 2, 2].set(sigma_lateral)
    return stress_stack


def compute_dp_coefficients(
    fric_angle_deg: float | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Map Mohr-Coulomb friction and cohesion to Drucker-Prager factors.

    The constitutive law uses ``sqrt(J2) - mu_1 * p - mu_2 * c = 0``.
    These coefficients match the Mohr-Coulomb surface on the triaxial-extension
    meridian, following the convention used by the JAX-MPM collapse benchmark.
    """
    fric_angle_rad = jnp.deg2rad(fric_angle_deg)
    sin_phi = jnp.sin(fric_angle_rad)
    denominator = jnp.sqrt(3.0) * (3.0 + sin_phi)
    mu_1 = 6.0 * sin_phi / denominator
    mu_2 = 6.0 * jnp.cos(fric_angle_rad) / denominator
    return mu_1, mu_2


def compute_global_measures(sim_state: Any) -> dict[str, jnp.ndarray]:
    """Compute global collapse measures from the final material-point state."""
    mp_state = sim_state.world.material_points[0]
    pos = mp_state.position_stack
    x = pos[:, 0]
    y = pos[:, 1]

    return {
        "final_height": jnp.max(y) - jnp.min(y),
        "final_center_of_mass": jnp.stack((jnp.mean(x), jnp.mean(y))),
        "final_runout_distance": jnp.max(x) - jnp.min(x),
    }


def project_volume_fraction_field(
    sim_state: Any,
    *,
    origin: tuple[float, float] = ORIGIN,
    end: tuple[float, float] = END,
    cell_size: float = CELL_SIZE,
) -> jnp.ndarray:
    """Bin final solid volume fraction onto background-grid cells."""
    mp_state = sim_state.world.material_points[0]
    pos = mp_state.position_stack[:, :2]

    x0, y0 = origin
    x1, y1 = end
    nx = int(round((x1 - x0) / cell_size))
    ny = int(round((y1 - y0) / cell_size))

    cell_x = jnp.clip(
        jnp.floor((pos[:, 0] - x0) / cell_size).astype(jnp.int32), 0, nx - 1
    )
    cell_y = jnp.clip(
        jnp.floor((pos[:, 1] - y0) / cell_size).astype(jnp.int32), 0, ny - 1
    )
    flat_idx = cell_x + nx * cell_y
    solid_volume_stack = mp_state.mass_stack / GRAIN_DENSITY
    weights = solid_volume_stack / (cell_size**2)

    return jnp.bincount(flat_idx, weights=weights, length=nx * ny).reshape(ny, nx).T


def project_nodal_volume_fraction_field(
    sim_state: Any,
    *,
    origin: tuple[float, float] = ORIGIN,
    end: tuple[float, float] = END,
    cell_size: float = CELL_SIZE,
) -> jnp.ndarray:
    """Project solid volume fraction with the simulation's quadratic mapping."""
    import hydraxmpm as hdx

    material_points = sim_state.world.material_points[0]
    domain = hdx.GridDomain.create(origin, end, cell_size, padding=0)
    mapping = hdx.ShapeFunctionMapping("quadratic", dim=2)
    cache = mapping.compute(
        material_points.position_stack[:, :2],
        domain.origin,
        domain.grid_size,
        domain._inv_cell_size,
    )
    solid_volume_stack = material_points.mass_stack / GRAIN_DENSITY
    nodal_solid_volume = mapping.scatter_to_grid(
        cache,
        jnp.ones_like(solid_volume_stack),
        solid_volume_stack,
        domain.num_cells,
        normalize=False,
    )
    return (nodal_solid_volume / cell_size**2).reshape(domain.grid_size)


def compute_bulk_height_profile(
    volume_fraction: jnp.ndarray,
    *,
    cell_size: float = CELL_SIZE,
    reference_solid_fraction: float = INITIAL_SOLID_VOLUME_FRACTION,
) -> jnp.ndarray:
    """Return equivalent solid-volume height at every horizontal grid node.

    The profile is the vertical integral of solid volume fraction divided by
    the initial packing fraction,

    ``h_v(x) = integral(phi_s(x, y), dy) / phi_s0``.

    It is linear in projected solid volume, contains no occupancy threshold,
    and preserves material area when integrated horizontally.
    """
    if volume_fraction.ndim != 2:
        raise ValueError("volume_fraction must be a two-dimensional grid")
    if cell_size <= 0.0:
        raise ValueError("cell_size must be positive")
    if reference_solid_fraction <= 0.0:
        raise ValueError("reference_solid_fraction must be positive")
    return cell_size * jnp.sum(volume_fraction, axis=1) / reference_solid_fraction


def save_measure_bundle(
    global_measures: dict[str, jnp.ndarray],
    local_field: jnp.ndarray,
    *,
    prefix: str = "collapse_ref",
) -> tuple[Path, Path]:
    """Save global and local measures under ``projects/collapse/output``."""
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    npz_path = OUTPUT_DIR / f"{prefix}_measures.npz"
    global_arrays = {
        name: jnp.asarray(value) for name, value in global_measures.items()
    }
    jnp.savez(npz_path, **global_arrays, local_field=jnp.asarray(local_field))

    json_path = OUTPUT_DIR / f"{prefix}_global.json"
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                name: jnp.asarray(value).tolist()
                for name, value in global_measures.items()
            },
            fp,
            indent=2,
        )

    return npz_path, json_path


def simulate_collapse(
    fric_angle: float = 20.0,
    c0: float = 150.0,
    base_friction: float = BASE_FRICTION,
    *,
    save_bundle: bool = False,
    prefix: str = "collapse_ref",
    visualize: bool = False,
    num_steps: int | None = None,
    compute_local: bool = True,
    compute_height_profile: bool = False,
    return_final_state: bool = False,
) -> dict[str, Any]:
    """Run one forward collapse simulation.

    Parameters
    ----------
    fric_angle:
        Mohr-Coulomb friction angle (deg) used to compute the distinct DP
        friction and cohesion factors ``mu_1`` and ``mu_2``.
    c0:
        Mohr-Coulomb cohesion mapped into the DP law.
    base_friction:
        Coulomb friction coefficient on the horizontal bottom wall. The other
        three domain walls remain frictionless.
    save_bundle:
        If True, save global/local outputs under the collapse output directory.
    prefix:
        Prefix for saved files.
    visualize:
        If True, stream simulation data to Rerun.
    num_steps:
        Optional override for number of explicit solver steps. If None, uses
        ``int(TOTAL_TIME / DT)``.
    compute_local:
        If False, skip the final volume-fraction projection. This is useful for
        inverse runs that consume only terminal global measures. Saving a
        bundle still computes the local field because it is part of the bundle.
    compute_height_profile:
        If True, project the final state with the quadratic mapping and return
        a differentiable equivalent solid-volume height profile on the fixed
        horizontal grid.
    return_final_state:
        If True, include the final simulation state for HydraxMPM
        postprocessing. The inverse forward kernel leaves this disabled.
    """
    import hydraxmpm as hdx

    particles_per_axis = int(round(PPC**0.5))
    if particles_per_axis**2 != PPC:
        raise ValueError("PPC must be a perfect square for the 2D particle lattice")
    sep = CELL_SIZE / particles_per_axis
    x = jnp.arange(0.0, COLUMN_WIDTH, sep) + 2.0 * sep
    y = jnp.arange(0.0, COLUMN_HEIGHT, sep) + 2.0 * sep
    xv, yv = jnp.meshgrid(x, y)
    position_stack = jnp.column_stack((xv.ravel(), yv.ravel()))
    num_particles = position_stack.shape[0]

    density_stack = jnp.full((num_particles,), BULK_DENSITY)
    column_surface_elevation = 2.0 * sep + COLUMN_HEIGHT
    stress_stack = initialize_lithostatic_stress(
        position_stack,
        density_stack,
        surface_elevation=column_surface_elevation,
    )
    mu_1, mu_2 = compute_dp_coefficients(fric_angle)
    law = hdx.DruckerPrager(
        nu=0.3,
        K=7e5,
        mu_1=mu_1,
        mu_2=mu_2,
        c0=c0,
        mu_1_hat=0.0,
        rho_0=SEPARATION_DENSITY_RATIO * BULK_DENSITY,
    )
    law_state = law.create_state(stress_stack=stress_stack)

    sim_builder = hdx.SimBuilder()
    sim_builder.add_material_points(
        position_stack=position_stack,
        density_stack=density_stack,
        stress_stack=stress_stack,
        cell_size=CELL_SIZE,
        ppc=PPC,
    )
    sim_builder.add_grid(origin=ORIGIN, end=END, cell_size=CELL_SIZE)
    sim_builder.add_constitutive_law(law=law, law_state=law_state)
    sim_builder.couple(shapefunction="quadratic")
    sim_builder.add_gravity(
        gravity=jnp.array([0.0, -GRAVITY_MAGNITUDE]),
        is_apply_on_grid=True,
    )
    sim_builder.add_sdf_object(
        sdf_logic=hdx.DomainSDF(
            origin=ORIGIN,
            end=END,
            # DomainSDF order: left, bottom, right, top.
            frictions=[0.0, base_friction, 0.0, 0.0],
            wall_offset=0.75 * CELL_SIZE,
        )
    )
    sim_builder.add_sdf_collider(gap=sep)
    sim_builder.set_solver(scheme="usl_aflip", alpha=0.90)

    mpm_solver, sim_state = sim_builder.build(dt=DT)

    steps = int(num_steps) if num_steps is not None else int(TOTAL_TIME / DT)
    output_step = max(1, int(OUTPUT_TIME / DT))
    callback = None

    if visualize:
        viewer = hdx.RerunVisualizer(is_3d=False)
        viewer.log_static_domain(origin=ORIGIN, end=END, cell_size=CELL_SIZE)

        def log_simulation(current_state: Any) -> None:
            mp_state = current_state.world.material_points[0]
            viewer.log_time(
                current_step=int(current_state.step),
                current_time=float(current_state.time),
            )
            viewer.log_material_points(
                mp_state,
                v_min=0.0,
                v_max=0.5,
                property_name="velocity_stack",
            )

        callback = log_simulation

    def loop_body(i: int, state: Any) -> Any:
        next_state = mpm_solver(state)
        if callback is not None:
            jax.lax.cond(
                i % output_step == 0,
                lambda s: jax.debug.callback(callback, s),
                lambda s: None,
                next_state,
            )
        return next_state

    final_state = jax.lax.fori_loop(0, steps, loop_body, sim_state)
    global_measures = compute_global_measures(final_state)
    local_field = None
    if compute_local or save_bundle:
        local_field = project_volume_fraction_field(
            final_state,
            origin=ORIGIN,
            end=END,
            cell_size=CELL_SIZE,
        )
    height_profile = None
    if compute_height_profile:
        nodal_volume_fraction = project_nodal_volume_fraction_field(final_state)
        height_profile = compute_bulk_height_profile(nodal_volume_fraction)

    if save_bundle:
        assert local_field is not None
        save_measure_bundle(global_measures, local_field, prefix=prefix)

    result = {
        "global": global_measures,
        "local": local_field,
        "height_profile": height_profile,
        "fric_angle": fric_angle,
        "c0": c0,
        "base_friction": base_friction,
    }
    if return_final_state:
        result["final_state"] = final_state
    return result


def run_sim() -> dict[str, Any]:
    """Run the interactive forward benchmark with visualization and bundle export."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    result = simulate_collapse(
        fric_angle=20.0,
        c0=150.0,
        save_bundle=True,
        prefix="collapse_ref",
        visualize=True,
    )
    print("Global measures:", result["global"])
    print("Saved local field shape:", result["local"].shape)
    return result


if __name__ == "__main__":
    p = multiprocessing.Process(target=run_sim, args=())
    p.start()
    try:
        p.join()
    except KeyboardInterrupt:
        p.terminate()
        p.join()
        print("Worker dead. GPU memory released.")

    if p.exitcode == 0:
        print("Simulation finished naturally.")
    else:
        print(f"Process ended with code {p.exitcode}")
