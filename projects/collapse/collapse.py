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
END = (0.2, 0.04)
CELL_SIZE = 0.005
COLUMN_WIDTH = 0.05
COLUMN_HEIGHT = 0.025
PPC = 2
DT = 5e-5
TOTAL_TIME = 0.5
OUTPUT_TIME = 0.05


def compute_dp_mu(fric_angle_deg: float | jnp.ndarray) -> jnp.ndarray:
    """Compute Drucker–Prager friction coefficient from friction angle (deg)."""
    fric_angle_rad = jnp.deg2rad(fric_angle_deg)
    return 6.0 * jnp.sin(fric_angle_rad) / (jnp.sqrt(3.0) * (3.0 + jnp.sin(fric_angle_rad)))


def compute_global_measures(sim_state: Any) -> dict[str, jnp.ndarray]:
    """Compute global collapse measures from the final material-point state."""
    mp_state = sim_state.world.material_points[0]
    pos = mp_state.position_stack
    x = pos[:, 0]
    y = pos[:, 1]

    return {
        "final_height": jnp.max(y) - jnp.min(y),
        "final_center_of_mass": jnp.asarray([jnp.mean(x), jnp.mean(y)], dtype=jnp.float32),
        "final_runout_distance": jnp.max(x) - jnp.min(x),
    }


def project_volume_fraction_field(
    sim_state: Any,
    *,
    origin: tuple[float, float] = ORIGIN,
    end: tuple[float, float] = END,
    cell_size: float = CELL_SIZE,
) -> jnp.ndarray:
    """Project final solid volume fraction onto the background grid."""
    mp_state = sim_state.world.material_points[0]
    pos = mp_state.position_stack[:, :2]
    volume_stack = mp_state.volume_stack

    x0, y0 = origin
    x1, y1 = end
    nx = int(round((x1 - x0) / cell_size))
    ny = int(round((y1 - y0) / cell_size))

    cell_x = jnp.clip(jnp.floor((pos[:, 0] - x0) / cell_size).astype(jnp.int32), 0, nx - 1)
    cell_y = jnp.clip(jnp.floor((pos[:, 1] - y0) / cell_size).astype(jnp.int32), 0, ny - 1)
    flat_idx = cell_x + nx * cell_y
    weights = volume_stack / (cell_size**2)

    field = jnp.bincount(flat_idx, weights=weights, length=nx * ny).reshape(nx, ny)
    return field / jnp.maximum(jnp.max(field), 1e-12)


def save_measure_bundle(
    global_measures: dict[str, jnp.ndarray],
    local_field: jnp.ndarray,
    *,
    prefix: str = "collapse_ref",
) -> tuple[Path, Path]:
    """Save global and local measures as NPZ + JSON under ``projects/collapse/output``."""
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    npz_path = OUTPUT_DIR / f"{prefix}_measures.npz"
    global_arrays = {name: jnp.asarray(value) for name, value in global_measures.items()}
    jnp.savez(npz_path, **global_arrays, local_field=jnp.asarray(local_field))

    json_path = OUTPUT_DIR / f"{prefix}_global.json"
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump({name: jnp.asarray(value).tolist() for name, value in global_measures.items()}, fp, indent=2)

    return npz_path, json_path


def simulate_collapse(
    fric_angle: float = 20.0,
    c0: float = 10.0,
    *,
    save_bundle: bool = False,
    prefix: str = "collapse_ref",
    visualize: bool = False,
    num_steps: int | None = None,
) -> dict[str, Any]:
    """Run one forward collapse simulation.

    Parameters
    ----------
    fric_angle:
        Friction angle (deg) used to compute DP ``mu_1`` and ``mu_2``.
    c0:
        Mohr–Coulomb cohesion mapped into the DP law.
    save_bundle:
        If True, save global/local outputs under [output/](/home/hcheng/GrainLearning/HydraxMPM/projects/collapse/output).
    prefix:
        Prefix for saved files.
    visualize:
        If True, stream simulation data to Rerun.
    num_steps:
        Optional override for number of explicit solver steps. If None, uses
        ``int(TOTAL_TIME / DT)``.
    """
    import hydraxmpm as hdx

    sep = CELL_SIZE / PPC
    x = jnp.arange(0.0, COLUMN_WIDTH, sep) + 2.0 * sep
    y = jnp.arange(0.0, COLUMN_HEIGHT, sep) + 2.0 * sep
    xv, yv = jnp.meshgrid(x, y)
    position_stack = jnp.column_stack((xv.ravel(), yv.ravel()))
    num_particles = position_stack.shape[0]

    density_stack = jnp.full((num_particles,), 2650.0)
    mu = compute_dp_mu(fric_angle)
    law = hdx.DruckerPrager(
        nu=0.3,
        K=7e5,
        mu_1=mu,
        mu_2=mu,
        c0=c0,
        rho_0=2650.0,
    )
    law_state = law.create_state(stress_stack=jnp.zeros((num_particles, 3, 3)))

    sim_builder = hdx.SimBuilder()
    sim_builder.add_material_points(
        position_stack=position_stack,
        density_stack=density_stack,
        cell_size=CELL_SIZE,
        ppc=PPC,
    )
    sim_builder.add_grid(origin=ORIGIN, end=END, cell_size=CELL_SIZE)
    sim_builder.add_constitutive_law(law=law, law_state=law_state)
    sim_builder.couple(shapefunction="quadratic")
    sim_builder.add_gravity(gravity=jnp.array([0.0, -9.81]), is_apply_on_grid=True)
    sim_builder.add_sdf_object(
        sdf_logic=hdx.DomainSDF(
            origin=ORIGIN,
            end=END,
            frictions=[0.0, 0.9, 0.0, 0.0],
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
    local_field = project_volume_fraction_field(
        final_state,
        origin=ORIGIN,
        end=END,
        cell_size=CELL_SIZE,
    )

    if save_bundle:
        save_measure_bundle(global_measures, local_field, prefix=prefix)

    return {
        "global": global_measures,
        "local": local_field,
        "fric_angle": fric_angle,
        "c0": c0,
    }


def run_sim() -> dict[str, Any]:
    """Run the interactive forward benchmark with visualization and bundle export."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    result = simulate_collapse(
        fric_angle=20.0,
        c0=10.0,
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
