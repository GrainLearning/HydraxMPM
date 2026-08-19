import os
import json
from pathlib import Path

import jax
import jax.numpy as jnp


OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def compute_global_measures(sim_state):
    """Collect the final global collapse metrics from the material points."""
    mp_state = sim_state.world.material_points[0]
    pos = mp_state.position_stack
    x = pos[:, 0]
    y = pos[:, 1]

    height = jnp.max(y) - jnp.min(y)
    center_of_mass = jnp.asarray([jnp.mean(x), jnp.mean(y)], dtype=jnp.float32)
    runout = jnp.max(x) - jnp.min(x)
    return {
        "final_height": float(height),
        "final_center_of_mass": [float(center_of_mass[0]), float(center_of_mass[1])],
        "final_runout_distance": float(runout),
    }


def project_volume_fraction_field(sim_state, *, origin=(0.0, 0.0), end=(0.6, 0.11), cell_size=0.0025):
    """Project the final solid volume fraction onto the background grid."""
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
    weights = volume_stack / (cell_size ** 2)

    field = jnp.bincount(flat_idx, weights=weights, length=nx * ny).reshape(nx, ny)
    field = field / jnp.maximum(jnp.max(field), 1e-12)
    return field


def save_measure_bundle(global_measures, local_field, *, prefix="collapse_ref"):
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    npz_path = OUTPUT_DIR / f"{prefix}_measures.npz"
    jnp.savez(npz_path, **global_measures, local_field=jnp.asarray(local_field))

    json_path = OUTPUT_DIR / f"{prefix}_global.json"
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(global_measures, fp, indent=2)

    return npz_path, json_path


def simulate_collapse(
    fric_angle: float = 19.8,
    c0: float = 1e4,
    *,
    save_bundle: bool = False,
    prefix: str = "collapse_ref",
    visualize: bool = False,
):
    """Run one collapse forward solve with optional Rerun visualization."""
    import hydraxmpm as hdx
    import jax.numpy as jnp

    class ColumnParameters:
        def __init__(self, fric_angle, c0):
            self.fric_angle = float(fric_angle)
            self.c0 = float(c0)
            self.rho_0 = 2650.0
            self.rho_p = 2650.0
            self.K = 7e5

        def compute_mu(self):
            fric_angle_rad = jnp.deg2rad(self.fric_angle)
            mu = (
                6
                * jnp.sin(fric_angle_rad)
                / (jnp.sqrt(3) * (3 + jnp.sin(jnp.deg2rad(self.fric_angle))))
            )
            return mu

    class CollapseProcedure:
        origin: tuple[float, float] = (0.0, 0.0)
        end: tuple[float, float] = (0.6, 0.11)
        cell_size: float = 0.0025
        column_width: float = 0.2
        column_height: float = 0.1
        ppc: int = 2
        dt: float = 1e-5
        total_time: float = 2.0
        output_time: float = 0.05
        gap = 0.0025

        def build_template(self):
            default_params = ColumnParameters(fric_angle, c0)
            sep = self.cell_size / self.ppc
            x = jnp.arange(0, self.column_width, sep) + 2 * sep
            y = jnp.arange(0, self.column_height, sep) + 2 * sep
            xv, yv = jnp.meshgrid(x, y)

            position_stack = jnp.array(list(zip(xv.flatten(), yv.flatten())))
            num_particles = len(position_stack)
            density_stack = jnp.ones(num_particles) * default_params.rho_0

            law = hdx.DruckerPrager(
                nu=0.3,
                K=default_params.K,
                mu_1=default_params.compute_mu(),
                c0=default_params.c0,
                rho_0=default_params.rho_0,
            )
            law_state = law.create_state(stress_stack=jnp.zeros((num_particles, 3, 3)))

            sim_builder = hdx.SimBuilder()
            sim_builder.add_material_points(
                position_stack=position_stack,
                density_stack=density_stack,
                cell_size=self.cell_size,
                ppc=self.ppc,
            )
            sim_builder.add_grid(
                origin=self.origin,
                end=self.end,
                cell_size=self.cell_size,
            )
            sim_builder.add_constitutive_law(law=law, law_state=law_state)
            sim_builder.couple(shapefunction="quadratic")
            sim_builder.add_gravity(gravity=jnp.array([0.0, -9.81]), is_apply_on_grid=True)

            domain_sdf = hdx.DomainSDF(
                origin=self.origin,
                end=self.end,
                frictions=0.9,
                wall_offset=0.75 * self.cell_size,
            )
            sim_builder.add_sdf_object(sdf_logic=domain_sdf)
            sim_builder.add_sdf_collider(gap=sep)
            sim_builder.set_solver(scheme="usl_aflip", alpha=0.90)
            mpm_solver, sim_state = sim_builder.build(dt=self.dt)
            return mpm_solver, sim_state

        def run(self, solver, state, call_back=None):
            steps = int(self.total_time / self.dt)
            output_step = int(self.output_time / self.dt)

            def loop_body(i, val_state):
                next_state = solver(val_state)
                if call_back is not None:
                    jax.lax.cond(
                        i % output_step == 0,
                        lambda s: jax.debug.callback(call_back, s),
                        lambda s: None,
                        next_state,
                    )
                return next_state

            return jax.lax.fori_loop(0, steps, loop_body, state)

    collapse_procedure = CollapseProcedure()
    mpm_solver, sim_state = collapse_procedure.build_template()

    call_back = None
    if visualize:
        viewer = hdx.RerunVisualizer(is_3d=False)
        viewer.log_static_domain(
            origin=collapse_procedure.origin,
            end=collapse_procedure.end,
            cell_size=collapse_procedure.cell_size,
        )

        def log_simulation(current_state):
            mp_state = current_state.world.material_points[0]
            viewer.log_time(
                current_step=int(current_state.step),
                current_time=float(current_state.time),
            )
            viewer.log_material_points(
                mp_state,
                v_min=0,
                v_max=0.5,
                property_name="velocity_stack",
            )

        call_back = log_simulation

    final_state = collapse_procedure.run(mpm_solver, sim_state, call_back)
    global_measures = compute_global_measures(final_state)
    local_field = project_volume_fraction_field(
        final_state,
        origin=collapse_procedure.origin,
        end=collapse_procedure.end,
        cell_size=collapse_procedure.cell_size,
    )

    if save_bundle:
        save_measure_bundle(global_measures, local_field, prefix=prefix)

    return {
        "global": global_measures,
        "local": local_field,
        "fric_angle": fric_angle,
        "c0": c0,
    }


def run_sim():
    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    result = simulate_collapse(
        fric_angle=19.8,
        c0=1e4,
        save_bundle=True,
        prefix="collapse_ref",
        visualize=True,
    )
    print("Global measures:", result["global"])
    print("Saved local field shape:", result["local"].shape)
    return result


import multiprocessing

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
