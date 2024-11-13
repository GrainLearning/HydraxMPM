import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv

from ..particles.particles import MPMConfig
from .math_helpers import get_KE_stack, get_pressure_stack
from .mpm_plot_helpers import point_to_3D


def io_vtk_callback(
    config: MPMConfig, particle_output=None, rigid_stack_index=None, output_box=True
):
    if particle_output is None:
        particle_output = ()

    if output_box:
        bbox = pv.Box(
            bounds=np.array(
                list(
                    zip(
                        point_to_3D(config, config.origin),
                        point_to_3D(config, config.end),
                    )
                )
            ).flatten()
        )

        bbox.save(f"{config.dir_path }/output/{config.project}/box.vtk")

    def io_vtk(carry, step):
        (
            solver,
            particles,
            nodes,
            material_stack,
            forces_stack,
        ) = carry
        position_stack = particles.position_stack

        stress_stack = particles.stress_stack

        position_3D_stack = jnp.pad(
            position_stack,
            [(0, 0), config.padding],
            mode="constant",
            constant_values=0,
        )

        cloud = pv.PolyData(np.array(position_3D_stack))

        jax.debug.print("step {}", step)

        for key in particle_output:
            if key == "phi_stack":
                # FUTURE generalize for more material
                phi_stack = particles.get_solid_volume_fraction_stack(
                    material_stack[0].rho_p
                )
                cloud["phi_stack"] = np.array(phi_stack)
            elif key == "KE_stack":
                KE_stack = get_KE_stack(particles.mass_stack, particles.velocity_stack)
                KE_stack = jnp.nan_to_num(KE_stack, nan=0.0, posinf=0.0, neginf=0.0)
                cloud["KE_stack"] = np.array(KE_stack)
            elif key == "pressure_stack":
                pressure_stack = get_pressure_stack(stress_stack, config.dim)
                cloud["pressure_stack"] = np.array(pressure_stack)
            else:
                X_stack = particles.__getattribute__(key)

                if X_stack.shape == (config.num_points, config.dim):
                    X_3D_stack = jnp.pad(
                        X_stack,
                        [(0, 0), config.padding],
                        mode="constant",
                        constant_values=0,
                    )
                else:
                    X_stack = X_3D_stack
                cloud[key] = np.array(X_stack)

        cloud.save(f"{config.dir_path }/output/{config.project}/particles_{step}.vtk")

        if rigid_stack_index is not None:
            rigid_position_stack = forces_stack[rigid_stack_index].position_stack
            rigid_cloud = pv.PolyData(np.array(rigid_position_stack))

            rigid_cloud.save(
                f"{config.dir_path }/output/{config.project}/rigid_particles_{step}.vtk"
            )

    def callback(carry, step):
        return jax.debug.callback(io_vtk, carry, step)

    return callback
