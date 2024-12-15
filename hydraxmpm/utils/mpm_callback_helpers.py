import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv

from ..particles.particles import MPMConfig
from .math_helpers import get_KE_stack, get_pressure_stack
from .mpm_plot_helpers import point_to_3D
from typing import Dict, List, Tuple


def io_movie_callback(
    config: MPMConfig,
    scalar_name: str = "",
    timeseries_options: Dict = None,
    bbox_options: Dict = None,
    camera_options: Dict = None,
    plotter_options: Dict = None,
    plotter: pv.Plotter = None,
    file_path=None,
):
    if file_path is None:
        file_path = f"{config.dir_path }/{config.output_path}/{config.project}/movie.gif"

    if bbox_options is None:
        bbox_options = {}

    bbox_options.setdefault("color", "black")
    bbox_options.setdefault("style", "wireframe")

    if plotter_options is None:
        plotter_options = {}

    # plotter_options.setdefault("shape", (1, 1))

    if plotter is None:
        plotter = pv.Plotter(notebook=False, off_screen=True, **plotter_options)

    if camera_options is None:
        camera_options = {}
        if config.dim == 2:
            camera_options.setdefault("camera_position", "xy")
        else:
            camera_options.setdefault("camera_position", "xz")
            camera_options.setdefault("azimuth", 45)
            camera_options.setdefault("elevation", 30)
            camera_options.setdefault("zoom", 0.9)

    if timeseries_options is None:
        timeseries_options = {}

    if scalar_name is not None:
        timeseries_options.setdefault("scalars", scalar_name)

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

    plotter.open_gif(file_path)
    plotter.clear()

    def io_movie(carry, step):
        (
            solver,
            particles,
            nodes,
            material_stack,
            forces_stack,
        ) = carry

        position_3D_stack = jnp.pad(
            particles.position_stack,
            [(0, 0), config.padding],
            mode="constant",
            constant_values=0,
        )

        polydata = pv.PolyData(np.array(position_3D_stack))

        scalar_stack = None
        for mi, material in enumerate(material_stack):
            if scalar_name in material.__dict__.keys():
                scalar_stack = material.__getattribute__(scalar_name)

        if scalar_name in particles.__dict__.keys():
            scalar_stack = particles.__getattribute__(scalar_name)

        if scalar_name == "pressure_stack":
            scalar_stack = get_pressure_stack(particles.stress_stack, config.dim)


        if scalar_stack is not None:
            polydata.point_data[scalar_name] = np.array(scalar_stack)

        plotter.add_mesh(polydata, **timeseries_options)

        plotter.add_mesh(bbox, **bbox_options)

        if step == 0:
            
            plotter.camera.tight(padding=0.10, adjust_render_window=True)

        plotter.write_frame()
        plotter.clear()

        if step == config.num_steps - 1:
            jax.debug.print("Saving is slow may take a while")
            plotter.close()

    def callback(carry, step):
        return jax.debug.callback(io_movie, carry, step)

    return callback


def io_material_point_callback(
    config: MPMConfig,
    particle_output=None,
    material_output=None,
    particle_ids=None,
    start_output=0,
):
    if particle_output is None:
        particle_output = ()

    if material_output is None:
        material_output = ()

    if particle_ids is None:
        particle_ids = slice(0, None)

    def io_mp(carry, step):
        (
            solver,
            particles,
            nodes,
            material_stack,
            forces_stack,
        ) = carry

        if step < start_output:
            return

        for mi, material in enumerate(material_stack):
            mat_out_dict = {}
            for key in material_output:
                X_stack = material.__getattribute__(key).at[particle_ids].get()
                mat_out_dict[key] = X_stack

                jnp.savez(
                    f"{config.dir_path }/{config.output_path}/{config.project}/numpy_mat_{mi}_{step}",
                    **mat_out_dict,
                )
        particle_out_dict = {}
        for key in particle_output:
            if key == "phi_stack":
                # FUTURE generalize for more material
                X_stack = particles.get_solid_volume_fraction_stack(
                    material_stack[0].rho_p
                )
            else:
                X_stack = particles.__getattribute__(key).at[particle_ids].get()

            particle_out_dict[key] = X_stack
            jnp.savez(
                f"{config.dir_path }/{config.output_path}/{config.project}/numpy_particle_{step}",
                **particle_out_dict,
            )

    def callback(carry, step):
        return jax.debug.callback(io_mp, carry, step)

    return callback


def io_vtk_callback(
    config: MPMConfig,
    particle_output=None,
    material_output=None,
    rigid_stack_index=None,
    output_box=True,
    start_output=0,
    output_dir="output",
):
    if particle_output is None:
        particle_output = ()

    if material_output is None:
        material_output = ()

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

        bbox.save(f"{config.dir_path }/{config.output_path}/{config.project}/box.vtk")

    def io_vtk(carry, step):
        (
            solver,
            particles,
            nodes,
            material_stack,
            forces_stack,
        ) = carry
        # jax.debug.print("{}",step)
        if step < start_output:
            return
        position_stack = particles.position_stack

        stress_stack = particles.stress_stack

        position_3D_stack = jnp.pad(
            position_stack,
            [(0, 0), config.padding],
            mode="constant",
            constant_values=0,
        )

        cloud = pv.PolyData(np.array(position_3D_stack))

        for key in material_output:
            for material in material_stack:
                X_stack = material.__getattribute__(key)

                if X_stack.shape == (config.num_points, config.dim):
                    X_3D_stack = jnp.pad(
                        X_stack,
                        [(0, 0), config.padding],
                        mode="constant",
                        constant_values=0,
                    )
                    X_stack = X_3D_stack

                cloud[key] = np.array(X_stack)

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

                    X_stack = X_3D_stack
                cloud[key] = np.array(X_stack)

        cloud.save(
            f"{config.dir_path }/{output_dir}/{config.project}/particles_{step}.vtk"
        )

        if rigid_stack_index is not None:
            rigid_position_stack = forces_stack[rigid_stack_index].position_stack
            rigid_cloud = pv.PolyData(np.array(rigid_position_stack))

            rigid_cloud.save(
                f"{config.dir_path }/{output_dir}/{config.project}/rigid_particles_{step}.vtk"
            )

    def callback(carry, step):
        return jax.debug.callback(io_vtk, carry, step)

    return callback
