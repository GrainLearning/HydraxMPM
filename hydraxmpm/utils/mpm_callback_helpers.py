from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv

from ..solvers.mpm_solver import MPMSolver
from .math_helpers import get_KE_stack, get_pressure_stack
from .mpm_plot_helpers import point_to_3D
from ..material_points.material_points import MaterialPoints
import inspect


# a function that creates a function
def io_helper_vtk(
    output_box: bool = True,
    start_output: int = 0,
    remove_word_stack=True,
):
    # helper function
    def add_meta_data(solver, cloud, step):
        cloud.add_field_data(np.array(solver.grid.origin), "origin")
        cloud.add_field_data(np.array(solver.grid.end), "end")
        cloud.add_field_data(np.array(solver.grid.grid_size), "grid_size")
        cloud.add_field_data(solver.config.dt, "dt")
        cloud.add_field_data(solver.config.dim, "dim")
        cloud.add_field_data(step * solver.config.dt, "t")
        return cloud

    # helper function
    def clean_word(key, remove_word_stack):
        if remove_word_stack:
            return key.replace("_stack", "")
        return key

    def io_vtk(solver, step):
        jax.debug.print("[Callback] output {}", step)
        if (step == 0) & output_box:
            bbox = pv.Box(
                bounds=np.array(
                    list(
                        zip(
                            point_to_3D(solver._padding, solver.grid.origin),
                            point_to_3D(solver._padding, solver.grid.end),
                        )
                    )
                ).flatten()
            )

            bbox.save(f"{solver.config.output_path}/box.vtk")

        if step < start_output:
            return

        material_points_output = solver.config.output.get("material_points", ())

        if len(material_points_output) > 0:
            position_stack = solver.material_points.position_stack

            position_3D_stack = jnp.pad(
                position_stack,
                [(0, 0), solver.config._padding],
                mode="constant",
                constant_values=0,
            )

            cloud = pv.PolyData(np.array(position_3D_stack))
            cloud = add_meta_data(solver, cloud, step)
            for key in material_points_output:
                output_name = clean_word(key, remove_word_stack)
                cloud[output_name] = solver.material_points.__getattribute__(key)

            cloud.save(f"{solver.config.output_path}/particles_{step}.vtk")

        solver_output = solver.config.output.get("solver", ())

        solver_p2g_output = [x for x in solver_output if "p2g" in x]

        if len(solver_p2g_output) > 0:
            position_stack = solver.grid.position_stack

            position_3D_stack = jnp.pad(
                position_stack,
                [(0, 0), solver.config._padding],
                mode="constant",
                constant_values=0,
            )

            cloud = pv.PolyData(np.array(position_3D_stack))
            cloud = add_meta_data(solver, cloud, step)
            for key in solver_p2g_output:
                output_name = clean_word(key, remove_word_stack)
                cloud[output_name] = solver.__getattribute__(key)
            cloud.save(f"{solver.config.output_path}/p2g_{step}.vtk")

    def callback(carry, step):
        return jax.debug.callback(io_vtk, carry, step)

    return callback
