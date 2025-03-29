from functools import partial
from typing import Callable, Optional, Self, Tuple
from typing_extensions import runtime

import equinox as eqx
import jax
import jax.numpy as jnp

from ..common.types import TypeFloat

from .mpm_solver import MPMSolver
import time

import os
import shutil
import sys
from pathlib import Path
from datetime import timedelta


class Solution(eqx.Module):
    step: TypeFloat
    sim_time: TypeFloat
    output_time: TypeFloat
    dt: TypeFloat
    solver: MPMSolver
    output_dir: Optional[str] = eqx.field(default=None, static=True)


def get_script_directory():
    """Get the directory of the script that imported this module."""
    main_module = sys.modules.get("__main__")
    script_path = Path(main_module.__file__).resolve()
    return script_path.parent


def create_dir(directory_path, override=True):
    script_dir = get_script_directory()
    directory_path = (script_dir / directory_path).as_posix()

    # Check if the directory exists
    if os.path.exists(directory_path) and override:
        # Remove the directory and all its contents
        shutil.rmtree(directory_path)

    # Create an empty directory
    os.makedirs(directory_path)
    return directory_path


def save_files(step, config, name="", **kwargs):
    if len(kwargs) > 0:
        jnp.savez(
            f"{config.output_path}/{name}.{step.astype(int)}",
            **kwargs,
        )


def save_all(carry):
    step, next_solver, prev_solver, store_interval, output_time, _dt = carry

    shape_map_arrays, material_point_arrays, forces_arrays = prev_solver.get_output(
        next_solver, _dt
    )

    jax.debug.callback(
        save_files,
        step,
        prev_solver.config,
        "material_points",
        **material_point_arrays,
    )

    jax.debug.callback(
        save_files,
        step,
        prev_solver.config,
        "shape_map",
        **shape_map_arrays,
    )
    # jax.debug.callback(
    #     save_files, step, prev_solver.config, "solver", **shape_map_arrays
    # )

    jax.debug.callback(save_files, step, prev_solver.config, "forces", **forces_arrays)

    jax.debug.print("Saved output at step: {} time: {} dt {}", step, output_time, _dt)
    return output_time + store_interval


# @partial(
#     jax.jit,
#     static_argnames=(
#         "adaptive",
#         "total_time",
#         "dt",
#         "store_interval",
#         "dt_alpha",
#         "output_dir",
#         "override_dir",
#     ),
# )
@eqx.filter_jit
def run_mpm(
    solver: MPMSolver,
    total_time: float,
    store_interval: float,
    dt: Optional[float] = 0.0,
    output_dir: Optional[str] = None,
    override_dir: Optional[bool] = False,
    adaptive=True,
    dt_alpha: Optional[float] = 0.5,
):
    """Run the MPM solver.

    **Arguments:**

    - `solver`: The MPM solver.

    **Returns:**

    - `step`: The current step.
    - `solver`: The MPM solver.
    """
    if adaptive:
        _dt = solver._get_timestep(dt_alpha)
    else:
        _dt = dt

    if output_dir is not None:
        output_dir_full = create_dir(output_dir, override=override_dir)

    save_all((0, solver, solver, store_interval, 0.0, _dt))

    def main_loop(carry):
        step, prev_sim_time, prev_output_time, _dt, prev_solver = carry

        # if timestep overshoots,
        # we clip so we can save the state at the correct time
        if output_dir is not None:
            _dt = jnp.clip(prev_sim_time + _dt, max=prev_output_time) - prev_sim_time

        next_solver = prev_solver.update(step, _dt)

        next_sim_time = prev_sim_time + _dt
        if output_dir is not None:
            next_output_time = jax.lax.cond(
                abs(next_sim_time - prev_output_time) < 1e-12,
                lambda _: save_all(_),
                lambda _: prev_output_time,
                (
                    step + 1,
                    next_solver,
                    prev_solver,
                    store_interval,
                    prev_output_time,
                    _dt,
                ),
            )

        if adaptive:
            next_dt = solver._get_timestep(dt_alpha)
        else:
            next_dt = dt

        return (step + 1, next_sim_time, next_output_time, next_dt, next_solver)

    step, sim_time, output_time, dt, solver = eqx.internal.while_loop(
        lambda carry: carry[1] < total_time,
        main_loop,
        # step,sim_time,output_time,solver
        (0, 0.0, store_interval, _dt, solver),
        kind="lax",
    )

    return Solution(
        step=step,
        sim_time=sim_time,
        solver=solver,
        output_time=output_time,
        dt=dt,
        output_dir=output_dir_full,
    )
