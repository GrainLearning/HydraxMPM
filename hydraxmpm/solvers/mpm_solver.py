# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-

from typing import Callable, Optional, Self, Tuple, Dict

import equinox as eqx
import jax
import jax.numpy as jnp

from ..common.base import Base
from ..common.types import (
    TypeFloatScalarAStack,
    TypeFloatVector3AStack,
    TypeFloatVectorAStack,
    TypeInt,
    TypeUInt,
    TypeUIntScalarAStack,
    TypeFloat,
)

from ..shapefunctions.mapping import ShapeFunctionMapping
from ..constitutive_laws.constitutive_law import ConstitutiveLaw
from ..forces.force import Force
from ..grid.grid import Grid
from ..forces.boundary import Boundary
from ..forces.slipstickboundary import SlipStickBoundary
from ..material_points.material_points import MaterialPoints

from ..utils.math_helpers import (
    get_hencky_strain_stack,
    get_inertial_number_stack,
    get_pressure_stack,
    get_q_vm_stack,
    get_scalar_shear_strain_stack,
    get_strain_rate_from_L_stack,
)

import os
import shutil


def _numpy_tuple_deep(x) -> tuple:
    return tuple(map(tuple, jnp.array(x).tolist()))


def create_dir(directory_path, override=True):
    if os.path.exists(directory_path) and override:
        shutil.rmtree(directory_path)

    os.makedirs(directory_path)
    return directory_path


def save_files(step, output_dir, name="", **kwargs):
    if len(kwargs) > 0:
        jnp.savez(f"{output_dir}/{name}.{step.astype(int)}", **kwargs)


class MPMSolver(Base):
    """
    MPM solver base class for running MPM simulations which contains all components.

    This class also provides initialization convenience functions
    to create a solver from a dictionary of options.

    Attributes:
        material_points: (:class:`MaterialPoints`) MPM material_points object # MaterialPoints # see #[MaterialPoints]. # TODO
        grid: (:class:`Grid`) Regular background grid see #[Nodes]. # TODO
        constitutive_laws: (:class:`ConstitutiveLaw`) List of constitutive_laws see #[Materials]. # TODO
        forces: (:class:`Force`) List of forces # see #[Forces]. # TODO
    """

    # Modules
    material_points: MaterialPoints
    grid: Grid
    forces: Tuple[Force, ...] = eqx.field(default=())
    constitutive_laws: Tuple[ConstitutiveLaw, ...] = eqx.field(default=())
    callbacks: Tuple[Callable, ...] = eqx.field(static=True, default=())
    _setup_done: bool = eqx.field(default=False)
    shape_map: ShapeFunctionMapping = eqx.field(init=False)
    shapefunction: str = eqx.field(static=True, default="linear")

    dt: TypeFloat = eqx.field(default=1e-3)

    dim: int = eqx.field(static=True)
    ppc: int = eqx.field(static=True, default=1)

    _padding: tuple = eqx.field(init=False, static=True, repr=False)

    output_vars: Dict | Tuple[str, ...] = eqx.field(static=True)  # run sim

    def __init__(
        self,
        *,
        dim,
        material_points: MaterialPoints,
        grid: Grid,
        constitutive_laws: Optional[
            Tuple[ConstitutiveLaw, ...] | ConstitutiveLaw
        ] = None,
        forces: Optional[Tuple[Force, ...] | Force] = None,
        ppc=1,
        shapefunction="linear",
        output_vars: Optional[dict | Tuple[str, ...]] = None,
        **kwargs,
    ) -> Self:
        assert material_points.position_stack.shape[1] == dim, (
            "Dimension mismatch of material points, check if dim is set correctly. Either"
            "the material_points or the dim is set incorrectly."
        )
        assert len(grid.origin) == dim, (
            "Dimension mismatch of origin. Either "
            "the origin or the dim is set incorrectly."
        )

        self.output_vars = output_vars

        self.dim = dim
        self.ppc = ppc
        self.shapefunction = shapefunction
        self._padding = (0, 3 - self.dim)

        self.material_points = material_points

        self.grid = grid

        self.forces = (
            forces if isinstance(forces, tuple) else (forces,) if forces else ()
        )
        self.constitutive_laws = (
            constitutive_laws
            if isinstance(constitutive_laws, tuple)
            else (constitutive_laws,)
            if constitutive_laws
            else ()
        )

        self.shape_map = ShapeFunctionMapping(
            shapefunction=self.shapefunction,
            num_points=self.material_points.num_points,
            num_cells=self.grid.num_cells,
            dim=dim,
        )

        super().__init__(**kwargs)

    def setup(self: Self, **kwargs) -> Self:
        # we run this once after initialization
        if self._setup_done:
            return self

        # initialize pressure and density...
        new_constitutive_laws = []

        new_material_points = self.material_points
        new_material_points = new_material_points.init_volume_from_cellsize(
            self.grid.cell_size, self.ppc
        )

        for constitutive_law in self.constitutive_laws:
            new_constitutive_law, new_material_points = constitutive_law.init_state(
                new_material_points
            )
            new_constitutive_laws.append(new_constitutive_law)

        new_constitutive_laws = tuple(new_constitutive_laws)

        new_grid = self.grid.init_padding(self.shapefunction)

        new_forces = []

        # TODO init stat for forces
        for force in self.forces:
            if isinstance(force, Boundary) or isinstance(force, SlipStickBoundary):
                new_force, new_grid = force.init_ids(grid=new_grid, dim=self.dim)
            else:
                new_force = force
            new_forces.append(new_force)

        new_forces = tuple(new_forces)

        params = self.__dict__
        print("🐢.. Setting up MPM solver")
        print(f"Material Points: {new_material_points.num_points}")
        print(f"Grid: {new_grid.num_cells} ({new_grid.grid_size})")

        params.update(
            material_points=new_material_points,
            grid=new_grid,
            forces=new_forces,
            constitutive_laws=new_constitutive_laws,
            _setup_done=True,
        )

        return self.__class__(**params)

    def _update_forces_on_points(
        self,
        material_points: MaterialPoints,
        grid: Grid,
        forces: Tuple[Force, ...],
        step: TypeInt,
        dt: TypeFloat,
    ) -> Tuple[MaterialPoints, Tuple[Force, ...]]:
        # called within solver .update method
        new_forces = []
        for force in forces:
            material_points, new_force = force.apply_on_points(
                material_points=material_points,
                grid=grid,
                step=step,
                dt=dt,
                dim=self.dim,
            )
            new_forces.append(new_force)
        return material_points, tuple(new_forces)

    def _update_forces_grid(
        self: Self,
        material_points: MaterialPoints,
        grid: Grid,
        forces: Tuple[Force, ...],
        step: TypeInt,
        dt: TypeFloat,
    ) -> Tuple[Grid, Tuple[Force, ...]]:
        # called within solver .update method
        new_forces = []
        for force in forces:
            grid, new_force = force.apply_on_grid(
                material_points=material_points,
                grid=grid,
                step=step,
                dt=dt,
                dim=self.dim,
                shape_map=self.shape_map,
            )
            new_forces.append(new_force)

        return grid, tuple(new_forces)

    def _update_constitutive_laws(
        self: Self,
        material_points: MaterialPoints,
        constitutive_laws: Tuple[ConstitutiveLaw, ...],
        dt,
    ) -> Tuple[MaterialPoints, Tuple[ConstitutiveLaw, ...]]:
        # called within solver .update method
        new_materials = []
        for material in constitutive_laws:
            material_points, new_material = material.update(
                material_points=material_points,
                dt=dt,
                dim=self.dim,
            )
            new_materials.append(new_material)

        return material_points, tuple(new_materials)

    def _get_timestep(self, dt_alpha: TypeFloat = 0.5) -> TypeFloat:
        dt = 1e9
        for constitutive_laws in self.constitutive_laws:
            dt = jnp.minimum(
                dt,
                constitutive_laws.get_dt_crit(
                    material_points=self.material_points,
                    cell_size=self.grid.cell_size,
                    dt_alpha=dt_alpha,
                ),
            )
        return dt

    def get_output(self, new_solver, dt):
        material_points_output = self.output_vars.get("material_points", ())

        material_point_arrays = {}
        for key in material_points_output:
            # workaround around
            # properties of one class depend on properties of another
            output = new_solver.material_points.__getattribute__(key)

            if callable(output):
                output = output(
                    dt=dt,
                    rho_p=new_solver.constitutive_laws[0].rho_p,
                    d=new_solver.constitutive_laws[0].d,
                    eps_e_stack=new_solver.constitutive_laws[0].eps_e_stack,
                    eps_e_stack_prev=self.constitutive_laws[0].eps_e_stack,
                    W_stack=new_solver.constitutive_laws[0].W_stack,
                )

            material_point_arrays[key] = output

        shape_map_arrays = {}
        shape_map_output = self.output_vars.get("shape_map", ())

        for key in shape_map_output:
            output = new_solver.shape_map.__getattribute__(key)
            if callable(output):
                output = output(
                    material_points=new_solver.material_points,
                    grid=new_solver.grid,
                    dt=dt,
                )
            shape_map_arrays[key] = output

        forces_arrays = {}
        forces_output = self.output_vars.get("forces", ())
        for key in forces_output:
            for force in new_solver.forces:
                key_array = force.__dict__.get(key, None)
                if key_array is not None:
                    forces_arrays[key] = key_array

        return shape_map_arrays, material_point_arrays, forces_arrays

    @eqx.filter_jit
    def run(
        self: Self,
        *,
        total_time: float,
        store_interval: float,
        adaptive=False,
        dt: Optional[float] = 0.0,
        dt_alpha: Optional[float] = 0.5,
        dt_max: Optional[float] = None,
        output_dir: Optional[str] = None,
        override_dir: Optional[bool] = False,
    ):
        if adaptive:
            _dt = self._get_timestep(dt_alpha)
        else:
            _dt = dt
        if (override_dir) and (output_dir is not None):
            create_dir(output_dir)

        def save_all(args):
            step, next_solver, prev_solver, store_interval, output_time, _dt = args
            shape_map_arrays, material_point_arrays, forces_arrays = (
                prev_solver.get_output(next_solver, _dt)
            )
            jax.debug.callback(
                save_files, step, output_dir, "material_points", **material_point_arrays
            )
            jax.debug.callback(
                save_files, step, output_dir, "shape_map", **shape_map_arrays
            )
            jax.debug.callback(save_files, step, output_dir, "forces", **forces_arrays)
            jax.debug.print("Saved output at step: {} time: {:.3f} ", step, output_time)
            return output_time + store_interval

        save_all((0, self, self, store_interval, 0.0, _dt))

        def main_loop(carry):
            step, prev_sim_time, prev_output_time, _dt, prev_solver = carry

            # if timestep overshoots,
            # we clip so we can save the state at the correct time
            if output_dir is not None:
                _dt = (
                    jnp.clip(prev_sim_time + _dt, max=prev_output_time) - prev_sim_time
                )

            next_solver = prev_solver.update(step, _dt)

            next_sim_time = prev_sim_time + _dt

            if output_dir is not None:
                next_output_time = jax.lax.cond(
                    abs(next_sim_time - prev_output_time) < 1e-12,
                    lambda args: save_all(args),
                    lambda args: prev_output_time,
                    (
                        step + 1,
                        next_solver,
                        prev_solver,
                        store_interval,
                        prev_output_time,
                        _dt,
                    ),
                )
            else:
                next_output_time = prev_output_time

            if adaptive:
                next_dt = next_solver._get_timestep(dt_alpha)
                next_dt = jnp.clip(next_dt, None, dt_max)
            else:
                next_dt = dt

            return (step + 1, next_sim_time, next_output_time, next_dt, next_solver)

        step, sim_time, output_time, dt, new_solver = eqx.internal.while_loop(
            lambda carry: carry[1] < total_time,
            main_loop,
            # step, sim_time, output_time, solver
            (0, 0.0, store_interval, _dt, self),
            kind="lax",
        )

        return new_solver
