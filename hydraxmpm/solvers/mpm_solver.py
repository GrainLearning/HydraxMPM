from functools import partial
from sqlite3 import adapt
from typing import Callable, Optional, Self, Tuple

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
from ..material_points.material_points import MaterialPoints

from ..solvers.config import Config
from ..utils.math_helpers import (
    get_hencky_strain_stack,
    get_inertial_number_stack,
    get_pressure_stack,
    get_q_vm_stack,
    get_scalar_shear_strain_stack,
    get_strain_rate_from_L_stack,
)


def _numpy_tuple_deep(x) -> tuple:
    return tuple(map(tuple, jnp.array(x).tolist()))


class MPMSolver(Base):
    """
    MPM solver base class for running MPM simulations which contains all components.

    This class also provides initialization convenience functions
    to create a solver from a dictionary of options.

    Attributes:
        config: (:class:`Config`) Solver configuration #see #[Config]
        material_points: (:class:`MaterialPoints`) MPM material_points object # MaterialPoints # see #[MaterialPoints]. # TODO
        grid: (:class:`Grid`) Regular background grid see #[Nodes]. # TODO
        constitutive_laws: (:class:`ConstitutiveLaw`) List of constitutive_laws see #[Materials]. # TODO
        forces: (:class:`Force`) List of forces # see #[Forces]. # TODO
    """

    # Modules
    config: Config = eqx.field(static=True)
    material_points: MaterialPoints
    grid: Grid
    forces: Tuple[Force, ...] = eqx.field(default=())
    constitutive_laws: Tuple[ConstitutiveLaw, ...] = eqx.field(default=())
    callbacks: Tuple[Callable, ...] = eqx.field(static=True, default=())
    _setup_done: bool = eqx.field(default=False)
    shape_map: ShapeFunctionMapping = eqx.field(init=False)

    dt: TypeFloat = eqx.field(default=1e-3)

    def __init__(
        self,
        config: Config,
        material_points: MaterialPoints,
        grid: Grid,
        constitutive_laws: Optional[
            Tuple[ConstitutiveLaw, ...] | ConstitutiveLaw
        ] = None,
        forces: Optional[Tuple[Force, ...] | Force] = None,
        **kwargs,
    ) -> Self:
        assert material_points.position_stack.shape[1] == config.dim, (
            "Dimension mismatch of material points and config, check if dim is set correctly. Either"
            "the material_points or the config.dim is set incorrectly."
        )
        assert len(grid.origin) == config.dim, (
            "Dimension mismatch of origin and config. Either "
            "the origin or the config.dim is set incorrectly."
        )

        self.config = config

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
            shapefunction=config.shapefunction,
            num_points=self.material_points.num_points,
            num_cells=self.grid.num_cells,
            dim=config.dim,
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
            self.grid.cell_size, self.config.ppc
        )

        for constitutive_law in self.constitutive_laws:
            new_constitutive_law, new_material_points = constitutive_law.init_state(
                new_material_points
            )
            new_constitutive_laws.append(new_constitutive_law)

        new_constitutive_laws = tuple(new_constitutive_laws)

        new_grid = self.grid.init_padding(self.config.shapefunction)

        new_forces = []

        for force in self.forces:
            if isinstance(force, Boundary):
                new_force, new_grid = force.init_ids(grid=new_grid, dim=self.config.dim)
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
                dim=self.config.dim,
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
                dim=self.config.dim,
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
                dim=self.config.dim,
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
        material_points_output = new_solver.config.output.get("material_points", ())

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
        shape_map_output = new_solver.config.output.get("shape_map", ())

        for key in shape_map_output:
            # shape_map_arrays[key] = new_solver.shape_map.__getattribute__(key)
            output = new_solver.shape_map.__getattribute__(key)
            if callable(output):
                output = output(
                    material_points=new_solver.material_points,
                    grid=new_solver.grid,
                    dt=dt,
                )
            shape_map_arrays[key] = output

        forces_arrays = {}
        forces_output = new_solver.config.output.get("forces", ())
        for key in forces_output:
            for force in new_solver.forces:
                key_array = force.__dict__.get(key, None)
                if key_array is not None:
                    forces_arrays[key] = key_array

        return shape_map_arrays, material_point_arrays, forces_arrays
