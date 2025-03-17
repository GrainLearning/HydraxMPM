from functools import partial
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
        callbacks: (:class:`Callable`) # List of callback functions # see #[Callbacks]. # TODO
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

    def __init__(
        self,
        config: Config,
        material_points: MaterialPoints,
        grid: Grid,
        constitutive_laws: Optional[
            Tuple[ConstitutiveLaw, ...] | ConstitutiveLaw
        ] = None,
        forces: Optional[Tuple[Force, ...] | Force] = None,
        callbacks: Optional[Tuple[Callable, ...] | Callable] = None,
        **kwargs,
    ) -> Self:
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

        self.callbacks = (
            callbacks
            if isinstance(callbacks, tuple)
            else (callbacks,)
            if callbacks
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
        # print("setup {}", new_material_points.volume_stack)
        for constitutive_law in self.constitutive_laws:
            new_constitutive_law, new_material_points = constitutive_law.init_state(
                # self.material_points
                new_material_points
            )
            new_constitutive_laws.append(new_constitutive_law)

        new_constitutive_laws = tuple(new_constitutive_laws)

        # Pad boundary one shapefunction width
        new_grid = self.grid.init_padding(self.config.shapefunction)

        new_forces = []

        for force in self.forces:
            if isinstance(force, Boundary):
                new_force = force.init_ids(
                    grid_size=new_grid.grid_size, dim=self.config.dim, dt=self.config.dt
                )
            else:
                new_force = force
            new_forces.append(new_force)

        new_forces = tuple(new_forces)

        params = self.__dict__
        print("Setting up MPM solver")
        print(f"MaterialPoints: {new_material_points.num_points}")
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
    ) -> Tuple[MaterialPoints, Tuple[Force, ...]]:
        # called within solver .update method
        new_forces = []
        for force in forces:
            material_points, new_force = force.apply_on_points(
                material_points=material_points,
                grid=grid,
                step=step,
                dt=self.config.dt,
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
    ) -> Tuple[Grid, Tuple[Force, ...]]:
        # called within solver .update method
        new_forces = []
        for force in forces:
            grid, new_force = force.apply_on_grid(
                material_points=material_points,
                grid=grid,
                step=step,
                dt=self.config.dt,
                dim=self.config.dim,
            )
            new_forces.append(new_force)

        return grid, tuple(new_forces)

    def _update_constitutive_laws(
        self: Self,
        material_points: MaterialPoints,
        constitutive_laws: Tuple[ConstitutiveLaw, ...],
    ) -> Tuple[MaterialPoints, Tuple[ConstitutiveLaw, ...]]:
        # called within solver .update method
        new_materials = []
        for material in constitutive_laws:
            material_points, new_material = material.update(
                material_points=material_points, dt=self.config.dt, dim=self.config.dim
            )
            new_materials.append(new_material)

        return material_points, tuple(new_materials)

    @jax.jit
    def run(self: Self, unroll=False):
        def main_loop(step, solver):
            def save_files(step, name="", **kwargs):
                if len(kwargs) > 0:
                    jnp.savez(
                        f"{self.config.output_path}/{name}.{step.astype(int)}",
                        **kwargs,
                    )

            new_solver = solver.update(step)

            def save_all():
                # output material points

                solver_arrays, material_point_arrays, forces_arrays = self.get_output(
                    new_solver
                )

                jax.debug.callback(
                    save_files, step, "material_points", **material_point_arrays
                )
                jax.debug.callback(save_files, step, "solver", **solver_arrays)

                jax.debug.callback(save_files, step, "forces", **forces_arrays)

                jax.debug.print("[{} / {}]  Saved output", step, self.config.num_steps)

            jax.lax.cond(
                step % self.config.store_every == 0,
                lambda _: save_all(),
                lambda _: None,
                operand=False,
            )

            return new_solver

        return jax.lax.fori_loop(0, self.config.num_steps, main_loop, self)

    def get_output(self, new_solver):
        material_points_output = new_solver.config.output.get("material_points", ())

        material_point_arrays = {}
        for key in material_points_output:
            # workaround around
            # properties of one class depend on properties of another
            output = new_solver.material_points.__getattribute__(key)

            if callable(output):
                output = output(
                    dt=new_solver.config.dt,
                    rho_p=new_solver.constitutive_laws[0].rho_p,
                    d=new_solver.constitutive_laws[0].d,
                    eps_e_stack=new_solver.constitutive_laws[0].eps_e_stack,
                    eps_e_stack_prev=self.constitutive_laws[0].eps_e_stack,
                    W_stack=new_solver.constitutive_laws[0].W_stack,
                )

            material_point_arrays[key] = output

        solver_arrays = {}
        solver_output = new_solver.config.output.get("solver", ())
        for key in solver_output:
            solver_arrays[key] = new_solver.__getattribute__(key)

        forces_arrays = {}
        forces_output = new_solver.config.output.get("forces", ())
        for key in forces_output:
            for force in new_solver.forces:
                key_array = force.__dict__.get(key, None)
                if key_array is not None:
                    forces_arrays[key] = key_array

        return solver_arrays, material_point_arrays, forces_arrays

    def map_p2g(self, X_stack, return_solver=False):
        """Assumes shapefunctions/interactions have already been generated"""

        mass_stack = self.material_points.mass_stack

        def p2g(point_id, shapef, shapef_grad_padded, intr_dist_padded):
            intr_X = X_stack.at[point_id].get()
            intr_mass = mass_stack.at[point_id].get()
            scaled_X = shapef * intr_mass * intr_X

            scaled_mass = shapef * intr_mass
            return scaled_X, scaled_mass

        scaled_X_stack, scaled_mass_stack = self.shape_map.vmap_intr_scatter(p2g)

        zeros_N_mass_stack = jnp.zeros_like(self.grid.mass_stack)

        out_shape = X_stack.shape[1:]
        zero_node_X_stack = jnp.zeros((self.grid.num_cells, *out_shape))

        nodes_mass_stack = zeros_N_mass_stack.at[self.shape_map._intr_hash_stack].add(
            scaled_mass_stack
        )
        nodes_X_stack = zero_node_X_stack.at[self.shape_map._intr_hash_stack].add(
            scaled_X_stack
        )

        def divide(X_generic, mass):
            result = jax.lax.cond(
                mass > self.grid.small_mass_cutoff,
                lambda x: x / mass,
                # lambda x: 0.0 * jnp.zeros_like(x),
                lambda x: jnp.nan * jnp.zeros_like(x),
                X_generic,
            )
            return result

        if return_solver:
            return self, jax.vmap(divide)(nodes_X_stack, nodes_mass_stack)
        return jax.vmap(divide)(nodes_X_stack, nodes_mass_stack)

    def map_p2g2g(self, X_stack=None, return_solver=False):
        new_self, N_stack = self.map_p2g(X_stack, return_solver=True)

        def vmap_intr_g2p(intr_hashes, intr_shapef, intr_shapef_grad, intr_dist_padded):
            return intr_shapef * N_stack.at[intr_hashes].get()

        scaled_N_stack = new_self.shape_map.vmap_intr_gather(vmap_intr_g2p)

        out_shape = N_stack.shape[1:]

        @partial(jax.vmap, in_axes=(0))
        def update_P_stack(scaled_N_stack):
            return jnp.sum(scaled_N_stack, axis=0)

        if return_solver:
            return new_self, update_P_stack(
                scaled_N_stack.reshape(-1, self.shape_map._window_size, *out_shape)
            )
        else:
            return update_P_stack(
                scaled_N_stack.reshape(-1, self.shape_map._window_size, *out_shape)
            )

    @property
    def p2g_position_stack(self):
        return self.grid.position_stack

    @property
    def p2g_position_mesh(self):
        return self.grid.position_mesh

    @property
    def p2g_p_stack(self):
        stress_stack = self.map_p2g(self.material_points.stress_stack)
        return get_pressure_stack(stress_stack)

    @property
    def p2g_q_vm_stack(self):
        stress_stack = self.map_p2g(self.material_points.stress_stack)
        return get_q_vm_stack(stress_stack)

    @property
    def p2g_q_p_stack(self):
        stress_stack = self.map_p2g(self.material_points.stress_stack)
        q_stack = get_q_vm_stack(stress_stack)
        p_stack = get_pressure_stack(stress_stack)
        return q_stack / p_stack

    @property
    def p2g_KE_stack(self):
        KE_stack = self.material_points.KE_stack
        return self.map_p2g(KE_stack)

    @property
    def p2g_KE_stack(self):
        KE_stack = self.material_points.KE_stack
        return self.map_p2g(KE_stack)

    @property
    def p2g_dgamma_dt_stack(self):
        depsdt_stack = self.map_p2g(self.material_points.depsdt_stack)
        return get_scalar_shear_strain_stack(depsdt_stack)

    @property
    def p2g_gamma_stack(self):
        eps_stack = self.map_p2g(self.material_points.eps_stack)
        return get_scalar_shear_strain_stack(eps_stack)

    @property
    def p2g_specific_volume_stack(self):
        specific_volume_stack = self.material_points.specific_volume_stack(
            rho_p=self.constitutive_laws[0].rho_p
        )
        return self.map_p2g(X_stack=specific_volume_stack)

    @property
    def p2g_viscosity_stack(self):
        q_stack = self.p2g_q_vm_stack
        dgamma_dt_stack = self.p2g_dgamma_dt_stack
        return (jnp.sqrt(3) * q_stack) / dgamma_dt_stack

    @property
    def p2g_inertial_number_stack(self):
        pdgamma_dt_stack = self.p2g_dgamma_dt_stack
        p_stack = self.p2g_p_stack
        inertial_number_stack = get_inertial_number_stack(
            p_stack,
            pdgamma_dt_stack,
            p_dia=self.constitutive_laws[0].d,
            rho_p=self.constitutive_laws[0].rho_p,
        )
        # inertial_number_stack = self.material_points.inertial_number_stack(
        #     rho_p=self.constitutive_laws[0].rho_p, d=self.constitutive_laws[0].d
        # )
        return inertial_number_stack

    @property
    def p2g_PE_stack(self):
        PE_stack = self.material_points.PE_stack(
            self.config.dt, self.constitutive_laws[0].W_stack
        )
        return self.map_p2g(X_stack=PE_stack)

    @property
    def p2g_KE_PE_stack(self):
        return self.p2g_KE_stack / self.p2g_PE_stack
