from functools import partial
from typing import Callable, Optional, Self, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from ..common.base import Base
from ..common.types import (
    TypeFloatScalarAStack,
    TypeFloatVector3AStack,
    TypeInt,
    TypeUInt,
    TypeUIntScalarAStack,
)
from ..constitutive_laws.constitutive_law import ConstitutiveLaw
from ..forces.force import Force
from ..grid.grid import Grid
from ..forces.boundary import Boundary
from ..material_points.material_points import MaterialPoints
from ..shapefunctions.cubic import vmap_linear_cubicfunction
from ..shapefunctions.linear import vmap_linear_shapefunction
from ..solvers.config import Config
from ..utils.math_helpers import (
    get_hencky_strain_stack,
    get_pressure_stack,
    get_q_vm_stack,
    get_scalar_shear_strain_stack,
    get_strain_rate_from_L_stack,
)


def _numpy_tuple_deep(x) -> tuple:
    return tuple(map(tuple, jnp.array(x).tolist()))


# dictionary defnitions to lookup some shape functions
shapefunction_definitions = {
    "linear": vmap_linear_shapefunction,
    "cubic": vmap_linear_cubicfunction,
}
shapefunction_nodal_positions_1D = {
    "linear": jnp.arange(2),
    "cubic": jnp.arange(4) - 1,  # center point in middle
}


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

    # Node-particle connectivity (interactions, shapefunctions, etc.)
    _shapefunction_call: Callable = eqx.field(init=False, static=True)
    _intr_id_stack: TypeUIntScalarAStack = eqx.field(init=False)
    _intr_hash_stack: TypeUIntScalarAStack = eqx.field(init=False)
    _intr_shapef_stack: TypeFloatScalarAStack = eqx.field(init=False)
    _intr_shapef_grad_stack: TypeFloatVector3AStack = eqx.field(init=False)
    _intr_dist_stack: TypeFloatVector3AStack = eqx.field(init=False)
    _forward_window: tuple = eqx.field(
        repr=False, init=False, static=True, converter=lambda x: _numpy_tuple_deep(x)
    )
    _backward_window: tuple = eqx.field(
        repr=False, init=False, static=True, converter=lambda x: _numpy_tuple_deep(x)
    )
    _window_size: int = eqx.field(init=False, static=True)

    _setup_done: bool = eqx.field(default=False)

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

        # Set connectivity and shape function
        self._shapefunction_call = shapefunction_definitions[self.config.shapefunction]
        window_1D = shapefunction_nodal_positions_1D[self.config.shapefunction]

        self._forward_window = jnp.array(
            jnp.meshgrid(*[window_1D] * self.config.dim)
        ).T.reshape(-1, self.config.dim)

        self._backward_window = self._forward_window[::-1] - 1
        self._window_size = len(self._backward_window)

        self._intr_shapef_stack = jnp.zeros(
            self.material_points.num_points * self._window_size
        )
        self._intr_shapef_grad_stack = jnp.zeros(
            (self.material_points.num_points * self._window_size, 3)
        )

        self._intr_dist_stack = jnp.zeros(
            (self.material_points.num_points * self._window_size, 3)
        )  #  needed for APIC / AFLIP

        self._intr_id_stack = jnp.arange(
            self.material_points.num_points * self._window_size
        ).astype(jnp.uint32)

        self._intr_hash_stack = jnp.zeros(
            self.material_points.num_points * self._window_size
        ).astype(jnp.uint32)

    def setup(self: Self, **kwargs) -> Self:
        # we run this once after initialization
        if self._setup_done:
            return self

        # initialize pressure and density...
        new_constitutive_laws = []
        new_material_points = self.material_points
        new_material_points.init_volume_from_cellsize(
            self.grid.cell_size, self.config.ppc
        )
        for constitutive_law in self.constitutive_laws:
            new_constitutive_law, new_material_points = constitutive_law.init_state(
                self.material_points
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

    def _get_particle_grid_interaction(
        self: Self,
        intr_id: TypeUInt,
        material_points: MaterialPoints,
        grid: Grid,
        return_point_id=False,
    ):
        # Create mapping between material_points and grid nodes.
        # Shape functions, and connectivity information are calculated here

        point_id = (intr_id / self._window_size).astype(jnp.uint32)

        stencil_id = (intr_id % self._window_size).astype(jnp.uint16)

        # Relative position of the particle to the node.
        particle_pos = material_points.position_stack.at[point_id].get()

        rel_pos = (particle_pos - jnp.array(grid.origin)) * grid._inv_cell_size

        stencil_pos = jnp.array(self._forward_window).at[stencil_id].get()

        intr_grid_pos = jnp.floor(rel_pos) + stencil_pos

        intr_hash = jnp.ravel_multi_index(
            intr_grid_pos.astype(jnp.int32), grid.grid_size, mode="wrap"
        ).astype(jnp.uint32)

        intr_dist = rel_pos - intr_grid_pos

        shapef, shapef_grad_padded = self._shapefunction_call(
            intr_dist, grid._inv_cell_size, self.config.dim, self.config._padding
        )

        # is there a more efficient way to do this?
        intr_dist_padded = jnp.pad(
            intr_dist,
            self.config._padding,
            mode="constant",
            constant_values=0.0,
        )

        # transform to grid coordinates
        intr_dist_padded = -1.0 * intr_dist_padded * grid.cell_size

        if return_point_id:
            return (
                intr_dist_padded,
                intr_hash,
                shapef,
                shapef_grad_padded,
                point_id,
            )
        return intr_dist_padded, intr_hash, shapef, shapef_grad_padded

    def _get_particle_grid_interactions_batched(self):
        """get particle grid interactions / shapefunctions
        Batched version of get_interaction."""
        (
            new_intr_dist_stack,
            new_intr_hash_stack,
            new_intr_shapef_stack,
            new_intr_shapef_grad_stack,
        ) = jax.vmap(
            self._get_particle_grid_interaction, in_axes=(0, None, None, None)
        )(self._intr_id_stack, self.material_points, self.grid, False)

        return eqx.tree_at(
            lambda state: (
                state._intr_dist_stack,
                state._intr_hash_stack,
                state._intr_shapef_stack,
                state._intr_shapef_grad_stack,
            ),
            self,
            (
                new_intr_dist_stack,
                new_intr_hash_stack,
                new_intr_shapef_stack,
                new_intr_shapef_grad_stack,
            ),
        )

    # particle to grid, get interactions
    def vmap_interactions_and_scatter(self, p2g_func: Callable):
        """Map particle to grid, also gets interaction data"""

        @jax.checkpoint
        def vmap_intr(intr_id: TypeUInt):
            intr_dist_padded, intr_hash, shapef, shapef_grad_padded, point_id = (
                self._get_particle_grid_interaction(
                    intr_id, self.material_points, self.grid, return_point_id=True
                )
            )

            out_stack = p2g_func(point_id, shapef, shapef_grad_padded, intr_dist_padded)

            return intr_dist_padded, intr_hash, shapef, shapef_grad_padded, out_stack

        (
            new_intr_dist_stack,
            new_intr_hash_stack,
            new_intr_shapef_stack,
            new_intr_shapef_grad_stack,
            out_stack,
        ) = jax.vmap(vmap_intr)(self._intr_id_stack)

        return eqx.tree_at(
            lambda state: (
                state._intr_dist_stack,
                state._intr_hash_stack,
                state._intr_shapef_stack,
                state._intr_shapef_grad_stack,
            ),
            self,
            (
                new_intr_dist_stack,
                new_intr_hash_stack,
                new_intr_shapef_stack,
                new_intr_shapef_grad_stack,
            ),
        ), out_stack

    def vmap_intr_scatter(self, p2g_func: Callable):
        """map particle to grid, does not get interaction data with relative distance"""

        def vmap_p2g(intr_id, intr_shapef, intr_shapef_grad, intr_dist):
            point_id = (intr_id / self._window_size).astype(jnp.uint32)
            return p2g_func(point_id, intr_shapef, intr_shapef_grad, intr_dist)

        return jax.vmap(vmap_p2g)(
            self._intr_id_stack,
            self._intr_shapef_stack,
            self._intr_shapef_grad_stack,
            self._intr_dist_stack,  # relative distance node coordinates
        )

    # Grid to particle
    def vmap_intr_gather(self, g2p_func: Callable):
        def vmap_g2p(intr_hash, intr_shapef, intr_shapef_grad, intr_dist):
            return g2p_func(intr_hash, intr_shapef, intr_shapef_grad, intr_dist)

        return jax.vmap(vmap_g2p)(
            self._intr_hash_stack,
            self._intr_shapef_stack,
            self._intr_shapef_grad_stack,
            self._intr_dist_stack,
        )

    @jax.jit
    def run(self: Self, unroll=False):
        def main_loop(step, solver):
            new_solver = solver.update(step)
            return new_solver

        # @scan_tqdm(self.config.num_steps / self.config.store_every)
        def scan_fn(solver, step):
            step_next = step + self.config.store_every
            new_solver = jax.lax.fori_loop(step, step_next, main_loop, solver)
            for callback in self.callbacks:
                callback(new_solver, step_next)

            return new_solver, []

        xs = jnp.arange(0, self.config.num_steps, self.config.store_every).astype(
            jnp.int32
        )

        carry, _ = jax.lax.scan(
            scan_fn,
            self,
            xs=xs,
            unroll=unroll,
        )

        return carry

    def map_p2g(self, key, X_stack=None, return_solver=False):
        """Assumes shapefunctions/interactions have already been generated"""
        if X_stack is None:
            X_stack = self.material_points.__getattribute__(key)
        mass_stack = self.material_points.mass_stack

        def p2g(point_id, shapef, shapef_grad_padded, intr_dist_padded):
            intr_X = X_stack.at[point_id].get()
            intr_mass = mass_stack.at[point_id].get()
            scaled_X = shapef * intr_mass * intr_X

            scaled_mass = shapef * intr_mass
            return scaled_X, scaled_mass

        new_self, (scaled_X_stack, scaled_mass_stack) = (
            self.vmap_interactions_and_scatter(p2g)
        )

        zeros_N_mass_stack = jnp.zeros_like(new_self.grid.mass_stack)

        out_shape = X_stack.shape[1:]
        zero_node_X_stack = jnp.zeros((new_self.grid.num_cells, *out_shape))

        nodes_mass_stack = zeros_N_mass_stack.at[new_self._intr_hash_stack].add(
            scaled_mass_stack
        )
        nodes_X_stack = zero_node_X_stack.at[new_self._intr_hash_stack].add(
            scaled_X_stack
        )

        def divide(X_generic, mass):
            result = jax.lax.cond(
                mass > new_self.grid.small_mass_cutoff,
                lambda x: x / mass,
                # lambda x: 0.0 * jnp.zeros_like(x),
                lambda x: jnp.nan * jnp.zeros_like(x),
                X_generic,
            )
            return result

        if return_solver:
            return new_self, jax.vmap(divide)(nodes_X_stack, nodes_mass_stack)
        return jax.vmap(divide)(nodes_X_stack, nodes_mass_stack)

    def map_p2g2g(self, key, X_stack=None, return_solver=False):
        new_self, N_stack = self.map_p2g(key, X_stack, return_solver=True)

        def vmap_intr_g2p(intr_hashes, intr_shapef, intr_shapef_grad, intr_dist_padded):
            return intr_shapef * N_stack.at[intr_hashes].get()

        scaled_N_stack = new_self.vmap_intr_gather(vmap_intr_g2p)

        out_shape = N_stack.shape[1:]

        @partial(jax.vmap, in_axes=(0))
        def update_P_stack(scaled_N_stack):
            return jnp.sum(scaled_N_stack, axis=0)

        if return_solver:
            return new_self, update_P_stack(
                scaled_N_stack.reshape(-1, self._window_size, *out_shape)
            )
        else:
            return update_P_stack(
                scaled_N_stack.reshape(-1, self._window_size, *out_shape)
            )

    @property
    def p2g_p_stack(self):
        stress_stack = self.map_p2g("stress_stack")
        return get_pressure_stack(stress_stack)

    @property
    def p2g_q_vm_stack(self):
        stress_stack = self.map_p2g("stress_stack")
        return get_q_vm_stack(stress_stack)

    @property
    def p2g_q_p_stack(self):
        stress_stack = self.map_p2g("stress_stack")
        q_stack = get_q_vm_stack(stress_stack)
        p_stack = get_pressure_stack(stress_stack)
        return q_stack / p_stack

    @property
    def p2g_KE_stack(self):
        KE_stack = self.map_p2g("KE_stack")
        return KE_stack

    @property
    def p2g_dgamma_dt_stack(self):
        depsdt_stack = self.map_p2g("depsdt_stack")
        dgammadt_stack = get_scalar_shear_strain_stack(depsdt_stack)
        return dgammadt_stack

    @property
    def p2g_gamma_stack(self):
        eps_stack = self.map_p2g("eps_stack")
        return get_scalar_shear_strain_stack(eps_stack)

    # @property
    # TODO
    # def p2g_phi_stack(self):
    #     return self.map_p2g("phi_stack")
