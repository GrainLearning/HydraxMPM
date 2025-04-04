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
    TypeFloatVector,
    TypeFloat,
    TypeUInt,
    TypeUIntScalarAStack,
)

from ..grid.grid import Grid
from ..material_points.material_points import MaterialPoints
from .cubic import vmap_cubic_shapefunction

from .quadratic import vmap_quadratic_shapefunction
from .linear import vmap_linear_shapefunction


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


# dictionary defnitions to lookup some shape functions
shapefunction_definitions = {
    "linear": vmap_linear_shapefunction,
    "quadratic": vmap_quadratic_shapefunction,
    "cubic": vmap_cubic_shapefunction,
}
shapefunction_nodal_positions_1D = {
    "linear": jnp.arange(2),
    "quadratic": jnp.arange(4) - 1,
    "cubic": jnp.arange(4) - 1,
}


class ShapeFunctionMapping(Base):
    """
    Mapping of shape functions between material points and grid.
    """

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

    dim: int = eqx.field(static=True)  # config

    # internal variables
    _padding: tuple = eqx.field(init=False, static=True, repr=False)

    def __init__(
        self,
        shapefunction: str,
        num_points: int,
        dim: int,
        **kwargs,
    ) -> Self:
        # Set connectivity and shape function
        self._shapefunction_call = shapefunction_definitions[shapefunction]
        window_1D = shapefunction_nodal_positions_1D[shapefunction]

        self._forward_window = jnp.array(jnp.meshgrid(*[window_1D] * dim)).T.reshape(
            -1, dim
        )

        self._backward_window = self._forward_window[::-1] - 1
        self._window_size = len(self._backward_window)

        self._intr_shapef_stack = jnp.zeros(num_points * self._window_size)
        self._intr_shapef_grad_stack = jnp.zeros((num_points * self._window_size, 3))

        self._intr_dist_stack = jnp.zeros(
            (num_points * self._window_size, 3)
        )  #  needed for APIC / AFLIP

        self._intr_id_stack = jnp.arange(num_points * self._window_size).astype(
            jnp.uint32
        )

        self._intr_hash_stack = jnp.zeros(num_points * self._window_size).astype(
            jnp.uint32
        )
        self.dim = dim
        self._padding = (0, 3 - self.dim)

    def _get_particle_grid_interaction(
        self: Self,
        intr_id: TypeUInt,
        node_type_stack: TypeUIntScalarAStack,
        position_stack: jnp.array,  # workaround type checking
        origin: TypeFloatVector,
        _inv_cell_size: TypeFloat,
        grid_size: tuple,
        return_point_id=False,
    ):
        # Create mapping between material_points and grid nodes.
        # Shape functions, and connectivity information are calculated here

        point_id = (intr_id / self._window_size).astype(jnp.uint32)

        stencil_id = (intr_id % self._window_size).astype(jnp.uint16)

        # Relative position of the particle to the node.
        particle_pos = position_stack.at[point_id].get()

        rel_pos = (particle_pos - origin) * _inv_cell_size

        stencil_pos = jnp.array(self._forward_window).at[stencil_id].get()

        intr_grid_pos = jnp.floor(rel_pos) + stencil_pos

        intr_hash = jnp.ravel_multi_index(
            intr_grid_pos.astype(jnp.int32), grid_size, mode="wrap"
        ).astype(jnp.uint32)

        intr_node_type = node_type_stack.at[intr_hash].get()

        intr_dist = rel_pos - intr_grid_pos

        shapef, shapef_grad_padded = self._shapefunction_call(
            intr_dist,
            _inv_cell_size,
            self.dim,
            self._padding,
            # intr_node_type=intr_node_type,
            intr_node_type=0,
        )

        # is there a more efficient way to do this?
        intr_dist_padded = jnp.pad(
            intr_dist,
            self._padding,
            mode="constant",
            constant_values=0.0,
        )

        # transform to grid coordinates
        intr_dist_padded = -1.0 * intr_dist_padded * (1.0 / _inv_cell_size)

        if return_point_id:
            return (
                intr_dist_padded,
                intr_hash,
                shapef,
                shapef_grad_padded,
                point_id,
            )
        return intr_dist_padded, intr_hash, shapef, shapef_grad_padded

    def _get_particle_grid_interactions_batched(
        self, material_points: MaterialPoints, grid: Grid
    ):
        """get particle grid interactions / shapefunctions
        Batched version of get_interaction."""
        (
            new_intr_dist_stack,
            new_intr_hash_stack,
            new_intr_shapef_stack,
            new_intr_shapef_grad_stack,
        ) = jax.vmap(
            self._get_particle_grid_interaction,
            in_axes=(0, None, None, None, None, None, None),
        )(
            self._intr_id_stack,
            grid.type_stack.reshape(-1),
            material_points.position_stack,
            jnp.array(grid.origin),
            grid._inv_cell_size,
            grid.grid_size,
            False,
        )

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
    def vmap_interactions_and_scatter(
        self,
        p2g_func: Callable,
        material_points: MaterialPoints = None,
        grid: Grid = None,
        position_stack: TypeFloatVectorAStack = None,
    ):
        """Map particle to grid, also gets interaction data"""

        if material_points is not None:
            position_stack = material_points.position_stack

        def vmap_intr(intr_id: TypeUInt):
            intr_dist_padded, intr_hash, shapef, shapef_grad_padded, point_id = (
                self._get_particle_grid_interaction(
                    intr_id,
                    grid.type_stack.reshape(-1),
                    position_stack,
                    jnp.array(grid.origin),
                    grid._inv_cell_size,
                    grid.grid_size,
                    return_point_id=True,
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

    def map_p2g(self, X_stack, mass_stack=None, grid=None, return_self=False):
        """Assumes shapefunctions/interactions have already been generated"""

        def p2g(point_id, shapef, shapef_grad_padded, intr_dist_padded):
            intr_X = X_stack.at[point_id].get()
            intr_mass = mass_stack.at[point_id].get()
            scaled_X = shapef * intr_mass * intr_X

            scaled_mass = shapef * intr_mass
            return scaled_X, scaled_mass

        scaled_X_stack, scaled_mass_stack = self.vmap_intr_scatter(p2g)

        zeros_N_mass_stack = jnp.zeros_like(grid.mass_stack)

        out_shape = X_stack.shape[1:]
        zero_node_X_stack = jnp.zeros((grid.num_cells, *out_shape))

        nodes_mass_stack = zeros_N_mass_stack.at[self._intr_hash_stack].add(
            scaled_mass_stack
        )
        nodes_X_stack = zero_node_X_stack.at[self._intr_hash_stack].add(scaled_X_stack)

        def divide(X_generic, mass):
            result = jax.lax.cond(
                mass > grid.small_mass_cutoff,
                lambda x: x / mass,
                # lambda x: 0.0 * jnp.zeros_like(x),
                lambda x: jnp.nan * jnp.zeros_like(x),
                X_generic,
            )
            return result

        if return_self:
            return self, jax.vmap(divide)(nodes_X_stack, nodes_mass_stack)
        return jax.vmap(divide)(nodes_X_stack, nodes_mass_stack)

    def map_p2g2g(self, X_stack, mass_stack=None, grid=None, return_self=False):
        new_self, N_stack = self.map_p2g(X_stack, mass_stack, grid, return_self=True)

        def vmap_intr_g2p(intr_hashes, intr_shapef, intr_shapef_grad, intr_dist_padded):
            return intr_shapef * N_stack.at[intr_hashes].get()

        scaled_N_stack = new_self.vmap_intr_gather(vmap_intr_g2p)

        out_shape = N_stack.shape[1:]

        @partial(jax.vmap, in_axes=(0))
        def update_P_stack(scaled_N_stack):
            return jnp.sum(scaled_N_stack, axis=0)

        if return_self:
            return new_self, update_P_stack(
                scaled_N_stack.reshape(-1, self._window_size, *out_shape)
            )
        else:
            return update_P_stack(
                scaled_N_stack.reshape(-1, self._window_size, *out_shape)
            )

    def grid_position_stack(self, grid, **kwargs):
        return grid.position_stack

    def grid_mesh(self, grid, **kwargs):
        return grid.position_mesh

    def p2g_p_stack(self, material_points, grid, **kwargs):
        stress_stack = self.map_p2g(
            material_points.stress_stack,
            material_points.mass_stack,
            grid,
        )
        return get_pressure_stack(stress_stack)

    def p2g_q_stack(self, material_points, grid, **kwargs):
        stress_stack = self.map_p2g(
            material_points.stress_stack,
            material_points.mass_stack,
            grid,
        )
        return get_q_vm_stack(stress_stack)

    def p2g_q_p_stack(self, material_points, grid, **kwargs):
        stress_stack = self.map_p2g(
            material_points.stress_stack,
            material_points.mass_stack,
            grid,
        )
        q_stack = get_q_vm_stack(stress_stack)
        p_stack = get_pressure_stack(stress_stack)
        return q_stack / p_stack

    def p2g_KE_stack(self, material_points, grid, **kwargs):
        return self.map_p2g(
            material_points.KE_stack,
            material_points.mass_stack,
            grid,
        )

    def p2g_dgamma_dt_stack(self, material_points, grid, **kwargs):
        depsdt_stack = self.map_p2g(
            material_points.depsdt_stack,
            material_points.mass_stack,
            grid,
        )
        return get_scalar_shear_strain_stack(depsdt_stack)

    def p2g_positions(self, material_points, grid, **kwargs):
        return self.map_p2g(
            material_points.position_stack,
            material_points.mass_stack,
            grid,
        )

    def p2g_positions_intrp(self, position_stack, grid, **kwargs):
        return self.map_p2g(
            position_stack,
            kwargs.get("mass_stack", jnp.ones(position_stack.shape[0])),
            grid,
        )

    def p2g_gamma_stack(self, material_points, grid, **kwargs):
        eps_stack = self.map_p2g(
            material_points.eps_stack,
            material_points.mass_stack,
            grid,
        )
        return get_scalar_shear_strain_stack(eps_stack)


# @property
# def p2g_specific_volume_stack(self):
#     specific_volume_stack = self.material_points.specific_volume_stack(
#         rho_p=self.constitutive_laws[0].rho_p
#     )
#     return self.map_p2g(X_stack=specific_volume_stack)

# @property
# def p2g_viscosity_stack(self):
#     q_stack = self.p2g_q_vm_stack
#     dgamma_dt_stack = self.p2g_dgamma_dt_stack
#     return (jnp.sqrt(3) * q_stack) / dgamma_dt_stack

# @property
# def p2g_inertial_number_stack(self):
#     pdgamma_dt_stack = self.p2g_dgamma_dt_stack
#     p_stack = self.p2g_p_stack
#     inertial_number_stack = get_inertial_number_stack(
#         p_stack,
#         pdgamma_dt_stack,
#         p_dia=self.constitutive_laws[0].d,
#         rho_p=self.constitutive_laws[0].rho_p,
#     )

#     return inertial_number_stack

# @property
# def p2g_PE_stack(self):
#     PE_stack = self.material_points.PE_stack(
#         self.dt,
#         self.constitutive_laws[0].W_stack,
#     )
#     return self.map_p2g(X_stack=PE_stack)

# @property
# def p2g_KE_PE_stack(self):
#     return self.p2g_KE_stack / self.p2g_PE_stack
