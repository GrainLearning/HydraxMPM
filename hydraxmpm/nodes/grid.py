import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from typing_extensions import Callable, Self, Tuple

from ..common.types import (
    TypeFloat,
    TypeFloatScalarAStack,
    TypeFloatVector3AStack,
    TypeFloatVectorAStack,
    TypeIntScalarAStack,
    TypeUIntScalarAStack,
    TypeFloatVectorPStack,
)
from ..config.mpm_config import MPMConfig
from ..shapefunctions.cubic import vmap_linear_cubicfunction
from ..shapefunctions.linear import vmap_linear_shapefunction

from jaxtyping import UInt


class Grid(eqx.Module):
    """Represents the background grid mappings.

    This class creates the connectivity between particles and grid nodes, and mapping data between them.

    Functionality is intended to be used internally, but can also be called externally.

    Example usage outside core module:
    ```python
    import hydraxmpm as hdx
    import jax.numpy as jnp

    # Create MPM configuration
    config = hdx.MPMConfig(...)

    # Create grid
    grid = hdx.Grid(config)

    # Example particle values
    position_stack = jnp.array([...])
    mass_stack = ...
    velocity_stack = ...

    # option A. make connectivity and perform p2g
    def p2g_func(point_id, intr_shapef, intr_shapef_grad):
        mass = mass_stack[point_id]
        velocity = velocity_stack[point_id]
        scaled_mass = intr_shapef * mass
        scaled_velocity = intr_shapef * velocity
        return scaled_mass, scaled_velocity

    # Combine interaction finding and scattering for efficiency
    intr_scaled_mass, intr_scaled_velocity = grid.vmap_interactions_and_scatter(
        p2g_func, position_stack
    )

    # option B. make connectivity and perform p2g
    # Get interacting nodes based on particle positions
    grid = hdx.get_interactions(grid, position_stack)

    # Scatter particle data to grid using vectorized function
    intr_scaled_mass, intr_scaled_velocity = grid.vmap_intr_scatter(p2g_func)

    # ... (use scattered data for further computations)

    # perform g2p...
    ```

    Attributes:
        config: (:class:`MPMConfig`): MPM configuration object.
        intr_id_stack (jnp.ndarray): ids of interacting nodes for each particle (connectivity information).
        intr_hash_stack: (jnp.ndarray): spatial hash of interacting nodes for lookup.
        intr_shapef_stack: shape function for particle-node interactions.
        intr_shapef_grad_stack: gradients of the shape functions for particle-node interactions.
        intr_dist_stack: distances between particles and interacting nodes (grid coordinates)

    """

    config: MPMConfig = eqx.field(static=True)

    shapefunction_call: Callable = eqx.field(init=False, static=True)

    intr_id_stack: TypeUIntScalarAStack = eqx.field(init=False)
    intr_hash_stack: TypeUIntScalarAStack = eqx.field(init=False)
    intr_shapef_stack: TypeFloatScalarAStack = eqx.field(init=False)
    intr_shapef_grad_stack: TypeFloatVector3AStack = eqx.field(init=False)
    intr_dist_stack: TypeFloatVector3AStack = eqx.field(init=False)

    def __init__(self, config: MPMConfig) -> Self:
        self.config = config
        # post init
        self.intr_id_stack = jnp.arange(
            self.config.num_points * self.config.window_size, device=self.config.device
        ).astype(jnp.uint32)

        self.intr_hash_stack = jnp.zeros(
            self.config.num_points * self.config.window_size, device=self.config.device
        ).astype(jnp.uint32)

        self.intr_dist_stack = jnp.zeros(
            (self.config.num_points * self.config.window_size, 3),
            device=self.config.device,
        )  # 3D needed for APIC / AFLIP

        self.intr_shapef_stack = jnp.zeros(
            (self.config.num_points * self.config.window_size),
            device=self.config.device,
        )
        self.intr_shapef_grad_stack = jnp.zeros(
            (self.config.num_points * self.config.window_size, 3),
            device=self.config.device,
        )

        if self.config.shapefunction == "linear":
            self.shapefunction_call = vmap_linear_shapefunction
        elif self.config.shapefunction == "cubic":
            self.shapefunction_call = vmap_linear_cubicfunction

    def get_interactions(self, position_stack: TypeFloatVectorPStack) -> Self:
        def vmap_intr(intr_id: chex.ArrayBatched) -> Tuple[chex.Array, chex.Array]:
            point_id = (intr_id / self.config.window_size).astype(jnp.uint32)

            stencil_id = (intr_id % self.config.window_size).astype(jnp.uint16)

            # Relative position of the particle to the node.
            particle_pos = position_stack.at[point_id].get()

            rel_pos = (
                particle_pos - jnp.array(self.config.origin)
            ) * self.config.inv_cell_size

            stencil_pos = jnp.array(self.config.forward_window).at[stencil_id].get()

            intr_grid_pos = jnp.floor(rel_pos) + stencil_pos

            intr_hash = jnp.ravel_multi_index(
                intr_grid_pos.astype(jnp.int32), self.config.grid_size, mode="wrap"
            )

            intr_dist = rel_pos - intr_grid_pos

            shapef, shapef_grad_padded = self.shapefunction_call(intr_dist, self.config)

            # is there a more efficient way to do this?
            intr_dist_padded = jnp.pad(
                intr_dist,
                self.config.padding,
                mode="constant",
                constant_values=0.0,
            )

            # transform to grid coordinates
            intr_dist_padded = -1.0 * intr_dist_padded * self.config.cell_size

            return intr_dist_padded, intr_hash, shapef, shapef_grad_padded

        (
            new_intr_dist_stack,
            new_intr_hash_stack,
            new_intr_shapef_stack,
            new_intr_shapef_grad_stack,
        ) = jax.vmap(vmap_intr)(self.intr_id_stack)

        return eqx.tree_at(
            lambda state: (
                state.intr_dist_stack,
                state.intr_hash_stack,
                state.intr_shapef_stack,
                state.intr_shapef_grad_stack,
            ),
            self,
            (
                new_intr_dist_stack,
                new_intr_hash_stack,
                new_intr_shapef_stack,
                new_intr_shapef_grad_stack,
            ),
        )

    def vmap_interactions_and_scatter(
        self, p2g_func: Callable, position_stack: TypeFloatVectorPStack
    ) -> Self:
        def vmap_intr(intr_id: UInt) -> Tuple[chex.Array, chex.Array]:
            point_id = (intr_id / self.config.window_size).astype(jnp.uint32)

            stencil_id = (intr_id % self.config.window_size).astype(jnp.uint16)

            # Relative position of the particle to the node.
            particle_pos = position_stack.at[point_id].get()

            rel_pos = (
                particle_pos - jnp.array(self.config.origin)
            ) * self.config.inv_cell_size

            stencil_pos = jnp.array(self.config.forward_window).at[stencil_id].get()

            intr_grid_pos = jnp.floor(rel_pos) + stencil_pos

            intr_hash = jnp.ravel_multi_index(
                intr_grid_pos.astype(jnp.int32), self.config.grid_size, mode="wrap"
            )

            intr_dist = rel_pos - intr_grid_pos

            shapef, shapef_grad_padded = self.shapefunction_call(intr_dist, self.config)

            shapef = jax.lax.cond(
                ((intr_hash < 0) | (intr_hash >= self.config.num_cells)),
                lambda x: 0.0,
                lambda x: x,
                shapef,
            )

            # is there a more efficient way to do this?
            intr_dist_padded = jnp.pad(
                intr_dist,
                self.config.padding,
                mode="constant",
                constant_values=0.0,
            )

            # transform to grid coordinates
            intr_dist_padded = -1.0 * intr_dist_padded * self.config.cell_size

            out_stack = p2g_func(point_id, shapef, shapef_grad_padded, intr_dist_padded)

            return intr_dist_padded, intr_hash, shapef, shapef_grad_padded, out_stack

        (
            new_intr_dist_stack,
            new_intr_hash_stack,
            new_intr_shapef_stack,
            new_intr_shapef_grad_stack,
            out_stack,
        ) = jax.vmap(vmap_intr)(self.intr_id_stack)

        return eqx.tree_at(
            lambda state: (
                state.intr_dist_stack,
                state.intr_hash_stack,
                state.intr_shapef_stack,
                state.intr_shapef_grad_stack,
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
        """map particle to grid, not relative distances not included in mapping"""

        def vmap_p2g(intr_id, intr_shapef, intr_shapef_grad):
            point_id = (intr_id / self.config.window_size).astype(jnp.uint32)
            return p2g_func(point_id, intr_shapef, intr_shapef_grad)

        return jax.vmap(vmap_p2g)(
            self.intr_id_stack, self.intr_shapef_stack, self.intr_shapef_grad_stack
        )

    def vmap_intr_gather(self, g2p_func: Callable):
        def vmap_g2p(intr_hash, intr_shapef, intr_shapef_grad):
            return g2p_func(intr_hash, intr_shapef, intr_shapef_grad)

        return jax.vmap(vmap_g2p)(
            self.intr_hash_stack, self.intr_shapef_stack, self.intr_shapef_grad_stack
        )

    # adding relative distance
    def vmap_intr_scatter_dist(self, p2g_func: Callable):
        def vmap_p2g(intr_id, intr_shapef, intr_shapef_grad, intr_dist):
            point_id = (intr_id / self.config.window_size).astype(jnp.uint32)
            return p2g_func(point_id, intr_shapef, intr_shapef_grad, intr_dist)

        return jax.vmap(vmap_p2g)(
            self.intr_id_stack,
            self.intr_shapef_stack,
            self.intr_shapef_grad_stack,
            self.intr_dist_stack,
        )

    def vmap_intr_gather_dist(self, g2p_func: Callable):
        def vmap_g2p(intr_hash, intr_shapef, intr_shapef_grad, intr_dist):
            return g2p_func(intr_hash, intr_shapef, intr_shapef_grad, intr_dist)

        return jax.vmap(vmap_g2p)(
            self.intr_hash_stack,
            self.intr_shapef_stack,
            self.intr_shapef_grad_stack,
            self.intr_dist_stack,
        )
