from typing_extensions import Self, Tuple, Callable

import chex
import jax
import jax.numpy as jnp
from functools import partial

from ..config.mpm_config import MPMConfig

import equinox as eqx

from ..shapefunctions.linear import vmap_linear_shapefunction
from ..shapefunctions.cubic import vmap_linear_cubicfunction
from ..shapefunctions.shapefunctions import SHAPEFUNCTION


class Grid(eqx.Module):
    intr_id_stack: jax.Array
    intr_shapef_stack: chex.Array
    intr_hash_stack: chex.Array
    intr_shapef_grad_stack: chex.Array
    intr_dist_stack: chex.Array

    shapefunction_call: Callable = eqx.field(static=True)

    config: MPMConfig = eqx.field(static=True)

    def __init__(
        self: Self,
        config: MPMConfig = None,
    ) -> Self:
        self.config = config

        self.intr_id_stack = jnp.arange(config.num_points * config.window_size).astype(
            jnp.uint32
        )

        self.intr_hash_stack = jnp.zeros(config.num_points * config.window_size).astype(
            jnp.uint32
        )

        self.intr_dist_stack = jnp.zeros(
            (config.num_points * config.window_size, 3)
        )  # 3D needed for APIC / AFLIP

        self.intr_shapef_stack = jnp.zeros((config.num_points * config.window_size))
        self.intr_shapef_grad_stack = jnp.zeros(
            (config.num_points * config.window_size, 3)
        )

        if config.shapefunction == SHAPEFUNCTION.linear:
            self.shapefunction_call = vmap_linear_shapefunction
        elif config.shapefunction == SHAPEFUNCTION.cubic:
            self.shapefunction_call = vmap_linear_cubicfunction

    def get_interactions(self, position_stack: chex.Array) -> Self:
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
                intr_grid_pos.astype(jnp.uint32), self.config.grid_size, mode="wrap"
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

    def vmap_intr_scatter(self, p2g_func: Callable):
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
            return g2p_func(intr_hash, intr_shapef, intr_shapef_grad,intr_dist)

        return jax.vmap(vmap_g2p)(
            self.intr_hash_stack,
            self.intr_shapef_stack,
            self.intr_shapef_grad_stack,
            self.intr_dist_stack,
        )
