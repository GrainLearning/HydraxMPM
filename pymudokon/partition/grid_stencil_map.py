"""State and functions for the background MPM grid nodes."""
# TODO: Add support for Sparse grid. This feature is currently experimental in JAX.

import jax.experimental
from typing_extensions import Self

from jax.experimental import pallas as pl

import chex
import jax
import jax.numpy as jnp
from functools import partial
import numpy as np

from ..config.mpm_config import MPMConfig

import equinox as eqx


def get_hash_id(grid_pos, grid_size):
    return jnp.ravel_multi_index(grid_pos, (grid_size), mode="wrap")
0

class GridStencilMap(eqx.Module):
    cell_start_stack: chex.Array
    cell_count_stack: chex.Array
    cell_id_stack: chex.Array
    point_hash_stack: chex.Array
    point_id_stack: chex.Array

    origin: tuple = eqx.field(static=True, converter=lambda x: tuple(x))
    grid_size: tuple = eqx.field(static=True, converter=lambda x: tuple(x))
    inv_cell_size: float = eqx.field(static=True, converter=lambda x: float(x))
    num_points: float = eqx.field(static=True, converter=lambda x: int(x))
    num_cells: float = eqx.field(static=True, converter=lambda x: int(x))

    window_size: int = eqx.field(static=True, converter=lambda x: int(x))
    forward_window: tuple = eqx.field(
        static=True, converter=lambda x: tuple(map(tuple, x))
    )
    backward_window: tuple = eqx.field(
        static=True, converter=lambda x: tuple(map(tuple, x))
    )
    
    unroll_grid_kernels: bool = eqx.field(static=True, converter=lambda x: bool(x))
    

    def __init__(
        self: Self,
        config: MPMConfig = None,
        origin: tuple = None,
        grid_size: tuple = None,
        num_cells: int = None,
        num_points: int = None,
        cell_size: float = None,
        forward_window: int = None,
        backward_window: int = None,
        unroll_grid_kernels: bool = True
    ) -> Self:
        if config:
            num_cells = config.num_cells
            num_points = config.num_points
            origin = config.origin
            cell_size = config.cell_size
            grid_size = config.grid_size
            forward_window = config.forward_window
            backward_window = config.backward_window
            unroll_grid_kernels = config.unroll_grid_kernels

        self.cell_start_stack = jnp.zeros(num_cells).astype(jnp.int32)
        self.cell_count_stack = jnp.zeros(num_cells).astype(jnp.int32)
        self.point_hash_stack = jnp.zeros(num_points).astype(jnp.int32)
        self.point_id_stack = jnp.arange(num_points).astype(jnp.int32)
        self.cell_id_stack = jnp.arange(num_cells).astype(jnp.int32)

        self.origin = origin
        self.inv_cell_size = 1.0 / cell_size
        self.grid_size = grid_size
        self.num_cells = num_cells
        self.num_points = num_points
        self.window_size = len(forward_window)
        self.backward_window = backward_window
        self.forward_window = forward_window
        
        self.unroll_grid_kernels = unroll_grid_kernels

    def get_hash(self, position_stack):
        @partial(jax.vmap)
        def vmap_get_hash(position):
            rel_pos = (position - jnp.array(self.origin)) * self.inv_cell_size

            grid_pos = jnp.floor(rel_pos).astype(jnp.int32)

            return get_hash_id(grid_pos, self.grid_size)

        return vmap_get_hash(position_stack)

    @partial(jax.jit)
    def partition(self, position_stack):
        new_point_hash_stack = self.get_hash(position_stack)

        new_cell_count_stack = (
            jnp.zeros(self.num_cells, dtype=jnp.int32).at[new_point_hash_stack].add(1)
        )

        new_point_id_stack = jnp.argsort(new_point_hash_stack)

        new_cell_start_stack = jnp.searchsorted(
            a=new_point_hash_stack,
            v=self.cell_id_stack,
            sorter=new_point_id_stack,
            method="sort",
        )

        return eqx.tree_at(
            lambda state: (
                state.point_hash_stack,
                state.cell_start_stack,
                state.cell_count_stack,
                state.point_id_stack,
            ),
            self,
            (
                new_point_hash_stack,
                new_cell_start_stack,
                new_cell_count_stack,
                new_point_id_stack,
            ),
        )

    def vmap_grid_scatter_fori(self, g2p_func, init_vals, is_grid_hash=True):
        @partial(jax.vmap, in_axes=(0))
        def vmap_g2p(point_id, point_hash):
            point_grid_pos = jnp.array(jnp.unravel_index(point_hash, self.grid_size))

            def inner(w_id, carry):
                window = jnp.array(self.forward_window)[w_id]

                neighbor = point_grid_pos + window

                if is_grid_hash:
                    neighbor = get_hash_id(neighbor, self.grid_size)

                new_carry_scan = g2p_func(point_id, neighbor, w_id, carry)

                return new_carry_scan

            carry = jax.lax.fori_loop(
                lower=0,
                upper=self.window_size,
                body_fun=inner,
                unroll=self.unroll_grid_kernels,
                init_val=(init_vals),
            )

            return carry

        return vmap_g2p(self.point_id_stack, self.point_hash_stack)

    def vmap_grid_gather_fori(self, p2g_func, init_vals):
        @partial(jax.vmap, in_axes=(0))
        def vmap_p2g(cell_id):
            cell_pos = jnp.array(jnp.unravel_index(cell_id, self.grid_size))

            def inner_1(w_id, carry):
                window = jnp.array(self.backward_window)[w_id]

                neighbor = cell_pos + window

                neighbor_id = get_hash_id(neighbor, self.grid_size)

                count = self.cell_count_stack.at[neighbor_id].get(
                    mode="fill", fill_value=0
                )
                # jax.debug.print(" cell_pos {} window {} | neighbor {} neighbor_id {} {}",cell_pos,window,neighbor,cell_id,count)

                def inner_2(step, carry_fori):
                    p_id = self.cell_start_stack.at[neighbor_id].get() + step
                    p_id = self.point_id_stack.at[p_id].get()
                    new_carry_fori = p2g_func(p_id, cell_id, w_id, carry_fori)
                    return new_carry_fori

                new_carry = jax.lax.fori_loop(
                    lower=0,
                    upper=count,
                    body_fun=inner_2,
                    init_val=carry,
                )

                return new_carry
                # return carry

            carry = jax.lax.fori_loop(
                lower=0,
                upper=self.window_size,
                body_fun=inner_1,
                unroll=self.unroll_grid_kernels,
                init_val=(init_vals),
            )
            return carry

        return vmap_p2g(self.cell_id_stack)
