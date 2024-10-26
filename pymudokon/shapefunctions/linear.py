"""Most basic linear shape functions."""

from functools import partial
from typing import Tuple
from typing_extensions import Self

import chex

import jax.numpy as jnp
import equinox as eqx

from ..config.mpm_config import MPMConfig


class LinearShapeFunction(eqx.Module):
    shapef_stack: chex.Array = eqx.field(converter=lambda x: jnp.asarray(x))
    shapef_grad_stack: chex.Array = eqx.field(converter=lambda x: jnp.asarray(x))

    # statics
    dim: int = eqx.field(static=True, converter=lambda x: int(x))
    window_size: int = eqx.field(static=True, converter=lambda x: int(x))
    num_points: int = eqx.field(static=True, converter=lambda x: int(x))
    origin: int = eqx.field(static=True, converter=lambda x: tuple(x))
    inv_cell_size: float = eqx.field(static=True, converter=lambda x: float(x))

    def __init__(
        self,
        config: MPMConfig,
        num_points: int = None,
        dim: int = None,
        inv_cell_size: float = None,
        origin: tuple = None,
    ):
        if config:
            window_size = config.window_size
            dim = config.dim
            num_points = config.num_points
            inv_cell_size = config.inv_cell_size
            origin = config.origin

        self.num_points = num_points
        self.window_size = window_size
        self.dim = dim
        self.origin = origin
        self.inv_cell_size = inv_cell_size

        self.shapef_stack = jnp.zeros((self.num_points,self.window_size))
        self.shapef_grad_stack = jnp.zeros((self.num_points, self.window_size, 3))

    def get_shapefunctions(
        self: Self,
        grid,
        particles: chex.Array,
    ) -> Tuple[chex.Array]:
        padding = (0, 3 - self.dim)

        def vmap_g2p_shp(p_id, grid_pos, w_id, carry):
            shape_f_prev, shape_f_grad_prev = carry

            rel_pos = (
                particles.position_stack.at[p_id].get() - jnp.array(self.origin)
            ) * self.inv_cell_size

            dist = rel_pos - grid_pos
            abs_dist = jnp.abs(dist)

            basis = jnp.where(abs_dist < 1.0, 1.0 - abs_dist, 0.0)

            dbasis = jnp.where(
                abs_dist < 1.0, -jnp.sign(dist) * self.inv_cell_size, 0.0
            )

            shapef = jnp.prod(basis)

            if self.dim == 2:
                shapef_grad = jnp.array(
                    [
                        dbasis.at[0].get() * basis.at[1].get(),
                        dbasis.at[1].get() * basis.at[0].get(),
                    ]
                )
            elif self.dim == 3:
                shapef_grad = jnp.array(
                    [
                        dbasis.at[0].get() * basis.at[1].get() * basis.at[2].get(),
                        dbasis.at[1].get() * basis.at[0].get() * basis.at[2].get(),
                        dbasis.at[2].get() * basis.at[0].get() * basis.at[1].get(),
                    ]
                )
            else:
                shapef_grad = dbasis

            shapef_grad_padded = jnp.pad(
                shapef_grad,
                padding,
                mode="constant",
                constant_values=0.0,
            )

            # jax.debug.print("grid_pos {} self.inv_cell_size {} rel_pos {} dist {} w_id {}",grid_pos,self.inv_cell_size,rel_pos, dist,w_id)
            new_shapef = shape_f_prev.at[w_id].set(shapef)
            new_shapef_grad = shape_f_grad_prev.at[w_id, :].set(shapef_grad_padded)

            return (new_shapef, new_shapef_grad)

        new_shapef_stack, new_shapef_grad_stack = grid.vmap_grid_scatter_fori(
            vmap_g2p_shp,
            (jnp.zeros(self.window_size), jnp.zeros((self.window_size, 3))),
            is_grid_hash=False,
        )

        return eqx.tree_at(
            lambda state: (
                state.shapef_stack,
                state.shapef_grad_stack,
            ),
            self,
            (new_shapef_stack, new_shapef_grad_stack),
        )
