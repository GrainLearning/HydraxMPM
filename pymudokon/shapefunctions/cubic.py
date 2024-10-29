"""Module for containing the cubic shape functions.

References:
    - De Vaucorbeil, Alban, et al. 'Material point method after 25 years: theory,
    implementation, and applications.'
"""

from functools import partial
from typing import Tuple
from typing_extensions import Self

import chex
import jax
import jax.numpy as jnp
from jax import Array

import equinox as eqx

from ..config.mpm_config import MPMConfig


class CubicShapeFunction(eqx.Module):
    """Cubic B-spline shape functions for the particle-node interactions.

    It is recommended that each background cell is populated by
    2 (1D), 4 (2D), 8 (3D) material points. The optimal integration points are
    at 0.2113, 0.7887 determined by Gauss quadrature rule.


    """

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

        self.shapef_stack = jnp.zeros((self.num_points, self.window_size))
        self.shapef_grad_stack = jnp.zeros((self.num_points, self.window_size, 3))

    def __call__(self, grid, particles):
        padding = (0, 3 - self.dim)

        def vmap_g2p_shp(p_id, grid_pos, w_id, carry):
            shape_f_prev, shape_f_grad_prev = carry

            rel_pos = (
                particles.position_stack.at[p_id].get() - jnp.array(self.origin)
            ) * self.inv_cell_size

            dist = rel_pos - grid_pos


            condlist = [
                (dist >= -2) * (dist < -1),
                (dist >= -1) * (dist < 0),
                (dist >= 0) * (dist < 1),
                (dist >= 1) * (dist < 2),
            ]

            _piecewise = partial(jnp.piecewise, x=dist, condlist=condlist)

            h = self.inv_cell_size

            def middle_splines():
                basis = _piecewise(
                    funclist=[
                        # (1/6)x**3 + x**2 + 2x + 4/3
                        lambda x: ((1.0 / 6.0 * x + 1.0) * x + 2.0) * x + 4.0 / 3.0,
                        # -1/2 x**3 - x**2 +2/3
                        lambda x: (-0.5 * x - 1) * x * x + 2.0 / 3.0,
                        # 1/2 x**3 - x**2 + 2/3
                        lambda x: (0.5 * x - 1) * x * x + 2.0 / 3.0,
                        # -1/6 x**3 + x**2 -2x + 4/3
                        lambda x: ((-1.0 / 6.0 * x + 1.0) * x - 2.0) * x + 4.0 / 3.0,
                    ]
                )
                dbasis = _piecewise(
                    funclist=[
                        # (1/2)x**2 + 2x + 2
                        lambda x: h * ((0.5 * x + 2) * x + 2.0),
                        # -3/2 x**2 - 2x
                        lambda x: h * (-3.0 / 2.0 * x - 2.0) * x,
                        # 3/2 x**2 - 2x
                        lambda x: h * (3.0 / 2.0 * x - 2.0) * x,
                        # -1/2 x**2 + 2x -2
                        lambda x: h * ((-0.5 * x + 2) * x - 2.0),
                    ]
                )
                return basis, dbasis

            def boundary_splines():
                basis = _piecewise(
                    funclist=[
                        # 1/6 x**3 + x**2 + 2x + 4/3
                        lambda x: ((1.0 / 6.0 * x + 1.0) * x + 2.0) * x + 4.0 / 3.0,
                        # -1/6 x**3 +x + 1
                        lambda x: (-1.0 / 6.0 * x * x + 1.0) * x + 1.0,
                        # 1/6 x**3 - x  + 1
                        lambda x: ((1.0 / 6.0) * x * x - 1.0) * x + 1.0,
                        # -1/6 x**3 + x**2 -2x + 4/3
                        lambda x: ((-1.0 / 6.0 * x + 1.0) * x - 2.0) * x + 4.0 / 3.0,
                    ]
                )
                dbasis = _piecewise(
                    funclist=[
                        # 1/2 x**2 + 2x + 2
                        lambda x: h * ((0.5 * x + 2) * x + 2.0),
                        # -1/2 x**2 +1
                        lambda x: h * (-0.5 * x * x + 1.0),
                        # 1/2 x**2 - 1
                        lambda x: h * (0.5 * x * x - 1.0),
                        # -1/2 x**2 + 2x -2
                        lambda x: h * ((-0.5 * x + 2) * x - 2.0),
                    ]
                )
                return basis, dbasis

            def boundary_0_p_h():
                basis = _piecewise(
                    funclist=[
                        lambda x: jnp.float32(0.0),
                        # -1/3 x**3 -x**2 + 2/3
                        lambda x: (-1.0 / 3.0 * x - 1.0) * x * x + 2.0 / 3.0,
                        # 1/2 x**3 -x**2 + 2/3
                        lambda x: (0.5 * x - 1) * x * x + 2.0 / 3.0,
                        # -1/6 x**3 + x**2 -2x + 4/3
                        lambda x: ((-1.0 / 6.0 * x + 1.0) * x - 2.0) * x + 4.0 / 3.0,
                    ]
                )
                dbasis = _piecewise(
                    funclist=[
                        lambda x: jnp.float32(0.0),
                        # -x**2 -2x
                        lambda x: h * (-x - 2) * x,
                        # 3/2 x**2 -2x
                        lambda x: h * (3.0 / 2.0 * x - 2.0) * x,
                        # -1/2 x**2 + 2x -2
                        lambda x: h * ((-0.5 * x + 2) * x - 2.0),
                    ]
                )
                return basis, dbasis

            def boundary_N_m_h():
                basis = _piecewise(
                    funclist=[
                        # (1/6) x**3 + x**2 + 2x + 4/3
                        lambda x: ((1.0 / 6.0 * x + 1.0) * x + 2.0) * x + 4.0 / 3.0,
                        # -1/2 x**3 - x**2 + 2/3
                        lambda x: (-0.5 * x - 1) * x * x + 2.0 / 3.0,
                        # 1/3 x**3 -x**2 + 2/3
                        lambda x: (1.0 / 3.0 * x - 1.0) * x * x + 2.0 / 3.0,
                        lambda x: jnp.float32(0.0),
                    ]
                )
                dbasis = _piecewise(
                    funclist=[
                        # (1/2) x**2 + 2x + 2
                        lambda x: h * ((0.5 * x + 2) * x + 2.0),
                        # -3/2 x**2 - 2x
                        lambda x: h * (-3.0 / 2.0 * x - 2.0) * x,
                        #  x**2 -2x
                        lambda x: h * (x - 2.0) * x,
                        lambda x: jnp.float32(0.0),
                    ]
                )
                return basis, dbasis

            # 0th index is middle
            # 1st index is boundary 0 or N
            # 3rd index is left side of closes boundary 0 + h
            # 4th index is right side of closes boundary N -h

            basis, dbasis = jax.lax.switch(
                # index= intr_node_type,
                index=0,
                branches=[
                    middle_splines,
                    boundary_splines,
                    boundary_0_p_h,
                    boundary_N_m_h,
                ],
            )

            shapef = jnp.prod(basis)

            # jax.debug.print(
            #     "p_id {} w_id {} rel_pos {} grid_pos {} dist {} shapef {}",
            #     p_id,
            #     w_id,
            #     rel_pos,
            #     grid_pos,
            #     dist,
            #     shapef,
            # )
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

    # def calculate_shapefunction(
    #     self: Self,
    #     origin: chex.Array,
    #     inv_node_spacing: jnp.float32,
    #     grid_size: chex.Array,
    #     position_stack: chex.Array,
    #     species_stack: chex.Array
    # ) -> Tuple[Self, Array]:
    #     """Calculate shape functions and its gradients."""
    #     stencil_size, dim = self.stencil.shape

    #     num_particles = position_stack.shape[0]

    #     intr_id_stack = jnp.arange(num_particles * stencil_size).astype(jnp.int32)

    #     # Calculate the particle-node pair interactions
    #     # see `ShapeFunction class` for more details
    #     intr_dist_stack, intr_hash_stack = self.vmap_intr(
    #         intr_id_stack, position_stack, origin, inv_node_spacing, grid_size
    #     )

    #     # intr_node_type = jnp.zeros(intr_id_stack).astype(jnp.int16)
    #     # from here we can calculate intr hash type node_type[intr_hash]
    #     # same as gather....
    #     intr_shapef_stack, intr_shapef_grad_stack = self.vmap_intr_shp(
    #         intr_dist_stack, intr_hash_stack, species_stack, inv_node_spacing
    #     )
    #     # intr_shapef_stack = self.intr_shapef_stack
    #     # intr_shapef_grad_stack = self.intr_shapef_grad_stack
    #     # print(intr_dist_stack.shape)

    #     intr_dist_3d_stack = jnp.pad(
    #         intr_dist_stack,
    #         [(0, 0), (0, 3 - dim)],
    #         mode="constant",
    #         constant_values=0,
    #     )

    #     return self.replace(
    #         intr_shapef_stack=intr_shapef_stack,
    #         intr_shapef_grad_stack=intr_shapef_grad_stack,
    #         intr_id_stack=intr_id_stack,
    #         intr_hash_stack=intr_hash_stack,
    #     ), intr_dist_3d_stack

    # @partial(jax.vmap, in_axes=(None, 0, 0, None, None))
    # def vmap_intr_shp(
    #     self,
    #     intr_dist: Array,
    #     intr_hash: jnp.int32,
    #     node_species_stack: Array,
    #     h: jax.numpy.float32
    # ) -> Tuple[Array, Array]:
    #     """Vectorized cubic shape function calculation.

    #     Calculate the shape function, and then its gradients.

    #     Args:
    #     intr_dist (Array):
    #     Particle-node pair interactions distance.
    #     intr_species (Array):
    #     Node type of the background grid. See
    #     :meth:`pymudokon.core.nodes.Nodes.set_species` for details.
    #     inv_node_spacing (jax.numpy.float32):
    #     Inverse node spacing.

    #     Returns:
    #     Tuple[Array, Array]:
    #     Shape function and its gradient.
    #     """
    #     # intr_node_type = node_species_stack.at[intr_hash].get()

    #     condlist = [
    #                 (intr_dist >= -2)*(intr_dist < -1),
    #                 (intr_dist >= -1)*(intr_dist < 0),
    #                 (intr_dist >= 0)*(intr_dist < 1),
    #                 (intr_dist >=1)*(intr_dist < 2)
    #             ]

    #     _piecewise = partial(jnp.piecewise,
    #                 x = intr_dist,
    #                 condlist = condlist
    #                 )

    #     def middle_splines():
    #         basis = _piecewise(funclist=[
    #                 # (1/6)x**3 + x**2 + 2x + 4/3
    #                 lambda x: ((1.0 / 6.0 * x + 1.0) * x + 2.0) * x + 4.0 / 3.0,
    #                 # -1/2 x**3 - x**2 +2/3
    #                 lambda x: (-0.5 * x - 1) * x * x + 2.0 / 3.0,
    #                 # 1/2 x**3 - x**2 + 2/3
    #                 lambda x: (0.5 * x - 1) * x * x + 2.0 / 3.0,
    #                 # -1/6 x**3 + x**2 -2x + 4/3
    #                 lambda x: ((-1.0 / 6.0 * x + 1.0) * x - 2.0) * x + 4.0 / 3.0
    #                 ])
    #         dbasis = _piecewise(funclist=[
    #                 # (1/2)x**2 + 2x + 2
    #                 lambda x: h * ((0.5 * x + 2) * x + 2.0),
    #                 # -3/2 x**2 - 2x
    #                 lambda x: h * (-3.0 / 2.0 * x - 2.0) * x,
    #                 # 3/2 x**2 - 2x
    #                 lambda x: h * (3.0 / 2.0 * x - 2.0) * x,
    #                 # -1/2 x**2 + 2x -2
    #                 lambda x: h * ((-0.5 * x + 2) * x - 2.0)
    #                 ])
    #         return basis, dbasis

    #     def boundary_splines():
    #         basis = _piecewise(funclist=[
    #                 # 1/6 x**3 + x**2 + 2x + 4/3
    #                 lambda x: ((1.0 / 6.0 * x + 1.0) * x + 2.0) * x + 4.0 / 3.0,
    #                 # -1/6 x**3 +x + 1
    #                 lambda x: (-1.0 / 6.0 * x * x + 1.0) * x + 1.0,
    #                 # 1/6 x**3 - x  + 1
    #                 lambda x:  ((1.0 / 6.0 )*x*x -1.0)*x  +1.0,
    #                 # -1/6 x**3 + x**2 -2x + 4/3
    #                 lambda x: ((-1.0 / 6.0 * x + 1.0) * x - 2.0) * x + 4.0 / 3.0
    #                 ])
    #         dbasis = _piecewise(funclist=[
    #                 # 1/2 x**2 + 2x + 2
    #                 lambda x: h * ((0.5 * x + 2) * x + 2.0),
    #                 # -1/2 x**2 +1
    #                 lambda x: h * (-0.5 * x * x + 1.0),
    #                 # 1/2 x**2 - 1
    #                 lambda x: h * (0.5 * x * x - 1.0),
    #                 # -1/2 x**2 + 2x -2
    #                 lambda x: h * ((-0.5 * x + 2) * x - 2.0)
    #                 ])
    #         return basis, dbasis

    #     def boundary_0_p_h():
    #         basis = _piecewise(funclist=[
    #                 lambda x:  jnp.float32(0.0),
    #                 # -1/3 x**3 -x**2 + 2/3
    #                 lambda x: (-1.0 / 3.0 * x - 1.0) * x * x + 2.0 / 3.0,
    #                 # 1/2 x**3 -x**2 + 2/3
    #                 lambda x: (0.5 * x - 1) * x * x + 2.0 / 3.0,
    #                 # -1/6 x**3 + x**2 -2x + 4/3
    #                 lambda x: ((-1.0 / 6.0 * x + 1.0) * x - 2.0) * x + 4.0 / 3.0,
    #                 ])
    #         dbasis = _piecewise(funclist=[
    #                 lambda x:  jnp.float32(0.0),
    #                 # -x**2 -2x
    #                 lambda x: h * (-x - 2) * x,
    #                 # 3/2 x**2 -2x
    #                 lambda x: h * (3.0 / 2.0 * x - 2.0) * x,
    #                 # -1/2 x**2 + 2x -2
    #                 lambda x: h * ((-0.5 * x + 2) * x - 2.0),
    #                 ])
    #         return basis, dbasis

    #     def boundary_N_m_h():
    #         basis = _piecewise(funclist=[
    #                 # (1/6) x**3 + x**2 + 2x + 4/3
    #                 lambda x: ((1.0 / 6.0 * x + 1.0) * x + 2.0) * x + 4.0 / 3.0,
    #                 # -1/2 x**3 - x**2 + 2/3
    #                 lambda x: (-0.5 * x - 1) * x * x + 2.0 / 3.0,
    #                 # 1/3 x**3 -x**2 + 2/3
    #                 lambda x: (1.0 / 3.0 * x - 1.0) * x * x + 2.0 / 3.0,
    #                 lambda x:  jnp.float32(0.0),
    #                 ])
    #         dbasis = _piecewise(funclist=[
    #                 # (1/2) x**2 + 2x + 2
    #                 lambda x: h * ((0.5 * x + 2) * x + 2.0),
    #                 # -3/2 x**2 - 2x
    #                 lambda x: h * (-3.0 / 2.0 * x - 2.0) * x,
    #                 #  x**2 -2x
    #                 lambda x: h * (x - 2.0) * x,
    #                 lambda x:  jnp.float32(0.0),
    #                 ])
    #         return basis, dbasis

    #     # 0th index is middle
    #     # 1st index is boundary 0 or N
    #     # 3rd index is left side of closes boundary 0 + h
    #     # 4th index is right side of closes boundary N -h

    #     basis, dbasis = jax.lax.switch(
    #         # index= intr_node_type,
    #         index = 0,
    #         branches =[
    #         middle_splines,
    #         boundary_splines,
    #         boundary_0_p_h,
    #         boundary_N_m_h,
    #         ]
    #     )
    #     intr_shapef = jnp.prod(basis)

    #     dim = basis.shape[0]
    #     if dim == 2:
    #         intr_shapef_grad = jnp.array(
    #             [
    #                 dbasis[0] * basis[1],
    #                 dbasis[1] * basis[0],
    #                 0.0,
    #             ]
    #         )
    #     elif dim == 3:
    #         intr_shapef_grad = jnp.array(
    #             [
    #                 dbasis[0] * basis[1] * basis[2],
    #                 dbasis[1] * basis[0] * basis[2],
    #                 dbasis[2] * basis[0] * basis[1],
    #             ]
    #         )
    #     else:
    #         intr_shapef_grad = jnp.array([dbasis, 0.0, 0.0])
    #     return intr_shapef, intr_shapef_grad
