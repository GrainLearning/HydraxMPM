# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-
"""
Explanation:
    This module handles cubic shape functions for MPM simulations.
    It provides a function to compute the shape function values and gradients
    based on the relative distance from particles to grid nodes.
    
    The function is stored as a `Callable` within the ShapeFunctionMapping class

    It has a support of [-2, 2].
    
    References:
    - De Vaucorbeil, Alban, et al. "Material point method after 25 years: Theory, implementation, and applications." Adva

"""

from functools import partial
from typing import Tuple, Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


def vmap_cubic_shapefunction(
    intr_dist: Float[Array, "dim"],
    inv_cell_size: float,
    dim: int,
    padding: Tuple[int, int],
    intr_node_type: Int[Array, ""] | int,  # Dynamic integer scalar (0, 1, 2, 3)
    **kwargs: Any,
) -> Tuple[Float[Array, ""], Float[Array, "3"]]:
    
    """
    Cubic B-Spline shape function with boundary conditions.
    """
    # Pre-define conditions for piecewise construction
    # Note: These are boolean masks on the static shape of intr_dist
    condlist = [
        (intr_dist >= -2) * (intr_dist < -1),
        (intr_dist >= -1) * (intr_dist < 0),
        (intr_dist >= 0) * (intr_dist < 1),
        (intr_dist >= 1) * (intr_dist < 2),
    ]

    # helper for cleaner code
    _piecewise = partial(jnp.piecewise, x=intr_dist, condlist=condlist)
    h = inv_cell_size

    # spline definitions for different node types
    # Case 1: Boundary Nodes (Left/Right Edge)
    def boundary_splines():
        # checked
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
        # checked
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

    # Case 2: Near Boundary Left (0 + h)
    def boundary_0_p_h():
        # checked
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
        # checked
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

    # Case 0: Internal Nodes (Standard B-Spline)
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

    # Case 3: Near Boundary Right (L - h)
    def boundary_N_m_h():
        # checked
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
        # checked
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

    # Selects the correct spline function based on node type
    basis, dbasis = jax.lax.switch(
        index=intr_node_type,
        branches=[
            middle_splines,
            boundary_splines,
            boundary_0_p_h,
            boundary_N_m_h,
        ],
    )

    if dim == 2:
        shapef_grad = jnp.array(
            [
                dbasis.at[0].get() * basis.at[1].get(),
                dbasis.at[1].get() * basis.at[0].get(),
            ]
        )
    elif dim == 3:
        shapef_grad = jnp.array(
            [
                dbasis.at[0].get() * basis.at[1].get() * basis.at[2].get(),
                dbasis.at[1].get() * basis.at[0].get() * basis.at[2].get(),
                dbasis.at[2].get() * basis.at[0].get() * basis.at[1].get(),
            ]
        )
    else:
        # 1D fallback
        shapef_grad = dbasis

    shapef = jnp.prod(basis)

    # pad gradients to 3D
    shapef_grad_padded = jnp.pad(
        shapef_grad,
        padding,
        mode="constant",
        constant_values=0.0,
    )

    return (shapef, shapef_grad_padded)
