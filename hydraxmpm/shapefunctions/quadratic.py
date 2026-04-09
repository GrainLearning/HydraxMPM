# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-
"""
Explanation:
    This module handles quadratic shape functions for MPM simulations.
    It provides a function to compute the shape function values and gradients
    based on the relative distance from particles to grid nodes.
    
    The function is stored as a `Callable` within the ShapeFunctionMapping class

    It has a support of [-1.5, 1.5].

    References:
    - De Vaucorbeil, Alban, et al. "Material point method after 25 years: Theory, implementation, and applications." Adva
    - Jiang, Chenfanfu, et al. "The affine particle-in-cell method."

"""

from functools import partial
from typing import Tuple, Any

import jax.numpy as jnp
from jaxtyping import Array, Float

def vmap_quadratic_shapefunction(
    intr_dist: Float[Array, "dim"],
    inv_cell_size: float,
    dim: int,
    padding: Tuple[int, int],
    intr_node_type, # Kept for API consistency, unused in standard quadratic
    **kwargs: Any,
) ->  Tuple[Float[Array, ""], Float[Array, "3"]]:
    """
    Quadratic B-Spline shape function.

    """
    # Pre-define conditions for piecewise construction
    # Note: These are boolean masks on the static shape of intr_dist
    condlist = [
        (intr_dist >= -3 / 2) * (intr_dist <= -1 / 2),
        (intr_dist > -1 / 2) * (intr_dist <= 1 / 2),
        (intr_dist > 1 / 2) * (intr_dist <= 3 / 2),
    ]

    # helper for cleaner code
    _piecewise = partial(jnp.piecewise, x=intr_dist, condlist=condlist)
    h = jnp.array(inv_cell_size)

    
    # spline definitions for different node types

    # Range 1: 0.5(x + 1.5)^2  ->  0.5x^2 + 1.5x + 1.125
    # Range 2: 0.75 - x^2      -> -x^2 + 0.75
    # Range 3: 0.5(1.5 - x)^2  ->  0.5x^2 - 1.5x + 1.125
    
    def quadratic_splines():
        basis = _piecewise(
            funclist=[
                # 1/(2h^2) * x^2 + 3/(2h) * x + 9/8
                lambda x: (1 / 2) * x * x + (3 / 2) * x + 9 / 8,
                # -1/h^2 * x^2 + 3/4
                lambda x: -x * x + 3 / 4,
                # 1/(2h^2) * x^2 - 3/(2h) * x + 9/8
                lambda x: (1 / 2) * x * x - (3 / 2) * x + 9 / 8,
            ]
        )
        dbasis = _piecewise(
            funclist=[
                #  x + 3/(2)
                lambda x: h * (x + 3 / 2),
                # -2* x
                lambda x: h * (-2 * x),
                # h*(x+1.5)
                lambda x: h * (x - 3 / 2),
            ]
        )
        return basis, dbasis

    basis, dbasis = quadratic_splines()

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
