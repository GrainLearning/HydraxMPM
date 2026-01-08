# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-
"""
Explanation:
    This module handles linear shape functions for MPM simulations.
    It provides a function to compute the shape function values and gradients
    based on the relative distance from particles to grid nodes.
    
    The function is stored as a `Callable` within the ShapeFunctionMapping class

    It has a support of [-1, 1].

    References:
    - De Vaucorbeil, Alban, et al. "Material point method after 25 years: Theory, implementation, and applications." Adva
    
"""
from typing import Tuple, Any

import jax.numpy as jnp
from jaxtyping import Array, Float


def vmap_linear_shapefunction(
    intr_dist: Float[Array, "dim"],
    inv_cell_size: float, 
    dim: int,
    padding: Tuple[int, int],
    **kwargs: Any,
) -> Tuple[Float[Array, ""], Float[Array, "3"]]:
    
    """
    Calculates Linear Shape Function (1-Linear / Tent) and gradients.
    """

    # compute 1D basis functions along each axis
    # N(x) = 1 - |x|  for |x| < 1
    abs_intr_dist = jnp.abs(intr_dist)

    # ensures we don't divide by zero if mass is low, 
    # though strictly for shape functions 0.0 is the correct region outside support.
    basis = jnp.where(abs_intr_dist < 1.0, 1.0 - abs_intr_dist, 0.0)

    # compute 1D derivatives
    # dN(x) = -sign(x) * (1/h)
    dbasis = jnp.where(abs_intr_dist < 1.0, -jnp.sign(intr_dist) * inv_cell_size, 0.0)

    # tensor product for gradient
    # grad N(x,y) = [ dNx*Ny, Nx*dNy ]
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
        # Fallback for 1D
        shapef_grad = dbasis

    # tensor Product for value
    # N(x,y) = Nx * Ny
    shapef = jnp.prod(basis)


    # pad gradients to 3D
    shapef_grad_padded = jnp.pad(
        shapef_grad,
        padding,
        mode="constant",
        constant_values=0.0,
    )

    return (shapef, shapef_grad_padded)
