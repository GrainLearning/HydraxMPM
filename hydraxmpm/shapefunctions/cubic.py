# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

from ..common.types import TypeInt, TypeFloat, TypeFloat3, TypeFloatVector


def vmap_cubic_shapefunction(
    intr_dist: TypeFloatVector,
    inv_cell_size: TypeFloat,
    dim: TypeInt,
    padding: Tuple[int, int],
    intr_node_type: TypeInt,
    **kwargs,
) -> Tuple[TypeFloat, TypeFloat3]:
    condlist = [
        (intr_dist >= -2) * (intr_dist < -1),
        (intr_dist >= -1) * (intr_dist < 0),
        (intr_dist >= 0) * (intr_dist < 1),
        (intr_dist >= 1) * (intr_dist < 2),
    ]

    _piecewise = partial(jnp.piecewise, x=intr_dist, condlist=condlist)

    h = inv_cell_size

    # type 1 nodes - left / right boundary
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

    # type 2 nodes - left boundary + h
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

    # type 3 nodes middle
    # checked
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

    # type 4 nodes - right boundary - h
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

    # 0th index is middle
    # 1st index is boundary 0 or N
    # 3rd index is left side of closes boundary 0 + h
    # 4th index is right side of closes boundary N -h

    basis, dbasis = jax.lax.switch(
        # index=0,
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
        shapef_grad = dbasis

    shapef = jnp.prod(basis)

    shapef_grad_padded = jnp.pad(
        shapef_grad,
        padding,
        mode="constant",
        constant_values=0.0,
    )

    return (shapef, shapef_grad_padded)
