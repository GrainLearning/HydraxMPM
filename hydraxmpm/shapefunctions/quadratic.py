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


def vmap_quadratic_shapefunction(
    intr_dist: TypeFloatVector,
    inv_cell_size: TypeFloat,
    dim: TypeInt,
    padding: Tuple[int, int],
    intr_node_type: TypeInt,
    **kwargs,
) -> Tuple[TypeFloat, TypeFloat3]:
    condlist = [
        (intr_dist >= -3 / 2) * (intr_dist <= -1 / 2),
        (intr_dist > -1 / 2) * (intr_dist <= 1 / 2),
        (intr_dist > 1 / 2) * (intr_dist <= 3 / 2),
    ]

    _piecewise = partial(jnp.piecewise, x=intr_dist, condlist=condlist)

    h = jnp.array(inv_cell_size)

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

    shapef_grad_padded = jnp.pad(
        shapef_grad,
        padding,
        mode="constant",
        constant_values=0.0,
    )

    return (shapef, shapef_grad_padded)
