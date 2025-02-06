from typing import Tuple

import jax.numpy as jnp

from ..common.types import TypeInt, TypeFloat, TypeFloat3, TypeFloatVector


def vmap_linear_shapefunction(
    intr_dist: TypeFloatVector,
    inv_cell_size: TypeFloat,
    dim: TypeInt,
    padding: Tuple[int, int],
) -> Tuple[TypeFloat, TypeFloat3]:
    abs_intr_dist = jnp.abs(intr_dist)

    basis = jnp.where(abs_intr_dist < 1.0, 1.0 - abs_intr_dist, 0.0)

    dbasis = jnp.where(abs_intr_dist < 1.0, -jnp.sign(intr_dist) * inv_cell_size, 0.0)

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
