"""Base class for single integration point benchmark module"""

from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

from ..materials.material import Material


@jax.jit
def simple_shear(material: Material, strain_rate_tensors: Array, volumes: Array, dt: jnp.float32 | Array) -> Material:
    def scan_fn(carry, control):
        material, volume = carry
        strain_rate = control.reshape(1, 3, 3)
        print(strain_rate.shape, volume.shape, dt.shape)
        material, stress = material.update_stress_benchmark(strain_rate, volume, dt)

        carry = (material, volume)
        accumulate = (material, volume, stress)
        return carry, accumulate

    carry, accumulate = jax.lax.scan(scan_fn, (material, volumes), strain_rate_tensors)
    return carry, accumulate
