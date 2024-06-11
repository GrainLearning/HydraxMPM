"""Base class for single integration point benchmark module"""

from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

from ..materials.material import Material


@jax.jit
def simple_shear(
    material: Material,
    eps_path: Array,
    volumes: Array,
    dt: jnp.float32 | Array,
    output_step=int,
    output_function: Callable = None,
) -> Material:
    """Perfom a simple shear benchmark a material.

    Args:
        material (Material): Material object.
        eps_path (Array): Strain path.
        volumes (Array): Volumes of the material points.
        dt (jnp.float32 | Array): Time step.
        output_step (_type_, optional): Output number of steps. Defaults to int.
        output_function (Callable, optional): Callback function to process output. Defaults to None.

    Returns:
        Material: Updated material object.
    """
    num_benchmarks, num_steps = eps_path.shape

    eps_inc_target = jnp.zeros((num_benchmarks, 3, 3))
    eps_inc_prev = jnp.zeros((num_benchmarks, 3, 3))

    def body_loop(step, carry):
        eps_inc_prev, eps_inc_target, material, volumes, dt, output_function = carry
        eps_inc_target = eps_inc_target.at[:, 0, 1].add(eps_path[:, step])

        strain_increment = eps_inc_target - eps_inc_prev

        strain_rate = strain_increment / dt

        stress, material = material.update_stress_benchmark(strain_rate, volumes, dt)

        eps_inc_prev = eps_inc_target

        jax.lax.cond(
            step % output_step == 0,
            lambda x: jax.experimental.io_callback(output_function, None, x),
            lambda x: None,
            (step, stress, material, eps_inc_target, num_steps, strain_rate, dt),
        )
        return (eps_inc_prev, eps_inc_target, material, volumes, dt, output_function)

    package = jax.lax.fori_loop(
        0, num_steps, body_loop, (eps_inc_prev, eps_inc_target, material, volumes, dt, output_function)
    )

    return material
