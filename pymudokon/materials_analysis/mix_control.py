"""Base class for single integration point benchmark module"""

from typing import Callable

import jax
import jax.numpy as jnp
import optax
from jax import Array
import chex
from ..materials.material import Material


def update_from_params(
    material: Material,
    servo_params: Array,
    strain_rate_control: Array,
    mask: Array,
    volume_fraction: Array,
    dt: jnp.float32,
):
    trail_strain_rate = strain_rate_control

    trail_strain_rate = trail_strain_rate.at[mask].set(servo_params)

    material, trail_stress = material.update_stress_benchmark(
        trail_strain_rate.reshape(1, 3, 3), jnp.array([volume_fraction]), dt
    )

    trail_stress = trail_stress.reshape((3, 3))

    return material, trail_stress, trail_strain_rate


def mixed_loss(servo_params, strain_rate_control: Array, mask, stress_control, dt, material, volume_fraction):
    material, trail_stress, _ = update_from_params(
        material, servo_params, strain_rate_control, mask, volume_fraction, dt
    )

    loss = optax.losses.l2_loss(trail_stress.at[mask].get(), stress_control.at[mask].get())

    return jnp.mean(loss)


# @jax.jit
def mix_control(
    material: Material,
    strain_rate_stack: chex.Array,
    stress_stack: chex.Array,
    mask: chex.Array,
    volume_fraction: jnp.float32,
    dt: jnp.float32 | Array,
    learning_rate: float = 1e-3,
    num_opt_iter: int = 20,
) -> Material:
    # supports only 1 integration point (particle)
    chex.assert_shape(strain_rate_stack, (None, 3, 3))
    chex.assert_shape(stress_stack, (None, 3, 3))
    chex.assert_shape(mask, (3, 3))
    chex.assert_shape(volume_fraction, ())
    chex.assert_shape(material.stress_ref, (1, 3, 3))

    servo_params = jnp.zeros((3, 3)).at[mask].get()

    def scan_fn(carry, control):
        material, volume_fraction, servo_params = carry
        strain_rate_control, stress_control = control

        solver = optax.adabelief(learning_rate=learning_rate)

        opt_state = solver.init(servo_params)

        def run_solver_(i, carry):
            servo_params, opt_state = carry
            grad = jax.grad(mixed_loss)(
                servo_params, strain_rate_control, mask, stress_control, dt, material, volume_fraction
            )
            updates, opt_state = solver.update(grad, opt_state)
            servo_params = optax.apply_updates(servo_params, updates)
            return servo_params, opt_state

        servo_params, opt_state = jax.lax.fori_loop(0, num_opt_iter, run_solver_, (servo_params, opt_state))

        material, stress, strain_rate = update_from_params(
            material, servo_params, strain_rate_control, mask, volume_fraction, dt
        )

        carry = (material, volume_fraction, servo_params)
        accumulate = (material, volume_fraction, stress, strain_rate)
        return carry, accumulate

    return jax.lax.scan(
        scan_fn,
        (material, volume_fraction, servo_params),
        (strain_rate_stack, stress_stack),
    )
