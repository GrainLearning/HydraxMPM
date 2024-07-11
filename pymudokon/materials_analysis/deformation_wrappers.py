import jax
import jax.numpy as jnp
import chex
from .mix_control import mix_control


def triaxial_compression_wrapper(
    material, material_params, total_time=25.0, dt=0.01, strain_target=0.4, confine_pressure=0.0, volume_fraction=0.5
):
    num_steps = int(total_time / dt)

    deps_xx = strain_target / num_steps

    strain_rate_stack = jnp.zeros((num_steps, 3, 3)).at[:num_steps, [0, 1, 2], [0, 1, 2]].set(deps_xx)

    mask = jnp.zeros((3, 3)).at[[1, 2], [1, 2]].set(1).astype(bool)

    stress_stack = jnp.zeros((num_steps, 3, 3)).at[:num_steps, [0, 1, 2], [0, 1, 2]].set(-confine_pressure)

    material = material.create(*material_params)

    return mix_control(material, strain_rate_stack, stress_stack, mask, volume_fraction, dt)


def simple_shear_wrapper(material, material_params, total_time=25.0, dt=0.0001, target=0.05, volume_fraction=0.5):
    num_steps = int(total_time / dt)

    # deps_xy = jnp.linspace(0, target, num_steps)
    # strain_rate_stack = jnp.zeros((num_steps, 3, 3)).at[:num_steps, 1, 0].set(deps_xy)
    # strain_rate_stack = strain_rate_stack.at[:num_steps, 0, 1].set(deps_xy)
    deps_xy = target / num_steps
    strain_rate_stack = jnp.zeros((num_steps, 3, 3)).at[:num_steps, (0, 1), (1, 0)].set(deps_xy)

    mask = jnp.zeros((3, 3)).astype(bool)

    stress_stack = jnp.zeros((num_steps, 3, 3))

    material = material.create(*material_params)

    return mix_control(material, strain_rate_stack, stress_stack, mask, volume_fraction, dt)


def isotropic_compression_wrapper(
    material, material_params, total_time=25.0, dt=0.0001, target=0.05, volume_fraction=0.5
):
    num_steps = int(total_time / dt)

    deps_xx_yy_zz = target / num_steps
    strain_rate_stack = jnp.zeros((num_steps, 3, 3)).at[:num_steps, (0, 1, 2), (0, 1, 2)].set(-deps_xx_yy_zz)

    mask = jnp.zeros((3, 3)).astype(bool)

    stress_stack = jnp.zeros((num_steps, 3, 3))

    material = material.create(*material_params)

    return mix_control(material, strain_rate_stack, stress_stack, mask, volume_fraction, dt)
