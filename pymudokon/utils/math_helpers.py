"""
Module containing helper functions for mathematical operations.

To be used for post-processing and analysis of the simulation results.
"""

import jax
import jax.numpy as jnp

def get_pressure(stress: jax.Array)-> jax.Array:
    """Get the pressure from the stress tensor.

    Args:
        stress (jax.Array): Cauchy stress tensor `(num_samples, 3, 3)`.

    Returns:
        jax.Array: Pressure `(num_samples,)`.
    """
    stress = stress.reshape(-1, 3, 3)
    vmap_get_pressure = jax.vmap(lambda sigma: -(1 / 3.0) * (jnp.trace(sigma)), in_axes=(0))
    return vmap_get_pressure(stress)

def get_dev_stress(stress: jax.Array, pressure=None):
    """Get the deviatoric stress from the stress tensor.

    Args:
        stress (jax.Array): Cauchy stress tensor `(num_samples, 3, 3)`.
        pressure (jax.Array, optional): Pressure. Defaults to None.

    Returns:
        jax.Array: Pressure `(num_samples, 3, 3)`.
    """
    stress = stress.reshape(-1, 3, 3)
    if pressure is None:
        pressure = get_pressure(stress)

    vmap_get_dev_stress = jax.vmap(lambda sigma, p: sigma + jnp.eye(3) * p, in_axes=(0, 0))

    return vmap_get_dev_stress(stress, pressure)

def get_q_vm(stress: jax.Array, dev_stress=None):
    """Get the von Mises stress from the stress tensor sqrt(3/2*J2).

    Args:
        stress (jax.Array): Stress tensor `(num_samples, 3, 3)`.
        dev_stress (jax.Array, optional): Deviatoric stress tensors. Defaults to None.

    Returns:
        jax.Array: Von Mises stress `(num_samples,)`.
    """
    stress = stress.reshape(-1, 3, 3)
    if dev_stress is None:
        dev_stress = get_dev_stress(stress)

    vmap_get_q_vm = jax.vmap(lambda s: jnp.sqrt(3 * 0.5 * jnp.trace(s @ s.T)), in_axes=(0))
    return vmap_get_q_vm(dev_stress)

def get_tau(stress: jax.Array, dev_stress=None):
    """Get the shear stress from the stress (scalar) sqrt(1/2*J2).

    Args:
        stress (jax.Array): Stress tensor `(num_samples, 3, 3)`.
        dev_stress (_type_, optional): Deviatoric stress tensor `(num_samples, 3, 3)`. Defaults to None.

    Returns:
        jax.Array: Deviatoric stress scalar`(num_samples,)`.
    """
    stress = stress.reshape(-1, 3, 3)
    if dev_stress is None:
        dev_stress = get_dev_stress(stress)

    vmap_get_tau = jax.vmap(lambda s: 0.5 * jnp.trace(s @ s.T), in_axes=(0))
    return vmap_get_tau(dev_stress)

def get_volumetric_strain(strain: jax.Array,dim=3):
    """Get the volumetric strain from the strain tensor.

    Args:
        strain (jax.Array): strain tensor `(num_samples, 3, 3)`.

    Returns:
        jax.Array: Volumetric strain `(num_samples,)`.
    """
    strain = strain.reshape(-1, dim, dim)
    vmap_get_volumetric_strain = jax.vmap(lambda eps: -(jnp.trace(eps)), in_axes=(0))
    return vmap_get_volumetric_strain(strain)

def get_dev_strain(strain: jax.Array, volumetric_strain=None,dim =3):
    strain = strain.reshape(-1, dim, dim)
    if volumetric_strain is None:
        volumetric_strain = get_volumetric_strain(strain)

    vmap_get_dev_strain = jax.vmap(lambda eps, eps_v: eps + (1.0 / dim) * jnp.eye(3) * eps_v, in_axes=(0, 0))
    return vmap_get_dev_strain(strain, volumetric_strain)

def get_gamma(strain: jax.Array, dev_strain=None, dim=3):
    strain = strain.reshape(-1, dim, dim)
    if dev_strain is None:
        dev_strain = get_dev_strain(strain)

    vmap_get_gamma = jax.vmap(lambda eps: jnp.trace(eps @ eps.T), in_axes=(0))
    return vmap_get_gamma(dev_strain)


def get_KE(mass: jax.Array, vel: jax.Array):
    """Get the kinetic energy from the mass and velocity of a node or particle.

    Args:
        mass (jax.Array): Mass `(num_samples,)`.
        vel (jax.Array): Velocity `(num_samples, dim)`.

    Returns:
        jax.Array: Kinetic energy `(num_samples,)`.
    """
    vmap_get_KE = jax.vmap(lambda m, v: 0.5 * m * jnp.sum(v ** 2, axis=-1), in_axes=(0, 0))
    return vmap_get_KE(mass, vel)

