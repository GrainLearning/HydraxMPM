"""Implementation, state and functions for isotropic linear elastic material.

The module contains the following main components:

- LinearElasticContainer:
    JAX pytree (NamedTuple) that stores the parameters and state of the material.
- init:
    Initialize the state for the linear isotropic elastic material.
- vmap_update:
    Vectorized stress update for each particle.
- update_stress:
    Update stress and strain for all particles on a container level.
"""

from typing import NamedTuple, Tuple, Union

import jax
import jax.numpy as jnp

from ..core.particles import ParticlesContainer


class LinearElasticContainer(NamedTuple):
    """State for the isotropic linear elastic material.

    Attributes:
        E (jnp.float32):
            Young's modulus.
        nu (Union[jnp.array, jnp.float32]):
            Poisson's ratio.
        G (Union[jnp.array, jnp.float32]):
            Shear modulus.
        K (Union[jnp.array, jnp.float32]):
            Bulk modulus.
        eps_e (Union[jnp.array, jnp.int32]):
            Elastic strain tensors.
            Shape is `(number of particles,3,3)`.
    """
    # TODO we can replace this with incremental form / or hyperelastic form
    E: Union[jnp.array, jnp.float32]
    nu: Union[jnp.array, jnp.float32]
    G: Union[jnp.array, jnp.float32]
    K: Union[jnp.array, jnp.float32]

    # arrays
    eps_e: Union[jnp.array, jnp.float32]
    
def init(
    E: jnp.float32, nu: jnp.float32, num_particles, dim: jnp.int16 = 3
) -> LinearElasticContainer:
    """Initialize the isotropic linear elastic material.

    Args:
        E (jnp.float32): Young's modulus.
        nu (jnp.float32): Poisson's ratio.
        num_particles (jnp.int16): Number of particles.
        dim (jnp.int16, optional): Dimension. Defaults to 3.

    Returns:
        LinearElasticContainer: Updated state for the isotropic linear elastic material.

    Example:
        >>> # Create a linear elastic material for plane strain and 2 particles
        >>> import pymudokon as pm
        >>> import jax.numpy as jnp
        >>> material_state = pm.materials.linearelastic.init(1000, 0.2, 2, 2)
    """
    # TODO: How can we avoid the need for num_particles and dim?

    G = E / (2.0 * (1.0 + nu))
    K = E / (3.0 * (1.0 - 2.0 * nu))
    eps_e = jnp.zeros((num_particles, dim, dim), dtype=jnp.float32)
    return LinearElasticContainer(
        E=E, nu=nu, G=G, K=K, eps_e=eps_e)


def vmap_update(
    eps_e_prev: Union[jnp.array, jnp.float32],
    vel_grad,
    G: jnp.float32,
    K: jnp.float32,
    dt: jnp.float32,
) -> Tuple[Union[jnp.array, jnp.float32], Union[jnp.array, jnp.float32]]:
    """Update stress and strain for a single particle.

    Parallelized stress update for each particle. This function is called with vmap.

    Args:
        eps_e_prev (Union[jnp.array, jnp.float32]):
            Vectorized previous elastic strain tensor.
            Shape of full tensor is `(number of particles,dimension,dimension)`.
        vel_grad (Union[jnp.array, jnp.float32]):
            Vectorized velocity gradient tensor.
            Shape of full tensor is `(number of particles,dimension,dimension)`.
        G (jnp.float32):
            Shear modulus.
        K (jnp.float32):
            Bulk modulus.
        dt (jnp.float32):
            Time step.

    Returns:
        Tuple[Union[jnp.array, jnp.float32], Union[jnp.array, jnp.float32]]:
            Updated stress and elastic strain tensors.

    Example:
        >>> import pymudokon as pm
        >>> import jax.numpy as jnp
        >>> eps_e = jnp.ones((2, 2, 2), dtype=jnp.float32)
        >>> vel_grad = jnp.ones((2, 2, 2), dtype=jnp.float32)
        >>> stress, eps_e = jax.vmap(vmap_update, in_axes=(0, 0, None, None, None))(
        >>> eps_e, vel_grad, 1000, 1000, 0.001)
    """
    dim = vel_grad.shape[0]

    # strain increment
    deps = 0.5 * (vel_grad + vel_grad.T) * dt

    eps_e = eps_e_prev + deps

    eps_e_v = -jnp.trace(eps_e)

    eps_e_d = eps_e + (eps_e_v / dim) * jnp.eye(dim)

    s = 2.0 * G * eps_e_d

    # pad for 3x3 stress tensor
    if dim == 2:
        s = jnp.pad(s, ((0, 1), (0, 1)), mode="constant")

    p = K * eps_e_v * jnp.eye(3)

    return s - p, eps_e


def update_stress(
    particles: ParticlesContainer, material: LinearElasticContainer, dt: jnp.float32
) -> Tuple[ParticlesContainer, LinearElasticContainer]:
    """Update stress and strain for all particles.

    Called by the MPM solver (e.g., see :func:`~usl.update`).

    Args:
        particles (ParticlesContainer): _description_
        material (LinearElasticContainer): _description_
        dt (jnp.float32): _description_

    Returns:
        Tuple[ParticlesContainer, LinearElasticContainer]:
        Updated particles and material state.

    Example:
        >>> import pymudokon as pm
        >>> # ...  Assume particles_state and material_state are initialized
        >>> particles_state, material_state = \
            pm.materials.linearelastic.update_stress(particles_state, material_state, 0.001)
    """
    stress, eps_e = jax.vmap(vmap_update, in_axes=(0, 0, None, None, None))(
        material.eps_e, particles.velgrad_array, material.G, material.K, 0.001
    )

    material = material._replace(eps_e=eps_e)

    particles = particles._replace(stresses_array=stress)

    return particles, material
