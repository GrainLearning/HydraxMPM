"""Implementation, state and functions for isotropic linear elastic material."""

import dataclasses
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array
from typing_extensions import Self

from ..core.particles import Particles
from .base_mat import BaseMaterial


def vmap_update(
    eps_e_prev: Array,
    vel_grad: Array,
    G: jnp.float32,
    K: jnp.float32,
    dt: jnp.float32,
) -> Tuple[Array, Array]:
    """Parallelized stress update for each particle.

    Tensors are mapped per particle via vmap (num_particles, dim) -> (dim,).

    Args:
        eps_e_prev (Array):
            Vectorized previous elastic strain tensor.
            Shape of full tensor is `(number of particles,dimension,dimension)`.
        vel_grad (Array):
            Vectorized velocity gradient tensor.
            Shape of full tensor is `(number of particles,dimension,dimension)`.
        G (jnp.float32):
            Shear modulus.
        K (jnp.float32):
            Bulk modulus.
        dt (jnp.float32):
            Time step.

    Returns:
        Tuple[Array, Array]:
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

    # pad for 3D stress tensor
    if dim == 2:
        s = jnp.pad(s, ((0, 1), (0, 1)), mode="constant")

    p = K * eps_e_v * jnp.eye(3)

    return s - p, eps_e


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=False)
class LinearIsotropicElastic(BaseMaterial):
    """State for the isotropic linear elastic material.

    Attributes:
        E (jnp.float32):
            Young's modulus.
        nu (Array):
            Poisson's ratio.
        G (Array):
            Shear modulus.
        K (Array):
            Bulk modulus.
        eps_e (Array):
            Elastic strain tensors.
            Shape is `(number of particles,dim,dim)`.
    """

    # TODO we can replace this with incremental form / or hyperelastic form
    E: Array
    nu: Array
    G: Array
    K: Array

    # arrays
    eps_e: Array

    @classmethod
    def register(cls: Self, E: jnp.float32, nu: jnp.float32, num_particles, dim: jnp.int16 = 3) -> Self:
        """Initialize the isotropic linear elastic material.

        Args:
            E (jnp.float32):
                Young's modulus.
            nu (jnp.float32):
                Poisson's ratio.
            num_particles (jnp.int16):
                Number of particles.
            dim (jnp.int16, optional):
                Dimension. Defaults to 3.

        Returns:
            LinearIsotropicElastic:
                Updated state for the isotropic linear elastic material.

        Example:
            >>> # Create a linear elastic material for plane strain and 2 particles
            >>> import pymudokon as pm
            >>> import jax.numpy as jnp
            >>> material = pm.LinearIsotropicElastic.register(1000, 0.2, 2, 2)
        """
        # TODO: How can we avoid the need for num_particles and dim?

        G = E / (2.0 * (1.0 + nu))
        K = E / (3.0 * (1.0 - 2.0 * nu))
        eps_e = jnp.zeros((num_particles, dim, dim), dtype=jnp.float32)
        return cls(E=E, nu=nu, G=G, K=K, eps_e=eps_e)

    @jax.jit
    def update_stress(
        self: Self,
        particles: Particles,
        dt: jnp.float32,  # unused
    ) -> Tuple[Particles, Self]:
        """Update stress and strain for all particles.

        Called by the MPM solver (e.g., see :func:`~usl.update`).

        Args:
            self (LinearIsotropicElastic):
                self reference.
            particles (ParticlesContainer):
                State of the particles prior to the update.

            dt (jnp.float32):
                Time step.

        Returns:
            Tuple[ParticlesContainer, LinearIsotropicElastic]:
                Updated particles and material state.

        Example:
            >>> import pymudokon as pm
            >>> # ...  Assume particles and material are initialized
            >>> particles, material = material.update_stress(particles, 0.001)
        """
        stress, eps_e = jax.vmap(vmap_update, in_axes=(0, 0, None, None, None), out_axes=(0, 0))(
            self.eps_e, particles.velgrads, self.G, self.K, 0.001
        )

        material = self.replace(eps_e=eps_e)

        particles = particles.replace(stresses=stress)

        return particles, material
