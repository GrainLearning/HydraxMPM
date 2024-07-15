"""Constitutive model for a nearly incompressible Newtonian fluid."""

from typing import Tuple

import jax
import jax.numpy as jnp
import chex
from jax import Array
from typing_extensions import Self

from ..particles.particles import Particles
from .material import Material


def vmap_update(
    vel_grad: Array,
    masses: Array,
    volumes: Array,
    volumes_original: Array,
    K: jnp.float32,
    gamma: jnp.float32,
    viscosity: jnp.float32,
) -> Tuple[Array]:
    """Compression is negative in this formulation."""
    dim = vel_grad.shape[0]

    density = masses / volumes

    density_original = masses / volumes_original

    pressure = K * ((density / density_original) ** gamma - 1.0)

    strain_rate = 0.5 * (vel_grad + vel_grad.T)

    if dim == 2:
        strain_rate = jnp.pad(strain_rate, ((0, 1), (0, 1)), mode="constant")

    vol_strain_rate = jnp.trace(strain_rate)

    dev_strain_rate = strain_rate - (vol_strain_rate / 3) * jnp.eye(3)

    return -pressure * jnp.eye(3) + 2.0 * viscosity * dev_strain_rate


@chex.dataclass
class NewtonFluid(Material):
    """Constitutive model for a nearly incompressible Newtonian fluid.

    Attributes:
        K (jnp.float32):
            Bulk modulus.
        viscosity (jnp.float32):
            Viscosity.
        gamma (jnp.float32):
            parameter for the equation of state. Defaults to 7.0 (water).
    """

    K: jnp.float32
    viscosity: jnp.float32
    gamma: jnp.float32

    @classmethod
    def create(cls: Self, K: jnp.float32, viscosity: jnp.float32, gamma: jnp.float32 = 7.0) -> Self:
        """Initialize Newtonian fluid material.

        Args:
            K (jnp.float32):
                Bulk modulus.
            viscosity (jnp.float32):
                Viscosity.
            gamma (jnp.float32):
                parameter for the equation of state. Defaults to 7.0 (water).

        Returns:
            NewtonFluid:
                Initialized material.

        Example:
            >>> import pymudokon as pm
            >>> material = pm.NewtonFluid.register(K=1.0e6, viscosity=0.1)
        """
        return cls(K=K, viscosity=viscosity, gamma=gamma, stress_ref=None)

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
        stress = jax.vmap(vmap_update, in_axes=(0, 0, 0, 0, None, None, None), out_axes=(0))(
            particles.velgrads,
            particles.masses,
            particles.volumes,
            particles.volumes_original,
            self.K,
            self.gamma,
            self.viscosity,
        )

        return particles.replace(stresses=stress), self
