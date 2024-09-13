"""Constitutive model for a nearly incompressible Newtonian fluid."""

from typing import Tuple
from typing_extensions import Self

import chex
import jax
import jax.numpy as jnp

from ..particles.particles import Particles
from .material import Material


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
        phi_ref (jnp.float32): Reference solid volume fraction.
    """

    K: jnp.float32
    viscosity: jnp.float32
    gamma: jnp.float32

    @classmethod
    def create(
        cls: Self,
        K: jnp.float32 = 2.0 * 10**6,
        viscosity: jnp.float32 = 0.001,
        gamma: jnp.float32 = 7.0,
        phi_ref: jnp.float32 = 1.0,
        absolute_density: jnp.float32 = 1000.0,
    ) -> Self:
        """Create a Newtonian fluid material.

        Assumes reference volume fraction is 1.0.
        """
        return cls(
            K=K,
            viscosity=viscosity,
            gamma=gamma,
            absolute_density=absolute_density,
        )

    def update_from_particles(
        self: Self, particles: Particles, dt: jnp.float32
    ) -> Tuple[Particles, Self]:
        """Update the material state and particle stresses for MPM solver."""
        # uses reference density to calculate the pressure
        # Todo check if rho_p / rho_ref == phi/phi_ref...

        phi_stack = particles.volume0_stack / particles.volume_stack

        stress_stack, self = self.update(
            particles.stress_stack, particles.F_stack, particles.L_stack, phi_stack, dt
        )

        return particles.replace(stress_stack=stress_stack), self

    def update(
        self: Self,
        stress_prev_stack: chex.Array,
        F_stack: chex.Array,
        L_stack: chex.Array,
        phi_stack: chex.Array,
        dt: jnp.float32,
    ) -> Tuple[chex.Array, Self]:
        stress = jax.vmap(self.vmap_update_stress, in_axes=(0, 0), out_axes=(0))(
            L_stack, phi_stack
        )

        return stress, self

    def vmap_update_stress(
        self: Self, L: chex.Array, phi: chex.Array
    ) -> Tuple[chex.Array]:
        """Compression is negative in this formulation."""
        dim = L.shape[0]

        pressure = self.K * (phi**self.gamma - 1.0)

        deps_dt = 0.5 * (L + L.T)

        if dim == 2:
            deps_dt = deps_dt.at[:, [2, 2]].set(0.0)

        deps_v_dt = jnp.trace(deps_dt)

        deps_dev_dt = deps_dt - (deps_v_dt / 3) * jnp.eye(3)

        return -pressure * jnp.eye(3) + self.viscosity * deps_dev_dt
