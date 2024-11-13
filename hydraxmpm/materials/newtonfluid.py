"""Constitutive model for a nearly incompressible Newtonian fluid."""

from typing import Tuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from typing_extensions import Self

from ..config.mpm_config import MPMConfig
from ..particles.particles import Particles
from .material import Material


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


    def __init__(
        self: Self,
        config: MPMConfig,
        K: jnp.float32 = 2.0 * 10**6,
        viscosity: jnp.float32 = 0.001,
        gamma: jnp.float32 = 7.0,
    ) -> Self:
        """Create a Newtonian fluid material.

        Assumes reference volume fraction is 1.0.
        """

        self.K = K
        self.viscosity = viscosity
        self.gamma = gamma
        
        super().__init__(config)



    def update_from_particles(self: Self, particles: Particles) -> Tuple[Particles, Self]:
        """Update the material state and particle stresses for MPM solver."""
        
        # uses reference density to calculate the pressure
        # Todo check if rho_p / rho_ref == phi/phi_ref...
        phi_stack = particles.volume0_stack / particles.volume_stack

        vmap_update_ip = jax.vmap(fun=self.update_ip, in_axes=(0, 0, 0, 0))
        
        new_stress_stack = vmap_update_ip(
            particles.stress_stack,
            particles.F_stack,
            particles.L_stack,
            phi_stack,
        )
        new_particles = eqx.tree_at(
            lambda state: (state.stress_stack),
            particles,
            (new_stress_stack),
        )

        return new_particles, self

    def update_ip(
        self: Self,
        stress_prev: chex.Array,
        F: chex.Array,
        L: chex.Array,
        phi: chex.Array
    ) -> Tuple[chex.Array, Self]:
        
        pressure = self.K * (phi**self.gamma - 1.0)
                
        deps_dt = 0.5 * (L + L.T)

        if self.config.dim == 2:
            deps_dt = deps_dt.at[:, [2, 2]].set(0.0)

        deps_v_dt = jnp.trace(deps_dt)

        deps_dev_dt = deps_dt - (deps_v_dt / 3) * jnp.eye(3)

        return -pressure * jnp.eye(3) + self.viscosity * deps_dev_dt

