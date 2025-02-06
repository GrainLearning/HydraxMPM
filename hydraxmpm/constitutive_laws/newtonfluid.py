"""Constitutive model for a nearly incompressible Newtonian fluid."""

from typing import Tuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from typing_extensions import Self

from ..particles.particles import Particles
from .constitutive_law import Material

from ..common.types import TypeFloat, TypeFloatMatrix3x3, TypeInt


class NewtonFluid(Material):
    """Nearly incompressible Newtonian fluid.

    Attributes:
        K: Bulk modulus.
        viscosity: Viscosity.
        gamma: Exponent.
    """

    K: TypeFloat
    viscosity: TypeFloat
    gamma: TypeFloat

    dim: TypeInt = eqx.field(static=True)

    def __init__(
        self: Self,
        K: TypeFloat = 2.0 * 10**6,
        viscosity: TypeFloat = 0.001,
        gamma: TypeFloat = 7.0,
        **kwargs,
    ) -> Self:
        """Initialize the nearly incompressible Newtonian fluid material."""

        self.K = K
        self.viscosity = viscosity
        self.gamma = gamma

        self.dim = kwargs.get("dim", 3)

    def update_from_particles(
        self: Self, particles: Particles
    ) -> Tuple[Particles, Self]:
        """Update the material state and particle stresses for MPM solver."""

        # uses reference density to calculate the pressure
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
        stress_prev: TypeFloatMatrix3x3,
        F: TypeFloatMatrix3x3,
        L: TypeFloatMatrix3x3,
        phi: TypeFloat,
    ) -> TypeFloatMatrix3x3:
        pressure = self.K * (phi**self.gamma - 1.0)

        deps_dt = 0.5 * (L + L.T)

        if self.dim == 2:
            deps_dt = deps_dt.at[:, [2, 2]].set(0.0)

        deps_v_dt = jnp.trace(deps_dt)

        deps_dev_dt = deps_dt - (deps_v_dt / 3) * jnp.eye(3)

        return -pressure * jnp.eye(3) + self.viscosity * deps_dev_dt
