"""Implementation, state and functions for isotropic linear elastic material."""

from functools import partial
from typing import Tuple
from typing_extensions import Self

import chex
import jax
import jax.numpy as jnp

from ..particles.particles import Particles
from ..utils.math_helpers import get_sym_tensor_stack
from .material import Material


def get_mu_I(I, mu_s, mu_d, I0):
    return mu_s + (mu_d - mu_s) * (1 / (1 + I0 / I))


def get_I_phi(phi, phi_c, I_phi):
    return I_phi * jnp.log(phi_c / phi)


def get_pressure(dgammadt, I, d, rho_p):
    return rho_p * ((dgammadt * d) / I) ** 2


@chex.dataclass
class MuI(Material):
    mu_s: jnp.float32
    mu_d: jnp.float32
    I_0: jnp.float32
    phi_c: jnp.float32
    I_phi: jnp.float32
    rho_p: jnp.float32
    d: jnp.float32

    @classmethod
    def create(
        cls: Self,
        mu_s: jnp.float32,
        mu_d: jnp.float32,
        I_0: jnp.float32,
        phi_c: jnp.float32,
        I_phi: jnp.float32,
        rho_p: jnp.float32,
        d: jnp.float32,
        absolute_density: jnp.float32 = 0.0,
    ) -> Self:
        return cls(
            mu_s=mu_s,
            mu_d=mu_d,
            I_0=I_0,
            d=d,
            rho_p=rho_p,
            I_phi=I_phi,
            phi_c=phi_c,
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
        def get_deviatoric_strain_rate(L, F):
            dot_F = L @ F
            u, s, vh = jnp.linalg.svd(dot_F)
            eps = jnp.log(s)
            jax.debug.print("eps {} F {}", eps, F)
            return eps

        eps = jax.vmap(get_deviatoric_strain_rate)(L_stack, F_stack)

        deps_dt_stack = get_sym_tensor_stack(L_stack)
        jax.debug.print("deps_dt_stack {}", deps_dt_stack)

        stress_next_stack = self.vmap_viscoplastic(deps_dt_stack, phi_stack)

        return stress_next_stack, self

    @partial(jax.vmap, in_axes=(None, 0, 0), out_axes=(0))
    def vmap_viscoplastic(self, strain_rate: chex.Array, volume_fraction: chex.Array):
        I = get_I_phi(volume_fraction, self.phi_c, self.I_phi)

        mu_I = get_mu_I(I, self.mu_s, self.mu_d, self.I_0)

        volumetric_strain_rate = -jnp.trace(
            strain_rate
        )  # compressive strain rate is positive

        deviatoric_strain_rate = strain_rate + (
            1 / 3.0
        ) * volumetric_strain_rate * jnp.eye(3)

        dgamma_dt = jnp.sqrt(
            0.5 * (deviatoric_strain_rate @ deviatoric_strain_rate.T).trace()
        )

        # p = self.rho_p * (I / (shear_strain * self.d)) ** 2
        p = get_pressure(dgamma_dt, I, self.d, self.rho_p)
        # jax.debug.print("dgamma_dt {}", dgamma_dt)
        # Two assumptions made
        # (1) Drucker-Prager yield criterion with von-mises plastic potential and
        # (2) Alignment condition
        J2 = mu_I * p

        viscosity = J2 / dgamma_dt

        stress_next = -p * jnp.eye(3) + viscosity * deviatoric_strain_rate

        return stress_next

    def get_p_ref(self, phi, dgamma_dt):
        I = get_I_phi(phi, self.phi_c, self.I_phi)

        return get_pressure(dgamma_dt, I, self.d, self.rho_p)
