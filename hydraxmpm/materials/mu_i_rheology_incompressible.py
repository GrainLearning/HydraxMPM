"""Implementation, state and functions for isotropic linear elastic material."""

from typing import Tuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from typing_extensions import Self

from ..config.mpm_config import MPMConfig
from ..particles.particles import Particles
from ..utils.math_helpers import (
    get_dev_strain,
    get_inertial_number,
    get_scalar_shear_strain,
    get_sym_tensor,
)
from .material import Material


def get_mu_I(I, mu_s, mu_d, I0):
    return mu_s + (mu_d - mu_s) * (1 / (1 + I0 / I))


def get_mu_I_regularized_exp(I, mu_s, mu_d, I0, pen, dgamma_dt):
    s = 1.0 / jnp.sqrt(dgamma_dt**2 + pen**2)
    return mu_s * s + (mu_d - mu_s) * (1.0 / (1.0 + I0 / I))


def get_I_phi(phi, phi_c, I_phi):
    return -I_phi * jnp.log(phi / phi_c)


def get_pressure(dgammadt, I, d, rho_p):
    return rho_p * ((dgammadt * d) / I) ** 2


class MuI_incompressible(Material):
    mu_s: jnp.float32
    mu_d: jnp.float32
    I_0: jnp.float32
    rho_p: jnp.float32
    d: jnp.float32
    K: jnp.float32

    """
    (nearly) incompressible mu I
    
    Tensorial form similar to
    Jop, Pierre, Yoël Forterre, and Olivier Pouliquen. "A constitutive law for dense granular flows." Nature 441.7094 (2006): 727-730.
    
    mu I regularized by
    Franci, Alessandro, and Massimiliano Cremonesi. "3D regularized μ (I)-rheology for granular flows simulation." Journal of Computational Physics 378 (2019): 257-277.
    
    Pressure term by
    
    Salehizadeh, A. M., and A. R. Shafiei. "Modeling of granular column collapses with μ (I) rheology using smoothed particle hydrodynamic method." Granular Matter 21.2 (2019): 32.

    """

    def __init__(
        self: Self,
        config: MPMConfig,
        mu_s: jnp.float32,
        mu_d: jnp.float32,
        I_0: jnp.float32,
        rho_p: jnp.float32,
        d: jnp.float32,
        K: jnp.float32 = 1.0,
    ) -> Self:
        self.mu_s = mu_s

        self.mu_d = mu_d

        self.I_0 = I_0

        self.rho_p = rho_p

        self.d = d

        self.K = K

        super().__init__(config)

    def get_p_ref(self, phi):
        return jnp.maximum(self.K * (phi - 1.0), 1e-12)

    def update_from_particles(
        self: Self, particles: Particles
    ) -> Tuple[Particles, Self]:
        """Update the material state and particle stresses for MPM solver."""

        # not really solid volume fraction but delta change in density relative to original
        density_stack = particles.mass_stack / particles.volume_stack
        density0_stack = particles.mass_stack / particles.volume0_stack

        phi_stack = density_stack / density0_stack

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
        phi: chex.Array,
    ) -> Tuple[chex.Array, Self]:
        deps_dt = get_sym_tensor(L)

        p = jnp.maximum(self.K * (phi - 1.0), 1e-12)

        deps_dev_dt = get_dev_strain(deps_dt, dim=self.config.dim)

        dgamma_dt = get_scalar_shear_strain(deps_dt, dim=self.config.dim)

        I = get_inertial_number(p, dgamma_dt, self.d, self.rho_p)

        # regularize I
        I = jnp.maximum(I, 1e-9)

        alpha = 0.000001
        eta_E_s = p * self.mu_s / jnp.sqrt(dgamma_dt * dgamma_dt + alpha * alpha)

        mu_I_delta = (self.mu_d - self.mu_s) / (1.0 + self.I_0 / I)

        eta_delta = p * mu_I_delta / dgamma_dt

        eta = eta_E_s + eta_delta

        stress_next = -p * jnp.eye(3) + eta * deps_dev_dt

        return stress_next
