"""Implementation, state and functions for isotropic linear elastic material."""

from functools import partial
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import optimistix as optx
from typing_extensions import Self

from .material import Material


def get_mu_I(I, mu_s, mu_d, I0):
    return mu_s + (mu_d - mu_s) * (1.0 / (1 + I0 / I))


def get_mu_I_correction(mu_I, p, p0, k, d):
    p_star = p * (d / k)
    return mu_I * (1.0 - (p_star / p0) ** 0.5)


# def get_I_correction(I, p, p_phi, I_phi):
#     return I_phi * (p / p_phi) + I


# def get_I_phi(phi, phi_c, I_phi):
#     return I_phi * jnp.log(phi_c / phi)


# def get_pressure(dgammadt, I, d, rho_p):
#     return rho_p * ((dgammadt * d) / I) ** 2


@chex.dataclass
class MuISoft(Material):
    mu_s: jnp.float32
    mu_d: jnp.float32
    I_0: jnp.float32
    phi_c: jnp.float32
    I_phi: jnp.float32
    rho_p: jnp.float32
    d: jnp.float32

    p_phi: jnp.float32
    p0: jnp.float32
    k: jnp.float32

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
        p_phi: jnp.float32,
        p0: jnp.float32,
        k: jnp.float32,
        stress_ref: chex.Array = None,
        num_particles: jnp.int32 = 1,
        dim: jnp.int16 = 3,
    ) -> Self:
        if stress_ref is None:
            stress_ref = jnp.zeros((num_particles, dim, dim), dtype=jnp.float32)

        return cls(
            stress_ref=stress_ref,
            mu_s=mu_s,
            mu_d=mu_d,
            I_0=I_0,
            d=d,
            rho_p=rho_p,
            I_phi=I_phi,
            phi_c=phi_c,
            p_phi=p_phi,
            p0=p0,
            k=k,
        )

    def update_stress_benchmark(
        self: Self,
        stress_prev: chex.Array,
        strain_rate: chex.Array,
        volume_fraction: chex.Array,
        dt: jnp.float32,
    ) -> Tuple[chex.Array, Self]:
        # jax.debug.print("volume_fraction {}", volume_fraction)
        stress_next = self.vmap_viscoplastic(strain_rate, volume_fraction, self.stress_ref)
        return self, stress_next

    @partial(jax.vmap, in_axes=(None, 0, 0, 0), out_axes=(0))
    def vmap_viscoplastic(
        self, strain_rate: chex.Array, volume_fraction: chex.Array, stress_ref: chex.Array
    ):
        volumetric_strain_rate = -jnp.trace(strain_rate)  # compressive strain rate is positive

        deviatoric_strain_rate = strain_rate + (1 / 3.0) * volumetric_strain_rate * jnp.eye(3)

        dgamma_dt = jnp.sqrt(0.5 * (deviatoric_strain_rate @ deviatoric_strain_rate.T).trace())

        def dil(p, args):
            p_star = p * (self.d / self.k)
            term1 = p_star / self.p_phi + self.I_phi * jnp.log(self.phi_c / volume_fraction)
            term2 = (dgamma_dt * self.d) / jnp.sqrt(p / self.rho_p)
            sol = term1 - term2
            return sol

        solver = optx.Newton(rtol=1e-3, atol=1e-12)
        sol = optx.root_find(dil, solver, 0.1, throw=False)
        p = sol.value

        I = (dgamma_dt * self.d) / jnp.sqrt(p / self.rho_p)
        # I = p * (self.d / self.k) / self.p_phi + self.I_phi * jnp.log(self.phi_c / volume_fraction)
        mu_I = get_mu_I(I, self.mu_s, self.mu_d, self.I_0)

        mu_I_corr = get_mu_I_correction(mu_I, p, self.p0, self.k, self.d)

        # Two assumptions made
        # (1) Drucker-Prager yield criterion with von-mises plastic potential and
        # (2) Alignment condition
        # J2 = mu_I * p
        # J2 = mu_I_corr * p
        J2 = mu_I_corr * p

        viscosity = J2 / (dgamma_dt)

        stress_next = -p * jnp.eye(3) + viscosity * deviatoric_strain_rate

        return stress_next
