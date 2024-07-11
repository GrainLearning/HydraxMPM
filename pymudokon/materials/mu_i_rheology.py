"""Implementation, state and functions for isotropic linear elastic material."""

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import chex
from typing_extensions import Self

from .material import Material


def get_mu_I(I, mu_s, mu_d, I0):
    return mu_s + (mu_d - mu_s) * (1 / (1 + I0 / I))


def get_I_phi(phi, phi_c, I_phi):
    return -I_phi * jnp.log(phi / phi_c)


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
        )

    @jax.jit
    def update_stress_benchmark(
        self: Self,
        strain_rate: chex.Array,
        volume_fraction: chex.Array,
        dt: jnp.float32,
    ) -> Tuple[chex.Array, Self]:
        # jax.debug.print("volume_fraction {}", volume_fraction)
        stress_next = self.vmap_viscoplastic(strain_rate, volume_fraction, self.stress_ref)
        return self, stress_next
        # return self, stress_next

    @partial(jax.vmap, in_axes=(None, 0, 0, 0), out_axes=(0))
    def vmap_viscoplastic(self, strain_rate: chex.Array, volume_fraction: chex.Array, stress_ref: chex.Array):
        I = get_I_phi(volume_fraction, self.I_0, self.I_phi)

        mu_I = get_mu_I(I, self.mu_s, self.mu_d, self.I_0)

        volumetric_strain_rate = jnp.trace(strain_rate)  # compressive strain rate is positive

        deviatoric_strain_rate = strain_rate - (1 / 3.0) * volumetric_strain_rate * jnp.eye(3)

        shear_strain = jnp.sqrt(0.5 * (deviatoric_strain_rate @ deviatoric_strain_rate.T).trace())

        p = self.rho_p * (I / (shear_strain * self.d)) ** 2

        # Two assumptions made
        # (1) Drucker-Prager yield criterion with von-mises plastic potential and
        # (2) Alignment condition
        J2 = mu_I * p

        viscosity = J2 / shear_strain

        stress_next = p * jnp.eye(3) + viscosity * deviatoric_strain_rate

        return stress_next
