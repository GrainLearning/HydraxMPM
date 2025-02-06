"""Implementation, state and functions for isotropic linear elastic material."""

from functools import partial
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import optimistix as optx
from typing_extensions import Self

from ...particles.particles import Particles
from ...utils.math_helpers import (
    get_dev_strain,
    get_inertial_number,
    get_scalar_shear_strain,
    get_sym_tensor_stack,
)
from ..constitutive_law import Material


def get_mu_I(I, mu_s, mu_d, I0):
    return mu_s + (mu_d - mu_s) * (1 / (1 + I0 / I))


def get_mu_I_regularized_exp(I, mu_s, mu_d, I0, pen, dgamma_dt):
    # s = (1.0-jnp.exp(-dgamma_dt/pen))
    s = 1.0 / jnp.sqrt(dgamma_dt**2 + pen**2)
    return mu_s * s + (mu_d - mu_s) * (1.0 / (1.0 + I0 / I))


def get_I_phi(phi, phi_c, I_phi):
    return -I_phi * jnp.log(phi / phi_c)


def get_pressure(dgammadt, I, d, rho_p):
    return rho_p * ((dgammadt * d) / I) ** 2


@chex.dataclass
class MRMSteady(Material):
    mu_s: jnp.float32
    mu_d: jnp.float32
    I_0: jnp.float32
    phi_c: jnp.float32
    I_phi: jnp.float32
    rho_p: jnp.float32
    d: jnp.float32
    k_p: jnp.float32
    lam: jnp.float32
    dim: jnp.int32

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
        k_p: jnp.float32,
        lam: jnp.float32,
        absolute_density: jnp.float32 = 0.0,
        dim: jnp.int32 = 3,
    ) -> Self:
        jax.debug.print("{}", K)
        return cls(
            mu_s=mu_s,
            mu_d=mu_d,
            I_0=I_0,
            d=d,
            rho_p=rho_p,
            I_phi=I_phi,
            phi_c=phi_c,
            k_p=k_p,
            lam=lam,
            absolute_density=absolute_density,
            dim=dim,
        )

    def update_from_particles(
        self: Self, particles: Particles, dt: jnp.float32
    ) -> Tuple[Particles, Self]:
        """Update the material state and particle stresses for MPM solver."""

        phi_stack = particles.get_solid_volume_fraction_stack(self.rho_p)

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
        deps_dt_stack = get_sym_tensor_stack(L_stack)

        stress_next_stack = self.vmap_viscoplastic(
            deps_dt_stack, phi_stack, stress_prev_stack
        )

        # Debug
        # jax.debug.print("{}",stress_next_stack)
        # I_stack_debug = jax.vmap(get_I_phi, in_axes=(0,None,None))(phi_stack, self.phi_c,self.I_phi)
        # jax.debug.print("I mean {} min {} max {}",I_stack_debug.mean(),I_stack_debug.min(), I_stack_debug.max())
        # get_dev_strain = get_dev_strain_stack(deps_dt_stack,dim=self.dim)
        # jax.debug.print("{}",get_dev_strain)
        # jax.debug.print("phi_stack{}",phi_stack)
        return stress_next_stack, self

    @partial(jax.vmap, in_axes=(None, 0, 0, 0), out_axes=(0))
    def vmap_viscoplastic(
        self, strain_rate: chex.Array, phi: chex.Array, stress_prev: chex.Array
    ):
        deps_dev_dt = get_dev_strain(strain_rate, dim=self.dim)

        dgamma_dt = get_scalar_shear_strain(strain_rate, dim=self.dim)

        d_k = self.d / self.k_p

        def retief_rheology(p, args):
            I = get_inertial_number(p, dgamma_dt, self.d, self.rho_p)
            # I = jnp.maximum(I,1e-12)

            rhs = jnp.log(phi / self.phi_c)
            lhs_f = -I / self.I_phi

            p_star = p * d_k
            lhs_s = self.lam * jnp.log(1 + p_star)

            return rhs - lhs_f - lhs_s

        def find_root():
            solver = optx.Newton(rtol=1e-12, atol=1e-12)
            I = get_I_phi(phi, self.phi_c, self.I_phi)
            # I = jnp.maximum(I,1e-12)

            p_guess = jax.lax.cond(
                phi > self.phi_c,
                lambda: ((phi / self.phi_c) ** (1.0 / self.lam) - 1.0) * (1.0 / d_k),
                lambda: get_pressure(dgamma_dt, I, self.d, self.rho_p),
            )
            sol = optx.minimise(
                retief_rheology, solver, p_guess, throw=False, max_steps=20
            )
            return jnp.nan_to_num(sol.value, nan=1.0, posinf=0.0, neginf=0.0)

        p = find_root()

        I = get_inertial_number(p, dgamma_dt, self.d, self.rho_p)
        # I = jnp.maximum(I,1e-12)

        alpha = 0.000001
        # eta_E_s = p*self.mu_s/jnp.sqrt(dgamma_dt*dgamma_dt + alpha*alpha)

        eta_E_s = p * self.mu_s / dgamma_dt

        mu_I_delta = (self.mu_d - self.mu_s) / (1.0 + self.I_0 / I)

        eta_delta = p * mu_I_delta / dgamma_dt

        eta = eta_E_s + eta_delta

        stress_next = -p * jnp.eye(3) + eta * deps_dev_dt
        return stress_next

    def get_p_ref(self, phi, dgamma_dt):
        I = get_I_phi(phi, self.phi_c, self.I_phi)

        d_k = self.d / self.k_p

        def retief_rheology(p, args):
            I = get_inertial_number(p, dgamma_dt, self.d, self.rho_p)
            # I = jnp.maximum(I,1e-12)

            rhs = jnp.log(phi / self.phi_c)
            lhs_f = -I / self.I_phi

            p_star = p * d_k
            lhs_s = self.lam * jnp.log(1 + p_star)

            return rhs - lhs_f - lhs_s

        def find_root():
            solver = optx.Newton(rtol=1e-12, atol=1e-12)
            I = get_I_phi(phi, self.phi_c, self.I_phi)
            # I = jnp.maximum(I,1e-12)

            p_guess = jax.lax.cond(
                phi > self.phi_c,
                lambda: ((phi / self.phi_c) ** (1.0 / self.lam) - 1.0) * (1.0 / d_k),
                lambda: get_pressure(dgamma_dt, I, self.d, self.rho_p),
            )
            sol = optx.minimise(
                retief_rheology, solver, p_guess, throw=False, max_steps=40
            )
            return jnp.nan_to_num(sol.value, nan=1.0, posinf=0.0, neginf=0.0)

        p = find_root()

        return p


# # left = phi / self.phi_c
# # lam = 1.0/0.33
# # right = (1.0 + p*(self.d/1e8))**lam *jnp.exp(-I/self.I_phi)

# # p_phi = 0.33
# p_phi = 1.0
# lam = (1.0/p_phi)
# K= 1e3
# I = get_inertial_number(p,dgamma_dt,self.d,self.rho_p)

# left = phi / self.phi_c
# # right = (1.0 + p*(self.d/K))**lam *jnp.exp(-I/self.I_phi)

# right = (1.0 + p)**lam2 *jnp.exp(-I/self.I_phi)
