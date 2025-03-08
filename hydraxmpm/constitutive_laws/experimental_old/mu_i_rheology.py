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
class MuI(Material):
    mu_s: jnp.float32
    mu_d: jnp.float32
    I_0: jnp.float32
    phi_c: jnp.float32
    I_phi: jnp.float32
    rho_p: jnp.float32
    d: jnp.float32
    dim: jnp.int32
    K: jnp.float32

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
        K: jnp.float32 = 1.0,
        absolute_density: jnp.float32 = 0.0,
        dim: jnp.int32 = 3,
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
            K=K,
            dim=dim,
        )

    def update_from_particles(
        self: Self, particles: Particles, dt: jnp.float32
    ) -> Tuple[Particles, Self]:
        """Update the material state and particle stresses for MPM solver."""

        density_stack = particles.mass_stack / particles.volume_stack
        phi_stack = density_stack / self.rho_p

        # density_stack = particles.mass_stack/particles.volume_stack
        # density0_stack = particles.mass_stack/particles.volume0_stack

        # phi_stack = density_stack/density0_stack
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
        # get_dev_strain = get_dev_strain_stack(deps_dt_stack,dim=self.dim)
        # jax.debug.print("{}",get_dev_strain)
        # jax.debug.print("phi_stack{}",phi_stack)
        stress_next_stack = self.vmap_viscoplastic(
            deps_dt_stack, phi_stack, stress_prev_stack
        )
        # jax.debug.print("{}",stress_next_stack)
        # I_stack_debug = jax.vmap(get_I_phi, in_axes=(0,None,None))(phi_stack, self.phi_c,self.I_phi)
        # jax.debug.print("I mean {} min {} max {}",I_stack_debug.mean(),I_stack_debug.min(), I_stack_debug.max())

        return stress_next_stack, self

    @partial(jax.vmap, in_axes=(None, 0, 0, 0), out_axes=(0))
    def vmap_viscoplastic(
        self, strain_rate: chex.Array, phi: chex.Array, stress_prev: chex.Array
    ):
        deps_dev_dt = get_dev_strain(strain_rate, dim=self.dim)

        dgamma_dt = get_scalar_shear_strain(strain_rate, dim=self.dim)

        # I = jnp.maximum(I,self.I_phi/10)
        #
        # p = jnp.nan_to_num(get_pressure(dgamma_dt, I, self.d, self.rho_p),1e-12)
        # p = jnp.maximum(p,1e-12)
        lam2 = 0.0186

        def flow():
            def retief_rheology(p, args):
                # I = get_inertial_number(p,dgamma_dt,self.d,self.rho_p)
                # left = phi / self.phi_c
                # lam = 1.0/0.33
                # right = (1.0 + p*(self.d/1e8))**lam *jnp.exp(-I/self.I_phi)

                # p_phi = 0.33
                p_phi = 1.0
                lam = 1.0 / p_phi
                K = 1e3
                I = get_inertial_number(p, dgamma_dt, self.d, self.rho_p)

                left = phi / self.phi_c
                # right = (1.0 + p*(self.d/K))**lam *jnp.exp(-I/self.I_phi)

                right = (1.0 + p) ** lam2 * jnp.exp(-I / self.I_phi)

                return right - left

            def find_root():
                solver = optx.Newton(rtol=1e-12, atol=1e-12)
                I = get_I_phi(phi, self.phi_c, self.I_phi)

                p_guess = jax.lax.cond(
                    phi > self.phi_c,
                    lambda: (phi / self.phi_c) ** (1.0 / lam2) - 1.0,
                    lambda: get_pressure(dgamma_dt, I, self.d, self.rho_p),
                )
                sol = optx.minimise(
                    retief_rheology, solver, p_guess, throw=False, max_steps=20
                )
                return jnp.nan_to_num(sol.value, nan=1.0, posinf=0.0, neginf=0.0)

            p = find_root()

            I = get_inertial_number(p, dgamma_dt, self.d, self.rho_p)
            I = jnp.maximum(I, 1e-12)
            # I =
            alpha = 0.0001
            eta_E_s = p * self.mu_s / jnp.sqrt(dgamma_dt * dgamma_dt + alpha * alpha)

            mu_I_delta = (self.mu_d - self.mu_s) / (1.0 + self.I_0 / I)

            eta_delta = p * mu_I_delta / dgamma_dt

            eta = eta_E_s + eta_delta

            stress_next = -p * jnp.eye(3) + eta * deps_dev_dt
            return stress_next

        def stop():
            return stress_prev

        return jax.lax.cond(dgamma_dt < 1e-6, stop, flow)
        # return jax.lax.cond(dgamma_dt<1e-9,stop,flow )
        # return jax.lax.cond(dgamma_dt<1e-9,stop,flow )

    # @partial(jax.vmap, in_axes=(None, 0, 0), out_axes=(0))
    # def vmap_viscoplastic(self, strain_rate: chex.Array, volume_fraction: chex.Array):

    #     p = self.K*(volume_fraction-1.0)

    #     # I = get_I_phi(volume_fraction, self.phi_c, self.I_phi)

    #     # mu_I = get_mu_I(I, self.mu_s, self.mu_d, self.I_0)

    #     # eps_v_dt = get_volumetric_strain(strain_rate)
    #     # # volumetric_strain_rate = -jnp.trace(
    #     # #     strain_rate
    #     # # )  # compressive strain rate is positive

    #     # deviatoric_strain_rate = strain_rate + (
    #     #     1 / 3.0
    #     # ) * volumetric_strain_rate * jnp.eye(3)

    #     # dgamma_dt = jnp.sqrt(
    #     #     0.5 * (deviatoric_strain_rate @ deviatoric_strain_rate.T).trace()
    #     # )
    #     deps_dev_dt = get_dev_strain(strain_rate,dim =self.dim)

    #     dgamma_dt = get_scalar_shear_strain(dev_strain=deps_dev_dt,dim = self.dim)

    #     I = get_inertial_number(p,dgamma_dt,self.d, self.rho_p)

    #     # mu_I = get_mu_I_regularized_exp(
    #     #     I,
    #     #     self.mu_s,
    #     #     self.mu_d,
    #     #     self.I_0,
    #     #     0.0001,
    #     #     dgamma_dt
    #     #     )
    #     def flow():
    #         # p = get_pressure(dgamma_dt, I, self.d, self.rho_p)

    #         # Two assumptions made
    #         # (1) Drucker-Prager yield criterion with von-mises plastic potential and
    #         # (2) Alignment condition
    #         # J2 = mu_I *p

    #         # viscosity = J2 / dgamma_dt
    #         alpha = 0.001
    #         eta_E_s = p*self.mu_s/(dgamma_dt*dgamma_dt + alpha*alpha)

    #         eta_delta = p*((self.mu_d - self.mu_s) * (1.0 / (1.0 + self.I_0 / I)))/dgamma_dt

    #         viscosity = eta_E_s+eta_delta
    #         stress_next = -p * jnp.eye(3) + viscosity * deps_dev_dt

    #         return stress_next

    #     def stop():
    #         return jnp.zeros((3,3))
    #     # return jax.lax.cond(dgamma_dt<1e-12,flow,flow )
    #     return jax.lax.cond(dgamma_dt<1e-12,stop,flow )

    def get_p_ref(self, phi, dgamma_dt):
        I = get_I_phi(phi, self.phi_c, self.I_phi)

        return get_pressure(dgamma_dt, I, self.d, self.rho_p)
