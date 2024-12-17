from functools import partial
from operator import ne
from turtle import pos
from typing import Tuple, Union

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from typing_extensions import Self

from ..config.ip_config import IPConfig
from ..config.mpm_config import MPMConfig
from ..particles.particles import Particles

from ..utils.math_helpers import (
    # get_dev_strain,
    # get_hencky_strain_stack,
    get_dev_strain,
    get_sym_tensor_stack,
    get_hencky_strain,
    get_pressure,
    get_dev_stress,
    get_q_vm,
    get_volumetric_strain,
    get_scalar_shear_strain,
)
from .material import Material


def yield_function(p_hat, px_hat, q, M):
    return q * q / M * M + p_hat * p_hat - px_hat * p_hat


def get_state_boundary_layer(p_hat, q, M, cp, lam, p_t, ln_N):
    return (
        ln_N
        - lam * jnp.log(p_hat / (1.0 + p_t))
        - cp * jnp.log(1.0 + (q / p_hat) ** 2 / M**2)
    )


def get_p_hat(deps_e_v, kap, p_hat_prev):
    """Compute non-linear pressure."""
    p_hat = p_hat_prev / (1.0 - (1.0 / kap) * deps_e_v)
    return jnp.nanmax(jnp.array([p_hat, 1]))
    # return p_hat


def get_px_hat_mcc(px_hat_prev, cp, deps_p_v):
    """Compute non-linear pressure."""
    px_hat = px_hat_prev / (1.0 - (1.0 / cp) * deps_p_v)
    return jnp.nanmax(jnp.array([px_hat, 1]))
    # return px_hat


def get_s(deps_e_d, G, s_prev):
    return 2.0 * G * deps_e_d + s_prev


def get_K(kap, p_hat):
    return (1.0 / kap) * (p_hat)


def get_G(nu, K):
    return (3.0 * (1.0 - 2.0 * nu)) / (2.0 * (1.0 + nu)) * K


def give_phi_ref(p, R, ln_N, lam, kap, p_t):
    """Assume q =0, give reference solid volume fraction"""
    p_hat = p + p_t

    ln_eta = get_state_boundary_layer(p_hat, 0.0, 1, (lam - kap), lam, p_t, ln_N)
    ln_v = (lam - kap) * jnp.log(R) + ln_eta

    phi = 1.0 / jnp.exp(ln_v)

    return phi


def give_p_ref(phi, R, ln_N, lam, kap, p_t):
    ln_v = jnp.log(1.0 / phi)

    ln_eta = ln_v - (lam - kap) * jnp.log(R)

    p_hat = (1.0 + p_t) * jnp.exp((ln_N - ln_eta) / lam)

    return p_hat - p_t


class ModifiedCamClay(Material):
    nu: jnp.float32
    M: jnp.float32
    R: jnp.float32
    lam: jnp.float32
    kap: jnp.float32
    p_t: jnp.float32
    cp: jnp.float32
    chi: jnp.float32
    ln_N: jnp.float32

    eps_e_stack: chex.Array
    px_hat_stack: chex.Array
    p_ref_stack: chex.Array
    phi_ref_stack: chex.Array

    ln_v_c: chex.Array
    phi_c: chex.Array

    rho_p: jnp.float32

    def __init__(
        self: Self,
        config: Union[MPMConfig, IPConfig],
        nu: jnp.float32,
        M: jnp.float32,
        R: jnp.float32,
        lam: jnp.float32,
        kap: jnp.float32,
        ln_N: jnp.float32,
        p_t: jnp.float32 = 0.0,
        chi: jnp.float32 = 0,
        rho_p: jnp.float32 = None,
        p_ref_stack: chex.Array = None,
        phi_ref_stack: chex.Array = None,
    ) -> Self:
        self.config = config

        self.nu = nu

        self.M = M

        self.R = R

        self.lam = lam

        self.kap = kap

        self.cp = lam - kap

        self.rho_p = rho_p

        self.ln_N = ln_N

        self.p_t = p_t

        self.chi = chi

        if p_t > 0.0:
            self.ln_v_c = self.ln_N + self.lam * jnp.log((1 + self.p_t) / self.p_t)
        else:
            self.ln_v_c = self.ln_N

        self.phi_c = 1 / jnp.exp(self.ln_v_c)

        if p_ref_stack is None:
            vmap_give_p_ref = partial(
                jax.vmap, in_axes=(0, None, None, None, None, None)
            )(give_p_ref)
            p_ref_stack = vmap_give_p_ref(phi_ref_stack, R, ln_N, lam, kap, p_t)

        if phi_ref_stack is None:
            vmap_give_phi_ref = partial(
                jax.vmap, in_axes=(0, None, None, None, None, None)
            )(give_phi_ref)

            p_ref_stack = vmap_give_phi_ref(phi_ref_stack, R, ln_N, lam, kap, p_t)

        self.phi_ref_stack = phi_ref_stack

        self.p_ref_stack = p_ref_stack

        self.eps_e_stack = jnp.zeros((config.num_points, 3, 3))

        self.px_hat_stack = (p_ref_stack + p_t) * (1.0 / R)

    def update_from_particles(
        self: Self, particles: Particles
    ) -> Tuple[Particles, Self]:
        """Update the material state and particle stresses for MPM solver."""
        phi_stack = particles.get_solid_volume_fraction_stack(self.rho_p)

        deps_stack = get_sym_tensor_stack(particles.L_stack) * self.config.dt

        new_stress_stack, new_eps_e_stack, new_px_hat_stack = self.vmap_update_ip(
            deps_stack,
            self.eps_e_stack,
            particles.stress_stack,
            self.px_hat_stack,
            self.p_ref_stack,
            phi_stack,
        )

        new_self = eqx.tree_at(
            lambda state: (state.eps_e_stack, state.px_hat_stack),
            self,
            (new_eps_e_stack, new_px_hat_stack),
        )

        new_particles = eqx.tree_at(
            lambda state: (state.stress_stack),
            particles,
            (new_stress_stack),
        )

        return new_particles, new_self

    def update(
        self: Self,
        stress_prev_stack: chex.Array,
        F_stack: chex.Array,
        L_stack: chex.Array,
        phi_stack: chex.Array,
    ) -> Tuple[chex.Array, Self]:
        deps_stack = get_sym_tensor_stack(L_stack) * self.config.dt

        new_stress_stack, new_eps_e_stack, new_px_hat_stack = self.vmap_update_ip(
            deps_stack,
            self.eps_e_stack,
            stress_prev_stack,
            self.px_hat_stack,
            self.p_ref_stack,
            phi_stack,
        )

        new_self = eqx.tree_at(
            lambda state: (state.eps_e_stack, state.px_hat_stack),
            self,
            (new_eps_e_stack, new_px_hat_stack),
        )

        return (new_stress_stack, new_self)
        return (stress_prev_stack, self)

    @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0, 0), out_axes=(0, 0, 0))
    def vmap_update_ip(
        self: Self,
        deps_next: chex.Array,
        eps_e_prev: chex.Array,
        stress_prev: jnp.float32,
        px_hat_prev: jnp.float32,
        p_ref: jnp.float32,
        phi: jnp.float32,
    ) -> Tuple[chex.Array, Self]:
        p_prev = get_pressure(stress_prev)

        p_hat_prev = p_prev + self.p_t

        s_prev = get_dev_stress(stress_prev, pressure=p_prev)

        deps_e_v_tr = get_volumetric_strain(deps_next)

        p_hat_tr = get_p_hat(deps_e_v_tr, self.kap, p_hat_prev)

        deps_e_d_tr = get_dev_strain(deps_next, deps_e_v_tr)

        K_tr = get_K(self.kap, p_hat_tr)

        G_tr = get_G(self.nu, K_tr)

        s_tr = get_s(deps_e_d_tr, G_tr, s_prev)

        q_tr = get_q_vm(dev_stress=s_tr)

        yf = yield_function(p_hat_tr, px_hat_prev, q_tr, self.M)

        def elastic_update():
            stress_next = s_tr - (p_hat_tr - self.p_t) * jnp.eye(3)
            eps_e_tr = eps_e_prev + deps_next
            return stress_next, eps_e_tr, px_hat_prev

        def pull_to_ys():
            def residuals(sol, args):
                pmulti, deps_p_v = sol

                pmulti = jnp.nanmax(jnp.array([pmulti, 0.0]))

                p_hat_next = get_p_hat(deps_e_v_tr - deps_p_v, self.kap, p_hat_prev)

                K_next = get_K(self.kap, p_hat_next)

                G_next = get_G(self.nu, K_next)

                factor = 1 / (1 + 6.0 * G_next * pmulti)

                s_next = s_tr * factor

                q_next = q_tr * factor

                px_hat_next = get_px_hat_mcc(px_hat_prev, self.cp, deps_p_v)

                deps_v_p_fr = pmulti * (2.0 * p_hat_next - px_hat_next) * self.M**2

                yf_next = yield_function(p_hat_next, px_hat_next, q_next, self.M)

                R = jnp.array([yf_next, deps_v_p_fr - deps_p_v])

                aux = (p_hat_next, s_next, px_hat_next, G_next, K_next)

                return R, aux

            def find_roots():
                """Find roots of the residuals function."""

                # avoiding non-finite values

                init_val = jnp.array([0.0, 0.0])

                solver = optx.Newton(
                    rtol=1e-8,
                    atol=1e-8,
                )
                sol = optx.root_find(
                    residuals, solver, init_val, throw=False, has_aux=True, max_steps=10
                )
                return sol.value

            pmulti_curr, deps_p_v_next = jax.lax.stop_gradient(find_roots())
            R, aux = residuals([pmulti_curr, deps_p_v_next], None)

            p_hat_next, s_next, px_hat_next, G_next, K_next = aux

            s_next = eqx.error_if(s_next, jnp.isnan(s_next).any(), "s_next is nan")

            p_next = p_hat_next - self.p_t

            stress_next = s_next - (p_next) * jnp.eye(3)

            eps_e_v_next = (p_next - p_ref) / K_next

            eps_e_d_next = s_next / (2.0 * G_next)

            eps_e_next = eps_e_d_next - (1.0 / 3) * eps_e_v_next * jnp.eye(3)

            return stress_next, eps_e_next, px_hat_next

        stress_next, eps_e_next, px_hat_next = jax.lax.cond(
            (p_hat_tr >= 0.0),
            lambda: jax.lax.cond(yf > 0.0, pull_to_ys, elastic_update),
            lambda: (-self.p_t * jnp.eye(3), jnp.zeros((3, 3)), self.p_t),
        )

        return stress_next, eps_e_next, px_hat_next
