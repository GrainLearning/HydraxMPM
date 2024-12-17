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

from ...config.ip_config import IPConfig
from ...config.mpm_config import MPMConfig
from ...particles.particles import Particles
from ...utils.math_helpers import (
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
from ..material import Material


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
    # return jnp.maximum(p_hat, 1e-1)
    # return p_hat
    return jnp.nanmax(jnp.array([p_hat, 1e-1]))


def get_px_hat(px_hat_prev, cp, dH):
    """Compute non-linear pressure."""
    px_hat = px_hat_prev / (1.0 - (1 / cp) * dH)
    # return jnp.maximum(px_hat, 0.5e-1)\
    return jnp.nanmax(jnp.array([px_hat, 1e-1]))
    # return px_hat


def get_dH(p_hat, q, M, Mf, dgamma_p_d, deps_p_v, tol=0.5):
    "Regularization of UH https://www.sciencedirect.com/science/article/pii/S0266352X23007498"
    eta = q / p_hat
    eta = jnp.nanmax(jnp.array([eta, 0.0]))

    def limit():
        """Characteristic and steady state"""
        return ((Mf**4 - eta**4) / ((2 * eta) * (M**2 + eta**2))) * dgamma_p_d

    def valid():
        """solution with singularity at eta=M"""
        return (Mf**4 - eta**4) / (M**4 - eta**4) * deps_p_v

    return jax.lax.cond(jnp.abs(M - eta) < tol, limit, valid)


def get_px_hat_mcc(px_hat_prev, cp, deps_p_v):
    """Compute non-linear pressure."""
    px_hat = px_hat_prev / (1.0 - (1.0 / cp) * deps_p_v)
    return jnp.nanmax(jnp.array([px_hat, 1e-1]))
    # return px_hat


def get_s(deps_e_d, G, s_prev):
    return 2.0 * G * deps_e_d + s_prev


def get_K(kap, p_hat):
    return (1.0 / kap) * (p_hat)


def get_G(nu, K):
    return (3.0 * (1.0 - 2.0 * nu)) / (2.0 * (1.0 + nu)) * K


# def get_H(q, p_hat, Mf, M, deps_p_v, H_prev):
#     """Get hardening parameter"""
#     m = q / p_hat
#     curr = (Mf**4 - m**4) / (M**4 - m**4)
#     # return curr * deps_p_v + H_prev
#     # denominator may be zero due to numerical issues
#     return jnp.nan_to_num(curr, posinf=0.0) * deps_p_v + H_prev


def get_R(xi, cp):
    """Get overconsolidation parameter."""
    return jnp.exp(-xi / cp)


def get_Mf(M, R):
    """Get potential failure stress ratio from Hvorslev envelope."""
    term_sqrt = jnp.sqrt((12 * (3 - M) / M**2) * R + 1)
    return 6.0 / (term_sqrt + 1)


class UH(Material):
    """
    chi - yield surface parameter
    """

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
    px0_hat_stack: chex.Array
    px_hat_stack: chex.Array
    p0_hat_stack: chex.Array
    H_stack: chex.Array
    ln_v_c: chex.Array

    rho_p: jnp.float32

    def __init__(
        self: Self,
        config: Union[MPMConfig, IPConfig],
        nu: jnp.float32,
        M: jnp.float32,
        R: jnp.float32,
        lam: jnp.float32,
        kap: jnp.float32,
        p_t: jnp.float32 = 0.0,
        chi: jnp.float32 = 0,
        rho_p: jnp.float32 = None,
        ln_N: jnp.float32 = None,
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

        print("ln_v_c", self.ln_v_c)

        self.p0_hat_stack = p_ref_stack + p_t

        self.H_stack = jnp.zeros(config.num_points)

        self.eps_e_stack = jnp.zeros((config.num_points, 3, 3))

        self.px0_hat_stack = (p_ref_stack + p_t) * (1.0 / R)

        self.px_hat_stack = self.p0_hat_stack

    @classmethod
    def give_ref_phi(cls, p, R, ln_N, lam, kap, p_t):
        """Assume q =0, give reference solid volume fraction"""
        p_hat = p + p_t
        # ln_eta = ln_N - lam * jnp.log(p_hat / (1 + p_t))

        ln_eta = get_state_boundary_layer(p_hat, 0.0, 1, (lam - kap), lam, p_t, ln_N)
        ln_v = (lam - kap) * jnp.log(R) + ln_eta

        phi = 1.0 / jnp.exp(ln_v)

        jax.debug.print(
            "[give_ref_phi] ln_eta {} ln_v {} p_hat_prev {} phi {}",
            ln_eta,
            ln_v,
            p_hat,
            phi,
        )
        return phi

    @classmethod
    def give_ref_p(cls, phi, R, ln_N, lam, kap, p_t):
        ln_v = jnp.log(1.0 / phi)

        ln_eta = ln_v - (lam - kap) * jnp.log(R)

        p_hat = (1.0 + p_t) * jnp.exp((ln_N - ln_eta) / lam)

        jax.debug.print(
            "[give_ref_p] ln_eta {} ln_v {} p_hat_prev {} phi {}",
            ln_eta,
            ln_v,
            p_hat,
            phi,
        )

        ln_eta_calc = get_state_boundary_layer(
            p_hat, 0.0, 1, (lam - kap), lam, p_t, ln_N
        )

        jax.debug.print(
            "[give_ref_p] ln_eta_calc {} ln_v {} p_hat_prev {} phi {}",
            ln_eta_calc,
            ln_v,
            p_hat - p_t,
            phi,
        )
        return p_hat - p_t

    def update_from_particles(
        self: Self, particles: Particles
    ) -> Tuple[Particles, Self]:
        """Update the material state and particle stresses for MPM solver."""
        phi_stack = particles.get_solid_volume_fraction_stack(self.rho_p)

        deps_stack = get_sym_tensor_stack(particles.L_stack) * self.config.dt

        new_stress_stack, new_eps_e_stack, new_H_stack, new_px_hat_stack = (
            self.vmap_update_ip(
                deps_stack,
                self.eps_e_stack,
                particles.stress_stack,
                self.p0_hat_stack,
                self.px0_hat_stack,
                self.px_hat_stack,
                self.H_stack,
                phi_stack,
            )
        )

        new_self = eqx.tree_at(
            lambda state: (state.eps_e_stack, state.H_stack, state.px_hat_stack),
            self,
            (new_eps_e_stack, new_H_stack, new_px_hat_stack),
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

        new_stress_stack, new_eps_e_stack, new_H_stack, new_px_hat_stack = (
            self.vmap_update_ip(
                deps_stack,
                self.eps_e_stack,
                stress_prev_stack,
                self.p0_hat_stack,
                self.px0_hat_stack,
                self.px_hat_stack,
                self.H_stack,
                phi_stack,
            )
        )

        new_self = eqx.tree_at(
            lambda state: (state.eps_e_stack, state.H_stack, state.px_hat_stack),
            self,
            (new_eps_e_stack, new_H_stack, new_px_hat_stack),
        )

        return (new_stress_stack, new_self)

    @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0), out_axes=(0, 0, 0, 0))
    def vmap_update_ip(
        self: Self,
        deps_next: chex.Array,
        eps_e_prev: chex.Array,
        stress_prev: jnp.float32,
        p0_hat: jnp.float32,
        px0_hat: jnp.float32,
        px_hat_prev: jnp.float32,
        H_prev: jnp.float32,
        phi_curr: jnp.float32,
    ) -> Tuple[chex.Array, Self]:
        p_prev = get_pressure(stress_prev)

        ln_v = jnp.log(1.0 / phi_curr)

        p_hat_prev = p_prev + self.p_t

        s_prev = get_dev_stress(stress_prev, pressure=p_prev)

        deps_e_v_tr = get_volumetric_strain(deps_next)

        # eps_e_v_prev = get_volumetric_strain(eps_e_prev)

        # deps_e_v_tr = jnp.nanmax(jnp.array([deps_e_v_tr, 0.0 ]))

        p_hat_tr = get_p_hat(deps_e_v_tr, self.kap, p_hat_prev)

        deps_e_d_tr = get_dev_strain(deps_next, deps_e_v_tr)

        K_tr = get_K(self.kap, p_hat_tr)

        G_tr = get_G(self.nu, K_tr)

        s_tr = get_s(deps_e_d_tr, G_tr, s_prev)

        q_tr = get_q_vm(dev_stress=s_tr)

        yf = yield_function(p_hat_tr, px_hat_prev, q_tr, self.M)

        # phi = 1.0 / jnp.exp(ln_v)

        # phi_c = 1 / jnp.exp(self.ln_v_c)

        def elastic_update():
            stress_next = s_tr - (p_hat_tr - self.p_t) * jnp.eye(3)
            eps_e_tr = eps_e_prev + deps_next
            return stress_next, eps_e_tr, H_prev, px_hat_prev

        def pull_to_ys():
            def residuals(sol, args):
                pmulti, deps_p_v = sol

                # pmulti = eqx.error_if(pmulti, ln_v > self.ln_v_c, "ln_v > self.ln_v_c")

                pmulti = jnp.nanmax(jnp.array([pmulti, 0.0]))

                # deps_p_v = jnp.nanmax(jnp.array([deps_p_v, 0.0]))

                # deps_p_v = jnp.nanmin(jnp.array([deps_p_v, deps_e_v_tr]))

                # pmulti = eqx.error_if(
                #     pmulti, pmulti < 0.0, "pmulti negative"
                # )

                # deps_e =

                p_hat_next = get_p_hat(deps_e_v_tr - deps_p_v, self.kap, p_hat_prev)

                # p_hat_next = eqx.error_if(
                #     p_hat_next, p_hat_next < 0.0, "p_hat_next negative"
                # )
                K_next = get_K(self.kap, p_hat_next)

                G_next = get_G(self.nu, K_next)

                # G_next = eqx.error_if(G_next, G_next < 0.0, "G_next negative")

                # pmulti = eqx.error_if(pmulti, jnp.isnan(pmulti), "pmulti nan")
                # pmulti = eqx.error_if(G_next, jnp.isnan(G_next), "G_next nan")

                factor = 1 / (1 + 6.0 * G_next * pmulti)
                s_next = s_tr * factor

                q_next = q_tr * factor

                # ln_v_eta = get_state_boundary_layer(
                #     p_hat_next, q_next, self.M, self.cp, self.lam, self.p_t, self.ln_N
                # )

                # jnp.nan_to_num(ln_v_eta,nan=ln_v,posinf=ln_v,neginf=ln_v)

                # state_variable = ln_v_eta - ln_v

                # R_param = get_R(state_variable, self.cp)

                # Mf = get_Mf(self.M, R_param)

                # deps_e_d_next = 3*pmulti*s_next

                # dgamma_p_next = (4/ jnp.sqrt(3)) * get_scalar_shear_strain(
                #     dev_strain=deps_e_d_next
                #  )

                # dgamma_p_next = 2 * pmulti * q_next

                # dH = get_dH(p_hat_next, q_next, self.M, Mf, deps_p_v)
                # dH = get_dH2(p_hat_next, q_next, self.M, Mf, deps_p_v)
                # dH = get_dH(
                # p_hat_next, q_next, self.M, Mf, dgamma_p_next, deps_p_v, tol=1.0
                # )
                # jnp.maximum(dH,1e-10)

                # px_hat_next = get_px_hat(px_hat_prev, self.cp, dH)

                px_hat_next = get_px_hat_mcc(px_hat_prev, self.cp, deps_p_v)

                deps_v_p_fr = pmulti * (2.0 * p_hat_next - px_hat_next) * self.M**2

                yf_next = yield_function(p_hat_next, px_hat_next, q_next, self.M)

                R = jnp.array([yf_next, deps_v_p_fr - deps_p_v])

                # H_next = dH
                H_next = 0.0
                # R = R.at[0].set(R[0] / (K_tr * self.kap))
                # pmulti = eqx.error_if(
                #     ln, jnp.isnan(pmulti), "pmulti nan"
                # )
                # H_next = ln_v
                # jax.debug.print(
                #     "R {} state_variable {} p_prev {}", R, state_variable, p_prev
                # )
                aux = (p_hat_next, s_next, px_hat_next, H_next, G_next, K_next)

                return R, aux

            def find_roots():
                """Find roots of the residuals function."""

                # avoiding non-finite values

                init_val = jnp.array([0.0, 0.0])

                solver = optx.Newton(
                    rtol=1e-12,
                    atol=1e-12,
                )
                sol = optx.root_find(
                    residuals,
                    solver,
                    init_val,
                    throw=False,
                    has_aux=True,
                    max_steps=60,
                    # options={"lower":jnp.array([0.0, 0.0])}
                    # options=dict(lower=jnp.array([0.0, 0.0])),
                    # max_steps=5
                )
                return sol.value

            # def do_return_mapping():
            pmulti_curr, deps_p_v_next = jax.lax.stop_gradient(find_roots())
            R, aux = residuals([pmulti_curr, deps_p_v_next], None)

            # return aux, pmulti_curr, deps_p_v_next

            # aux, pmulti_curr, deps_p_v_next = jax.lax.cond(
            #     ln_v < self.ln_v_c,
            #     do_return_mapping,
            #     do_return_mapping,
            #     # lambda: (
            #     #     (
            #     #         p_hat_tr,  # p_hat_next
            #     #         jnp.zeros((3, 3)),  # s_next
            #     #         px_hat_prev,  # px_hat_next
            #     #         H_prev,  # H_next
            #     #         G_tr,  # G_next
            #     #         K_tr,  # K_next
            #     #     ),
            #     #     0.0,  # pmulti_curr
            #     #     0.0,  # deps_p_v_next
            #     # ),
            # )
            p_hat_next, s_next, px_hat_next, H_next, G_next, K_next = aux

            # jax.debug.print("{} {}",pmulti_curr,deps_p_v_next)

            # jax.debug.print(
            #     "ln_v {} phi {} ln_v_c {} phi_c {} p_hat_next {}",
            #     ln_v,
            #     phi,
            #     self.ln_v_c,
            #     phi_c,
            #     p_hat_next,
            # )
            # jax.debug.print("{}",state_variable)

            # p_hat_next = eqx.error_if(
            #     p_hat_next, jnp.isnan(p_hat_next).any(), "p_hat_next is nan"
            # )
            s_next = eqx.error_if(s_next, jnp.isnan(s_next).any(), "s_next is nan")

            stress_next = s_next - (p_hat_next - self.p_t) * jnp.eye(3)

            eps_e_v_next = (p_hat_next - p0_hat) / K_next

            # eps_e_v_next = jnp.maximum(eps_e_v_next, 0.0)

            eps_e_d_next = s_next / (2.0 * G_next)

            eps_e_next = eps_e_d_next - (1.0 / 3) * eps_e_v_next * jnp.eye(3)

            # stress_next = eqx.error_if(
            #     stress_next,
            #     jnp.isnan(stress_next).any(),
            #     " oops ln_v > self.ln_v_c",                # on_error="nan",
            # )
            # K = eqx.error_if(
            #     K, jnp.isnan(K).any(), "oops should not be going in this branch"
            # )
            # pmulti_curr = eqx.error_if(
            #     pmulti_curr, jnp.isnan(pmulti_curr).any(), "pmulti_curr is nan"
            # )
            # deps_p_v_next = eqx.error_if(
            #     deps_p_v_next, jnp.isnan(deps_p_v_next).any(), "deps_p_v_next is nan"
            # )
            # p_hat_next = eqx.error_if(
            #     p_hat_next, jnp.isnan(p_hat_next).any(), "p_hat_next is nan"
            # )

            # stress_next = eqx.error_if(
            #     stress_next, jnp.isnan(stress_next).any(), "stress_next is nan"
            # )

            # if jnp.isnan(stress_next).any():
            #     jax.debug.print(f"{ln_v} {self.ln_v_c}")

            return stress_next, eps_e_next, H_next, px_hat_next

        # optionally add (p_hat_stack > 0) ...
        #
        stress_next, eps_e_next, H_next, px_hat_next = jax.lax.cond(
            ln_v < self.ln_v_c,
            lambda: jax.lax.cond(yf > 0.0, pull_to_ys, elastic_update),
            lambda: (-self.p_t * jnp.eye(3), jnp.zeros((3, 3)), 0.0, self.p_t),
            # lambda: jax.lax.cond(yf > 0.0, pull_to_ys, elastic_update),
        )

        return stress_next, eps_e_next, H_next, px_hat_next

        # return (-self.p_t*jnp.eye(3), jnp.zeros((3, 3)), 0.0,self.p_t)
