from functools import partial
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
)
from ..material import Material


# def plot_yield_surface2(
#     ax,
#     p_range: Tuple,
#     M: jnp.float32,
#     px0: jnp.float32,
#     pt: jnp.float32,
#     ps: jnp.float32,
#     chi: jnp.float32,
#     color="black",
#     linestyle="--",
# ):
#     p_stack = jnp.arange(p_range[0], p_range[1], p_range[2])
#     print(p_stack)

#     def return_mapping(p):
#         def solve_yf(sol, args):
#             q = sol
#             print(q)
#             return yield_function2(p, pt, px0, ps, q, M, chi)

#         solver = optx.Newton(rtol=1e-6, atol=1e-6)
#         sol = optx.root_find(solve_yf, solver, p, throw=False)
#         return sol.value

#     q_stack = jax.vmap(return_mapping)(p_stack)

#     ax.plot(p_stack, q_stack, color=color, linestyle=linestyle)
#     return ax


# def yield_function(p_hat, px0_hat, q, M, cp, H=0.0):
#     """Split is useful if we want to use one form for state boundary layer"""
#     return jnp.log(p_hat / px0_hat) + jnp.log(1.0 + ((q / p_hat) ** 2) / M**2) - H / cp


# def yield_function(p_hat, px0_hat, q, M, cp, chi, p_s, H=0.0):
#     inner = (
#         1.0 + ((1.0 + chi) * q * q) / (M * M * p_hat * p_hat - chi * q * q)
#     ) * p_hat + p_s
#     return jnp.log(inner) - jnp.log(px0_hat + p_s) - (1.0 / cp) * H


def yield_function(p_hat, px0_hat, q, M, cp, H=0.0):
    px_hat = px0_hat * jnp.exp(H / cp)
    return q * q / M * M + p_hat * p_hat - px_hat * p_hat

def yield_function_log(p_hat, px0_hat, q, M, cp, H=0.0):
    """Split is useful if we want to use one form for state boundary layer"""
    return jnp.log(p_hat / px0_hat) + jnp.log(1.0 + ((q / p_hat) ** 2) / M**2) - H / cp


def get_p_hat(deps_e_v, kap, p_hat_prev):
    """Compute non-linear pressure."""
    p_hat = p_hat_prev / (1.0 - (1.0 / kap) * deps_e_v)
    return jnp.maximum(p_hat, 1e-1)


def get_px_hat(deps_p_v, cp, px_hat_prev):
    """Compute non-linear pressure."""
    px_hat = px_hat_prev / (1.0 - (1.0 / cp) * deps_p_v)
    return jnp.maximum(px_hat, 1e-1)


def get_s(deps_e_d, G, s_prev):
    return 2.0 * G * deps_e_d + s_prev


def get_K(kap, p_hat):
    return (1.0 / kap) * (p_hat)


def get_G(nu, K):
    return (3.0 * (1.0 - 2.0 * nu)) / (2.0 * (1.0 + nu)) * K


def get_H(q, p_hat, Mf, M, deps_p_v, H_prev):
    """Get hardening parameter"""
    m = q / p_hat
    curr = (Mf**4 - m**4) / (M**4 - m**4)
    # return curr * deps_p_v + H_prev
    # denominator may be zero due to numerical issues
    return jnp.nan_to_num(curr, posinf=0.0) * deps_p_v + H_prev


def get_R(xi, cp):
    """Get overconsolidation parameter."""
    return jnp.exp(-xi / cp)


def get_Mf(M, R):
    """Get potential failure stress ratio from Hvorslev envelope."""
    term_sqrt = jnp.sqrt((12 * (3 - M) / M**2) * R + 1)
    return 6.0 / (term_sqrt + 1)


# def get_xi(ln_v_m, ln_v):
#     """Get state variable specific volume on NCL - current specific volume."""
#     return ln_v_m - ln_v


# def get_ln_v_m(p_hat, q, ln_N, lam, kap, M, p_t):
#     eta_2 = (q / p_hat) ** 2

#     return (
#         ln_N
#         - lam * jnp.log(p_hat / (1 + p_t))
#         - (lam - kap) * jnp.log(1.0 + eta_2 / M * M)
#     )


# def get_Mc(M, m_par, xi):
#     """Characteristic state stress ratio"""
#     return M * jnp.exp(-m_par * xi)

# def get_ln_v_m(p_hat, q, ln_Z, lam, kap, M, chi, ps):
#     """Distance from ACL to NCL in natural log specific volume space / pressure space."""

#     eta = (q / p_hat) ** 2
#     M_2 = M * M

#     term_2 = -lam * jnp.log((p + ps) / (1.0 + ps))

#     term_3 = -(lam - kap) * jnp.log(
#         (((M_2 + m_2) / (M_2 - chi * m_2)) * p + ps) / (p + ps)
#     )

#     return ln_Z + term_2 + term_3


def get_specific_volume_ncl(p_hat, ln_N, lam, p_t):
    return ln_N - lam * jnp.log(p_hat / (1.0 + p_t))


# def yield_surface(p, q, H, cp, px0, M, chi, ps):
#     q_p_2 = (q / p) ** 2

#     return H - cp * jnp.log(
#         (((M * M + q_p_2) / (M * M - chi * q_p_2)) * p + ps) / (px0 + ps)
#     )


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

    rho_p: jnp.float32

    def __init__(
        self: Self,
        config: Union[MPMConfig, IPConfig],
        nu: jnp.float32,
        M: jnp.float32,
        R: jnp.float32,
        lam: jnp.float32,
        kap: jnp.float32,
        p_t: jnp.float32,
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

        self.p0_hat_stack = p_ref_stack + p_t

        a = get_specific_volume_ncl(self.p0_hat_stack, ln_N, lam, p_t)

        print(f"{1/jnp.exp(a)=}")

        self.H_stack = jnp.zeros(config.num_points)

        self.eps_e_stack = jnp.zeros((config.num_points, 3, 3))

        # R = current pressure / reference pressure
        # self.px0_hat_stack = (p_ref_stack + p_t) * R
        self.px0_hat_stack = self.p0_hat_stack
        self.px_hat_stack = self.p0_hat_stack

    @classmethod
    def give_ref_phi(p, R, ln_N, lam, kap, p_t):
        """Assume q =0, give reference solid volume fraction"""
        p_hat = p + p_t
        ln_eta = ln_N - lam * jnp.log(p_hat / (1 + p_t))
        ln_v = (lam - kap) * jnp.log(R) + ln_eta

        phi = 1.0 / jnp.exp(ln_v)
        return phi

    @classmethod
    def give_ref_p(cls, phi, R, ln_N, lam, kap, p_t):
        ln_v = jnp.log(1.0 / phi)
        ln_eta = ln_v - (lam - kap) * jnp.log(R)

        p_hat = (1.0 + p_t) * jnp.exp((ln_N - ln_eta) / lam)
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

        p_hat_prev = p_prev + self.p_t

        s_prev = get_dev_stress(stress_prev, pressure=p_prev)

        deps_e_v_tr = get_volumetric_strain(deps_next)

        p_hat_tr = get_p_hat(deps_e_v_tr, self.kap, p_hat_prev)

        deps_e_d_tr = get_dev_strain(deps_next, deps_e_v_tr)

        K_tr = get_K(self.kap, p_hat_tr)

        G_tr = get_G(self.nu, K_tr)

        s_tr = get_s(deps_e_d_tr, G_tr, s_prev)

        q_tr = get_q_vm(dev_stress=s_tr)

        yf = yield_function(p_hat_tr, px0_hat, q_tr, self.M, self.cp, H_prev)

        # yf = yield_function(p_hat_tr, px_hat_prev, q_tr, self.M, self.cp)

        def elastic_update():
            stress_next = s_tr - (p_hat_tr - self.p_t) * jnp.eye(3)
            eps_e_tr = eps_e_prev + deps_next
            return stress_next, eps_e_tr, H_prev, px_hat_prev

        def pull_to_ys():
            def residuals(sol, args):
                pmulti, deps_p_v = sol

                p_hat_next = get_p_hat(deps_e_v_tr - deps_p_v, self.kap, p_hat_prev)

                K_next = get_K(self.kap, p_hat_next)

                G_next = get_G(self.nu, K_next)

                factor = (self.M**2) / (self.M**2 + 6.0 * G_next * pmulti)

                s_next = s_tr * factor

                q_next = q_tr * factor

                yf0 = yield_function_log(
                    p_hat_next,
                    p_hat_next,
                    q_next,
                    self.M,
                    self.cp,
                )
                                

                ln_v_ncl = get_specific_volume_ncl(
                    p_hat_next, self.ln_N, self.lam, self.p_t
                )

                ln_v_eta = ln_v_ncl - self.cp * yf0

                ln_v = jnp.log(1.0 / phi_curr)

                state_variable = ln_v_eta - ln_v

                R_param = get_R(state_variable, self.cp)

                Mf = get_Mf(self.M, R_param)

                H_next = get_H(
                    q_next, p_hat_next, Mf, self.M, deps_p_v, H_prev
                )  # integrate hardening parameter


                px_hat_next = get_px_hat(deps_p_v, self.cp, px_hat_prev)

                deps_v_p_fr = pmulti * (2.0 * p_hat_next - px_hat_next)

                yf_current = yield_function_log(
                    p_hat_next, px0_hat, q_next, self.M, self.cp, H_next
                )
                # yf_ref = yield_function(
                #     p_hat_next, px_hat_next, q_next, self.M, self.cp
                # )


                R = jnp.array([yf_current, deps_v_p_fr - deps_p_v])

                # R = R.at[0].set(R[0] / (K_tr * self.kap))
                # H_next = 0.0

                aux = (p_hat_next, s_next, px_hat_next, H_next, G_next, K_next)

                return R, aux

            def find_roots():
                """Find roots of the residuals function."""

                # avoiding non-finite values

                init_val = jnp.array([0.0, 0.0])

                solver = optx.Newton(rtol=1e-8, atol=1e-8)
                sol = optx.root_find(
                    residuals,
                    solver,
                    init_val,
                    throw=False,
                    has_aux=True,
                    max_steps=40,
                    # max_steps=5,
                )
                return sol.value

            pmulti_curr, deps_p_v_next = jax.lax.stop_gradient(find_roots())

            # jax.debug.print("{} {}",pmulti_curr,deps_p_v_next)
            R, aux = residuals([pmulti_curr, deps_p_v_next], None)

            p_hat_next, s_next, px_hat_next, H_next, G_next, K_next = aux

            stress_next = s_next - (p_hat_next - self.p_t) * jnp.eye(3)

            eps_e_v_next = (p_hat_next - p0_hat) / K_next
            # eps_e_v_next = self.kap * jnp.log(
            # (p_hat_next + self.p_s) / (p0_hat + self.p_s)
            # )

            eps_e_d_next = s_next / (2.0 * G_next)

            eps_e_next = eps_e_d_next - (1.0 / 3) * eps_e_v_next * jnp.eye(3)

            return stress_next, eps_e_next, H_next, px_hat_next

        return jax.lax.cond(yf > 0.0, pull_to_ys, elastic_update)
