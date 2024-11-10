from functools import partial
from typing import Tuple, Union
from typing_extensions import Self

import chex
import jax
import jax.numpy as jnp
import optimistix as optx

from ...particles.particles import Particles
from ...utils.math_helpers import (
    get_dev_strain,
    get_dev_stress,
    get_pressure,
    get_pressure_stack,
    get_q_vm,
    get_sym_tensor_stack,
    get_volumetric_strain,
    # get_hencky_strain_stack,
    get_hencky_strain,
)
from ..common import get_timestep
from ..material import Material
from ...config.mpm_config import MPMConfig
from ...config.ip_config import IPConfig
import equinox as eqx


def plot_yield_surface(
    ax, p_range: Tuple, M: jnp.float32, p_c: jnp.float32, color="black", linestyle="--"
):
    p_stack = jnp.arange(p_range[0], p_range[1], p_range[2])

    def return_mapping(p):
        def solve_yf(sol, args):
            q = sol

            return yield_function(p, p_c, q, M)

        solver = optx.Newton(rtol=1e-6, atol=1e-6)
        sol = optx.root_find(solve_yf, solver, p, throw=False)
        return sol.value

    q_stack = jax.vmap(return_mapping)(p_stack)

    ax.plot(p_stack, q_stack, color=color, linestyle=linestyle)
    return ax


def get_elas_non_linear_pressure(p_ref, ps, kap, eps_e_v):
    return (p_ref + ps) * jnp.exp(eps_e_v / kap) - ps


def get_non_linear_hardening_pressure(pc0, ps, kap, lam, eps_p_v):
    return (pc0 + ps) * jnp.exp(eps_p_v / (lam - kap)) - ps


def get_elas_dev_stress(eps_e_d, G):
    return 2.0 * G * eps_e_d


def yield_function(p, p_c, q, M):
    """Compute the modified Cam Clay yield function."""
    return q**2 / M**2 - p * (p_c - p)


def yield_function(p, q, H, cp, pc0, M, chi, ps):
    term1 = jnp.log(
        (1.0 + ((1.0 + chi) * q * q) / (M * M * p * p - chi * q * q)) * p + ps
    )

    term2 = -jnp.log(pc0 + ps)

    term3 = -H / cp

    return term1 + term2 + term3


def plastic_potential(stress, H, cp, pc0, M_c):
    """Unified Hardening plastic potential function."""
    q = get_q_vm(stress)
    p = get_pressure(stress)
    return jnp.log(p / pc0) + jnp.log(1.0 + (q**2) / (p**2 * M_c**2)) - H / cp


def get_flattened_triu_3x3(vals):
    """Convert flattened upper triangular values to 3x3 symmetric tensor."""
    new = jnp.zeros((3, 3))
    inds = jnp.triu_indices_from(new)
    new = new.at[inds].set(vals)
    new = new.at[inds[1], inds[0]].set(vals)
    return new


def get_flattened_triu(A):
    """Get flattened upper triangular components of a 3x3 symmetric tensor."""
    return A.at[jnp.triu_indices(A.shape[0])].get()


def get_K(kap, p, ps):
    return (1.0 / kap) * (p + ps)


def get_G(nu, K):
    return (3.0 * (1.0 - 2.0 * nu)) / (2.0 * (1.0 + nu)) * K


def get_H(q, p, Mf, M, deps_p_v, H_prev):
    """Get hardening parameter"""
    m = q / p
    curr = (Mf**4 - m**4) / (M**4 - m**4)
    # denominator may be zero due to numerical issues
    return jnp.nan_to_num(curr, posinf=0.0) * deps_p_v + H_prev


def get_R(xi, cp):
    """Get overconsolidation parameter."""
    return jnp.exp(-xi / cp)


def get_Mf(M, R):
    """Get potential failure stress ratio from Hvorslev envelope."""
    term_sqrt = jnp.sqrt((12 * (3 - M) / M**2) * R + 1)
    return 6.0 / (term_sqrt + 1)


def get_Mc(M, m_par, xi):
    """Characteristic state stress ratio"""
    return M * jnp.exp(-m_par * xi)


def get_xi(ln_v_m, ln_v):
    """Get state variable specific volume on NCL - current specific volume."""
    return ln_v_m - ln_v


def get_ln_v_m(p, q, ln_Z, lam, kap, M, chi, ps):
    """Distance from ACL to NCL in natural log specific volume space / pressure space."""

    m_2 = (q / p) ** 2
    M_2 = M * M

    term_2 = -lam * jnp.log((p + ps) / (1.0 + ps))

    term_3 = -(lam - kap) * jnp.log(
        (((M_2 + m_2) / (M_2 - chi * m_2)) * p + ps) / (p + ps)
    )

    return ln_Z + term_2 + term_3


class CSUH(Material):
    nu: jnp.float32
    M: jnp.float32
    R: jnp.float32
    lam: jnp.float32
    kap: jnp.float32
    ln_N: jnp.float32
    ln_v_c: jnp.float32
    ln_Z: jnp.float32
    ps: jnp.float32

    m_par: jnp.float32
    chi: jnp.float32
    cp: jnp.float32

    eps_p_stack: chex.Array
    p_c_ref_stack: chex.Array
    p_ref_stack: chex.Array
    phi_ref_stack: chex.Array
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
        ln_N: jnp.float32,
        m_par: jnp.float32,
        chi: jnp.float32,
        rho_p: jnp.float32 = None,
        ln_v_c: jnp.float32 = None,
        ln_Z: jnp.float32 = None,
        p_ref_stack: chex.Array = None,
        phi_ref_stack: chex.Array = None,
    ) -> Self:
        self.config = config

        self.p_ref_stack = p_ref_stack

        self.nu = nu
        self.M = M
        self.R = R
        self.lam = lam
        self.kap = kap
        self.m_par = m_par
        self.chi = chi

        self.H_stack = jnp.zeros(config.num_points)

        self.eps_p_stack = jnp.zeros((config.num_points, 3, 3))

        self.ln_N = ln_N
        self.ln_v_c = ln_v_c

        self.cp = lam - kap

        self.rho_p = rho_p

        if ln_v_c is None:
            self.ps = jnp.exp((ln_N - ln_Z) / lam) - 1.0
            self.ln_v_c = ln_Z + self.lam * jnp.log((1 + self.ps) / self.ps)
            self.ln_Z = ln_Z

        elif ln_Z is None:
            self.ps = jnp.exp((ln_N - ln_v_c) / lam)
            self.ln_Z = ln_v_c - self.lam * jnp.log((1 + self.ps) / self.ps)
            self.ln_v_c = ln_v_c
        else:
            raise ValueError

        if phi_ref_stack is None:
            self.phi_ref_stack = self.get_phi_ref(p_ref_stack)

        self.p_c_ref_stack = p_ref_stack

    def get_phi_ref(self, p_ref_stack: chex.Array, dim=3):
        """Assuming we are on ncl."""

        # works only for special case when q=0,
        # not implemented for other cases
        ln_v_m_stack = jax.vmap(
            get_ln_v_m,
            in_axes=(0, None, None, None, None, None, None, None),
        )(
            p_ref_stack,
            0,
            self.ln_Z,
            self.lam,
            self.kap,
            self.M,
            self.chi,
            self.ps,
        )

        xi = -self.cp * jnp.log(self.R)

        ln_v_stack = ln_v_m_stack - xi

        phi_stack = 1.0 / jnp.exp(ln_v_stack)

        # print("ln_v_stack",ln_v_stack, self.ln_v_c)
        # print("phi_stack",phi_stack)
        return phi_stack

    def update_from_particles(
        self: Self, particles: Particles
    ) -> Tuple[Particles, Self]:
        """Update the material state and particle stresses for MPM solver."""
        phi_stack = particles.get_solid_volume_fraction_stack(self.rho_p)

        new_stress_stack, new_eps_p_stack, new_H_stack = self.vmap_update_ip(
            particles.F_stack,
            self.eps_p_stack,
            self.p_ref_stack,
            self.p_c_ref_stack,
            self.H_stack,
            phi_stack,
        )

        jax.debug.print("{}", new_stress_stack)
        new_particles = eqx.tree_at(
            lambda state: (state.stress_stack),
            particles,
            (new_stress_stack),
        )
        new_self = eqx.tree_at(
            lambda state: (state.eps_p_stack, state.H_stack),
            self,
            (new_eps_p_stack, new_H_stack),
        )

        return new_particles, new_self

    def update(
        self: Self,
        stress_prev_stack: chex.Array,
        F_stack: chex.Array,
        L_stack: chex.Array,
        phi_stack: chex.Array,
    ) -> Tuple[chex.Array, Self]:
        new_stress_stack, new_eps_p_stack, new_H_stack = self.vmap_update_ip(
            F_stack,
            self.eps_p_stack,
            self.p_ref_stack,
            self.p_c_ref_stack,
            self.H_stack,
            phi_stack,
        )
        # jax.debug.print("{}",new_H_stack)
        new_self = eqx.tree_at(
            lambda state: (state.eps_p_stack, state.H_stack),
            self,
            (new_eps_p_stack, new_H_stack),
        )

        return (new_stress_stack, new_self)

    @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0, 0), out_axes=(0, 0, 0))
    def vmap_update_ip(
        self: Self,
        F: chex.Array,
        eps_p_prev: chex.Array,
        p_ref: jnp.float32,
        p_c_ref: jnp.float32,
        H_prev: jnp.float32,
        phi_curr: jnp.float32,
    ) -> Tuple[chex.Array, Self]:
        eps, u, vh = get_hencky_strain(F)

        eps_e_tr = eps - eps_p_prev

        eps_e_v_tr = get_volumetric_strain(eps_e_tr)

        eps_e_d_tr = get_dev_strain(eps_e_tr, eps_e_v_tr)

        p_tr = get_elas_non_linear_pressure(p_ref, self.ps, self.kap, eps_e_v_tr)

        K_tr = get_K(self.kap, p_tr, self.ps)

        G_tr = get_G(self.nu, K_tr)

        s_tr = get_elas_dev_stress(eps_e_d_tr, G_tr)

        q_tr = get_q_vm(dev_stress=s_tr, dim=self.config.dim)

        # eps_p_v_prev = get_volumetric_strain(eps_p_prev)

        # p_c_tr = get_non_linear_hardening_pressure(
        #     p_c_ref, self.ps, self.kap, self.lam, eps_p_v_prev
        # )

        H_tr = H_prev

        yf = yield_function(
            p_tr, q_tr, H_tr, self.cp, p_c_ref, self.M, self.chi, self.ps
        )

        def elastic_update():
            stress_next = s_tr - p_tr * jnp.eye(3)

            return stress_next, eps_p_prev, H_prev

        def pull_to_ys():
            def residuals(sol, args):
                pmulti_curr, *deps_p_flat = sol

                deps_p_next = get_flattened_triu_3x3(deps_p_flat)

                deps_p_v_next = get_volumetric_strain(deps_p_next)

                eps_p_next = eps_p_prev + deps_p_next

                eps_e_next = eps - eps_p_next

                eps_e_v_next = get_volumetric_strain(eps_e_next)

                # eps_p_v_next = get_volumetric_strain(eps_p_next)

                eps_e_d_next = get_dev_strain(eps_e_next, eps_e_v_next)

                p_next = get_elas_non_linear_pressure(
                    p_ref, self.ps, self.kap, eps_e_v_next
                )

                # p_c_next = get_non_linear_hardening_pressure(
                #     p_c_ref, self.ps, self.kap, self.lam, eps_p_v_next
                # )

                K_next = get_K(self.kap, p_next, self.ps)

                G_next = get_G(self.nu, K_next)

                s_next = get_elas_dev_stress(eps_e_d_next, G_next)

                q_next = get_q_vm(dev_stress=s_next, dim=self.config.dim)

                ln_v_m = get_ln_v_m(
                    p_next,
                    q_next,
                    self.ln_Z,
                    self.lam,
                    self.kap,
                    self.M,
                    self.chi,
                    self.ps,
                )

                ln_v = jnp.log(1.0 / phi_curr)
                xi = get_xi(ln_v_m, ln_v)

                R_param = get_R(xi, self.cp)

                Mf = get_Mf(self.M, R_param)

                H_next = get_H(
                    q_next, p_next, Mf, self.M, deps_p_v_next, H_prev
                )  # integrate hardening parameter

                yf_next = yield_function(
                    p_next, q_next, H_next, self.cp, p_c_ref, self.M, self.chi, self.ps
                )

                M_c = get_Mc(self.M, self.m_par, xi)
                # yf_next = yield_function(p_next, p_c_next, q_next, self.M)

                stress_next = s_next - p_next * jnp.eye(3)

                flow_vector = jax.grad(plastic_potential, argnums=0)(
                    stress_next, H_next, self.cp, p_c_ref, M_c
                )  # flow vector is non associated

                deps_p_fr = pmulti_curr * flow_vector

                deps_p_fr_flat = get_flattened_triu(deps_p_next - deps_p_fr)

                R = jnp.array([yf_next, *deps_p_fr_flat])

                R = R.at[0].set(R[0] / (K_tr * self.kap))

                aux = (p_next, s_next, eps_p_next, H_next)

                return R, aux

            def find_roots():
                """Find roots of the residuals function."""

                # 1st components plastic multiplier,
                # other 6 components of the plastic strain tensor
                init_val = jnp.zeros(7)

                solver = optx.Newton(rtol=1e-8, atol=1e-8)
                sol = optx.root_find(
                    residuals,
                    solver,
                    init_val,
                    throw=False,
                    has_aux=True,
                    # max_steps=20
                    max_steps=5,
                )
                return sol.value

            pmulti_curr, *deps_p_flat = jax.lax.stop_gradient(find_roots())

            R, aux = residuals([pmulti_curr, *deps_p_flat], None)

            p_next, s_next, eps_p_next, H_next = aux

            stress_next = s_next - p_next * jnp.eye(3)

            return stress_next, eps_p_next, H_next

        return jax.lax.cond(yf > 0, pull_to_ys, elastic_update)
