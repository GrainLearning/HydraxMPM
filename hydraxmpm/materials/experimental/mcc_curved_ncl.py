from functools import partial
from typing import Tuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from typing_extensions import Self

from ...config.mpm_config import MPMConfig
from ...utils.math_helpers import (
    get_dev_strain,
    # get_hencky_strain_stack,
    get_hencky_strain,
    get_pressure,
    get_q_vm,
    get_volumetric_strain,
)
from ..material import Material


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


def plastic_potential(stress, p_c, M, dim=3):
    """Unified Hardening plastic potential function."""
    p = get_pressure(stress, dim=dim)
    q = get_q_vm(stress=stress, dim=dim)
    return yield_function(p, p_c, q, M)


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


class MCC_Curved_NCL(Material):
    nu: jnp.float32
    M: jnp.float32
    R: jnp.float32
    lam: jnp.float32
    kap: jnp.float32
    ln_N: jnp.float32
    ln_v_c: jnp.float32
    ps: jnp.float32
    eps_p_stack: chex.Array
    p_c_stack: chex.Array
    p_ref_stack: chex.Array
    phi_ref_stack: chex.Array
    ln_Z : jnp.float32

    def __init__(
        self: Self,
        config: MPMConfig,
        nu: jnp.float32,
        M: jnp.float32,
        R: jnp.float32,
        lam: jnp.float32,
        kap: jnp.float32,
        ln_N: jnp.float32,
        ln_v_c: jnp.float32=None,
        ln_Z:jnp.float32 = None,
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

        self.eps_p_stack = jnp.zeros((config.num_points, 3, 3))

        self.p_c_stack = p_ref_stack * R

        if ln_v_c is None:
            self.ps = jnp.exp((ln_N - ln_Z)/lam) -1.0
            self.ln_v_c = ln_Z + self.lam * jnp.log((1 + self.ps) / self.ps)
            self.ln_Z = ln_Z
        elif ln_Z is None:  
            self.ps = jnp.exp(ln_N / ln_v_c) * jnp.exp(1.0 / lam)
            self.ln_Z = ln_v_c - self.lam * jnp.log((1 + self.ps) / self.ps)
            self.ln_v_c = ln_v_c
        else:
            raise ValueError
        
        self.ln_N = ln_N
        
        # self.ln_v_c = ln_v_c

        if phi_ref_stack is None:
            self.phi_ref_stack = self.get_phi_ref(p_ref_stack)

    def get_phi_ref(self, p_ref_stack: chex.Array, dim=3):
        """Assuming we are on ncl."""

        @partial(jax.vmap)
        def vmap_get_ln_v_m(p):
            m = 0.0  # m = q/p = 0
            return (
                self.ln_Z
                - self.lam * jnp.log((p + self.ps) / (1 + self.ps))
                - (self.lam - self.kap) * jnp.log(1.0 + m**2 / self.M**2)
            )

        vmap_get_ln_v_m()
        ln_v_m_stack = vmap_get_ln_v_m(p_ref_stack)

        phi_stack = 1.0 / jnp.exp(ln_v_m_stack)

        return phi_stack

    def update(
        self: Self,
        stress_prev_stack: chex.Array,
        F_stack: chex.Array,
        L_stack: chex.Array,
        phi_stack: chex.Array,
    ) -> Tuple[chex.Array, Self]:
        return (stress_prev_stack, self)

    def update(
        self: Self,
        stress_prev_stack: chex.Array,
        F_stack: chex.Array,
        L_stack: chex.Array,
        phi_stack: chex.Array,
    ) -> Tuple[chex.Array, Self]:
        new_stress_stack, new_eps_p_stack = self.vmap_update_ip(
            F_stack, self.eps_p_stack, self.p_ref_stack, self.p_c_stack
        )

        new_self = eqx.tree_at(
            lambda state: (state.eps_p_stack),
            self,
            (new_eps_p_stack),
        )

        return (new_stress_stack, new_self)

    @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0), out_axes=(0, 0))
    def vmap_update_ip(
        self: Self,
        F: chex.Array,
        eps_p_prev: chex.Array,
        p_ref: jnp.float32,
        p_c_ref: jnp.float32,
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

        eps_p_v_prev = get_volumetric_strain(eps_p_prev)

        p_c_tr = get_non_linear_hardening_pressure(
            p_c_ref, self.ps, self.kap, self.lam, eps_p_v_prev
        )

        yf = yield_function(p_tr, p_c_tr, q_tr, self.M)

        def elastic_update():
            stress_next = s_tr - p_tr * jnp.eye(3)

            return stress_next, eps_p_prev

        def pull_to_ys():
            def residuals(sol, args):
                pmulti_curr, *deps_p_flat = sol

                deps_p_next = get_flattened_triu_3x3(deps_p_flat)

                eps_p_next = eps_p_prev + deps_p_next

                eps_e_next = eps - eps_p_next

                eps_e_v_next = get_volumetric_strain(eps_e_next)

                eps_p_v_next = get_volumetric_strain(eps_p_next)

                eps_e_d_next = get_dev_strain(eps_e_next, eps_e_v_next)

                p_next = get_elas_non_linear_pressure(
                    p_ref, self.ps, self.kap, eps_e_v_next
                )

                p_c_next = get_non_linear_hardening_pressure(
                    p_c_ref, self.ps, self.kap, self.lam, eps_p_v_next
                )

                K_next = get_K(self.kap, p_next, self.ps)

                G_next = get_G(self.nu, K_next)

                s_next = get_elas_dev_stress(eps_e_d_next, G_next)

                q_next = get_q_vm(dev_stress=s_next, dim=self.config.dim)

                yf_next = yield_function(p_next, p_c_next, q_next, self.M)

                stress_next = s_next - p_next * jnp.eye(3)

                flow_vector = jax.grad(plastic_potential, argnums=0)(
                    stress_next, p_c_next, self.M, self.config.dim
                )  # flow vector is associated with respect to current yield surface

                deps_p_fr = pmulti_curr * flow_vector

                deps_p_fr_flat = get_flattened_triu(deps_p_next - deps_p_fr)

                R = jnp.array([yf_next, *deps_p_fr_flat])

                R = R.at[0].set(R[0] / (K_tr * self.kap))

                aux = (p_next, s_next, eps_p_next)

                return R, aux

            def find_roots():
                """Find roots of the residuals function."""

                # 1st components plastic multiplier,
                # other 6 components of the plastic strain tensor
                init_val = jnp.zeros(7)

                solver = optx.Newton(rtol=1e-8, atol=1e-8)
                sol = optx.root_find(
                    residuals, solver, init_val, throw=False, has_aux=True, max_steps=20
                )
                return sol.value

            pmulti_curr, *deps_p_flat = jax.lax.stop_gradient(find_roots())

            R, aux = residuals([pmulti_curr, *deps_p_flat], None)

            p_next, s_next, eps_p_next = aux

            stress_next = s_next - p_next * jnp.eye(3)

            return stress_next, eps_p_next

        return jax.lax.cond(yf > 0, pull_to_ys, elastic_update)
