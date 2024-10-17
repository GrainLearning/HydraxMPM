"""Modified Cam Clay regularized"""

from functools import partial
from typing import Tuple
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


def get_non_linear_hardening_pressure(pc0, ps, kap, lam, eps_e_v):
    return (pc0 + ps) * jnp.exp(eps_e_v / (lam - kap)) - ps


def get_elas_dev_stress(eps_e_d, s_ref, G):
    return 2.0 * G * eps_e_d + s_ref


def yield_function(p, p_c, q, M):
    """Compute the modified Cam Clay yield function."""
    return q**2 / M**2 - p * (p_c - p)


def get_K(kap, p, ps):
    return (1.0 / kap) * (p + ps)


def get_G(nu, K):
    return (3.0 * (1.0 - 2.0 * nu)) / (2.0 * (1.0 + nu)) * K


@chex.dataclass
class ModifiedCamClayReg(Material):

    nu: jnp.float32
    M: jnp.float32
    R: jnp.float32
    lam: jnp.float32
    kap: jnp.float32
    ln_N: jnp.float32
    ln_v_c: jnp.float32
    ps: jnp.float32
    dim: jnp.int16
    eps_p_v_stack: chex.Array
    p_c_stack: chex.Array
    F_p_stack: chex.Array
    stress_ref_stack: chex.Array

    @classmethod
    def create(
        cls: Self,
        nu: jnp.float32,
        M: jnp.float32,
        R: jnp.float32,
        lam: jnp.float32,
        kap: jnp.float32,
        ln_N: jnp.float32,
        ln_v_c: jnp.float32,
        stress_ref_stack: chex.Array = None,
        absolute_density: jnp.float32 = 1.0,
        dim: jnp.int16 = 3,
    ) -> Self:
        # Check if kappa less than lambda

        num_particles = stress_ref_stack.shape[0]
        eps_e_stack = jnp.zeros((num_particles, 3, 3))
        eps_p_v_stack = jnp.zeros(num_particles)

        if stress_ref_stack is None:
            stress_ref_stack = jnp.zeros((num_particles, 3, 3), dtype=jnp.float32)

        p_ref_stack = get_pressure_stack(stress_ref_stack, dim)

        p_c_stack = p_ref_stack * R

        ps = jnp.exp(ln_N / ln_v_c) * jnp.exp(1.0 / lam)
        # ps = 0
        # print(ps)
        return cls(
            stress_ref_stack=stress_ref_stack,
            eps_e_stack=eps_e_stack,
            p_c_stack=p_c_stack,
            nu=nu,
            M=M,
            R=R,
            lam=lam,
            kap=kap,
            ps=ps,
            ln_v_c=ln_v_c,
            ln_N=ln_N,
            absolute_density=absolute_density,
            eps_p_v_stack=eps_p_v_stack,
            dim=dim,
        )

    def get_phi_ref(self, stress_ref: chex.Array, dim=3):
        """Assuming we are on ncl."""
        ln_Z = self.ln_v_c - self.lam * jnp.log((1 + self.ps) / self.ps)

        @partial(jax.vmap)
        def vmap_get_ln_v_m(p):
            m = 0.0  # m = q/p = 0
            return (
                ln_Z
                - self.lam * jnp.log((p + self.ps) / (1 + self.ps))
                - (self.lam - self.kap) * jnp.log(1.0 + m**2 / self.M**2)
            )

        p_stack = get_pressure_stack(stress_ref, dim)

        ln_v_m_stack = vmap_get_ln_v_m(p_stack)

        phi_stack = 1.0 / jnp.exp(ln_v_m_stack)
        return phi_stack

    def update(
        self: Self,
        stress_prev_stack: chex.Array,
        F_stack: chex.Array,
        L_stack: chex.Array,
        phi_stack: chex.Array,
        dt: jnp.float32,
    ) -> Tuple[chex.Array, Self]:

        # eps_stack, *_ = get_hencky_strain_stack(F_stack)

        # jax.debug.print("{}", eps_stack)
        # deps_stack = get_sym_tensor_stack(L_stack) * dt
        stress_next_stack, eps_e_next_stack, p_c_next_stack = self.vmap_update_stress(
            F_stack,
            F_p_stack,
            self.stress_ref_stack,
            self.eps_e_stack,
            self.p_c_stack,
            eps_p_v_stack,
        )

        return (
            stress_prev_stack,
            self,
            # stress_next_stack,
            # self.replace(eps_e_stack=eps_e_next_stack, p_c_stack=p_c_next_stack),
        )

    @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0, 0), out_axes=(0, 0, 0))
    def vmap_update_stress(self, F, F_p, stress_ref, eps_e_prev, p_c_ref, eps_p_v_prev):
        # dim = self.dim
        F_e_tr = F @ jnp.linalg.inv(F_p)

        eps_e_tr, u, vh = get_hencky_strain(F_e_tr)

        # Reference pressure and deviatoric stress
        p_ref = get_pressure(stress_ref, self.dim)

        s_ref = get_dev_stress(stress_ref, p_ref, self.dim)

        eps_e_v_tr = get_volumetric_strain(eps_e_tr)

        eps_e_d_tr = get_dev_strain(eps_e_tr, eps_e_v_tr)

        p_tr = get_elas_non_linear_pressure(p_ref, self.ps, self.kap, eps_e_v_tr)

        K_tr = get_K(self.kap, p_tr, self.ps)

        G_tr = get_G(self.nu, K_tr)

        s_tr = get_elas_dev_stress(eps_e_d_tr, s_ref, G_tr)

        q_tr = get_q_vm(dev_stress=s_tr)

        # p_c_tr = p_c_prev
        p_c_tr = get_non_linear_hardening_pressure(
            p_c_ref, self.ps, self.kap, self.lam, eps_p_v_prev
        )

        yf = yield_function(p_tr, p_c_tr, q_tr, self.M)

    #     is_ep = yf > 0

    #     def elastic_update():

    #         stress_next = s_tr - p_tr * jnp.eye(3)

    #         return stress_next, eps_e_tr, p_c_prev

    #     def pull_to_ys():
    #         stress_next = s_tr - p_tr * jnp.eye(3)

    #         def residuals(sol, args):
    #             pmulti, deps_p_v = sol

    #             p_next = get_elas_non_linear_pressure(
    #                 deps_e_v - deps_p_v, self.kap, p_prev
    #             )

    #             K_next = get_K(self.kap, p_next)

    #             G_next = get_G(self.nu, K_next)

    #             s_next = (self.M**2 / (self.M**2 + 6.0 * G_next * pmulti)) * s_tr

    #             q_next = (self.M**2 / (self.M**2 + 6.0 * G_next * pmulti)) * q_tr

    #             p_c_next = get_non_linear_hardening_pressure(
    #                 deps_p_v,
    #                 self.lam,
    #                 self.kap,
    #                 p_c_prev,
    #             )

    #             deps_v_p_fr = pmulti * jax.grad(yield_function, argnums=0)(
    #                 p_next, p_c_next, q_next, self.M
    #             )
    #             yf = yield_function(p_next, p_c_next, q_next, self.M)

    #             R = jnp.array([yf, deps_p_v - deps_v_p_fr])

    #             R = R.at[0].set(R[0] / (K_tr * self.kap))

    #             aux = (p_next, s_next, p_c_next, G_next)

    #             return R, aux

    #         def find_roots():
    #             init_val = jnp.array([0.0, 0.0])

    #             solver = optx.Newton(rtol=1e-8, atol=1e-8)
    #             sol = optx.root_find(
    #                 residuals,
    #                 solver,
    #                 init_val,
    #                 throw=False,
    #                 has_aux=True,
    #                 max_steps=20,
    #             )

    #             return sol.value

    #         pmulti, deps_p_v_next = jax.lax.stop_gradient(find_roots())

    #         R, aux = residuals((pmulti, deps_p_v_next), None)
    #         p_next, s_next, p_c_next, G_next = aux

    #         stress_next = s_next - p_next * jnp.eye(3)

    #         eps_e_v_next = eps_e_v_tr - deps_p_v_next

    #         eps_e_d_next = (s_next - s_ref) / (2.0 * G_next)

    #         eps_e_next = eps_e_d_next - (1.0 / 3) * eps_e_v_next * jnp.eye(3)

    #         return stress_next, eps_e_next, p_c_next

    #     return jax.lax.cond(is_ep, pull_to_ys, elastic_update)

    # def get_timestep(self, cell_size, density, pressure=None, factor=0.1):
    #     if pressure is None:
    #         K = get_K(self.kap, self.p_c.max())
    #     else:
    #         K = get_K(self.kap, pressure)

    #     G = get_G(self.nu, K)

    #     dt = get_timestep(cell_size, K, G, density, factor)
    #     return dt

    # @classmethod
    # def get_p_ref_phi(cls, phi_ref, phi_c, lam, kap):

    #     v_ref = 1.0 / phi_ref

    #     Gamma = 1.0 / phi_c  # phi to specific volume

    #     log_N = jnp.log(Gamma) + (lam - kap) * jnp.log(2)

    #     # p on ICL
    #     log_p = (log_N - jnp.log(v_ref)) / lam

    #     return jnp.exp(log_p)

    # def estimate_timestep(self, p, density, cell_size, factor):

    #     K = get_K(self.kap, p)
    #     G = get_G(self.nu, K)

    #     return get_timestep(cell_size, K, G, density, factor)
