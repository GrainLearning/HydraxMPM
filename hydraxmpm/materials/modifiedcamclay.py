"""Implementation, state and functions for isotropic linear elastic material."""

from functools import partial
from typing import Tuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from typing_extensions import Self, Union

from ..config.ip_config import IPConfig
from ..config.mpm_config import MPMConfig
from ..particles.particles import Particles
from ..utils.math_helpers import (
    get_dev_strain,
    get_pressure,
    get_q_vm,
    get_sym_tensor,
    get_volumetric_strain,
)
from .common import get_timestep
from .material import Material


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


def get_elas_non_linear_pressure(deps_e_v, kap, p_prev):
    """Compute non-linear pressure."""
    out = p_prev / (1.0 - (1.0 / kap) * deps_e_v)
    return out
    # return jnp.maximum(,0.0)
    # return jnp.nan_to_num(out, neginf=0, nan=0)


def get_elas_dev_stress(eps_e_d, G):
    return 2.0 * G * eps_e_d


def get_non_linear_hardening_pressure(deps_p_v, lam, kap, p_c_prev):
    out = p_c_prev / (1.0 - (1.0 / (lam - kap)) * deps_p_v)
    return out
    # return jnp.nan_to_num(out, neginf=0, nan=0)


def yield_function(p, p_c, q, M):
    """Compute the modified Cam Clay yield function."""
    return q**2 / M**2 - p * (p_c - p)


def get_K(kap, p):
    return (1.0 / kap) * p


def get_G(nu, K):
    return (3.0 * (1.0 - 2.0 * nu)) / (2.0 * (1.0 + nu)) * K


class ModifiedCamClay(Material):
    """
    Return mapping follows implementation of

    """

    p_c_stack: chex.Array
    eps_e_stack: chex.Array
    p_ref_stack: chex.Array

    rho_p: float = eqx.field(static=True, converter=lambda x: float(x))
    kap: float = eqx.field(static=True, converter=lambda x: float(x))
    lam: float = eqx.field(static=True, converter=lambda x: float(x))
    R: float = eqx.field(static=True, converter=lambda x: float(x))
    M: float = eqx.field(static=True, converter=lambda x: float(x))
    nu: float = eqx.field(static=True, converter=lambda x: float(x))

    def __init__(
        self: Self,
        config: Union[MPMConfig, IPConfig],
        nu: jnp.float32,
        M: jnp.float32,
        R: jnp.float32,
        lam: jnp.float32,
        kap: jnp.float32,
        p_ref_stack: chex.Array,
        rho_p: jnp.float32 = 0.0,
    ) -> Self:
        # Check if kappa less than lambda

        eps_e_stack = jnp.zeros((config.num_points, 3, 3))

        p_c_stack = p_ref_stack * R

        self.p_ref_stack = p_ref_stack
        self.eps_e_stack = eps_e_stack
        self.p_c_stack = p_c_stack
        self.nu = nu
        self.M = M
        self.R = R
        self.lam = lam
        self.kap = kap
        self.rho_p = rho_p

        self.config = config

    # def update_from_particles(
    #     self: Self, particles: Particles
    # ) -> Tuple[Particles, Self]:
    #     """Update the material state and particle stresses for MPM solver."""

    #     stress_stack, self = self.update(
    #         particles.stress_stack, particles.F_stack, particles.L_stack, None, dt
    #     )

    #     return particles.replace(stress_stack=stress_stack), self

    # def update(
    #     self: Self,
    #     stress_prev_stack: chex.Array,
    #     F_stack: chex.Array,
    #     L_stack: chex.Array,
    #     phi_stack: chex.Array,
    #     dt: jnp.float32,
    # ) -> Tuple[chex.Array, Self]:
    #     deps_stack = get_sym_tensor_stack(L_stack) * dt
    #     stress_next_stack, eps_e_next_stack, p_c_next_stack = self.vmap_update_stress(
    #         deps_stack,
    #         self.stress_ref_stack,
    #         stress_prev_stack,
    #         self.eps_e_stack,
    #         self.p_c_stack,
    #     )

    #     return (
    #         stress_next_stack,
    #         self.replace(eps_e_stack=eps_e_next_stack, p_c_stack=p_c_next_stack),
    # )

    def update_from_particles(
        self: Self, particles: Particles
    ) -> Tuple[Particles, Self]:
        """Update the material state and particle stresses for MPM solver."""
        # phi_stack = particles.get_solid_volume_fraction_stack(self.rho_p)

        new_stress_stack, new_eps_e_stack, new_p_c_stack = self.vmap_update_ip(
            particles.L_stack,
            self.p_ref_stack,
            particles.stress_stack,
            self.eps_e_stack,
            self.p_c_stack,
        )

        # jax.debug.print("{}", new_stress_stack)
        new_particles = eqx.tree_at(
            lambda state: (state.stress_stack),
            particles,
            (new_stress_stack),
        )
        new_self = eqx.tree_at(
            lambda state: (state.eps_e_stack, state.p_c_stack),
            self,
            (new_eps_e_stack, new_p_c_stack),
        )

        return new_particles, new_self

    def update(
        self: Self,
        stress_prev_stack: chex.Array,
        F_stack: chex.Array,
        L_stack: chex.Array,
        phi_stack: chex.Array,
    ) -> Tuple[chex.Array, Self]:
        new_stress_stack, new_eps_e_stack, new_p_c_stack = self.vmap_update_ip(
            L_stack,
            self.p_ref_stack,
            stress_prev_stack,
            self.eps_e_stack,
            self.p_c_stack,
        )

        new_self = eqx.tree_at(
            lambda state: (state.eps_e_stack, state.p_c_stack),
            self,
            (new_eps_e_stack, new_p_c_stack),
        )

        return (new_stress_stack, new_self)

    @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0), out_axes=(0, 0, 0))
    def vmap_update_ip(self, L, p_ref, stress_prev, eps_e_prev, p_c_prev):
        deps_next = get_sym_tensor(L) * self.config.dt
        dim = deps_next.shape[0]

        p_prev = get_pressure(stress_prev, dim)

        # Trail Elastic strain
        eps_e_tr = eps_e_prev + deps_next

        eps_e_v_tr = get_volumetric_strain(eps_e_tr)

        eps_e_d_tr = get_dev_strain(eps_e_tr, eps_e_v_tr)

        deps_e_v = get_volumetric_strain(deps_next)

        p_tr = get_elas_non_linear_pressure(deps_e_v, self.kap, p_prev)

        K_tr = get_K(self.kap, p_tr)

        G_tr = get_G(self.nu, K_tr)

        s_tr = get_elas_dev_stress(eps_e_d_tr, G_tr)

        q_tr = get_q_vm(dev_stress=s_tr)

        p_c_tr = p_c_prev

        yf = yield_function(p_tr, p_c_tr, q_tr, self.M)

        is_ep = yf > 0

        def elastic_update():
            stress_next = s_tr - p_tr * jnp.eye(3)

            return stress_next, eps_e_tr, p_c_prev

        def pull_to_ys():
            stress_next = s_tr - p_tr * jnp.eye(3)

            def residuals(sol, args):
                pmulti, deps_p_v = sol

                p_next = get_elas_non_linear_pressure(
                    deps_e_v - deps_p_v, self.kap, p_prev
                )

                K_next = get_K(self.kap, p_next)

                G_next = get_G(self.nu, K_next)

                s_next = (self.M**2 / (self.M**2 + 6.0 * G_next * pmulti)) * s_tr

                q_next = (self.M**2 / (self.M**2 + 6.0 * G_next * pmulti)) * q_tr

                p_c_next = get_non_linear_hardening_pressure(
                    deps_p_v,
                    self.lam,
                    self.kap,
                    p_c_prev,
                )

                deps_v_p_fr = pmulti * jax.grad(yield_function, argnums=0)(
                    p_next, p_c_next, q_next, self.M
                )
                yf = yield_function(p_next, p_c_next, q_next, self.M)

                R = jnp.array([yf, deps_p_v - deps_v_p_fr])

                R = R.at[0].set(R[0] / (K_tr * self.kap))

                aux = (p_next, s_next, p_c_next, G_next)

                return R, aux

            def find_roots():
                init_val = jnp.array([0.0, 0.0])

                solver = optx.Newton(rtol=1e-8, atol=1e-8)
                sol = optx.root_find(
                    residuals,
                    solver,
                    init_val,
                    throw=False,
                    has_aux=True,
                    max_steps=5,
                )

                return sol.value

            pmulti, deps_p_v_next = jax.lax.stop_gradient(find_roots())

            R, aux = residuals((pmulti, deps_p_v_next), None)
            p_next, s_next, p_c_next, G_next = aux

            stress_next = s_next - p_next * jnp.eye(3)

            eps_e_v_next = eps_e_v_tr - deps_p_v_next

            eps_e_d_next = s_next / (2.0 * G_next)

            eps_e_next = eps_e_d_next - (1.0 / 3) * eps_e_v_next * jnp.eye(3)

            return stress_next, eps_e_next, p_c_next

        return jax.lax.cond(is_ep, pull_to_ys, elastic_update)

    def get_timestep(self, cell_size, density, pressure=None, factor=0.1):
        if pressure is None:
            K = get_K(self.kap, self.p_c.max())
        else:
            K = get_K(self.kap, pressure)

        G = get_G(self.nu, K)

        dt = get_timestep(cell_size, K, G, density, factor)
        return dt

    @classmethod
    def get_p_ref(cls, phi_ref, ln_N, lam, kap):
        v_ref = 1.0 / phi_ref

        # p on ICL
        log_p = (ln_N - jnp.log(v_ref)) / lam

        return jnp.exp(log_p)

    # def estimate_timestep(self, p, density, cell_size, factor):
    #     K = get_K(self.kap, p)
    #     G = get_G(self.nu, K)

    #     return get_timestep(cell_size, K, G, density, factor)
    # @classmethod
    # def create_from_phi_ref(
    #     cls: Self,
    #     nu: jnp.float32,
    #     M: jnp.float32,
    #     R: jnp.float32,
    #     lam: jnp.float32,
    #     kap: jnp.float32,
    #     phi_c: jnp.float32,
    #     rho_p: jnp.float32,
    #     phi_ref_stack: chex.Array = None,
    #     absolute_density: jnp.float32 = 1.0,
    #     dim: jnp.int16 = 3,
    # ):
    #     p_ref_stack = jax.vmap(cls.get_p_ref_phi, in_axes=(0, None, None, None))(
    #         phi_ref_stack, phi_c, lam, kap
    #     )

    #     def create_stress_ref(p_ref):
    #         return -jnp.eye(3) * p_ref

    #     stress_ref_stack = jax.vmap(create_stress_ref)(p_ref_stack)

    #     return cls.create(
    #         nu=nu,
    #         M=M,
    #         R=R,
    #         lam=lam,
    #         kap=kap,
    #         Vs=1.0,
    #         phi_c=phi_c,
    #         rho_p=rho_p,
    #         stress_ref_stack=stress_ref_stack,
    #         absolute_density=1.0,
    #         dim=dim,
    #     )
