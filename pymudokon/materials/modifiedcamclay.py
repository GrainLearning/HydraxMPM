"""Implementation, state and functions for isotropic linear elastic material."""

from functools import partial
from typing import Tuple
from typing_extensions import Self

import chex
import jax
import jax.numpy as jnp
import optimistix as optx

from ..particles.particles import Particles
from ..utils.math_helpers import (
    get_dev_strain,
    get_dev_stress,
    get_pressure,
    get_pressure_stack,
    get_q_vm,
    get_sym_tensor_stack,
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

    return p_prev / (1.0 - (1.0 / kap) * deps_e_v)


def get_elas_dev_stress(eps_e_d, s_ref, G):
    return 2.0 * G * eps_e_d + s_ref


def get_non_linear_hardening_pressure(deps_p_v, lam, kap, p_c_prev):
    return p_c_prev / (1.0 - (1.0 / (lam - kap)) * deps_p_v)


def yield_function(p, p_c, q, M):
    """Compute the modified Cam Clay yield function."""
    return q**2 / M**2 - p * (p_c - p)


def get_K(kap, p):
    return (1.0 / kap) * p


def get_G(nu, K):
    return (3.0 * (1.0 - 2.0 * nu)) / (2.0 * (1.0 + nu)) * K


@chex.dataclass
class ModifiedCamClay(Material):
    p_c_stack: chex.Array
    eps_e_stack: chex.Array
    stress_ref_stack: chex.Array
    nu: jnp.float32
    M: jnp.float32
    R: jnp.float32
    lam: jnp.float32
    kap: jnp.float32
    Vs: jnp.float32

    @classmethod
    def create(
        cls: Self,
        nu: jnp.float32,
        M: jnp.float32,
        R: jnp.float32,
        lam: jnp.float32,
        kap: jnp.float32,
        Vs: jnp.float32,
        stress_ref_stack: chex.Array = None,
        absolute_density: jnp.float32 = 1.0,
        dim: jnp.int16 = 3,
    ) -> Self:
        """Create a new instance of the Modified Cam Clay model.

        Args:
            cls (Self): Self type reference
            E (jnp.float32): Young's modulus.
            nu (jnp.float32): Poisson's ratio.
            M (jnp.float32): Slope of Critcal state line.
            R (jnp.float32): Overconsolidation ratio.
            lam (jnp.float32): Compression index.
            kap (jnp.float32): Decompression index.
            Vs (jnp.float32): Specific volume.
            stress_ref_stack (Array): Reference stress tensor.
            num_particles (_type_): Number of particles.
            dim (jnp.int16, optional): Dimension of the domain. Defaults to 3.
        """
        num_particles = stress_ref_stack.shape[0]
        eps_e_stack = jnp.zeros((num_particles, 3, 3), dtype=jnp.float32)

        if stress_ref_stack is None:
            stress_ref_stack = jnp.zeros((num_particles, 3, 3), dtype=jnp.float32)

        p_ref_stack = get_pressure_stack(stress_ref_stack, dim)

        p_c_stack = p_ref_stack * R

        import warnings

        warnings.warn("Warning modified cam clay requires particles with at least 1Pa")

        return cls(
            stress_ref_stack=stress_ref_stack,
            eps_e_stack=eps_e_stack,
            p_c_stack=p_c_stack,
            nu=nu,
            M=M,
            R=R,
            lam=lam,
            kap=kap,
            Vs=Vs,
            absolute_density=absolute_density,
        )

    def update_from_particles(
        self: Self, particles: Particles, dt: jnp.float32
    ) -> Tuple[Particles, Self]:
        """Update the material state and particle stresses for MPM solver."""
        stress_stack, self = self.update(
            particles.stress_stack, particles.F_stack, particles.L_stack, None, dt
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
        deps_stack = get_sym_tensor_stack(L_stack) * dt
        stress_next_stack, eps_e_next_stack, p_c_next_stack = self.vmap_update_stress(
            deps_stack,
            self.stress_ref_stack,
            stress_prev_stack,
            self.eps_e_stack,
            self.p_c_stack,
        )

        return (
            stress_next_stack,
            self.replace(eps_e_stack=eps_e_next_stack, p_c_stack=p_c_next_stack),
        )

    @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0), out_axes=(0, 0, 0))
    def vmap_update_stress(
        self,
        deps_next,
        stress_ref,
        stress_prev,
        eps_e_prev,
        p_c_prev,
    ):
        dim = deps_next.shape[0]

        p_prev = get_pressure(stress_prev, dim)

        # Reference pressure and deviatoric stress
        p_ref = get_pressure(stress_ref, dim)

        s_ref = get_dev_stress(stress_ref, p_ref, dim)

        # Trail Elastic strain
        eps_e_tr = eps_e_prev + deps_next

        eps_e_v_tr = get_volumetric_strain(eps_e_tr)

        eps_e_d_tr = get_dev_strain(eps_e_tr, eps_e_v_tr)

        deps_e_v = get_volumetric_strain(deps_next)

        p_tr = get_elas_non_linear_pressure(deps_e_v, self.kap, p_prev)

        K_tr = get_K(self.kap, p_tr)

        G_tr = get_G(self.nu, K_tr)

        s_tr = get_elas_dev_stress(eps_e_d_tr, s_ref, G_tr)

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

                solver = optx.Newton(rtol=1e-3, atol=1e-3)
                sol = optx.root_find(
                    residuals, solver, init_val, throw=False, has_aux=True, max_steps=20
                )

                return sol.value

            pmulti, deps_p_v_next = jax.lax.stop_gradient(find_roots())

            R, aux = residuals((pmulti, deps_p_v_next), None)
            p_next, s_next, p_c_next, G_next = aux
            stress_next = s_next - p_next * jnp.eye(3)

            eps_e_v_next = eps_e_v_tr - deps_p_v_next

            eps_e_d_next = (s_next - s_ref) / (2.0 * G_next)

            eps_e_next = eps_e_d_next - (1 / 3) * eps_e_v_next * jnp.eye(3)

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
    def get_p_ref_phi(cls, phi_ref, phi_c, lam, kap):
        
        v_ref = 1.0/phi_ref
        
        
        Gamma = 1.0/phi_c # phi to specific volume
        
        log_N = jnp.log(Gamma) +( lam-kap)*jnp.log(2)
        
        # p on ICL
        
        log_p = (log_N - jnp.log(v_ref))/lam
        
        return jnp.exp(log_p)