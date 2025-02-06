"""Implementation, state and functions for isotropic linear elastic material."""

from functools import partial

import chex
import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array
from typing_extensions import Self

from ..utils.math_helpers import (
    get_dev_strain,
    get_dev_stress,
    get_pressure,
    get_pressure_stack,
    get_q_vm,
    get_volumetric_strain,
    phi_to_e,
)
from .common import get_flattened_triu, get_symmetric_tensor_from_flattened_triu
from .material import Material


def get_elas_non_linear_pressure(deps_e_v, kap, p_prev, V0):
    """Compute non-linear pressure."""

    return p_prev / (1.0 - (V0 / kap) * deps_e_v)


def get_elas_dev_stress(eps_e_d, s_ref, G):
    return 2.0 * G * eps_e_d + s_ref


# def get_non_linear_hardening_pressure(deps_p_v, lam, kap, p_c_prev, V0):
#     return p_c_prev / (1.0 - (1.0 / (lam - kap)) * deps_p_v)


def yield_function(p, q, H, cp, px0, M, X, ps):
    first_term = jnp.log((1.0 + ((1.0 + X) * q**2) / (p**2 * M**2 - X * q**2)) * p + ps)
    second_term = -jnp.log(px0 + ps)
    third_term = -H / cp
    return first_term + second_term + third_term


def plastic_potential(stress, H, cp, px0, M_c):
    q = get_q_vm(stress)
    p = get_pressure(stress)
    return jnp.log(p / px0) + jnp.log(1 + (q**2) / (p**2 * M_c**2)) - H / cp


def get_Mc(m_state, eta_state, M):
    return M * jnp.exp(-m_state * eta_state)


def get_e_m(p, q, Z, M, lam, kap, ps, X):
    m = q / p

    inner_log1 = (p + ps) / (1.0 + ps)

    inner_log2 = (((M**2 + m**2) / (M**2 - X * m**2)) * p + ps) / (p + ps)

    return Z - lam * jnp.log(inner_log1) - (lam - kap) * jnp.log(inner_log2)


def get_e_state_variable(volume_fraction, e_m):
    e = phi_to_e(volume_fraction)
    return e_m - e


def get_Mf(M, state_variable, lam, kap):
    R = jnp.exp(-state_variable / (lam - kap))
    term_sqrt = jnp.sqrt((12 * (3.0 - M) / M**2) * R + 1.0)
    return 6.0 / (term_sqrt + 1)


def get_H(q, p, Mf, M_c, deps_p_v, H_prev):
    m = q / p
    return ((Mf**4 - m**4) / (M_c**4 - m**4)) * deps_p_v + H_prev


def get_K(V0, kap, p):
    return (V0 / kap) * p


def get_G(nu, K):
    return (3.0 * (1.0 - 2.0 * nu)) / (2.0 * (1.0 + nu)) * K


@chex.dataclass
class CSUHModel(Material):
    px0: Array
    H: Array
    eps_e: Array
    stress_ref: Array

    nu: jnp.float32

    M: jnp.float32
    R: jnp.float32
    lam: jnp.float32
    kap: jnp.float32
    V0: jnp.float32
    N: jnp.float32

    Z: jnp.float32
    X: jnp.float32
    m_state: jnp.float32
    Ps: jnp.float32

    @classmethod
    def create(
        cls: Self,
        nu: jnp.float32,
        M: jnp.float32,
        R: jnp.float32,
        lam: jnp.float32,
        kap: jnp.float32,
        V0: jnp.float32,
        N: jnp.float32,
        X: jnp.float32,
        m_state: jnp.float32,
        Z: jnp.float32 = None,
        Ps: jnp.float32 = None,
        stress_ref: Array = None,
        num_particles: jnp.int32 = 1,
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
            stress_ref (Array): Reference stress tensor.
            num_particles (_type_): Number of particles.
            dim (jnp.int16, optional): Dimension of the domain. Defaults to 3.
        """

        eps_e = jnp.zeros((num_particles, 3, 3), dtype=jnp.float32)

        if stress_ref is None:
            stress_ref = jnp.zeros((num_particles, 3, 3), dtype=jnp.float32)

        H = jnp.zeros(num_particles, dtype=jnp.float32)
        px0 = get_pressure_stack(stress_ref, dim)

        px0 = px0 * R
        e0 = V0 - 1.0

        if (Z is None) & (Ps is None):
            inner = jnp.exp((N - e0) / lam) + 1.0

            Z = N - lam * jnp.log(inner)
            Ps = jnp.exp((N - e0) / lam)
        elif Ps is None:
            Ps = jnp.exp((N - Z) / lam) - 1
        elif Z is None:
            Z = (V0 - 1.0) - lam * jnp.log((1.0 + Ps) / Ps)

        # jax.debug.print("{}", Z)
        return cls(
            stress_ref=stress_ref,
            px0=px0,
            H=H,
            eps_e=eps_e,
            nu=nu,
            M=M,
            R=R,
            lam=lam,
            kap=kap,
            V0=V0,
            N=N,
            Ps=Ps,
            Z=Z,
            X=X,
            m_state=m_state,
        )

    def update_stress_benchmark(
        self: Self,
        stress_prev: chex.Array,
        strain_rate: Array,
        volume_fraction: Array,
        dt: jnp.float32,
    ) -> Self:
        deps = strain_rate * dt

        stress_next, eps_e_next, H_next = self.vmap_update_stress(
            volume_fraction,
            deps,
            self.stress_ref,
            stress_prev,
            self.eps_e,
            self.px0,
            self.H,
        )
        # jax.debug.print("{}", H_next)
        return self.replace(eps_e=eps_e_next, H=H_next), stress_next

    @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0, 0, 0), out_axes=(0, 0, 0))
    def vmap_update_stress(
        self,
        volume_fraction_next,
        deps_next,
        stress_ref,
        stress_prev,
        eps_e_prev,
        px0,
        H_prev,
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

        p_tr = get_elas_non_linear_pressure(deps_e_v, self.kap, p_prev, self.V0)

        K_tr = get_K(self.V0, self.kap, p_tr)

        G_tr = get_G(self.nu, K_tr)

        s_tr = get_elas_dev_stress(eps_e_d_tr, s_ref, G_tr)

        H_tr = H_prev

        cp = self.lam - self.kap

        q_tr = get_q_vm(dev_stress=s_tr)

        yf = yield_function(p_tr, q_tr, H_tr, cp, px0, self.M, self.X, self.Ps)

        is_ep = yf > 0

        def elastic_update():
            stress_next = s_tr - p_tr * jnp.eye(3)
            return stress_next, eps_e_tr, H_tr

        def pull_to_ys():
            # stress_next = s_tr - p_tr * jnp.eye(3)

            def residuals(sol, args):
                pmulti, *deps_p_flat = sol

                deps_p = get_symmetric_tensor_from_flattened_triu(deps_p_flat)

                deps_p_v = get_volumetric_strain(deps_p)

                deps_p_d = get_dev_strain(deps_p, deps_p_v)

                p_next = get_elas_non_linear_pressure(
                    deps_e_v - deps_p_v, self.kap, p_prev, self.V0
                )
                K_tr = get_K(self.V0, self.kap, p_tr)

                G_tr = get_G(self.nu, K_tr)

                s_next = s_tr - get_elas_dev_stress(deps_p_d, jnp.zeros((3, 3)), G_tr)

                q_next = get_q_vm(dev_stress=s_next)

                e_m = get_e_m(
                    p_next, q_next, self.Z, self.M, self.lam, self.kap, self.Ps, self.X
                )

                state_variable = get_e_state_variable(volume_fraction_next, e_m)

                Mf = get_Mf(self.M, state_variable, self.lam, self.kap)

                Mc = get_Mc(self.m_state, state_variable, self.M)

                H_next = get_H(q_next, p_next, Mf, Mc, deps_p_v, H_prev)

                yf = yield_function(
                    p_next, q_next, H_next, cp, px0, self.M, self.X, self.Ps
                )

                stress_next = s_next - p_next * jnp.eye(3)
                deps_p_fr = pmulti * jax.grad(plastic_potential, argnums=0)(
                    stress_next, H_next, cp, px0, Mc
                )
                deps_p_fr_flat = get_flattened_triu(deps_p - deps_p_fr)

                R = jnp.array([yf, *deps_p_fr_flat])
                R = R.at[0].set(R[0] / ((K_tr * self.kap) / self.V0))

                aux = (p_next, s_next, H_next)

                return R, aux

            def find_roots():
                init_val = jnp.zeros(7)

                solver = optx.Newton(rtol=1e-3, atol=1e-3)
                sol = optx.root_find(
                    residuals, solver, init_val, throw=False, has_aux=True, max_steps=20
                )

                # p_next, s_next, H_next = sol.aux
                # _, *deps_p_flat = sol.value
                return sol.value

            pmulti, *deps_p_flat = jax.lax.stop_gradient(find_roots())

            R, aux = residuals([pmulti, *deps_p_flat], None)
            p_next, s_next, H_next = aux
            stress_next = s_next - p_next * jnp.eye(3)

            eps_e_next = eps_e_tr - get_symmetric_tensor_from_flattened_triu(
                deps_p_flat
            )

            return stress_next, eps_e_next, H_next

        return jax.lax.cond(is_ep, pull_to_ys, elastic_update)
