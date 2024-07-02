"""Implementation, state and functions for isotropic linear elastic material."""

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import chex
from typing_extensions import Self

from .material import Material


# def yield_function(p, p_c, q, M):
#     """Compute the modified Cam Clay yield function."""
#     p_s = 0.5 * p_c
#     return (p_s - p) ** 2 + (q / M) ** 2 - p_s**2


@chex.dataclass
class DruckerPrager(Material):
    eps_e: chex.Array
    E: jnp.float32
    nu: jnp.float32
    G: jnp.float32
    K: jnp.float32

    @classmethod
    def create(
        cls: Self,
        E: jnp.float32,
        nu: jnp.float32,
        num_particles: jnp.int32 = 1,
        stress_ref: chex.Array = None,
        dim: jnp.int16 = 3,
    ) -> Self:
        K = E / (3.0 * (1.0 - 2.0 * nu))
        G = E / (2.0 * (1.0 + nu))

        eps_e = jnp.zeros((num_particles, dim, dim), dtype=jnp.float32)

        if stress_ref is None:
            stress_ref = jnp.zeros((num_particles, dim, dim), dtype=jnp.float32)

        return cls(
            stress_ref=stress_ref,
            eps_e=eps_e,
            E=E,
            nu=nu,
            G=G,
            K=K,
        )

    @jax.jit
    def update_stress_benchmark(
        self: Self,
        strain_rate: chex.Array,
        volume: chex.Array,
        dt: jnp.float32,
    ) -> Tuple[chex.Array, Self]:
        chex.assert_shape(volume, (1,))

        chex.assert_shape(strain_rate, (1, 3, 3))

        chex.assert_shape(self.eps_e, (1, 3, 3))

        chex.assert_shape(self.stress_ref, (1, 3, 3))

        strain_rate = strain_rate.reshape(-1, 3, 3)
        deps = strain_rate * dt

        # Elastic trail predictor step
        s_tr, p_tr, eps_e_tr, s_ref, p_ref = self.vmap_elastic_trail_step(self.stress_ref, self.eps_e, deps)

        return self.replace(eps_e=eps_e_tr), s_tr - p_tr * jnp.eye(3)

        # # Plastic return mapping
        # stress_next, p_c_next, eps_e_next, eps_v_p_next = self.vmap_plastic_return_mapping(
        #     s_tr, p_tr, self.p_c, eps_e_tr, self.eps_v_p, volumes, s_ref, p_ref
        # )

        # return stress_next, self.replace(eps_e=eps_e_next, p_c=p_c_next, eps_v_p=eps_v_p_next)

    @partial(jax.vmap, in_axes=(None, 0, 0, 0), out_axes=(0, 0, 0, 0, 0))
    def vmap_elastic_trail_step(
        self, stress_ref: chex.Array, eps_e_prev: chex.Array, deps_next: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        dim = deps_next.shape[0]

        eps_e_tr = eps_e_prev + deps_next

        eps_e_v_tr = -jnp.trace(eps_e_tr)

        eps_e_d_tr = eps_e_tr + (eps_e_v_tr / dim) * jnp.eye(dim)

        s_tr = 2.0 * self.G * eps_e_d_tr

        # pad for 3D stress tensor
        if dim == 2:
            s_tr = jnp.pad(s_tr, ((0, 1), (0, 1)), mode="constant")

        p_tr = self.K * eps_e_v_tr

        p_ref = -jnp.trace(stress_ref) / dim
        s_ref = stress_ref + p_ref * jnp.eye(3)

        p_tr = p_tr + p_ref

        s_tr = s_tr + s_ref

        return s_tr, p_tr, eps_e_tr, s_ref, p_ref

    # @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0), out_axes=(0, 0, 0, 0))
    # def vmap_plastic_return_mapping(
    #     self,
    #     s_tr: Array,
    #     p_tr: Array,
    #     p_c_prev: Array,
    #     eps_e_tr: Array,
    #     eps_v_p_prev: Array,
    #     volume: Array,
    #     s_ref: Array,
    #     p_ref: Array,
    # ) -> Tuple[Array, Array, Array, Array]:
    #     """Vectorized plastic return mapping algorithm for modified Cam-Clay.

    #     Args:
    #         s_tr (Array): Trail deviatoric stress tensor.
    #         p_tr (Array): Trail pressure.
    #         p_c_prev (Array): Preconsolidation pressure.
    #         eps_e_tr (Array): Elastic strain tensor.
    #         eps_v_p_prev (Array): Volumetric plastic strain.
    #         volume (Array): Volume of the particles.
    #         s_ref (Array): Reference deviatoric stress tensor.
    #         p_ref (Array): Reference pressure.

    #     Returns:
    #         Tuple[Array, Array, Array, Array]:
    #             Tuple containing the updated stress, preconsolidation pressure, elastic strain tensor, plastic volumetric strain tensor.
    #     """
    #     specific_volume = volume / self.Vs

    #     v_lam_tilde = specific_volume / (self.lam - self.kap)

    #     q_tr = jnp.sqrt(1.5 * (s_tr @ s_tr.T).trace())

    #     is_ep = yield_function(p_tr, p_c_prev, q_tr, self.M) > 0

    #     # The following variables are global and passed implicity to down to functions where used
    #     # s_tr, p_tr, p_c_prev, eps_e_tr, eps_v_p_prev, volume, s_ref, p_ref, q_tr, is_ep
    #     # self,v_lam_tilde, is_ep

    #     def accept_elas() -> Tuple[Array, Array, Array, Array]:
    #         """If yield function is negative, return elastic solution."""
    #         stress = s_tr - p_tr * jnp.eye(3)
    #         return stress, p_c_prev, eps_e_tr, eps_v_p_prev

    #     def reduced_equation_system(sol: Array) -> Tuple[Array, Array]:
    #         """Solve reduced system of equations for plastic return mapping."""
    #         pmultp, eps_p_v_next = sol

    #         deps_p_v = eps_p_v_next - eps_v_p_prev

    #         p_next = p_tr - self.K * deps_p_v

    #         q_next = (self.M**2 / (self.M**2 + 6.0 * self.G * pmultp)) * q_tr

    #         p_c_next = p_c_prev * (1.0 + v_lam_tilde * deps_p_v)

    #         p_s_next = 0.5 * p_c_next

    #         s_next = (self.M**2 / (self.M**2 + 6.0 * self.G * pmultp)) * s_tr

    #         R = jnp.array(
    #             [
    #                 yield_function(p_next, p_c_next, q_next, self.M),
    #                 eps_p_v_next - eps_v_p_prev + 2.0 * pmultp * (p_s_next - p_next),
    #             ]
    #         )

    #         aux_data = p_next, s_next, p_c_next, eps_p_v_next

    #         # normalize residual for convergence check
    #         R = R.at[0].set(R[0] / (self.E * p_c_prev))

    #         conv = jnp.linalg.norm(R)

    #         return R, aux_data

    #     def pull_to_yield_surface():
    #         """Solve the plastic return mapping algorithm using Newton Rhapson method."""
    #         sol = jnp.array([0.0, eps_v_p_prev])

    #         R = jnp.array([1.0, 1.0])

    #         aux_data = p_tr, s_tr, p_c_prev, eps_v_p_prev

    #         def body_loop(step, carry):
    #             R, sol, aux_data = carry

    #             R, aux_data = reduced_equation_system(sol)

    #             jac, *_ = jax.jacfwd(reduced_equation_system, has_aux=True)(sol)
    #             inv_jac = jnp.linalg.inv(jac)

    #             sol = sol - inv_jac @ R

    #             return R, sol, aux_data

    #         # R, sol, aux_data = jax.lax.while_loop(
    #         # lambda carry: is_ep & (jnp.abs(jnp.linalg.norm(carry[0])) > 1e-2),
    #         # body_loop,
    #         # (R, sol, aux_data)
    #         # )
    #         R, sol, aux_data = jax.lax.fori_loop(
    #             0, 50, body_loop, (R, sol, aux_data)
    #         )  # autodiff supported for static jax loops

    #         p_next, s_next, p_c_next, eps_p_v_next = aux_data

    #         stress = s_next - p_next * jnp.eye(3)

    #         eps_e_next = (s_next - s_ref) / (2.0 * self.G) - (p_next - p_ref) / (3.0 * self.K) * jnp.eye(3)
    #         return stress, p_c_next, eps_e_next, eps_p_v_next

    #     return jax.lax.cond(is_ep, pull_to_yield_surface, accept_elas)
