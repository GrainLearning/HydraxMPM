"""Implementation, non-associated Drucker-Prager model with isotropic linear elasticity,
and linear hardening.

[1] de Souza Neto, E.A., Peric, D. and Owen, D.R., 2011. Computational methods for
plasticity.
"""

from functools import partial
from typing import Tuple
from typing_extensions import Self

import chex
import jax
import jax.numpy as jnp
import optax
import optimistix as optx

from ..particles.particles import Particles
from ..utils.math_helpers import (
    get_dev_strain,
    get_dev_stress,
    get_pressure,
    get_q_vm,
    get_sym_tensor_stack,
    get_volumetric_strain,
)
from .common import (
    get_bulk_modulus,
    get_lin_elas_dev,
    get_lin_elas_vol,
    get_shear_modulus,
)
from .material import Material


def yield_function(
    q: jnp.float32, p: jnp.float32, M: jnp.float32, M2: jnp.float32, c: jnp.float32
):
    """Drucker-Prager yield function."""
    return q - M * p - M2 * c


@chex.dataclass
class DruckerPrager(Material):
    r"""Non-associated Drucker-Prager model.

    The Drucker-Prager model is a smooth approximation to the Mohr-Coulomb model.

    This formulation is in small strain and elastic law is  isotropic linear elasticity.

    The implementation follows [1] with the exception that pressure and
    volumetric strain are positive in compression.

    [1] de Souza Neto, E.A., Peric, D. and Owen, D.R., 2011. Computational methods for
    plasticity.



    Attributes:
        E: Young's modulus.
        nu: Poisson's ratio.
        G: Shear modulus.
        K: Bulk modulus.
        M: Mohr-Coulomb friction parameter.
        M_hat: Mohr-Coulomb dilatancy parameter.
        M2: Mohr-Coulomb cohesion parameter.
        c0: Initial cohesion parameter.
        eps_acc_stack: Accumulated plastic strain for linear hardening
        eps_e_stack: Elastic strain tensor.
        H: Hardening modulus
    """

    E: jnp.float32
    nu: jnp.float32
    G: jnp.float32
    K: jnp.float32

    eps_e_stack: chex.Array
    eps_p_acc_stack: chex.Array
    stress_ref_stack: chex.Array

    M: jnp.float32 = 0.0
    M_hat: jnp.float32 = 0.0
    M2: jnp.float32 = 0.0
    c0: jnp.float32 = 0.0

    H: jnp.float32 = 0.0

    @classmethod
    def create(
        cls: Self,
        E: jnp.float32,
        nu: jnp.float32,
        M: jnp.float32 = 0.0,
        M2: jnp.float32 = 0.0,
        M_hat: jnp.float32 = 0.0,
        c0: jnp.float32 = 0.0,
        H: jnp.float32 = 0.0,
        num_particles: jnp.int32 = 1,
        dim: jnp.int16 = 3,
        stress_ref_stack: chex.Array = None,
        absolute_density: jnp.float32 = 1.0,
    ) -> Self:
        """Create a non-associated Drucker-Prager material model."""

        K = get_bulk_modulus(E, nu)
        G = get_shear_modulus(E, nu)

        eps_e_stack = jnp.zeros((num_particles, dim, dim), dtype=jnp.float32)
        eps_p_acc_stack = jnp.zeros(num_particles, dtype=jnp.float32)

        if stress_ref_stack is None:
            stress_ref_stack = jnp.zeros((num_particles, dim, dim), dtype=jnp.float32)

        return cls(
            E=E,
            nu=nu,
            M=M,
            M2=M2,
            M_hat=M_hat,
            c0=c0,
            H=H,
            G=G,
            K=K,
            eps_e_stack=eps_e_stack,
            eps_p_acc_stack=eps_p_acc_stack,
            stress_ref_stack=stress_ref_stack,
            absolute_density=absolute_density,
        )

    # def update_from_particles(
    #     self: Self, particles: Particles, dt: jnp.float32
    # ) -> Tuple[Particles, Self]:
    #     """Update the material state and particle stresses for MPM solver."""
    #     stress_stack, self = self.update(
    #         particles.stress_stack, particles.F_stack, particles.L_stack, None, dt
    #     )

    #     return particles.replace(stress_stack=stress_stack), self
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
        """Update stress using the Drucker-Prager model"""

        deps_stack = get_sym_tensor_stack(L_stack) * dt

        stress_next_stack, eps_e_next_stack, eps_p_acc_next_stack = (
            self.vmap_update_stress(
                deps_stack,
                self.stress_ref_stack,
                self.eps_e_stack,
                self.eps_p_acc_stack,
            )
        )

        return (
            stress_next_stack,
            self.replace(
                eps_e_stack=eps_e_next_stack, eps_p_acc_stack=eps_p_acc_next_stack
            ),
        )

    @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0), out_axes=(0, 0, 0))
    def vmap_update_stress(self, deps_next, stress_ref, eps_e_prev, eps_p_acc_prev):
        dim = deps_next.shape[0]

        # Reference pressure and deviatoric stress
        p_ref = get_pressure(stress_ref, dim)

        s_ref = get_dev_stress(stress_ref, p_ref, dim)

        eps_e_tr = eps_e_prev + deps_next

        eps_e_v_tr = get_volumetric_strain(eps_e_tr)

        eps_e_d_tr = get_dev_strain(eps_e_tr, eps_e_v_tr, 3)

        s_tr = get_lin_elas_dev(eps_e_d_tr, self.G)

        p_tr = get_lin_elas_vol(eps_e_v_tr, self.K)

        p_tr = p_tr + p_ref

        s_tr = s_tr + s_ref

        q_tr = get_q_vm(dev_stress=s_tr, pressure=p_tr)

        c = self.c0 + self.H * eps_p_acc_prev  # linear hardening

        yf = yield_function(q_tr, p_tr, self.M, self.M2, c=c)

        is_ep = yf > 0

        def elastic_update(
            is_ep,
        ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
            """If yield function is negative, return elastic solution."""
            stress = s_tr - p_tr * jnp.eye(3)

            return stress, eps_e_tr, eps_p_acc_prev

        def pull_to_ys(is_ep) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
            """If yield function is negative, return elastic solution."""

            def residuals(sol, args):
                """Reduced system for non-associated flow rule."""
                pmulti = sol

                # solve non associated flow rules
                # volumetric plastic strain increment
                deps_p_v = -pmulti * self.M_hat

                # deviatoric plastic strain increment
                # flow vector is coaxial to deviatoric stress
                deps_p_dev = ((pmulti * jnp.sqrt(3)) / 2.0) * (s_tr / q_tr)

                # Trail isotropic linear elastic law
                p_next = p_tr - self.K * deps_p_v
                s_next = s_tr - 2.0 * self.G * deps_p_dev
                q_next = get_q_vm(dev_stress=s_next, pressure=p_next)

                # linear hardening
                eps_p_acc_next = eps_p_acc_prev + self.M2 * deps_p_v

                c = self.c0 + self.H * eps_p_acc_next

                aux = p_next, s_next, eps_p_acc_next

                R = yield_function(q_next, p_next, self.M, self.M2, c=c)

                return jnp.abs(R), aux

            solver = optx.OptaxMinimiser(
                optax.adabelief(learning_rate=0.0001), rtol=1e-11, atol=1e-11
            )

            sol = optx.minimise(
                residuals,
                solver,
                0.0,
                throw=False,
                has_aux=True,
            )

            # jax.debug.print("sol {} is_ep {}", sol.result, is_ep)
            p_next, s_next, eps_p_acc_next = sol.aux

            stress_next = s_next - p_next * jnp.eye(3)

            eps_e_next = (s_next - s_ref) / (2.0 * self.G) - (p_next - p_ref) / (
                3.0 * self.K
            ) * jnp.eye(3)
            # jax.debug.print("plastic step")
            return stress_next, eps_e_next, eps_p_acc_next
            # return 1

            # stress = s_tr - p_tr * jnp.eye(3)
            # return stress, eps_e_tr, eps_p_acc_prev

        # stress_next = s_tr - p_tr * jnp.eye(3)
        stress_next, eps_e_next, eps_p_acc_next = jax.lax.cond(
            is_ep, pull_to_ys, elastic_update, None
        )

        return stress_next, eps_e_next, eps_p_acc_next
        # jax.debug.print("is_ep_result {}", is_ep_result)
        # return stress_next, eps_e_tr, eps_p_acc_prev

    # # def update_stress(
    # #     self: Self, particles: Particles, dt: jnp.float32
    # # ) -> Tuple[Particles, Self]:
    # #     """Update stress and strain for all particles."""
    # #     vel_grad = particles.velgrads

    # #     vel_grad_T = jnp.transpose(vel_grad, axes=(0, 2, 1))

    # #     deps = 0.5 * (vel_grad + vel_grad_T) * dt
    # #     # Elastic step
    # #     s_tr, p_tr, eps_e_tr, s_ref, p_ref = self.vmap_elastic_trail_step(
    # #         self.stress_ref, self.eps_e, deps
    # #     )

    # #     # Plastic return mapping
    # #     stress_next, eps_p_acc_next, eps_e_next = self.vmap_plastic_return_mapping(
    # #         s_tr, p_tr, eps_e_tr, self.eps_p_acc, s_ref, p_ref
    # #     )

    # #     particles = particles.replace(stresses=stress_next)

    #     return particles, self

    # @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0, 0), out_axes=(0, 0, 0))
    # def vmap_plastic_return_mapping(
    #     self,
    #     s_tr: chex.ArrayBatched,
    #     p_tr: chex.ArrayBatched,
    #     eps_e_tr: chex.ArrayBatched,
    #     eps_p_acc: chex.ArrayBatched,
    #     s_ref: chex.ArrayBatched,
    #     p_ref: chex.ArrayBatched,
    # ) -> Tuple[chex.ArrayBatched, chex.ArrayBatched, chex.ArrayBatched]:
    #     """Plastic implicit return mapping algorithm."""
    #     J2_tr = jnp.sqrt(0.5 * (s_tr @ s_tr.T).trace())

    #     c = self.c0 + self.H * eps_p_acc  # linear hardening

    #     yf = yield_function(J2_tr, p_tr, self.eta, self.xi, c=c)

    #     is_ep = yf > 0

    #     def elastic_update() -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    #         """If yield function is negative, return elastic solution."""
    #         stress = s_tr - p_tr * jnp.eye(3)
    #         return stress, eps_p_acc, eps_e_tr

    #     def return_mapping():
    #         """If yield function is positive, pull back to the yield surface."""
    #         sol = 0.0

    #         R = 1.0

    #         aux_data = p_tr, s_tr, eps_p_acc

    #         def reduced_system_cone(sol):
    #             """Reduced system for non-associated flow rule."""
    #             p_multi = sol
    #             # solve non associated flow rules
    #             # volumetric plastic strain increment
    #             deps_p_v = -p_multi * self.eta_hat

    #             # deviatoric plastic strain increment
    #             # flow vector is coaxial to deviatoric stress
    #             deps_p_gamma = p_multi * s_tr / (2 * jnp.sqrt(J2_tr))

    #             # Trail isotropic linear elastic law
    #             p_next = p_tr - self.K * deps_p_v
    #             s_next = s_tr - 2.0 * self.G * deps_p_gamma

    #             # linear hardening
    #             eps_p_acc_next = eps_p_acc + self.xi * deps_p_v
    #             c = self.c0 + self.H * eps_p_acc_next

    #             aux_data = p_next, s_next, eps_p_acc_next
    #             R = yield_function(J2_tr, p_next, self.eta, self.xi, c=c)

    #             # R = R / (self.E * p_next)

    #             return R, aux_data

    #         def reduced_system_apex(sol):
    #             p_multi = sol
    #             deps_p_v = -p_multi * self.eta_hat
    #             p_next = p_tr - self.K * deps_p_v
    #             eps_p_acc_next = eps_p_acc + self.xi * deps_p_v
    #             c = self.c0 + self.H * eps_p_acc_next
    #             R = yield_function(0, p_next, self.eta, self.xi, c=c)

    #             # R = R / (self.E * p_next)

    #             s_next = jnp.zeros((3, 3), dtype=jnp.float32)

    #             aux_data = p_next, s_next, eps_p_acc_next

    #             return R, aux_data

    #         def NewtonRhapson(step, carry):
    #             """Newton-Raphson iteration cone."""
    #             R, sol, aux_data = carry

    #             R, aux_data = jax.lax.cond(
    #                 is_cone,
    #                 reduced_system_cone,
    #                 reduced_system_apex,
    #                 sol,
    #             )

    #             d, *_ = jax.lax.cond(
    #                 is_cone,
    #                 jax.jacfwd(reduced_system_cone, has_aux=True),
    #                 jax.jacfwd(reduced_system_apex, has_aux=True),
    #                 sol,
    #             )

    #             # d, *_ = jax.jacfwd(reduced_system_cone, has_aux=True)(sol)

    #             sol = sol - R / d

    #             return R, sol, aux_data

    #         is_cone = True
    #         R, sol, aux_data = jax.lax.fori_loop(0, 30, NewtonRhapson,
    # (R, sol, aux_data))

    #         is_cone = False
    #         R, sol, aux_data = jax.lax.cond(
    #             J2_tr - self.G * sol >= 0,  # noqa: E999
    #             lambda carry: carry,
    #             lambda carry: jax.lax.fori_loop(0, 30, NewtonRhapson, carry),
    #             (R, sol, aux_data),
    #         )

    #         p_next, s_next, eps_p_acc_next = aux_data

    #         stress_next = s_next - p_next * jnp.eye(3)

    #         eps_e_next = (s_next - s_ref) / (2.0 * self.G) - (p_next - p_ref) / (
    #             3.0 * self.K
    #         ) * jnp.eye(3)

    #         return stress_next, eps_p_acc_next, eps_e_next

    #     return jax.lax.cond(is_ep, return_mapping, elastic_update)

    # def eta_outer(phi):
    #     return 6 * jnp.sin(phi) / (jnp.sqrt(3) * (3 - jnp.sin(phi)))

    # def eta_inner(phi):
    #     return 6 * jnp.sin(phi) / (jnp.sqrt(3) * (3 + jnp.sin(phi)))

    # def eta_plain_strain(phi):
    #     return 3 * jnp.tan(phi) / jnp.sqrt(9 + 12 * jnp.tan(phi) ** 2)

    # def xi_outer(phi):
    #     return 6 * jnp.cos(phi) / (jnp.sqrt(3) * (3 - jnp.sin(phi)))

    # def xi_inner(phi):
    #     return 6 * jnp.cos(phi) / (jnp.sqrt(3) * (3 + jnp.sin(phi)))

    # def xi_plain_strain(phi):
    #     return 3 / (9 + 12 * jnp.tan(phi) ** 2)

    # def sin_angle(phi):
    #     return jnp.sin(phi)

    # eta = jax.lax.switch(
    #     cone_approximation,
    #     [eta_outer, eta_inner, eta_plain_strain, sin_angle],
    #     friction_angle,
    # )

    # eta_hat = jax.lax.switch(
    #     cone_approximation,
    #     [eta_outer, eta_inner, eta_plain_strain, sin_angle],
    #     dilatancy_angle,
    # )

    # xi = jax.lax.switch(
    #     cone_approximation, [xi_outer, xi_inner, xi_plain_strain], friction_angle
    # )
