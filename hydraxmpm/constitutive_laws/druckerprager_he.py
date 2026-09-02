# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-

from typing import Tuple, Optional, Self
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array

from .constitutive_law import (
    ConstitutiveLawState,
    ConstitutiveLaw,
    Convergence,
    dummy_convergence,
)
from ..material_points.material_points import MaterialPointState
# from ..utils.math_helpers import (
#     get_dev_strain,
#     get_J2,
#     get_volumetric_strain,
#     get_pressure,
#     get_dev_stress,
#     get_sym_tensor,
#     get_spin_tensor,
#     get_jaumann_increment,
#     get_inertial_number,
#     inv_2x2_robust,
# )


class DruckerPragerHEState(ConstitutiveLawState):
    be_stack: Float[Array, "num_points 3 3"]
    eps_v_pradhana_stack: Float[Array, "num_points"]
    convergence: Optional[Convergence] = None


class DruckerPragerHE(ConstitutiveLaw):
    """
    Non-associated Drucker-Prager model with linear hardening and
    isotropic linear elasticity. Follows return mapping algorithm described in [1].


    - [1] de Souza Neto, Eduardo A., Djordje Peric, and David RJ Owen. Computational methods for plasticity: theory and applications. John Wiley & Sons, 2008.
    """

    # Return mapping
    NUM_NEWTON_ITERS = 5
    NUM_RESIDUALS = 1
    NUM_UNKNOWNS = 1

    K: float | Float[Array, ""]
    G: float | Float[Array, ""]
    rho_p: float | Float[Array, ""]
    M_csl: float | Float[Array, ""]

    def __init__(
        self,
        *,
        K: float | Float[Array, ""],
        nu: float | Float[Array, ""],
        M_csl: float | Float[Array, ""],
        rho_p: float | Float[Array, ""] = 2000.0,
        debug_convergence: bool = False,
    ):
        self.K = K
        E = 3.0 * K * (1.0 - 2.0 * nu)
        self.G = E / (2.0 * (1.0 + nu))
        self.M_csl = M_csl

        self.rho_p = rho_p

        self.debug_convergence = debug_convergence
    
    def give_v_0_stack(self, density0_stack):
        """Calculate reference specific volume v_0 = rho_p / rho_0."""
        return self.rho_p / density0_stack

    def give_density_stack(
        self,
        p_stack,
        density0_stack,
    ):
        """Get current density given current pressure and reference density."""
        # Point-wise specific volume: v_0 = rho_p / rho_0
        v_0 = self.rho_p / density0_stack

        # Specific volume at current pressure: v = v_0 * exp(-p / K)
        v = v_0 * jnp.exp(-p_stack / self.K)

        return self.rho_p / v

    def create_state(
        self,
        stress_stack: Float[Array, "num_points 3 3"],
    ) -> DruckerPragerHEState:

        def invert_stress(tau):

            p_tau = jnp.trace(tau) / 3.0

            s_tau = tau - p_tau * jnp.eye(3)

            eps_v = p_tau / jnp.maximum(self.K, 1e-12)
            eps_d = s_tau / jnp.maximum(2.0 * self.G, 1e-12)

            # Total strain tensor
            eps_total = eps_d + (1.0 / 3.0) * eps_v * jnp.eye(3)

            return jax.scipy.linalg.expm(-2.0 * eps_total)

        convergence = None
        if self.debug_convergence:

            jacobians = jnp.zeros(
                (
                    stress_stack.shape[0],
                    self.NUM_NEWTON_ITERS,
                    self.NUM_UNKNOWNS,
                    self.NUM_UNKNOWNS,
                )
            )
            residuals = jnp.zeros(
                (stress_stack.shape[0], self.NUM_NEWTON_ITERS, self.NUM_RESIDUALS)
            )
            unknowns = jnp.zeros(
                (stress_stack.shape[0], self.NUM_NEWTON_ITERS, self.NUM_UNKNOWNS)
            )

            convergence = Convergence(
                residuals=residuals,
                unknowns=unknowns,
                jacobians=jacobians,
            )

        be_initial = jax.vmap(invert_stress)(stress_stack)
        eps_v_init = jnp.zeros(stress_stack.shape[0])

        return DruckerPragerHEState(
            be_stack=be_initial,
            convergence=convergence,
            eps_v_pradhana_stack=eps_v_init,
        )

    def update(
        self,
        mp_state: MaterialPointState,
        law_state: DruckerPragerHEState,
        dt: float | Float[Array, ""],
    ) -> Tuple[MaterialPointState, DruckerPragerHEState]:
        """Vectorized update."""

        new_stress, new_be, new_eps_v_pradhana, convergence = jax.vmap(
            self._update_stress, in_axes=(0, 0, 0, 0, None)
        )(
            mp_state.F_inc_stack,
            law_state.be_stack,
            law_state.eps_v_pradhana_stack,
            mp_state.density_stack,
            dt,
        )

        new_mp = eqx.tree_at(lambda m: (m.stress_stack), mp_state, (new_stress))
        new_law = eqx.tree_at(
            lambda l: (l.be_stack, l.convergence, l.eps_v_pradhana_stack),
            law_state,
            (new_be, convergence, new_eps_v_pradhana),
        )

        return new_mp, new_law

    def _update_stress(self, F_inc, be_prev, eps_v_pradhana_prev, rho, dt):

        cohesion = 0.0
        be_tr = F_inc @ be_prev @ F_inc.T

        eigvals, V = jnp.linalg.eigh(be_tr)
        eigvals = jnp.maximum(eigvals, 1e-12)

        # Hencky Strain (Compression Positive)
        eps_e_tr = -0.5 * jnp.log(eigvals)
        eps_e_v_tr = jnp.sum(eps_e_tr)
        eps_e_q_tr = eps_e_tr - (eps_e_v_tr / 3.0)

        # Predictor Kirchhoff Stress
        p_tr_raw = self.K * eps_e_v_tr
        
        # p_tr = self.K * eps_e_v_tr
        
        p_prad = self.K * eps_v_pradhana_prev
        p_tr = p_tr_raw + p_prad
        
        s_tr = 2.0 * self.G * eps_e_q_tr

        sqrt_J2_tr = jnp.sqrt(jnp.maximum(0.5 * jnp.sum(s_tr**2), 1e-12))
        q_tr = jnp.sqrt(3.0) * sqrt_J2_tr

        # Drucker-Prager Yield
        yf = q_tr - self.M_csl * (p_tr + cohesion)

        is_ep = yf > 0.0
        # is_tension = p_tr <= 1e-6
        is_tension = p_tr <= (-cohesion + 1e-6)
        def apex_projection():
 
            p_target = cohesion

            eps_v_pred_next = (p_target - p_tr_raw) / self.K
            convergence = None
            if self.debug_convergence:
                convergence = dummy_convergence(
                    self.debug_convergence,
                    self.NUM_NEWTON_ITERS,
                    self.NUM_RESIDUALS,
                    self.NUM_UNKNOWNS,
                )
            return jnp.zeros((3, 3)), jnp.eye(3), eps_v_pred_next, convergence
        
        
        def elastic_update():

            stress_principal = s_tr + p_tr
            tau_next = (V * stress_principal) @ V.T
            be_next = (V * jnp.exp(-2.0 * eps_e_tr)) @ V.T

            convergence = None
            if self.debug_convergence:
                convergence = dummy_convergence(
                    self.debug_convergence,
                    self.NUM_NEWTON_ITERS,
                    self.NUM_RESIDUALS,
                    self.NUM_UNKNOWNS,
                )
            return tau_next, be_next, 0.0, convergence

        def pull_to_ys():
            # 1. Closed-form projection
            # Since your residual was: res = (q_next - M_csl * (p_tr + cohesion))
            # The solution where res = 0 is simply:
            q_next = self.M_csl * (p_tr + cohesion)

            # Ensure q_next is not negative (though p_tr + cohesion > 0 usually)
            q_next = jnp.maximum(q_next, 0.0)

            # 2. Scale the deviatoric stress
            # s_next / s_tr = q_next / q_tr
            scaling = q_next / jnp.maximum(q_tr, 1e-12)
            s_next = s_tr * scaling

            # 3. Update the elastic strains
            # Assuming p_tr remains constant (as in your Newton version)
            eps_e_next = (s_next / (2.0 * self.G)) + (p_tr / (3.0 * self.K))

            # 4. Reconstruct Stress and be
            stress_principal = s_next + p_tr
            tau_next = (V * stress_principal) @ V.T
            be_next = (V * jnp.exp(-2.0 * eps_e_next)) @ V.T

            # Dummy convergence for API compatibility
            convergence = None
            if self.debug_convergence:
                convergence = dummy_convergence(
                    self.debug_convergence,
                    self.NUM_NEWTON_ITERS,
                    self.NUM_RESIDUALS,
                    self.NUM_UNKNOWNS,
                )

            return tau_next, be_next, 0.0, convergence

        # def pull_to_ys():
        #     # https://github.com/patrick-kidger/optimistix/issues/132
        #     # Here we have safe values to avoid nan during compile time
        #     # safe_mask = lambda x, default: jnp.where(is_ep, x, default)
        #     q_tr_safe = jnp.maximum(q_tr, 1e-6)

        #     # using logarithmic critical state pressure for better numerical stability
        #     # u_p_tr=jnp.log(p_tr)

        #     def residuals(x_vec):
        #         pmulti = x_vec[0]
        #         q_next = q_tr - 3.0 * self.G * pmulti

        #         # res_yf = (q_next - self.M_csl * p_tr) / self.K
        #         res_yf = (q_next - self.M_csl * (p_tr + cohesion)) / self.K

        #         return jnp.atleast_1d(res_yf), q_next

        #     def step_fn(carry, _):

        #         R_func = lambda x: residuals(x)[0]

        #         # Forward mode AD for jacobian
        #         J = jax.jacfwd(R_func)(carry)
        #         R = R_func(carry)

        #         dx = jnp.linalg.solve(J + jnp.eye(J.shape[0]) * 1e-12, -R)
        #         dx = dx.reshape(carry.shape)

        #         x_new = carry + dx

        #         is_bad = jnp.any(jnp.isnan(x_new)) | jnp.any(jnp.isinf(x_new))
        #         x_new = jnp.where(is_bad, carry, x_new)

        #         convergence = None
        #         if self.debug_convergence:
        #             convergence = Convergence(
        #                 residuals=R,
        #                 unknowns=x_new,
        #                 jacobians=J,
        #             )
        #         return x_new, convergence

        #     x_init = jnp.array([0.0])

        #     # solve
        #     x_final, convergence = jax.lax.scan(
        #         step_fn, x_init, None, length=5, unroll=True
        #     )

        #     _, aux = residuals(x_final)
        #     pmulti_f = x_final

        #     q_next = aux

        #     s_next = s_tr * (q_next / jnp.maximum(q_tr, 1e-12))

        #     eps_e_next = (s_next / (2.0 * self.G)) + (p_tr / (3.0 * self.K))

        #     stress_principal = s_next + p_tr
        #     tau_next = (V * stress_principal) @ V.T

        #     be_next = (V * jnp.exp(-2.0 * eps_e_next)) @ V.T

        #     return tau_next, be_next, convergence

        stress_next, be_next, eps_v_pradhana, convergence = jax.lax.cond(
            is_tension,
            apex_projection,
            lambda: jax.lax.cond(is_ep, pull_to_ys, elastic_update),
        )

        return stress_next, be_next, eps_v_pradhana, convergence

    def get_dt_crit(
        self, mp_state: MaterialPointState, cell_size: float, alpha: float = 0.5
    ):
        """Critical timestep for stability based on P-wave speed."""
        rho_stack = mp_state.rho_stack

        c_p = jnp.sqrt((self.K + (4.0 / 3.0) * self.G) / rho_stack)

        vel_mag = jnp.linalg.norm(mp_state.velocity_stack, axis=1)
        max_speed = jnp.max(c_p + vel_mag)

        return (alpha * cell_size) / (max_speed + 1e-9)

        # p_next = jnp.exp(u_p)

        # factor = self.M_csl**2/ (self.M_csl**2 + 6.0 * self.G * pmulti)

        # q_next = q_tr_safe * factor

        # # deps_s_p = pmulti

        # # deps_s_p_dt = deps_s_p / dt

        # # I_p_next = get_plastic_inertial_number(
        # #     jnp.maximum(p_next, 1_000), deps_s_p_dt, self.d, self.rho_p
        # # )

        # # dI = I_p_next - I_p_prev
        # # I_p_next=0.0
        # # I_p_prev =0.0
        # # dI = I_p_next - I_p_prev

        # # dI = 0.0

        # # deps_p_v = - dI / self.I_v
        # # deps_p_v = 0.0
        # # eps_e_v = eps_e_v_tr - deps_p_v
        # eps_e_v = eps_e_v_tr

        # p_next_el = self.K * eps_e_v

        # yf_next = q_next - self.M_csl * p_next

        # R = jnp.array([yf_next, (jnp.log(p_next) - jnp.log(p_next_el)) ])

        # I_p_next =0.0
        # aux = (factor, I_p_next)
        # return R, aux
