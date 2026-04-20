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

from .constitutive_law import ConstitutiveLaw, ConstitutiveLawState
from ..material_points.material_points import MaterialPointState
from ..utils.math_helpers import (
    get_dev_strain,
    get_J2,
    get_volumetric_strain,
    get_pressure,
    get_dev_stress,
    get_sym_tensor,
    get_spin_tensor,
    get_jaumann_increment,
)

class DruckerPragerState(ConstitutiveLawState):
    eps_e_stack: Float[Array, "num_points 3 3"]
    eps_p_acc_stack: Float[Array, "num_points"]
    p_0_stack: Float[Array, "num_points"]


class DruckerPrager(ConstitutiveLaw):
    """
    Non-associated Drucker-Prager model with linear hardening and 
    isotropic linear elasticity. Follows return mapping algorithm described in [1].
    

    - [1] de Souza Neto, Eduardo A., Djordje Peric, and David RJ Owen. Computational methods for plasticity: theory and applications. John Wiley & Sons, 2008.
    """

    K: float | Float[Array, ""]
    G: float | Float[Array, ""]
    mu_1: float | Float[Array, ""]
    mu_2: float | Float[Array, ""]
    c0: float | Float[Array, ""]
    mu_1_hat: float | Float[Array, ""]
    H: float | Float[Array, ""]
    rho_0: float | Float[Array, ""]

    def __init__(
        self,
        *,
        K: float | Float[Array, ""],
        nu: float | Float[Array, ""],
        mu_1: float | Float[Array, ""],
        mu_2: float | Float[Array, ""] = 0.0,
        c0: float | Float[Array, ""] = 0.0,
        mu_1_hat: float | Float[Array, ""] = 0.0,
        H: float | Float[Array, ""] = 0.0,
        rho_0: float | Float[Array, ""] = 1000.0,
        requires_F_reset: bool = False,
    ):
        self.K = K
        E = 3.0 * K * (1.0 - 2.0 * nu)
        self.G = E / (2.0 * (1.0 + nu))
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.c0 = c0
        self.mu_1_hat = mu_1_hat
        self.H = H
        self.rho_0 = rho_0
        self.requires_F_reset = requires_F_reset

    def create_state(self, stress_stack: Float[Array, "num_points 3 3"]) -> DruckerPragerState:
        """Initializes the material state."""
        num_points = stress_stack.shape[0]

        # Initial pressure from the material point state
        p_0_stack = jax.vmap(get_pressure)(stress_stack)
        
        return DruckerPragerState(
            eps_e_stack=jnp.zeros((num_points, 3, 3)),
            eps_p_acc_stack=jnp.zeros(num_points),
            p_0_stack=p_0_stack,
        )

    def update(
        self, 
        mp_state: MaterialPointState, 
        law_state: DruckerPragerState, 
        dt: float | Float[Array, ""]
    ) -> Tuple[MaterialPointState, DruckerPragerState]:
        """Vectorized update for the MPM solver."""
        
        new_stress, new_eps_e, new_eps_p_acc = jax.vmap(
            self._update_stress,
            in_axes=(0, 0, 0, 0, 0, 0, None)
        )(
            mp_state.L_stack,
            mp_state.stress_stack,
            law_state.eps_e_stack,
            law_state.eps_p_acc_stack,
            law_state.p_0_stack,
            mp_state.density_stack,
            dt
        )

        new_mp = eqx.tree_at(lambda m: m.stress_stack, mp_state, new_stress)
        new_law = eqx.tree_at(
            lambda l: (l.eps_e_stack, l.eps_p_acc_stack),
            law_state,
            (new_eps_e, new_eps_p_acc)
        )

        return new_mp, new_law

    def _update_stress(
        self, 
        L: Float[Array, "3 3"], 
        stress_prev: Float[Array, "3 3"],
        eps_e_prev: Float[Array, "3 3"], 
        eps_p_acc_prev: float | Float[Array, ""],
        p_0: float | Float[Array, ""], 
        rho: float | Float[Array, ""], 
        dt: float | Float[Array, ""]
    ):
        # kinematics
        D = get_sym_tensor(L)
        W = get_spin_tensor(L)
        deps = D * dt

        # objective rate correction (Jaumann)
        # Rotates previous tensors to current configuration
        eps_e_prev_rot = get_jaumann_increment(eps_e_prev, W, dt)
        
        # elastic Predictor
        eps_e_tr = eps_e_prev_rot + deps
        vol_e_tr = get_volumetric_strain(eps_e_tr)
        dev_e_tr = get_dev_strain(eps_e_tr, vol_e_tr)

        p_tr = (self.K * vol_e_tr) + p_0
        s_tr = 2.0 * self.G * dev_e_tr
        sqrt_J2_tr = jnp.sqrt(jnp.maximum(get_J2(dev_stress=s_tr), 1e-12))

        # linear hardening cohesion
        c = self.c0 + self.H * eps_p_acc_prev
        yf = sqrt_J2_tr - self.mu_1 * p_tr - self.mu_2 * c
        
        is_ep = yf > 0.0

        def elastic_update():
            stress = s_tr + p_tr * jnp.eye(3)
            return stress, eps_e_tr, eps_p_acc_prev

        def plastic_update():
            # return mapping for non-associated Drucker-Prager
        
            # Pull to Cone
            pmulti = yf / (self.G + self.K * self.mu_1 * self.mu_1_hat + self.H * self.mu_2 * self.mu_2)
            
            p_cone = p_tr - self.K * pmulti * self.mu_1_hat
            sqrt_J2_cone = sqrt_J2_tr - self.G * pmulti
            
            # If sqrt_J2_cone < 0, we hit the apex
            is_apex = sqrt_J2_cone < 0.0

            def pull_to_cone():
                s_cone = s_tr * (1.0 - (self.G * pmulti) / sqrt_J2_tr)
                eps_p_acc_cone = eps_p_acc_prev + self.mu_2 * pmulti
                stress_cone = s_cone + p_cone * jnp.eye(3)
                
                # Reconstruct elastic strain
                eps_e_v_cone = (p_cone - p_0) / self.K
                eps_e_d_cone = s_cone / (2.0 * self.G)
                eps_e_cone = eps_e_d_cone + (1.0 / 3.0) * eps_e_v_cone * jnp.eye(3)
                
                return stress_cone, eps_e_cone, eps_p_acc_cone

            def pull_to_apex():
                # at Apex: sqrt_J2 = 0 and  p = - (mu_2/mu_1) * c
                # This requires solving for volumetric plastic strain increment
                # or linear hardening: p_next = p_tr - K*deps_p_v
                alpha = self.mu_2 / jnp.maximum(self.mu_1, 1e-8)
                
                # Solve deps_p_v
                deps_p_v = (self.mu_1 * p_tr + self.mu_2 * (self.c0 + self.H * eps_p_acc_prev)) / jnp.maximum(self.K * self.mu_1 - self.H * self.mu_2 * alpha, 1e-12)
                
                p_apex = p_tr - self.K * deps_p_v
                eps_p_acc_apex = eps_p_acc_prev + alpha * deps_p_v
                
                stress_apex = +p_apex * jnp.eye(3)
                eps_e_v_apex = (p_apex - p_0) / self.K
                eps_e_apex = (1.0 / 3.0) * eps_e_v_apex * jnp.eye(3)
                
                return stress_apex, eps_e_apex, eps_p_acc_apex

            return jax.lax.cond(is_apex, pull_to_apex, pull_to_cone)

        # disconnection/vacuum check (stress-free assumption)
        stress_next, eps_e_next, eps_p_acc_next = jax.lax.cond(
            rho >= self.rho_0,
            lambda: jax.lax.cond(is_ep, plastic_update, elastic_update),
            lambda: (jnp.zeros((3, 3)), jnp.zeros((3, 3)), eps_p_acc_prev)
        )

        return stress_next, eps_e_next, eps_p_acc_next

    def get_dt_crit(self, mp_state: MaterialPointState, cell_size: float, alpha: float = 0.5):
        """Critical timestep for stability based on P-wave speed."""
        rho_stack = mp_state.rho_stack

        c_p = jnp.sqrt((self.K + (4.0 / 3.0) * self.G) / rho_stack)
        
        vel_mag = jnp.linalg.norm(mp_state.velocity_stack, axis=1)
        max_speed = jnp.max(c_p + vel_mag)
        
        return (alpha * cell_size) / (max_speed + 1e-9)