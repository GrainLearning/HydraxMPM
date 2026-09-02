# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-

"""Implementation of the regularized µ(I) rheology for dense granular flows."""

import equinox as eqx
import jax
import jax.numpy as jnp

from .constitutive_law import ConstitutiveLawState, ConstitutiveLaw
from ..material_points.material_points import MaterialPointState

from jaxtyping import Float, Array
from typing import Tuple, Optional, Any


class MuIExplicitState(ConstitutiveLawState):
    """State for Mu(I) rheology.

    Stores the reference density to compute pressure via a linear Equation of State.
    """

    pass


class MuIExplicit(ConstitutiveLaw):
    """
    Incompressible µ(I) Rheology for dense granular flows.

    Based on:
    Jop et al. (2006) "A constitutive law for dense granular flows."

    With regularization from:
    Franci and Cremonesi (2019) "3D regularized μ(I)-rheology..."

    Features:
    - Linear Equation of State (Bulk Modulus).
    - Regularized viscosity to handle the static limit.
    - Tension cutoff (cohesionless).
    """

    # Material Constants
    mu_s: float | Float[Array, ""]  # Static friction coefficient
    mu_d: float | Float[Array, ""]  # Dynamic friction coefficient
    I_0: float | Float[Array, ""]  # Inertial number constant
    K: float | Float[Array, ""]  # Bulk Modulus
    rho_p: float | Float[Array, ""]  # Particle grain density
    d_p: float | Float[Array, ""]  # Mean particle diameter

    # Regularization parameters
    alpha: float | Float[Array, ""]
    alpha_sq: float | Float[Array, ""]

    # Stability parameters
    p_min_calc: float | Float[Array, ""] = 0.0

    debug_convergence: bool = eqx.field(static=True, default=False)

    def __init__(
        self,
        *,
        mu_s: float | Float[Array, ""],
        mu_d: float | Float[Array, ""],
        I_0: float | Float[Array, ""],
        d_p: float | Float[Array, ""],
        K: float | Float[Array, ""] = 1.0e6,
        rho_p: float | Float[Array, ""] = 2650.0,
        alpha: float | Float[Array, ""] = 1e-6,
        p_min_calc: float | Float[Array, ""] = 0.0,
        debug_convergence: bool = True,
    ):
        """
        Initialize the µ(I) Rheology model.

        Args:
            mu_s: Static friction coefficient.
            mu_d: Dynamic friction coefficient.
            I_0: Reference Inertial number (material constant).
            d_p: Particle diameter (needed for Inertial number calculation).
            K: Bulk Modulus for pressure calculation.
            rho_p: Density of the grains (not the bulk).
            alpha: Regularization parameter for low shear rates.
            p_min_calc: Minimum pressure used in denominator calculations for stability.
        """
        self.mu_s = mu_s
        self.mu_d = mu_d
        self.I_0 = I_0
        self.d_p = d_p
        self.K = K
        self.rho_p = rho_p

        self.alpha = alpha
        self.alpha_sq = alpha * alpha

        self.p_min_calc = p_min_calc

        self.debug_convergence = debug_convergence

    def create_state(self, mp_state: Optional[MaterialPointState] = None) -> MuIExplicitState:
        return MuIExplicitState()

    def give_v_0_stack(self, density0_stack):
        """Calculate reference specific volume v_0 = rho_p / rho_0."""
        return self.rho_p / density0_stack

    def give_density_stack(
        self,
        p_stack,
        density0_stack,
    ):
        """Get reference density given current pressure
        and density
        """

        # specific volume at reference pressure

        v_0 = self.give_v_0_stack(density0_stack)

        # specific volume at current pressure
        v = v_0 * jnp.exp(-p_stack / self.K)

        return self.rho_p / v

    def _update_stress(self, F_inc, rho, density0, dt):

        # velocity gradient
        # Compression is positive
        L = (jnp.eye(3) - F_inc) / dt

        deps_dt = 0.5 * (L + L.T)

        deps_v = jnp.trace(deps_dt)

        deps_dev = deps_dt - (deps_v / 3.0) * jnp.eye(3)

        dot_gamma = jnp.sqrt(jnp.maximum(2.0 * jnp.sum(deps_dev * deps_dev), 1e-16))

        # Hencky EOS: p = K * log(rho/density0)
        p = self.K * jnp.log(jnp.maximum(rho / density0, 1e-17))

        J = density0 / jnp.maximum(rho, 1e-9)

        def connected_update():

            p_safe = jnp.maximum(p, self.p_min_calc)

            eta_s = (self.mu_s * p_safe) / jnp.sqrt(dot_gamma**2 + self.alpha**2)

            delta_mu = self.mu_d - self.mu_s

            # #  pressure confinement term
            pconf = self.I_0 * jnp.sqrt(p_safe / self.rho_p)

            eta_d = (self.d_p * p_safe * delta_mu) / (pconf + self.d_p * dot_gamma)

            eta_total = eta_s + eta_d

            stress = p_safe * jnp.eye(3) + 2.0 * eta_total * deps_dev
            # convert to Kirchhoff stress
            return J * stress

        is_tension = p < 1e-12

        stress_next = jax.lax.cond(
            is_tension, lambda: jnp.zeros((3, 3)), connected_update
        )

        return stress_next

    def update(
        self, mp_state: MaterialPointState, law_state: MuIExplicitState, dt: float
    ) -> Tuple[MaterialPointState, MuIExplicitState]:

        new_stress_stack = jax.vmap(self._update_stress, in_axes=(0, 0, 0, None))(
            mp_state.F_inc_stack, mp_state.density_stack, mp_state.density0_stack, dt
        )

        new_mp = eqx.tree_at(lambda s: s.stress_stack, mp_state, new_stress_stack)

        return new_mp, law_state

    def get_dt_crit(
        self, mp_state, cell_size: float, alpha: float = 0.5
    ) -> Float[Array, ""]:
        """Critical timestep based on Bulk Modulus wave speed."""

        def particle_wave_speed(rho):
            # c = sqrt(K / rho)
            return jnp.sqrt(self.K / rho)

        rho_stack = mp_state.mass_stack / mp_state.volume_stack
        c_stack = jax.vmap(particle_wave_speed)(rho_stack)

        vel_mag_stack = jnp.linalg.norm(mp_state.velocity_stack, axis=1)

        max_wave_speed = jnp.max(c_stack + vel_mag_stack)

        return (alpha * cell_size) / (max_wave_speed + 1e-9)
