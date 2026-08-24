# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-

"""Implementation of the regularized µ(I) rheology for dense granular flows."""

from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..material_points.material_points import MaterialPointState
from .constitutive_law import ConstitutiveLaw, ConstitutiveLawState


class MuIState(ConstitutiveLawState):
    """State for Mu(I) rheology.

    Stores the reference density to compute pressure via a linear Equation of State.
    """
    density_ref_stack: Float[Array, "num_points"]


class MuIIncompressibleState(ConstitutiveLawState):
    """Pressure Lagrange multiplier carried at material points.

    Pressure is solved by an incompressible MPM solver.  It is not computed
    from density or the deformation gradient by this constitutive law.
    """

    pressure_stack: Float[Array, "num_points"]


class MuI_Incompressible(ConstitutiveLaw):
    """Isochoric local µ(I) rheology for use with a pressure-projection solver.

    The constitutive part supplies only the pressure-dependent deviatoric
    response.  ``USLIncompressibleAFLIP`` supplies pressure as a Lagrange
    multiplier and enforces the discrete constraint ``div(v) = 0``.

    ``max_shear_viscosity`` and the explicit viscous CFL limit are numerical
    regularizations, not material parameters.  If either limit is active in a
    steady flow, the response is a capped Newtonian branch rather than the
    requested local mu(I) rheology, and the selected flow rate is cap-dependent.
    """

    mu_s: float | Float[Array, ""]
    mu_d: float | Float[Array, ""]
    I_0: float | Float[Array, ""]
    rho_p: float | Float[Array, ""]
    d_p: float | Float[Array, ""]
    p_min_calc: float | Float[Array, ""]
    cell_size: float | Float[Array, ""]
    viscosity_cfl: float | Float[Array, ""]
    max_shear_viscosity: Optional[float | Float[Array, ""]] = None

    def __init__(
        self,
        *,
        mu_s: float | Float[Array, ""],
        mu_d: float | Float[Array, ""],
        I_0: float | Float[Array, ""],
        d_p: float | Float[Array, ""],
        cell_size: float | Float[Array, ""],
        rho_p: float | Float[Array, ""] = 2650.0,
        p_min_calc: float | Float[Array, ""] = 0.0,
        viscosity_cfl: float | Float[Array, ""] = 0.125,
        max_shear_viscosity: Optional[float | Float[Array, ""]] = None,
        requires_F_reset: bool = True,
    ):
        self.mu_s = mu_s
        self.mu_d = mu_d
        self.I_0 = I_0
        self.d_p = d_p
        self.rho_p = rho_p
        self.p_min_calc = p_min_calc
        self.cell_size = cell_size
        self.viscosity_cfl = viscosity_cfl
        self.max_shear_viscosity = max_shear_viscosity
        self.requires_F_reset = requires_F_reset

    def create_state_from_pressure(
        self, pressure_stack: Float[Array, "num_points"]
    ) -> MuIIncompressibleState:
        return MuIIncompressibleState(pressure_stack=jnp.maximum(pressure_stack, 0.0))

    def create_state(
        self, material_points: MaterialPointState
    ) -> MuIIncompressibleState:
        return self.create_state_from_pressure(material_points.pressure_stack)

    def _kinematics(self, L):
        """Return deviatoric strain rate and its Jop et al. invariant."""
        D = 0.5 * (L + L.T)
        D_dev = D - (jnp.trace(D) / 3.0) * jnp.eye(3)
        shear_rate = jnp.sqrt(2.0 * jnp.sum(D_dev * D_dev))
        return D_dev, shear_rate

    def _viscosity_components(self, shear_rate, pressure):
        """Return the static-friction and finite dynamic viscosity terms."""
        pressure = jnp.maximum(pressure, 0.0)
        pressure_safe = jnp.maximum(pressure, self.p_min_calc)

        shear_rate_safe = jnp.maximum(
            shear_rate, jnp.finfo(shear_rate.dtype).tiny
        )
        eta_static = self.mu_s * pressure_safe / shear_rate_safe
        pressure_scale = self.I_0 * jnp.sqrt(pressure_safe / self.rho_p)
        eta_dynamic = (
            self.d_p * pressure_safe * (self.mu_d - self.mu_s)
        ) / (pressure_scale + self.d_p * shear_rate + 1.0e-30)
        return eta_static, eta_dynamic

    def _limit_viscosity(self, viscosity, density, dt):
        """Apply fixed and explicit-CFL viscosity safeguards."""
        dt_safe = jnp.maximum(dt, jnp.finfo(jnp.asarray(dt).dtype).tiny)
        viscosity_limit = (
            self.viscosity_cfl * density * self.cell_size**2 / dt_safe
        )
        if self.max_shear_viscosity is not None:
            viscosity = jnp.minimum(viscosity, self.max_shear_viscosity)
        return jnp.minimum(viscosity, viscosity_limit)

    def _update_stress(self, L, pressure, density, dt):
        """Return total compression-positive stress for one material point."""
        D_dev, shear_rate = self._kinematics(L)
        pressure = jnp.maximum(pressure, 0.0)
        eta_static, eta_dynamic = self._viscosity_components(
            shear_rate, pressure
        )
        viscosity = self._limit_viscosity(
            eta_static + eta_dynamic, density, dt
        )

        return pressure * jnp.eye(3) + 2.0 * viscosity * D_dev

    def update(self, material_points_state, law_state, dt):
        density_stack = (
            material_points_state.mass_stack / material_points_state.volume0_stack
        )
        stress_stack = jax.vmap(self._update_stress, in_axes=(0, 0, 0, None))(
            material_points_state.L_stack,
            law_state.pressure_stack,
            density_stack,
            dt,
        )
        material_points_state = eqx.tree_at(
            lambda state: state.stress_stack,
            material_points_state,
            stress_stack,
        )
        return material_points_state, law_state

    def get_dt_crit(self, mp_state, cell_size: float, alpha: float = 0.5):
        """Limit the step by advection and the explicit viscosity update."""
        del alpha
        speed = jnp.max(jnp.linalg.norm(mp_state.velocity_stack, axis=1))
        advective_dt = cell_size / (speed + 1.0e-9)
        if self.max_shear_viscosity is None:
            return advective_dt
        density = jnp.min(mp_state.mass_stack / mp_state.volume0_stack)
        viscous_dt = (
            self.viscosity_cfl * density * cell_size**2 / self.max_shear_viscosity
        )
        return jnp.minimum(advective_dt, viscous_dt)


class MuI_regularized(MuI_Incompressible):
    """Exponentially regularized incompressible local mu(I).

    Only the divergent static contribution ``mu_s p / shear_rate`` is
    regularized, exactly as in Eqs. (13)--(16) of Franci and Cremonesi (2019).
    The Papanastasiou form has the finite zero-shear limit
    ``mu_s p / regularization_rate``.

    ``regularization_rate`` is the paper's lambda in inverse seconds.  The
    inherited viscosity limits are separate safeguards required by this
    explicit MPM momentum update; they are not part of the cited model.
    """

    regularization_rate: float | Float[Array, ""]

    def __init__(
        self,
        *,
        mu_s: float | Float[Array, ""],
        mu_d: float | Float[Array, ""],
        I_0: float | Float[Array, ""],
        d_p: float | Float[Array, ""],
        cell_size: float | Float[Array, ""],
        rho_p: float | Float[Array, ""] = 2650.0,
        regularization_rate: float | Float[Array, ""] = 1.0e-2,
        p_min_calc: float | Float[Array, ""] = 0.0,
        viscosity_cfl: float | Float[Array, ""] = 0.125,
        requires_F_reset: bool = True,
    ):
        if regularization_rate <= 0.0:
            raise ValueError("regularization_rate must be positive")
        super().__init__(
            mu_s=mu_s,
            mu_d=mu_d,
            I_0=I_0,
            d_p=d_p,
            cell_size=cell_size,
            rho_p=rho_p,
            p_min_calc=p_min_calc,
            viscosity_cfl=viscosity_cfl,
            requires_F_reset=requires_F_reset,
        )
        self.regularization_rate = regularization_rate

    def _viscosity_components(self, shear_rate, pressure):
        pressure_safe = jnp.maximum(
            jnp.maximum(pressure, 0.0), self.p_min_calc
        )
        rate = self.regularization_rate
        ratio = jnp.where(
            shear_rate > 0.0,
            -jnp.expm1(-shear_rate / rate)
            / jnp.maximum(shear_rate, jnp.finfo(shear_rate.dtype).tiny),
            1.0 / rate,
        )
        eta_static = self.mu_s * pressure_safe * ratio

        pressure_scale = self.I_0 * jnp.sqrt(pressure_safe / self.rho_p)
        eta_dynamic = (
            self.d_p * pressure_safe * (self.mu_d - self.mu_s)
        ) / (pressure_scale + self.d_p * shear_rate + 1.0e-30)
        return eta_static, eta_dynamic


class MuI_LC(ConstitutiveLaw):
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
    I_0: float | Float[Array, ""]   # Inertial number constant
    K: float | Float[Array, ""]     # Bulk Modulus
    rho_p: float | Float[Array, ""] # Particle grain density
    d_p: float | Float[Array, ""]   # Mean particle diameter

    # Regularization parameters
    alpha: float | Float[Array, ""]
    alpha_sq: float | Float[Array, ""]

    # Stability parameters
    p_min_calc: float | Float[Array, ""] = 0.0

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
        requires_F_reset: bool = True,
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
            requires_F_reset: Whether to reset shear deformation gradient (True for flow models).
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
        self.requires_F_reset = requires_F_reset

    def create_state_from_density(
        self,
        density_stack: Float[Array, "num_points"],
        pressure_stack: Optional[Float[Array, "num_points"]] = None,
    ) -> MuIState:
        """
        Initializes state. Calculates reference density from current P and Rho
        assuming p = K * (rho/rho_ref - 1).
        """
        if pressure_stack is None:
            pressure_stack = jnp.zeros_like(density_stack)

        # Invert linear EOS: rho_ref = rho / (p/K + 1)
        density_ref_stack = density_stack / ((pressure_stack / self.K) + 1.0)
        jax.debug.print("Initial density range: [{min:.2f}, {max:.2f}]",
            min=jnp.min(density_ref_stack),
            max=jnp.max(density_ref_stack)
        )
        return MuIState(density_ref_stack=density_ref_stack)

    def create_state(
        self,
        material_points: MaterialPointState
    ) -> MuIState:
        """Creates state using initial material point configuration."""

        density_initial_stack = material_points.mass_stack / material_points.volume0_stack

        return self.create_state_from_density(
            density_stack=density_initial_stack,
            pressure_stack=material_points.pressure_stack
        )

    def _update_stress(
        self,
        L: Float[Array, "3 3"],
        mass,
        volume,
        density_ref,
        dt
    ) -> Float[Array, "3 3"]:
        """Calculates the Cauchy stress for a single particle."""

        deps_dt = 0.5 * (L + L.T)



        deps_v = jnp.trace(deps_dt)
        deps_dev = deps_dt - (deps_v / 3.0) * jnp.eye(3)


        dot_gamma = jnp.sqrt(2.0 * jnp.sum(deps_dev * deps_dev))


        current_density = mass / volume
        rho_ratio = current_density / density_ref

        # Linear EOS
        p = self.K * (rho_ratio - 1.0)

        is_connected = (rho_ratio > 1.0)

        def connected_update():

            p_safe = jnp.maximum(p, self.p_min_calc)

            eta_s = (self.mu_s * p_safe) / jnp.sqrt(dot_gamma**2 + self.alpha**2)

            delta_mu = self.mu_d - self.mu_s

            #  pressure confinement term
            pconf = self.I_0 * jnp.sqrt(p_safe / self.rho_p)


            eta_d = (self.d_p * p_safe * delta_mu) / (pconf + self.d_p * dot_gamma)

            eta_total = eta_s + eta_d


            stress = p_safe * jnp.eye(3) + 2.0 * eta_total * deps_dev
            return stress


        stress_next = jax.lax.cond(
            is_connected,
            connected_update,
            lambda: jnp.zeros((3, 3))
        )

        return stress_next

    def update(
        self,
        material_points_state: MaterialPointState,
        law_state: MuIState,
        dt: float | Float[Array, "..."]
    ) -> Tuple[MaterialPointState, MuIState]:

        # Vectorized update
        new_stress_stack = jax.vmap(
            self._update_stress,
            in_axes=(0, 0, 0, 0, None)
        )(
            material_points_state.L_stack,
            material_points_state.mass_stack,
            material_points_state.volume_stack,
            law_state.density_ref_stack,
            dt
        )

        new_mp_state = eqx.tree_at(
            lambda s: s.stress_stack,
            material_points_state,
            new_stress_stack
        )

        return new_mp_state, law_state

    def get_dt_crit(self, mp_state, cell_size: float, alpha: float = 0.5) -> Float[Array, ""]:
        """Critical timestep based on Bulk Modulus wave speed."""

        def particle_wave_speed(rho):
            # c = sqrt(K / rho)
            return jnp.sqrt(self.K / rho)

        rho_stack = mp_state.mass_stack / mp_state.volume_stack
        c_stack = jax.vmap(particle_wave_speed)(rho_stack)

        vel_mag_stack = jnp.linalg.norm(mp_state.velocity_stack, axis=1)

        max_wave_speed = jnp.max(c_stack + vel_mag_stack)

        return (alpha * cell_size) / (max_wave_speed + 1e-9)
