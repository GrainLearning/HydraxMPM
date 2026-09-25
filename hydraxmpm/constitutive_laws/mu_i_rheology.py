# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-

"""Implementation of µ(I) rheology with selectable static regularization."""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..material_points.material_points import MaterialPointState
from .constitutive_law import ConstitutiveLaw, ConstitutiveLawState

StaticRegularizationType = Literal["exp", "sr"]


def _static_regularization_exp(shear_rate, regularization_rate):
    """Return ``(1-exp(-gamma/rate))/gamma`` and its zero-rate limit.

    The shear-rate invariant is nonnegative. The condition below evaluates the
    regular expression for positive rates and supplies its analytical
    ``1 / regularization_rate`` limit exactly at zero; it is not a sign test for
    a signed shear component.
    """
    shear_rate_safe = jnp.maximum(shear_rate, jnp.finfo(shear_rate.dtype).tiny)
    return jnp.where(
        shear_rate > 0.0,
        -jnp.expm1(-shear_rate / regularization_rate) / shear_rate_safe,
        1.0 / regularization_rate,
    )


def _static_regularization_sr(shear_rate, alpha):
    """Return ``1 / sqrt(shear_rate**2 + alpha**2)``."""
    return 1.0 / jnp.sqrt(shear_rate**2 + alpha**2)


def _static_regularization_ratio(
    shear_rate,
    regularization_type: StaticRegularizationType,
    alpha,
    regularization_rate,
):
    """Dispatch static-viscosity regularization used in eta_static."""
    if regularization_type == "exp":
        return _static_regularization_exp(shear_rate, regularization_rate)
    if regularization_type == "sr":
        return _static_regularization_sr(shear_rate, alpha)
    raise ValueError("static_regularization must be one of {'exp', 'sr'}")


def _apply_viscosity_cfl(
    viscosity,
    density,
    dt,
    cell_size,
    viscosity_cfl,
):
    """Cap viscosity at the explicit diffusion limit for the current discretization."""
    dt_safe = jnp.maximum(dt, jnp.finfo(jnp.asarray(dt).dtype).tiny)
    viscosity_limit = viscosity_cfl * density * cell_size**2 / dt_safe
    return jnp.minimum(viscosity, viscosity_limit)


def _viscosity_components(
    shear_rate,
    pressure,
    *,
    mu_s,
    mu_d,
    I_0,
    d_p,
    rho_p,
    p_min_calc,
    static_regularization,
    alpha,
    regularization_rate,
):
    """Return the common static and dynamic apparent-viscosity contributions."""
    pressure_safe = jnp.maximum(jnp.maximum(pressure, 0.0), p_min_calc)
    ratio = _static_regularization_ratio(
        shear_rate,
        static_regularization,
        alpha,
        regularization_rate,
    )
    eta_static = mu_s * pressure_safe * ratio
    pressure_scale = I_0 * jnp.sqrt(pressure_safe / rho_p)
    eta_dynamic = (
        d_p
        * pressure_safe
        * (mu_d - mu_s)
        / (pressure_scale + d_p * shear_rate + 1.0e-30)
    )
    return eta_static, eta_dynamic


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


class MuI_IC(ConstitutiveLaw):
    """Isochoric local µ(I) rheology for use with a pressure-projection solver.

    The constitutive part supplies only the pressure-dependent deviatoric
    response.  ``USLIncompressibleAFLIP`` supplies pressure as a Lagrange
    multiplier and enforces the discrete constraint ``div(v) = 0``.

    ``static_regularization='exp'`` applies the exponential form
    ``(1 - exp(-shear_rate / regularization_rate)) / shear_rate``. The
    ``'sr'`` option applies the square-root form
    ``1 / sqrt(shear_rate**2 + alpha**2)``.

    The explicit viscous CFL limit is a numerical regularization, not a
    material parameter. If it is active in a steady flow, the response is a
    capped Newtonian branch rather than the requested local mu(I) rheology.
    """

    mu_s: float | Float[Array, ""]
    mu_d: float | Float[Array, ""]
    I_0: float | Float[Array, ""]
    rho_p: float | Float[Array, ""]
    d_p: float | Float[Array, ""]
    p_min_calc: float | Float[Array, ""]
    cell_size: float | Float[Array, ""]
    viscosity_cfl: float | Float[Array, ""]
    alpha: float | Float[Array, ""]
    regularization_rate: float | Float[Array, ""]
    static_regularization: StaticRegularizationType = eqx.field(static=True)

    def __init__(
        self,
        *,
        mu_s: float | Float[Array, ""],
        mu_d: float | Float[Array, ""],
        I_0: float | Float[Array, ""],
        d_p: float | Float[Array, ""],
        cell_size: float | Float[Array, ""],
        rho_p: float | Float[Array, ""] = 2650.0,
        alpha: float | Float[Array, ""] = 74.0,
        regularization_rate: float | Float[Array, ""] = 28.0,
        static_regularization: StaticRegularizationType = "exp",
        p_min_calc: float | Float[Array, ""] = 0.0,
        viscosity_cfl: float | Float[Array, ""] = 0.125,
        requires_F_reset: bool = True,
    ):
        self.mu_s = mu_s
        self.mu_d = mu_d
        self.I_0 = I_0
        self.d_p = d_p
        self.rho_p = rho_p
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if regularization_rate <= 0.0:
            raise ValueError("regularization_rate must be positive")
        if static_regularization not in ("exp", "sr"):
            raise ValueError("static_regularization must be one of {'exp', 'sr'}")
        self.alpha = alpha
        self.regularization_rate = regularization_rate
        self.static_regularization = static_regularization
        self.p_min_calc = p_min_calc
        self.cell_size = cell_size
        self.viscosity_cfl = viscosity_cfl
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
        return _viscosity_components(
            shear_rate,
            pressure,
            mu_s=self.mu_s,
            mu_d=self.mu_d,
            I_0=self.I_0,
            d_p=self.d_p,
            rho_p=self.rho_p,
            p_min_calc=self.p_min_calc,
            static_regularization=self.static_regularization,
            alpha=self.alpha,
            regularization_rate=self.regularization_rate,
        )

    def _update_stress(self, L, pressure, density, dt):
        """Return total compression-positive stress for one material point."""
        D_dev, shear_rate = self._kinematics(L)
        pressure = jnp.maximum(pressure, 0.0)
        eta_static, eta_dynamic = self._viscosity_components(shear_rate, pressure)
        viscosity = _apply_viscosity_cfl(
            eta_static + eta_dynamic,
            density,
            dt,
            self.cell_size,
            self.viscosity_cfl,
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
        """Return the advective limit; viscosity is capped."""
        del alpha
        speed = jnp.max(jnp.linalg.norm(mp_state.velocity_stack, axis=1))
        advective_dt = cell_size / (speed + 1.0e-9)
        return advective_dt


class MuI_IC_regularized(MuI_IC):
    """Compatibility name for ``MuI_IC(static_regularization='exp')``."""

    def __init__(
        self,
        *,
        mu_s: float | Float[Array, ""],
        mu_d: float | Float[Array, ""],
        I_0: float | Float[Array, ""],
        d_p: float | Float[Array, ""],
        cell_size: float | Float[Array, ""],
        rho_p: float | Float[Array, ""] = 2650.0,
        regularization_rate: float | Float[Array, ""] = 28.0,
        p_min_calc: float | Float[Array, ""] = 0.0,
        viscosity_cfl: float | Float[Array, ""] = 0.125,
        requires_F_reset: bool = True,
    ):
        super().__init__(
            mu_s=mu_s,
            mu_d=mu_d,
            I_0=I_0,
            d_p=d_p,
            cell_size=cell_size,
            rho_p=rho_p,
            regularization_rate=regularization_rate,
            static_regularization="exp",
            p_min_calc=p_min_calc,
            viscosity_cfl=viscosity_cfl,
            requires_F_reset=requires_F_reset,
        )


class MuI_LC(ConstitutiveLaw):
    """Linear-EOS compressible µ(I) rheology for dense granular flows.

    Based on:
    Jop et al. (2006) "A constitutive law for dense granular flows."

    Features:
    - Linear Equation of State (Bulk Modulus).
    - Selectable exponential or square-root regularization of the static term.
    - Tension cutoff (cohesionless).

    The exponential form follows Franci and Cremonesi (2019), "3D regularized
    μ(I)-rheology...", and has a finite zero-shear limit controlled by
    ``regularization_rate``. The square-root form is controlled by ``alpha``.
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
    regularization_rate: float | Float[Array, ""]
    static_regularization: StaticRegularizationType = eqx.field(static=True)

    # Stability parameters
    p_min_calc: float | Float[Array, ""] = 0.0
    cell_size: float | Float[Array, ""]
    viscosity_cfl: float | Float[Array, ""]

    def __init__(
        self,
        *,
        mu_s: float | Float[Array, ""],
        mu_d: float | Float[Array, ""],
        I_0: float | Float[Array, ""],
        d_p: float | Float[Array, ""],
        cell_size: float | Float[Array, ""] = 1.0,
        K: float | Float[Array, ""] = 1.0e6,
        rho_p: float | Float[Array, ""] = 2650.0,
        alpha: float | Float[Array, ""] = 74.0,
        regularization_rate: float | Float[Array, ""] = 28.0,
        static_regularization: StaticRegularizationType = "sr",
        p_min_calc: float | Float[Array, ""] = 0.0,
        viscosity_cfl: float | Float[Array, ""] = 0.125,
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
            alpha: Parameter for the square-root (``'sr'``) regularization.
            static_regularization: Static-term regularization model: ``'exp'``
                for the exponential form or ``'sr'`` for the square-root form.
            regularization_rate: Rate for the exponential (``'exp'``)
                regularization.
            p_min_calc: Minimum pressure used in denominator calculations for stability.
            viscosity_cfl: Explicit viscosity-CFL cap coefficient.
            requires_F_reset: Whether to reset shear deformation gradient for flow.
        """
        self.mu_s = mu_s
        self.mu_d = mu_d
        self.I_0 = I_0
        self.d_p = d_p
        self.K = K
        self.rho_p = rho_p
        self.cell_size = cell_size
        self.viscosity_cfl = viscosity_cfl
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if regularization_rate <= 0.0:
            raise ValueError("regularization_rate must be positive")
        if static_regularization not in ("exp", "sr"):
            raise ValueError("static_regularization must be one of {'exp', 'sr'}")
        self.alpha = alpha
        self.regularization_rate = regularization_rate
        self.static_regularization = static_regularization

        self.p_min_calc = p_min_calc
        self.requires_F_reset = requires_F_reset

    def _viscosity_components(self, shear_rate, pressure):
        return _viscosity_components(
            shear_rate,
            pressure,
            mu_s=self.mu_s,
            mu_d=self.mu_d,
            I_0=self.I_0,
            d_p=self.d_p,
            rho_p=self.rho_p,
            p_min_calc=self.p_min_calc,
            static_regularization=self.static_regularization,
            alpha=self.alpha,
            regularization_rate=self.regularization_rate,
        )

    def create_state_from_density(
        self,
        density_stack: Float[Array, "num_points"],
        pressure_stack: Float[Array, "num_points"] | None = None,
    ) -> MuIState:
        """
        Initializes state. Calculates reference density from current P and Rho
        assuming p = K * (rho/rho_ref - 1).
        """
        if pressure_stack is None:
            pressure_stack = jnp.zeros_like(density_stack)

        # Invert linear EOS: rho_ref = rho / (p/K + 1)
        density_ref_stack = density_stack / ((pressure_stack / self.K) + 1.0)
        return MuIState(density_ref_stack=density_ref_stack)

    def create_state(self, material_points: MaterialPointState) -> MuIState:
        """Creates state using initial material point configuration."""

        density_initial_stack = (
            material_points.mass_stack / material_points.volume0_stack
        )

        return self.create_state_from_density(
            density_stack=density_initial_stack,
            pressure_stack=material_points.pressure_stack,
        )

    def _update_stress(
        self, L: Float[Array, "3 3"], mass, volume, density_ref, dt
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

        is_connected = rho_ratio > 1.0

        def connected_update():
            p_safe = jnp.maximum(p, self.p_min_calc)

            eta_static, eta_dynamic = self._viscosity_components(dot_gamma, p_safe)
            viscosity = _apply_viscosity_cfl(
                eta_static + eta_dynamic,
                current_density,
                dt,
                self.cell_size,
                self.viscosity_cfl,
            )

            stress = p_safe * jnp.eye(3) + 2.0 * viscosity * deps_dev
            return stress

        stress_next = jax.lax.cond(
            is_connected, connected_update, lambda: jnp.zeros((3, 3))
        )

        return stress_next

    def update(
        self,
        material_points_state: MaterialPointState,
        law_state: MuIState,
        dt: float | Float[Array, "..."],
    ) -> tuple[MaterialPointState, MuIState]:
        # Vectorized update
        new_stress_stack = jax.vmap(self._update_stress, in_axes=(0, 0, 0, 0, None))(
            material_points_state.L_stack,
            material_points_state.mass_stack,
            material_points_state.volume_stack,
            law_state.density_ref_stack,
            dt,
        )

        new_mp_state = eqx.tree_at(
            lambda s: s.stress_stack, material_points_state, new_stress_stack
        )

        return new_mp_state, law_state

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
