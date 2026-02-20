# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-

import equinox as eqx
import jax
import jax.numpy as jnp

from .constitutive_law import ConstitutiveLawState
from .constitutive_law import ConstitutiveLaw

from ..material_points.material_points import MaterialPointState

from jaxtyping import Float, Array

from typing import Any, Tuple, Optional, Self


def get_bulk_modulus(E, nu):
    return E / (3.0 * (1.0 - 2.0 * nu))


def get_shear_modulus(E, nu):
    return E / (2.0 * (1.0 + nu))


def get_lame_modulus(E, nu):
    return E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


class LinearElasticState(ConstitutiveLawState):
    """
    State for Linear Elasticity.

    We store 'stress_ref' here.
    In an incremental formulation, this usually represents the Initial Stress (sigma_0)
    (e.g., geostatic stress) that the material started with.
    """

    stress_ref_stack: Float[Array, "num_points 3 3"]
    pass


class LinearElasticLaw(ConstitutiveLaw):
    """Isotropic linear elastic material solved in incremental form.

    Attributes:
        E: Young's modulus.
        nu: Poisson's ratio.
        G: Shear modulus.
        K: Bulk modulus.
        lam: Lame modulus.
    """

    E: float | Float[Array, ""]
    nu: float | Float[Array, ""]
    G: float | Float[Array, ""]
    K: float | Float[Array, ""]
    lam: float | Float[Array, ""]

    def __init__(
        self: Self,
        E: float | Float[Array, ""],
        nu: float | Float[Array, ""],
        requires_F_reset: bool = True,
    ) -> Self:
        """Initialize the isotropic linear elastic material."""

        self.E = E

        self.nu = nu

        self.K = get_bulk_modulus(E, nu)
        self.G = get_shear_modulus(E, nu)
        self.lam = get_lame_modulus(E, nu)

        self.requires_F_reset = requires_F_reset

    def create_state(
        self,
        material_points: MaterialPointState = None,
        stress_ref_stack: Optional[Float[Array, "num_points 3 3"]] = None,
        density_stack: Optional[Float[Array, "num_points"]] = None,
    ) -> LinearElasticState:
        """Create the constitutive law state for the given material points."""

        if material_points is not None:
            stress_ref_stack = material_points.stress_stack
        elif density_stack is not None:
            num_points = density_stack.shape[0]
            stress_ref_stack = jnp.zeros((num_points, 3, 3))

        return LinearElasticState(
            stress_ref_stack=stress_ref_stack,
        )

    def _update_stress(
        self,
        L: Float[Array, "3 3"],  # Velocity Gradient
        stress_prev: Float[Array, "3 3"],
        dt,
    ) -> Float[Array, "3 3"]:

        # Strain rate symmetric part of velocity gradient L
        deps_dt = 0.5 * (L + L.T)

        deps = deps_dt * dt

        return (
            stress_prev + self.lam * jnp.trace(deps) * jnp.eye(3) + 2.0 * self.G * deps
        )

    def update(
        self, material_points_state: MaterialPointState, law_state, dt
    ) -> Tuple[MaterialPointState, Any]:

        new_stress_stack = jax.vmap(self._update_stress, in_axes=(0, 0, None))(
            material_points_state.L_stack, material_points_state.stress_stack, dt
        )

        new_material_points_state = eqx.tree_at(
            lambda s: s.stress_stack, material_points_state, new_stress_stack
        )

        return new_material_points_state, law_state

    def get_dt_crit(
        self: Self,
        material_points_state: MaterialPointState,
        cell_size: float,
        alpha: float = 0.5,
    ) -> Float[Array, ""]:
        """
        CFL condition based on P-wave speed.
        v_p = sqrt( (K + 4/3G) / rho )
        """
        rho = material_points_state.mass_stack / material_points_state.volume_stack

        # Dilational wave speed
        # constrained modulus M = K + 4/3G
        M = self.K + (4.0 / 3.0) * self.G
        c_dil = jnp.sqrt(M / rho)

        # Particle velocity
        vel_mag = jnp.linalg.norm(material_points_state.velocity_stack, axis=1)

        max_signal_speed = jnp.max(c_dil + vel_mag)

        return (alpha * cell_size) / (max_signal_speed + 1e-9)
