# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-

"""Constitutive model for a nearly incompressible Newtonian fluid."""

import equinox as eqx
import jax
import jax.numpy as jnp

from .constitutive_law import ConstitutiveLawState
from .constitutive_law import ConstitutiveLaw

from ..material_points.material_points import MaterialPointState

from jaxtyping import Float, Array

from typing import Any, Tuple, Optional


class NewtonFluidState(ConstitutiveLawState):
    pass


class NewtonFluid(ConstitutiveLaw):
    """Nearly incompressible Newtonian fluid.
    Attributes:
        K: Bulk modulus.
        viscosity: Viscosity.
        gamma: Exponent.
    """

    K: float
    viscosity: float
    beta: float
    debug_convergence: bool = eqx.field(static=True, default=False)

    def __init__(
        self,
        K: float = 2e6,
        viscosity: float = 1e-3,
        beta: float = 7.0,
    ):
        """Initialize the nearly incompressible Newtonian fluid material."""

        self.K = K
        self.viscosity = viscosity
        self.beta = beta

    def create_state(self, mp_state: Optional[MaterialPointState] = None) -> NewtonFluidState:
        return NewtonFluidState()

    def give_density_stack(
        self,
        p_stack,
        density0_stack,
    ):
        """Get reference density given current pressure
        and density
        """

        ratio_inv = jnp.maximum(p_stack / self.K + 1.0, 1e-9)

        return density0_stack * (ratio_inv ** (1.0 / self.beta))

    def _update_stress(self, F_inc, density, density0, dt):
        """Calculates Cauchy stress for a single material point."""

        # Strain rate symmetric part of velocity gradient L
        L = (jnp.eye(3) - F_inc) / dt

        deps_dt = 0.5 * (L + L.T)

        deps_v = jnp.trace(deps_dt)

        deps_dev = deps_dt - (deps_v / 3.0) * jnp.eye(3)

        # EOS
        ratio = density / density0
        ratio = jnp.maximum(ratio, 1e-6)  # Stability
        p = self.K * (ratio**self.beta - 1.0)

        # clip negative pressure to disallow tension
        p = jnp.maximum(p, 0.0)

        # Cauchy stress calculation
        stress = p * jnp.eye(3) + (2.0 * self.viscosity * deps_dev)
        J = density0 / jnp.maximum(density, 1e-9)

        return J * stress

    def update(
        self, mp_state: MaterialPointState, law_state: NewtonFluidState, dt: float
    ) -> Tuple[MaterialPointState, NewtonFluidState]:

        new_stress_stack = jax.vmap(self._update_stress, in_axes=(0, 0, 0,None))(
            mp_state.F_inc_stack,
            mp_state.density_stack,
            mp_state.density0_stack,
            dt
        )

        new_mp = eqx.tree_at(lambda s: s.stress_stack, mp_state, new_stress_stack)

        return new_mp, law_state

    def get_dt_crit(
        self, mp_state, cell_size: float, alpha: float = 0.5
    ) -> Float[Array, ""]:
        """Calculates CFL limit based on sound speed."""

        def particle_c(density):
            return jnp.sqrt(self.K * self.beta / density)

        density = mp_state.mass_stack / mp_state.volume_stack
        c_sound = jax.vmap(particle_c)(density)

        # Particle Velocity
        vel_mag = jnp.linalg.norm(mp_state.velocity_stack, axis=1)

        # Max signal speed
        max_signal = jnp.max(c_sound + vel_mag)

        # Avoid div zero
        return (alpha * cell_size) / (max_signal + 1e-9)
