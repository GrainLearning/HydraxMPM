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


class LinearElasticHEState(ConstitutiveLawState):
    """
    State for Large Strain Linear Elasticity (Hyperelastic).
    We store the elastic Left Cauchy-Green tensor 'be'.
    """

    be_stack: Float[Array, "num_points 3 3"]


class LinearElasticHE(ConstitutiveLaw):
    """Isotropic linear elastic material solved in incremental form.

    Attributes:
        nu: Poisson's ratio.
        G: Shear modulus.
        K: Bulk modulus.

    """
    nu: float | Float[Array, ""]
    G: float | Float[Array, ""]
    K: float | Float[Array, ""]

    tension_cutoff: bool = eqx.field(static=True, default=True)
    debug_convergence: bool = eqx.field(static=True, default=False)

    def __init__(
        self: Self,
        K: float | Float[Array, ""],
        nu: float | Float[Array, ""],
        tension_cutoff: bool = True,
        debug_convergence: bool = eqx.field(static=True, default=False)
    ) -> Self:
        """Initialize the isotropic linear elastic material."""
        self.nu = nu
        self.K = K
        E = 3.0 * K * (1.0 - 2.0 * nu)
        self.G = E / (2.0 * (1.0 + nu))

        self.tension_cutoff = tension_cutoff
        debug_convergence: bool = eqx.field(static=True, default=False)

    def give_density_stack(
        self,
        p_stack,
        density0_stack,
    ):
        """Get reference density given current pressure
        and density
        """

        return density0_stack * jnp.exp(p_stack / self.K)

   
    def create_state(
        self,
        stress_stack: Float[Array, "num_points 3 3"],
    ) -> LinearElasticHEState:

        def invert_stress(tau):

            p_tau = jnp.trace(tau) / 3.0

            s_tau = tau - p_tau * jnp.eye(3)

            eps_v = p_tau / jnp.maximum(self.K, 1e-12)
            eps_d = s_tau / jnp.maximum(2.0 * self.G, 1e-12)

            # Total strain tensor
            eps_total = eps_d + (1.0 / 3.0) * eps_v * jnp.eye(3)

            return jax.scipy.linalg.expm(-2.0 * eps_total)


        be_initial = jax.vmap(invert_stress)(stress_stack)
        return LinearElasticHEState(be_stack=be_initial)


    def _update_stress(self, F_inc, be_prev, density,  dt):

        be_next = F_inc @ be_prev @ F_inc.T

        eigvals, V = jnp.linalg.eigh(be_next)
        eigvals = jnp.maximum(eigvals, 1e-12)  # Numerical stability

        # Hencky Strain (Compression Positive)
        eps_e = -0.5 * jnp.log(eigvals)
        eps_e_v = jnp.sum(eps_e)
        eps_e_q = eps_e - (eps_e_v / 3.0)

        # Stress Update
        p = self.K * eps_e_v

        s = 2.0 * self.G * eps_e_q
        stress_principal = s + p

        # Reconstruct Kirchhoff Stress Tensor & elastic strain
        tau = (V * stress_principal) @ V.T
        be_next = (V * jnp.exp(-2.0 * eps_e)) @ V.T

        # Handle tension cut-off (no negative pressure)
        if self.tension_cutoff:
            is_tension = p <= 1e-6
            return jax.lax.cond(
                is_tension,
                lambda: (jnp.zeros((3, 3)), jnp.eye(3)),
                lambda: (tau, be_next),
            )

        return tau, be_next

    def update(
        self,
        mp_state: MaterialPointState,
        law_state: LinearElasticHEState,
        dt: float | Float[Array, ""],
    ) -> Tuple[MaterialPointState, LinearElasticHEState]:

        new_stress, new_be = jax.vmap(self._update_stress, in_axes=(0, 0, 0, None))(
            mp_state.F_inc_stack,
            law_state.be_stack,
            mp_state.density_stack,
            dt,
        )

        new_mp = eqx.tree_at(lambda m: (m.stress_stack), mp_state, (new_stress))
        new_law = eqx.tree_at(
            lambda l: l.be_stack,
            law_state,
            new_be,
        )

        return new_mp, new_law

    def get_dt_crit(
        self: Self,
        material_points_state: MaterialPointState,
        cell_size: float,
        alpha: float = 0.5,
    ) -> Float[Array, ""]:
        """
        CFL condition based on P-wave speed.
        v_p = sqrt( (K + 4/3G) / density )
        """
        density = material_points_state.mass_stack / material_points_state.volume_stack

        # Dilational wave speed
        # constrained modulus M = K + 4/3G
        M = self.K + (4.0 / 3.0) * self.G
        c_dil = jnp.sqrt(M / density)

        # Particle velocity
        vel_mag = jnp.linalg.norm(material_points_state.velocity_stack, axis=1)

        max_signal_speed = jnp.max(c_dil + vel_mag)

        return (alpha * cell_size) / (max_signal_speed + 1e-9)
