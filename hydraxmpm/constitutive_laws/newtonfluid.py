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
    density_ref_stack: Float[Array, "num_points"]
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


    def __init__(
        self,
        K: float = 2e6, 
        viscosity: float = 1e-3, 
        beta: float = 7.0,
        requires_F_reset: bool = True
    ):
        """Initialize the nearly incompressible Newtonian fluid material."""

        self.K = K
        self.viscosity = viscosity
        self.beta = beta

    
        # inherited from base class for bvps
        self.requires_F_reset = requires_F_reset 


    def create_state_from_density(
        self,
        density_stack: Float[Array, "num_points"],
        pressure_stack: Optional[Float[Array, "num_points"]] = None,
    ) -> NewtonFluidState:
        """ initializes the constitutive state based on given density."""
        
        if pressure_stack is None:
            pressure_stack = jnp.zeros_like(density_stack)

        factor = (pressure_stack / self.K + 1.0) ** (1.0 / self.beta)

        density_ref_stack = density_stack / factor

        print(jnp.mean(density_ref_stack))
        return NewtonFluidState(density_ref_stack=density_ref_stack)
    
    def create_state(
        self,
        material_points: MaterialPointState
    ) -> NewtonFluidState:
        """defaults to using initial density to create state."""


        density_initial_stack = material_points.mass_stack/ material_points.volume0_stack
        
        return self.create_state_from_density(
            material_points.pressure_stack,
            density_stack=density_initial_stack
        )
 

    
    def _update_stress(
        self,
        L, 
        mass,
        volume,
        density_ref,
    ):
        """ Calculates Cauchy stress for a single material point."""

        # Strain rate symmetric part of velocity gradient L
        deps_dt = 0.5 * (L + L.T)

        # Get deviatoric part of strain rate via decomposition
        deps_dt_v = jnp.trace(deps_dt) # volumetric part
        dev_deps_dt = deps_dt - (1.0/3.0) * deps_dt_v * jnp.eye(3) # deviatoric part

        # Equation of State (EOS, Pressure)
        density = mass / volume

        # EOS
        # p = K * ((rho/rho_0)^beta - 1)
        ratio = density / density_ref

        # Prevent extreme negative pressures 
        ratio = jnp.maximum(ratio, 1e-6) 

        p = self.K * (ratio**self.beta - 1.0)

        # clip negative pressure if you want to disallow tension
        p = jnp.maximum(p, 0.0) 

        # Cauchy stress calculation
        # sigma = -pI + 2*mu*dev(D)
        stress = -p * jnp.eye(3) + (2.0 * self.viscosity * dev_deps_dt)


        return stress

    

    def update(
        self,
        material_points_state: MaterialPointState, 
        law_state: NewtonFluidState,
        dt
    ) -> Tuple[Any, NewtonFluidState]:

        # Vectorize the physics over all particles
        new_stress_stack = jax.vmap(self._update_stress, in_axes=(0,  0, 0, 0, ))(
            material_points_state.L_stack,
            material_points_state.mass_stack,
            material_points_state.volume_stack,
            law_state.density_ref_stack,
        )

        # Return updated particles and (unchanged) law state
        new_p_state = eqx.tree_at(
            lambda s: s.stress_stack,
            material_points_state,
            new_stress_stack
        )
        
        return new_p_state, law_state
    
    def get_dt_crit(self, mp_state, cell_size: float, alpha: float = 0.5) -> Float[Array, ""]:
        """Calculates CFL limit based on sound speed."""
        
        def particle_c(rho):
            return jnp.sqrt(self.K * self.beta / rho)

        rho = mp_state.mass_stack / mp_state.volume_stack
        c_sound = jax.vmap(particle_c)(rho)
        
        # Particle Velocity
        vel_mag = jnp.linalg.norm(mp_state.velocity_stack, axis=1)
        
        # Max signal speed
        max_signal = jnp.max(c_sound + vel_mag)
        
        # Avoid div zero
        return (alpha * cell_size) / (max_signal + 1e-9)
