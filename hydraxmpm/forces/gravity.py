# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-

"""
    Explanation:
            This module contains the gravity forces.
            
            Gravity is stores as a force state so it can be updated.

            e.g., by gravity packing or tilting a plane

            Gravity can be applied to particles or background grid nodes
"""



from .force import Force, BaseForceState

from jaxtyping import Array, Float, Int, UInt, Bool

import jax
import jax.numpy as jnp


from typing import Self, Optional, Dict, Tuple, Callable, List, Any
import equinox as eqx


from ..material_points.material_points import BaseMaterialPointState

from ..grid.grid import GridState


class GravityState(BaseForceState):
    """State for the gravity force"""
    gravity: Float[Array, "dim"]



class Gravity(Force):
    """
    Gravity force logic.

    Attributes:
        callback: Optional function to modify gravity application
        is_apply_on_grid: If True, apply gravity on grid nodes; else on particles
        p_idx_list: List of particle set indices to apply gravity on
        g_idx_list: List of grid indices to apply gravity on
        f_idx: Index of the gravity force state in the force states
        
    """

    # Optional callback for custom gravity application
    callback: Optional[Callable] = eqx.field(static=True)
    
    # control
    is_apply_on_grid: bool = eqx.field(static=True, converter=lambda x: bool(x))
    
    # Connectivity
    p_idx_list: list[int] = eqx.field(static=True)
    g_idx_list: list[int] = eqx.field(static=True)
    f_idx : int = eqx.field(static=True, default =0)


    def __init__(
        self: Self,
        is_apply_on_grid: Optional[bool] = False,
        callback: Optional[Callable] = None,
        p_idx_list: list[int] = None,
        g_idx_list: list[int] = None,
        f_idx = 0
    ) -> Self:
        """Initialize Gravity force."""

        if g_idx_list is None:
            g_idx_list = [0] # select only first grid

        if p_idx_list is None:
            p_idx_list = [0] # select only first particle set

        self.is_apply_on_grid = is_apply_on_grid
        self.callback = callback
        self.f_idx = f_idx
        self.g_idx_list = g_idx_list
        self.p_idx_list = p_idx_list

    def create_state(self: Self, gravity: Float[Array, "dim"]) -> GravityState:
        """ Helper function to create the state for the gravity force."""
        return GravityState(gravity=gravity)

    def apply_grid_forces(
            self, 
            mp_states, 
            grid_states, 
            f_states, 
            intr_caches, 
            couplings,
            dt, 
            time
            ):
        """
        Apply gravity on grid forces
        """
    
        if not self.is_apply_on_grid:
            return grid_states, f_states
        
        f_state = f_states[self.f_idx]
        
        # Loop over all grid ids
        for g_idx in self.g_idx_list:

            grid_state = grid_states[g_idx]

            if self.callback:
                f_state, grid_state = self.callback(self, f_state, grid_state, mp_states,  dt, time, g_idx)

            # F = m * g
            # Add to existing grid force directly
            gravity_impulse = grid_state.mass_stack[:,None] * f_state.gravity

            new_force_stack = grid_state.force_stack + gravity_impulse
            
            grid_states[g_idx] =  eqx.tree_at(lambda g: g.force_stack, grid_state, new_force_stack)
            

        f_states[self.f_idx] = f_state

        return grid_states, f_states

    def apply_pre_p2g(
            self, 
            mp_states, 
            grid_states, 
            f_states, 
            intr_caches, 
            couplings,
            dt, 
            time
            ):
        """
        Apply gravity on particle forces
        """
        # Same logic as above but applied to particles
        if self.is_apply_on_grid:
            return mp_states, f_states
        
        f_state = f_states[self.f_idx]

        # Loop over all particle indices
        for p_idx in self.p_idx_list:

            mp_state = mp_states[p_idx]

            if self.callback:
                f_state, mp_state  = self.callback(self, f_state, grid_states, mp_state,  dt, time, p_idx)

            gravity_impulse = mp_state.mass_stack[:,None]  * f_state.gravity


            mp_state = eqx.tree_at(
                lambda state: (state.force_stack),
                mp_state,
                (mp_state.force_stack + gravity_impulse),
            )

            mp_states[self.p_idx] = mp_state

        f_states[self.f_idx] = f_state

        return mp_states, f_states
