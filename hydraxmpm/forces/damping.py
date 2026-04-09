# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-

from .force import Force, BaseForceState

from jaxtyping import Array, Float, Int, UInt, Bool

import jax
import jax.numpy as jnp


from typing import Self, Optional, Dict, Tuple, Callable, List, Any
import equinox as eqx


from ..material_points.material_points import BaseMaterialPointState

from ..grid.grid import GridDomain


class DampingState(BaseForceState):
    """State for the Cundall damping force"""
    alpha: Float[Array, ""] 

class Damping(Force):
    """
    Cundall (Local) Damping logic.
    F_damp = -alpha * |F_total| * sign(v)
    
    Damping is specifically useful for quasi-static simulations (e.g., soil settling).
    """


    is_apply_on_grid: bool = eqx.field(static=True, converter=lambda x: bool(x))
    
    # Connectivity
    p_idx_list: list[int] = eqx.field(static=True)
    g_idx_list: list[int] = eqx.field(static=True)
    f_idx: int = eqx.field(static=True, default=0)

    def __init__(
        self: Self,
        is_apply_on_grid: bool = True,
        p_idx_list: list[int] = None,
        g_idx_list: list[int] = None,
        f_idx=0,
    ) -> Self:
        if g_idx_list is None: g_idx_list = [0]
        if p_idx_list is None: p_idx_list = [0]

        self.is_apply_on_grid = is_apply_on_grid
        self.f_idx = f_idx
        self.g_idx_list = g_idx_list
        self.p_idx_list = p_idx_list


    def create_state(self: Self, alpha: float) -> DampingState:
        """Helper function to create the state for the damping force."""
        return DampingState(alpha=jnp.array(alpha))

    def apply_grid_forces(
        self, world, mechanics, sim_cache, sdf_logics, couplings, grid_domains, dt, time
    ):
        """
        Apply Cundall damping on grid forces.
        Note: This should ideally be called AFTER gravity and internal stress forces 
        are already in the force_stack.
        """
        grids = list(sim_cache.grids)
        f_states = list(mechanics.forces)

        if not self.is_apply_on_grid:
            return world, mechanics, sim_cache

        f_state = f_states[self.f_idx]

        for g_idx in self.g_idx_list:
            grid_cache = grids[g_idx]

            # The current force_stack contains (Internal Forces + Gravity)
            f_total = grid_cache.force_stack
            v_grid = grid_cache.moment_stack / (grid_cache.mass_stack[:, None] + 1e-22)  
            
            # F_damping -alpha * |F| * sign(v)
            damping_force = -f_state.alpha * jnp.abs(f_total) * jnp.sign(v_grid)
            

            
            # Add damping to the existing force stack
            new_force_stack = f_total + damping_force

            grids[g_idx] = eqx.tree_at(
                lambda g: g.force_stack, grid_cache, new_force_stack
            )

        sim_cache = eqx.tree_at(lambda s: s.grids, sim_cache, tuple(grids))
        return world, mechanics, sim_cache

    def apply_pre_p2g(
        self, world, mechanics, sim_cache, sdf_logics, couplings, grid_domains, dt, time
    ):
        """
        Apply damping on particles (if is_apply_on_grid is False).
        """
        mp_states = list(world.material_points)
        f_states = list(mechanics.forces)

        if self.is_apply_on_grid:
            return world, mechanics, sim_cache

        f_state = f_states[self.f_idx]

        for p_idx in self.p_idx_list:
            mp_state = mp_states[p_idx]
            
           
            f_total = mp_state.force_stack
            v_part = mp_state.velocity_stack 
            
            damping_force = -f_state.alpha * jnp.abs(f_total) * jnp.sign(v_part)
            
            mp_state = eqx.tree_at(
                lambda state: state.force_stack,
                mp_state,
                (f_total + damping_force),
            )
            mp_states[p_idx] = mp_state

        world = eqx.tree_at(lambda w: w.material_points, world, tuple(mp_states))
        return world, mechanics, sim_cache
