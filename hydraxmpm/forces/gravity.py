# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-

"""
Gravity force module for MPM solvers.

This module defines the Gravity class, which applies gravitational forces to either the background grid (Eulerian) or directly to material points (Lagrangian) in a Material Point Method (MPM) simulation. Gravity can be linearly ramped up over a number of steps, which is useful for gradually introducing body forces and avoiding numerical instabilities at the start of a simulation.


"""

from typing import List, Optional, Tuple, Any

import equinox as eqx
import jax
import jax.numpy as jnp
from typing_extensions import Self

from ..common.types import TypeFloat, TypeFloatVector, TypeInt
from ..grid.grid import Grid
from ..material_points.material_points import MaterialPoints
from .force import Force


class Gravity(Force):
    """
    Gravity force for MPM simulations.

    This class applies gravity either to the background grid (nodes) or directly to the material points (particles), depending on the 'particle_gravity' flag. Gravity can be linearly ramped up over a number of steps using the 'increment' and 'stop_ramp_step' parameters.

    Linear ramping: If 'increment' is set, gravity is increased by this value each step, up to 'stop_ramp_step'. This helps avoid sudden force application and improves numerical stability.
    """

    gravity: TypeFloatVector
    increment: Optional[TypeFloatVector]
    stop_ramp_step: Optional[TypeInt]
    particle_gravity: bool = eqx.field(static=True, converter=lambda x: bool(x))
    dt: TypeFloat = eqx.field(static=True)

    def __init__(
        self: Self,
        gravity: TypeFloatVector | List | Tuple,
        increment: Optional[TypeFloatVector | List | Tuple] = None,
        stop_ramp_step: Optional[TypeInt] = 0,
        particle_gravity: Optional[bool] = True,
        **kwargs,
    ) -> Self:
        """
        Initialize the Gravity force.

        Args:
            gravity: Gravity vector (e.g., [0, -9.81, 0]).
            increment: Optional vector for ramping gravity over steps.
            stop_ramp_step: Step at which ramping stops.
            particle_gravity: If True, applies gravity to particles; else to grid.
            **kwargs: Additional arguments (e.g., dt).
        """
        self.gravity = jnp.array(gravity)
        self.increment = jnp.zeros_like(self.gravity) if increment is None else increment
        self.stop_ramp_step = stop_ramp_step
        self.particle_gravity = particle_gravity
        self.dt = kwargs.get("dt", 0.001)

    def apply_on_grid(
        self: Self,
        material_points: Optional[MaterialPoints] = None,
        grid: Optional[Grid] = None,
        step: Optional[TypeInt] = 0,
        dt: TypeFloat = 0.01,
        dim: TypeInt = 3,
        **kwargs: Any,
    ) -> Tuple[Grid, Self]:
        """
        Apply gravity to the grid nodes (Eulerian approach).

        Args:
            material_points: Not used here.
            grid: The grid to apply gravity to.
            step: Current simulation step (for ramping).
            dt: Time step size.
            dim: Problem dimension (unused).
            **kwargs: Additional arguments.

        Returns:
            Updated grid and (possibly updated) Gravity instance.
        """
        if self.particle_gravity:
            # Gravity is applied to particles, not grid
            return grid, self

        # Compute ramped gravity if increment is set
        if self.increment is not None:
            gravity = self.gravity + self.increment * jnp.minimum(step, self.stop_ramp_step)
        else:
            gravity = self.gravity

        # Compute gravity-induced moment for each node
        moment_gravity = grid.mass_stack.reshape(-1, 1) * gravity * dt
        new_moment_nt_stack = grid.moment_nt_stack + moment_gravity

        # Update grid with new moments (non-thermal)
        new_grid = eqx.tree_at(
            lambda state: (state.moment_nt_stack),
            grid,
            (new_moment_nt_stack),
        )
        return new_grid, self

    def apply_on_points(
        self: Self,
        material_points: Optional[MaterialPoints] = None,
        grid: Optional[Grid] = None,
        step: Optional[TypeInt] = 0,
        dt: Optional[TypeFloat] = 0.01,
        dim: TypeInt = 3,
    ) -> Tuple[MaterialPoints, Self]:
        """
        Apply gravity directly to material points (Lagrangian approach).

        Args:
            material_points: The particles to apply gravity to.
            grid: Not used here.
            step: Current simulation step (for ramping).
            dt: Time step size (unused).
            dim: Problem dimension (unused).

        Returns:
            Updated material points and (possibly updated) Gravity instance.
        """
        if not self.particle_gravity:
            # Gravity is applied to grid, not particles
            return material_points, self

        # Compute ramped gravity if increment is set
        if self.increment is not None:
            gravity = self.gravity + self.increment * jnp.minimum(step, self.stop_ramp_step)
        else:
            gravity = self.gravity

        def get_gravitational_force(mass: TypeFloat):
            return mass * gravity

        # Update force_stack for each particle
        new_particles = eqx.tree_at(
            lambda state: (state.force_stack),
            material_points,
            (jax.vmap(get_gravitational_force)(material_points.mass_stack)),
        )
        return new_particles, self
