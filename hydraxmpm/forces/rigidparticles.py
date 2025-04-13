# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-

"""Module for imposing zero/non-zero boundaries via rigid particles."""

from functools import partial
from typing import Callable, Tuple, Optional, Self, Any

import jax
import jax.numpy as jnp
from jax import Array

from ..grid.grid import Grid
from ..material_points.material_points import MaterialPoints
from .force import Force

import equinox as eqx

from ..common.types import (
    TypeFloat,
    TypeInt,
    # TypeUIntScalarAStack,
    TypeFloatVectorAStack,
    TypeFloatVector,
)

from ..shapefunctions.mapping import ShapeFunctionMapping


class RigidParticles(Force):
    """Shapes are discretized into rigid particles to impose boundary conditions.


    Correction to Bardenhagen's contact algorithm presented by

    L. Gao, et. al, 2022, MPM modeling of pile installation in sand - Computers and geotechniques


    The rigid particles are used to impose boundary conditions on the grid.

    """

    position_stack: TypeFloatVectorAStack
    velocity_stack: TypeFloatVectorAStack

    com: Optional[TypeFloatVector] = None

    mu: TypeFloat

    gap_factor: TypeFloat

    update_rigid_particles: Optional[Callable] = eqx.field(static=True)

    shape_map: ShapeFunctionMapping

    def __init__(
        self: Self,
        position_stack: TypeFloatVectorAStack,
        velocity_stack: TypeFloatVectorAStack = None,
        mu: TypeFloat = 0.0,
        com: Optional[TypeFloatVector] = None,
        gap_factor: Optional[TypeFloat] = 1.0,
        update_rigid_particles: Optional[Callable] = None,
        **kwargs,
    ) -> Self:
        """Initialize the rigid particles."""

        if velocity_stack is None:
            velocity_stack = jnp.zeros_like(position_stack)

        self.position_stack = position_stack

        self.velocity_stack = velocity_stack

        self.mu = mu

        self.update_rigid_particles = update_rigid_particles

        self.com = com

        self.gap_factor = gap_factor

        num_points, dim = position_stack.shape

        self.shape_map = ShapeFunctionMapping(
            shapefunction=kwargs.get("shapefunction", "cubic"),
            dim=dim,
            num_points=num_points,
        )

        super().__init__(**kwargs)

    def apply_on_grid(
        self: Self,
        material_points: Optional[MaterialPoints] = None,
        grid: Optional[Grid] = None,
        step: Optional[TypeInt] = 0,
        dt: Optional[TypeFloat] = 0.01,
        dim: TypeInt = 3,
        shape_map: Optional[ShapeFunctionMapping] = None,
        **kwargs: Any,
    ):
        """Apply the boundary conditions on the nodes moments.

        #         Procedure:
        #             - Get the normals of the non-rigid particles on the grid.
        #             - Get the velocities on the grid due to the velocities of the
        #                 rigid particles.
        #             - Get contacting nodes and apply the velocities on the grid.
        #"""

        def vmap_velocities_p2g_rigid(
            point_id, intr_shapef, intr_shapef_grad, intr_dist
        ):
            intr_velocities = self.velocity_stack.at[point_id].get()
            r_scaled_velocity = intr_shapef * intr_velocities
            return r_scaled_velocity

        new_r_shape_map, r_scaled_velocity_stack = (
            self.shape_map.vmap_interactions_and_scatter(
                vmap_velocities_p2g_rigid, position_stack=self.position_stack, grid=grid
            )
        )

        r_nodes_vel_stack = (
            jnp.zeros_like(grid.moment_nt_stack)
            .at[new_r_shape_map._intr_hash_stack]
            .add(r_scaled_velocity_stack)
        )

        mp_position_intrp_stack = shape_map.map_p2g(
            X_stack=material_points.position_stack,
            mass_stack=material_points.mass_stack,
            grid=grid,
        )

        rp_position_intrp_stack = new_r_shape_map.map_p2g(
            self.position_stack, jnp.ones(dim), grid
        )

        @partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, 0))
        def vmap_nodes(moment_nt, mass, normal, r_vel, mp_pos, rp_pos):
            """Apply the velocities on the grid from the rigid particles."""
            # skip the nodes with small mass, due to numerical instability

            normal = normal / (jnp.linalg.vector_norm(normal) + 1e-10)

            rel_pos = jnp.dot(mp_pos - rp_pos, normal)

            r_contact_mask = rel_pos + self.gap_factor * grid.cell_size <= 0

            vel_nt = moment_nt / (mass + 1e-10)

            delta_vel = vel_nt - r_vel

            delta_vel_dot_normal = jnp.dot(delta_vel, normal)

            delta_vel_padded = jnp.pad(
                delta_vel,
                new_r_shape_map._padding,
                mode="constant",
                constant_values=0,
            )

            norm_padded = jnp.pad(
                normal,
                new_r_shape_map._padding,
                mode="constant",
                constant_values=0,
            )

            delta_vel_cross_normal = jnp.cross(
                delta_vel_padded, norm_padded
            )  # works only for vectors of len 3
            norm_delta_vel_cross_normal = jnp.linalg.vector_norm(delta_vel_cross_normal)
            # Add epsilon

            omega = delta_vel_cross_normal / (norm_delta_vel_cross_normal + 1e-10)
            mu_prime = jnp.minimum(
                self.mu, norm_delta_vel_cross_normal / (delta_vel_dot_normal + 1e-10)
            )

            normal_cross_omega = jnp.cross(
                norm_padded, omega
            )  # works only for vectors of len 3

            tangent = (
                (norm_padded + mu_prime * normal_cross_omega)
                .at[: new_r_shape_map.dim]
                .get()
            )

            # sometimes tangent become nan if velocity is zero at initialization
            # which causes problems
            tangent = jnp.nan_to_num(tangent)

            new_nodes_vel_nt = jax.lax.cond(
                ((r_contact_mask) & (delta_vel_dot_normal > 0.0)),
                # (r_contact_mask),
                lambda x: x - delta_vel_dot_normal * tangent,
                # lambda x: x - delta_vel_dot_normal*node_normals, # no friction debug
                lambda x: x,
                vel_nt,
            )
            node_moments_nt = new_nodes_vel_nt * mass
            return node_moments_nt

        # jax.debug.print("r_nodes_vel_stack {}", r_nodes_vel_stack.max())

        moment_nt_stack = vmap_nodes(
            grid.moment_nt_stack,
            grid.mass_stack,
            grid.normal_stack,
            r_nodes_vel_stack,
            mp_position_intrp_stack,
            rp_position_intrp_stack,
        )

        if self.update_rigid_particles:
            new_position_stack, new_velocity_stack, new_com = (
                self.update_rigid_particles(
                    step, self.position_stack, self.velocity_stack, self.com, dt
                )
            )
        else:
            new_position_stack = self.position_stack
            new_velocity_stack = self.velocity_stack
            new_com = self.com

        new_grid = eqx.tree_at(
            lambda state: (state.moment_nt_stack),
            grid,
            (moment_nt_stack),
        )

        new_self = eqx.tree_at(
            lambda state: (state.position_stack, state.velocity_stack, state.com),
            self,
            (new_position_stack, new_velocity_stack, new_com),
        )

        return new_grid, new_self
