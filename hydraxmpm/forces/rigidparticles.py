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

    alpha: TypeFloat
    beta: TypeFloat

    update_rigid_particles: Optional[Callable] = eqx.field(static=True)

    shape_map: ShapeFunctionMapping

    def __init__(
        self: Self,
        position_stack: TypeFloatVectorAStack,
        velocity_stack: TypeFloatVectorAStack = None,
        mu: TypeFloat = 0.0,
        com: Optional[TypeFloatVector] = None,
        alpha: TypeFloat = 0.0001,
        beta: TypeFloat = 2.0,
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

        self.alpha = alpha

        self.beta = beta

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

        new_shape_map, r_scaled_velocity_stack = (
            self.shape_map.vmap_interactions_and_scatter(
                vmap_velocities_p2g_rigid, position_stack=self.position_stack, grid=grid
            )
        )

        r_nodes_vel_stack = (
            jnp.zeros_like(grid.moment_nt_stack)
            .at[new_shape_map._intr_hash_stack]
            .add(r_scaled_velocity_stack)
        )

        r_nodes_contact_mask_stack = (
            jnp.zeros_like(grid.mass_stack, dtype=jnp.bool_)
            .at[new_shape_map._intr_hash_stack]
            .set(True)
        )

        intr_vec_dist_stack = jnp.sqrt(
            jnp.sum(jnp.pow(new_shape_map._intr_dist_stack, 2), axis=1)
        )

        r_nodes_min_dist_stack = (
            jnp.zeros_like(grid.mass_stack)
            .at[new_shape_map._intr_hash_stack]
            .min(intr_vec_dist_stack)
        )

        @partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, 0))
        def vmap_nodes(moment_nt, mass, normal, r_vel, r_contact_mask, r_min_dist):
            """Apply the velocities on the grid from the rigid particles."""
            # skip the nodes with small mass, due to numerical instability
            vel_nt = jax.lax.cond(
                mass > grid.small_mass_cutoff,
                lambda x: x / mass,
                lambda x: jnp.zeros_like(x),
                moment_nt,
            )

            # normalize the normals
            normal = jax.lax.cond(
                mass > grid.small_mass_cutoff,
                lambda x: x / jnp.linalg.vector_norm(x),
                lambda x: jnp.zeros_like(x),
                normal,
            )
            normal = jnp.nan_to_num(normal)

            # check if the velocity direction of the normal and apply contact
            # dot product is 0 when the vectors are orthogonal
            # and 1 when they are parallel
            # if othogonal no contact is happening
            # if parallel the contact is happening
            delta_vel = vel_nt - r_vel

            # modification

            x = jax.lax.cond(
                r_min_dist <= 0,
                lambda x: 1.0 - 2 * (-x * (1.0 / 1.25) ** (0.58)),
                lambda x: 2 * (x * (1.0 / 1.25) ** (0.58)) - 1.0,
                r_min_dist,
            )

            fp = (1.0 - self.alpha * (x**self.beta)) / (
                1.0 + self.alpha * (x**self.beta)
            )

            delta_vel *= fp

            # end modification

            delta_vel_dot_normal = jnp.dot(delta_vel, normal)

            delta_vel_padded = jnp.pad(
                delta_vel,
                new_shape_map._padding,
                mode="constant",
                constant_values=0,
            )

            norm_padded = jnp.pad(
                normal,
                new_shape_map._padding,
                mode="constant",
                constant_values=0,
            )

            delta_vel_cross_normal = jnp.cross(
                delta_vel_padded, norm_padded
            )  # works only for vectors of len 3
            norm_delta_vel_cross_normal = jnp.linalg.vector_norm(delta_vel_cross_normal)

            omega = delta_vel_cross_normal / norm_delta_vel_cross_normal
            mu_prime = jnp.minimum(
                self.mu, norm_delta_vel_cross_normal / delta_vel_dot_normal
            )

            normal_cross_omega = jnp.cross(
                norm_padded, omega
            )  # works only for vectors of len 3

            tangent = (
                (norm_padded + mu_prime * normal_cross_omega)
                .at[: new_shape_map.dim]
                .get()
            )

            # sometimes tangent become nan if velocity is zero at initialization
            # which causes problems
            tangent = jnp.nan_to_num(tangent)

            new_nodes_vel_nt = jax.lax.cond(
                ((r_contact_mask) & (delta_vel_dot_normal > 0.0)),
                lambda x: x - delta_vel_dot_normal * tangent,
                # lambda x: x - delta_vel_dot_normal*node_normals, # no friction debug
                lambda x: x,
                vel_nt,
            )
            node_moments_nt = new_nodes_vel_nt * mass
            return node_moments_nt

        moment_nt_stack = vmap_nodes(
            grid.moment_nt_stack,
            grid.mass_stack,
            grid.normal_stack,
            r_nodes_vel_stack,
            r_nodes_contact_mask_stack,
            r_nodes_min_dist_stack,
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
