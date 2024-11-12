"""Module for imposing zero/non-zero boundaries via rigid particles."""

from functools import partial
from typing import Callable, Tuple
from typing_extensions import Self

import chex
import jax
import jax.numpy as jnp
from jax import Array

from ..nodes.nodes import Nodes
from ..particles.particles import Particles
from ..forces.forces import Forces
from ..config.mpm_config import MPMConfig

from ..nodes.grid import Grid
import equinox as eqx


class RigidParticles(Forces):
    """Shapes are discretized into rigid particles to impose boundary conditions.

    The rigid particles are used to impose boundary conditions on the grid.

    """

    position_stack: Array
    velocity_stack: Array
    com: Array
    mu: float

    grid: Grid

    update_rigid_particles: Callable = eqx.field(static=True)

    def __init__(
        self: Self,
        config: MPMConfig,
        position_stack: Array,
        velocity_stack: Array = None,
        mu: float = 0.0,
        com: Array = None,
        update_rigid_particles: Callable = None,
    ) -> Self:
        """Initialize the rigid particles."""
        num_rigid = position_stack.shape[0]

        if velocity_stack is None:
            velocity_stack = jnp.zeros((num_rigid, config.dim))

        self.position_stack = position_stack
        self.velocity_stack = velocity_stack

        self.grid = Grid(config)
        self.mu = mu
        self.update_rigid_particles = update_rigid_particles
        self.com = com
        super().__init__(config)

    def apply_on_nodes(
        self: Self,
        nodes: Nodes,
        particles: Particles = None,
        step: jnp.int32 = 0,
    ) -> Tuple[Nodes, Self]:
        """Apply the boundary conditions on the nodes moments.

        Procedure:
            - Get the normals of the non-rigid particles on the grid.
            - Get the velocities on the grid due to the velocities of the
                rigid particles.
            - Get contacting nodes and apply the velocities on the grid.
        """

        def vmap_velocities_p2g_non_rigid(
            point_id, intr_shapef, intr_shapef_grad, intr_dist
        ):
            intr_velocities = self.velocity_stack.at[point_id].get()
            r_scaled_velocity = intr_shapef * intr_velocities
            return r_scaled_velocity

        new_grid, r_scaled_velocity_stack = self.grid.vmap_interactions_and_scatter(
            vmap_velocities_p2g_non_rigid, self.position_stack
        )

        r_nodes_vel_stack = (
            jnp.zeros_like(nodes.moment_nt_stack)
            .at[new_grid.intr_hash_stack]
            .add(r_scaled_velocity_stack)
        )

        r_nodes_contact_mask_stack = (
            jnp.zeros_like(nodes.mass_stack, dtype=jnp.bool_)
            .at[new_grid.intr_hash_stack]
            .set(True)
        )

        @partial(jax.vmap, in_axes=(0, 0, 0, 0, 0))
        def vmap_nodes(
            moment_nt,
            mass,
            normal,
            r_vel,
            r_contact_mask,
        ):
            """Apply the velocities on the grid from the rigid particles."""
            # skip the nodes with small mass, due to numerical instability
            vel_nt = jax.lax.cond(
                mass > nodes.small_mass_cutoff,
                lambda x: x / mass,
                lambda x: jnp.zeros_like(x),
                moment_nt,
            )

            # normalize the normals
            normal = jax.lax.cond(
                mass > nodes.small_mass_cutoff,
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

            delta_vel_dot_normal = jnp.dot(delta_vel, normal)

            delta_vel_padded = jnp.pad(
                delta_vel,
                self.config.padding,
                mode="constant",
                constant_values=0,
            )

            norm_padded = jnp.pad(
                normal,
                self.config.padding,
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
                .at[: self.config.dim]
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
            nodes.moment_nt_stack,
            nodes.mass_stack,
            nodes.normal_stack,
            r_nodes_vel_stack,
            r_nodes_contact_mask_stack,
        )

        if self.update_rigid_particles:
            new_position_stack, new_velocity_stack, new_com = (
                self.update_rigid_particles(
                    step,
                    self.position_stack,
                    self.velocity_stack,
                    self.com,
                    self.config,
                )
            )
        else:
            new_position_stack = self.position_stack
            new_velocity_stack = self.velocity_stack
            new_com = self.com

        new_nodes = eqx.tree_at(
            lambda state: (state.moment_nt_stack),
            nodes,
            (moment_nt_stack),
        )

        new_self = eqx.tree_at(
            lambda state: (state.position_stack, state.velocity_stack, state.com),
            self,
            (new_position_stack, new_velocity_stack, new_com),
        )

        return new_nodes, new_self
