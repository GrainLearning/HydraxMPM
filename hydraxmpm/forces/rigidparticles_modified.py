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
from .force import Forces
from ..config.mpm_config import MPMConfig

from ..nodes.grid import Grid
import equinox as eqx


class RigidParticlesModified(Forces):
    """Shapes are discretized into rigid particles to impose boundary conditions.


    Correction to Bardenhagen's contact algorithm presented by

    L. Gao, et. al, 2022, MPM modeling of pile installation in sand - Computers and geotechniques


    The rigid particles are used to impose boundary conditions on the grid.

    """

    position_stack: Array
    velocity_stack: Array
    com: Array
    mu: float

    alpha: float
    beta: float

    grid: Grid
    thickness: tuple = eqx.field(static=True)
    truncate_outbound: bool = eqx.field(static=True)

    update_rigid_particles: Callable = eqx.field(static=True)

    def __init__(
        self: Self,
        config: MPMConfig,
        position_stack: Array,
        velocity_stack: Array = None,
        mu: float = 0.0,
        com: Array = None,
        alpha: jnp.float32 = 0.8,
        beta: jnp.float32 = 2.0,
        update_rigid_particles: Callable = None,
        thickness=0,
        truncate_outbound=True,
    ) -> Self:
        """Initialize the rigid particles."""
        num_rigid = position_stack.shape[0]

        if velocity_stack is None:
            velocity_stack = jnp.zeros((num_rigid, config.dim), device=config.device)

        self.position_stack = jax.device_put(position_stack, device=config.device)
        self.velocity_stack = jax.device_put(velocity_stack, device=config.device)

        self.grid = Grid(config)
        self.mu = mu
        self.update_rigid_particles = update_rigid_particles

        self.com = com

        self.alpha = alpha

        self.beta = beta
        self.thickness = config.cell_size * thickness * jnp.ones(config.dim)
        self.truncate_outbound = truncate_outbound
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

        def check_in_domain(pos):
            ls_valid = pos > jnp.array(self.config.origin) + jnp.array(self.thickness)
            gt_valid = pos < jnp.array(self.config.end) - jnp.array(self.thickness)
            return jnp.all(ls_valid * gt_valid)

        is_valid_stack = jax.vmap(check_in_domain)(self.position_stack)

        def vmap_velocities_p2g_rigid(
            point_id, intr_shapef, intr_shapef_grad, intr_dist
        ):
            intr_velocities = self.velocity_stack.at[point_id].get()
            r_scaled_velocity = intr_shapef * intr_velocities
            return r_scaled_velocity

        new_grid, r_scaled_velocity_stack = self.grid.vmap_interactions_and_scatter(
            vmap_velocities_p2g_rigid, self.position_stack
        )

        def null_outbound_interactions(intr_hash, is_valid):
            return jax.lax.cond(
                is_valid,
                lambda: intr_hash,
                lambda: -1 * jnp.ones(self.config.window_size).astype(jnp.int32),
            )

        new_intr_hash_stack = jax.vmap(null_outbound_interactions)(
            new_grid.intr_hash_stack.reshape(
                (self.config.num_points, self.config.window_size)
            ),
            is_valid_stack,
        ).reshape(-1)

        r_nodes_vel_stack = (
            jnp.zeros_like(nodes.moment_nt_stack)
            .at[new_intr_hash_stack]
            .add(r_scaled_velocity_stack)
        )

        r_nodes_contact_mask_stack = (
            jnp.zeros_like(nodes.mass_stack, dtype=jnp.bool_)
            .at[new_intr_hash_stack]
            .set(True)
        )

        intr_vec_dist_stack = jnp.sqrt(
            jnp.sum(jnp.pow(self.grid.intr_dist_stack, 2), axis=1)
        )

        r_nodes_min_dist_stack = (
            jnp.zeros_like(nodes.mass_stack)
            .at[new_intr_hash_stack]
            .min(intr_vec_dist_stack)
        )

        @partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, 0))
        def vmap_nodes(moment_nt, mass, normal, r_vel, r_contact_mask, r_min_dist):
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
            r_nodes_min_dist_stack,
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
