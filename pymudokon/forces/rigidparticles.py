"""Module for imposing zero/non-zero boundaries via rigid particles."""

from functools import partial
from typing import Tuple
from typing_extensions import Self

import chex
import jax
import jax.numpy as jnp
from jax import Array

from ..nodes.nodes import Nodes
from ..particles.particles import Particles
from ..shapefunctions.shapefunctions import ShapeFunction


@chex.dataclass
class RigidParticles:
    """Shapes are discretized into rigid particles to impose boundary conditions.

    The rigid particles are used to impose boundary conditions on the grid.

    Attributes:
        position_stack: Positions of the rigid particles.
        velocity_stack: Velocities of the rigid particles.
        shapefunction: Shape function to interpolate the rigid particles on the grid.

    Example usage:
    >>> import jax.numpy as jnp
    >>> import pymudokon as pm
    >>> nodes = pm.Nodes.create(origin=jnp.array([0.0, 0.0]),
    ... end=jnp.array([1.0, 1.0]), node_spacing=0.5)
    >>> shapefunction = pm.LinearShapeFunction.create(nodes)
    >>> rigid_particles = pm.RigidParticles.create(
    ...     position_stack=jnp.array([[0.0, 0.0], [0.5, 0.5]]),
    ...     velocity_stack=jnp.array([[0.0, 0.0], [0.0, 0.0]]),
    ...     shapefunction=shapefunction,
    ... )
    >>> # add rigid particles to solver


    """

    position_stack: Array
    velocity_stack: Array
    shapefunction: ShapeFunction

    @classmethod
    def create(
        cls: Self,
        position_stack: Array,
        velocity_stack: Array,
        shapefunction: ShapeFunction,
    ) -> Self:
        """Initialize the rigid particles."""
        return cls(
            position_stack=position_stack,
            velocity_stack=velocity_stack,
            shapefunction=shapefunction,
        )

    def apply_on_nodes_moments(
        self: Self,
        nodes: Nodes,
        particles: Particles = None,
        shapefunctions: ShapeFunction = None,
        dt: jnp.float32 = 0.0,
    ) -> Tuple[Nodes, Self]:
        """Apply the boundary conditions on the nodes moments.

        Procedure:
            - Get the normals of the non-rigid particles on the grid.
            - Get the velocities on the grid due to the velocities of the
                rigid particles.
            - Get contacting nodes and apply the velocities on the grid.
        """
        # nr denotes non-rigid particles, r denotes rigid particles
        nr_stencil_size, dim = shapefunctions.stencil.shape

        @partial(jax.vmap, in_axes=(0, 0))
        def vmap_nr_p2g_grid_normals(
            intr_id: chex.ArrayBatched, intr_shapef_grad: chex.ArrayBatched
        ) -> chex.ArrayBatched:
            """Get the normals of the non-rigid particles on the grid."""
            nr_particle_id = (intr_id / nr_stencil_size).astype(jnp.int32)
            nr_intr_masses = particles.mass_stack.at[nr_particle_id].get()
            nr_intr_normal = (intr_shapef_grad * nr_intr_masses).at[:dim].get()
            return nr_intr_normal

        nr_intr_normal_stack = vmap_nr_p2g_grid_normals(
            shapefunctions.intr_id_stack, shapefunctions.intr_shapef_grad_stack
        )

        nodes_normal_stack = (
            jnp.zeros_like(nodes.moment_nt_stack)
            .at[shapefunctions.intr_hash_stack]
            .add(nr_intr_normal_stack)
        )

        r_shapefunctions, _ = self.shapefunction.calculate_shapefunction(
            origin=nodes.origin,
            inv_node_spacing=nodes.inv_node_spacing,
            grid_size=nodes.grid_size,
            position_stack=self.position_stack,
        )

        r_stencil_size, _ = r_shapefunctions.stencil.shape

        @partial(jax.vmap, in_axes=(0, 0))
        def vmap_rp2g_velocities(
            intr_id: chex.ArrayBatched, intr_shapef: chex.ArrayBatched
        ) -> chex.ArrayBatched:
            """Get velocities on the grid from velocities of the rigid particles."""
            particle_id = (intr_id / r_stencil_size).astype(jnp.int32)
            intr_velocities = self.velocity_stack.at[particle_id].get()
            r_scaled_velocity = intr_shapef * intr_velocities
            return r_scaled_velocity

        r_scaled_velocity_stack = vmap_rp2g_velocities(
            r_shapefunctions.intr_id_stack, r_shapefunctions.intr_shapef_stack
        )

        r_nodes_velocity_stack = (
            jnp.zeros_like(nodes.moment_nt_stack)
            .at[r_shapefunctions.intr_hash_stack]
            .add(r_scaled_velocity_stack)
        )

        r_nodes_contact_mask_stack = (
            jnp.zeros_like(nodes.mass_stack, dtype=bool)
            .at[r_shapefunctions.intr_hash_stack]
            .set(True)
        )

        @partial(jax.vmap, in_axes=(0, 0, 0, 0, 0))
        def vmap_nodes(
            node_moments_nt,
            nodes_masses,
            nodes_normals,
            r_nodes_velocities,
            r_nodes_contact_mask,
        ):
            """Apply the velocities on the grid from the rigid particles."""
            # skip the nodes with small mass, due to numerical instability
            nodes_vel_nt = jax.lax.cond(
                nodes_masses > nodes.small_mass_cutoff,
                lambda x: x / nodes_masses,
                lambda x: jnp.zeros_like(x),
                node_moments_nt,
            )

            # normalize the normals
            nodes_normals = jax.lax.cond(
                nodes_masses > nodes.small_mass_cutoff,
                lambda x: x / jnp.linalg.vector_norm(x),
                lambda x: jnp.zeros_like(x),
                nodes_normals,
            )

            # check if the velocity direction of the normal and apply contact
            # dot product is 0 when the vectors are orthogonal
            # and 1 when they are parallel
            # if othogonal no contact is happening
            # if parallel the contact is happening
            criteria = jnp.dot(nodes_vel_nt - r_nodes_velocities, nodes_normals)

            new_nodes_vel_nt = jax.lax.cond(
                ((r_nodes_contact_mask) & (criteria > 0.0)),
                lambda x: x - criteria * nodes_normals,
                lambda x: x,
                nodes_vel_nt,
            )

            node_moments_nt = new_nodes_vel_nt * nodes_masses

            return node_moments_nt

        moment_nt_stack = vmap_nodes(
            nodes.moment_nt_stack,
            nodes.mass_stack,
            nodes_normal_stack,
            r_nodes_velocity_stack,
            r_nodes_contact_mask_stack,
        )

        r_positions_next_stack = self.position_stack + self.velocity_stack * dt

        return nodes.replace(moment_nt_stack=moment_nt_stack), self.replace(
            position_stack=r_positions_next_stack
        )
