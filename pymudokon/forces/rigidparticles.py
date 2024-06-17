"""Node walls"""

from functools import partial
from typing import List, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax import Array
from typing_extensions import Self

from ..core.nodes import Nodes
from ..core.particles import Particles
from ..shapefunctions.shapefunction import ShapeFunction


@struct.dataclass
class RigidParticles:
    positions: Array
    velocities: Array
    shapefunction: ShapeFunction

    @classmethod
    def create(cls: Self, positions: Array, velocities: Array, shapefunction: ShapeFunction) -> Self:
        return cls(positions=positions, velocities=velocities, shapefunction=shapefunction)

    @jax.jit
    def apply_on_nodes_moments(
        self: Self,
        nodes: Nodes,
        particles: Particles = None,
        shapefunctions: ShapeFunction = None,
        dt: jnp.float32 = 0.0,
    ) -> Tuple[Nodes, Self]:
        # (1) Get grid normals from non-rigid particles
        nr_stencil_size, dim = shapefunctions.stencil.shape

        @partial(jax.vmap, in_axes=(0, 0))
        def vmap_nrp2g_grid_normals(intr_id, intr_shapef_grad):
            nr_particle_id = (intr_id / nr_stencil_size).astype(jnp.int32)
            nr_intr_masses = particles.masses.at[nr_particle_id].get()
            nr_intr_normal = intr_shapef_grad * nr_intr_masses
            return nr_intr_normal

        nr_intr_normal = vmap_nrp2g_grid_normals(shapefunctions.intr_ids, shapefunctions.intr_shapef_grad)

        nodes_normals = jnp.zeros_like(nodes.moments_nt).at[shapefunctions.intr_hashes].add(nr_intr_normal)

        # (2) Velocities on the grid due to the velocities of the rigid particles
        r_shapefunctions, _ = self.shapefunction.calculate_shapefunction(nodes, self.positions)

        r_stencil_size, _ = r_shapefunctions.stencil.shape

        @partial(jax.vmap, in_axes=(0, 0))
        def vmap_rp2g_velocities(intr_id, intr_shapef):
            particle_id = (intr_id / r_stencil_size).astype(jnp.int32)
            intr_velocities = self.velocities.at[particle_id].get()
            r_scaled_velocity = intr_shapef * intr_velocities
            return r_scaled_velocity

        r_scaled_velocity = vmap_rp2g_velocities(r_shapefunctions.intr_ids, r_shapefunctions.intr_shapef)

        r_nodes_velocities = jnp.zeros_like(nodes.moments_nt).at[r_shapefunctions.intr_hashes].add(r_scaled_velocity)

        r_nodes_contact_mask = jnp.zeros_like(nodes.masses, dtype=bool).at[r_shapefunctions.intr_hashes].set(True)

        # (3) Apply the velocities on the grid due to the velocities of the rigid particles
        @partial(jax.vmap, in_axes=(0, 0, 0, 0, 0))
        def vmap_nodes(node_moments_nt, nodes_masses, nodes_normals, r_nodes_velocities, r_nodes_contact_mask):
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

            # # if the velocity direction of the normal
            # # dot product is 0 when the vectors are orthogonalm and 1 when they are parallel
            # # if othogonal no contact is happening
            # # if parallel the contact is happening
            criteria = jnp.dot(nodes_vel_nt - r_nodes_velocities, nodes_normals)

            new_nodes_vel_nt = jax.lax.cond(
                ((r_nodes_contact_mask) & (criteria > 0.0)),
                lambda x: x - criteria * nodes_normals,
                lambda x: x,
                nodes_vel_nt,
            )

            node_moments_nt = new_nodes_vel_nt * nodes_masses

            return node_moments_nt

        moments_nt = vmap_nodes(nodes.moments_nt, nodes.masses, nodes_normals, r_nodes_velocities, r_nodes_contact_mask)

        r_positions_next = self.positions + self.velocities * dt

        return nodes.replace(moments_nt=moments_nt), self.replace(positions=r_positions_next)
        # # # set positions of the particles
        # return nodes.replace(
        #     moments_nt=moments_nt
        # ), self
