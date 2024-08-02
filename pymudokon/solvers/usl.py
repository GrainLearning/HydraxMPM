"""Implementation of the Explicit Update Stress Last (USL) Material Point Method (MPM).

References:
    - De Vaucorbeil, Alban, et al. 'Material point method after 25 years:
    theory, implementation, and applications.'
"""

from functools import partial
from typing import List, Tuple
from typing_extensions import Self

import chex
import jax
import jax.numpy as jnp

from ..forces.forces import Forces
from ..materials.material import Material
from ..nodes.nodes import Nodes
from ..particles.particles import Particles
from ..shapefunctions.shapefunctions import ShapeFunction
from .solver import Solver


@chex.dataclass
class USL(Solver):
    """Update Stress Last (USL) Material Point Method (MPM) solver.

    Attributes:
        alpha: FLIP-PIC ratio
        dt: time step of the solver

    """

    alpha: jnp.float32
    dt: jnp.float32

    @classmethod
    def create(
        cls,
        alpha: jnp.float32 = 0.99,
        dt: jnp.float32 = 0.00001,
    ):
        """Create a new instance of the USL solver."""
        return USL(alpha=alpha, dt=dt)

    def update(
        self: Self,
        particles: Particles,
        nodes: Nodes,
        shapefunctions: ShapeFunction,
        material_stack: List[Material],
        forces_stack: List[Forces],
    ):
        """Perform a single update step of the USL solver."""
        nodes = nodes.refresh()
        particles = particles.refresh()

        shapefunctions, _ = shapefunctions.calculate_shapefunction(
            origin=nodes.origin,
            inv_node_spacing=nodes.inv_node_spacing,
            grid_size=nodes.grid_size,
            position_stack=particles.position_stack,
        )

        nodes = self.p2g(
            particles=particles, nodes=nodes, shapefunctions=shapefunctions
        )

        # Apply forces here
        new_forces_stack = []
        for forces in forces_stack:
            nodes, forces = forces.apply_on_nodes_moments(
                particles=particles,
                nodes=nodes,
                shapefunctions=shapefunctions,
                dt=self.dt,
            )
            new_forces_stack.append(forces)

        particles = self.g2p(
            particles=particles,
            nodes=nodes,
            shapefunctions=shapefunctions,
        )

        new_material_stack = []
        for material in material_stack:
            particles, material = material.update_from_particles(
                particles=particles, dt=self.dt
            )
            new_material_stack.append(material)

        return (
            self,
            particles,
            nodes,
            shapefunctions,
            new_material_stack,
            new_forces_stack,
        )

    def p2g(
        self: Self,
        particles,
        nodes,
        shapefunctions,
    ):
        """Particle (MP)  to grid transfer function.

        Procedure is as follows:
        - Gather particle quantities to interactions.
        - Scale masses, moments, and forces by shape functions.
        - Calculate node internal force from scaled stresses, volumes.
        - Sum interaction quantities to nodes.
        """
        stencil_size, dim = shapefunctions.stencil.shape

        @partial(jax.vmap, in_axes=(0, 0, 0))
        def vmap_p2g(
            intr_id: chex.ArrayBatched,
            intr_shapef: chex.ArrayBatched,
            intr_shapef_grad: chex.ArrayBatched,
        ) -> Tuple[chex.ArrayBatched, chex.ArrayBatched, chex.ArrayBatched]:
            """Gather particle quantities to interactions."""
            # out of scope quantities (e.g., particles., are static)
            particle_id = (intr_id / stencil_size).astype(jnp.int32)

            intr_masses = particles.mass_stack.at[particle_id].get()
            intr_volumes = particles.volume_stack.at[particle_id].get()
            intr_velocities = particles.velocity_stack.at[particle_id].get()
            intr_ext_forces = particles.force_stack.at[particle_id].get()
            intr_stresses = particles.stress_stack.at[particle_id].get()

            scaled_mass = intr_shapef * intr_masses
            scaled_moments = scaled_mass * intr_velocities
            scaled_ext_force = intr_shapef * intr_ext_forces
            scaled_int_force = -1.0 * intr_volumes * intr_stresses @ intr_shapef_grad

            scaled_total_force = scaled_int_force[:dim] + scaled_ext_force

            return scaled_mass, scaled_moments, scaled_total_force

        # Get interaction id and respective particle belonging to interaction
        # form a batched interaction
        scaled_mass_stack, scaled_moment_stack, scaled_total_force_stack = vmap_p2g(
            shapefunctions.intr_id_stack,
            shapefunctions.intr_shapef_stack,
            shapefunctions.intr_shapef_grad_stack,
        )

        # Sum all interaction quantities.
        nodes_mass_stack = (
            jnp.zeros_like(nodes.mass_stack)
            .at[shapefunctions.intr_hash_stack]
            .add(scaled_mass_stack)
        )

        nodes_moment_stack = (
            jnp.zeros_like(nodes.moment_stack)
            .at[shapefunctions.intr_hash_stack]
            .add(scaled_moment_stack)
        )

        nodes_force_stack = (
            jnp.zeros_like(nodes.moment_stack)
            .at[shapefunctions.intr_hash_stack]
            .add(scaled_total_force_stack)
        )

        nodes_moment_nt_stack = nodes_moment_stack + nodes_force_stack * self.dt

        return nodes.replace(
            mass_stack=nodes_mass_stack,
            moment_stack=nodes_moment_stack,
            moment_nt_stack=nodes_moment_nt_stack,
        )

    def g2p(
        self,
        particles: Particles,
        nodes: Nodes,
        shapefunctions: chex.Array,
    ):
        """Grid to particle transfer function."""
        stencil_size, dim = shapefunctions.stencil.shape

        padding = (0, 3 - dim)

        @partial(jax.vmap, in_axes=(0, 0, 0))
        def vmap_intr_scatter(
            intr_hashes: chex.ArrayBatched,
            intr_shapef: chex.ArrayBatched,
            intr_shapef_grad: chex.ArrayBatched,
        ) -> Tuple[chex.ArrayBatched, chex.ArrayBatched, chex.ArrayBatched]:
            """Scatter quantities from nodes to interactions."""
            intr_masses = nodes.mass_stack.at[intr_hashes].get()
            intr_moments = nodes.moment_stack.at[intr_hashes].get()
            intr_moments_nt = nodes.moment_nt_stack.at[intr_hashes].get()

            # Small mass cutoff to avoid unphysical large velocities
            intr_vels = jax.lax.cond(
                intr_masses > nodes.small_mass_cutoff,
                lambda x: x / intr_masses,
                lambda x: jnp.zeros_like(x),
                intr_moments,
            )

            intr_vels_nt = jax.lax.cond(
                intr_masses > nodes.small_mass_cutoff,
                lambda x: x / intr_masses,
                lambda x: jnp.zeros_like(x),
                intr_moments_nt,
            )
            intr_delta_vels = intr_vels_nt - intr_vels

            intr_scaled_delta_vels = intr_shapef * intr_delta_vels

            intr_scaled_vels_nt = intr_shapef * intr_vels_nt

            # Pad velocities for plane strain
            intr_vels_nt_padded = jnp.pad(
                intr_vels_nt,
                padding,
                mode="constant",
                constant_values=0,
            )

            intr_scaled_velgrad = (
                intr_shapef_grad.reshape(-1, 1) @ intr_vels_nt_padded.reshape(-1, 1).T
            )

            return intr_scaled_delta_vels, intr_scaled_vels_nt, intr_scaled_velgrad

        @partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, 0, 0))
        def vmap_particles_update(
            intr_delta_vels_reshaped: chex.ArrayBatched,
            intr_vels_nt_reshaped: chex.ArrayBatched,
            intr_velgrad_reshaped: chex.ArrayBatched,
            p_velocities: chex.ArrayBatched,
            p_positions: chex.ArrayBatched,
            p_F: chex.ArrayBatched,
            p_volumes_orig: chex.ArrayBatched,
        ) -> Tuple[
            chex.ArrayBatched,
            chex.ArrayBatched,
            chex.ArrayBatched,
            chex.ArrayBatched,
            chex.ArrayBatched,
        ]:
            """Update particle quantities by summing interaction quantities."""
            p_velgrads_next = jnp.sum(intr_velgrad_reshaped, axis=0)

            delta_vels = jnp.sum(intr_delta_vels_reshaped, axis=0)
            vels_nt = jnp.sum(intr_vels_nt_reshaped, axis=0)

            p_velocities_next = (1.0 - self.alpha) * vels_nt + self.alpha * (
                p_velocities + delta_vels
            )

            p_positions_next = p_positions + vels_nt * self.dt

            if dim == 2:
                p_velgrads_next = p_velgrads_next.at[2, 2].set(0)

            p_F_next = (jnp.eye(3) + p_velgrads_next * self.dt) @ p_F

            if dim == 2:
                p_F_next = p_F_next.at[2, 2].set(1)

            p_volumes_next = jnp.linalg.det(p_F_next) * p_volumes_orig
            return (
                p_velocities_next,
                p_positions_next,
                p_F_next,
                p_volumes_next,
                p_velgrads_next,
            )

        (
            intr_scaled_delta_vel_stack,
            intr_scaled_vel_nt_stack,
            intr_scaled_velgrad_stack,
        ) = vmap_intr_scatter(
            shapefunctions.intr_hash_stack,
            shapefunctions.intr_shapef_stack,
            shapefunctions.intr_shapef_grad_stack,
        )

        (
            p_velocity_next_stack,
            p_position_next_stack,
            p_F_stack_next,
            p_volume_stack_next,
            p_L_stack_next,
        ) = vmap_particles_update(
            intr_scaled_delta_vel_stack.reshape(-1, stencil_size, dim),
            intr_scaled_vel_nt_stack.reshape(-1, stencil_size, dim),
            intr_scaled_velgrad_stack.reshape(-1, stencil_size, 3, 3),
            particles.velocity_stack,
            particles.position_stack,
            particles.F_stack,
            particles.volume0_stack,
        )

        return particles.replace(
            velocity_stack=p_velocity_next_stack,
            position_stack=p_position_next_stack,
            F_stack=p_F_stack_next,
            volume_stack=p_volume_stack_next,
            L_stack=p_L_stack_next,
        )
