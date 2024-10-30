"""Implementation of the Explicit Update Stress Last (USL) Material Point Method (MPM).

References:
    - De Vaucorbeil, Alban, et al. 'Material point method after 25 years:
    theory, implementation, and applications.'
"""

from functools import partial
from typing import List, Tuple
from pyvista import Grid
from typing_extensions import Self

import chex
import jax
import jax.numpy as jnp

from ..nodes.nodes import Nodes
from ..particles.particles import Particles
from ..config.mpm_config import MPMConfig
import equinox as eqx


from .solver import Solver


class USL(Solver):
    """Update Stress Last (USL) Material Point Method (MPM) solver.

    Attributes:
        alpha: FLIP-PIC ratio
        dt: time step of the solver

    """

    alpha: float = eqx.field(static=True, converter=lambda x: float(x))

    def __init__(self, config: MPMConfig, alpha: float = 0.99):
        self.alpha = alpha
        super().__init__(config)

    def update(
        self: Self,
        particles: Particles,
        nodes: Nodes,
        material_stack: List,
        forces_stack: List,
        step: int = 0,
    ):
        particles = particles.refresh()
        nodes = nodes.refresh()

        nodes = nodes.get_interactions(particles.position_stack)

        nodes = self.p2g(particles=particles, nodes=nodes)

        # Apply forces here
        new_forces_stack = []
        for forces in forces_stack:
            nodes, new_forces = forces.apply_on_nodes(
                particles=particles,
                nodes=nodes,
                step=step,
            )
            new_forces_stack.append(new_forces)

        particles = self.g2p(particles=particles, nodes=nodes)

        new_material_stack = []
        for material in material_stack:
            particles, new_material = material.update_from_particles(particles=particles)
            new_material_stack.append(new_material)

        return (
            self,
            particles,
            nodes,
            new_material_stack,
            new_forces_stack,
        )

    def p2g(self, particles, nodes):
        def vmap_intr_p2g(point_id, intr_shapef, intr_shapef_grad):
            intr_masses = particles.mass_stack.at[point_id].get()
            intr_volumes = particles.volume_stack.at[point_id].get()
            intr_velocities = particles.velocity_stack.at[point_id].get()
            intr_ext_forces = particles.force_stack.at[point_id].get()
            intr_stresses = particles.stress_stack.at[point_id].get()

            scaled_mass = intr_shapef * intr_masses
            scaled_moments = scaled_mass * intr_velocities
            scaled_ext_force = intr_shapef * intr_ext_forces
            scaled_int_force = -1.0 * intr_volumes * intr_stresses @ intr_shapef_grad

            scaled_total_force = scaled_int_force[: self.config.dim] + scaled_ext_force
            return scaled_mass, scaled_moments, scaled_total_force

        # Get interaction id and respective particle belonging to interaction

        # form a batched interaction
        scaled_mass_stack, scaled_moment_stack, scaled_total_force_stack = (
            nodes.vmap_intr_scatter(vmap_intr_p2g)
        )

        # Sum all interaction quantities.
        new_mass_stack = (
            jnp.zeros_like(nodes.mass_stack)
            .at[nodes.intr_hash_stack]
            .add(scaled_mass_stack)
        )

        new_moment_stack = (
            jnp.zeros_like(nodes.moment_stack)
            .at[nodes.intr_hash_stack]
            .add(scaled_moment_stack)
        )

        new_force_stack = (
            jnp.zeros_like(nodes.moment_stack)
            .at[nodes.intr_hash_stack]
            .add(scaled_total_force_stack)
        )

        nodes_moment_nt_stack = new_moment_stack + new_force_stack * self.config.dt

        return eqx.tree_at(
            lambda state: (
                state.mass_stack,
                state.moment_stack,
                state.moment_nt_stack,
            ),
            nodes,
            (new_mass_stack, new_moment_stack, nodes_moment_nt_stack),
        )

    def g2p(self, particles: Particles, nodes: Nodes):
        def vmap_intr_g2p(
            intr_hashes: chex.ArrayBatched,
            intr_shapef: chex.ArrayBatched,
            intr_shapef_grad: chex.ArrayBatched,
        ):
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
                self.config.padding,
                mode="constant",
                constant_values=0,
            )

            intr_scaled_velgrad = (
                intr_shapef_grad.reshape(-1, 1) @ intr_vels_nt_padded.reshape(-1, 1).T
            )

            return intr_scaled_delta_vels, intr_scaled_vels_nt, intr_scaled_velgrad

        (
            new_intr_scaled_delta_vel_stack,
            new_intr_scaled_vel_nt_stack,
            new_intr_scaled_velgrad_stack,
        ) = nodes.vmap_intr_gather(vmap_intr_g2p)

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

            p_positions_next = p_positions + vels_nt * self.config.dt

            if self.config.dim == 2:
                p_velgrads_next = p_velgrads_next.at[2, 2].set(0)

            p_F_next = (jnp.eye(3) + p_velgrads_next * self.config.dt) @ p_F

            if self.config.dim == 2:
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
            new_velocity_stack,
            new_position_stack,
            new_F_stack,
            new_volume_stack,
            new_L_stack,
        ) = vmap_particles_update(
            new_intr_scaled_delta_vel_stack.reshape(
                -1, self.config.window_size, self.config.dim
            ),
            new_intr_scaled_vel_nt_stack.reshape(
                -1, self.config.window_size, self.config.dim
            ),
            new_intr_scaled_velgrad_stack.reshape(-1, self.config.window_size, 3, 3),
            particles.velocity_stack,
            particles.position_stack,
            particles.F_stack,
            particles.volume0_stack,
        )
        return eqx.tree_at(
            lambda state: (
                state.volume_stack,
                state.F_stack,
                state.L_stack,
                state.position_stack,
                state.velocity_stack,
            ),
            particles,
            (
                new_volume_stack,
                new_F_stack,
                new_L_stack,
                new_position_stack,
                new_velocity_stack,
            ),
        )
 