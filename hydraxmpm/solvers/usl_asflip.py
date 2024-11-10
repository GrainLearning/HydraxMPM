from functools import partial
from typing import Tuple
from typing_extensions import Self

import chex
import jax
import jax.numpy as jnp

from ..nodes.nodes import Nodes
from ..particles.particles import Particles
from .solver import Solver
import equinox as eqx


class USL_ASFLIP(Solver):
    """Explicit Update Stress Last (USL) Affine Particle in Cell (APIC) MPM solver."""

    alpha: jnp.float32
    phi_c: jnp.float32
    beta_min: jnp.float32
    beta_max: jnp.float32
    Dp: chex.Array
    Dp_inv: chex.Array
    Bp_stack: chex.Array

    def __init__(self, config, alpha=1.0, phi_c=0.5, beta_min=0.0, beta_max=0.0):
        # jax.debug.print("USL_APIC solver supported for cubic shape functions only")
        self.Dp = (1.0 / 3.0) * config.cell_size * config.cell_size * jnp.eye(3)

        self.Dp_inv = jnp.linalg.inv(self.Dp)

        self.Bp_stack = jnp.zeros((config.num_points, 3, 3))

        self.alpha = alpha

        self.phi_c = phi_c
        self.beta_min = beta_min
        self.beta_max = beta_max
        super().__init__(config)

    def update(self, particles, nodes, material_stack, forces_stack, step):
        nodes = nodes.refresh()
        particles = particles.refresh()

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

        particles, self = self.g2p(particles=particles, nodes=nodes)

        new_material_stack = []
        for material in material_stack:
            particles, new_material = material.update_from_particles(
                particles=particles
            )
            new_material_stack.append(new_material)

        return (
            self,
            particles,
            nodes,
            new_material_stack,
            new_forces_stack,
        )

    def p2g(self: Self, particles: Particles, nodes: Nodes) -> Nodes:
        def vmap_intr_p2g(point_id, intr_shapef, intr_shapef_grad, intr_dist):
            intr_masses = particles.mass_stack.at[point_id].get()
            intr_volumes = particles.volume_stack.at[point_id].get()
            intr_velocities = particles.velocity_stack.at[point_id].get()
            intr_ext_forces = particles.force_stack.at[point_id].get()
            intr_stresses = particles.stress_stack.at[point_id].get()

            intr_Bp = self.Bp_stack.at[point_id].get()  # APIC affine matrix

            affine_velocity = (
                intr_Bp @ jnp.linalg.inv(self.Dp)
            ) @ intr_dist  # intr_dist is 3D

            scaled_mass = intr_shapef * intr_masses
            scaled_moments = scaled_mass * (
                intr_velocities + affine_velocity.at[: self.config.dim].get()
            )

            scaled_ext_force = intr_shapef * intr_ext_forces
            scaled_int_force = -1.0 * intr_volumes * intr_stresses @ intr_shapef_grad

            scaled_total_force = (
                scaled_int_force.at[: self.config.dim].get() + scaled_ext_force
            )

            scaled_normal = (intr_shapef_grad * intr_masses).at[: self.config.dim].get()

            return scaled_mass, scaled_moments, scaled_total_force, scaled_normal

        # form a batched interaction
        (
            new_nodes,
            (
                scaled_mass_stack,
                scaled_moment_stack,
                scaled_total_force_stack,
                scaled_normal_stack,
            ),
        ) = nodes.vmap_interactions_and_scatter(vmap_intr_p2g, particles.position_stack)

        # Sum all interaction quantities.
        new_mass_stack = (
            jnp.zeros_like(new_nodes.mass_stack)
            .at[new_nodes.intr_hash_stack]
            .add(scaled_mass_stack)
        )

        new_moment_stack = (
            jnp.zeros_like(new_nodes.moment_stack)
            .at[new_nodes.intr_hash_stack]
            .add(scaled_moment_stack)
        )

        new_force_stack = (
            jnp.zeros_like(new_nodes.moment_stack)
            .at[new_nodes.intr_hash_stack]
            .add(scaled_total_force_stack)
        )

        new_normal_stack = (
            jnp.zeros_like(new_nodes.normal_stack)
            .at[new_nodes.intr_hash_stack]
            .add(scaled_normal_stack)
        )

        nodes_moment_nt_stack = new_moment_stack + new_force_stack * self.config.dt

        return eqx.tree_at(
            lambda state: (
                state.mass_stack,
                state.moment_stack,
                state.moment_nt_stack,
                state.normal_stack,
            ),
            new_nodes,
            (new_mass_stack, new_moment_stack, nodes_moment_nt_stack, new_normal_stack),
        )

    def g2p(self: Self, particles: Particles, nodes: Nodes) -> Tuple[Particles, Self]:
        def vmap_intr_g2p(
            intr_hashes: chex.ArrayBatched,
            intr_shapef: chex.ArrayBatched,
            intr_shapef_grad: chex.ArrayBatched,
            intr_dist: chex.ArrayBatched,
        ):
            intr_masses = nodes.mass_stack.at[intr_hashes].get()
            intr_moments = nodes.moment_stack.at[intr_hashes].get()
            intr_moments_nt = nodes.moment_nt_stack.at[intr_hashes].get()

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

            intr_scaled_vels_nt = intr_shapef * intr_vels_nt

            intr_delta_vels = intr_vels_nt - intr_vels
            intr_scaled_delta_vels = intr_shapef * intr_delta_vels

            # Pad velocities for plane strain
            intr_vels_nt_padded = jnp.pad(
                intr_vels_nt,
                self.config.padding,
                mode="constant",
                constant_values=0,
            )

            # APIC affine matrix
            intr_Bp = (
                intr_shapef
                * intr_vels_nt_padded.reshape(-1, 1)
                @ intr_dist.reshape(-1, 1).T
            )

            intr_scaled_velgrad = intr_Bp @ self.Dp_inv

            return (
                intr_scaled_delta_vels,
                intr_scaled_vels_nt,
                intr_scaled_velgrad,
                intr_Bp,
            )

        (
            new_intr_scaled_delta_vel_stack,
            new_intr_scaled_vel_nt_stack,
            new_intr_scaled_velgrad_stack,
            new_intr_Bp_stack,
        ) = nodes.vmap_intr_gather_dist(vmap_intr_g2p)

        @partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, 0, 0, 0))
        def vmap_particles_update(
            intr_delta_vels_reshaped,
            intr_vels_nt_reshaped,
            intr_velgrad_reshaped,
            intr_Bp,
            p_velocities,
            p_positions,
            p_F,
            p_volumes_orig,
        ):
            # Update particle quantities
            p_velgrads_next = jnp.sum(intr_velgrad_reshaped, axis=0)

            delta_vels = jnp.sum(intr_delta_vels_reshaped, axis=0)
            vels_nt = jnp.sum(intr_vels_nt_reshaped, axis=0)

            p_Bp_next = jnp.sum(intr_Bp, axis=0)
            if self.config.dim == 2:
                p_Bp_next = p_Bp_next.at[2, 2].set(0)

            T = self.alpha * (p_velocities + delta_vels - vels_nt)
            p_velocities_next = vels_nt + T

            if self.config.dim == 2:
                p_velgrads_next = p_velgrads_next.at[2, 2].set(0)

            p_F_next = (jnp.eye(3) + p_velgrads_next * self.config.dt) @ p_F

            if self.config.dim == 2:
                p_F_next = p_F_next.at[2, 2].set(1)

            p_volumes_next = jnp.linalg.det(p_F_next) * p_volumes_orig

            # solid volume fraction
            phi = p_volumes_orig / p_volumes_next  # mass is constant

            Beta_p = jax.lax.cond(
                phi < self.phi_c, lambda: self.beta_max, lambda: self.beta_min
            )
            p_positions_next = p_positions + (vels_nt + Beta_p * T) * self.config.dt

            return (
                p_velocities_next,
                p_positions_next,
                p_F_next,
                p_volumes_next,
                p_velgrads_next,
                p_Bp_next,
            )

        (
            new_velocity_stack,
            new_position_stack,
            new_F_stack,
            new_volume_stack,
            new_L_stack,
            new_Bp_stack,
        ) = vmap_particles_update(
            new_intr_scaled_delta_vel_stack.reshape(
                -1, self.config.window_size, self.config.dim
            ),
            new_intr_scaled_vel_nt_stack.reshape(
                -1, self.config.window_size, self.config.dim
            ),
            new_intr_scaled_velgrad_stack.reshape(-1, self.config.window_size, 3, 3),
            new_intr_Bp_stack.reshape(-1, self.config.window_size, 3, 3),
            particles.velocity_stack,
            particles.position_stack,
            particles.F_stack,
            particles.volume0_stack,
        )

        new_solver = eqx.tree_at(
            lambda state: (state.Bp_stack),
            self,
            (new_Bp_stack),
        )

        new_particles = eqx.tree_at(
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

        return new_particles, new_solver
