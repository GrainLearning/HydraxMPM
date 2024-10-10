from functools import partial
from typing import Tuple
from typing_extensions import Self

import chex
import jax
import jax.numpy as jnp

from ..nodes.nodes import Nodes
from ..particles.particles import Particles
from ..shapefunctions.shapefunctions import ShapeFunction
from ..solvers.solver import Solver


@chex.dataclass
class USL_APIC(Solver):
    """
    Explicit Update Stress Last (USL) Affine Particle in Cell (APIC) MPM solver.

    !!! warning "Warning"

        Only cubic shape functions are supported for this solver at the moment.

    **References:**

    * Jiang, Chenfanfu, et al. "The affine particle-in-cell method."
        ACM Transactions on Graphics (TOG) 34.4 (2015): 1-10.

    """

    Dp: chex.Array
    Dp_inv: chex.Array
    Bp_stack: chex.Array

    @classmethod
    def create(
        cls,
        cell_size,
        dim,
        num_particles,
        dt: jnp.float32 = 0.00001,
    ):
        # jax.debug.print("USL_APIC solver supported for cubic shape functions only")
        Dp = (1.0 / 3.0) * cell_size * cell_size * jnp.eye(3)

        Dp_inv = jnp.linalg.inv(Dp)

        Bp_stack = jnp.zeros((num_particles, 3, 3))

        return USL_APIC(dt=dt, Dp=Dp, Dp_inv=Dp_inv, Bp_stack=Bp_stack)

    def update(
        self, particles, nodes, shapefunctions, material_stack, forces_stack, step
    ):
        nodes = nodes.refresh()
        particles = particles.refresh()

        shapefunctions, intr_dist_3d_stack = shapefunctions.calculate_shapefunction(
            origin=nodes.origin,
            inv_node_spacing=nodes.inv_node_spacing,
            grid_size=nodes.grid_size,
            position_stack=particles.position_stack,
            species_stack=nodes.species_stack,
        )

        # transform from grid space to particle space
        intr_dist_3d_stack = -1.0 * intr_dist_3d_stack * nodes.node_spacing

        nodes = self.p2g(
            particles=particles,
            nodes=nodes,
            shapefunctions=shapefunctions,
            intr_dist_3d_stack=intr_dist_3d_stack,
        )

        new_forces_stack = []
        for forces in forces_stack:
            nodes, forces = forces.apply_on_nodes_moments(
                particles=particles,
                nodes=nodes,
                shapefunctions=shapefunctions,
                dt=self.dt,
                step=step,
            )
            new_forces_stack.append(forces)

        particles, self = self.g2p(
            particles=particles,
            nodes=nodes,
            shapefunctions=shapefunctions,
            intr_dist_3d_stack=intr_dist_3d_stack,
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
        particles: Particles,
        nodes: Nodes,
        shapefunctions: ShapeFunction,
        intr_dist_3d_stack: jax.Array,
    ) -> Nodes:
        stencil_size, dim = shapefunctions.stencil.shape

        @partial(jax.vmap, in_axes=(0, 0, 0, 0))
        def vmap_p2g(intr_id, intr_shapef, intr_shapef_grad, intr_dist_3d):
            particle_id = (intr_id / stencil_size).astype(jnp.int32)

            intr_masses = particles.mass_stack.at[particle_id].get()
            intr_volumes = particles.volume_stack.at[particle_id].get()
            intr_velocities = particles.velocity_stack.at[particle_id].get()
            intr_ext_forces = particles.force_stack.at[particle_id].get()
            intr_stresses = particles.stress_stack.at[particle_id].get()

            intr_Bp = self.Bp_stack.at[particle_id].get()  # APIC affine matrix

            affine_velocity = (intr_Bp @ jnp.linalg.inv(self.Dp)) @ intr_dist_3d

            scaled_mass = intr_shapef * intr_masses
            scaled_moments = scaled_mass * (
                intr_velocities + affine_velocity.at[:dim].get()
            )
            scaled_ext_force = intr_shapef * intr_ext_forces
            scaled_int_force = -1.0 * intr_volumes * intr_stresses @ intr_shapef_grad

            scaled_total_force = scaled_int_force.at[:dim].get() + scaled_ext_force

            return scaled_mass, scaled_moments, scaled_total_force

        scaled_mass_stack, scaled_moment_stack, scaled_total_force_stack = vmap_p2g(
            shapefunctions.intr_id_stack,
            shapefunctions.intr_shapef_stack,
            shapefunctions.intr_shapef_grad_stack,
            intr_dist_3d_stack,
        )

        nodes_mass_stack = nodes.mass_stack.at[shapefunctions.intr_hash_stack].add(
            scaled_mass_stack
        )

        nodes_moment_stack = nodes.moment_stack.at[shapefunctions.intr_hash_stack].add(
            scaled_moment_stack
        )

        nodes_force_stack = (
            jnp.zeros_like(nodes.moment_nt_stack)
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
        self: Self,
        particles: Particles,
        nodes: Nodes,
        shapefunctions: ShapeFunction,
        intr_dist_3d_stack: jax.Array,
    ) -> Tuple[Particles, Self]:
        stencil_size, dim = shapefunctions.stencil.shape

        padding = (0, 3 - dim)

        @partial(jax.vmap, in_axes=(0, 0, 0, 0))
        def vmap_intr_scatter(intr_hashes, intr_shapef, intr_shapef_grad, intr_dist_3d):
            intr_masses = nodes.mass_stack.at[intr_hashes].get()
            intr_moments_nt = nodes.moment_nt_stack.at[intr_hashes].get()

            intr_vels_nt = jax.lax.cond(
                intr_masses > nodes.small_mass_cutoff,
                lambda x: x / intr_masses,
                lambda x: jnp.zeros_like(x),
                intr_moments_nt,
            )

            intr_scaled_vels_nt = intr_shapef * intr_vels_nt

            # Pad velocities for plane strain
            intr_vels_nt_padded = jnp.pad(
                intr_vels_nt,
                padding,
                mode="constant",
                constant_values=0,
            )

            # APIC affine matrix
            intr_Bp = (
                intr_shapef
                * intr_vels_nt_padded.reshape(-1, 1)
                @ intr_dist_3d.reshape(-1, 1).T
            )

            intr_scaled_velgrad = intr_Bp @ self.Dp_inv

            return intr_scaled_vels_nt, intr_scaled_velgrad, intr_Bp

        @partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, 0))
        def vmap_particles_update(
            intr_vels_nt_reshaped,
            intr_velgrad_reshaped,
            intr_Bp,
            p_positions,
            p_F,
            p_volumes_orig,
        ):
            # Update particle quantities
            p_velgrads_next = jnp.sum(intr_velgrad_reshaped, axis=0)

            vels_nt = jnp.sum(intr_vels_nt_reshaped, axis=0)

            p_Bp_next = jnp.sum(intr_Bp, axis=0)
            if dim == 2:
                p_Bp_next = p_Bp_next.at[2, 2].set(0)

            p_velocities_next = vels_nt

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
                p_Bp_next,
            )

        intr_scaled_vels_nt_stack, intr_scaled_velgrad_stack, intr_Bp_stack = (
            vmap_intr_scatter(
                shapefunctions.intr_hash_stack,
                shapefunctions.intr_shapef_stack,
                shapefunctions.intr_shapef_grad_stack,
                intr_dist_3d_stack,
            )
        )

        (
            p_velocities_next_stack,
            p_positions_next_stack,
            p_F_next_stack,
            p_volumes_next_stack,
            p_velgrads_next_stack,
            p_Bp_next_stack,
        ) = vmap_particles_update(
            intr_scaled_vels_nt_stack.reshape(-1, stencil_size, dim),
            intr_scaled_velgrad_stack.reshape(-1, stencil_size, 3, 3),
            intr_Bp_stack.reshape(-1, stencil_size, 3, 3),
            particles.position_stack,
            particles.F_stack,
            particles.volume0_stack,
        )

        return particles.replace(
            velocity_stack=p_velocities_next_stack,
            position_stack=p_positions_next_stack,
            F_stack=p_F_next_stack,
            volume_stack=p_volumes_next_stack,
            L_stack=p_velgrads_next_stack,
        ), self.replace(Bp_stack=p_Bp_next_stack)
