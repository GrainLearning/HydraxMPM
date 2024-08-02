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
    """Explicit Update Stress Last (USL) Affine Particle in Cell (APIC) MPM solver."""

    Dp: chex.Array
    Dp_inv: chex.Array
    Bp: chex.Array

    @classmethod
    def create(
        cls,
        cell_size,
        dim,
        num_particles,
        dt: jnp.float32 = 0.00001,
    ):
        jax.debug.print("USL_APIC solver supported for cubic shape functions only")
        Dp = (1.0 / 3.0) * cell_size * cell_size * jnp.eye(3)

        Dp_inv = jnp.linalg.inv(Dp)

        Bp = jnp.zeros((num_particles, 3, 3))

        return USL_APIC(dt=dt, Dp=Dp, Dp_inv=Dp_inv, Bp=Bp)

    def update(self, particles, nodes, shapefunctions, material_stack, forces_stack):
        nodes = nodes.refresh()
        particles = particles.refresh()

        shapefunctions, intr_dist_3d = shapefunctions.calculate_shapefunction(
            origin=nodes.origin,
            inv_node_spacing=nodes.inv_node_spacing,
            grid_size=nodes.grid_size,
            positions=particles.positions,
        )

        # transform from grid space to particle space
        intr_dist_3d = -1.0 * intr_dist_3d * nodes.node_spacing

        nodes = self.p2g(
            particles=particles,
            nodes=nodes,
            shapefunctions=shapefunctions,
            intr_dist_3d=intr_dist_3d,
        )

        new_forces_stack = []
        for forces in forces_stack:
            nodes, forces = forces.apply_on_nodes_moments(
                particles=particles,
                nodes=nodes,
                shapefunctions=shapefunctions,
                dt=self.dt,
            )
            new_forces_stack.append(forces)

        particles, self = self.g2p(
            particles=particles,
            nodes=nodes,
            shapefunctions=shapefunctions,
            intr_dist_3d=intr_dist_3d,
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
        intr_dist_3d: jax.Array,
    ) -> Nodes:
        stencil_size, dim = shapefunctions.stencil.shape

        @partial(jax.vmap, in_axes=(0, 0, 0, 0))
        def vmap_p2g(intr_id, intr_shapef, intr_shapef_grad, intr_dist_3d):
            particle_id = (intr_id / stencil_size).astype(jnp.int32)

            intr_masses = particles.masses.at[particle_id].get()
            intr_volumes = particles.volumes.at[particle_id].get()
            intr_velocities = particles.velocities.at[particle_id].get()
            intr_ext_forces = particles.forces.at[particle_id].get()
            intr_stresses = particles.stresses.at[particle_id].get()

            intr_Bp = self.Bp.at[particle_id].get()  # APIC affine matrix

            affine_velocity = (intr_Bp @ jnp.linalg.inv(self.Dp)) @ intr_dist_3d

            scaled_mass = intr_shapef * intr_masses
            scaled_moments = scaled_mass * (
                intr_velocities + affine_velocity.at[:dim].get()
            )
            scaled_ext_force = intr_shapef * intr_ext_forces
            scaled_int_force = -1.0 * intr_volumes * intr_stresses @ intr_shapef_grad

            scaled_total_force = scaled_int_force.at[:dim].get() + scaled_ext_force

            return scaled_mass, scaled_moments, scaled_total_force

        scaled_mass, scaled_moments, scaled_total_force = vmap_p2g(
            shapefunctions.intr_ids,
            shapefunctions.intr_shapef,
            shapefunctions.intr_shapef_grad,
            intr_dist_3d,
        )

        nodes_masses = nodes.masses.at[shapefunctions.intr_hashes].add(scaled_mass)

        nodes_moments = nodes.moments.at[shapefunctions.intr_hashes].add(scaled_moments)

        nodes_forces = (
            jnp.zeros_like(nodes.moments_nt)
            .at[shapefunctions.intr_hashes]
            .add(scaled_total_force)
        )

        nodes_moments_nt = nodes_moments + nodes_forces * self.dt

        return nodes.replace(
            masses=nodes_masses,
            moments=nodes_moments,
            moments_nt=nodes_moments_nt,
        )

    @jax.jit
    def g2p(
        self: Self,
        particles: Particles,
        nodes: Nodes,
        shapefunctions: ShapeFunction,
        intr_dist_3d: jax.Array,
    ) -> Tuple[Particles, Self]:
        stencil_size, dim = shapefunctions.stencil.shape

        padding = (0, 3 - dim)

        @partial(jax.vmap, in_axes=(0, 0, 0, 0))
        def vmap_intr_scatter(intr_hashes, intr_shapef, intr_shapef_grad, intr_dist_3d):
            intr_masses = nodes.masses.at[intr_hashes].get()
            intr_moments_nt = nodes.moments_nt.at[intr_hashes].get()

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

        intr_scaled_vels_nt, intr_scaled_velgrad, intr_Bp = vmap_intr_scatter(
            shapefunctions.intr_hashes,
            shapefunctions.intr_shapef,
            shapefunctions.intr_shapef_grad,
            intr_dist_3d,
        )

        (
            p_velocities_next,
            p_positions_next,
            p_F_next,
            p_volumes_next,
            p_velgrads_next,
            p_Bp_next,
        ) = vmap_particles_update(
            intr_scaled_vels_nt.reshape(-1, stencil_size, dim),
            intr_scaled_velgrad.reshape(-1, stencil_size, 3, 3),
            intr_Bp.reshape(-1, stencil_size, 3, 3),
            particles.positions,
            particles.F,
            particles.volumes_original,
        )

        return particles.replace(
            velocities=p_velocities_next,
            positions=p_positions_next,
            F=p_F_next,
            volumes=p_volumes_next,
            velgrads=p_velgrads_next,
        ), self.replace(Bp=p_Bp_next)
