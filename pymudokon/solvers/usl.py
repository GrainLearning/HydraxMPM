"""Implementation of the Explicit Update Stress Last (USL) Material Point Method (MPM).

References:
    - De Vaucorbeil, Alban, et al. 'Material point method after 25 years: theory, implementation, and applications.'
"""

from functools import partial
from typing import List

import jax
import jax.numpy as jnp
from flax import struct
from typing_extensions import Self

from ..core.nodes import Nodes
from ..core.particles import Particles
from ..forces.forces import Forces
from ..materials.material import Material
from ..shapefunctions.shapefunction import ShapeFunction
from .solver import Solver


@jax.jit
def p2g(
    nodes: Nodes,
    particles: Particles,
    shapefunctions: ShapeFunction,
    dt: jnp.float32,
) -> Nodes:
    stencil_size, dim = shapefunctions.stencil.shape

    padding = (0, 3 - dim)

    @partial(jax.vmap, in_axes=(0, 0, 0))
    def vmap_p2g(intr_id, intr_shapef, intr_shapef_grad):
        particle_id = (intr_id / stencil_size).astype(jnp.int32)

        intr_masses = particles.masses.at[particle_id].get()
        intr_volumes = particles.volumes.at[particle_id].get()
        intr_velocities = particles.velocities.at[particle_id].get()
        intr_ext_forces = particles.forces.at[particle_id].get()
        intr_stresses = particles.stresses.at[particle_id].get()

        intr_shapef_grad = jnp.pad(
            intr_shapef_grad,
            padding,
            mode="constant",
            constant_values=0,
        )

        scaled_mass = intr_shapef * intr_masses
        scaled_moments = scaled_mass * intr_velocities
        scaled_ext_force = intr_shapef * intr_ext_forces
        scaled_int_force = -1.0 * intr_volumes * intr_stresses @ intr_shapef_grad


        scaled_total_force = scaled_int_force[:dim] + scaled_ext_force

        # scaled_total_force = scaled_int_force[:dim]
        # jax.debug.print("particle_id {}, node_id {}, intr_shapef_grad {}",particle_id,intr_hash, intr_shapef_grad)
        # jax.debug.print("particle_id {}, node_id {}, scaled_total_force {}",particle_id,intr_hash, scaled_total_force)
        return scaled_mass, scaled_moments, scaled_total_force

    scaled_mass, scaled_moments, scaled_total_force = vmap_p2g(
        shapefunctions.intr_ids, shapefunctions.intr_shapef, shapefunctions.intr_shapef_grad
    )

    nodes_masses = nodes.masses.at[shapefunctions.intr_hashes].add(scaled_mass)

    nodes_moments = nodes.moments.at[shapefunctions.intr_hashes].add(scaled_moments)

    nodes_forces = jnp.zeros_like(nodes.moments_nt).at[shapefunctions.intr_hashes].add(scaled_total_force)
    # jax.debug.print("nodes_forces {}",nodes_forces)
    nodes_moments_nt = nodes_moments + nodes_forces * dt
    
    return nodes.replace(
        masses=nodes_masses,
        moments=nodes_moments,
        moments_nt=nodes_moments_nt,
    )


@jax.jit
def g2p(
    nodes: Nodes,
    particles: Particles,
    shapefunctions: ShapeFunction,
    alpha: jnp.float32,
    dt: jnp.float32,
) -> Particles:
    """Grid to particle transfer.

    Procedure is as follows:
    - Prepare shape functions and their gradients.
    - Calculate node velocities (by dividing moments with mass)
    - Scatter node quantities to particle-node interactions.
    - Calculate interaction velocities and velocity gradients.
    - Sum interaction quantities to particles.
    - Update particle quantities
    (e.g., velocities, positions, volumes, velocity gradients, deformation gradients, etc.)


    Args:
        nodes (Nodes): MPM background nodes.
        particles (Particles): Material points / particles
        shapefunctions (Interactions): A shape function e.g. `LinearShapeFunction`.
        alpha (jnp.float32): Flip & PIC ratio.
        dt (jnp.float32): Time step

    Returns:
        Particles: Updated particles
    """
    _, dim = shapefunctions.stencil.shape

    @partial(jax.vmap, in_axes=(0, 0, 0))
    def vmap_intr_scatter(intr_hashes, intr_shapef, intr_shapef_grad):
        intr_masses = nodes.masses.at[intr_hashes].get()
        intr_moments = nodes.moments.at[intr_hashes].get()
        intr_moments_nt = nodes.moments_nt.at[intr_hashes].get()

        intr_vels = jax.lax.cond(
            intr_masses > nodes.small_mass_cutoff, 
            lambda x: x / intr_masses, 
            lambda x: jnp.zeros_like(x), 
            intr_moments
        )

        intr_vels_nt = jax.lax.cond(
            intr_masses > nodes.small_mass_cutoff,
            lambda x: x / intr_masses,
            lambda x: jnp.zeros_like(x),
            intr_moments_nt,
        )
        intr_delta_vels = intr_vels_nt - intr_vels

        intr_scaled_delta_vels = intr_shapef*intr_delta_vels 

        intr_scaled_vels_nt = intr_shapef*intr_vels_nt
        # jax.debug.print("intr_moments_nt {}",intr_moments_nt)
        # jax.debug.print("intr_masses {}",intr_masses)
        # jax.debug.print("intr_vels_nt {}",intr_vels_nt)
        intr_scaled_velgrad = intr_shapef_grad.reshape(-1, 1) @ intr_vels_nt.reshape(-1, 1).T

        return intr_scaled_delta_vels, intr_scaled_vels_nt, intr_scaled_velgrad

    @partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, 0, 0))
    def vmap_particles_update(
        intr_delta_vels_reshaped,
        intr_vels_nt_reshaped,
        intr_velgrad_reshaped,
        p_velocities,
        p_positions,
        p_F,
        p_volumes_orig,
    ):
        # Update particle quantities
        p_velgrads_next = jnp.sum(intr_velgrad_reshaped, axis=0)

        # jax.debug.print("p_velgrads_next {}",p_velgrads_next)

        delta_vels = jnp.sum(intr_delta_vels_reshaped, axis=0)
        vels_nt = jnp.sum(intr_vels_nt_reshaped, axis=0)
        p_velocities_next = (1.0 - alpha) * vels_nt + alpha * (p_velocities + delta_vels)

        p_positions_next = p_positions + vels_nt * dt

        p_F_next = (jnp.eye(dim) + p_velgrads_next * dt)@p_F

        p_volumes_next =  jnp.linalg.det(p_F_next)*p_volumes_orig
        return p_velocities_next, p_positions_next, p_F_next, p_volumes_next, p_velgrads_next

    intr_scaled_delta_vels, intr_scaled_vels_nt, intr_scaled_velgrad = vmap_intr_scatter(
        shapefunctions.intr_hashes, shapefunctions.intr_shapef, shapefunctions.intr_shapef_grad
    )

    p_velocities_next, p_positions_next, p_F_next, p_volumes_next, p_velgrads_next = vmap_particles_update(
        intr_scaled_delta_vels.reshape(-1, *shapefunctions.stencil.shape),
        intr_scaled_vels_nt.reshape(-1, *shapefunctions.stencil.shape),
        intr_scaled_velgrad.reshape(-1, *shapefunctions.stencil.shape, shapefunctions.stencil.shape[1]),
        particles.velocities,
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
    )

    # # Sum interactions to particles
    # particle_delta_vel_inc = jnp.sum(intr_scaled_delta_vels.reshape(-1, stencil_size, dim), axis=1)

    # particle_vel_inc = jnp.sum(intr_scaled_vels_nt.reshape(-1, stencil_size, dim), axis=1)

    # particle_velgrads = jnp.sum(intr_scaledgrad_vels_nt.reshape(-1, stencil_size, dim, dim), axis=1)

    # # Update particle quantities
    # particles_velocities = (1.0 - alpha) * particle_vel_inc + alpha * (particles.velocities + particle_delta_vel_inc)

    # particles_positions = particles.positions + particle_vel_inc * dt

    # # deformation gradient and volume update
    # F = jnp.repeat(jnp.eye(dim).reshape(1, dim, dim), num_particles, axis=0) + particle_velgrads * dt

    # F = jax.lax.batch_matmul(F, particles.F)

    # J = jnp.linalg.det(F)

    # particles_volume = particles.volumes_original * J

    # return particles

    # return particles.replace(
    #     velocities=particles_velocities,
    #     positions=particles_positions,
    #     F=F,
    #     volumes=particles_volume,
    #     velgrads=particle_velgrads,
    # )


@struct.dataclass
class USL(Solver):
    """Explicit Update Stress Last (USL) MPM solver.

    Inherits data from `Solver`.

    Attributes:
        alpha: Flip & PIC ratio.
    """

    alpha: jnp.float32

    @classmethod
    def create(
        cls: Self,
        particles: Particles,
        nodes: Nodes,
        shapefunctions: ShapeFunction,
        materials: List[Material],
        forces: List[Forces] = None,
        alpha: jnp.float32 = 0.98,
        dt: jnp.float32 = 0.00001,
    ) -> Self:
        """Create a USL solver.

        Args:
            cls (Self): self reference.
            particles (Particles): Particles in the simulation.
            nodes (Nodes): Nodes in the simulation.
            shapefunctions (ShapeFunction): Shape functions in the simulation, e.g., `LinearShapeFunction`.
            materials (List[Material]): List of materials in the simulation, e.g., `LinearIsotropicElastic`.
            forces (List[Force], optional): List of forces. Defaults to None.
            alpha (jnp.float32, optional): FLIP-PIC ratio. Defaults to 0.99.
            dt (jnp.float32, optional): Time step. Defaults to 0.00001.

        Returns:
            USL: Initialized USL solver.

        Example:
            >>> import pymudokon as pm
            >>> particles = pm.Particles.create(positions=jnp.array([[1.0, 2.0], [0.3, 0.1]]))
            >>> nodes = pm.Nodes.create(
            ...     origin=jnp.array([0.0, 0.0]),
            ...     end=jnp.array([1.0, 1.0]),
            ...     node_spacing=0.5,
            ... )
            >>> shapefunctions = pm.LinearShapeFunction.create(2, 2)
            >>> material = pm.LinearIsotropicElastic.create(E=1000.0, nu=0.2, num_particles=2, dim=2)
            >>> particles, nodes, shapefunctions = pm.Discretize(particles, nodes, shapefunctions)
            >>> usl = pm.USL.create(
            ...     particles=particles,
            ...     nodes=nodes,
            ...     shapefunctions=shapefunctions,
            ...     materials=[material],
            ...     alpha=0.1,
            ...     dt=0.001,
            ... )
        """
        if forces is None:
            forces = []

        return cls(
            particles=particles,
            nodes=nodes,
            shapefunctions=shapefunctions,
            materials=materials,
            forces=forces,
            alpha=alpha,
            dt=dt,
        )

    @jax.jit
    def update(self: Self) -> Self:
        """Solve 1 iteration of USL MPM.

        Args:
            self (Self): Current USL solver.

        Returns:
            Self: Updated USL solver.
        """
        nodes = self.nodes.refresh()

        particles = self.particles.refresh()

        shapefunctions = self.shapefunctions.calculate_shapefunction(nodes=nodes, particles=particles)

        nodes = p2g(
            nodes=nodes,
            particles=particles,
            shapefunctions=shapefunctions,
            dt=self.dt,
        )

        # for loop is statically unrolled, may result in large compile times
        # for many materials/forces
        forces = []
        for force in self.forces:
            nodes, out_force = force.apply_on_nodes_moments(
                particles=particles, nodes=nodes, shapefunctions=shapefunctions, dt=self.dt
            )
            forces.append(out_force)
        # forces = self.forces

        particles = g2p(
            particles=particles,
            nodes=nodes,
            shapefunctions=shapefunctions,
            alpha=self.alpha,
            dt=self.dt,
        )

        materials = []
        for mat in self.materials:
            particles, out_mat = mat.update_stress(particles=particles, dt=self.dt)
            materials.append(out_mat)

        return self.replace(
            particles=particles,
            nodes=nodes,
            materials=materials,
            forces=forces,
            shapefunctions=shapefunctions,
        )
