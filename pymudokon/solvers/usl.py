"""Implementation of the Explicit Update Stress Last (USL) Material Point Method (MPM).

References:
    - De Vaucorbeil, Alban, et al. 'Material point method after 25 years: theory, implementation, and applications.'
"""

from typing import List
from flax import struct
import jax
import jax.numpy as jnp
from typing_extensions import Self

from ..shapefunctions.shapefunction import ShapeFunction
from ..core.nodes import Nodes
from ..core.particles import Particles
from ..forces.forces import Forces
from ..materials.material import Material
from .solver import Solver
from functools import partial

@jax.jit
def p2g(
    nodes: Nodes,
    particles: Particles,
    shapefunctions: ShapeFunction,
    dt: jnp.float32,
) -> Nodes:
    """Particle to grid transfer.

    The procedure is as follows:

    - Prepare shape functions and and their gradients.
    - Map out shapefunction for a particle and its stencil.
    - Scale quantities with shape functions and gradients.
    - Gather quantities to nodes.
    - Integrate moments on nodes.

    Args:
        nodes (Nodes): MPM background nodes.
        particles (Particles): Material points / particles
        shapefunctions (ShapeFunction): A shape function e.g. `LinearShapeFunction`
        dt (jnp.float32):
            Time step.

    Returns:
        Nodes: Updated nodes
    """
    stencil_size, dim = shapefunctions.stencil.shape

    intr_shapef = shapefunctions.intr_shapef

    # padding is applied if we are in plane strain
    intr_shapef_grad = jnp.pad(
        shapefunctions.intr_shapef_grad,
        ((0, 0), (0, 1), (0, 0)),
        mode="constant",
        constant_values=0,
    )  # plane strain #TODO - generalize for 3D

    # interactions
    intr_masses = jnp.repeat(particles.masses, stencil_size).reshape(-1, 1, 1)
    intr_volumes = jnp.repeat(particles.volumes, stencil_size).reshape(-1, 1, 1)
    intr_velocities = jnp.repeat(particles.velocities, stencil_size, axis=0).reshape(-1, dim, 1)
    intr_ext_forces = jnp.repeat(particles.forces, stencil_size, axis=0).reshape(-1, dim, 1)
    intr_stresses = jnp.repeat(particles.stresses, stencil_size, axis=0).reshape(-1, 3, 3)

    # scaled quantities with shape functions and gradients
    scaled_mass = intr_masses * intr_shapef
    scaled_moments = jax.lax.batch_matmul(intr_velocities, scaled_mass)
    scaled_ext_forces = jax.lax.batch_matmul(intr_ext_forces, intr_shapef)
    scaled_int_forces = jax.lax.batch_matmul(intr_stresses, intr_shapef_grad)
    scaled_int_forces = jax.lax.batch_matmul(scaled_int_forces, -1.0 * intr_volumes)
    scaled_total_forces = scaled_int_forces[:, :2] + scaled_ext_forces  # unpad and add #TODO - generalize for 3D
        
    # node (reshape) and gather interactions
    nodes_masses = nodes.masses.at[shapefunctions.intr_hashes,].add(scaled_mass.reshape(-1), mode="drop")

    nodes_moments = nodes.moments.at[shapefunctions.intr_hashes].add(scaled_moments.reshape(-1, 2), mode="drop")

    nodes_forces = (
        jnp.zeros_like(nodes.moments_nt)
        .at[shapefunctions.intr_hashes]
        .add(scaled_total_forces.reshape(-1, 2), mode="drop")
    )

    # integrate node moments
    nodes_moments_nt = nodes_moments + nodes_forces * dt

    return nodes.replace(
        masses=nodes_masses,
        moments=nodes_moments,
        moments_nt=nodes_moments_nt,
    )

@jax.jit
def p2g_batch(
        nodes: Nodes,
        particles: Particles,
        shapefunctions: ShapeFunction,
        dt: jnp.float32,
    )-> Nodes:
    
    stencil_size, dim = shapefunctions.stencil.shape

    intr_shapef = shapefunctions.intr_shapef

    # padding is applied if we are in plane strain
    intr_shapef_grad = jnp.pad(
        shapefunctions.intr_shapef_grad,
        ((0, 0), (0, 1), (0, 0)),
        mode="constant",
        constant_values=0,
    )  # plane strain #TODO - generalize for 3D

    # interactions
    p_ids = jnp.repeat(particles.ids, stencil_size).reshape(-1)
    
        # interactions
    # intr_masses = jnp.repeat(particles.masses, stencil_size).reshape(-1, 1, 1)
    
    @partial(jax.vmap, in_axes=(0,0,0,None,None,None,None, None))    
    def vmap_p2g(
        p_ids,
        intr_shapef,
        intr_shapef_grad,
        p_masses, p_volumes, p_velocities, p_forces, p_stresses):
        
        intr_masses = p_masses.at[p_ids].get(indices_are_sorted=True, unique_indices=True)
        intr_volumes = p_volumes.at[p_ids].get(indices_are_sorted=True, unique_indices=True)
        intr_velocities = p_velocities.at[p_ids].get(indices_are_sorted=True, unique_indices=True)
        intr_ext_forces = p_forces.at[p_ids].get(indices_are_sorted=True, unique_indices=True)
        intr_stresses = p_stresses.at[p_ids].get(indices_are_sorted=True, unique_indices=True)
        
        scaled_mass = intr_shapef*intr_masses
        scaled_moments = scaled_mass*intr_velocities
        scaled_ext_force = scaled_mass*intr_ext_forces

        scaled_int_force = -1.0*intr_volumes*intr_stresses@intr_shapef_grad

        scaled_total_force = scaled_int_force[:2,0] + scaled_ext_force

        return scaled_mass,scaled_moments,scaled_total_force
    
    scaled_mass,scaled_moments,scaled_total_force = vmap_p2g(p_ids,
                            intr_shapef.reshape(-1,1),
                            intr_shapef_grad,
                            particles.masses,
                            particles.volumes,
                            particles.velocities,
                            particles.forces,
                            particles.stresses
                            )
 
    # # node (reshape) and gather interactions
    nodes_masses = nodes.masses.at[shapefunctions.intr_hashes,].add(scaled_mass.reshape(-1), mode="drop")

    nodes_moments = nodes.moments.at[shapefunctions.intr_hashes].add(scaled_moments.reshape(-1, 2), mode="drop")

    nodes_forces = (
        jnp.zeros_like(nodes.moments_nt)
        .at[shapefunctions.intr_hashes]
        .add(scaled_total_force.reshape(-1, 2), mode="drop")
    )

    # integrate node moments
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
    num_particles, dim = particles.positions.shape
    stencil_size = shapefunctions.stencil.shape[0]
    # Prepare shape functions

    intr_shapef = shapefunctions.intr_shapef
    intr_shapef_grad = shapefunctions.intr_shapef_grad
    intr_shapef_grad_T = jnp.transpose(intr_shapef_grad, axes=(0, 2, 1))

    # Calculate the node velocities
    # small masses may cause numerical instability
    # so we set zero velocities at these nodes
    node_masses_reshaped = nodes.masses.reshape(-1, 1)
    nodes_velocities = nodes.moments / node_masses_reshaped

    nodes_velocities = jnp.where(
        node_masses_reshaped > nodes.small_mass_cutoff,
        nodes_velocities,
        jnp.zeros_like(nodes_velocities),
    )
    nodes_velocities_nt = nodes.moments_nt / node_masses_reshaped

    nodes_velocities_nt = jnp.where(
        node_masses_reshaped > nodes.small_mass_cutoff,
        nodes_velocities_nt,
        jnp.zeros_like(nodes_velocities_nt),
    )

    # Scatter node quantities to particle-node interactions
    intr_vels = jnp.take(nodes_velocities, shapefunctions.intr_hashes, axis=0, mode="fill", fill_value=0.0).reshape(
        -1, dim, 1
    )

    intr_vels_nt = jnp.take(
        nodes_velocities_nt, shapefunctions.intr_hashes, axis=0, mode="fill", fill_value=0.0
    ).reshape(-1, dim, 1)

    # Calculate the interaction velocities and velocity gradients
    intr_delta_vels = intr_vels_nt - intr_vels

    intr_scaled_delta_vels = intr_delta_vels * intr_shapef

    intr_scaled_vels_nt = intr_vels_nt * intr_shapef

    intr_scaledgrad_vels_nt = jax.lax.batch_matmul(intr_vels_nt, intr_shapef_grad_T)

    # Sum interactions to particles
    particle_delta_vel_inc = jnp.sum(intr_scaled_delta_vels.reshape(-1, stencil_size, 2), axis=1)

    particle_vel_inc = jnp.sum(intr_scaled_vels_nt.reshape(-1, stencil_size, 2), axis=1)

    particle_velgrads = jnp.sum(intr_scaledgrad_vels_nt.reshape(-1, stencil_size, 2, 2), axis=1)

    # Update particle quantities
    particles_velocities = (1.0 - alpha) * particle_vel_inc + alpha * (particles.velocities + particle_delta_vel_inc)

    particles_positions = particles.positions + particle_vel_inc * dt

    # deformation gradient and volume update
    F = jnp.repeat(jnp.eye(dim).reshape(1, dim, dim), num_particles, axis=0) + particle_velgrads * dt

    F = jax.lax.batch_matmul(F, particles.F)

    J = jnp.linalg.det(F)

    particles_volume = particles.volumes_original * J

    return particles.replace(
        velocities=particles_velocities,
        positions=particles_positions,
        F=F,
        volumes=particles_volume,
        velgrads=particle_velgrads,
    )

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
        alpha: jnp.float32 = 0.99,
        dt: jnp.float32 = 0.00001
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
            dt=dt
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

        shapefunctions = self.shapefunctions.get_interactions(particles=particles, nodes=nodes)

        shapefunctions = shapefunctions.calculate_shapefunction(nodes=nodes)

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
