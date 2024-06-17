"""Implementation of the Explicit Update Stress Last (USL) Material Point Method (MPM).

References:
    - Jiang, C., Schroeder, C., Teran, J., Stomakhin, A. and Selle, A., 2016. The material point method for simulating continuum materials. In Acm siggraph 2016 courses (pp. 1-52).
    - Jiang, C., Schroeder, C., Selle, A., Teran, J. and Stomakhin, A., 2015. The affine particle-in-cell method. ACM Transactions on Graphics (TOG), 34(4), pp.1-10.
"""

from functools import partial
from typing import List, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from typing_extensions import Self

from ..core.nodes import Nodes
from ..core.particles import Particles
from ..forces.forces import Forces
from ..materials.material import Material
from ..shapefunctions.shapefunction import ShapeFunction
from ..shapefunctions.cubic import CubicShapeFunction
from ..shapefunctions.linear import LinearShapeFunction
from .solver import Solver


@struct.dataclass
class USL_APIC(Solver):
    """Explicit Update Stress Last (USL) Affine Particle in Cell (APIC) MPM solver.

    Inherits data from `Solver`.

    Attributes:
        Dp: APIC interpolation stencil.
    """

    Dp: jax.Array
    Bp: jax.Array

    @classmethod
    def create(
        cls: Self,
        particles: Particles,
        nodes: Nodes,
        shapefunctions: ShapeFunction,
        materials: List[Material],
        forces: List[Forces] = None,
        dt: jnp.float32 = 0.00001,
        Bp: jax.Array = None,
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
            ...     Dp=0.1,
            ...     dt=0.001,
            ... )
        """
        if forces is None:
            forces = []

        num_particles, dim = particles.positions.shape

        if isinstance(shapefunctions, LinearShapeFunction):
            raise ValueError("APIC with linear shape functions not supported.")
        elif isinstance(shapefunctions, CubicShapeFunction):
            Dp = (1.0 / 3.0) * nodes.node_spacing * nodes.node_spacing * jnp.eye(dim)
        else:
            raise ValueError("Shape function not implemented.")

        if Bp is None:
            Bp = jnp.zeros((num_particles, dim, dim))

        return cls(
            particles=particles,
            nodes=nodes,
            shapefunctions=shapefunctions,
            materials=materials,
            forces=forces,
            Dp=Dp,
            dt=dt,
            Bp=Bp,
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

        shapefunctions, intr_dist = self.shapefunctions.calculate_shapefunction(
            nodes=nodes, positions=particles.positions
        )

        # transform from grid space to particle space
        intr_dist = -1.0 * intr_dist * nodes.node_spacing

        nodes = self.p2g(nodes=nodes, particles=particles, shapefunctions=shapefunctions, intr_dist=intr_dist)

        # for loop is statically unrolled, may result in large compile times
        # for many materials/forces
        forces = []
        for force in self.forces:
            nodes, out_force = force.apply_on_nodes_moments(
                particles=particles, nodes=nodes, shapefunctions=shapefunctions, dt=self.dt
            )
            forces.append(out_force)
        # forces = self.forces

        particles, self = self.g2p(
            particles=particles,
            nodes=nodes,
            shapefunctions=shapefunctions,
            intr_dist=intr_dist,
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

    @jax.jit
    def p2g(
        self: Self, nodes: Nodes, particles: Particles, shapefunctions: ShapeFunction, intr_dist: jax.Array
    ) -> Nodes:
        stencil_size, dim = shapefunctions.stencil.shape

        padding = (0, 3 - dim)

        @partial(jax.vmap, in_axes=(0, 0, 0, 0))
        def vmap_p2g(intr_id, intr_shapef, intr_shapef_grad, intr_dist):
            particle_id = (intr_id / stencil_size).astype(jnp.int32)

            intr_masses = particles.masses.at[particle_id].get()
            intr_volumes = particles.volumes.at[particle_id].get()
            intr_velocities = particles.velocities.at[particle_id].get()
            intr_ext_forces = particles.forces.at[particle_id].get()
            intr_stresses = particles.stresses.at[particle_id].get()

            intr_Bp = self.Bp.at[particle_id].get()  # APIC affine matrix

            affine_velocity = (intr_Bp @ jnp.linalg.inv(self.Dp)) @ intr_dist

            intr_shapef_grad = jnp.pad(
                intr_shapef_grad,
                padding,
                mode="constant",
                constant_values=0,
            )

            scaled_mass = intr_shapef * intr_masses
            scaled_moments = scaled_mass * (intr_velocities + affine_velocity)
            scaled_ext_force = intr_shapef * intr_ext_forces
            scaled_int_force = -1.0 * intr_volumes * intr_stresses @ intr_shapef_grad

            scaled_total_force = scaled_int_force[:dim] + scaled_ext_force

            return scaled_mass, scaled_moments, scaled_total_force

        scaled_mass, scaled_moments, scaled_total_force = vmap_p2g(
            shapefunctions.intr_ids, shapefunctions.intr_shapef, shapefunctions.intr_shapef_grad, intr_dist
        )

        nodes_masses = nodes.masses.at[shapefunctions.intr_hashes].add(scaled_mass)

        nodes_moments = nodes.moments.at[shapefunctions.intr_hashes].add(scaled_moments)

        nodes_forces = jnp.zeros_like(nodes.moments_nt).at[shapefunctions.intr_hashes].add(scaled_total_force)

        nodes_moments_nt = nodes_moments + nodes_forces * self.dt

        return nodes.replace(
            masses=nodes_masses,
            moments=nodes_moments,
            moments_nt=nodes_moments_nt,
        )

    @jax.jit
    def g2p(
        self: Self, nodes: Nodes, particles: Particles, shapefunctions: ShapeFunction, intr_dist: jax.Array
    ) -> Tuple[Particles, Self]:
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

        @partial(jax.vmap, in_axes=(0, 0, 0, 0))
        def vmap_intr_scatter(intr_hashes, intr_shapef, intr_shapef_grad, intr_dist):
            intr_masses = nodes.masses.at[intr_hashes].get()
            intr_moments_nt = nodes.moments_nt.at[intr_hashes].get()

            intr_vels_nt = jax.lax.cond(
                intr_masses > nodes.small_mass_cutoff,
                lambda x: x / intr_masses,
                lambda x: jnp.zeros_like(x),
                intr_moments_nt,
            )

            intr_scaled_vels_nt = intr_shapef * intr_vels_nt

            intr_scaled_velgrad = intr_shapef_grad.reshape(-1, 1) @ intr_vels_nt.reshape(-1, 1).T

            # APIC affine matrix
            intr_Bp = intr_shapef * intr_vels_nt.reshape(-1, 1) @ intr_dist.reshape(-1, 1).T

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

            p_velocities_next = vels_nt

            p_positions_next = p_positions + vels_nt * self.dt

            p_F_next = (jnp.eye(dim) + p_velgrads_next * self.dt) @ p_F

            p_volumes_next = jnp.linalg.det(p_F_next) * p_volumes_orig
            return p_velocities_next, p_positions_next, p_F_next, p_volumes_next, p_velgrads_next, p_Bp_next

        intr_scaled_vels_nt, intr_scaled_velgrad, intr_Bp = vmap_intr_scatter(
            shapefunctions.intr_hashes, shapefunctions.intr_shapef, shapefunctions.intr_shapef_grad, intr_dist
        )

        p_velocities_next, p_positions_next, p_F_next, p_volumes_next, p_velgrads_next, p_Bp_next = (
            vmap_particles_update(
                intr_scaled_vels_nt.reshape(-1, *shapefunctions.stencil.shape),
                intr_scaled_velgrad.reshape(-1, *shapefunctions.stencil.shape, shapefunctions.stencil.shape[1]),
                intr_Bp.reshape(-1, *shapefunctions.stencil.shape, shapefunctions.stencil.shape[1]),
                particles.positions,
                particles.F,
                particles.volumes_original,
            )
        )

        return particles.replace(
            velocities=p_velocities_next,
            positions=p_positions_next,
            F=p_F_next,
            volumes=p_volumes_next,
            velgrads=p_velgrads_next,
        ), self.replace(Bp=p_Bp_next)
