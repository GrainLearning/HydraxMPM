"""State and functions for the material points (called particles)."""

import dataclasses

import jax
import jax.numpy as jnp
from jax import Array
from typing_extensions import Self

from .base import Base


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=False)
class Particles(Base):
    """Dataclass containing the state of the MPM particles.

    Attributes:
        original_density (jnp.float32): Original density.
        positions (Array): Spatial coordinate vectors `(num_particles, dimension)`.
        velocities (Array): Spatial velocity vectors `(num_particles, dimension)`.
        masses (Array): Masses `(num_particles,)`.
        species (Array): Material ID or type `(num_particles,)`.
        volumes (Array): Current volumes  `(num_particles,)`.
        volumes_original (Array): Original volumes `(num_particles,)`.
        velgrads (Array): Velocity gradient tensors `(num_particles, dimension, dimension)`.
        stresses (Array): Cauchy stress tensors `(num_particles, 3, 3)`.
        forces (Array): External force vectors `(num_particles, dimension)`.
        F (Array): Deformation gradient tensors `(num_particles, dimension, dimension)`.
    """

    positions: Array
    velocities: Array
    masses: Array
    species: Array
    volumes: Array
    volumes_original: Array
    velgrads: Array
    stresses: Array
    forces: Array
    F: Array

    original_density: jnp.float32

    @classmethod
    def register(
        cls: Self,
        positions: Array,
        velocities: Array = None,
        masses: Array = None,
        species: Array = None,
        volumes: Array = None,
        velgrads: Array = None,
        stresses: Array = None,
        forces: Array = None,
        F: Array = None,
        original_density: jnp.float32 = 1.0,
    ) -> Self:
        """Initialize the state of the MPM particles.

        Expects arrays of the same length (i.e., first dimension of num_particles) for all the input arrays.

        Args:
            cls (Particles):
                self type reference
            positions (Array): Spatial coordinate vectors `(num_particles, dims)`.
            velocities (Array, optional): Velocity vectors `(num_particles, dims)`, defaults to zeros.
            masses (Array, optional): Masses array `(num_particles, )` defaults to zeros.
            species (Array, optional): Material ID or type of the particle `(num_particles, )`, defaults to zeros.
            volumes (Array, optional): Volumes `(num_particles, )`, defaults to zeros.
            velgrads (Array, optional): Velocity gradient tensors `(num_particles, dims, dims )`, defaults to zeros.
            stresses (Array, optional): Cauchy stress tensors `(num_particles, 3, 3 )`, defaults to zeros.
            forces (Array, optional): External force vectors `(num_particles, dims)`, defaults to zeros.
            F (Array, optional): Deformation gradient tensors `(num_particles, dims, dims)`, defaults identity matrices.
            original_density (jnp.float32, optional): Original density (Scalar). Defaults to 1.0.

        Returns:
            Particles: Updated state for the MPM particles.

        Example:
            >>> # create two particles with constant velocities
            >>> import pymudokon as pm
            >>> import jax.numpy as jnp
            >>> particles = pm.Particles.register(
            >>>     positions = jnp.array([[0.0, 0.0], [1.0, 1.0]]),
            >>>     velocities = jnp.array([[0.0, 0.0], [1.0, 2.0]]
            >>> )
        )
        """
        _num_particles, _dim = positions.shape

        if velocities is None:
            velocities = jnp.zeros((_num_particles, _dim))

        if masses is None:
            masses = jnp.zeros((_num_particles))

        if species is None:
            species = jnp.zeros((_num_particles), dtype=jnp.int32)

        if volumes is None:
            volumes = jnp.zeros((_num_particles))

        volumes_original = volumes

        if velgrads is None:
            velgrads = jnp.zeros((_num_particles, _dim, _dim))

        if stresses is None:
            stresses = jnp.zeros((_num_particles, 3, 3))

        if forces is None:
            forces = jnp.zeros((_num_particles, _dim))

        if F is None:
            F = jnp.stack([jnp.eye(_dim)] * _num_particles)

        original_density = original_density

        return cls(
            positions=positions,
            velocities=velocities,
            masses=masses,
            species=species,
            volumes=volumes,
            volumes_original=volumes_original,
            velgrads=velgrads,
            stresses=stresses,
            forces=forces,
            F=F,
            original_density=original_density,
        )

    @jax.jit
    def calculate_volume(
        self: Self,
        node_spacing: jnp.float32,
        particles_per_cell: jnp.int32,
    ) -> Self:
        """Calculate the particles' volumes according to grid discretization and particles per cell.

        Should be called after initialization, and before updating state with the MPM solver.

        Args:
            cls (Particles): Self reference
            particles (Particles): Particles state.
            node_spacing (jnp.float32): Node spacing of background grid.
            particles_per_cell (jnp.int32): Number of particles in each cell.

        Returns:
            Particles: Updated state for the MPM particles.

        Example:
            >>> import pymudokon as pm
            >>> import jax.numpy as jnp
            >>> # ... create particles
            >>> particles = particles.calculate_volume(0.5, 2)
        """
        num_particles, dim = self.positions.shape

        volumes = jnp.ones(num_particles) * (node_spacing**dim) / particles_per_cell

        volumes_original = volumes

        return self.replace(volumes=volumes, volumes_original=volumes_original)

    @jax.jit
    def refresh(self) -> Self:
        """Refresh the state of the particles.

        Typically called before each time step to reset the state of the particles (e.g. in :func:`~usl.update`).

        Args:
            cls (Particles): Particles state.

        Returns:
            Particles: Updated state for the MPM particles.

        Example:
            >>> import pymudokon as pm
            >>> import jax.numpy as jnp
            >>> # ... create particles
            >>> particles = particles.refresh()
        """
        return self.replace(velgrads=self.velgrads.at[:].set(0.0))
