"""State and functions for the material points (called particles)."""

import jax
from flax import struct
from typing_extensions import Self


@struct.dataclass
class Particles:
    """Dataclass containing the state of the MPM particles.

    Attributes:
        original_density (jax.numpy.float32): Original density.
        positions (jax.Array): Spatial coordinate vectors `(num_particles, dimension)`.
        velocities (jax.Array): Spatial velocity vectors `(num_particles, dimension)`.
        masses (jax.Array): Masses `(num_particles,)`.
        species (jax.Array): Material ID or type `(num_particles,)`.
        volumes (jax.Array): Current volumes  `(num_particles,)`.
        volumes_original (jax.Array): Original volumes `(num_particles,)`.
        velgrads (jax.Array): Velocity gradient tensors `(num_particles, dimension, dimension)`.
        stresses (jax.Array): Cauchy stress tensors `(num_particles, 3, 3)`.
        forces (jax.Array): External force vectors `(num_particles, dimension)`.
        F (jax.Array): Deformation gradient tensors `(num_particles, dimension, dimension)`.
    """

    positions: jax.Array
    velocities: jax.Array
    masses: jax.Array
    species: jax.Array
    volumes: jax.Array
    volumes_original: jax.Array
    velgrads: jax.Array
    stresses: jax.Array
    forces: jax.Array
    F: jax.Array

    ids: jax.Array

    original_density: jax.numpy.float32

    @classmethod
    def create(
        cls: Self,
        positions: jax.Array,
        velocities: jax.Array = None,
        masses: jax.Array = None,
        species: jax.Array = None,
        volumes: jax.Array = None,
        velgrads: jax.Array = None,
        stresses: jax.Array = None,
        forces: jax.Array = None,
        F: jax.Array = None,
        original_density: jax.numpy.float32 = 1.0,
    ) -> Self:
        """Initialize the state of the MPM particles.

        Expects arrays of the same length (i.e., first dimension of num_particles) for all the input arrays.

        Args:
            cls (Particles):
                self type reference
            positions (jax.Array): Spatial coordinate vectors `(num_particles, dims)`.
            velocities (jax.Array, optional): Velocity vectors `(num_particles, dims)`, defaults to zeros.
            masses (jax.Array, optional): Masses array `(num_particles, )` defaults to zeros.
            species (jax.Array, optional): Material ID or type of the particle `(num_particles, )`, defaults to zeros.
            volumes (jax.Array, optional): Volumes `(num_particles, )`, defaults to zeros.
            velgrads (jax.Array, optional): Velocity gradient tensors `(num_particles, dims, dims )`, defaults to zeros.
            stresses (jax.Array, optional): Cauchy stress tensors `(num_particles, 3, 3 )`, defaults to zeros.
            forces (jax.Array, optional): External force vectors `(num_particles, dims)`, defaults to zeros.
            F (jax.Array, optional):
                Deformation gradient tensors `(num_particles, dims, dims)`, defaults identity matrices.
            original_density (jax.numpy.float32, optional): Original density (Scalar). Defaults to 1.0.

        Returns:
            Particles: Updated state for the MPM particles.

        Example:
            >>> # create two particles with constant velocities
            >>> import pymudokon as pm
            >>> import jax.numpy as jax.numpy
            >>> particles = pm.Particles.register(
            >>>     positions = jax.numpy.array([[0.0, 0.0], [1.0, 1.0]]),
            >>>     velocities = jax.numpy.array([[0.0, 0.0], [1.0, 2.0]]
            >>> )
        )
        """
        _num_particles, dim = positions.shape

        if velocities is None:
            velocities = jax.numpy.zeros((_num_particles, dim))

        if masses is None:
            masses = jax.numpy.zeros((_num_particles))

        if species is None:
            species = jax.numpy.zeros((_num_particles), dtype=jax.numpy.int32)

        if volumes is None:
            volumes = jax.numpy.zeros((_num_particles))

        volumes_original = volumes

        if velgrads is None:
            velgrads = jax.numpy.zeros((_num_particles, dim, dim))

        if stresses is None:
            stresses = jax.numpy.zeros((_num_particles, 3, 3))

        if forces is None:
            forces = jax.numpy.zeros((_num_particles, dim))

        if F is None:
            F = jax.numpy.stack([jax.numpy.eye(dim)] * _num_particles)

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
            ids=jax.numpy.arange(_num_particles),
        )

    @jax.jit
    def calculate_volume(
        self: Self,
        node_spacing: jax.numpy.float32,
        particles_per_cell: jax.numpy.int32,
    ) -> Self:
        """Calculate the particles' volumes according to grid discretization and particles per cell.

        Should be called after initialization, and before updating state with the MPM solver.

        Args:
            cls (Particles): Self reference
            particles (Particles): Particles state.
            node_spacing (jax.numpy.float32): Node spacing of background grid.
            particles_per_cell (jax.numpy.int32): Number of particles in each cell.

        Returns:
            Particles: Updated state for the MPM particles.

        Example:
            >>> import pymudokon as pm
            >>> import jax.numpy as jax.numpy
            >>> # ... create particles
            >>> particles = particles.calculate_volume(0.5, 2)
        """
        num_particles, dim = self.positions.shape

        volumes = jax.numpy.ones(num_particles) * (node_spacing**dim) / particles_per_cell

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
            >>> import jax.numpy as jax.numpy
            >>> # ... create particles
            >>> particles = particles.refresh()
        """
        return self.replace(velgrads=self.velgrads.at[:].set(0.0))
