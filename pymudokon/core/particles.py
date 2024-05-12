"""State and functions for managing the material points (called particles)."""

import dataclasses

import jax
import jax.numpy as jnp
from jax import Array
from typing_extensions import Self

from .base import Base


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=False)
class Particles(Base):
    """State for the MPM particles.

    Attributes:
        original_density (jnp.float32):
            Original density of the particles.
        positions (Array):
            Position vectors of the particles `(num_particles, dimension)`.
        velocities (Array):
            Velocity vectors of the particles `(num_particles, dimension)`.
        masses (Array):
            Masses of the particles `(num_particles,)`.
        species (Array):
            Species (material ID) of the particles `(num_particles,)`.
        volumes (Array):
            Current volumes of the particles `(num_particles,)`.
        volumes_original (Array):
            Original volumes of the particles `(num_particles,)`.
        velgrads (Array):
            Velocity gradient tensors of the particles `(num_particles, dimension, dimension)`.
        stresses (Array):
            Stress tensors of the particles  `(num_particles, 3, 3)`.
        forces (Array):
            External force vectors on the particles `(num_particles, dimension)`.
        F (Array):
            Deformation gradient tensors of the particles `(num_particles, dimension, dimension)`.
    """

    # arrays
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
            positions (Array):
                Position vectors of the particles `(num_particles, dims)`.
            velocities (Array, optional):
                Velocity vectors of the particles `(num_particles, dims)`, defaults to zeros.
            masses (Array, optional):
                Masses array of the particles`(num_particles, )` defaults to zeros.
            species (Array, optional):
                Species / material id of the particle `(num_particles, )`, defaults to zeros.
            volumes (Array, optional):
                Volumes of the particles `(num_particles, )`, defaults to zeros.
            velgrads (Array, optional):
                Velocity gradient tensors of the particles `(num_particles, dims, dims )`, defaults to zeros.
            stresses (Array, optional):
                Stress tensor (Cauchy) of the particles `(num_particles, 3, 3 )`, defaults to zeros.
            forces (Array, optional):
                Force vectors of the particles `(num_particles, dims)`, defaults to zeros.
            F (Array, optional):
                Deformation gradient tensors of particles, `(num_particles, dims, dims)`, defaults to identity matrices.
            original_density (jnp.float32, optional):
                original density. Defaults to 1.0.

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
            cls (Particles):
                Self reference
            particles (Particles):
                Particles state.
            node_spacing (jnp.float32):
                Node spacing of background grid.
            particles_per_cell (jnp.int32):
                number of particles in each cell.

        Returns:
            Particles:
                Updated state for the MPM particles.

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
            cls (Particles):
                Particles state.

        Returns:
            Particles:
                Updated state for the MPM particles.

        Example:
            >>> import pymudokon as pm
            >>> import jax.numpy as jnp
            >>> # ... create particles
            >>> particles = particles.refresh()
        """
        return self.replace(velgrads=self.velgrads.at[:].set(0.0))
