"""State and functions for the material points (called particles)."""
# TODO: Add support for different initial volume calculation ratios.

import chex
import jax
import jax.numpy as jnp
from typing_extensions import Self


@chex.dataclass
class Particles:
    """Material points State

    Example usage:
            >>> # create two particles with constant velocities in 2D plane strain problem
            >>> import pymudokon as pm
            >>> import jnp as jnp
            >>> particles = pm.Particles.register(
            >>>     positions = jnp.array([[0.0, 0.0], [1.0, 1.0]]),
            >>>     velocities = jnp.array([[0.0, 0.0], [1.0, 2.0]]
            >>> )

        One might need to discretize particles on the grid before using them in the MPM solver.

        This can be done either using a utility function in `pm.discretize` or by calling directly:
            >>> # ... create particles
            >>> particles = particles.calculate_volume(node_spacing = 0.5, particles_per_cell = 2)
            >>> # ... use particles in MPM solver

    Attributes:
        original_density: Original density.
        positions: Spatial coordinate vectors `(num_particles, dimension)`.
        velocities: Spatial velocity vectors `(num_particles, dimension)`.
        forces: External force vectors `(num_particles, dimension)`.
        masses: Masses `(num_particles,)`.
        species: Material ID or type `(num_particles,)`.
        volumes: Current volumes  `(num_particles,)`.
        volumes_original: Original volumes `(num_particles,)`.
        velgrads: Velocity gradient tensors `(num_particles, 3, 3)`.
        stresses: Cauchy stress tensors `(num_particles, 3, 3)`.
        F: Deformation gradient tensors `(num_particles, 3, 3)`.
    """

    positions: chex.Array
    velocities: chex.Array
    forces: chex.Array
    masses: chex.Array
    volumes: chex.Array
    volumes_original: chex.Array
    velgrads: chex.Array
    stresses: chex.Array
    F: chex.Array
    ids: chex.Array
    density_ref: jnp.float32

    @classmethod
    def create(
        cls: Self,
        positions: chex.Array,
        velocities: chex.Array = None,
        masses: chex.Array = None,
        volumes: chex.Array = None,
        velgrads: chex.Array = None,
        stresses: chex.Array = None,
        forces: chex.Array = None,
        F: chex.Array = None,
        original_density: jnp.float32 = 1.0,
    ) -> Self:
        """Initialize particles state.

        Expects arrays of the same length (i.e., first dimension of num_particles) for all the input arrays.

        Args:
            cls: self type reference
            positions: Spatial coordinate vectors `(num_particles, dims)`.
            velocities (optional): Velocity vectors `(num_particles, dims)`, defaults to zeros.
            forces (optional): External force vectors `(num_particles, dims)`, defaults to zeros.
            masses (optional): Masses array `(num_particles, )` defaults to zeros.
            species (optional): Material ID or type of the particle `(num_particles, )`, defaults to zeros.
            volumes (optional): Volumes `(num_particles, )`, defaults to zeros.
            velgrads (optional): Velocity gradient tensors `(num_particles, dims, dims )`, defaults to zeros.
            stresses (optional): Cauchy stress tensors `(num_particles, 3, 3 )`, defaults to zeros.
            F ( optional): Deformation gradient tensors `(num_particles, 3, 3)`, defaults identity matrices.
            original_density (optional): Original density (Scalar). Defaults to 1.0.

        Returns:
            Particles: Updated state for the MPM particles.
        )
        """
        num_particles, dim = positions.shape

        if velocities is None:
            velocities = jnp.zeros((num_particles, dim))

        if forces is None:
            forces = jnp.zeros((num_particles, dim))

        if masses is None:
            masses = jnp.zeros((num_particles))

        if volumes is None:
            volumes = jnp.zeros((num_particles))

        volumes_original = volumes

        if velgrads is None:
            velgrads = jnp.zeros((num_particles, 3, 3))

        if stresses is None:
            stresses = jnp.zeros((num_particles, 3, 3))

        if F is None:
            F = jnp.stack([jnp.eye(3)] * num_particles)

        density_ref = original_density

        stress_ref = stresses

        return cls(
            positions=positions,
            velocities=velocities,
            masses=masses,
            volumes=volumes,
            volumes_original=volumes_original,
            velgrads=velgrads,
            stresses=stresses,
            forces=forces,
            F=F,
            density_ref=density_ref,
            ids=jnp.arange(num_particles),
        )

    def calculate_volume(
        self: Self,
        node_spacing: jnp.float32,
        particles_per_cell: jnp.int32,
    ) -> Self:
        """Calculate the particles' volumes given a node spacing and number of particles per cell.

        Should be called after initialization, and before updating state with the MPM solver.

        Assumes each cell should have the same number of particles and volume ratio.

        Args:
            cls: Self reference
            particles: Particles state.
            node_spacing: Node spacing of background grid.
            particles_per_cell: Number of particles in each cell.

        Returns:
            Particles: Updated particles state.
        """
        num_particles, dim = self.positions.shape

        volumes = jnp.ones(num_particles) * (node_spacing**dim) / particles_per_cell

        volumes_original = volumes

        return self.replace(volumes=volumes, volumes_original=volumes_original)

    def refresh(self) -> Self:
        """Refresh the state of the particles.

        Typically called before each time step to reset the state of the particles.

        Args:
            cls: Particles state.

        Returns:
            Particles: Updated state for the MPM particles.
        """
        return self.replace(velgrads=self.velgrads.at[:].set(0.0))
