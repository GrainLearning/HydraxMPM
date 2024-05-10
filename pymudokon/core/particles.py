"""State and functions for managing the material points (called particles).

The module contains the following main components:

- ParticlesContainer:
    JAX pytree (NamedTuple) that stores the state of the MPM particles.
- init:
    Initialize the state for the MPM particles.
- calculate_volume:
    Calculate the volumes of the particles based on background grid information.
- refresh:
    Refresh/reset the state for the MPM particles.
"""

from typing import NamedTuple, Union

import jax
import jax.numpy as jnp


class ParticlesContainer(NamedTuple):
    """State for the MPM particles.

    Attributes:
        original_density (jnp.float32):
            Original density of the particles.
        positions_array (Union[jnp.array, jnp.float32]):
            Position vectors of the particles.
            Shape is `(number of particles, dimension)`.
        velocities_array (Union[jnp.array, jnp.float32]):
            Velocity vectores of the particles.
            Shape is `(number of particles, dimension)`.
        masses_array (Union[jnp.array, jnp.float32]):
            Masses of the particles.
            Shape is `(number of particles,)`.
        species_array (Union[jnp.array, jnp.int32]):
            Species (material ID) of the particles.
            Shape is `(number of particles,)`.
        volumes_array (Union[jnp.array, jnp.float32]):
            Current volumes of the particles.
            Shape is `(number of particles,)`.
        volumes_original_array (Union[jnp.array, jnp.float32]):
            Original volumes of the particles.
            Typically not changed during simulation.
            Shape is `(number of particles,)`.
        velgrad_array (Union[jnp.array, jnp.float32]):
            Velocity gradient tensors of the particles.
            Shape is `(number of particles, dimension, dimension)`.
        stresses_array (Union[jnp.array, jnp.float32]):
            Stress tensors of the particles.
            Kept as a 3x3 matrix for plane strain.
            Shape is `(number of particles, 3, 3)`.
        forces_array (Union[jnp.array, jnp.float32]):
            External force vectors on the particles.
        F_array (Union[jnp.array, jnp.float32]):
            Deformation gradient tensors of the particles.
    """
    original_density: jnp.float32

    # arrays
    positions_array: Union[jnp.array, jnp.float32]
    velocities_array: Union[jnp.array, jnp.float32]
    masses_array: Union[jnp.array, jnp.float32]
    species_array: Union[jnp.array, jnp.int32]
    volumes_array: Union[jnp.array, jnp.float32]
    volumes_original_array: Union[jnp.array, jnp.float32]
    velgrad_array: Union[jnp.array, jnp.float32]
    stresses_array: Union[jnp.array, jnp.float32]
    forces_array: Union[jnp.array, jnp.float32]
    F_array: Union[jnp.array, jnp.float32]


def init(
    positions: Union[jnp.array, jnp.float32],
    velocities: Union[jnp.array, jnp.float32] = None,
    masses: Union[jnp.array, jnp.float32] = None,
    species: Union[jnp.array, jnp.int32] = None,
    volumes: Union[jnp.array, jnp.float32] = None,
    velgrad: Union[jnp.array, jnp.float32] = None,
    stresses: Union[jnp.array, jnp.float32] = None,
    forces: Union[jnp.array, jnp.float32] = None,
    F: Union[jnp.array, jnp.float32] = None,
    density: jnp.float32 = 1.0,
) -> ParticlesContainer:
    """Initialize the state for the MPM particles.

    Expects arrays of the same length (i.e., first dimension of number of particles) for all the input arrays.

    Args:
        positions (Union[jnp.array, jnp.float32]):
            Position vectors of the particles.
            Expects shape of `(number of particles, dimensions)`.
        velocities (Union[jnp.array, jnp.float32], optional):
            Velocity vectors of the particles.
            Expects shape of `(number of particles, dimensions)`.
            Defaults to zeros.
        masses (Union[jnp.array, jnp.float32], optional):
            Masses array of the particles.
            Expects shape of `(number of particles, )`.
            Defaults to zeros.
        species (Union[jnp.array, jnp.int32], optional):
            Species / material id of the particles.
            Expects shape of `(number of particles, )`.
            Defaults to zeros.
        volumes (Union[jnp.array, jnp.float32], optional):
            Volumes of the particles.
            Expects shape of `(number of particles, )`.
            Defaults to zeros.
        velgrad (Union[jnp.array, jnp.float32], optional):
            Velocity gradient tensors of the particles.
            Expects shape of `(number of particles, dimensions, dimensions )`.
            Defaults to zeros.
        stresses (Union[jnp.array, jnp.float32], optional):
            Stress tensor (Cauchy) of the particles.
            Expects shape of `(number of particles, 3, 3 )`.
            Defaults to zeros.
        forces (Union[jnp.array, jnp.float32], optional):
            Force vectors of the particles.
            Expects shape of `(number of particles, dimensions)`.
            Defaults to zeros.
        F (Union[jnp.array, jnp.float32], optional):
            Deformation gradient tensors of particles.
            Expects shape of `(number of particles, dimensions, dimensions)`.
            Defaults to identity matrices.
        density (jnp.float32, optional):
            original density. Defaults to 1.0.

    Returns:
        ParticlesContainer: Updated state for the MPM particles.

    Example:
        >>> # create two particles with constant velocities
        >>> import pymudokon as pm
        >>> import jax.numpy as jnp
        >>> positions = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 1.0]])
        >>> velocities = jnp.array([[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]])
        >>> particles_state = pm.particles.init(positions, velocities)
    """
    _num_particles, _dim = positions.shape

    if velocities is None:
        velocities_array = jnp.zeros((_num_particles, _dim))
    else:
        velocities_array = velocities

    if masses is None:
        masses_array = jnp.zeros((_num_particles))
    else:
        masses_array = masses

    if species is None:
        species_array = jnp.zeros((_num_particles), dtype=jnp.int32)
    else:
        species_array = species

    if volumes is None:
        volumes_array = jnp.zeros((_num_particles))
    else:
        volumes_array = volumes

    volumes_original_array = volumes

    if velgrad is None:
        velgrad_array = jnp.zeros((_num_particles, _dim, _dim))
    else:
        velgrad_array = velgrad

    if stresses is None:
        stresses_array = jnp.zeros((_num_particles, 3, 3))
    else:
        stresses_array = stresses

    if forces is None:
        forces_array = jnp.zeros((_num_particles, _dim))
    else:
        forces_array = forces

    if F is None:
        F_array = jnp.stack([jnp.eye(_dim)] * _num_particles)

    else:
        F_array = F

    return ParticlesContainer(
        positions_array=positions,
        velocities_array=velocities_array,
        masses_array=masses_array,
        species_array=species_array,
        volumes_array=volumes_array,
        volumes_original_array=volumes_original_array,
        velgrad_array=velgrad_array,
        stresses_array=stresses_array,
        forces_array=forces_array,
        F_array=F_array,
        original_density=density
    )


@jax.jit
def calculate_volume(
    particles_state: ParticlesContainer,
    node_spacing: jnp.float32,
    particles_per_cell: jnp.int32,
) -> ParticlesContainer:
    """Calculate the volumes' of the particles based on the node spacing and particles per cell.

    Updates the volumes array and original volume array of the particles.

    Should be called after initialization, and before updating state with the MPM solver.

    Args:
        particles_state (ParticlesContainer):
            Particles state.
        node_spacing (jnp.float32):
            Node spacing of background grid.
        particles_per_cell (jnp.int32):
            Number of particles in each cell.

    Returns:
        ParticlesContainer: Updated state for the MPM particles.

    Example:
        >>> import pymudokon as pm
        >>> import jax.numpy as jnp
        >>> # ... create particles_state
        >>> particles_state = pm.particles.calculate_volume(particles_state, 0.5, 2)
    """
    num_particles, dim = particles_state.positions_array.shape

    volumes_array = jnp.ones(num_particles) * (node_spacing**dim) / particles_per_cell

    volumes_original_array = volumes_array

    return particles_state._replace(
        volumes_array=volumes_array, volumes_original_array=volumes_original_array
    )


@jax.jit
def refresh(particles_state: ParticlesContainer) -> ParticlesContainer:
    """Refresh the state of the particles.

    Typically called before each time step to reset the state of the particles (e.g. in :func:`~usl.update`).

    Args:
        particles_state (ParticlesContainer): Particles state.

    Returns:
        ParticlesContainer: Updated state for the MPM particles.

    Example:
        >>> import pymudokon as pm
        >>> import jax.numpy as jnp
        >>> # ... create particles_state
        >>> particles_state = pm.particles.refresh(particles_state)
    """
    return particles_state._replace(velgrad_array=particles_state.velgrad_array.at[:].set(0.0))
