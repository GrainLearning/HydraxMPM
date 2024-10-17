"""State and functions for the material points (called particles)."""
# TODO: Add support for different initial volume calculation ratios.

from typing_extensions import Self

import chex
import jax.numpy as jnp

from jax.sharding import Sharding
import jax

@chex.dataclass
class Particles:
    """Material points state.

    Attributes:
        position_stack: Spatial coordinate vectors `(num_particles, dimension)`.
        velocity_stack: Spatial velocity vectors `(num_particles, dimension)`.
        force_stack: External force vectors `(num_particles, dimension)`.
        mass_stack: Masses `(num_particles,)`.
        volume_stack: Current volumes  `(num_particles,)`.
        volume0_stack: Original volumes `(num_particles,)`.
        L_stack: Velocity gradient tensors `(num_particles, 3, 3)`.
        stress_stack: Cauchy stress tensors `(num_particles, 3, 3)`.
        F_stack: Deformation gradient tensors `(num_particles, 3, 3)`.
        id_stack: Particle IDs `(num_particles,)`.

    Example usage:
            >>> # create two particles with constant velocities in 2D plane strain
            >>> import pymudokon as pm
            >>> import jnp as jnp
            >>> particles = pm.Particles.register(
            >>>     position_stack = jnp.array([[0.0, 0.0], [1.0, 1.0]]),
            >>>     velocity_stack = jnp.array([[0.0, 0.0], [1.0, 2.0]]
            >>> )

        One might need to discretize particles on the grid before using them in the
        MPM solver.

        This can be done either using a utility function in `pm.discretize` or by
        calling directly:
            >>> # ... create particles
            >>> particles = particles.calculate_volume(
                node_spacing = 0.5, particles_per_cell = 2)
            >>> # ... use particles in MPM solver


    """

    position_stack: chex.Array
    velocity_stack: chex.Array
    force_stack: chex.Array
    mass_stack: chex.Array
    volume_stack: chex.Array
    volume0_stack: chex.Array
    L_stack: chex.Array
    stress_stack: chex.Array
    F_stack: chex.Array
    id_stack: chex.Array
    dim: int
    num_particles: int

    @classmethod
    def create(
        cls: Self,
        position_stack: chex.Array,
        velocity_stack: chex.Array = None,
        mass_stack: chex.Array = None,
        volume_stack: chex.Array = None,
        L_stack: chex.Array = None,
        stress_stack: chex.Array = None,
        force_stack: chex.Array = None,
        F_stack: chex.Array = None,
    ) -> Self:
        """Create the initial state of the particles."""
        num_particles, dim = position_stack.shape

        if velocity_stack is None:
            velocity_stack = jnp.zeros((num_particles, dim))

        if force_stack is None:
            force_stack = jnp.zeros((num_particles, dim))

        if mass_stack is None:
            mass_stack = jnp.zeros((num_particles))

        if volume_stack is None:
            volume_stack = jnp.zeros((num_particles))

        volume0_stack = volume_stack

        if L_stack is None:
            L_stack = jnp.zeros((num_particles, 3, 3))

        if stress_stack is None:
            stress_stack = jnp.zeros((num_particles, 3, 3))

        if F_stack is None:
            F_stack = jnp.stack([jnp.eye(3)] * num_particles)

        return cls(
            position_stack=position_stack,
            velocity_stack=velocity_stack,
            mass_stack=mass_stack,
            volume_stack=volume_stack,
            volume0_stack=volume0_stack,
            L_stack=L_stack,
            stress_stack=stress_stack,
            force_stack=force_stack,
            F_stack=F_stack,
            id_stack=jnp.arange(num_particles),
            num_particles=num_particles,
            dim = dim
        )
    
    def distributed(self: Self, device: Sharding):
        position_stack = jax.device_put(self.position_stack,device)
        velocity_stack = jax.device_put(self.velocity_stack,device)
        force_stack = jax.device_put(self.force_stack,device)
        mass_stack = jax.device_put(self.mass_stack,device)
        volume_stack = jax.device_put(self.volume_stack,device)
        volume0_stack = jax.device_put(self.volume0_stack,device)
        L_stack = jax.device_put(self.L_stack,device)
        stress_stack = jax.device_put(self.stress_stack,device)
        F_stack = jax.device_put(self.F_stack,device)
        id_stack = jax.device_put(self.id_stack,device)
        return self.replace(
            position_stack = position_stack,
            velocity_stack = velocity_stack,
            force_stack = force_stack,
            mass_stack = mass_stack,
            volume_stack = volume_stack,
            volume0_stack = volume0_stack,
            L_stack = L_stack,
            stress_stack = stress_stack,
            F_stack = F_stack,
            id_stack = id_stack
        )


    def calculate_volume(
        self: Self,
        node_spacing: jnp.float32,
        particles_per_cell: jnp.int32,
    ) -> Self:
        """Calculate the particles' initial volumes.

        Should be called after initialization, and before updating state
        with the MPM solver.

        Assumes each cell should have the same number of particles and volume ratio.

        Args:
            cls: Self reference
            particles: Particles state.
            node_spacing: Node spacing of background grid.
            particles_per_cell: Number of particles in each cell.

        Returns:
            Particles: Updated particles state.
        """
        num_particles, dim = self.position_stack.shape

        volume_stack = (
            jnp.ones(num_particles) * (node_spacing**dim) / particles_per_cell
        )

        volume0_stack = volume_stack

        return self.replace(volume_stack=volume_stack, volume0_stack=volume0_stack)

    def refresh(self) -> Self:
        """Refresh the state of the particles.

        Typically called before each time step to reset the state of the particles.

        Args:
            cls: Particles state.

        Returns:
            Particles: Updated state for the MPM particles.
        """
        return self.replace(L_stack=self.L_stack.at[:].set(0.0))

    def get_phi_stack(self, rho_p):
        
        density_stack = self.mass_stack/self.volume_stack

        return density_stack/rho_p
