"""State and functions for the material points (called particles)."""


from typing_extensions import Self

import chex
import jax.numpy as jnp

from jax.sharding import Sharding
import jax

import equinox as eqx

class Particles(eqx.Module):
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

    position_stack: chex.Array  = eqx.field(converter=lambda x: jnp.asarray(x))
    velocity_stack: chex.Array  = eqx.field(converter=lambda x: jnp.asarray(x))
    force_stack: chex.Array = eqx.field(converter=lambda x: jnp.asarray(x))
    mass_stack: chex.Array = eqx.field(converter=lambda x: jnp.asarray(x))
    volume_stack: chex.Array = eqx.field(converter=lambda x: jnp.asarray(x))
    volume0_stack: chex.Array = eqx.field(converter=lambda x: jnp.asarray(x))

    L_stack: chex.Array = eqx.field(converter=lambda x: jnp.asarray(x))
    stress_stack: chex.Array = eqx.field(converter=lambda x: jnp.asarray(x))
    F_stack: chex.Array = eqx.field(converter=lambda x: jnp.asarray(x))
    dim: int = eqx.field(static=True, converter=lambda x: int(x))
    num_points: int = eqx.field(static=True, converter=lambda x: int(x))

    
    def __init__(
        self: Self,
        config=None,
        position_stack: chex.Array= None,
        velocity_stack: chex.Array = None,
        mass_stack: chex.Array = None,
        volume_stack: chex.Array = None,
        L_stack: chex.Array = None,
        stress_stack: chex.Array = None,
        force_stack: chex.Array = None,
        F_stack: chex.Array = None,
    ) -> Self:
        if config:
            self.num_points = config.num_points
            self.dim =  config.dim
        else:
            self.num_points,self.dim = position_stack.shape
            
        if position_stack is None:
            self.position_stack = jnp.zeros((self.num_points,self.dim))
        else:
            self.position_stack = position_stack

        if velocity_stack is None:
            self.velocity_stack = jnp.zeros((self.num_points, self.dim))
        else:
            self.velocity_stack = velocity_stack
            
        if force_stack is None:
            self.force_stack = jnp.zeros((self.num_points, self.dim))
        else:
            self.force_stack = force_stack
            
        if mass_stack is None:
            self.mass_stack = jnp.zeros((self.num_points))
        else:
            self.mass_stack = mass_stack
        
        if volume_stack is None:
            self.volume_stack = jnp.zeros((self.num_points))
        else:
            self.volume_stack = volume_stack
        
        self.volume0_stack = self.volume_stack

        if L_stack is None:
            self.L_stack = jnp.zeros((self.num_points, 3, 3))
        else:
            self.L_stack = self.L_stack
            
        if stress_stack is None:
            self.stress_stack = jnp.zeros((self.num_points, 3, 3))
        else:
            self.stress_stack = stress_stack

        if F_stack is None:
            self.F_stack = jnp.stack([jnp.eye(3)] * self.num_points)
        else:
            self.F_stack = F_stack


    def refresh(self) -> Self:
        """Refresh the state of the particles.

        Typically called before each time step to reset the state of the particles.

        Args:
            cls: Particles state.

        Returns:
            Particles: Updated state for the MPM particles.
        """
        return eqx.tree_at(
            lambda state: (
                state.L_stack
            ),
            self,
            (self.L_stack.at[:].set(0.0)),
        )
        
    def calculate_volume(
        self: Self,
        cell_size: jnp.float32,
        ppc: jnp.int32,
    ):
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

        volume_stack = (
            jnp.ones(self.num_points) * (cell_size**self.dim) / ppc
        )
        return volume_stack

    
    # def distributed(self: Self, device: Sharding):
    #     position_stack = jax.device_put(self.position_stack,device)
    #     velocity_stack = jax.device_put(self.velocity_stack,device)
    #     force_stack = jax.device_put(self.force_stack,device)
    #     mass_stack = jax.device_put(self.mass_stack,device)
    #     volume_stack = jax.device_put(self.volume_stack,device)
    #     volume0_stack = jax.device_put(self.volume0_stack,device)
    #     L_stack = jax.device_put(self.L_stack,device)
    #     stress_stack = jax.device_put(self.stress_stack,device)
    #     F_stack = jax.device_put(self.F_stack,device)
    #     id_stack = jax.device_put(self.id_stack,device)
    #     return self.replace(
    #         position_stack = position_stack,
    #         velocity_stack = velocity_stack,
    #         force_stack = force_stack,
    #         mass_stack = mass_stack,
    #         volume_stack = volume_stack,
    #         volume0_stack = volume0_stack,
    #         L_stack = L_stack,
    #         stress_stack = stress_stack,
    #         F_stack = F_stack,
    #         id_stack = id_stack
    #     )





    # def get_phi_stack(self, rho_p):
        
    #     density_stack = self.mass_stack/self.volume_stack

    #     return density_stack/rho_p
