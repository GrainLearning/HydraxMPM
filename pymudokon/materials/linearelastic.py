"""Implementation, state and functions for isotropic linear elastic material."""

import dataclasses
from typing import Tuple
from flax import struct
import jax
import jax.numpy as jnp
from jax import Array
from typing_extensions import Self

from ..core.particles import Particles
from .material import Material


from functools import partial


@struct.dataclass
class LinearIsotropicElastic(Material):
    """State for the isotropic linear elastic material.

    Attributes:
        E (jnp.float32): Young's modulus.
        nu (Array): Poisson's ratio.
        G (Array): Shear modulus.
        K (Array): Bulk modulus.
        eps_e (Array): Elastic strain tensors `(number of particles,dim,dim)`.
    """

    # TODO we can replace this with incremental form / or hyperelastic form
    E: jnp.float32
    nu: jnp.float32
    G: jnp.float32
    K: jnp.float32

    # arrays
    stress_ref: Array
    eps_e: Array

    @classmethod
    def create(cls: Self, E: jnp.float32, nu: jnp.float32, num_particles, dim: jnp.int16 = 3, stress_ref:Array=None) -> Self:
        """Initialize the isotropic linear elastic material.

        Args:
            E (jnp.float32): Young's modulus.
            nu (jnp.float32): Poisson's ratio.
            num_particles (jnp.int16): Number of particles.
            dim (jnp.int16, optional): Dimension. Defaults to 3.

        Returns:
            LinearIsotropicElastic: Initial state of the isotropic linear elastic material.

        Example:
            >>> # Create a linear elastic material for plane strain and 2 particles
            >>> import pymudokon as pm
            >>> material = pm.LinearIsotropicElastic.create(1000, 0.2, 2, 2)
        """
        # TODO: How can we avoid the need for num_particles and dim?

        G = E / (2.0 * (1.0 + nu))
        K = E / (3.0 * (1.0 - 2.0 * nu))
        eps_e = jnp.zeros((num_particles, dim, dim), dtype=jnp.float32)
        
        if stress_ref is None:
            stress_ref = jnp.zeros((num_particles, 3, 3), dtype=jnp.float32)
        return cls(E=E, nu=nu, G=G, K=K, eps_e=eps_e,stress_ref=stress_ref)

    @jax.jit
    def update_stress(
        self: Self,
        particles: Particles,
        dt: jnp.float32,  # unused
    ) -> Tuple[Particles, Self]:
        """Update stress and strain for all particles.

        Called by the MPM solver (e.g., see :func:`~usl.update`).

        Args:
            self (LinearIsotropicElastic): Self reference.
            particles (ParticlesContainer): State of the particles prior to the update.

            dt (jnp.float32): Time step.

        Returns:
            Tuple[ParticlesContainer, LinearIsotropicElastic]: Updated particles and material state.

        Example:
            >>> import pymudokon as pm
            >>> # ...  Assume particles and material are initialized
            >>> particles, material = material.update_stress(particles, 0.001)
        """
        vel_grad = particles.velgrads

        vel_grad_T = jnp.transpose(vel_grad, axes=(0, 2, 1))

        deps = 0.5 * (vel_grad + vel_grad_T) * dt

        stress, eps_e = self.vmap_update(
            self.eps_e, deps, self.stress_ref, self.G, self.K
        )

        material = self.replace(eps_e=eps_e)

        particles = particles.replace(stresses=stress)

        return particles, material

    @jax.jit
    def update_stress_benchmark(
        self: Self,
        strain_rate: Array,
        volumes: Array,
        dt: jnp.float32,
        update_history: bool = True,
    ) -> Self:
        """Update stress for a single element benchmark.

        Args:
            self (LinearIsotropicElastic): Self reference.
            strain_rate (Array): Strain rate tensor `(dim,dim)`.
            volumes: Volumes of the particles.
            dt (jnp.float32): Time step.
            update_history (bool, optional): Update the history. Defaults to True.

        Returns:
            LinearIsotropicElastic: Updated material state.

        Example:
            >>> import pymudokon as pm
            >>> import jax.numpy as jnp
            >>> # ...  Assume material is initialized
            >>> material = material.update_stress_benchmark(jnp.eye(2), 0.5, 0.001)
        """
        deps = strain_rate * dt
        stress, eps_e = self.vmap_update(
            self.eps_e, deps, self.stress_ref, self.G, self.K
        )
        return stress, self.replace(eps_e=eps_e)

    @partial(jax.vmap, in_axes=(None, 0, 0, 0, None, None), out_axes=(0, 0))
    def vmap_update(self:Self, eps_e_prev: Array, deps: Array, stress_ref: Array, G: jnp.float32, K: jnp.float32) -> Tuple[Array, Array]:
        """Parallelized stress update for each particle.

        Tensors are mapped per particle via vmap (num_particles, dim) -> (dim,).

        Args:
            eps_e_prev (Array): Vectorized previous elastic strain tensor `(number of particles,dimension,dimension)`.
            deps (Array): Vectorized strain increment.
                Shape of full tensor is `(number of particles,dimension,dimension)`.
            G (jnp.float32): Shear modulus.
            K (jnp.float32): Bulk modulus.
            dt (jnp.float32): Time step.

        Returns:
            Tuple[Array, Array]: Updated stress and elastic strain tensors.

        Example:
            >>> import pymudokon as pm
            >>> import jax.numpy as jnp
            >>> eps_e = jnp.ones((2, 2, 2), dtype=jnp.float32)
            >>> deps = jnp.eye(3, dtype=jnp.float32)*0.001
            >>> stress, eps_e = jax.vmap(vmap_update, in_axes=(0, 0, None, None), out_axes=(0, 0))(
            ... eps_e, deps, 100, 100
            ...)
        """
        dim = deps.shape[0]

        eps_e = eps_e_prev + deps

        eps_e_v = -jnp.trace(eps_e)

        eps_e_d = eps_e + (eps_e_v / dim) * jnp.eye(dim)

        s = 2.0 * G * eps_e_d

        # pad for 3D stress tensor
        if dim == 2:
            s = jnp.pad(s, ((0, 1), (0, 1)), mode="constant")


        p = K * eps_e_v * jnp.eye(3)

        p_ref = -jnp.trace(stress_ref) / dim
        s_ref = stress_ref + p_ref * jnp.eye(3)
        
        p = p + p_ref

        s = s + s_ref
        return s - p, eps_e