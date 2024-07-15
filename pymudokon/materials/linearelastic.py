"""Isotropic linear elastic material and constitutive update."""
# update stress benchmark

from functools import partial
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from typing_extensions import Self

from ..particles.particles import Particles
from .material import Material


@chex.dataclass
class LinearIsotropicElastic(Material):
    """Small strain isotropic linear elastic material solved in incremental form.

    Attributes:
        E: Young's modulus.
        nu: Poisson's ratio.
        G: Shear modulus.
        K: Bulk modulus.

    Can be used as a standalone module or as part of the MPM solver.

    Example usage as standalone module:
        >>> # Create a linear elastic material  and 1 particles
        >>> import pymudokon as pm
        >>> material = pm.LinearIsotropicElastic.create(
        ... 1000, 0.2, 1, stress_ref=jnp.zeros((1, 3, 3), dtype=jnp.float32))
        >>> # Isotropic compression
        >>> material.update_stress_benchmark(jnp.eye(2)*0.99, 0.5, 0.001)
    """

    E: jnp.float32
    nu: jnp.float32
    G: jnp.float32
    K: jnp.float32
    lam: jnp.float32

    @classmethod
    def create(cls: Self, E: jnp.float32, nu: jnp.float32, num_particles, stress_ref: chex.Array = None) -> Self:
        """Initialize the isotropic linear elastic material.

        Args:
            E: Young's modulus.
            nu: Poisson's ratio.
            num_particles: Number of particles.
            stress_ref: Reference stress tensor. Defaults to empty.

        Returns:
            LinearIsotropicElastic: Initial state of the isotropic linear elastic material.
        """
        G = E / (2.0 * (1.0 + nu))
        K = E / (3.0 * (1.0 - 2.0 * nu))

        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        if stress_ref is None:
            stress_ref = jnp.zeros((num_particles, 3, 3), dtype=jnp.float32)

        return cls(E=E, nu=nu, G=G, K=K, lam=lam, stress_ref=stress_ref)

    def update_stress(
        self: Self,
        particles: Particles,
        dt: jnp.float32,  # unused
    ) -> Tuple[Particles, Self]:
        """Update stress and strain for all particles.

        Called by the MPM solver.

        Args:
            self: Self reference.
            particles: State of the particles prior to the update.
            dt: Time step.

        Returns:
            Tuple: Updated particles and material state.

        """
        vel_grad = particles.velgrads

        vel_grad_T = jnp.transpose(vel_grad, axes=(0, 2, 1))

        deps = 0.5 * (vel_grad + vel_grad_T) * dt

        stresses = self.vmap_update(particles.stresses, deps)

        particles = particles.replace(stresses=stresses)

        return particles, self

    def update_stress_benchmark(
        self: Self, stress_prev: chex.Array, strain_rate: chex.Array, volumes: chex.Array, dt: jnp.float32
    ) -> Self:
        """Update stress for a single element benchmark.

        Args:
            self: Self reference.
            strain_rate: Strain rate tensor `(dim,dim)`.
            volumes: Volumes of the particles.
            dt: Time step.

        Returns:
            Self: Updated material state.
        """
        raise NotImplementedError("feature not implemented yet.")
        deps = strain_rate * dt
        stress = self.vmap_update(self.eps_e, deps, self.stress_ref, self.G, self.K)
        return stress, self

    @partial(jax.vmap, in_axes=(None, 0, 0), out_axes=(0))
    def vmap_update(
        self: Self,
        stresses_prev: chex.ArrayBatched,
        deps: chex.ArrayBatched,
    ) -> chex.ArrayBatched:
        """Vectorized stress update.

        Args:
            self: Self reference.
            stresses_prev: Previous stress tensor.
            deps: Vectorized strain increment.

        Returns: Updated stress tensor
        """
        dim = deps.shape[0]

        if dim == 2:
            deps = jnp.pad(deps, ((0, 1), (0, 1)), mode="constant")

        return stresses_prev + self.lam * jnp.trace(deps) * jnp.eye(3) + 2.0 * self.G * deps
