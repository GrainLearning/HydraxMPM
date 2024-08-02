"""Isotropic linear elastic material and constitutive update."""

from typing import Tuple
from typing_extensions import Self

import chex
import jax
import jax.numpy as jnp

from ..particles.particles import Particles
from ..utils.math_helpers import get_sym_tensor_stack
from .common import get_bulk_modulus, get_lame_modulus, get_shear_modulus
from .material import Material


@chex.dataclass
class LinearIsotropicElastic(Material):
    """Small strain isotropic linear elastic material solved in incremental form.

    Attributes:
        E: Young's modulus.
        nu: Poisson's ratio.
        G: Shear modulus.
        K: Bulk modulus.
        lam: Lame modulus.

    Can be used as a standalone module or as part of the MPM solver.
    """

    E: jnp.float32
    nu: jnp.float32
    G: jnp.float32
    K: jnp.float32
    lam: jnp.float32

    @classmethod
    def create(
        cls: Self,
        E: jnp.float32,
        nu: jnp.float32,
    ) -> Self:
        """Initialize the isotropic linear elastic material."""
        K = get_bulk_modulus(E, nu)
        G = get_shear_modulus(E, nu)
        lam = get_lame_modulus(E, nu)

        return cls(E=E, nu=nu, G=G, K=K, lam=lam, absolute_density=None)

    def update_from_particles(
        self: Self, particles: Particles, dt: jnp.float32
    ) -> Tuple[Particles, Self]:
        """Update the material state and particle stresses for MPM solver."""
        stress_stack, self = self.update(
            particles.stress_stack, particles.F_stack, particles.L_stack, None, dt
        )

        return particles.replace(stress_stack=stress_stack), self

    def update(
        self: Self,
        stress_prev_stack: chex.Array,
        F_stack: chex.Array,
        L_stack: chex.Array,
        phi_stack: chex.Array,
        dt: jnp.float32,
    ) -> Tuple[chex.Array, Self]:
        """Main stress update function."""
        dim = stress_prev_stack.shape[1]
        deps_stack = get_sym_tensor_stack(L_stack) * dt

        def vmap_update(stress_prev, deps):
            if dim == 2:
                deps = deps.at[:, [2, 2]].set(0.0)
            return (
                stress_prev
                + self.lam * jnp.trace(deps) * jnp.eye(3)
                + 2.0 * self.G * deps
            )

        stress_next_stack = jax.vmap(vmap_update)(stress_prev_stack, deps_stack)

        return stress_next_stack, self
