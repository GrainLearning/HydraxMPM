from typing import Tuple
from typing_extensions import Self, Optional

import chex
import jax
import jax.numpy as jnp

from ..particles.particles import Particles
from ..utils.math_helpers import get_sym_tensor
from .common import get_bulk_modulus, get_lame_modulus, get_shear_modulus
from .material import Material
from functools import partial

import equinox as eqx

from ..config.mpm_config import MPMConfig


class LinearIsotropicElastic(Material):
    """Isotropic linear elastic material solved in incremental form.

    Attributes:
        E: Young's modulus.
        nu: Poisson's ratio.
        G: Shear modulus.
        K: Bulk modulus.
        lam: Lame modulus.
    """

    E: jnp.float32
    nu: jnp.float32
    G: jnp.float32
    K: jnp.float32
    lam: jnp.float32

    def __init__(self: Self, config: Optional[MPMConfig], E: float, nu: float) -> Self:
        """Initialize the isotropic linear elastic material."""

        self.E = E
        self.nu = nu
        self.K = get_bulk_modulus(E, nu)
        self.G = get_shear_modulus(E, nu)
        self.lam = get_lame_modulus(E, nu)
        super().__init__(config)

    def update_from_particles(
        self: Self, particles: Particles
    ) -> Tuple[Particles, Self]:
        """Update the material state and particle stresses for MPM solver."""

        vmap_update_ip = jax.vmap(fun=self.update_ip, in_axes=(0, 0, 0, None))
        new_stress_stack = vmap_update_ip(
            particles.stress_stack,
            particles.F_stack,
            particles.L_stack,
            None,
        )

        new_particles = eqx.tree_at(
            lambda state: (state.stress_stack),
            particles,
            (new_stress_stack),
        )

        return new_particles, self

    def update_ip(
        self: Self,
        stress_prev: chex.Array,
        F: chex.Array,
        L: chex.Array,
        phi: jnp.float32 = None,
    ) -> Tuple[chex.Array, Self]:
        """Update stress on a single integration point"""
        deps = get_sym_tensor(L) * self.config.dt

        if self.config.dim == 2:
            deps = deps.at[:, [2, 2]].set(0.0)

        return (
            stress_prev + self.lam * jnp.trace(deps) * jnp.eye(3) + 2.0 * self.G * deps
        )
