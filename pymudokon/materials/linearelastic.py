"""Isotropic linear elastic material and constitutive update."""

from typing import Tuple
from typing_extensions import Self

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


class LinearIsotropicElastic(eqx.Module):
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

    # static fields
    dim: int = eqx.field(static=True, converter=lambda x: int(x))
    dt: float = eqx.field(static=True, converter=lambda x: float(x))

    def __init__(
        self: Self,
        config: MPMConfig,
        E: float,
        nu: float,
        dim: int = 3,
        dt: float = 0.1,
    ) -> Self:
        """Initialize the isotropic linear elastic material."""

        if config:
            dim = config.dim
            dt = config.dt

        self.dim = dim
        self.dt = dt

        self.E = E
        self.nu = nu
        self.K = get_bulk_modulus(E, nu)
        self.G = get_shear_modulus(E, nu)
        self.lam = get_lame_modulus(E, nu)

    def __call__(
        self: Self, particles: Particles
    ) -> Tuple[Particles, Self]:
        """Update the material state and particle stresses for MPM solver."""

        vmap_update_ip = jax.vmap(
            fun=self.update_ip,
            in_axes=( 0, 0, 0, None)
        )
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
        """Main stress update function."""

        deps = get_sym_tensor(L) * self.dt

        if self.dim == 2:
            deps = deps.at[:, [2, 2]].set(0.0)

        return (
            stress_prev + self.lam * jnp.trace(deps) * jnp.eye(3) + 2.0 * self.G * deps
        )
