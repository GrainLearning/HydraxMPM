"""Constitutive model for a nearly incompressible Newtonian fluid."""

from functools import partial
from typing import Optional, Self, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from ..common.types import TypeFloat, TypeFloatMatrix3x3, TypeInt
from ..material_points.material_points import MaterialPoints
from .constitutive_law import ConstitutiveLaw
from ..utils.math_helpers import get_dev_strain


def give_p(K, rho, rho_0, alpha):
    return K * ((rho / rho_0) ** alpha - 1.0)


def give_rho(K, rho_0, p, alpha):
    return rho_0 * ((p / K) + 1) ** (1.0 / alpha)


class NewtonFluid(ConstitutiveLaw):
    """Nearly incompressible Newtonian fluid.

    Attributes:
        K: Bulk modulus.
        viscosity: Viscosity.
        gamma: Exponent.
    """

    K: TypeFloat
    viscosity: TypeFloat
    alpha: TypeFloat

    def __init__(
        self: Self,
        K: TypeFloat = 2.0 * 10**6,
        viscosity: TypeFloat = 0.001,
        alpha: TypeFloat = 7.0,
        **kwargs,
    ) -> Self:
        """Initialize the nearly incompressible Newtonian fluid material."""

        self.K = K
        self.viscosity = viscosity
        self.alpha = alpha
        super().__init__(**kwargs)

    def init_state(self: Self, material_points: MaterialPoints):
        p_0_stack = material_points.p_stack
        vmap_give_rho_ref = partial(
            jax.vmap,
            in_axes=(None, None, 0, None),
        )(give_rho)

        rho = vmap_give_rho_ref(self.K, self.rho_0, p_0_stack, self.alpha)

        deps_dt_stack = material_points.deps_dt_stack
        rho_rho_0_stack = rho / self.rho_0

        vmap_update_ip = jax.vmap(fun=self.update_ip, in_axes=(0, 0))
        new_stress_stack = vmap_update_ip(deps_dt_stack, rho_rho_0_stack)

        material_points = material_points.replace(stress_stack=new_stress_stack)

        return self.post_init_state(material_points, rho=rho, rho_0=self.rho_0)

    def update(
        self: Self,
        material_points: MaterialPoints,
        dt: TypeFloat,
        dim: Optional[TypeInt] = 3,
    ) -> Tuple[MaterialPoints, Self]:
        """Update the material state and particle stresses for MPM solver."""

        vmap_update_ip = jax.vmap(fun=self.update_ip, in_axes=(0, 0))
        deps_dt_stack = material_points.deps_dt_stack
        rho_rho_0_stack = material_points.rho_stack / self.rho_0

        new_stress_stack = vmap_update_ip(deps_dt_stack, rho_rho_0_stack)

        new_particles = eqx.tree_at(
            lambda state: (state.stress_stack),
            material_points,
            (new_stress_stack),
        )

        return new_particles, self

    def update_ip(
        self: Self,
        deps_dt: TypeFloatMatrix3x3,
        rho_rho_0: TypeFloat,
    ) -> TypeFloatMatrix3x3:
        deps_dev_dt = get_dev_strain(deps_dt)

        p = self.K * ((rho_rho_0) ** self.alpha - 1.0)
        p = jnp.clip(p, 0, None)
        return -p * jnp.eye(3) + self.viscosity * deps_dev_dt

    def get_dt_crit(self, material_points, cell_size, dt_alpha=0.5):
        """Get critical timestep of material poiints for stability."""

        def vmap_dt_crit(rho, vel):
            cdil = jnp.sqrt(self.K / rho)

            c = jnp.abs(vel) + cdil * jnp.ones_like(vel)
            return c

        c_stack = jax.vmap(vmap_dt_crit)(
            material_points.rho_stack,
            material_points.velocity_stack,
        )
        return (dt_alpha * cell_size) / jnp.max(c_stack)
