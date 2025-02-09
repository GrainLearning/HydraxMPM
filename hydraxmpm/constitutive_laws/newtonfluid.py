"""Constitutive model for a nearly incompressible Newtonian fluid."""

from functools import partial
from typing import Optional, Self, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from ..common.types import TypeFloat, TypeFloatMatrix3x3, TypeInt
from ..material_points.material_points import MaterialPoints
from .constitutive_law import ConstitutiveLaw


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

    init_by_density: bool = True

    def __init__(
        self: Self,
        K: TypeFloat = 2.0 * 10**6,
        viscosity: TypeFloat = 0.001,
        alpha: TypeFloat = 7.0,
        init_by_density: bool = True,
        **kwargs,
    ) -> Self:
        """Initialize the nearly incompressible Newtonian fluid material."""

        self.K = K
        self.viscosity = viscosity
        self.alpha = alpha
        self.init_by_density = init_by_density
        super().__init__(**kwargs)

    def init_state(self: Self, material_points: MaterialPoints):
        # There are two ways to initialize via a reference pressure or reference density
        # these can be given as a scalar or array

        p_0 = self.p_0
        if p_0 is None:
            p_0 = material_points.p_stack

        rho_0 = self.rho_0

        rho = self.rho_0

        if self.init_by_density:
            if eqx.is_array(rho_0):
                vmap_give_p_ref = partial(
                    jax.vmap,
                    in_axes=(None, 0, None, None),
                )(give_p)
            else:
                vmap_give_p_ref = give_p

            p_0 = vmap_give_p_ref(self.K, rho_0, rho_0, self.alpha)
        else:
            if eqx.is_array(p_0):
                vmap_give_rho_ref = partial(
                    jax.vmap,
                    in_axes=(None, None, 0, None),
                )(give_rho)
            else:
                vmap_give_rho_ref = give_rho

                rho = vmap_give_rho_ref(self.K, self.rho_0, p_0, self.alpha)

        material_points = material_points.init_stress_from_p_0(p_0)
        # if there is pressure, then density is not on reference density
        material_points = material_points.init_mass_from_rho_0(rho)
        params = self.__dict__
        params.update(rho_0=rho_0, p_0=p_0)
        return self.__class__(**params), material_points

    def update(
        self: Self,
        material_points: MaterialPoints,
        dt: TypeFloat,
        dim: Optional[TypeInt] = 3,
    ) -> Tuple[MaterialPoints, Self]:
        """Update the material state and particle stresses for MPM solver."""

        vmap_update_ip = jax.vmap(fun=self.update_ip, in_axes=(0, 0, 0, 0, None))

        new_stress_stack = vmap_update_ip(
            material_points.stress_stack,
            material_points.F_stack,
            material_points.L_stack,
            material_points.rho_stack,
            dim,
        )
        new_particles = eqx.tree_at(
            lambda state: (state.stress_stack),
            material_points,
            (new_stress_stack),
        )

        return new_particles, self

    def update_ip(
        self: Self,
        stress_prev: TypeFloatMatrix3x3,
        F: TypeFloatMatrix3x3,
        L: TypeFloatMatrix3x3,
        rho: TypeFloat,
        dim: TypeInt,
    ) -> TypeFloatMatrix3x3:
        pressure = give_p(self.K, rho, self.rho_0, self.alpha)

        deps_dt = 0.5 * (L + L.T)

        if dim == 2:
            deps_dt = deps_dt.at[:, [2, 2]].set(0.0)

        deps_v_dt = jnp.trace(deps_dt)

        deps_dev_dt = deps_dt - (deps_v_dt / 3) * jnp.eye(3)

        return -pressure * jnp.eye(3) + self.viscosity * deps_dev_dt
