from typing import Optional, Self, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from ..common.types import TypeFloat, TypeFloatMatrix3x3, TypeInt
from ..material_points.material_points import MaterialPoints
from ..utils.math_helpers import get_sym_tensor_stack
from .constitutive_law import ConstitutiveLaw


def get_bulk_modulus(E, nu):
    return E / (3.0 * (1.0 - 2.0 * nu))


def get_shear_modulus(E, nu):
    return E / (2.0 * (1.0 + nu))


def get_lame_modulus(E, nu):
    return E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


class LinearIsotropicElastic(ConstitutiveLaw):
    """Isotropic linear elastic material solved in incremental form.

    Attributes:
        E: Young's modulus.
        nu: Poisson's ratio.
        G: Shear modulus.
        K: Bulk modulus.
        lam: Lame modulus.
    """

    E: TypeFloat
    nu: TypeFloat
    G: TypeFloat = eqx.field(init=False)
    K: TypeFloat = eqx.field(init=False)

    lam: TypeFloat

    def __init__(
        self: Self,
        E: TypeFloat,
        nu: TypeFloat,
        **kwargs,
    ) -> Self:
        """Initialize the isotropic linear elastic material."""

        self.E = E
        self.nu = nu

        # post init
        self.K = get_bulk_modulus(E, nu)
        self.G = get_shear_modulus(E, nu)
        self.lam = get_lame_modulus(E, nu)

        super().__init__(**kwargs)

    def update(
        self: Self,
        material_points: MaterialPoints,
        dt: TypeFloat,
        dim: Optional[TypeInt] = 3,
    ) -> Tuple[MaterialPoints, Self]:
        """Update the material state and particle stresses for MPM solver."""

        vmap_update_ip = jax.vmap(fun=self.update_ip, in_axes=(None, 0, 0, 0))

        new_stress_stack = vmap_update_ip(
            dim,
            material_points.stress_stack,
            material_points.F_stack,
            material_points.deps_stack,
        )

        new_material_points = eqx.tree_at(
            lambda state: (state.stress_stack),
            material_points,
            (new_stress_stack),
        )
        return new_material_points, self

    def update_ip(
        self: Self,
        dim: TypeInt,
        stress_prev: TypeFloatMatrix3x3,
        F: TypeFloatMatrix3x3,
        deps: TypeFloatMatrix3x3,
    ) -> TypeFloatMatrix3x3:
        """Update stress on a single integration point"""

        if dim == 2:
            deps = deps.at[:, [2, 2]].set(0.0)

        return (
            stress_prev + self.lam * jnp.trace(deps) * jnp.eye(3) + 2.0 * self.G * deps
        )
