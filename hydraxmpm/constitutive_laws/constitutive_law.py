from typing import Optional, Self, Tuple, Union

import equinox as eqx
import jax.numpy as jnp
import jax

from ..common.base import Base
from ..common.types import (
    TypeFloat,
    TypeFloatMatrixPStack,
    TypeFloatScalarAStack,
    TypeInt,
    TypeFloatScalarPStack,
)
from ..material_points.material_points import MaterialPoints

from ..utils.math_helpers import get_double_contraction, get_sym_tensor


class ConstitutiveLaw(Base):
    rho_0: Optional[Union[TypeFloatScalarAStack, TypeFloat]] = None
    p_0: Optional[Union[TypeFloatScalarAStack, TypeFloat]] = None
    d: float = eqx.field(static=True, default=1.0)
    rho_p: float = eqx.field(static=True, default=1.0)

    # for elastoplastic models
    eps_e_stack: Optional[TypeFloatMatrixPStack] = None

    init_by_density: bool = eqx.field(static=True, default=False)

    approx_stress_power: bool = eqx.field(static=True, default=False)

    approx_strain_energy_density: bool = eqx.field(static=True, default=False)

    W_stack: Optional[TypeFloatScalarPStack] = None

    P_stack: Optional[TypeFloatScalarPStack] = None

    def __init__(self, **kwargs):
        self.d = kwargs.get("d")
        self.p_0 = kwargs.get("p_0")
        phi_0 = kwargs.get("phi_0")
        self.rho_p = kwargs.get("rho_p", 1.0)

        self.init_by_density = kwargs.get("init_by_density", False)

        if phi_0 is None:
            self.rho_0 = kwargs.get("rho_0", 1.0)
        else:
            self.rho_0 = self.rho_p * phi_0

        # strain energy density
        self.approx_stress_power = kwargs.get("approx_stress_power", False)
        self.P_stack = kwargs.get("P_stack")

        self.approx_strain_energy_density = kwargs.get(
            "approx_strain_energy_density", False
        )
        self.W_stack = kwargs.get("W_stack")

        super().__init__(**kwargs)

    def update(
        self: Self,
        material_points: MaterialPoints,
        dt: TypeFloat,
        dim: Optional[TypeInt] = 3,
    ) -> Tuple[MaterialPoints, Self]:
        pass

    def init_state(
        self: Self,
        material_points: MaterialPoints,
        **kwargs,
    ) -> Tuple[Self, MaterialPoints]:
        p_0 = self.p_0
        if p_0 is None:
            p_0 = material_points.p_stack

        material_points = material_points.init_stress_from_p_0(p_0)

        material_points = material_points.init_mass_from_rho_0(self.rho_0)

        W_stack = None
        if self.approx_strain_energy_density:
            W_stack = jnp.zeros(material_points.num_points)

        P_stack = None
        if self.approx_stress_power:
            P_stack = jnp.zeros(material_points.num_points)

        params = self.__dict__

        params.update(
            p_0=p_0,
            W_stack=W_stack,
            P_stack=P_stack,
            **kwargs,
        )
        return self.__class__(**params), material_points

    def post_update(self, next_stress_stack, deps_dt_stack, dt, **kwargs):
        """

        Get stress power, strain energy density (Explicit euler)

        """
        # TODO is there a smarter way to do this, without all the ifs?
        if (self.approx_stress_power) and (self.approx_strain_energy_density):
            P_stack = self.get_stress_power(next_stress_stack, deps_dt_stack)
            W_stack = P_stack * dt + self.W_stack
            return self.replace(W_stack=W_stack, P_stack=P_stack)
        elif self.approx_stress_power:
            P_stack = self.get_stress_power(next_stress_stack, deps_dt_stack)
            return self.replace(P_stack=P_stack)
        elif self.approx_strain_energy_density:
            P_stack = self.get_stress_power(next_stress_stack, deps_dt_stack)
            W_stack = P_stack * dt + self.W_stack
            return self.replace(W_stack=W_stack)

    def get_stress_power(self, stress_stack, deps_dt_stack):
        """
        Compute stress power
        P=sigma:D
        """

        def vmap_stress_power(stress_next, deps_dt):
            return get_double_contraction(stress_next, deps_dt)

        P_stack = jax.vmap(vmap_stress_power)(stress_stack, deps_dt_stack)

        return P_stack

    @property
    def phi_0(self):
        """Assumes dry case"""
        return self.rho_0 / self.rho_p

    @property
    def lnv_0(self):
        v = 1.0 / self.phi_0
        return jnp.log(v)
