# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-

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

from ..utils.math_helpers import get_double_contraction


class ConvergenceControlConfig(eqx.Module):
    error_check: bool = eqx.field(static=True, default=False)
    rtol: float = eqx.field(static=True, default=1e-6)
    atol: float = eqx.field(static=True, default=1e-6)
    max_iter: int = eqx.field(static=True, default=100)
    throw: bool = eqx.field(static=True, default=True)
    lower_bound: tuple = eqx.field(static=True, default=(1e-6, 1e-6))
    upper_bound: tuple = eqx.field(static=True, default=(1e6, 1e6))


class ConstitutiveLaw(Base):
    rho_0: Optional[Union[TypeFloatScalarAStack, TypeFloat]] = None
    d: Optional[float] = eqx.field(static=True, default=1.0)
    rho_p: float = eqx.field(static=True, default=1.0)

    # for elastoplastic models
    eps_e_stack: Optional[TypeFloatMatrixPStack] = None

    approx_stress_power: bool = eqx.field(static=True, default=False)

    approx_strain_energy_density: bool = eqx.field(static=True, default=False)

    W_stack: Optional[TypeFloatScalarPStack] = None

    P_stack: Optional[TypeFloatScalarPStack] = None

    def __init__(self, **kwargs):
        self.d = kwargs.get("d")
        self.rho_p = kwargs.get("rho_p", 1.0)

        phi_0 = kwargs.get("phi_0")
        ln_v_0 = kwargs.get("ln_v_0")

        rho_0 = kwargs.get("rho_0")

        if rho_0 is not None:
            self.rho_0 = rho_0
        elif ln_v_0 is not None:
            phi_0 = 1.0 / jnp.exp(ln_v_0)
            self.rho_0 = self.rho_p * phi_0
        elif phi_0 is not None:
            self.rho_0 = self.rho_p * phi_0

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

    def post_init_state(
        self: Self,
        material_points: MaterialPoints,
        **kwargs,
    ):
        # this solves cases when rho_0 is not rho,
        # i.e., current density may be different from reference,
        # for example Newtonian fluids
        rho_0 = kwargs.get("rho_0", self.rho_0)
        rho = kwargs.get("rho", rho_0)
        material_points = material_points.init_mass_from_rho_0(rho)

        W_stack = None
        if self.approx_strain_energy_density:
            W_stack = jnp.zeros(material_points.num_points)

        P_stack = None
        if self.approx_stress_power:
            P_stack = jnp.zeros(material_points.num_points)

        params = self.__dict__
        params.update(
            W_stack=W_stack,
            P_stack=P_stack,
            **kwargs,
        )
        return self.__class__(**params), material_points

    def init_state(
        self: Self,
        material_points: MaterialPoints,
        **kwargs,
    ) -> Tuple[Self, MaterialPoints]:
        return self.post_init_state(material_points, **kwargs)

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
    def ln_v_0(self):
        v = 1.0 / self.phi_0
        return jnp.log(v)

    def get_dt_crit(self, material_points, cell_size, alpha=0.5, **kwargs):
        """Get critical timestep of material poiints for stability."""
        return kwargs.get("dt", 1e-6)
