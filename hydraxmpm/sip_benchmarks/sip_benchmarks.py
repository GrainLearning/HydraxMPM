# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-

from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from ..common.base import Base
from ..common.types import TypeFloat, TypeFloatMatrix3x3AStack
from ..material_points.material_points import MaterialPoints

from ..utils.math_helpers import get_pressure
from typing import Tuple


class SIPBenchmark(Base):
    L_control_stack: Optional[TypeFloatMatrix3x3AStack] = None
    X_control_stack: Optional[Any] = None
    L_unknown_indices: Optional[Any] = None
    init_material_points: bool = False

    def loss_stress(self, stress_guest, stress_target):
        return stress_guest - stress_target


class TriaxialConsolidatedUndrained(SIPBenchmark):
    """
    Triaxial consolidated undrained test

    This test is strain rate controlled and the volumetric strain is zero.

    (∂/∂t)ε_v= (∂/∂t)ε_xx + (∂/∂t)ε_yy + (∂/∂t)ε_zz = 0


    we adjust the radial strainrate to keep the volumetric strain zero

    (∂/∂t)ε_r = -(1/2)(∂/∂t)ε_xx = -(1/2)(∂/∂t)ε_yy = (∂/∂t)ε_zz


    >>> compression positive
    >>> L =
    >>> [-(0.5)*deps_r_dt,0,0]
    >>> [',-(0.5)*deps_r_dt,0]
    >>> [',',-(0.5)*deps_zz_dt]
    with
    >>> deps_r_dt = -1/2 * deps_zz_dt



    """

    deps_zz_dt: TypeFloat

    num_steps: int

    p0: TypeFloat = None

    init_material_points: bool = False

    def __init__(
        self,
        deps_zz_dt: TypeFloat,
        num_steps: int,
        p0: TypeFloat = None,
        init_material_points: Optional[bool] = False,
        **kwargs,
    ):
        self.deps_zz_dt = deps_zz_dt

        self.p0 = p0

        self.num_steps = num_steps

        self.init_material_points = init_material_points
        self.L_control_stack = kwargs.get("L_control_stack", None)
        self.X_control_stack = kwargs.get("X_control_stack", None)
        self.L_unknown_indices = None
        super().__init__(**kwargs)

    def init_state(self, material_points: MaterialPoints, **kwargs):
        # deps_zz_dt_stack = jnp.ones(self.num_steps) * self.deps_zz_dt

        if eqx.is_array(self.deps_zz_dt):
            if len(self.deps_zz_dt.shape) > 2:
                deps_zz_dt_stack = self.deps_zz_dt
            else:
                deps_zz_dt_stack = jnp.ones(self.num_steps) * self.deps_zz_dt
        else:
            deps_zz_dt_stack = jnp.ones(self.num_steps) * self.deps_zz_dt

        def get_L(deps_zz_dt):
            deps_r_dt = -1 / 2 * deps_zz_dt

            L = jnp.zeros((3, 3))
            L = L.at[0, 0].set(deps_r_dt)
            L = L.at[1, 1].set(deps_r_dt)
            L = L.at[2, 2].set(deps_zz_dt)
            return L

        L_control_stack = jax.vmap(get_L)(deps_zz_dt_stack)

        new_material_points = material_points

        if self.init_material_points:
            L_stack = L_control_stack.at[0].get()
            new_material_points = new_material_points.replace(
                L_stack=jnp.tile(L_stack, (material_points.num_points, 1, 1))
            )
            if self.p0 is not None:
                new_material_points = new_material_points.init_stress_from_p_0(
                    jnp.array(self.p0)
                )

        return TriaxialConsolidatedUndrained(
            deps_zz_dt=self.deps_zz_dt,
            p0=self.p0,
            num_steps=self.num_steps,
            L_control_stack=L_control_stack,
        ), new_material_points


class TriaxialConsolidatedDrained(SIPBenchmark):
    """
    Triaxial consolidated drained test

    # in this test the stress ratio is kept constant

    p =(1/3)*(2σ_rr+σ_zz)

    q =(σ_rr-σ_zz)


    # This test is strain rate controlled and the volumetric strain is zero.

    # σ_rr = σ_xx = σ_yy ≠ σ_zz

    # we adjust the radial strain rate to keep confining pressure constant

    >>> compression positive
    >>> L =
    >>> [?,0,0]
    >>> [',?,0]
    >>> [',',-deps_zz_dt]

    with
    >>> sigma =
    >>> [sigma_rr,0,0]
    >>> [0,sigma_rr,0]
    >>> [0,0,sigma_zz]


    we input a confing pressure and axial strain rate

    assumes initial state is sigma_x=sigma_y=sigma_z=p0


    """

    deps_zz_dt: TypeFloat

    num_steps: int

    p0: TypeFloat

    init_material_points: bool = False

    def __init__(
        self,
        deps_zz_dt: TypeFloat,
        num_steps: int,
        p0: TypeFloat = None,
        init_material_points: Optional[bool] = False,
        **kwargs,
    ):
        self.deps_zz_dt = deps_zz_dt

        self.p0 = p0

        self.num_steps = num_steps

        self.init_material_points = init_material_points
        self.L_control_stack = kwargs.get("L_control_stack", None)
        self.X_control_stack = kwargs.get("X_control_stack", None)
        self.L_unknown_indices = [0, 1], [0, 1]
        super().__init__(**kwargs)

    def init_state(self, material_points: MaterialPoints, **kwargs):
        deps_zz_dt_stack = jnp.ones(self.num_steps) * self.deps_zz_dt

        p_stack = jnp.ones(self.num_steps) * self.p0  # confining pressure

        def get_L(deps_zz_dt):
            L = jnp.zeros((3, 3))
            # L = L.at[0, 0].set(-(0.5) * deps_r_dt)
            # L = L.at[1, 1].set(-(0.5) * deps_r_dt)
            L = L.at[2, 2].set(deps_zz_dt)
            return L

        L_control_stack = jax.vmap(get_L)(deps_zz_dt_stack)

        new_material_points = material_points

        if self.init_material_points:
            L_stack = L_control_stack.at[0].get()
            new_material_points = new_material_points.replace(
                L_stack=jnp.tile(L_stack, (material_points.num_points, 1, 1))
            )
            if self.p0 is not None:
                new_material_points = new_material_points.init_stress_from_p_0(
                    jnp.array(self.p0)
                )

        return TriaxialConsolidatedDrained(
            deps_zz_dt=self.deps_zz_dt,
            p0=self.p0,
            num_steps=self.num_steps,
            L_control_stack=L_control_stack,
            X_control_stack=p_stack,
        ), new_material_points

    def loss_stress(self, stress_guest, X_target):
        sigma_xx_yy = stress_guest.at[self.L_unknown_indices].get()
        R = sigma_xx_yy + X_target * jnp.ones(2)  # stress tensor has opposite sign
        R_norm = jnp.linalg.norm(R)

        return R_norm**2


class ConstantPressureShear(SIPBenchmark):
    """

    Pressure controlled shear or drained shear


    Strain rate control
    >>> [?,x/2,0]
    >>> [',?,0]
    >>> [',',?]


    """

    deps_xy_dt: TypeFloat | Tuple[TypeFloat, ...]
    p0: TypeFloat
    num_steps: int

    init_material_points: bool = False

    def __init__(
        self,
        deps_xy_dt: TypeFloat | Tuple[TypeFloat, ...],
        p0: TypeFloat,
        num_steps: int,
        init_material_points: Optional[bool] = False,
        **kwargs,
    ):
        self.deps_xy_dt = deps_xy_dt
        self.p0 = p0

        self.num_steps = num_steps

        self.init_material_points = init_material_points
        self.L_control_stack = kwargs.get("L_control_stack", None)
        self.X_control_stack = kwargs.get("X_control_stack", None)
        self.L_unknown_indices = [0, 1, 2], [0, 1, 2]

        super().__init__(**kwargs)

    def init_state(self, material_points: MaterialPoints, **kwargs):
        if eqx.is_array(self.p0):
            if len(self.p0.shape) > 2:
                p_stack = self.p0
            else:
                p_stack = jnp.ones(self.num_steps) * self.p0
        else:
            p_stack = jnp.ones(self.num_steps) * self.p0

        if eqx.is_array(self.deps_xy_dt):
            if len(self.deps_xy_dt.shape) > 2:
                deps_xy_dt_stack = self.deps_xy_dt
            else:
                deps_xy_dt_stack = jnp.ones(self.num_steps) * self.deps_xy_dt
        else:
            deps_xy_dt_stack = jnp.ones(self.num_steps) * self.deps_xy_dt

        def get_L(deps_xy_dt):
            L = jnp.zeros((3, 3))
            L = L.at[0, 1].set(deps_xy_dt)
            return L

        L_control_stack = jax.vmap(get_L)(deps_xy_dt_stack)

        new_material_points = material_points

        if self.init_material_points:
            L_stack = L_control_stack.at[0].get()
            new_material_points = new_material_points.replace(
                L_stack=jnp.tile(L_stack, (material_points.num_points, 1, 1))
            )
            new_material_points = new_material_points.init_stress_from_p_0(
                p_stack.at[0].get()
            )

        return ConstantPressureShear(
            deps_xy_dt=self.deps_xy_dt,
            p0=self.p0,
            num_steps=self.num_steps,
            L_control_stack=L_control_stack,
            X_control_stack=p_stack,
        ), new_material_points

    def loss_stress(self, stress_guest, X_target):
        p = get_pressure(stress_guest)
        return p - X_target


class IsotropicCompression(SIPBenchmark):
    """
    Isotropic compression

    A constant strain rate is applied

    x = (∂/∂t)ε_xx = (∂/∂t) ε_yy = (∂/∂t) ε_zz

    >>> L = [x/2,     0,      0]
    >>>     [0  ,     -x/2,   0]
    >>>     [0  ,     0,    x/2]


    """

    deps_xx_yy_zz_dt: TypeFloat | Tuple[TypeFloat, ...]
    p0: TypeFloat
    num_steps: int

    init_material_points: bool = False

    def __init__(
        self,
        deps_xx_yy_zz_dt: TypeFloat | Tuple[TypeFloat, ...],
        p0: TypeFloat,
        num_steps: int,
        init_material_points: Optional[bool] = False,
        **kwargs,
    ):
        self.deps_xx_yy_zz_dt = deps_xx_yy_zz_dt
        self.p0 = p0

        self.num_steps = num_steps

        self.init_material_points = init_material_points
        self.L_control_stack = kwargs.get("L_control_stack", None)
        self.X_control_stack = kwargs.get("X_control_stack", None)
        self.L_unknown_indices = None

        super().__init__(**kwargs)

    def init_state(self, material_points: MaterialPoints, **kwargs):
        p_stack = jnp.ones(self.num_steps) * self.p0

        if eqx.is_array(self.deps_xx_yy_zz_dt):
            if len(self.deps_xx_yy_zz_dt.shape) > 2:
                deps_xx_yy_zz_dt = self.deps_xx_yy_zz_dt
            else:
                deps_xx_yy_zz_dt = jnp.ones(self.num_steps) * self.deps_xx_yy_zz_dt
        else:
            deps_xx_yy_zz_dt = jnp.ones(self.num_steps) * self.deps_xx_yy_zz_dt

        def get_L(deps_xx_yy_zz_dt):
            L = jnp.zeros((3, 3))
            L = L.at[0, 0].set(-deps_xx_yy_zz_dt)
            L = L.at[1, 1].set(-deps_xx_yy_zz_dt)
            L = L.at[2, 2].set(-deps_xx_yy_zz_dt)
            return L

        L_control_stack = jax.vmap(get_L)(deps_xx_yy_zz_dt)

        new_material_points = material_points

        if self.init_material_points:
            L_stack = L_control_stack.at[0].get()
            new_material_points = new_material_points.replace(
                L_stack=jnp.tile(L_stack, (material_points.num_points, 1, 1))
            )
            new_material_points = new_material_points.init_stress_from_p_0(
                p_stack.at[0].get()
            )

        return IsotropicCompression(
            deps_xx_yy_zz_dt=deps_xx_yy_zz_dt,
            p0=self.p0,
            num_steps=self.num_steps,
            L_control_stack=L_control_stack,
        ), new_material_points


class ConstantVolumeShear(SIPBenchmark):
    """

    Volume controlled shear or drained shear

    """

    deps_xy_dt: TypeFloat | Tuple[TypeFloat, ...]
    p0: TypeFloat
    num_steps: int

    init_material_points: bool = False

    def __init__(
        self,
        deps_xy_dt: TypeFloat | Tuple[TypeFloat, ...],
        p0: TypeFloat,
        num_steps: int,
        init_material_points: Optional[bool] = False,
        **kwargs,
    ):
        self.deps_xy_dt = deps_xy_dt
        self.p0 = p0

        self.num_steps = num_steps

        self.init_material_points = init_material_points
        self.L_control_stack = kwargs.get("L_control_stack", None)
        self.X_control_stack = None

        self.L_unknown_indices = None
        super().__init__(**kwargs)

    def init_state(self, material_points: MaterialPoints, **kwargs):
        p_stack = jnp.ones(self.num_steps) * self.p0

        if eqx.is_array(self.deps_xy_dt):
            if len(self.deps_xy_dt.shape) > 2:
                deps_xy_dt_stack = self.deps_xy_dt
            else:
                deps_xy_dt_stack = jnp.ones(self.num_steps) * self.deps_xy_dt
        else:
            deps_xy_dt_stack = jnp.ones(self.num_steps) * self.deps_xy_dt

        def get_L(deps_xy_dt):
            L = jnp.zeros((3, 3))
            L = L.at[0, 1].set(deps_xy_dt)
            return L

        L_control_stack = jax.vmap(get_L)(deps_xy_dt_stack)

        new_material_points = material_points

        if self.init_material_points:
            L_stack = L_control_stack.at[0].get()
            new_material_points = new_material_points.replace(
                L_stack=jnp.tile(L_stack, (material_points.num_points, 1, 1))
            )
            new_material_points = new_material_points.init_stress_from_p_0(
                p_stack.at[0].get()
            )

        return ConstantPressureShear(
            deps_xy_dt=self.deps_xy_dt,
            p0=self.p0,
            num_steps=self.num_steps,
            L_control_stack=L_control_stack,
        ), new_material_points


class UniaxialCompression(SIPBenchmark):
    """
    Uniaxial compression

    A constant strain rate is applied

    x =   (∂/∂t) ε_zz

    >>> L = [0,     0,      0]
    >>>     [0  ,     0,   0]
    >>>     [0  ,     0,    x/2]


    """

    deps_zz_dt: TypeFloat | Tuple[TypeFloat, ...]
    p0: TypeFloat
    num_steps: int
    init_material_points: bool = False

    def __init__(
        self,
        deps_zz_dt: TypeFloat | Tuple[TypeFloat, ...],
        p0: TypeFloat,
        num_steps: int,
        init_material_points: Optional[bool] = False,
        **kwargs,
    ):
        self.deps_zz_dt = deps_zz_dt
        self.p0 = p0

        self.num_steps = num_steps

        self.init_material_points = init_material_points
        self.L_control_stack = kwargs.get("L_control_stack", None)
        self.X_control_stack = kwargs.get("X_control_stack", None)
        self.L_unknown_indices = None

        super().__init__(**kwargs)

    def init_state(self, material_points: MaterialPoints, **kwargs):
        p_stack = jnp.ones(self.num_steps) * self.p0

        if eqx.is_array(self.deps_zz_dt):
            if len(self.deps_zz_dt.shape) > 2:
                deps_zz_dt = self.deps_zz_dt
            else:
                deps_zz_dt = jnp.ones(self.num_steps) * self.deps_zz_dt
        else:
            deps_zz_dt = jnp.ones(self.num_steps) * self.deps_zz_dt

        def get_L(deps_zz_dt):
            L = jnp.zeros((3, 3))
            L = L.at[2, 2].set(-deps_zz_dt)
            return L

        L_control_stack = jax.vmap(get_L)(deps_zz_dt)

        new_material_points = material_points

        if self.init_material_points:
            L_stack = L_control_stack.at[0].get()
            new_material_points = new_material_points.replace(
                L_stack=jnp.tile(L_stack, (material_points.num_points, 1, 1))
            )
            new_material_points = new_material_points.init_stress_from_p_0(
                p_stack.at[0].get()
            )

        return UniaxialCompression(
            deps_zz_dt=deps_zz_dt,
            p0=self.p0,
            num_steps=self.num_steps,
            L_control_stack=L_control_stack,
        ), new_material_points
