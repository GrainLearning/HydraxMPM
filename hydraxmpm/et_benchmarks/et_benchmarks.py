from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from ..common.base import Base
from ..common.types import TypeFloat, TypeFloatMatrix3x3AStack, TypeInt
from ..material_points.material_points import MaterialPoints
from ..solvers.config import Config
from ..utils.math_helpers import get_pressure, get_q_vm
from typing import Tuple
import optimistix as optx


class ETBenchmark(Base):
    L_control_stack: Optional[TypeFloatMatrix3x3AStack] = None
    X_control_stack: Optional[Any] = None
    L_unknown_indices: Optional[Any] = None
    init_material_points: bool = False

    def loss_stress(self, stress_guest, stress_target):
        return stress_guest - stress_target


class TRX_CU(ETBenchmark):
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

    num_steps: Optional[TypeInt]

    p0: TypeFloat = None

    init_material_points: bool = False

    def __init__(
        self,
        deps_zz_dt: TypeFloat,
        num_steps: Optional[TypeInt] = None,
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

    def init_state(self, num_steps, material_points: MaterialPoints, **kwargs):
        deps_zz_dt_stack = jnp.ones(num_steps) * self.deps_zz_dt

        def get_L(deps_zz_dt):
            deps_r_dt = -1 / 2 * deps_zz_dt

            L = jnp.zeros((3, 3))
            L = L.at[0, 0].set(-(0.5) * deps_r_dt)
            L = L.at[1, 1].set(-(0.5) * deps_r_dt)
            L = L.at[2, 2].set(-(0.5) * deps_zz_dt)
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

        return TRX_CU(
            deps_zz_dt=self.deps_zz_dt,
            p0=self.p0,
            num_steps=num_steps,
            L_control_stack=L_control_stack,
        ), new_material_points


class TRX_CD(ETBenchmark):
    """
    Triaxial consolidated udrained test

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
    >>> [',',-(0.5)*deps_zz_dt]

    with
    >>> sigma =
    >>> [sigma_rr,0,0]
    >>> [0,sigma_rr,0]
    >>> [0,0,sigma_zz]


    we input a confing pressure and axial strain rate

    assumes initial state is sigma_x=sigma_y=sigma_z=p0


    """

    deps_zz_dt: TypeFloat

    num_steps: Optional[TypeInt]

    p0: TypeFloat

    init_material_points: bool = False

    def __init__(
        self,
        deps_zz_dt: TypeFloat,
        p0: TypeFloat = None,
        num_steps: Optional[TypeInt] = None,
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

    def init_state(self, num_steps, material_points: MaterialPoints, **kwargs):
        deps_zz_dt_stack = jnp.ones(num_steps) * self.deps_zz_dt

        p_stack = jnp.ones(num_steps) * self.p0  # confining pressure

        def get_L(deps_zz_dt):
            L = jnp.zeros((3, 3))
            # L = L.at[0, 0].set(-(0.5) * deps_r_dt)
            # L = L.at[1, 1].set(-(0.5) * deps_r_dt)
            L = L.at[2, 2].set(-(0.5) * deps_zz_dt)
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

        return TRX_CD(
            deps_zz_dt=self.deps_zz_dt,
            p0=self.p0,
            num_steps=num_steps,
            L_control_stack=L_control_stack,
            X_control_stack=p_stack,
        ), new_material_points

    def loss_stress(self, stress_guest, X_target):
        sigma_xx_yy = stress_guest.at[self.L_unknown_indices].get()
        R = sigma_xx_yy + X_target * jnp.ones(2)  # stress tensor has opposite sign
        R_norm = jnp.linalg.norm(R)

        return R_norm**2


class S_CD(ETBenchmark):
    """
    Drained simple shear

        Strain rate control
        >>> [?,x/2,0]
        >>> [',?,0]
        >>> [',',?]

        Pressure control

    """

    deps_xy_dt: TypeFloat | Tuple[TypeFloat, ...]
    p0: TypeFloat
    num_steps: Optional[TypeInt]

    init_material_points: bool = False

    def __init__(
        self,
        deps_xy_dt: TypeFloat | Tuple[TypeFloat, ...],
        p0: TypeFloat,
        num_steps: Optional[TypeInt] = None,
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

    def init_state(self, num_steps, material_points: MaterialPoints, **kwargs):
        p_stack = jnp.ones(num_steps) * self.p0

        if eqx.is_array(self.deps_xy_dt):
            if len(self.deps_xy_dt.shape) > 2:
                deps_xy_dt_stack = self.deps_xy_dt
            else:
                deps_xy_dt_stack = jnp.ones(num_steps) * self.deps_xy_dt
        else:
            deps_xy_dt_stack = jnp.ones(num_steps) * self.deps_xy_dt

        def get_L(deps_xy_dt):
            L = jnp.zeros((3, 3))
            L = L.at[0, 1].set(deps_xy_dt / 2)
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

        return S_CD(
            deps_xy_dt=self.deps_xy_dt,
            p0=self.p0,
            num_steps=num_steps,
            L_control_stack=L_control_stack,
            X_control_stack=p_stack,
        ), new_material_points

    def loss_stress(self, stress_guest, X_target):
        return get_pressure(stress_guest) - X_target


class ISO_C(ETBenchmark):
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
    num_steps: Optional[TypeInt]

    init_material_points: bool = False

    def __init__(
        self,
        deps_xx_yy_zz_dt: TypeFloat | Tuple[TypeFloat, ...],
        p0: TypeFloat,
        num_steps: Optional[TypeInt] = None,
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

    def init_state(self, num_steps, material_points: MaterialPoints, **kwargs):
        p_stack = jnp.ones(num_steps) * self.p0

        if eqx.is_array(self.deps_xx_yy_zz_dt):
            if len(self.deps_xx_yy_zz_dt.shape) > 2:
                deps_xx_yy_zz_dt = self.deps_xx_yy_zz_dt
            else:
                deps_xx_yy_zz_dt = jnp.ones(num_steps) * self.deps_xx_yy_zz_dt
        else:
            deps_xx_yy_zz_dt = jnp.ones(num_steps) * self.deps_xx_yy_zz_dt

        def get_L(deps_xx_yy_zz_dt):
            L = jnp.zeros((3, 3))
            L = L.at[0, 0].set(-deps_xx_yy_zz_dt / 2)
            L = L.at[1, 1].set(-deps_xx_yy_zz_dt / 2)
            L = L.at[2, 2].set(-deps_xx_yy_zz_dt / 2)
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

        return S_CD(
            deps_xy_dt=deps_xx_yy_zz_dt,
            p0=self.p0,
            num_steps=num_steps,
            L_control_stack=L_control_stack,
        ), new_material_points


# class ConstantVolumeSimpleShear(ETBenchmark):
#     """
#     Drained simple shear

#         Strain rate control
#         >>> [-y,x,0]
#         >>> [',-y,0]
#         >>> [',',-y]

#         Stress control
#         >>> [?,?,?]
#         >>> [',?,?]
#         >>> [',',?]
#     """

#     x: TypeFloat | Tuple[TypeFloat, ...]
#     num_steps: Optional[TypeInt]

#     def __init__(
#         self,
#         x: TypeFloat | Tuple[TypeFloat, ...],
#         num_steps: Optional[TypeInt] = None,
#         init_material_points: Optional[bool] = False,
#         **kwargs,
#     ):
#         self.x = x

#         self.num_steps = num_steps

#         self.init_material_points = init_material_points
#         self.L_control_stack = kwargs.get("L_control_stack", None)
#         self.L_unknown_indices = None
#         self.X_control_stack = None

#         super().__init__(**kwargs)

#     def init_state(self, config: Config, material_points: MaterialPoints, **kwargs):
#         num_steps = self.num_steps
#         if num_steps is None:
#             num_steps = config.num_steps

#         if isinstance(self.x, tuple):
#             x_stack = jnp.linspace(self.x[0], self.x[1], num_steps)
#         else:
#             x_stack = jnp.ones(num_steps) * self.x

#         def get_L(x):
#             L = jnp.zeros((3, 3))
#             L = L.at[0, 1].set(x)
#             return L

#         L_control_stack = jax.vmap(get_L)(x_stack)

#         new_material_points = material_points
#         if self.init_material_points:
#             L_stack = L_control_stack.at[0].get()
#             new_material_points = material_points.replace(
#                 L_stack=jnp.tile(L_stack, (material_points.num_points, 1, 1))
#             )
#         return ConstantVolumeSimpleShear(
#             x=self.x,
#             num_steps=num_steps,
#             L_control_stack=L_control_stack,
#         ), new_material_points


# class IsotropicCompression(ETBenchmark):
#     """
#     Strain rate control
#     >>> [-x,0,0]
#     >>> [',-x,0]
#     >>> [',',-x]

#     Stress control
#     >>> [?,?,?]
#     >>> [',?,?]
#     >>> [',',?]
#     """

#     x_range: tuple[TypeFloat, TypeFloat]

#     num_steps: Optional[TypeInt]

#     def __init__(
#         self,
#         x_range: tuple[TypeFloat, TypeFloat],
#         num_steps: Optional[TypeInt] = None,
#         **kwargs,
#     ):
#         self.x_range = x_range

#         self.num_steps = num_steps

#         self.L_control_stack = kwargs.get("L_control_stack", None)

#         super().__init__(**kwargs)

#     def init_state(self, config: Config, material_points: MaterialPoints, **kwargs):
#         num_steps = self.num_steps
#         if num_steps is None:
#             num_steps = config.num_steps

#         x_stack = jnp.linspace(self.x_range[0], self.x_range[1], num_steps)

#         def get_L(x):
#             L = jnp.zeros((3, 3))
#             L = L.at[[0, 1, 2], [0, 1, 2]].set(-x)
#             return L

#         L_control_stack = jax.vmap(get_L)(x_stack)

#         return self.__class__(
#             x_range=self.x_range,
#             num_steps=num_steps,
#             L_control_stack=L_control_stack,
#             stress_control_stack=None,
#             stress_mask_indices=None,
#             _setup_done=True,
#         ), material_points
