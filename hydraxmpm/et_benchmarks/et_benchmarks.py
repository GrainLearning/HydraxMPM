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


class ETBenchmark(Base):
    L_control_stack: Optional[TypeFloatMatrix3x3AStack] = None
    X_control_stack: Optional[Any] = None
    L_unknown_indices: Optional[Any] = None
    init_material_points: bool = False

    def loss_stress(self, stress_guest, stress_target):
        return stress_guest - stress_target


class TriaxialCompressionDrained(ETBenchmark):
    """
    Drained simple shear

        Strain rate control
        >>> [?,0,0]
        >>> [',?,0]
        >>> [',',x]

        Pressure control
    """

    x: TypeFloat
    p: TypeFloat
    num_steps: Optional[TypeInt]

    init_material_points: bool = False

    def __init__(
        self,
        x: TypeFloat,
        p: TypeFloat,
        num_steps: Optional[TypeInt] = None,
        init_material_points: Optional[bool] = False,
        **kwargs,
    ):
        self.x = x
        self.p = p

        self.num_steps = num_steps

        self.init_material_points = init_material_points
        self.L_control_stack = kwargs.get("L_control_stack", None)
        self.X_control_stack = kwargs.get("X_control_stack", None)
        self.L_unknown_indices = [0, 1], [0, 1]

        super().__init__(**kwargs)

    def init_state(self, config: Config, material_points: MaterialPoints, **kwargs):
        num_steps = self.num_steps
        if num_steps is None:
            num_steps = config.num_steps

        x_stack = jnp.ones(num_steps) * self.x
        p_stack = jnp.ones(num_steps) * self.p

        def get_L(x):
            L = jnp.zeros((3, 3))
            L = L.at[2, 2].set(x)
            return L

        L_control_stack = jax.vmap(get_L)(x_stack)

        new_material_points = material_points
        if self.init_material_points:
            L_stack = L_control_stack.at[0].get()
            new_material_points = new_material_points.replace(
                L_stack=jnp.tile(L_stack, (material_points.num_points, 1, 1))
            )
            new_material_points = new_material_points.init_stress_from_p_0(
                p_stack.at[0].get()
            )
        # print("L_control_stack", L_control_stack)
        return TriaxialCompressionDrained(
            x=self.x,
            p=self.p,
            num_steps=num_steps,
            L_control_stack=L_control_stack,
            X_control_stack=p_stack,
        ), new_material_points

    def loss_stress(self, stress_guest, X_target):
        # jax.debug.print("{}", X_target)
        return (
            stress_guest.at[self.L_unknown_indices].get() - jnp.ones(2) * X_target / 3
        )
        # p = get_pressure(stress_guest)
        # q = get_q_vm(stress_guest)
        # return q / p == 1 / 3
        # stress_guess_00 = stress_guest.at[0,0].get()

        # stress_guess_22 = stress_guest.at[2,2].get()
        # p= (1./3)*(stress_guess_00+
        # return stress_guess_00_11 -
        # return get_pressure(stress_guest) - X_target


class ConstantVolumeSimpleShear(ETBenchmark):
    """
    Drained simple shear

        Strain rate control
        >>> [-y,x,0]
        >>> [',-y,0]
        >>> [',',-y]

        Stress control
        >>> [?,?,?]
        >>> [',?,?]
        >>> [',',?]
    """

    x: TypeFloat | Tuple[TypeFloat, ...]
    num_steps: Optional[TypeInt]

    def __init__(
        self,
        x: TypeFloat | Tuple[TypeFloat, ...],
        num_steps: Optional[TypeInt] = None,
        init_material_points: Optional[bool] = False,
        **kwargs,
    ):
        self.x = x

        self.num_steps = num_steps

        self.init_material_points = init_material_points
        self.L_control_stack = kwargs.get("L_control_stack", None)
        self.L_unknown_indices = None
        self.X_control_stack = None

        super().__init__(**kwargs)

    def init_state(self, config: Config, material_points: MaterialPoints, **kwargs):
        num_steps = self.num_steps
        if num_steps is None:
            num_steps = config.num_steps

        if isinstance(self.x, tuple):
            x_stack = jnp.linspace(self.x[0], self.x[1], num_steps)
        else:
            x_stack = jnp.ones(num_steps) * self.x

        def get_L(x):
            L = jnp.zeros((3, 3))
            L = L.at[0, 1].set(x)
            return L

        L_control_stack = jax.vmap(get_L)(x_stack)

        new_material_points = material_points
        if self.init_material_points:
            L_stack = L_control_stack.at[0].get()
            new_material_points = material_points.replace(
                L_stack=jnp.tile(L_stack, (material_points.num_points, 1, 1))
            )
        return ConstantVolumeSimpleShear(
            x=self.x,
            num_steps=num_steps,
            L_control_stack=L_control_stack,
        ), new_material_points


class ConstantPressureSimpleShear(ETBenchmark):
    """
    Drained simple shear

        Strain rate control
        >>> [?,x,0]
        >>> [',?,0]
        >>> [',',?]

        Pressure control
    """

    x: TypeFloat | Tuple[TypeFloat, ...]
    p: TypeFloat
    num_steps: Optional[TypeInt]

    init_material_points: bool = False

    def __init__(
        self,
        x: TypeFloat | Tuple[TypeFloat, ...],
        p: TypeFloat,
        num_steps: Optional[TypeInt] = None,
        init_material_points: Optional[bool] = False,
        **kwargs,
    ):
        self.x = x
        self.p = p

        self.num_steps = num_steps

        self.init_material_points = init_material_points
        self.L_control_stack = kwargs.get("L_control_stack", None)
        self.X_control_stack = kwargs.get("X_control_stack", None)
        self.L_unknown_indices = [0, 1, 2], [0, 1, 2]

        super().__init__(**kwargs)

    def init_state(self, config: Config, material_points: MaterialPoints, **kwargs):
        num_steps = self.num_steps
        if num_steps is None:
            num_steps = config.num_steps
        p_stack = jnp.ones(num_steps) * self.p

        # if isinstance(self.x, tuple):
        # x_stack = jnp.linspace(self.x[0], self.x[1], num_steps)
        # elif eqx.is_array(self.x):
        x_stack = self.x
        # else:
        # x_stack = jnp.ones(num_steps) * self.x

        def get_L(x):
            L = jnp.zeros((3, 3))
            L = L.at[0, 1].set(x)
            return L

        L_control_stack = jax.vmap(get_L)(x_stack)

        new_material_points = material_points

        if self.init_material_points:
            L_stack = L_control_stack.at[0].get()

            new_material_points = new_material_points.replace(
                L_stack=jnp.tile(L_stack, (material_points.num_points, 1, 1))
            )

            new_material_points = new_material_points.init_stress_from_p_0(
                p_stack.at[0].get()
            )

        return ConstantPressureSimpleShear(
            x=self.x,
            p=self.p,
            num_steps=num_steps,
            L_control_stack=L_control_stack,
            X_control_stack=p_stack,
        ), new_material_points

    def loss_stress(self, stress_guest, X_target):
        return get_pressure(stress_guest) - X_target


class IsotropicCompression(ETBenchmark):
    """
    Strain rate control
    >>> [-x,0,0]
    >>> [',-x,0]
    >>> [',',-x]

    Stress control
    >>> [?,?,?]
    >>> [',?,?]
    >>> [',',?]
    """

    x_range: tuple[TypeFloat, TypeFloat]

    num_steps: Optional[TypeInt]

    def __init__(
        self,
        x_range: tuple[TypeFloat, TypeFloat],
        num_steps: Optional[TypeInt] = None,
        **kwargs,
    ):
        self.x_range = x_range

        self.num_steps = num_steps

        self.L_control_stack = kwargs.get("L_control_stack", None)

        super().__init__(**kwargs)

    def init_state(self, config: Config, material_points: MaterialPoints, **kwargs):
        num_steps = self.num_steps
        if num_steps is None:
            num_steps = config.num_steps

        x_stack = jnp.linspace(self.x_range[0], self.x_range[1], num_steps)

        def get_L(x):
            L = jnp.zeros((3, 3))
            L = L.at[[0, 1, 2], [0, 1, 2]].set(-x)
            return L

        L_control_stack = jax.vmap(get_L)(x_stack)

        return self.__class__(
            x_range=self.x_range,
            num_steps=num_steps,
            L_control_stack=L_control_stack,
            stress_control_stack=None,
            stress_mask_indices=None,
            _setup_done=True,
        ), material_points
