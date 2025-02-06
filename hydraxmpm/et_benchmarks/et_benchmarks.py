from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from ..common.base import Base
from ..common.types import TypeFloat, TypeFloatMatrix3x3AStack, TypeInt


class ETBenchmark(Base):
    L_control_stack: Optional[TypeFloatMatrix3x3AStack] = None
    stress_control_stack: Optional[TypeFloatMatrix3x3AStack] = None
    stress_mask_indices: Optional = None


class VolumeControlShear(ETBenchmark):
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

    x_range: tuple[TypeFloat, TypeFloat]
    y_range: tuple[TypeFloat, TypeFloat]
    num_steps: Optional[TypeInt]

    def __init__(
        self,
        x_range: tuple[TypeFloat, TypeFloat],
        y_range: tuple[TypeFloat, TypeFloat],
        num_steps: Optional[TypeInt] = None,
        **kwargs,
    ):
        self.x_range = x_range
        self.y_range = y_range

        self.num_steps = num_steps

        self.L_control_stack = kwargs.get("L_control_stack", None)

        super().__init__(**kwargs)

    def init_steps(self, num_steps: TypeInt, **kwargs):
        # set velocity gradient
        if self.num_steps is not None:
            num_steps = self.num_steps

        x_stack = jnp.linspace(self.x_range[0], self.x_range[1], num_steps)
        y_stack = jnp.linspace(self.y_range[0], self.y_range[1], num_steps)

        def get_L(x, y):
            L = jnp.zeros((3, 3))
            L = L.at[[0, 1], [1, 0]].set(x)
            L = L.at[[0, 1, 2], [0, 1, 2]].set(-y)
            return L

        L_control_stack = jax.vmap(get_L)(x_stack, y_stack)

        return self.__class__(
            x_range=self.x_range,
            y_range=self.y_range,
            num_steps=num_steps,
            L_control_stack=L_control_stack,
            stress_control_stack=None,
            stress_mask_indices=None,
            _setup_done=True,
        )


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

    def init_steps(self, num_steps: TypeInt, **kwargs):
        # set velocity gradient
        if self.num_steps is not None:
            num_steps = self.num_steps

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
        )
