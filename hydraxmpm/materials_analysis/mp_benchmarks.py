from typing import Tuple, Self

import chex
import jax
import jax.numpy as jnp

from ..materials.material import Material
from .mix_control import mix_control

import equinox as eqx

from ..config.ip_config import IPConfig


def mp_benchmark_volume_control_shear(
    config,
    material,
    phi_ref,
    stress_ref,
    x_range=(0, 0.1),
    y_range=(0.0, 0.1),
    F_ref=None,
    output=None,
    early_stop_iter=None,
    debug=False,
    return_carry=False,
):
    """
            Strain rate control
            >>> [-y,x,0]
            >>> [',-y,0]
            >>> [',',-y]

            Stress control
            >>> [?,?,?]
            >>> [',?,?]
            >>> [',',?]
    #"""
    x_stack = jnp.linspace(x_range[0], x_range[1], config.num_steps)
    y_stack = jnp.linspace(y_range[0], y_range[1], config.num_steps)

    def get_L(x, y):
        L = jnp.zeros((3, 3))
        L = L.at[[0, 1], [1, 0]].set(x)
        L = L.at[[0, 1, 2], [0, 1, 2]].set(-y)
        return L

    L_control_stack = jax.vmap(get_L)(x_stack, y_stack)

    carry, accumulated = mix_control(
        config=config,
        material=material,
        L_control_stack=L_control_stack.at[:early_stop_iter].get(),
        stress_control_stack=None,
        stress_mask_indices=None,
        stress_ref=stress_ref,
        F_ref=F_ref,
        phi_ref=phi_ref,
        output=output,
    )

    accumulated_next = []

    for i, _ in enumerate(output):
        accumulated_next.append(accumulated[i].at[0 :: config.store_every].get())

    if return_carry:
        return accumulated_next, carry
    return accumulated_next





#     return self.replace(
#         material=material_next,
#         stress_ref=stress_next,
#         F_ref=F_next,
#         phi_ref=phi_next,
#         accumulated=accumulated_next,
#     )


# class MPBenchmark(eqx.Module):
#     material: Material
#     output: Tuple[str]

#     config: IPConfig = eqx.field(static=True)

#     L_control_stack: chex.Array
#     stress_control_stack: chex.Array
#     stress_mask_indices: chex.Array

#     phi_ref: jnp.float32
#     stress_ref: chex.Array
#     F_ref: chex.Array

#     def __init__(
#         self: Self,
#         config: IPConfig,
#         material:Material,
#         L_control_stack: chex.Array,
#         stress_control_stack: chex.Array,
#         stress_mask_indices: chex.Array,
#         phi_ref: jnp.float32,
#         stress_ref: chex.Array,
#         F_ref: chex.Array = None,
#     ):
#         if F_ref is None:
#             F_ref = jnp.eye(3)

#         self.config = config
#         self.phi_ref = phi_ref
#         self.F_ref = F_ref
#         self.phi_ref = phi_ref
#         self.stress_ref = stress_ref

#         self.L_control_stack = L_control_stack

#         self.stress_control_stack = stress_control_stack

#         self.stress_mask_indices = stress_mask_indices

#         self.material = material

#     @classmethod
#     def create_volume_control_shear(
#         cls,
#         config,
#         material,
#         phi_ref,
#         stress_ref,
#         x_range=(0, 0.1),
#         y_range=(0.0, 0.1),
#         F_ref=None,
#         output=None,
#         early_stop_iter=None,
#         debug=False,
#     ):
#         """
#         Strain rate control
#         >>> [-y,x,0]
#         >>> [',-y,0]
#         >>> [',',-y]

#         Stress control
#         >>> [?,?,?]
#         >>> [',?,?]
#         >>> [',',?]
#         """
#         x_stack = jnp.linspace(x_range[0], x_range[1], config.num_steps)
#         y_stack = jnp.linspace(y_range[0], y_range[1], config.num_steps)

#         def get_L(x, y):
#             L = jnp.zeros((3, 3))
#             L = L.at[[0, 1], [1, 0]].set(x)
#             L = L.at[[0, 1, 2], [0, 1, 2]].set(-y)
#             return L

#         L_control_stack = jax.vmap(get_L)(x_stack, y_stack)

#         return cls(
#             material=material,
#             phi_ref=phi_ref,
#             stress_ref=stress_ref,
#             output=output,
#             L_control_stack=L_control_stack.at[:early_stop_iter].get(),
#             stress_control_stack=None,
#             stress_mask_indices=None,
#             F_ref=None,
#         )

# def run(self):
#     carry, accumulated = mix_control(
#         material=self.material,
#         dt=self.dt,
#         L_control_stack=self.L_control_stack,
#         stress_control_stack=self.stress_control_stack,
#         stress_mask_indices=self.stress_mask_indices,
#         stress_ref=self.stress_ref,
#         F_ref=self.F_ref,
#         phi_ref=self.phi_ref,
#         output=self.output,
#     )
#     (material_next, stress_next, F_next, phi_next, step, servo_params) = carry

#     accumulated_next = []

#     for i, _ in enumerate(self.output):
#         accumulated_next.append(accumulated[i].at[0 :: self.store_every].get())

#     return self.replace(
#         material=material_next,
#         stress_ref=stress_next,
#         F_ref=F_next,
#         phi_ref=phi_next,
#         accumulated=accumulated_next,
#     )

# def get_time_stack(self):
#     step_stack = jnp.arange(0, self.load_steps, self.store_every)
#     time_stack = (
#         jnp.linspace(0, self.total_time, self.load_steps).at[step_stack].get()
#     )
#     return time_stack

# @classmethod
# def create_pressure_control_shear(
#     cls,
#     material,
#     total_time=200,
#     dt=0.0001,
#     x_range=(0, 0.1),
#     y_range=(0.0, 10000),
#     store_every=50,
#     phi_ref=0.7,
#     F_ref=None,
#     stress_ref=None,
#     output=None,
#     early_stop_iter=None,
# ):
#     """
#     Strain rate control
#     >>> [?,x,0]
#     >>> [',?,0]
#     >>> [',',?]

#     Stress control
#     >>> [-y/3,?,?]
#     >>> [',-y/3,?]
#     >>> [',',-y/3]
#     """
#     if "stress_ref_stack" in material.__dict__:  # noqa
#         if stress_ref is not None:
#             material = material.replace(stress_ref=stress_ref.reshape(1, 3, 3))
#         else:
#             stress_ref = material.stress_ref_stack.reshape(3, 3)

#     if F_ref is not None:
#         F_ref = jnp.eye(3)

#     stress_mask = jnp.zeros((3, 3)).at[[0, 1, 2], [0, 1, 2]].set(1).astype(bool)

#     stress_mask_indices = jnp.where(stress_mask)

#     load_steps = jnp.int32(total_time / dt)

#     x_stack = jnp.linspace(x_range[0], x_range[1], load_steps)
#     y_stack = jnp.linspace(y_range[0], y_range[1], load_steps)

#     def get_L(x):
#         return jnp.zeros((3, 3)).at[[0, 1], [1, 0]].set(x)

#     L_control_stack = jax.vmap(get_L)(x_stack)

#     def get_stress(y):
#         # y is pressure
#         return -jnp.eye(3) * y / 3.0

#     stress_control_stack = jax.vmap(get_stress)(y_stack)

#     return cls(
#         material=material,
#         total_time=total_time,
#         load_steps=load_steps,
#         dt=dt,
#         phi_ref=phi_ref,
#         stress_ref=stress_ref,
#         store_every=store_every,
#         output=output,
#         L_control_stack=L_control_stack.at[:early_stop_iter].get(),
#         stress_control_stack=stress_control_stack,
#         stress_mask_indices=stress_mask_indices,
#         F_ref=None,
#         accumulated=None,
#     )
