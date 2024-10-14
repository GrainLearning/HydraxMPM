"""Base class for single integration point benchmark module"""

from functools import partial
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import optimistix as optx

from ..materials.material import Material
from ..utils.math_helpers import (
    get_phi_from_L,
)


@partial(jax.jit, static_argnames=("output"))
def mix_control(
    material: Material,
    dt: jnp.float32,
    L_control_stack: chex.Array,
    stress_control_stack: chex.Array = None,
    stress_mask_indices: chex.Array = None,
    stress_ref: chex.Array = None,
    F_ref: chex.Array = None,
    phi_ref: jnp.float32 = None,
    output: Tuple[str] = None,
) -> Material:
    chex.assert_shape(phi_ref, ())
    chex.assert_shape(stress_ref, (3, 3))
    chex.assert_shape(L_control_stack, (None, 3, 3))

    if F_ref is None:
        F_ref = jnp.eye(3)

    if stress_ref is None:
        stress_ref = jnp.zeros((3, 3))

    if phi_ref is None:
        phi_ref = 0.0

    if output is None:
        output = []

    servo_params = None

    if stress_mask_indices is not None:
        servo_params = jnp.zeros((3,3)).at[stress_mask_indices].get()


    def scan_fn(carry, control):
        (
            material_prev,
            stress_prev,
            F_prev,
            phi_prev,
            step,
            servo_params,
        ) = carry

        L_control, stress_control = control
        
        if stress_mask_indices is not None:
            stress_control_target = stress_control.at[stress_mask_indices].get()
        
        def update_from_params(L_next):


            F_next = (jnp.eye(3) + L_next * dt) @ F_prev

            phi_next = get_phi_from_L(L_next, phi_prev, dt)

            stress_next, material_next = material_prev.update(
                stress_prev.reshape(1, 3, 3),
                F_next.reshape(1, 3, 3),
                L_next.reshape(1, 3, 3),
                jnp.array([phi_next]),
                dt,
            )

            stress_next = stress_next.reshape((3, 3))

            aux = (F_next, L_next, phi_next, material_next)
            return stress_next, aux

        def servo_controller(sol, args):
            
            L_next = L_control.at[stress_mask_indices].set(sol)

            stress_next, aux = update_from_params(L_next)

            stress_guess = stress_next.at[stress_mask_indices].get()

            R = stress_guess - stress_control_target
            return R, (stress_next, *aux)

        if stress_mask_indices is None:
            stress_next, aux_next = update_from_params(L_control)

        else:
            params = servo_params
            solver = optx.Newton(rtol=1e-8, atol=1e-1)

            sol = optx.root_find(
                servo_controller,
                solver,
                params,
                throw=False,
                has_aux=True,
                max_steps=20,
            )
            
            stress_next, *aux_next = sol.aux

        F_next, L_next, phi_next, material_next = aux_next

        carry = (
            material_next,
            stress_next,
            F_next,
            phi_next,
            step + 1,
            servo_params,
        )

        accumulate = []

        for key in output:
            if key == "stress":
                accumulate.append(stress_next)
            elif key == "F":
                accumulate.append(F_next)
            elif key == "L":
                accumulate.append(L_next)
            elif key == "phi":
                accumulate.append(phi_next)
            elif key in material_next:
                accumulate.append(jnp.squeeze(material_next[key]))
        return carry, accumulate

    carry, accumulate = jax.lax.scan(
        scan_fn,
        (material, stress_ref, F_ref, phi_ref, 0, servo_params),  # carry
        (L_control_stack, stress_control_stack),  # control
    )

    return carry, accumulate
