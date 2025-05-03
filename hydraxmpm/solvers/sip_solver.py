# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-

import inspect

from typing import Optional, Self, Tuple, Dict


import jax
import jax.numpy as jnp
import optimistix as optx
import equinox as eqx

from ..common.base import Base
from ..constitutive_laws.constitutive_law import ConstitutiveLaw
from ..sip_benchmarks.sip_benchmarks import SIPBenchmark
from ..material_points.material_points import MaterialPoints


def stress_update(L_next, material_points_prev, constitutive_law_prev, dt):
    """Performs a stress update step for a single material point."""

    # Pad dimensions
    L_next_padded = jnp.expand_dims(L_next, axis=0)

    # Advance velocity gradient and deformation gradient over a timestep
    material_points_next = material_points_prev.update_L_and_F_stack(
        L_stack_next=L_next_padded, dt=dt
    )

    # Call the constitutive law to update the stress
    material_points_next, constitutive_law_next = constitutive_law_prev.update(
        material_points=material_points_next, dt=dt
    )
    return material_points_next, constitutive_law_next


def apply_control(
    L_control,
    material_points_prev,
    constitutive_law_prev,
    et_benchmark,
    dt,
    X_0=None,
    x_target=None,
    rtol=1e-3,
    atol=1e-2,
    max_steps=20,
):
    """Applies a single control step, solving for unknowns if needed (mixed control)."""

    def servo_controller(sol, return_aux=False):
        # sol = jnp.nan_to_num(sol)
        L_next = L_control.at[et_benchmark.L_unknown_indices].set(sol)

        material_points_next, constitutive_law_next = stress_update(
            L_next, material_points_prev, constitutive_law_prev, dt
        )

        stress_guess = jnp.squeeze(material_points_next.stress_stack, axis=0)

        R = et_benchmark.loss_stress(stress_guess, x_target)

        # R = jnp.nan_to_num(R)

        # jax.debug.print("mixed control: sol {} R {}",sol,R)

        # jax.debug.breakpoint()

        if return_aux:
            return R, (material_points_next, constitutive_law_next)
        return R

    if et_benchmark.X_control_stack is None:
        material_points_next, constitutive_law_next = stress_update(
            L_control, material_points_prev, constitutive_law_prev, dt
        )

    else:

        def find_roots():
            newton = optx.Newton(rtol=rtol, atol=atol)
            # jax.debug.print("X_0 {}",X_0)
            sol = optx.root_find(
                servo_controller,
                newton,
                X_0,
                args=False,
                throw=False,
                has_aux=False,
                options=dict(lower=-100, upper=100),
                max_steps=max_steps,
            )

            return sol.value

        X_root = find_roots()

        R, aux = servo_controller(X_root, True)
        material_points_next, constitutive_law_next = aux

    return material_points_next, constitutive_law_next


def mix_control(
    solver,
    material_points: MaterialPoints,
    constitutive_law: ConstitutiveLaw,
    et_benchmark: SIPBenchmark,
    dt,
    rtol,
    atol,
    max_steps,
):
    """Scans through the control sequence of a single SIP benchmark."""
    # TODO have X_target become new X_0? or X_guess become X_0?
    if et_benchmark.X_control_stack is not None:
        X_0 = jnp.zeros_like(et_benchmark.X_control_stack.at[0].get())
    else:
        X_0 = None

    def scan_fn(carry, xs):
        (
            constitutive_law_prev,
            material_points_prev,
            step,
        ) = carry
        # jax.debug.print("{} \r",step)
        L_control, x_target = xs

        material_points_next, constitutive_law_next = apply_control(
            L_control,
            material_points_prev,
            constitutive_law_prev,
            et_benchmark,
            dt,
            X_0,
            x_target,
            rtol,
            atol,
            max_steps,
        )

        accumulate = solver.get_output(
            dt,
            material_points_next,
            constitutive_law_next,
            material_points_prev,
            constitutive_law_prev,
        )

        carry = (constitutive_law_next, material_points_next, step + 1)

        return carry, accumulate

    carry, accumulate = jax.lax.scan(
        scan_fn,
        (constitutive_law, material_points, 0),  # carry
        (
            et_benchmark.L_control_stack,
            et_benchmark.X_control_stack,
        ),  # control
    )

    return carry, accumulate


class SIPSolver(Base):
    """Element Test (ET) Solver. Applies element test loading conditions
    on one (or many) constitutive_law points."""

    material_points: MaterialPoints

    constitutive_law: ConstitutiveLaw

    sip_benchmarks: Tuple[SIPBenchmark, ...]

    _setup_done: bool = eqx.field(default=False)

    rtol: float = eqx.field(default=1e-10, static=True)
    atol: float = eqx.field(default=1e-2, static=True)
    max_steps: int = eqx.field(default=20, static=True)
    output_vars: Dict | Tuple[str, ...] = eqx.field(static=True)  # run sim

    def __init__(
        self,
        constitutive_law: ConstitutiveLaw,
        material_points: MaterialPoints = None,
        rtol: float = 1e-10,
        atol: float = 1e-2,
        max_steps: int = 20,
        sip_benchmarks: Optional[Tuple[SIPBenchmark, ...] | SIPBenchmark] = None,
        output_vars: Optional[dict | Tuple[str, ...]] = None,
        **kwargs,
    ) -> Self:
        self.output_vars = output_vars

        self.constitutive_law = constitutive_law

        self.rtol = rtol
        self.atol = atol
        self.max_steps = max_steps

        if material_points is None:
            material_points = MaterialPoints()

        self.material_points = material_points

        self.sip_benchmarks = (
            sip_benchmarks
            if isinstance(sip_benchmarks, tuple)
            else (sip_benchmarks,)
            if sip_benchmarks
            else ()
        )
        super().__init__(**kwargs)

    def setup(self: Self, **kwargs) -> Self:
        # we run this once after initialization
        if self._setup_done:
            return self

        # setup deformation modes
        new_sip_benchmarks = []
        new_material_points = self.material_points
        for benchmark in self.sip_benchmarks:
            new_deformation_mode, new_material_points = benchmark.init_state(
                new_material_points
            )
            new_sip_benchmarks.append(new_deformation_mode)

        new_sip_benchmarks = tuple(new_sip_benchmarks)

        new_constitutive_law, new_material_points = self.constitutive_law.init_state(
            new_material_points
        )

        params = self.__dict__

        params.update(
            sip_benchmarks=new_sip_benchmarks,
            material_points=new_material_points,
            constitutive_law=new_constitutive_law,
            _setup_done=True,
        )

        return self.__class__(**params)

    @eqx.filter_jit
    def run(self, dt: float):
        accumulate_list = []

        carry_list = []

        for et_benchmark in self.sip_benchmarks:
            carry, accumulate = mix_control(
                self,
                self.material_points,
                self.constitutive_law,
                et_benchmark,
                dt,
                self.rtol,
                self.atol,
                self.max_steps,
            )
            accumulate_list.append(accumulate)
            carry_list.append(carry)

        # ensure that the list of output arrays is "flat"
        # with respect to number of deformation mode
        if len(self.sip_benchmarks) == 1:
            accumulate_list = accumulate_list[0]
            carry_list = carry_list[0]

        return accumulate_list

    def get_output(
        self,
        dt,
        material_points,
        constitutive_law,
        material_points_prev=None,
        constitutive_law_prev=None,
    ):
        accumulate = []
        for key in self.output_vars:
            output = None
            # workaround around
            # properties of one class depend on properties of another
            for name, member in inspect.getmembers(material_points):
                if key == name:
                    output = material_points.__getattribute__(key)

                    if callable(output):
                        output = output(
                            dt=dt,
                            rho_p=constitutive_law.rho_p,
                            d=constitutive_law.d,
                            eps_e_stack=constitutive_law.eps_e_stack,
                            eps_e_stack_prev=constitutive_law_prev.eps_e_stack,
                        )
            for name, member in inspect.getmembers(constitutive_law):
                if key == name:
                    output = constitutive_law.__getattribute__(key)
                    if callable(output):
                        continue

            if eqx.is_array(output):
                output = output.squeeze(axis=0)

            if output is None:
                raise ValueError(f" {key} output not is not supported")
            accumulate.append(output)
        return accumulate
