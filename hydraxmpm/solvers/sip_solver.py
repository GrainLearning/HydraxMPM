# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

"""
Single Integration Point (SIP) Solver for Constitutive Model Testing
======================================================================

This module provides the  single integration point class (SIPSolver) and functions for running element tests and verifying constitutive models under various loading conditions.

It is designed to share the same API as the Material Point Method (MPM)solver for flexible, robust, and automated testing of constitutive laws.

The SIP benchmarks are defined elsewhere in the `hydraxmpm.sip_benchmarks` module.

Key Features:
- Supports stress/strain control, mixed control, and servo control for SIP element tests.
- Integrates with user-defined constitutive laws and SIP benchmark protocols.
- Handles both direct and root-finding-based control strategies.
- Designed for batch and automated testing of constitutive models.

Usage:
- Instantiate SIPSolver with a constitutive law, material points, and SIP benchmarks.
- Call `setup()` to initialize states.
- Call `run(dt)` to execute the test sequence.
"""

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


def stress_update(
    L_next, material_points_prev, constitutive_law_prev, dt, is_linear_approx=False
):
    """
    Perform a stress update for a single material point.

    Advances the velocity gradient and deformation gradient, then updates the stress
    using the constitutive law.
    """
    L_next_padded = jnp.expand_dims(L_next, axis=0)
    material_points_next = material_points_prev.update_L_and_F_stack(
        L_stack_next=L_next_padded, dt=dt, is_linear_approx=is_linear_approx
    )
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
    is_linear_approx=False,
):
    """
    Apply a single control step (stress, strain, or mixed) for SIP element tests.

    If the benchmark requires root-finding (mixed/servo control), uses a Newton solver.
    Otherwise, applies the control directly.
    """
    def servo_controller(sol, return_aux=False):
        L_next = L_control.at[et_benchmark.L_unknown_indices].set(sol)
        material_points_next, constitutive_law_next = stress_update(
            L_next,
            material_points_prev,
            constitutive_law_prev,
            dt,
            is_linear_approx=is_linear_approx,
        )
        stress_guess = jnp.squeeze(material_points_next.stress_stack, axis=0)
        R = et_benchmark.loss_stress(stress_guess, x_target)
        if return_aux:
            return R, (material_points_next, constitutive_law_next)
        return R

    if et_benchmark.X_control_stack is None:
        # Direct control (no root-finding)
        material_points_next, constitutive_law_next = stress_update(
            L_control,
            material_points_prev,
            constitutive_law_prev,
            dt,
            is_linear_approx=is_linear_approx,
        )
    else:
        # Mixed/servo control (root-finding)
        def find_roots():
            newton = optx.Newton(rtol=rtol, atol=atol)
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
    """
    Scan through a control sequence for a SIP benchmark (stress/strain path).

    Handles both direct and mixed/servo control protocols, accumulating outputs at each step.
    """
    if et_benchmark.X_control_stack is not None:
        X_0 = jnp.zeros_like(et_benchmark.X_control_stack.at[0].get())
    else:
        X_0 = None

    def scan_fn(carry, xs):
        constitutive_law_prev, material_points_prev, step = carry
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
            is_linear_approx=solver.is_linear_approx,
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
        (constitutive_law, material_points, 0),
        (et_benchmark.L_control_stack, et_benchmark.X_control_stack),
    )
    return carry, accumulate


class SIPSolver(Base):
    """
    Single Integration Point (SIP) Solver for Constitutive Model Testing.

    Applies element test loading conditions (stress, strain, or mixed control)
    to one or more constitutive law points. Supports batch and automated testing
    of constitutive models under various protocols.

    Attributes:
        material_points: MaterialPoints instance.
        constitutive_law: ConstitutiveLaw instance.
        sip_benchmarks: Tuple of SIPBenchmark instances.
        rtol: Relative tolerance for root-finding.
        atol: Absolute tolerance for root-finding.
        max_steps: Maximum root-finding steps.
        is_linear_approx: Use linearized update if True.
        output_vars: Output variable names or dict for result extraction.
    """
    material_points: MaterialPoints
    constitutive_law: ConstitutiveLaw
    sip_benchmarks: Tuple[SIPBenchmark, ...]
    _setup_done: bool = eqx.field(default=False)
    rtol: float = eqx.field(default=1e-10, static=True)
    atol: float = eqx.field(default=1e-2, static=True)
    max_steps: int = eqx.field(default=20, static=True)
    is_linear_approx: bool = eqx.field(default=False, static=True)
    output_vars: Dict | Tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        constitutive_law: ConstitutiveLaw,
        material_points: MaterialPoints = None,
        rtol: float = 1e-10,
        atol: float = 1e-2,
        max_steps: int = 20,
        sip_benchmarks: Optional[Tuple[SIPBenchmark, ...] | SIPBenchmark] = None,
        output_vars: Optional[dict | Tuple[str, ...]] = None,
        is_linear_approx=False,
        **kwargs,
    ) -> Self:
        """
        Initialize the SIPSolver.
        """
        self.output_vars = output_vars
        self.constitutive_law = constitutive_law
        self.rtol = rtol
        self.atol = atol
        self.max_steps = max_steps
        self.is_linear_approx = is_linear_approx
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
        """
        Initialize deformation modes and constitutive law states for all SIP benchmarks.
        Should be called once after initialization.
        """
        if self._setup_done:
            return self
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
        """
        Run the SIP solver for all benchmarks, accumulating outputs.
        """
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
        """
        Extract output variables from material points and constitutive law.
        Supports both array and callable outputs, and handles dynamic extraction
        based on the output_vars attribute.
        """
        accumulate = []
        for key in self.output_vars:
            output = None
            # Try extracting from material_points
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
            # Try extracting from constitutive_law
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
