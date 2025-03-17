import inspect
from functools import partial
from typing import Optional, Self, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
import equinox as eqx

from ..common.base import Base
from ..constitutive_laws.constitutive_law import ConstitutiveLaw
from ..et_benchmarks.et_benchmarks import ETBenchmark
from ..material_points.material_points import MaterialPoints
from .config import Config

from ..common.types import TypeInt


def stress_update(L_next, material_points_prev, constitutive_law_prev, dt):
    """Call the update function"""
    L_next_padded = jnp.expand_dims(L_next, axis=0)
    material_points_next = material_points_prev.update_L_and_F_stack(
        L_stack_next=L_next_padded, dt=dt
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
    rtol=1e-10,
    atol=1e-2,
    max_steps=20,
):
    def servo_controller(sol, args):
        L_next = L_control.at[et_benchmark.L_unknown_indices].set(sol)

        material_points_next, constitutive_law_next = stress_update(
            L_next, material_points_prev, constitutive_law_prev, dt
        )

        stress_guess = jnp.squeeze(material_points_next.stress_stack, axis=0)

        R = et_benchmark.loss_stress(stress_guess, x_target)

        return R, (material_points_next, constitutive_law_next)

    if et_benchmark.X_control_stack is None:
        material_points_next, constitutive_law_next = stress_update(
            L_control, material_points_prev, constitutive_law_prev, dt
        )
    else:
        solver = optx.Newton(rtol=rtol, atol=atol)

        sol = optx.root_find(
            servo_controller,
            solver,
            X_0,
            throw=False,
            has_aux=True,
            max_steps=max_steps,
        )

        material_points_next, constitutive_law_next = sol.aux

    return material_points_next, constitutive_law_next


class ETSolver(Base):
    """Element Test (ET) Solver. Applies element test loading conditions
    on one (or many) constitutive_law points."""

    config: Config = eqx.field(static=True)
    material_points: MaterialPoints

    constitutive_law: ConstitutiveLaw

    et_benchmarks: Tuple[ETBenchmark, ...]

    _setup_done: bool = eqx.field(default=False)

    rtol: float = eqx.field(default=1e-10, static=True)
    atol: float = eqx.field(default=1e-2, static=True)
    max_steps: int = eqx.field(default=20, static=True)
    # step: TypeInt

    def __init__(
        self,
        config: Config,
        constitutive_law: ConstitutiveLaw,
        material_points: MaterialPoints = None,
        rtol: float = 1e-10,
        atol: float = 1e-2,
        max_steps: int = 20,
        et_benchmarks: Optional[Tuple[ETBenchmark, ...] | ETBenchmark] = None,
        step: TypeInt = 0,
        **kwargs,
    ) -> Self:
        self.config = config
        self.constitutive_law = constitutive_law

        self.rtol = rtol
        self.atol = atol
        self.max_steps = max_steps
        # self.step = step

        if material_points is None:
            material_points = MaterialPoints()

        self.material_points = material_points

        self.et_benchmarks = (
            et_benchmarks
            if isinstance(et_benchmarks, tuple)
            else (et_benchmarks,)
            if et_benchmarks
            else ()
        )
        super().__init__(**kwargs)

    def setup(self: Self, **kwargs) -> Self:
        # we run this once after initialization
        if self._setup_done:
            return self

        # setup deformation modes
        new_et_benchmarks = []
        new_material_points = self.material_points
        for benchmark in self.et_benchmarks:
            new_deformation_mode, new_material_points = benchmark.init_state(
                self.config, new_material_points
            )
            new_et_benchmarks.append(new_deformation_mode)

        new_et_benchmarks = tuple(new_et_benchmarks)

        new_constitutive_law, new_material_points = self.constitutive_law.init_state(
            new_material_points
        )

        params = self.__dict__

        params.update(
            et_benchmarks=new_et_benchmarks,
            material_points=new_material_points,
            constitutive_law=new_constitutive_law,
            _setup_done=True,
        )

        return self.__class__(**params)

    def run_once_update(self: Self):
        assert len(self.et_benchmarks) == 1, "Only one deformation mode is supported"

        et_benchmark = self.et_benchmarks[0]

        if et_benchmark.X_control_stack is not None:
            X_0 = jnp.zeros_like(et_benchmark.X_control_stack.at[0].get())
            x_target = et_benchmark.X_control_stack.at[self.step].get()
        else:
            X_0 = None
            x_target = None

        material_points_next, constitutive_law_next = apply_control(
            et_benchmark.L_control_stack.at[self.step].get(),
            self.material_points,
            self.constitutive_law,
            et_benchmark,
            self.config.dt,
            X_0,
            x_target,
            self.rtol,
            self.atol,
            self.max_steps,
        )

        accumulate = self.get_output(
            material_points_next,
            constitutive_law_next,
            self.material_points,
            self.constitutive_law,
        )

        new_self = eqx.tree_at(
            lambda state: (state.material_points, state.constitutive_law, state.step),
            self,
            (material_points_next, constitutive_law_next, self.step + 1),
        )

        return new_self, accumulate

    @jax.jit
    def run_jit(self: Self, return_carry: bool = False):
        return self.run(return_carry)

    def run(self: Self, return_carry: bool = False):
        accumulate_list = []
        carry_list = []

        for et_benchmark in self.et_benchmarks:
            carry, accumulate = self.mix_control(
                self.material_points, self.constitutive_law, et_benchmark
            )
            accumulate_list.append(accumulate)
            carry_list.append(carry)

        # ensure that the list of output arrays is "flat"
        # with respect to number of deformation mode
        if len(self.et_benchmarks) == 1:
            accumulate_list = accumulate_list[0]
            carry_list = carry_list[0]

        if return_carry:
            return carry_list, accumulate_list
        return accumulate_list

    def mix_control(
        self: Self,
        material_points: MaterialPoints,
        constitutive_law: ConstitutiveLaw,
        et_benchmark: ETBenchmark,
    ):
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

            L_control, x_target = xs

            material_points_next, constitutive_law_next = apply_control(
                L_control,
                material_points_prev,
                constitutive_law_prev,
                et_benchmark,
                self.config.dt,
                X_0,
                x_target,
                self.rtol,
                self.atol,
                self.max_steps,
            )

            accumulate = self.get_output(
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

    def get_output(
        self,
        material_points,
        constitutive_law,
        material_points_prev=None,
        constitutive_law_prev=None,
    ):
        accumulate = []
        for key in self.config.output:
            output = None
            # workaround around
            # properties of one class depend on properties of another
            for name, member in inspect.getmembers(material_points):
                if key == name:
                    output = material_points.__getattribute__(key)

                    if callable(output):
                        output = output(
                            dt=self.config.dt,
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
