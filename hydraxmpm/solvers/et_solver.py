import inspect
from typing import Optional, Self, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx

from ..common.base import Base
from ..constitutive_laws.constitutive_law import ConstitutiveLaw
from ..et_benchmarks.et_benchmarks import ETBenchmark
from ..material_points.material_points import MaterialPoints
from .config import Config


class ETSolver(Base):
    """Element Test (ET) Solver. Applies element test loading conditions
    on one (or many) constitutive_law points."""

    config: Config = eqx.field(static=True)
    material_points: MaterialPoints
    constitutive_law: ConstitutiveLaw
    et_benchmarks: Tuple[ETBenchmark, ...]

    _setup_done: bool = eqx.field(default=False)

    def __init__(
        self,
        config: Config,
        constitutive_law: ConstitutiveLaw,
        material_points: MaterialPoints = None,
        et_benchmarks: Optional[Tuple[ETBenchmark, ...] | ETBenchmark] = None,
        **kwargs,
    ) -> Self:
        self.config = config
        self.constitutive_law = constitutive_law

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
            # To do setup materials

        constitutive_law, material_points = self.constitutive_law.init_state(
            self.material_points
        )
        # setup deformation modes
        new_et_benchmarks = []

        for deformation_mode in self.et_benchmarks:
            new_deformation_mode = deformation_mode.init_steps(
                num_steps=self.config.num_steps
            )
            new_et_benchmarks.append(new_deformation_mode)
        et_benchmarks = tuple(new_et_benchmarks)

        params = self.__dict__

        params.update(
            et_benchmarks=et_benchmarks,
            material_points=material_points,
            constitutive_law=constitutive_law,
            _setup_done=True,
        )

        return self.__class__(**params)

    # @jax.jit
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
        servo_params = None

        if et_benchmark.stress_mask_indices is not None:
            servo_params = jnp.zeros((3, 3)).at[et_benchmark.stress_mask_indices].get()

        def scan_fn(carry, control):
            (
                constitutive_law_prev,
                material_points_prev,
                step,
                servo_params,
            ) = carry

            L_control, stress_control = control
            if et_benchmark.stress_mask_indices is not None:
                stress_control_target = stress_control.at[
                    et_benchmark.stress_mask_indices
                ].get()

            def update_from_params(L_next):
                L_next_padded = jnp.expand_dims(L_next, axis=0)
                material_points_next = material_points_prev.update_L_and_F_stack(
                    L_stack_next=L_next_padded,
                    dt=self.config.dt,
                )
                # jnp.expand_dims(L_next, axis=0
                material_points_next, constitutive_law_next = (
                    constitutive_law_prev.update(
                        material_points=material_points_next, dt=self.config.dt
                    )
                )
                return material_points_next, constitutive_law_next

            def servo_controller(sol, args):
                L_next = L_control.at[et_benchmark.stress_mask_indices].set(sol)

                material_points_next, constitutive_law_next = update_from_params(L_next)

                stress_next = jnp.squeeze(material_points_next.stress_stack, axis=0)
                stress_guess = stress_next.at[et_benchmark.stress_mask_indices].get()

                R = stress_guess - stress_control_target
                return R, (material_points_next, constitutive_law_next)

            if et_benchmark.stress_mask_indices is None:
                material_points_next, constitutive_law_next = update_from_params(
                    L_control
                )

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

                material_points_next, constitutive_law_next = sol.aux

            carry = (
                constitutive_law_next,
                material_points_next,
                step + 1,
                servo_params,
            )

            accumulate = []
            for key in list(self.config.output):
                output = None
                for name, member in inspect.getmembers(material_points_next):
                    if key == name:
                        output = material_points_next.__getattribute__(key)
                for name, member in inspect.getmembers(constitutive_law_next):
                    if key == name:
                        output = constitutive_law_next.__getattribute__(key)

                # workaround around
                # properties of one class depend on properties of another
                if key == "phi_stack":
                    output = material_points_next.phi_stack(constitutive_law_next.rho_p)
                elif key == "specific_volume_stack":
                    output = material_points_next.specific_volume_stack(
                        constitutive_law_next.rho_p
                    )
                elif key == "inertial_number_stack":
                    output = material_points_next.inertial_number_stack(
                        constitutive_law_next.rho_p,
                        constitutive_law_next.d,
                    )

                if eqx.is_array(output):
                    output = output.squeeze(axis=0)

                if output is None:
                    raise ValueError(f" {key} output not is not supported")
                accumulate.append(output)
            return carry, accumulate

        return jax.lax.scan(
            scan_fn,
            (
                constitutive_law,
                material_points,
                0,
                servo_params,
            ),  # carry
            (
                et_benchmark.L_control_stack,
                et_benchmark.stress_control_stack,
            ),  # control
        )
