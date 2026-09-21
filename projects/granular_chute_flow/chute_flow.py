"""Periodic, gravity-driven granular chute-flow benchmark.

Coordinates are aligned with the chute: ``x`` is streamwise and ``y`` is normal
to the base. The background grid is periodic in ``x``; the base is represented by
a frictional plane SDF and the top is open.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import hydraxmpm as hdx

MODEL_NAMES = {
    "mu_ic": "mu_IC",
    "mu_lc": "mu_LC",
    "mu_ic_regularized": "mu_IC_regularized",
    "mu_i_incompressible": "mu_IC",
    "drucker_prager": "drucker_prager",
}


@dataclass(frozen=True)
class ChuteParameters:
    periodic_x_min: float = 0.0
    periodic_x_max: float = 0.10
    base_y: float = 0.0
    domain_height: float = 0.15
    fill_depth: float = 0.10
    cell_size: float = 0.02
    ppc: int = 4

    dt: float = 2.0e-4
    total_time: float = 15.0
    output_time: float = 0.05

    grain_density: float = 2650.0
    initial_solid_fraction: float = 0.60
    bulk_modulus: float = 1.0e6
    friction_angle_deg: float = 20.0
    dynamic_friction_angle_deg: float = 30.0
    # calibrated to match the Bagnold reference solution
    mu_i_lc_alpha: float = 74.0
    mu_i_ic_viscosity_cfl: float = 0.002
    mu_i_regularization_rate: float = 0.1
    chute_angle_deg: float = 24.0
    lateral_stress_ratio: float = 0.5
    base_friction: float = 0.7
    separation_density_ratio: float = 0.90
    constitutive_model: str = "drucker_prager"
    solver_alpha: float = 0.1

    def __post_init__(self):
        try:
            canonical_name = MODEL_NAMES[self.constitutive_model.lower()]
        except KeyError as error:
            raise ValueError(
                f"Unknown constitutive model {self.constitutive_model!r}"
            ) from error
        object.__setattr__(self, "constitutive_model", canonical_name)
        for name in (
            "dt",
            "total_time",
            "output_time",
            "mu_i_lc_alpha",
            "mu_i_regularization_rate",
            "mu_i_ic_viscosity_cfl",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def origin(self) -> tuple[float, float]:
        return (self.periodic_x_min, self.base_y)

    @property
    def end(self) -> tuple[float, float]:
        return (self.periodic_x_max, self.base_y + self.domain_height)

    @property
    def bulk_density(self) -> float:
        return self.grain_density * self.initial_solid_fraction

    @property
    def particles_per_axis(self) -> int:
        value = int(round(np.sqrt(self.ppc)))
        if value * value != self.ppc:
            raise ValueError("ppc must be a perfect square in this 2D benchmark")
        return value

    @property
    def particle_spacing(self) -> float:
        return self.cell_size / self.particles_per_axis

    @property
    def gravity(self) -> jax.Array:
        theta = jnp.deg2rad(self.chute_angle_deg)
        return jnp.array(
            [9.81 * jnp.sin(theta), -9.81 * jnp.cos(theta)],
            dtype=jnp.float32,
        )

    @property
    def drucker_prager_mu(self) -> jax.Array:
        phi = jnp.deg2rad(self.friction_angle_deg)
        return 6.0 * jnp.sin(phi) / (jnp.sqrt(3.0) * (3.0 + jnp.sin(phi)))


class ChuteProcedure:
    def __init__(self, params: ChuteParameters | None = None):
        self.params = params or ChuteParameters()
        self.dt = self.params.dt
        self.total_steps = int(round(self.params.total_time / self.dt))
        self.output_steps = max(1, int(round(self.params.output_time / self.dt)))

    def generate_particles(self):
        params = self.params
        length = params.periodic_x_max - params.periodic_x_min
        n_cells_x = int(round(length / params.cell_size))
        n_cells_y = int(round(params.fill_depth / params.cell_size))
        if not np.isclose(n_cells_x * params.cell_size, length):
            raise ValueError("periodic length must be an integer number of cells")
        if not np.isclose(n_cells_y * params.cell_size, params.fill_depth):
            raise ValueError("fill depth must be an integer number of cells")

        spacing = params.particle_spacing
        stream = (
            params.periodic_x_min
            + (jnp.arange(n_cells_x * params.particles_per_axis) + 0.5) * spacing
        )
        normal = (
            params.base_y
            + (jnp.arange(n_cells_y * params.particles_per_axis) + 0.5) * spacing
        )
        xx, yy = jnp.meshgrid(stream, normal, indexing="xy")
        position = jnp.stack([xx.ravel(), yy.ravel()], axis=-1).astype(jnp.float32)
        velocity = jnp.zeros_like(position)
        density = jnp.full(position.shape[0], params.bulk_density, dtype=jnp.float32)
        return position, velocity, density

    def initialize_lithostatic_stress(self, position, density):
        """Return a compression-positive stress field in chute coordinates."""
        params = self.params
        depth_below_surface = jnp.maximum(
            params.base_y + params.fill_depth - position[:, 1], 0.0
        )
        theta = jnp.deg2rad(params.chute_angle_deg)
        sigma_yy = density * 9.81 * jnp.cos(theta) * depth_below_surface
        sigma_xy = -density * 9.81 * jnp.sin(theta) * depth_below_surface
        sigma_xx = params.lateral_stress_ratio * sigma_yy
        sigma_zz = params.lateral_stress_ratio * sigma_yy

        stress = jnp.zeros((position.shape[0], 3, 3), dtype=jnp.float32)
        stress = stress.at[:, 0, 0].set(sigma_xx)
        stress = stress.at[:, 1, 1].set(sigma_yy)
        stress = stress.at[:, 2, 2].set(sigma_zz)
        stress = stress.at[:, 0, 1].set(sigma_xy)
        stress = stress.at[:, 1, 0].set(sigma_xy)
        return stress

    def build_template(self):
        params = self.params
        position, velocity, density = self.generate_particles()
        stress = self.initialize_lithostatic_stress(position, density)
        pressure = jnp.trace(stress, axis1=1, axis2=2) / 3.0

        model_name = MODEL_NAMES[params.constitutive_model.lower()]
        if model_name == "mu_LC":
            law = hdx.MuI_LC(
                mu_s=jnp.tan(jnp.deg2rad(params.friction_angle_deg)),
                mu_d=jnp.tan(jnp.deg2rad(params.dynamic_friction_angle_deg)),
                I_0=0.35,
                d_p=0.002,
                K=params.bulk_modulus,
                rho_p=params.grain_density,
                alpha=params.mu_i_lc_alpha,
            )
            law_state = law.create_state_from_density(
                density_stack=density,
                pressure_stack=pressure,
            )
        elif model_name == "mu_IC":
            law = hdx.MuI_IC(
                mu_s=jnp.tan(jnp.deg2rad(params.friction_angle_deg)),
                mu_d=jnp.tan(jnp.deg2rad(params.dynamic_friction_angle_deg)),
                I_0=0.35,
                d_p=0.002,
                rho_p=params.grain_density,
                cell_size=params.cell_size,
                viscosity_cfl=params.mu_i_ic_viscosity_cfl,
            )
            law_state = law.create_state_from_pressure(
                pressure_stack=pressure,
            )
        elif model_name == "mu_IC_regularized":
            law = hdx.MuI_IC_regularized(
                mu_s=jnp.tan(jnp.deg2rad(params.friction_angle_deg)),
                mu_d=jnp.tan(jnp.deg2rad(params.dynamic_friction_angle_deg)),
                I_0=0.35,
                d_p=0.002,
                rho_p=params.grain_density,
                regularization_rate=params.mu_i_regularization_rate,
                cell_size=params.cell_size,
                viscosity_cfl=params.mu_i_ic_viscosity_cfl,
            )
            law_state = law.create_state_from_pressure(
                pressure_stack=pressure,
            )
        elif model_name == "drucker_prager":
            law = hdx.DruckerPrager(
                nu=0.3,
                K=params.bulk_modulus,
                mu_1=jnp.tan(jnp.deg2rad(params.friction_angle_deg)),
                rho_0=params.separation_density_ratio * params.bulk_density,
            )
            law_state = law.create_state(stress_stack=stress)
        else:
            raise ValueError(
                f"Unknown constitutive model {params.constitutive_model!r}; "
                "choose 'drucker_prager', 'mu_LC', 'mu_IC', "
                "or 'mu_IC_regularized'."
            )

        builder = hdx.SimBuilder()
        builder.add_material_points(
            position_stack=position,
            velocity_stack=velocity,
            density_stack=density,
            stress_stack=stress,
            cell_size=params.cell_size,
            ppc=params.ppc,
        )
        builder.add_grid(
            origin=params.origin,
            end=params.end,
            cell_size=params.cell_size,
            periodic_axes=(True, False),
        )
        builder.add_constitutive_law(law=law, law_state=law_state)
        builder.couple(shapefunction="quadratic")
        builder.add_gravity(gravity=params.gravity, is_apply_on_grid=True)

        base = hdx.PlaneSDF(normal=(0.0, 1.0))
        builder.add_sdf_object(
            sdf_logic=base,
            center_of_mass=jnp.array([0.0, params.base_y]),
        )
        builder.add_sdf_collider(
            gap=params.particle_spacing,
            friction=params.base_friction,
        )
        if model_name in ("mu_IC", "mu_IC_regularized"):
            builder.set_solver(
                scheme="usl_incompressible_aflip",
                alpha=params.solver_alpha,
            )
        else:
            builder.set_solver(scheme="usl_aflip", alpha=params.solver_alpha)

        solver, state = builder.build(dt=params.dt)
        self.validate_initial_state(solver, state)
        return solver, state

    def validate_initial_state(self, solver, state):
        params = self.params
        mp = state.world.material_points[0]
        expected_volume = params.cell_size**2 / params.ppc
        expected_mass = (
            params.bulk_density
            * (params.periodic_x_max - params.periodic_x_min)
            * params.fill_depth
        )
        actual_mass = float(jnp.sum(mp.mass_stack))

        if not np.isclose(float(mp.volume_stack[0]), expected_volume):
            raise ValueError("particle volume is inconsistent with cell_size and ppc")
        if not np.isclose(actual_mass, expected_mass, rtol=1.0e-6):
            raise ValueError(
                f"particle mass {actual_mass} does not match bed mass {expected_mass}"
            )
        if solver.grid_domains[0].periodic_axes != (True, False):
            raise ValueError("streamwise grid axis is not periodic")
        if not bool(jnp.all(jnp.isfinite(mp.stress_stack))):
            raise ValueError("initial stress contains non-finite values")

    def wrap_periodic_streamwise(self, state):
        mp_states = list(state.world.material_points)
        mp = mp_states[0]
        params = self.params
        period = params.periodic_x_max - params.periodic_x_min
        wrapped_x = (
            jnp.mod(mp.position_stack[:, 0] - params.periodic_x_min, period)
            + params.periodic_x_min
        )
        updated_mp = eqx.tree_at(
            lambda item: item.position_stack,
            mp,
            mp.position_stack.at[:, 0].set(wrapped_x),
        )
        mp_states[0] = updated_mp
        world = eqx.tree_at(
            lambda item: item.material_points, state.world, tuple(mp_states)
        )
        return eqx.tree_at(lambda item: item.world, state, world)


def _save_particle_state(output_dir: Path, step: int, time: float, mp) -> None:
    """Write one raw particle snapshot for independent postprocessing."""
    snapshot_dir = output_dir / "particle_states"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        snapshot_dir / f"step_{step:08d}.npz",
        step=np.asarray(step),
        time=np.asarray(time),
        position=np.asarray(mp.position_stack),
        velocity=np.asarray(mp.velocity_stack),
        mass=np.asarray(mp.mass_stack),
        density=np.asarray(mp.density_stack),
        stress=np.asarray(mp.stress_stack),
    )


def run_sim(
    params: ChuteParameters | None = None,
    output_dir: Path | None = None,
    write_visuals: bool = True,
):
    procedure = ChuteProcedure(params)
    params = procedure.params
    solver, state = procedure.build_template()

    def make_advance_chunk(chunk_length):
        def advance_chunk(current_state):
            def step_once(carry, _):
                next_state = solver(carry)
                next_state = procedure.wrap_periodic_streamwise(next_state)
                return next_state, None

            return jax.lax.scan(
                step_once,
                current_state,
                xs=None,
                length=chunk_length,
            )[0]

        return eqx.filter_jit(advance_chunk)

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "output"
    else:
        output_dir = Path(__file__).resolve().parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "parameters.json").write_text(
        json.dumps(asdict(params), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    visualizer = (
        hdx.VTKVisualizer(output_dir=str(output_dir)) if write_visuals else None
    )

    history = {"time": [], "mean_speed": [], "mean_vx": []}

    print("Starting granular chute-flow benchmark")
    print(
        f"model={params.constitutive_model}, particles="
        f"{state.world.material_points[0].num_points}, dt={params.dt}, "
        f"steps={procedure.total_steps}"
    )

    full_chunks, remainder = divmod(procedure.total_steps, procedure.output_steps)
    chunk_lengths = [procedure.output_steps] * full_chunks
    if remainder:
        chunk_lengths.append(remainder)
    advance_functions = {
        length: make_advance_chunk(length) for length in set(chunk_lengths)
    }

    completed_steps = 0
    for chunk_length in chunk_lengths:
        state = advance_functions[chunk_length](state)
        completed_steps += chunk_length

        mp = state.world.material_points[0]
        arrays_finite = (
            jnp.all(jnp.isfinite(mp.position_stack))
            & jnp.all(jnp.isfinite(mp.velocity_stack))
            & jnp.all(jnp.isfinite(mp.stress_stack))
        )
        if not bool(arrays_finite):
            raise FloatingPointError(
                f"non-finite particle state at step {completed_steps}"
            )

        time = completed_steps * params.dt
        speed = float(jnp.mean(jnp.linalg.norm(mp.velocity_stack, axis=1)))
        mean_vx = float(jnp.mean(mp.velocity_stack[:, 0]))
        history["time"].append(time)
        history["mean_speed"].append(speed)
        history["mean_vx"].append(mean_vx)

        _save_particle_state(output_dir, completed_steps, time, mp)

        if write_visuals:
            visualizer.log_particles(
                mp,
                label="material_points",
                property_name="velocity_stack",
                time=time,
                step=completed_steps,
            )

        min_y = float(jnp.min(mp.position_stack[:, 1]))
        max_y = float(jnp.max(mp.position_stack[:, 1]))
        print(
            f"step={completed_steps:05d}, t={time:.4f}, "
            f"mean_vx={mean_vx:.4e}, mean_speed={speed:.4e}, "
            f"y=[{min_y:.4f}, {max_y:.4f}]",
            flush=True,
        )

    mp = state.world.material_points[0]
    print(f"Final particle count: {mp.num_points}")
    print(f"Saved outputs to {output_dir}")
    return state, history


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=lambda name: MODEL_NAMES.get(name.lower(), name),
        choices=tuple(dict.fromkeys(MODEL_NAMES.values())),
        default=ChuteParameters.constitutive_model,
    )
    parser.add_argument("--total-time", type=float, default=ChuteParameters.total_time)
    parser.add_argument(
        "--mu-i-lc-alpha", type=float, default=ChuteParameters.mu_i_lc_alpha
    )
    parser.add_argument(
        "--mu-i-regularization-rate",
        type=float,
        default=ChuteParameters.mu_i_regularization_rate,
    )
    parser.add_argument(
        "--mu-i-ic-viscosity-cfl",
        type=float,
        default=ChuteParameters.mu_i_ic_viscosity_cfl,
        help="Shared viscosity CFL coefficient for both incompressible models.",
    )
    parser.add_argument(
        "--output-time", type=float, default=ChuteParameters.output_time
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-visuals", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cli_params = replace(
        ChuteParameters(),
        constitutive_model=args.model,
        mu_i_lc_alpha=args.mu_i_lc_alpha,
        mu_i_regularization_rate=args.mu_i_regularization_rate,
        mu_i_ic_viscosity_cfl=args.mu_i_ic_viscosity_cfl,
        total_time=args.total_time,
        output_time=args.output_time,
    )
    run_sim(
        params=cli_params,
        output_dir=args.output_dir,
        write_visuals=not args.no_visuals,
    )
