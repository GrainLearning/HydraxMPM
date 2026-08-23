"""Periodic, gravity-driven granular chute-flow benchmark.

Coordinates are aligned with the chute: ``x`` is streamwise and ``y`` is normal
to the base. The background grid is periodic in ``x``; the base is represented by
a frictional plane SDF and the top is open.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import hydraxmpm as hdx


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
    total_time: float = 5.0
    output_time: float = 0.05

    grain_density: float = 2650.0
    initial_solid_fraction: float = 0.60
    bulk_modulus: float = 1.0e6
    friction_angle_deg: float = 20.0
    dynamic_friction_angle_deg: float = 30.0
    # Calibrated for the 24 degree Bagnold benchmark with 0.005 m cells.
    mu_i_max_shear_viscosity: float = 6.65
    mu_i_regularization_rate: float = 31.5
    chute_angle_deg: float = 24.0
    lateral_stress_ratio: float = 0.5
    base_friction: float = 0.7
    separation_density_ratio: float = 0.90

    constitutive_model: str = "drucker_prager"
    incompressible_wall_ghost_no_slip: bool = True
    steady_start_fraction: float = 0.8

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
        stream = params.periodic_x_min + (
            jnp.arange(n_cells_x * params.particles_per_axis) + 0.5
        ) * spacing
        normal = params.base_y + (
            jnp.arange(n_cells_y * params.particles_per_axis) + 0.5
        ) * spacing
        xx, yy = jnp.meshgrid(stream, normal, indexing="xy")
        position = jnp.stack([xx.ravel(), yy.ravel()], axis=-1).astype(jnp.float32)
        velocity = jnp.zeros_like(position)
        density = jnp.full(
            position.shape[0], params.bulk_density, dtype=jnp.float32
        )
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

        model_name = params.constitutive_model.lower()
        if model_name == "mu_i":
            law = hdx.MuI_LC(
                mu_s=jnp.tan(jnp.deg2rad(params.friction_angle_deg)),
                mu_d=jnp.tan(jnp.deg2rad(params.dynamic_friction_angle_deg)),
                I_0=0.35,
                d_p=0.002,
                K=params.bulk_modulus,
                rho_p=params.grain_density,
                alpha=1.0e-6,
            )
            law_state = law.create_state_from_density(
                density_stack=density,
                pressure_stack=pressure,
            )
        elif model_name == "mu_i_incompressible":
            law = hdx.MuI_Incompressible(
                mu_s=jnp.tan(jnp.deg2rad(params.friction_angle_deg)),
                mu_d=jnp.tan(jnp.deg2rad(params.dynamic_friction_angle_deg)),
                I_0=0.35,
                d_p=0.002,
                rho_p=params.grain_density,
                max_shear_viscosity=params.mu_i_max_shear_viscosity,
                cell_size=params.cell_size,
            )
            law_state = law.create_state_from_pressure(
                pressure_stack=pressure,
            )
        elif model_name == "mu_i_regularized":
            law = hdx.MuI_regularized(
                mu_s=jnp.tan(jnp.deg2rad(params.friction_angle_deg)),
                mu_d=jnp.tan(jnp.deg2rad(params.dynamic_friction_angle_deg)),
                I_0=0.35,
                d_p=0.002,
                rho_p=params.grain_density,
                regularization_rate=params.mu_i_regularization_rate,
                cell_size=params.cell_size,
            )
            law_state = law.create_state_from_pressure(
                pressure_stack=pressure,
            )
        elif model_name == "drucker_prager":
            law = hdx.DruckerPrager(
                nu=0.3,
                K=params.bulk_modulus,
                mu_1=params.drucker_prager_mu,
                rho_0=params.separation_density_ratio * params.bulk_density,
            )
            law_state = law.create_state(stress_stack=stress)
        else:
            raise ValueError(
                f"Unknown constitutive model {params.constitutive_model!r}; "
                "choose 'drucker_prager', 'mu_i', 'mu_i_incompressible', "
                "or 'mu_i_regularized'."
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
        if model_name in ("mu_i_incompressible", "mu_i_regularized"):
            builder.set_solver(
                scheme="usl_incompressible_aflip",
                alpha=0.1,
                wall_ghost_no_slip=params.incompressible_wall_ghost_no_slip,
            )
        else:
            builder.set_solver(scheme="usl_aflip", alpha=0.1)

        solver, state = builder.build(dt=params.dt)
        self.validate_initial_state(solver, state)
        return solver, state

    def validate_initial_state(self, solver, state):
        params = self.params
        mp = state.world.material_points[0]
        expected_volume = params.cell_size**2 / params.ppc
        expected_mass = params.bulk_density * (
            params.periodic_x_max - params.periodic_x_min
        ) * params.fill_depth
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

    def compute_velocity_profile(self, state):
        mp = state.world.material_points[0]
        position = np.asarray(mp.position_stack)
        velocity = np.asarray(mp.velocity_stack)
        n_bins = int(round(self.params.domain_height / self.params.particle_spacing))
        edges = np.linspace(
            self.params.base_y,
            self.params.base_y + self.params.domain_height,
            n_bins + 1,
        )
        centers = 0.5 * (edges[:-1] + edges[1:])
        counts, _ = np.histogram(position[:, 1], bins=edges)
        weighted, _ = np.histogram(position[:, 1], bins=edges, weights=velocity[:, 0])
        mean_vx = np.full(centers.shape, np.nan)
        np.divide(weighted, counts, out=mean_vx, where=counts > 0)
        return centers, mean_vx, counts

    @staticmethod
    def compute_time_averaged_profile(profiles):
        if not profiles:
            return np.array([]), np.array([])
        y = np.asarray(profiles[0][:, 0], dtype=float)
        stacked_vx = np.stack(
            [np.asarray(profile[:, 1], dtype=float) for profile in profiles], axis=0
        )
        valid_counts = np.sum(np.isfinite(stacked_vx), axis=0)
        summed = np.nansum(stacked_vx, axis=0)
        mean_vx = np.full(y.shape, np.nan)
        np.divide(summed, valid_counts, out=mean_vx, where=valid_counts > 0)
        return y, mean_vx


def _save_profile(path: Path, y, velocity, counts=None):
    columns = [y, velocity]
    header = "y,mean_vx"
    if counts is not None:
        columns.append(counts)
        header += ",particle_count"
    np.savetxt(
        path,
        np.column_stack(columns),
        delimiter=",",
        header=header,
        comments="",
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
    visualizer = (
        hdx.VTKVisualizer(output_dir=str(output_dir)) if write_visuals else None
    )

    history = {"time": [], "mean_speed": [], "mean_vx": []}
    steady_profiles = []

    print("Starting granular chute-flow benchmark")
    print(
        f"model={params.constitutive_model}, particles="
        f"{state.world.material_points[0].num_points}, dt={params.dt}, "
        f"steps={procedure.total_steps}"
    )

    full_chunks, remainder = divmod(
        procedure.total_steps, procedure.output_steps
    )
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

        time = float(state.time)
        speed = float(jnp.mean(jnp.linalg.norm(mp.velocity_stack, axis=1)))
        mean_vx = float(jnp.mean(mp.velocity_stack[:, 0]))
        history["time"].append(time)
        history["mean_speed"].append(speed)
        history["mean_vx"].append(mean_vx)

        profile_y, profile_vx, counts = procedure.compute_velocity_profile(state)
        profile = np.column_stack([profile_y, profile_vx])
        if time >= params.steady_start_fraction * params.total_time:
            steady_profiles.append(profile)

        _save_profile(
            output_dir / f"velocity_profile_{completed_steps:05d}.csv",
            profile_y,
            profile_vx,
            counts,
        )

        if write_visuals:
            visualizer.log_particles(
                mp,
                label="material_points",
                property_name="velocity_stack",
                time=time,
                step=completed_steps,
            )
            position = np.asarray(mp.position_stack)
            velocity = np.asarray(mp.velocity_stack)
            plt.figure(figsize=(7, 3.5))
            plt.quiver(
                position[:, 0],
                position[:, 1],
                velocity[:, 0],
                velocity[:, 1],
                np.linalg.norm(velocity, axis=1),
                cmap="viridis",
                scale=30,
                width=0.0035,
            )
            plt.xlim(params.periodic_x_min, params.periodic_x_max)
            plt.ylim(params.base_y, params.base_y + params.domain_height)
            plt.xlabel("streamwise coordinate x [m]")
            plt.ylabel("height above base y [m]")
            plt.colorbar(label="speed [m/s]")
            plt.tight_layout()
            plt.savefig(output_dir / f"snapshot_{completed_steps:05d}.png", dpi=180)
            plt.close()

        min_y = float(jnp.min(mp.position_stack[:, 1]))
        max_y = float(jnp.max(mp.position_stack[:, 1]))
        print(
            f"step={completed_steps:05d}, t={time:.4f}, "
            f"mean_vx={mean_vx:.4e}, mean_speed={speed:.4e}, "
            f"y=[{min_y:.4f}, {max_y:.4f}]",
            flush=True,
        )

    mp = state.world.material_points[0]
    final_y, final_vx, final_counts = procedure.compute_velocity_profile(state)
    _save_profile(
        output_dir / "final_velocity_profile.csv",
        final_y,
        final_vx,
        final_counts,
    )

    avg_y, avg_vx = procedure.compute_time_averaged_profile(steady_profiles)
    if avg_y.size:
        _save_profile(output_dir / "steady_velocity_profile.csv", avg_y, avg_vx)

    np.savetxt(
        output_dir / "mean_velocity.csv",
        np.column_stack(
            [history["time"], history["mean_speed"], history["mean_vx"]]
        ),
        delimiter=",",
        header="time,mean_speed,mean_vx",
        comments="",
    )

    if write_visuals and avg_y.size:
        plt.figure(figsize=(6, 4))
        plt.plot(avg_vx, avg_y, linewidth=2)
        plt.xlabel("mean streamwise velocity [m/s]")
        plt.ylabel("height above base [m]")
        plt.tight_layout()
        plt.savefig(output_dir / "steady_velocity_profile.png", dpi=180)
        plt.close()

    tail_start = max(0, int(0.8 * len(history["mean_vx"])))
    tail_velocity = history["mean_vx"][tail_start:]
    if len(tail_velocity) >= 2:
        relative_tail_change = abs(tail_velocity[-1] - tail_velocity[0]) / max(
            abs(tail_velocity[-1]), 1.0e-12
        )
        history["relative_tail_change"] = relative_tail_change
        print(
            f"Relative mean-vx change over final 20%: "
            f"{relative_tail_change:.3%}"
        )

    print(f"Final particle count: {mp.num_points}")
    print(f"Saved outputs to {output_dir}")
    return state, history


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=(
            "drucker_prager",
            "mu_i",
            "mu_i_incompressible",
            "mu_i_regularized",
        ),
        default=ChuteParameters.constitutive_model,
    )
    parser.add_argument("--total-time", type=float, default=ChuteParameters.total_time)
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
        total_time=args.total_time,
        output_time=args.output_time,
    )
    run_sim(
        params=cli_params,
        output_dir=args.output_dir,
        write_visuals=not args.no_visuals,
    )
