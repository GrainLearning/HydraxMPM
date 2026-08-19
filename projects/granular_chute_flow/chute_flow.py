"""Granular chute-flow benchmark using periodic streamwise boundaries.

The streamwise direction is periodic: particles exiting one side re-enter on the opposite side,
while thevertical direction remains bounded by a domain wall and gravity drives the flow.
"""

from __future__ import annotations

import os
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import hydraxmpm as hdx


class ChuteParameters:
    # Slim computational box while keeping a small margin around periodic cell.
    origin = (-0.03, 0.0)
    end = (0.07, 0.2)
    cell_size = 0.02
    ppc = 4
    dt = 2.0e-4
    total_time = 1.0
    output_time = 0.01
    rho_0 = 2650.0
    K = 1.0e8
    friction_angle = 20.0
    chute_angle_deg = 24.0
    periodic_x_min = 0.0
    periodic_x_max = 0.02
    fill_depth = 0.2
    constitutive_model = "drucker_prager"  # or "mu_i"

    @staticmethod
    def compute_mu():
        theta = jnp.deg2rad(ChuteParameters.friction_angle)
        mu = 6.0 * jnp.sin(theta) / (jnp.sqrt(3.0) * (3.0 + jnp.sin(theta)))
        return mu


class ChuteProcedure:
    def __init__(self):
        self.params = ChuteParameters()
        self.origin = self.params.origin
        self.end = self.params.end
        self.cell_size = self.params.cell_size
        self.ppc = self.params.ppc
        self.dt = self.params.dt
        self.total_steps = int(self.params.total_time / self.dt)
        self.output_steps = int(self.params.output_time / self.dt)

    def generate_particles(self):
        sep = self.cell_size / self.ppc
        stream = jnp.arange(
            self.params.periodic_x_min + sep,
            self.params.periodic_x_max,
            sep,
        )
        normal = jnp.arange(
            self.origin[1] + sep,
            self.origin[1] + self.params.fill_depth + sep,
            sep,
        )
        ss, nn = jnp.meshgrid(stream, normal)
        pos = jnp.stack([ss.ravel(), nn.ravel()], axis=-1).astype(jnp.float32)
        vel = jnp.zeros_like(pos)
        density = jnp.full(pos.shape[0], self.params.rho_0)
        return pos, vel, density

    def initialize_lithostatic_stress(self, pos, density):
        """Initialize a realistic confining stress field for a tilted chute.

        The soil is initially at rest in a gravity field that is tilted by the chute
        angle. We compute the vertical stress using the depth measured normal to the
        chute base and rotate the principal stress tensor into the global frame.
        """
        gravity_mag = jnp.linalg.norm(
            jnp.array([
                9.81 * jnp.sin(jnp.deg2rad(self.params.chute_angle_deg)),
                -9.81 * jnp.cos(jnp.deg2rad(self.params.chute_angle_deg)),
            ], dtype=jnp.float32)
        )
        y_depth = pos[:, 1] - self.origin[1]
        p_stack, q_stack = hdx.precondition_from_lithostatic(
            density_stack=density,
            depth_stack=y_depth,
            gravity=gravity_mag,
            slope_angle_deg=self.params.chute_angle_deg,
            k0=0.5,
        )
        stress_local = hdx.reconstruct_stress_from_triaxial(p_stack, q_stack)

        theta = jnp.deg2rad(self.params.chute_angle_deg)
        rot = jnp.array(
            [
                [jnp.cos(theta), -jnp.sin(theta), 0.0],
                [jnp.sin(theta),  jnp.cos(theta), 0.0],
                [0.0,            0.0,            1.0],
            ],
            dtype=jnp.float32,
        )
        stress_world = jax.vmap(lambda s: rot @ s @ rot.T)(stress_local)
        return stress_world

    def build_template(self):
        pos, vel, density = self.generate_particles()
        stress_stack = self.initialize_lithostatic_stress(pos, density)

        model_name = self.params.constitutive_model.lower()
        if model_name == "mu_i":
            law = hdx.MuI_LC(
                mu_s=self.params.compute_mu(),
                mu_d=self.params.compute_mu() * 1.5,
                I_0=0.35,
                d_p=0.002,
                K=self.params.K,
                rho_p=self.params.rho_0,
                alpha=1.0e-6,
            )
            law_state = law.create_state_from_density(density_stack=density)
        elif model_name == "drucker_prager":
            law = hdx.DruckerPrager(
                nu=0.3,
                K=self.params.K,
                mu_1=self.params.compute_mu(),
                rho_0=self.params.rho_0,
            )
            law_state = law.create_state(stress_stack=jnp.zeros((pos.shape[0], 3, 3)))
        else:
            raise ValueError(
                "Unknown constitutive model: "
                f"{self.params.constitutive_model}. Choose 'drucker_prager' or 'mu_i'."
            )

        sim_builder = hdx.SimBuilder()

        sim_builder.add_material_points(
            position_stack=pos,
            velocity_stack=vel,
            density_stack=density,
            stress_stack=stress_stack,
            cell_size=self.cell_size,
            ppc=self.ppc,
        )
        sim_builder.add_grid(
            origin=self.origin,
            end=self.end,
            cell_size=self.cell_size,
        )
        sim_builder.add_constitutive_law(law=law, law_state=law_state)
        sim_builder.couple(shapefunction="quadratic")
        theta = jnp.deg2rad(self.params.chute_angle_deg)
        gravity_world = jnp.array([
            9.81 * jnp.sin(theta),
            -9.81 * jnp.cos(theta),
        ], dtype=jnp.float32)
        sim_builder.add_gravity(gravity=gravity_world, is_apply_on_grid=True)

        # For the periodic streamwise direction, the left/right walls should not
        # behave like rigid contact walls. Only the normal-direction wall and the
        # rough lower boundary carry friction
        # Intended physical setup for the periodic chute:
        # - left/right walls are periodic
        # - bottom wall is a no-slip or rough base
        # - top is free
        domain_sdf = hdx.DomainSDF(
            origin=self.origin,
            end=self.end,
            frictions=[0.0, 0.7, 0.0, 0.0],
            wall_offset=0.75 * self.cell_size,
        )
        sim_builder.add_sdf_object(sdf_logic=domain_sdf)
        sim_builder.add_sdf_collider(gap=self.cell_size / self.ppc)
        sim_builder.set_solver(scheme="usl_aflip", alpha=0.9)

        return sim_builder.build(dt=self.dt)

    def wrap_periodic_streamwise(self, sim_state):
        world = sim_state.world
        mp_states = list(world.material_points)
        mp_state = mp_states[0]

        pos = mp_state.position_stack
        vel = mp_state.velocity_stack

        s_min = self.params.periodic_x_min
        s_max = self.params.periodic_x_max
        s_wrapped = jnp.mod(pos[:, 0] - s_min, s_max - s_min) + s_min

        n_min = self.origin[1]
        n_max = self.end[1]
        n_clipped = jnp.clip(pos[:, 1], n_min, n_max)

        world_pos = pos.at[:, 0].set(s_wrapped).at[:, 1].set(n_clipped)
        # Keep velocity continuous across periodic x faces.
        world_vel = vel

        updated_mp = eqx.tree_at(
            lambda s: (s.position_stack, s.velocity_stack),
            mp_state,
            (world_pos, world_vel),
        )

        mp_states[0] = updated_mp
        world = eqx.tree_at(lambda w: w.material_points, world, tuple(mp_states))
        return eqx.tree_at(lambda s: s.world, sim_state, world)

    def compute_velocity_profile(self, sim_state):
        mp_state = sim_state.world.material_points[0]
        pos = np.asarray(mp_state.position_stack)
        vel = np.asarray(mp_state.velocity_stack)

        y = pos[:, 1]
        vx = vel[:, 0]
        y_min = self.origin[1]
        y_max = self.end[1]
        bins = 20
        edges = np.linspace(y_min, y_max, bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        counts, _ = np.histogram(y, bins=edges)
        weighted, _ = np.histogram(y, bins=edges, weights=vx)
        avg_vx = np.divide(weighted, counts, out=np.zeros_like(weighted, dtype=float), where=counts > 0)
        return centers, avg_vx

    def compute_time_averaged_profile(self, profiles):
        if not profiles:
            return np.array([]), np.array([])
        y = np.asarray(profiles[0][:, 0], dtype=float)
        stacked_vx = np.stack([np.asarray(p[:, 1], dtype=float) for p in profiles], axis=0)
        avg_vx = np.mean(stacked_vx, axis=0)
        return y, avg_vx


def run_sim():
    procedure = ChuteProcedure()
    solver, sim_state = procedure.build_template()

    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    vis = hdx.VTKVisualizer(output_dir=str(output_dir))
    history = {"time": [], "mean_speed": [], "profile_y": [], "profile_vx": []}
    profile_history = []

    print("Starting granular chute flow benchmark")
    print(f"Domain: {procedure.origin} -> {procedure.end}, dt={procedure.dt}, steps={procedure.total_steps}")

    for step in range(procedure.total_steps):
        sim_state = solver(sim_state)
        sim_state = procedure.wrap_periodic_streamwise(sim_state)

        if step % procedure.output_steps == 0:
            mp_state = sim_state.world.material_points[0]
            mean_vel = jnp.mean(jnp.linalg.norm(mp_state.velocity_stack, axis=1))
            history["time"].append(step * procedure.dt)
            history["mean_speed"].append(float(mean_vel))
            vis.log_particles(
                mp_state,
                label="material_points",
                property_name="velocity_stack",
                time=float(step * procedure.dt),
                step=int(step),
            )

            pos = np.asarray(mp_state.position_stack)
            vel_local = np.asarray(mp_state.velocity_stack)

            plt.figure(figsize=(7, 3.5))
            plt.quiver(
                pos[:, 0], pos[:, 1],
                vel_local[:, 0], vel_local[:, 1],
                np.linalg.norm(vel_local, axis=1),
                cmap="viridis",
                scale=30,
                width=0.0035,
            )
            plt.xlim(procedure.params.periodic_x_min, procedure.params.periodic_x_max)
            plt.ylim(procedure.origin[1], procedure.end[1])
            plt.xlabel("periodic streamwise coordinate x")
            plt.ylabel("vertical coordinate y")
            plt.colorbar(label="speed")
            plt.tight_layout()
            plt.savefig(output_dir / f"snapshot_{step:05d}.png", dpi=180)
            plt.close()

            profile_y, profile_vx = procedure.compute_velocity_profile(sim_state)
            history["profile_y"].append(profile_y)
            history["profile_vx"].append(profile_vx)
            profile_history.append(np.column_stack([profile_y, profile_vx]))

            np.savetxt(
                output_dir / f"velocity_profile_{step:05d}.csv",
                np.column_stack([profile_y, profile_vx]),
                delimiter=",",
                header="y,mean_vx",
                comments="",
            )

            print(f"step={step:04d}, mean_speed={float(mean_vel):.4e}")

    final_mp = sim_state.world.material_points[0]
    final_mean = float(jnp.mean(jnp.linalg.norm(final_mp.velocity_stack, axis=1)))
    print("Final particle count:", final_mp.position_stack.shape[0])
    print("Final mean speed:", final_mean)

    times = history["time"]
    speeds = history["mean_speed"]
    plt.figure(figsize=(6, 4))
    plt.plot(times, speeds, linewidth=2)
    plt.xlabel("time [s]")
    plt.ylabel("mean speed [m/s]")
    plt.tight_layout()
    plt.savefig(output_dir / "mean_speed.png", dpi=180)
    plt.close()

    with open(output_dir / "mean_speed.csv", "w", encoding="utf-8") as f:
        f.write("time,mean_speed\n")
        for t, v in zip(times, speeds):
            f.write(f"{t},{v}\n")

    final_pos = np.asarray(final_mp.position_stack)
    vel_local = np.asarray(final_mp.velocity_stack)

    plt.figure(figsize=(7, 3.5))
    plt.quiver(
        final_pos[:, 0], final_pos[:, 1],
        vel_local[:, 0], vel_local[:, 1],
        np.linalg.norm(vel_local, axis=1),
        cmap="viridis",
        scale=30,
        width=0.0035,
    )
    plt.xlim(procedure.params.periodic_x_min, procedure.params.periodic_x_max)
    plt.ylim(procedure.origin[1], procedure.end[1])
    plt.xlabel("periodic streamwise coordinate x")
    plt.ylabel("vertical coordinate y")
    plt.colorbar(label="speed")
    plt.tight_layout()
    plt.savefig(output_dir / "final_snapshot.png", dpi=180)
    plt.close()

    final_profile_y, final_profile_vx = procedure.compute_velocity_profile(sim_state)
    np.savetxt(
        output_dir / "final_velocity_profile.csv",
        np.column_stack([final_profile_y, final_profile_vx]),
        delimiter=",",
        header="y,mean_vx",
        comments="",
    )

    steady_window = max(1, min(20, len(profile_history)))
    steady_profiles = profile_history[-steady_window:]
    avg_y, avg_vx = procedure.compute_time_averaged_profile(steady_profiles)
    if avg_y.size > 0:
        np.savetxt(
            output_dir / "steady_velocity_profile.csv",
            np.column_stack([avg_y, avg_vx]),
            delimiter=",",
            header="y,mean_vx",
            comments="",
        )
        plt.figure(figsize=(6, 4))
        plt.plot(avg_vx, avg_y, linewidth=2)
        plt.gca().invert_yaxis()
        plt.xlabel("mean streamwise velocity $v_x$")
        plt.ylabel("depth y")
        plt.tight_layout()
        plt.savefig(output_dir / "steady_velocity_profile.png", dpi=180)
        plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(final_profile_vx, final_profile_y, linewidth=2)
    plt.gca().invert_yaxis()
    plt.xlabel("mean streamwise velocity $v_x$")
    plt.ylabel("depth y")
    plt.tight_layout()
    plt.savefig(output_dir / "final_velocity_profile.png", dpi=180)
    plt.close()

    print(f"Saved visualization outputs to {output_dir}")


if __name__ == "__main__":
    run_sim()
