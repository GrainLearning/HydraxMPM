import sys
import os

import jax
import hydraxmpm as hdx
import jax.numpy as jnp

import equinox as eqx
from floor import SpectralSetup, BumpyFloorSDF


class CollapseParams(eqx.Module):
    bumps: SpectralSetup
    frictions: SpectralSetup
    tilt_angle: float = 0.0  # degrees

    sim_id: int = 0

    def get_gravity_vector(self):
        angle_rad = jnp.deg2rad(self.tilt_angle)
        gravity = hdx.rotation_2d(angle_rad, jnp.array([0.0, -9.81]))
        return gravity


class CollapseSetup:

    origin: tuple[float, float] = (0.0, -0.21)  # [m]
    end: tuple[float, float] = (2.0, 0.41)  # [m]
    # cell_size: float = 0.0025  # [m] # high res
    cell_size: float = 0.005

    column_width: float = 0.2  # [m]
    column_height: float = 0.2  # [m]

    ppc: int = 2
    dt: float = 1e-5

    total_time: float = 3.0
    output_time: float = 0.05

    gap = 0.0025

    law = "drucker_prager"

    # Material parameters
    fric_angle = 25.0  # [degrees]
    rho_0 = 2650.0  # [kg/m^3]
    rho_p = 2650.0  # [kg/m^3] assuming no compaction for simplicity
    K = 7e5  # [Pa] stiffness parameter

    # output
    viewer = None
    vtk_output = None

    def build_template(self, sim_params=None):

        if sim_params is None:
            default_bumps = SpectralSetup()
            default_frictions = SpectralSetup()
            sim_params = CollapseParams(bumps=default_bumps, frictions=default_frictions )

        sim_builder = hdx.SimBuilder()

        ###########################################################################
        # Create material points
        ###########################################################################

        sep = self.cell_size / self.ppc
        x = jnp.arange(0, self.column_width, sep) + 2 * sep
        y = jnp.arange(0, self.column_height, sep) + 2 * sep
        xv, yv = jnp.meshgrid(x, y)

        position_stack = jnp.array(list(zip(xv.flatten(), yv.flatten())))

        num_particles = len(position_stack)

        density_stack = jnp.ones(num_particles) * self.rho_0

        y_surface = self.column_height

        depths = jnp.maximum(y_surface - position_stack[:, 1], 0.05)

        p_stack, q_stack = hdx.precondition_from_lithostatic(
                density_stack=density_stack,
                depth_stack=depths,
                gravity=9.81,
                slope_angle_deg=sim_params.tilt_angle,
                k0=0.5
        )
        stress_stack = hdx.reconstruct_stress_from_triaxial(p_stack, q_stack)
        
        self.mp_id = sim_builder.add_material_points(
            position_stack=position_stack,
            density_stack=density_stack,
            stress_stack=stress_stack,
            cell_size=self.cell_size,
            ppc=self.ppc,
        )

        ###########################################################################
        # Create grid
        ###########################################################################

        self.grid_id = sim_builder.add_grid(
            origin=self.origin,
            end=self.end,
            cell_size=self.cell_size,
        )

        ###########################################################################
        # Law
        ###########################################################################

        fric_angle_rad = jnp.deg2rad(self.fric_angle)
        mu_s = (
            6
            * jnp.sin(fric_angle_rad)
            / (jnp.sqrt(3) * (3 + jnp.sin(jnp.deg2rad(self.fric_angle))))
        )

        if self.law == "mu_i":
            law = hdx.MuI_LC(
                K=self.K, mu_s=mu_s, mu_d=mu_s * 2, I_0=1e-2, d_p=0.025, alpha=1e-4
            )

            self.law_id = sim_builder.add_constitutive_law(
                law=law, density_stack=density_stack
            )

        elif self.law == "drucker_prager":

            law = hdx.DruckerPrager(
                nu=0.3,
                K=self.K,
                mu_1=mu_s,
                rho_0=self.rho_0,
            )
            law_state = law.create_state(stress_stack=stress_stack)
            self.law_id = sim_builder.add_constitutive_law(law=law, law_state=law_state)

        ###########################################################################
        # Couple
        ###########################################################################

        self.body_id = sim_builder.couple(
            shapefunction="quadratic",
        )

        ###########################################################################
        # Forces
        ###########################################################################

        # Gravity
        self.gravity_f_id = sim_builder.add_gravity(
            gravity=sim_params.get_gravity_vector(), is_apply_on_grid=True
        )

        # domain
        domain_sdf = hdx.DomainSDF(
            origin=self.origin,
            end=self.end,
            frictions=0.5,
            wall_offset=0.75 * self.cell_size,
        )
        self.domain_sdf_id = sim_builder.add_sdf_object(sdf_logic=domain_sdf)

        self.domain_collider_f_id = sim_builder.add_sdf_collider(
            gap=sep,
        )

        # bumpy floor
        bumpy_floor_sdf = BumpyFloorSDF(
            x_min=self.origin[0] + self.column_width,
            x_max=self.end[0],
            x_transition_width=0.5,
        )
        bumpy_floor_sdf_state = bumpy_floor_sdf.create_state(
            bumps=sim_params.bumps, frictions=sim_params.frictions
        )

        self.bumpy_floor_sdf_idx = sim_builder.add_sdf_object(
            sdf_logic=bumpy_floor_sdf, sdf_state=bumpy_floor_sdf_state
        )
        self.domain_collider_collider_id = sim_builder.add_sdf_collider(
            gap=sep,
        )

        ###########################################################################
        # Set solver
        ###########################################################################
        self.solver_id = sim_builder.set_solver(scheme="usl_aflip", alpha=0.90)
        mpm_solver, sim_state = sim_builder.build(dt=self.dt)

        ###########################################################################
        # Create viewer
        ###########################################################################

        sim_builder.summary(dt=self.dt)

        return mpm_solver, sim_state

    def run(self, solver, state, params=None):

        if params is not None:
            bumpy_floor = eqx.tree_at(
                lambda p: (p.bumps, p.frictions),
                state.world.sdfs[self.bumpy_floor_sdf_idx],
                (params.bumps, params.frictions),
            )
            state = eqx.tree_at(
                lambda s: s.world.sdfs[self.bumpy_floor_sdf_idx],
                state,
                bumpy_floor,
            )

        steps = int(self.total_time / self.dt)
        output_step = int(self.output_time / self.dt)

        sim_id = params.sim_id if params is not None else None

        def loop_body(i, val_state):

            next_state = solver(val_state)

            jax.lax.cond(
                i % output_step == 0,
                lambda s: jax.debug.callback(
                    self.log_simulation,
                    s,
                    sim_id,
                ),
                lambda s: None,
                next_state,
            )
            return next_state

        final_state = jax.lax.fori_loop(0, steps, loop_body, state)
        return final_state

    def setup_viewer(self, solver, state, batch_configs=None, N=1, grid_columns=1):
        import rerun as rr
        import rerun.blueprint as rrb

        self.viewer = hdx.RerunVisualizer(is_3d=False)

        if batch_configs is None:
            return

        batch_configs_list = [
            jax.tree_util.tree_map(lambda x: x[i], batch_configs) for i in range(N)
        ]

        view_list = []

        for i, params in enumerate(batch_configs_list):
            sim_id = params.sim_id

            view = rrb.Spatial2DView(
                name=f"Simulation view {i}", origin=f"Sim_A/sim_{sim_id}/"
            )
            view_list.append(view)

        blueprint = rrb.Grid(*view_list, grid_columns=grid_columns)
        rr.send_blueprint(blueprint, make_active=True)

        for i, params in enumerate(batch_configs_list):
            sim_id = params.sim_id
            print(f"Setting up viewer for sim_id: {sim_id}")

            bumpy_floor = eqx.tree_at(
                lambda p: (p.bumps, p.frictions),
                state.world.sdfs[self.bumpy_floor_sdf_idx],
                (params.bumps, params.frictions)
            )
            state = eqx.tree_at(
                lambda s: s.world.sdfs[self.bumpy_floor_sdf_idx],
                state,
                bumpy_floor,
            )
            self.log_static(solver, state, sim_id)

    def setup_vtk(
        self,
        relative_dir,
        output_dir,
        solver,
        state,
        batch_configs=None,
        N=1,
    ):
        self.vtk_output = hdx.VTKVisualizer(
            relative_dir=relative_dir, output_dir=output_dir
        )

        if batch_configs is None:
            return

        batch_configs_list = [
            jax.tree_util.tree_map(lambda x: x[i], batch_configs) for i in range(N)
        ]

        for i, params in enumerate(batch_configs_list):
            sim_id = params.sim_id
            bumpy_floor = eqx.tree_at(
                lambda p: (p.bumps, p.frictions),
                state.world.sdfs[self.bumpy_floor_sdf_idx],
                (params.bumps, params.frictions),
            )
            state = eqx.tree_at(
                lambda s: s.world.sdfs[self.bumpy_floor_sdf_idx],
                state,
                bumpy_floor,
            )
            self.log_static(solver, state, sim_id)
            self.log_simulation(state, sim_id)

    def log_static(self, solver, state, sim_id):

        if self.viewer is not None:
            self.viewer.log_static_domain(
                origin=self.origin,
                end=self.end,
                cell_size=self.cell_size,
                label=f"sim_{sim_id}/domain",
            )

            self.viewer.log_sdf(
                sdf_logic=solver.sdf_logics[self.bumpy_floor_sdf_idx],
                sdf_state=state.world.sdfs[self.bumpy_floor_sdf_idx],
                start=self.origin,
                end=self.end,
                static=True,
                resolution=1000,
                label=f"sim_{sim_id}/floor",
            )
        if self.vtk_output is not None:
            self.vtk_output.log_static_domain(
                origin=self.origin,
                end=self.end,
                cell_size=self.cell_size,
                label=f"sim_{sim_id}/domain",
            )
            self.vtk_output.log_sdf(
                sdf_logic=solver.sdf_logics[self.bumpy_floor_sdf_idx],
                sdf_state=state.world.sdfs[self.bumpy_floor_sdf_idx],
                start=self.origin,
                end=self.end,
                resolution=1000,
                label=f"sim_{sim_id}/floor",
                step=0,
                time=0.0,
            )

        if (self.vtk_output is not None) or (self.viewer is not None):
            jax.debug.print("Logged static domain for sim_id: {sim_id}", sim_id=sim_id)

    def log_simulation(self, sim_state, sim_id=None):


        sim_id = int(sim_id)
        mp_state = sim_state.world.material_points[0]

        step = sim_state.step
        time = sim_state.time

        if self.viewer is not None:
            self.viewer.log_time(current_step=int(step), current_time=float(time))

            self.viewer.log_material_points(
                mp_state,
                v_min=0,
                v_max=0.5,
                label=f"sim_{sim_id}/particles",
                property_name="velocity_stack",
                scale_radius=2
            )
        if self.vtk_output is not None:
            self.vtk_output.log_particles(
                mp_state,
                label=f"sim_{sim_id}/particles",
                property_name="pressure_stack",
                time=sim_state.time,
                step=sim_state.step,
            )

        if (self.vtk_output is not None) or (self.viewer is not None):
            jax.debug.print("log simulation ID: {sim_id}", sim_id=sim_id)

    def run_with_trajectory(self, solver, state, params=None):

        if params is not None:
            bumpy_floor = eqx.tree_at(
                lambda p: (p.bumps, p.frictions),
                state.world.sdfs[self.bumpy_floor_sdf_idx],
                (params.bumps, params.frictions),
            )
            gravity = eqx.tree_at(
                lambda f: f.gravity,
                state.mechanics.forces[self.gravity_f_id],
                params.get_gravity_vector(),
            )

            state = eqx.tree_at(
                lambda s: 
                (
                    s.world.sdfs[self.bumpy_floor_sdf_idx],
                    s.mechanics.forces[self.gravity_f_id],
                ),
                state,
                (bumpy_floor, gravity)
            )

        num_steps = int(self.total_time / self.dt)
        output_step = int(self.output_time / self.dt)
        num_snapshots = num_steps // output_step

        def inner_step(carry_state, _):
            def body(i, s):
                return solver(s)

            next_state = jax.lax.fori_loop(0, output_step, body, carry_state)

            jax.debug.callback(self.log_simulation, next_state, params.sim_id)

            jax.debug.print(
                "Snapshot for sim_id: {sim_id} at time {time}",
                sim_id=params.sim_id,
                time=next_state.time,
            )

            return next_state, next_state

        final_state, trajectory = jax.lax.scan(
            inner_step, state, xs=None, length=num_snapshots
        )

        return trajectory
