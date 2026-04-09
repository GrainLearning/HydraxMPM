def run_sim():
    import sys
    import os
    from functools import partial

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    import jax
    import hydraxmpm as hdx
    import jax.numpy as jnp

    import equinox as eqx

    class ColumnParameters:
        fric_angle = 19.8  # [degrees]
        rho_0 = 2650.0  # [kg/m^3]
        rho_p = 2650.0  # [kg/m^3] assuming no compaction for simplicity
        K = 7e5  # [Pa] stiffness parameter

        def compute_mu(self):
            "matching Mohr-Coulomb criterion under triaxial extension conditions"
            fric_angle_rad = jnp.deg2rad(self.fric_angle)
            mu = (
                6
                * jnp.sin(fric_angle_rad)
                / (jnp.sqrt(3) * (3 + jnp.sin(jnp.deg2rad(self.fric_angle))))
            )
            return mu

    class CollapseProcedure:

        origin: tuple[float, float] = (0.0, 0.0)  # [m]
        end: tuple[float, float] = (0.6, 0.11)  # [m]
        cell_size: float = 0.0025  # [m]

        column_width: float = 0.2  # [m]
        column_height: float = 0.1  # [m]

        ppc: int = 2
        dt: float = 1e-5

        total_time: float = 2.0
        output_time: float = 0.05

        gap = 0.0025 

        def build_template(self):

            default_params = ColumnParameters()

            sep = self.cell_size / self.ppc
            x = jnp.arange(0, self.column_width, sep) + 2 * sep
            y = jnp.arange(0, self.column_height, sep) + 2 * sep
            xv, yv = jnp.meshgrid(x, y)

            position_stack = jnp.array(list(zip(xv.flatten(), yv.flatten())))

            # law = hdx.ModifiedCamClay(
            #     nu=0.3, M=0.9, lam=0.2,
            #     kap=0.005, N=1.0,
            #     rho_p= default_params.rho_p,
            # )


            num_particles = len(position_stack)

            # law_state, stress_ref_stack, density_stack = law.create_state_from_ocr(
            #     p_stack=jnp.ones(num_particles) * 1000.0,  # kPa
            #     ocr_stack=jnp.ones(num_particles) * 1.0,
            # )

            density_stack = jnp.ones(num_particles) * default_params.rho_0

            law = hdx.DruckerPrager(
                nu= 0.3,
                K=default_params.K,
                mu_1=default_params.compute_mu(),
                rho_0=default_params.rho_0,
            )
            law_state = law.create_state(stress_stack=jnp.zeros((num_particles, 3, 3)))

            sim_builder = hdx.SimBuilder()

            self.mp_id = sim_builder.add_material_points(
                position_stack=position_stack,
                density_stack=density_stack,
                cell_size=self.cell_size,
                ppc=self.ppc,
            )

            self.grid_id = sim_builder.add_grid(
                origin=self.origin,
                end=self.end,
                cell_size=self.cell_size,
            )
            self.law_id = sim_builder.add_constitutive_law(law=law, law_state=law_state)

            self.body_id = sim_builder.couple(
                shapefunction="quadratic",
            )

            self.gravity_f_id = sim_builder.add_gravity(
                gravity=jnp.array([0.0, -9.81]), is_apply_on_grid=True
            )

            domain_sdf = hdx.DomainSDF(
                origin=self.origin,
                end=self.end,
                frictions=0.9,
                wall_offset=0.75 * self.cell_size,
            )

            self.domain_sdf_id = sim_builder.add_sdf_object(sdf_logic=domain_sdf)

            self.domain_collider_f_id = sim_builder.add_sdf_collider(
                gap=sep,
            )

            self.solver_id = sim_builder.set_solver(scheme="usl_aflip", alpha=0.90)
            mpm_solver, sim_state = sim_builder.build(dt=self.dt)
            return mpm_solver, sim_state

        def run(self, solver, state, call_back=None):
            steps = int(self.total_time / self.dt)
            output_step = int(self.output_time / self.dt)

            def loop_body(i, val_state):

                next_state = solver(val_state)

                if call_back is not None:
                    jax.lax.cond(
                        i % output_step == 0,
                        lambda s: jax.debug.callback(call_back, s),
                        lambda s: None,
                        next_state,
                    )
                return next_state

            final_state = jax.lax.fori_loop(0, steps, loop_body, state)
            return final_state

    collapse_parameters = ColumnParameters()

    collapse_procedure = CollapseProcedure()

    mpm_solver, sim_state = collapse_procedure.build_template()

    viewer = hdx.RerunVisualizer(is_3d=False)

    viewer.log_static_domain(
        origin=collapse_procedure.origin,
        end=collapse_procedure.end,
        cell_size=collapse_procedure.cell_size,
    )

    def log_simulation(sim_state):

        mp_state = sim_state.world.material_points[0]
        step = sim_state.step
        time = sim_state.time

        viewer.log_time(current_step=int(step), current_time=float(time))
        viewer.log_material_points(
            mp_state, v_min=0, v_max=0.5, property_name="velocity_stack"
        )

    collapse_procedure.run(
        mpm_solver,
        sim_state,
        log_simulation,
    )


import multiprocessing

if __name__ == "__main__":

    p = multiprocessing.Process(target=run_sim, args=())
    p.start()
    try:
        p.join()
    except KeyboardInterrupt:

        # Force JAX to release GPU memory by terminating the process
        p.terminate()

        p.join()

        print("Worker dead. GPU memory released.")

    if p.exitcode == 0:
        print("Simulation finished naturally.")
    else:
        print(f"Process ended with code {p.exitcode}")
