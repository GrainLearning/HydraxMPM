# """This example is coming back soon"""





def simulation_wrapper():

    import sys
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
    import jax


    import jax.numpy as jnp
    import numpy as np

    import hydraxmpm as hdx

    # from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    # import equinox as eqx

    # # Set this to the number of threads/cores you want to use as "devices"
    # os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    import equinox as eqx


    class HeartSDF(hdx.SDFObjectBase):
        """
        A Heart Shape.
        Based on Inigo Quilez's exact Heart SDF.
        
        Domain: 2D (Shape) or 3D (Infinite Column along Z)
        The 'tip' of the heart is at the local origin (0,0).
        """
        scale: float

        def __init__(self, scale=1.0):

            self.scale = scale

        def signed_distance_local(self, state, p_local):
            
    

            p = p_local / self.scale

            x = p[0]
            y = p[1]

            # 3. Apply Symmetry
            x = jnp.abs(x)


            dx_a = x - 0.25
            dy_a = y - 0.75
            dist_a = jnp.sqrt(dx_a**2 + dy_a**2) - jnp.sqrt(2.0)/4.0


            d1 = x**2 + (y - 1.0)**2
            

            w = 0.5 * jnp.maximum(x + y, 0.0)
            d2 = (x - w)**2 + (y - w)**2
            
            dist_b = jnp.sqrt(jnp.minimum(d1, d2)) * jnp.sign(x - y)

        
            dist_unit = jnp.where(
                (y + x) > 1.0,
                dist_a,
                dist_b
            )

            return dist_unit * self.scale
    # =========================================================
    # 1. SIMULATION PARAMETERS
    # =========================================================
    origin = jnp.array((0.0, 0.0))
    end = jnp.array((10.0, 10.0))
    domain_size = end - origin
    ppc = 2
    cell_size = (1 / 80.0) * 10  # 80 cells along each axis

    sim_builder = hdx.SimBuilder()

    dt = 0.5e-3
    total_steps = int(20/ dt)
    output_step = 0.01 / dt

    # =========================================================
    # 2. DEFINE Water BODY p_idx 0, g_idx 0
    # =========================================================
    # sdf_block = hdx.BoxSDF(size=(9.0, 4.0))
    sdf_block = hdx.BoxSDF(size=(10.0, 6.0))
    sdf = HeartSDF(scale=3.0)
    position_stack = hdx.generate_particles_in_sdf(
        sdf_obj=sdf_block,
        center_of_mass=jnp.array([5.0, 7.0]),
        # center_of_mass=jnp.array([5.5, 0.5]),
        bounds_min=origin,
        bounds_max=end,
        cell_size=cell_size,
        ppc=ppc,
        mode="regular",
    )

    density_stack = jnp.full((len(position_stack),), 1000.0   )

    water_p_idx = sim_builder.add_material_points(
        position_stack=position_stack,
        density_stack=density_stack,
        cell_size=cell_size,
        ppc=ppc
    )

    water_g_idx = sim_builder.add_grid(
        origin =origin,
        end =end,
        cell_size=cell_size,
    )

        # mu_s: float,
        # mu_d: float,
        # I_0: float,
        # d_p: float,
        # K: float = 1.0e6,
        # rho_p: float = 2650.0,
        # alpha: float = 1e-4,
        # p_min_calc: float = 10.0,

    water_c_id = sim_builder.add_constitutive_law(
         law = hdx.NewtonFluid(K=2e6, viscosity=1e-3, beta=7.0),
        # law = hdx.MuI_LC(
        #     K=2e6, mu_s=0.4, mu_d=1.41, I_0=1e-4,  d_p=0.025, alpha=1e-4
        # ),
         density_stack=density_stack
    )

    water_b_idx = sim_builder.couple(
        # p_idx=water_p_idx,
        # g_idx=water_g_idx,
        # c_idx=water_c_id,
        shapefunction="quadratic",
    )

    # =========================================================
    # clay
  

    # position_stack = hdx.generate_particles_in_sdf(
    #     sdf_obj=sdf_block,
    #      center_of_mass=jnp.array([5.0, 7.0]),
    #     # center_of_mass=jnp.array([5.5, 0.5]),
    #     bounds_min=origin,
    #     bounds_max=end,
    #     cell_size=cell_size,
    #     ppc=ppc,
    #     mode="regular",
    # )

    # p_stack = jnp.full((position_stack.shape[0],), 10_000.0)  # kPa

    # mcc = hdx.ModifiedCamClay(
    #     nu=0.3, M=0.9, lam=0.2, kap=0.05, N=2.0, p_ref=1000.0, K_min=100
    # )

    # mcc_state, stress_ref_stack,density_stack = mcc.create_state_from_ocr(
    #     p_stack=p_stack,
    #     ocr_stack =1.0
    # )

    # clay_p_idx = sim_builder.add_material_points(
    #     position_stack=position_stack,
    #     stress_stack=stress_ref_stack,
    #     density_stack=density_stack,
    #     cell_size=cell_size,
    #     ppc=ppc
    # )

    # clay_g_idx = sim_builder.add_grid(
    #     origin =origin,
    #     end =end,
    #     cell_size=cell_size,
    # )

    # clay_c_id = sim_builder.add_constitutive_law(
    #      law = mcc,
    #      law_state = mcc_state
    # )

    # clay_b_idx = sim_builder.couple(
    #     shapefunction="quadratic",
    # )

    # =========================================================

    # position_stack = hdx.generate_particles_in_sdf(
    #     sdf_obj=sdf_block,
    #     center_of_mass=jnp.array([8.0, 7.5]),
    #     # center_of_mass=jnp.array([5.5, 0.5]),
    #     bounds_min=origin,
    #     bounds_max=end,
    #     cell_size=cell_size,
    #     ppc=ppc,
    #     mode="regular",
    # )

    # density_stack = jnp.full((len(position_stack),), 2000.0)
    # el_p_idx = sim_builder.add_material_points(
    #     position_stack=position_stack,
    #     density_stack=density_stack,
    #     cell_size=cell_size,
    #     ppc=ppc,
    # )

    # el_idx = sim_builder.add_grid(
    #     origin=origin, end=end, cell_size=cell_size, padding=0
    # )
    # elastic_law = hdx.LinearElasticLaw(E=1e6, nu=0.3)
    # el_c_id = sim_builder.add_constitutive_law(law=elastic_law)

    # el_b_idx = sim_builder.couple(
    #     shapefunction="quadratic",
    # )

 
    # =========================================================
    # 2. DEFINE Water BODY p_idx 0, g_idx 0
    # =========================================================
    # sdf_block_rigid = hdx.BoxSDF(size=(1.0, 4.0))

    # position_stack = hdx.generate_particles_in_sdf(
    #     sdf_obj = sdf_block,
    #     center_of_mass=jnp.array([1.5, 2.5]),
    #     bounds_min=origin,
    #     bounds_max=end,
    #     cell_size=cell_size,
    #     ppc=ppc,
    #     mode="regular"
    # )

    # rigid_p_idx = sim_builder.add_material_points(
    #     position_stack=position_stack,
    #     is_rigid=True,
    # )

    # rigid_g_idx = sim_builder.add_grid(
    #     origin =origin,
    #     end =end,
    #     cell_size=cell_size,
    #     padding=0
    # )
    # rigid_b_idx = sim_builder.couple(
    #     # p_idx=water_p_idx,
    #     # g_idx=water_g_idx,
    #     # c_idx=water_c_id,
    #     shapefunction="quadratic",
    # )

    # =========================================================
    # 3. DEFINE FORCES
    # =========================================================

    grav_idx = sim_builder.add_gravity(
        gravity=jnp.array([0.0, -9.8]), is_apply_on_grid=True
    )

    boundary_idx = sim_builder.add_boundary(
        friction=0.0, origin=origin, end=end, gap=1e-4
    )

    sdf_star = hdx.StarSDF(points=5, inner_radius=1, outer_radius=2)
    sdf_collider_idx = sim_builder.add_sdf_collider(
        sdf_logic=sdf_star,
        center_of_mass=jnp.array([5.0, 2.5]),
        velocity=jnp.array([0.0, 0.0]),
        rotation=30.0 * jnp.pi / 180.0,
        angular_velocity=-180 * (jnp.pi / 180.0),
        gap=cell_size / 2,
        friction=0.9,
    )
    # grid_contact_idx = sim_builder.add_body_contact(
    #     couple_idx_actor= water_b_idx,
    #     couple_idx_receiver= clay_b_idx,
    #     is_reaction=True,
    #     # is_rigid= True,
    # )

    # =========================================================
    # Morph SDF test
    # =========================================================

    # =========================================================
    # 4. DEFINE SIM STATE AND CONTEXT
    # =========================================================
    solver_idx = sim_builder.set_solver(
        scheme="usl_aflip",
        # b_idx_list=[water_b_idx],
        # f_idx_list=[boundary_idx,grav_idx],
        alpha=0.90,
    )

    mpm_solver, sim_state = sim_builder.build(dt=dt)

    # # =========================================================
    # # 4. RUN SIMULATION
    # # =========================================================

    vis = hdx.RerunVisualizer(
        
        origin=origin, end=end, cell_size=cell_size, ppc=ppc,
        # recording_id="CompareModels",
        # root_path="RD",
        )

    # vis.log_sdf_boundary(
    #     sdf_logic=sdf_block,
    #     sdf_state=sdf_state,
    # )
    def log_simulation(sim_state: hdx.SimState):
        vis.log_simulation(sim_state)
        vis.log_sdf_boundary(
            sdf_logic=sdf_star,
            sdf_state=sim_state.forces[sdf_collider_idx],
        )

    def loop_body(i, sim_state):

        sim_state = mpm_solver.step(sim_state)
        jax.lax.cond(
            i % output_step == 0,
            lambda s: jax.debug.callback(log_simulation, s),
            lambda s: None,
            sim_state,
        )
        return sim_state

    @jax.jit
    def run_sim(sim_state: hdx.SimState):
        sim_state = jax.lax.fori_loop(0, total_steps, loop_body, sim_state)
        return sim_state

    sim_state = run_sim(sim_state)


import multiprocessing

if __name__ == "__main__":

    p = multiprocessing.Process(target=simulation_wrapper, args=())
    p.start()
    try:
        p.join()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Cleaning up...")

        # 1. Force JAX to clear backend caches (helps with VRAM)
        p.terminate()
        
        p.join()

        print("✅ Worker dead. GPU memory released.")
        
    # Check results if it finished naturally
    if p.exitcode == 0:
        print("Main: Simulation finished successfully.")
    else:
        print(f"Main: Process ended with code {p.exitcode}")