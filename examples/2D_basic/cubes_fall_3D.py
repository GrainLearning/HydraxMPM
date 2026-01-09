# """This example is coming back soon"""





def simulation_wrapper():


    import sys
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
    import jax

    # print(jax.devices("gpu"))
    # jax.config.update("jax_default_device", jax.devices("gpu")[1])

    import jax.numpy as jnp
    import numpy as np

    import hydraxmpm as hdx

    # from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


    # import equinox as eqx

    # # Set this to the number of threads/cores you want to use as "devices"
    # os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    import equinox as eqx
    # =========================================================
    # 1. SIMULATION PARAMETERS
    # =========================================================
    origin = jnp.array((0.0, 0.0, 0.0))
    end = jnp.array((10.0, 10.0, 10.0))
    domain_size = end - origin
    ppc = 2
    cell_size = (1 / 80.0) * 10  # 80 cells along each axis

    sim_builder = hdx.SimBuilder()

    dt = 0.5e-3
    total_steps = int(10/ dt)
    output_step = 0.01 / dt

    # =========================================================
    # 2. DEFINE Water BODY p_idx 0, g_idx 0
    # =========================================================
    # sdf_block = hdx.BoxSDF(size=(9.0, 4.0))
    sdf_block = hdx.BoxSDF(size=(9.0, 5.0, 9.0))
    
    
    # sdf_block = hdx.BoxSDF(size=(2.0, 2.0, 2.0))
    position_stack = hdx.generate_particles_in_sdf(
        sdf_obj=sdf_block,
        center_of_mass=jnp.array([5.0, 7.0, 5.0]),
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

    water_c_id = sim_builder.add_constitutive_law(
        #  law = hdx.NewtonFluid(K=2e6, viscosity=1e-3, beta=7.0),
        law = hdx.MuI_LC(
            K=2e6, mu_s=0.4, mu_d=1.41, I_0=1e-4,  d_p=0.025, alpha=1e-4
        ),
         density_stack=density_stack
    )

    water_b_idx = sim_builder.couple(
        # p_idx=water_p_idx,
        # g_idx=water_g_idx,
        # c_idx=water_c_id,
        shapefunction="cubic",
    )

    # =========================================================
    # clay
  

    # position_stack = hdx.generate_particles_in_sdf(
    #     sdf_obj=sdf_block,
    #      center_of_mass=jnp.array([5.0, 7.0, 5.0]),
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
        gravity=jnp.array([0.0, -9.8, 0.0]), is_apply_on_grid=True
    )

    boundary_idx = sim_builder.add_boundary(
        friction=0.0, origin=origin, end=end, gap=1e-4
    )


    def axis_angle_to_quat(axis, angle_rad):

        axis = axis / (jnp.linalg.norm(axis) + 1e-9) # Normalize
        half_angle = angle_rad / 2.0
        
        w = jnp.cos(half_angle)
        xyz = axis * jnp.sin(half_angle)
        
        # Return [w, x, y, z]
        return jnp.concatenate([jnp.array([w]), xyz])


    axis_z = jnp.array([0.0, 0.0, 1.0])
    deg2rad = jnp.pi / 180.0
    rot_quat = axis_angle_to_quat(axis_z, 30.0 * deg2rad)
    ang_vel_vector = axis_z * (-180.0 * deg2rad)
    
    sdf_star = hdx.StarSDF(points=5, inner_radius=1, outer_radius=2)
    sdf_collider_idx = sim_builder.add_sdf_collider(
        sdf_object=sdf_star,
        center_of_mass=jnp.array([5.0, 2.5, 5.0]),
        # velocity=jnp.array([0.0, 0.0]),
        # rotation=30.0 * jnp.pi / 180.0,
        angular_velocity=ang_vel_vector,
        gap=cell_size,
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

    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    # devices = jax.devices()
    # num_devices = len(devices)
    # print(f"🚀 Running on {num_devices} ")
    # # devices = devices[6:]

    # if num_devices > 1:


    #     mesh = Mesh(devices, axis_names=('ensemble',))
    #     sharding_particles = NamedSharding(mesh, P('ensemble')) 
   
    #     sharding_replicated = NamedSharding(mesh, P()) # Fully replicated
    #     # sharding_particles_1d = NamedSharding(mesh, P('ensemble')) # Add 1D sharding spec
    #     def get_sharding_spec(leaf: hdx.SimState):
    #             """
    #             Intelligently distributes the SimState.
    #             Particles -> Sharded across devices.
    #             Grids -> Replicated on all devices.
    #             """
                
    #             if not eqx.is_array(leaf):
    #                 return sharding_replicated
                
    #             # Heuristic: Shard the particle arrays, replicate everything else
    #             # Note: Check dim > 0 to avoid sharding scalars
    #             is_particle_array = (leaf.ndim > 0) and (leaf.shape[0] == len(position_stack))
                
    #             if is_particle_array:
    #                 return sharding_particles
    #             else:
    #                 return sharding_replicated

    #             # Apply the distribution
    #             # distributed_state = jax.tree.map(_sharding_decision, state)
    #             # return distributed_state
    #     # Apply transformations to SimState
    #     # 1. Shard Particles
    #     print("⏳ Distributing state across devices...")
    #     # sim_state = distribute_state(sim_state)
    #     sharding_tree = jax.tree.map(get_sharding_spec, sim_state)
    #     print("✅ State distributed.")

        
    #     print("⏳ Distributing state across devices...")
    #     sim_state = jax.device_put(sim_state, sharding_tree)
    #     print("✅ State distributed.")


    # # =========================================================
    # # 4. RUN SIMULATION
    # # =========================================================

    vis = hdx.RerunVisualizer(
        origin=origin, end=end, cell_size=cell_size, ppc=ppc,
        #   mode="save"
        
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
        jax.debug.print("Logged step")

    def loop_body(i, sim_state):

        sim_state = mpm_solver.step(sim_state)
        jax.lax.cond(
            i % output_step == 0,
            lambda s: jax.debug.callback(log_simulation, s),
            lambda s: None,
            sim_state,
        )
        return sim_state

    # @jax.jit
    # def run_sim(sim_state: hdx.SimState):
    #     sim_state = jax.lax.fori_loop(0, total_steps, loop_body, sim_state)
    #     return sim_state


    # 6. EXPLICIT JIT COMPILATION
    # We pass in_shardings and out_shardings to prevent the compiler 
    # from trying to force data to Device 0.
    # @jax.jit(in_shardings=(sharding_tree,), out_shardings=sharding_tree)
    def run_sim(sim_state: hdx.SimState):
        sim_state = jax.lax.fori_loop(0, total_steps, loop_body, sim_state)
        return sim_state

    # Run
    sim_state = run_sim(sim_state)
    
    
    # sim_state = run_sim(sim_state)

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