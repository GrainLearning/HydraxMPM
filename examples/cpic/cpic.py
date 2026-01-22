# """This example is coming back soon"""


def simulation_wrapper():

    import sys
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    import jax

    import jax.numpy as jnp
    import numpy as np

    import hydraxmpm as hdx
    from jaxtyping import Float, Array

    # from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    # import equinox as eqx

    # # Set this to the number of threads/cores you want to use as "devices"
    # os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    import equinox as eqx

    class LineSegmentSDF(hdx.SDFObjectBase):
        """
        A Finite Line Segment with thickness.
        Effectively a Capsule.
        """

        start: Float[Array, "dim"]
        end: Float[Array, "dim"]
        thickness: float

        def __init__(self, start, end, thickness=0.005):
            self.start = jnp.asarray(start)
            self.end = jnp.asarray(end)
            self.thickness = thickness

        def signed_distance_local(self, state, p):
            # Vector math to find distance to segment
            pa = p - self.start
            ba = self.end - self.start

            # Project p onto ba, clamp between 0 and 1
            h = jnp.clip(jnp.dot(pa, ba) / jnp.dot(ba, ba), 0.0, 1.0)

            # Distance to closest point
            dist = jnp.linalg.norm(pa - ba * h)

            return dist - self.thickness

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
            dist_a = jnp.sqrt(dx_a**2 + dy_a**2) - jnp.sqrt(2.0) / 4.0

            d1 = x**2 + (y - 1.0) ** 2

            w = 0.5 * jnp.maximum(x + y, 0.0)
            d2 = (x - w) ** 2 + (y - w) ** 2

            dist_b = jnp.sqrt(jnp.minimum(d1, d2)) * jnp.sign(x - y)

            dist_unit = jnp.where((y + x) > 1.0, dist_a, dist_b)

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

    dt = 0.5e-4
    total_steps = int(20 / dt)
    output_step = 0.01 / dt

    sdf_heart = HeartSDF(scale=4.0)

    sdf_state = sdf_heart.create_state(
        center_of_mass=jnp.array([6.5, 7.5]),
        rotation=90 * jnp.pi / 180.0,
    )

    position_stack = hdx.generate_particles_in_sdf(
        sdf_obj=sdf_heart,
        sdf_state=sdf_state,
        bounds_min=origin,
        bounds_max=end,
        cell_size=cell_size,
        ppc=ppc,
        mode="regular",
    )

    density_stack = jnp.full((len(position_stack),), 2000.0)

    linear_p_idx = sim_builder.add_material_points(
        position_stack=position_stack,
        density_stack=density_stack,
        cell_size=cell_size,
        ppc=ppc,
    )

    linear_g_idx = sim_builder.add_grid(
        origin=origin,
        end=end,
        cell_size=cell_size,
        padding=4,
    )

    linear_c_id = sim_builder.add_constitutive_law(
        # law=hdx.LinearElasticLaw(E=1e7, nu=0.2),
        law=hdx.NewtonFluid(K=2e6, viscosity=1e-3, beta=7.0),
        density_stack=density_stack,
    )

    linear_b_idx = sim_builder.couple(
        shapefunction="quadratic",
    )

    # =========================================================
    # 3. DEFINE FORCES
    # =========================================================

    grav_idx = sim_builder.add_gravity(
        gravity=jnp.array([0.0, -9.8]), is_apply_on_grid=True
    )

    domain_sdf = hdx.DomainSDF(
        origin=origin,
        end=end,
        frictions=0.9,
        wall_offset=0.75 * cell_size,
    )
    domain_idx = sim_builder.add_sdf_object(
        sdf_logic=domain_sdf,
    )

    domain_collider_idx = sim_builder.add_sdf_collider(
        gap=0.001 * cell_size,
        friction=1.0
    )

    knife_sdf = hdx.CapsuleSDF(
        height=4.0,
        radius=0.5 * cell_size,
    )
    knife_state = knife_sdf.create_state(
        center_of_mass=jnp.array([5.0, 2.0]),
        velocity=jnp.array([0.0, 0.0]),
        angular_velocity=0.0,
    )
    knife_idx = sim_builder.add_sdf_object(
        sdf_logic=knife_sdf,
        sdf_state=knife_state,

    )
    knife_collider_idx = sim_builder.add_sdf_collider(
        gap=0.1 * cell_size,
        friction=0.0
    )

    # =========================================================
    # 4. DEFINE SIM STATE AND CONTEXT
    # =========================================================
    solver_idx = sim_builder.set_solver(
        scheme="usl_aflip",
        # scheme="usl",
        # b_idx_list=[water_b_idx],
        # f_idx_list=[boundary_idx,grav_idx],
        alpha=0.90,
    )

    mpm_solver, sim_state = sim_builder.build(dt=dt)

    # # # =========================================================
    # # # 4. RUN SIMULATION
    # # # =========================================================

    vis = hdx.RerunVisualizer(
        origin=origin,
        end=end,
        cell_size=cell_size,
        ppc=ppc,
        # recording_id="CompareModels",
        # root_path="RD",
    )

    # vis.log_sdf_boundary(
    #     sdf_logic=domain_sdf,
    #     sdf_state=sim_state.world.sdfs[domain_idx],
    # )

    def log_simulation(sim_state: hdx.SimState):
        vis.log_simulation(sim_state)
        vis.log_sdf_boundary(
            sdf_logic=knife_sdf,
            sdf_state=sim_state.world.sdfs[knife_idx],
            resolution=200,
            # star
        )
        # vis.log_sdf_boundary(
        #     sdf_logic=chain_sdf,
        #     sdf_state=sim_state.world.sdfs[sdf_morph_idx],
        # )

        # vis.log_sdf_boundary(
        #     sdf_logic=cloud_logic,
        #     sdf_state=sim_state.world.sdfs[sdf_cloud_idx],
        #     resolution=60,
        # )

    def loop_body(i, sim_state):
        # sim_state = mpm_solver(sim_state)

        time = sim_state.time

        # sdfs = list(sim_state.world.sdfs)
        # # sdfs[sdf_morph_idx] = update_morph(time, sdfs[sdf_morph_idx])

        # current_sdf_state = sim_state.world.sdfs[sdf_cloud_idx]
        # current_mp_state = sim_state.world.material_points[water_p_idx]

        # sdfs[sdf_cloud_idx] = cloud_logic.update_cloud_from_mps(
        #     sdf_state=current_sdf_state,
        #     mp_state=current_mp_state,
        # )

        # world = eqx.tree_at(lambda w: w.sdfs, sim_state.world, tuple(sdfs))
        # sim_state = eqx.tree_at(lambda s: s.world, sim_state, world)
        sim_state = mpm_solver(sim_state)
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

    # vis.log_sdf_boundary(
    #     sdf_logic=sdf_block,
    #     sdf_state=sdf_state,
    # )
    # def log_simulation(sim_state: hdx.SimState):
    #     vis.log_simulation(sim_state)
    #     vis.log_sdf_boundary(
    #         sdf_logic=sdf_star,
    #         sdf_state=sim_state.forces[sdf_collider_idx],
    #     )

    # def loop_body(i, sim_state):

    #     sim_state = mpm_solver.step(sim_state)
    #     jax.lax.cond(
    #         i % output_step == 0,
    #         lambda s: jax.debug.callback(log_simulation, s),
    #         lambda s: None,
    #         sim_state,
    #     )
    #     return sim_state

    # @jax.jit
    # def run_sim(sim_state: hdx.SimState):
    #     sim_state = jax.lax.fori_loop(0, total_steps, loop_body, sim_state)
    #     return sim_state

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
