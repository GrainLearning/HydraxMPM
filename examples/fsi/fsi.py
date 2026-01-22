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

    class MorphSDFState(hdx.SDFObjectState):
        """
        State for a morphing object.
        blend_factor: 0.0 = Shape A, 1.0 = Shape B
        """

        blend_factor: float | Float[Array, ""]
        blend_rate: Float[Array, ""]

    # class MorphSDF(hdx.SDFObjectBase):
    #     """
    #     Linearly interpolates between two SDFs.

    #     Usage:
    #         morph = MorphSDF(start_shape=BoxSDF(...), end_shape=SphereSDF(...))
    #         state = morph.create_state(center, rotation, blend_factor=0.5)
    #     """

    #     shape_a: hdx.SDFObjectBase
    #     shape_b: hdx.SDFObjectBase

    #     def __init__(
    #         self, start_shape: hdx.SDFObjectBase, end_shape: hdx.SDFObjectBase
    #     ):
    #         self.shape_a = start_shape
    #         self.shape_b = end_shape

    #     def create_state(
    #         self,
    #         center_of_mass,
    #         rotation=None,
    #         blend_factor: float = 0.0,
    #         blend_rate: float = 0.0,
    #         velocity=None,
    #         angular_velocity=None,
    #     ) -> MorphSDFState:
    #         # call parent to get base fields
    #         base = super().create_state(
    #             center_of_mass, velocity, angular_velocity, rotation
    #         )

    #         return MorphSDFState(
    #             center_of_mass=base.center_of_mass,
    #             rotation=base.rotation,
    #             velocity=base.velocity,
    #             angular_velocity=base.angular_velocity,
    #             blend_factor=jnp.asarray(blend_factor),
    #             blend_rate=jnp.asarray(blend_rate)
    #         )

    #     def signed_distance_local(
    #         self, state, p_local: Float[Array, "dim"]
    #     ) -> Float[Array, ""]:

    #         d_a = self.shape_a.signed_distance_local(state, p_local)
    #         d_b = self.shape_b.signed_distance_local(state, p_local)

    #         # 2. Linear Interpolation
    #         # t=0 -> d_a, t=1 -> d_b
    #         t = jnp.clip(state.blend_factor, 0.0, 1.0)

    #         return (1.0 - t) * d_a + t * d_b

    #     def get_velocity(
    #         self, sdf_state: hdx.SDFObjectState, pos_world: Float[Array, "dim"], dt
    #     ) -> Float[Array, "dim"]:
    #         """
    #         Calculates kinematic velocity for a single point.
    #         """
    #         # Note everything is calculated in WORLD coordinates

    #         p_local = pos_world - sdf_state.center_of_mass

    #         # v = v_lin + w x r
    #         # Handle 2D vs 3D cross product
    #         if p_local.shape[0] == 2:
    #             # 2D Cross product: omega is scalar, r is vector
    #             # [-w * ry, w * rx]
    #             cross = jnp.array(
    #                 [
    #                     -sdf_state.angular_velocity * p_local[1],
    #                     sdf_state.angular_velocity * p_local[0],
    #                 ]
    #             )
    #         else:
    #             # 3D in WORLD frame
    #             cross = jnp.cross(sdf_state.angular_velocity, p_local)

    #         # Morph velocity via AD
    #         #  Spatial Gradient (d_phi / d_x) -> Normal
    #         # Note: We use the WORLD signed_distance, which includes rotation/translation logic
    #         grad_spatial_fn = jax.grad(self.signed_distance, argnums=1)
    #         spatial_grad = grad_spatial_fn(sdf_state, pos_world)

    #         # Parameter Gradient (d_phi / d_state)
    #         # This returns a MorphSDFState object full of gradients
    #         grad_state_fn = jax.grad(self.signed_distance, argnums=0)
    #         state_grads = grad_state_fn(sdf_state, pos_world)

    #         # Extract sensitivity to blend_factor
    #         dphi_dblend = state_grads.blend_factor

    #         # dphi/dt = (dphi/dblend) * (dblend/dt)
    #         dphi_dt = dphi_dblend * sdf_state.blend_rate

    #         # prevent artificial suction
    #         dphi_dt_clamped = jnp.minimum(dphi_dt, 0.0) 

    #         # C. Compute Morph Velocity Vector
    #         # v_morph = - ( dphi_dt / |grad phi|^2 ) * grad phi
    #         norm_sq = jnp.sum(spatial_grad**2)

    #         # Epsilon prevents NaN deep inside object
    #         v_morph = -(dphi_dt_clamped / (norm_sq + 1e-12)) * spatial_grad

    #         v_body = sdf_state.velocity + cross + v_morph

    #         return v_body

    # sdf_star = hdx.StarSDF(points=5, inner_radius=1, outer_radius=2)

    # sdf_sphere = hdx.SphereSDF(radius=4)

    # heart_sdf = HeartSDF(scale=2.5)
    # mdf_object = MorphSDF(
    #     start_shape=sdf_star,
    #     end_shape=sdf_sphere,
    # )

    # mdf_state = MorphSDFState(
    #     center_of_mass=jnp.array([5.0, 2.5]),
    #     velocity=jnp.array([0.0, 0.0]),
    #     rotation=30.0 * jnp.pi / 180.0,
    #     angular_velocity=0.0,
    #     blend_factor=0.0,
    #     blend_rate=0.0
    # )

    # def update_morph(sim_time, sdf_state):
    #     speed_factor = 5.0

    #     def compute_blend(t):
    #         omega = jnp.pi * speed_factor
    #         return 0.5 * (1.0 + jnp.sin(t * omega))

    #     b_factor, b_rate = jax.value_and_grad(compute_blend)(sim_time)

    #     # 3. Update BOTH fields in the state
    #     return eqx.tree_at(
    #         lambda s: (s.blend_factor, s.blend_rate), sdf_state, (b_factor, b_rate)
    #     )




    class ChainMorphSDF(hdx.SDFObjectBase):
        shapes: list[hdx.SDFObjectBase]

        def __init__(self, shapes: list[hdx.SDFObjectBase]):
            self.shapes = shapes

        def create_state(self, center_of_mass, rotation=None, blend_factor=0.0, blend_rate=0.0, velocity=None, angular_velocity=None):
            base = super().create_state(center_of_mass, velocity, angular_velocity, rotation)
            return MorphSDFState(
                center_of_mass=base.center_of_mass,
                rotation=base.rotation,
                velocity=base.velocity,
                angular_velocity=base.angular_velocity,
                blend_factor=jnp.asarray(blend_factor),
                blend_rate=jnp.asarray(blend_rate)
            )

        def signed_distance_local(self, state, p_local):
            # Compute distances for all shapes
            dists = jnp.stack([s.signed_distance_local(state, p_local) for s in self.shapes])
            
            T = state.blend_factor
            num_segments = len(self.shapes) - 1
            T = jnp.clip(T, 0.0, num_segments)
            
            # Identify segment
            idx = jnp.minimum(jnp.floor(T).astype(int), num_segments - 1)
            t = T - idx
            
            d_start = dists[idx]
            d_end   = dists[idx + 1]
            
            return (1.0 - t) * d_start + t * d_end

        def get_velocity(self, sdf_state, pos_world, dt):
            p_local = pos_world - sdf_state.center_of_mass
            if p_local.shape[0] == 2:
                cross = jnp.array([-sdf_state.angular_velocity * p_local[1], sdf_state.angular_velocity * p_local[0]])
            else:
                cross = jnp.cross(sdf_state.angular_velocity, p_local)

            grad_spatial_fn = jax.grad(self.signed_distance, argnums=1)
            spatial_grad = grad_spatial_fn(sdf_state, pos_world)

            grad_state_fn = jax.grad(self.signed_distance, argnums=0)
            state_grads = grad_state_fn(sdf_state, pos_world)

            dphi_dblend = state_grads.blend_factor
            dphi_dt = dphi_dblend * sdf_state.blend_rate
            dphi_dt_clamped = jnp.minimum(dphi_dt, 0.0) 

            norm_sq = jnp.sum(spatial_grad**2)
            v_morph = -(dphi_dt_clamped / (norm_sq + 1e-12)) * spatial_grad

            return sdf_state.velocity + cross + v_morph

    def update_morph(sim_time, sdf_state):
            # We want to go from 0 -> 2 -> 0
            max_val = 2.0 # (3 shapes - 1)
            period = 4.0  # Seconds for full cycle
            
            def compute_blend(t):
                # Triangle wave or Sine wave scaled to [0, max_val]
                norm_t = (1.0 + jnp.sin(t * 2 * jnp.pi / period - jnp.pi/2)) * 0.5
                return norm_t * max_val

            b_factor, b_rate = jax.value_and_grad(compute_blend)(sim_time)

            return eqx.tree_at(
                lambda s: (s.blend_factor, s.blend_rate), sdf_state, (b_factor, b_rate)
            )

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
    total_steps = int(20 / dt)
    output_step = 0.01 / dt

    # =========================================================
    # 2. DEFINE Water BODY p_idx 0, g_idx 0
    # =========================================================

    sdf_block = hdx.BoxSDF(size=(10.0, 3.0))
    

    
    position_stack = hdx.generate_particles_in_sdf(
        sdf_obj=sdf_block,
        center_of_mass=jnp.array([5.0, 4.0]),
        # center_of_mass=jnp.array([5.5, 0.5]),
        bounds_min=origin,
        bounds_max=end,
        cell_size=cell_size,
        ppc=ppc,
        mode="regular",
    )

    density_stack = jnp.full((len(position_stack),), 1000.0)

    water_p_idx = sim_builder.add_material_points(
        position_stack=position_stack,
        density_stack=density_stack,
        cell_size=cell_size,
        ppc=ppc,
    )

    water_g_idx = sim_builder.add_grid(
        origin=origin,
        end=end,
        cell_size=cell_size,
        padding=4,
    )

    water_c_id = sim_builder.add_constitutive_law(
        law=hdx.NewtonFluid(K=2e6, viscosity=1e-3, beta=7.0),
        # law = hdx.MuI_LC(
        #     K=2e6, mu_s=0.4, mu_d=1.41, I_0=1e-4,  d_p=0.025, alpha=1e-4
        # ),
        density_stack=density_stack,
    )

    water_b_idx = sim_builder.couple(
        shapefunction="quadratic",
    )


    # =========================================================
    # 2. DEFINE Water BODY p_idx 0, g_idx 0
    # =========================================================

    sdf_block = hdx.BoxSDF(size=(4.0, 2.0))
    

    
    position_stack = hdx.generate_particles_in_sdf(
        sdf_obj=sdf_block,
        center_of_mass=jnp.array([5.0, 8.0]),
        bounds_min=origin,
        bounds_max=end,
        cell_size=cell_size,
        ppc=ppc,
        mode="regular",
    )

    density_stack = jnp.full((len(position_stack),), 10000.0)

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
        law=hdx.LinearElasticLaw(E=1e7, nu=0.2),
        # law = hdx.MuI_LC(
        #     K=2e6, mu_s=0.4, mu_d=1.41, I_0=1e-4,  d_p=0.025, alpha=1e-4
        # ),
        density_stack=density_stack,
    )

    linear_b_idx = sim_builder.couple(
        shapefunction="quadratic",
    )

    # =========================================================
    # 3. DEFINE FORCES
    # =========================================================

    sdf_star = hdx.StarSDF(points=5, inner_radius=1.0, outer_radius=2.0)
    sdf_heart = HeartSDF(scale=2.0)
    sdf_sphere = hdx.SphereSDF(radius=3.5)
    sdf_plank = hdx.BoxSDF(size=(5.0, 1.0))
    chain_sdf = ChainMorphSDF(shapes=[sdf_star, sdf_heart, sdf_sphere, sdf_plank ])

    mdf_state = chain_sdf.create_state(
        center_of_mass=jnp.array([5.0, 5.0]), # Centered
        velocity=jnp.array([0.0, 0.0]),
        rotation=0.0,
        angular_velocity=1.0, # Slow spin
        blend_factor=0.0,
        blend_rate=0.0
    )

    grav_idx = sim_builder.add_gravity(
        gravity=jnp.array([0.0, -9.8]), is_apply_on_grid=True
    )

    domain_sdf = hdx.DomainSDF(
        origin=origin,
        end = end,
        frictions = 0.9,
        wall_offset = 0.5 * cell_size,
    )
    domain_idx = sim_builder.add_sdf_object(
        sdf_logic=domain_sdf,
    )

    domain_collider_idx = sim_builder.add_sdf_collider(gap=1e-6)



    # sdf_morph_idx = sim_builder.add_sdf_object(
    #     sdf_logic=chain_sdf,
    #     sdf_state=mdf_state,
    # )

    # morph_collider_idx = sim_builder.add_sdf_collider(
    #     gap=cell_size / 2,
    #     friction=0.9
    # )

    # cloud_logic = hdx.ParticleCloudSDF(smooth_k=10.0)

    # cloud_state = cloud_logic.create_state(
    #     points = position_stack,
    #     radii = jnp.full((len(position_stack),), 2* cell_size),
    # )
    # sdf_cloud_idx = sim_builder.add_sdf_object(
    #     sdf_logic=cloud_logic,
    #     sdf_state=cloud_state,
    # )

    # grid_contact_idx = sim_builder.add_body_contact(
    #     couple_idx_actor= water_b_idx,
    #     couple_idx_receiver= linear_b_idx,
    #     # is_reaction=True,
    #     # is_rigid= True,
    # )

    grid_contact_idx = sim_builder.add_body_contact(
        couple_idx_actor= linear_b_idx,
        couple_idx_receiver= water_b_idx,
        # is_reaction=True,
        # is_rigid= True,
    )

    # =========================================================
    # Morph SDF test
    # =========================================================

    # =========================================================
    # 4. DEFINE SIM STATE AND CONTEXT
    # =========================================================
    solver_idx = sim_builder.set_solver(
        # scheme="usl_aflip",
        scheme="usl_aflip",
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

    vis.log_sdf_boundary(
        sdf_logic=domain_sdf,
        sdf_state=sim_state.world.sdfs[domain_idx],
    )


    def log_simulation(sim_state: hdx.SimState):
        vis.log_simulation(sim_state)
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
