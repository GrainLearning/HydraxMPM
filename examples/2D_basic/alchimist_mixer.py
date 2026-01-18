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

        rounding smooths out corners, (0.0) are prone to glitch
        Domain: 2D (Shape) or 3D (Infinite Column along Z)
        The 'tip' of the heart is at the local origin (0,0).
        """

        scale: float
        rounding: float

        def __init__(self, scale=1.0, rounding =0.1):

            self.rounding = rounding
            self.scale = scale

        def signed_distance_local(self, state, p_local):

            p = p_local / self.scale

            x = p[0]
            y = p[1]

            # 3. Apply Symmetry
            x = jnp.abs(x + 1e-12)

            dx_a = x - 0.25
            dy_a = y - 0.75
            dist_a = jnp.sqrt(dx_a**2 + dy_a**2) - jnp.sqrt(2.0) / 4.0

            d1 = x**2 + (y - 1.0) ** 2

            w = 0.5 * jnp.maximum(x + y, 0.0)
            d2 = (x - w) ** 2 + (y - w) ** 2

            dist_b = jnp.sqrt(jnp.minimum(d1, d2)) * jnp.sign(x - y)

            dist_unit = jnp.where((y + x) > 1.0, dist_a, dist_b)

            return dist_unit * self.scale - self.rounding

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
        # Configuration
        hold_time = 1.5   # Seconds to stay as one shape
        morph_time = 1.5  # Seconds to transition
        step_time = hold_time + morph_time
        max_shape_idx = 3.0 # Star(0) -> Heart(1) -> Sphere(2) -> Plank(3)
        
        # We want to go 0 -> 1 -> 2 -> 3 -> 2 -> 1 -> 0 ...
        # Total cycle time for one full back-and-forth
        full_cycle_duration = step_time * max_shape_idx * 2
        
        def compute_blend(t):
            # 1. Ping-Pong Time
            # Map time to a 0 -> Max -> 0 triangle, but stretched to account for steps
            # This creates a value 'p' that goes 0..3..0 continuously
            
            # Triangle wave logic:
            # Normalized 0..1
            norm_t = (t % full_cycle_duration) / full_cycle_duration 
            # 0..2
            norm_t = norm_t * 2.0 
            # 0..1..0 (Linear triangle)
            ping_pong = 1.0 - jnp.abs(norm_t - 1.0)
            
            # Map to total number of steps (0..3.0)
            # This 'p' represents the continuous fractional shape index
            p = ping_pong * max_shape_idx
            
            # 2. Add Pauses (Stepping)
            # We separate 'p' into integer part (shape index) and fractional part
            idx = jnp.floor(p)
            frac = p - idx
            
            # We want 'frac' to stay at 0 for 'hold_time', then go 0->1 during 'morph_time'
            # Calculate the ratio of hold vs morph in the generic 0..1 window
            pause_ratio = hold_time / step_time
            
            # Remap fractional part:
            # If frac < pause_ratio: Output 0 (Hold)
            # If frac > pause_ratio: Smoothstep 0->1
            
            # (frac - start) / (end - start)
            active_t = (frac - pause_ratio) / (1.0 - pause_ratio)
            active_t = jnp.clip(active_t, 0.0, 1.0)
            
            # Smoothstep (3t^2 - 2t^3) ensures velocity starts and ends at 0
            smooth_t = active_t * active_t * (3.0 - 2.0 * active_t)
            
            return idx + smooth_t

        # Use AD to get Value and Rate
        b_val, b_rate = jax.value_and_grad(compute_blend)(sim_time)

        # --- EFFECTS LOGIC ---
        
        # 1. Rotation Speed (Mixer Effect)
        # Star (0.0): Fast Spin (5.0)
        # Heart (1.0): Still (0.0)
        # Sphere (2.0): Reverse Spin (-3.0)
        # Plank (3.0): Still (0.0)
        
        # We define a continuous function for omega based on the blend value
        # Using simple Gaussian-like bumps or Lerps
        
        # Is close to Star? (1 at 0.0, 0 at 1.0)
        w_star = jnp.exp(-2.0 * (b_val - 0.0)**2) * 5.0
        # Is close to Sphere?
        w_sphere = jnp.exp(-2.0 * (b_val - 2.0)**2) * -3.0
        
        target_omega = w_star + w_sphere
        
        # 2. Vertical Movement (The Crusher)
        # Move up and down. 
        # When b_val is near 3 (Plank), we want to be LOW to crush particles.
        # When b_val is near 0 (Star), we want to be MIDDLE.
        
        target_y = 6.0 - 2.5 * jnp.exp(-2.0 * (b_val - 3.0)**2)
        
        # Use AD for Velocity Consistency again!
        # If we just set position, velocity is wrong. Let AD compute velocity of the Y motion.  
            
        def compute_y_pos(t):
            bv = compute_blend(t)
            
            # Start at 2.5.
            # When blend is near 3.0 (Plank), drop down by 1.5 units (to y=1.0)
            base_y = 2.5
            drop_amount = 1.5
            
            return base_y - drop_amount * jnp.exp(-2.0 * (bv - 3.0)**2)


        y_pos, y_vel = jax.value_and_grad(compute_y_pos)(sim_time)
        
        new_com = sdf_state.center_of_mass.at[1].set(y_pos)
        new_vel = jnp.array([0.0, y_vel])

        return eqx.tree_at(
            lambda s: (s.blend_factor, s.blend_rate, s.angular_velocity, s.center_of_mass, s.velocity), 
            sdf_state, 
            (b_val, b_rate, target_omega, new_com, new_vel)
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
    total_steps = int(20.0 / dt) 
    output_step = int(0.016 / dt)

    # =========================================================
    # 2. DEFINE Water BODY p_idx 0, g_idx 0
    # =========================================================
    # sdf_block = hdx.BoxSDF(size=(9.0, 4.0))
    sdf_block = hdx.BoxSDF(size=(10.0, 6.0))
    # sdf = HeartSDF(scale=3.0)
    
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
        # law=hdx.NewtonFluid(K=2e6, viscosity=1e-3, beta=7.0),
        law = hdx.MuI_LC(
            K=2e6, mu_s=0.4, mu_d=1.41, I_0=1e-4,  d_p=0.025, alpha=1e-4
        ),
        density_stack=density_stack,
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

    sdf_star = hdx.StarSDF(points=5, inner_radius=1.0, outer_radius=2.0)
    sdf_heart = HeartSDF(scale=4.0)
    sdf_sphere = hdx.SphereSDF(radius=3.5)
    sdf_plank = hdx.BoxSDF(size=(5.0, 1.0))
    chain_sdf = ChainMorphSDF(shapes=[sdf_star, sdf_heart, sdf_sphere, sdf_plank ])

    mdf_state = chain_sdf.create_state(
        center_of_mass=jnp.array([5.0, 2.5]), # Centered
        velocity=jnp.array([0.0, 0.0]),
        rotation=0.0,
        angular_velocity=1.0, # Slow spin
        blend_factor=0.0,
        blend_rate=0.0
    )

    grav_idx = sim_builder.add_gravity(
        gravity=jnp.array([0.0, -9.8]), is_apply_on_grid=True
    )

    boundary_idx = sim_builder.add_boundary(
        friction=0.0, origin=origin, end=end, gap=1e-4
    )

    sdf_collider_idx = sim_builder.add_sdf_collider(
        sdf_logic=chain_sdf,
        f_state=mdf_state,
        #         gap = 1e-4,
        # center_of_mass=jnp.array([5.0, 2.5]),
        # velocity=jnp.array([0.0, 0.0]),
        # rotation=30.0 * jnp.pi / 180.0,
        # angular_velocity=-180 * (jnp.pi / 180.0),
        # gap=cell_size / 2,
        gap=1e-3,
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
        origin=origin,
        end=end,
        cell_size=cell_size,
        ppc=ppc,
        # recording_id="CompareModels",
        # root_path="RD",
    )

    def log_simulation(sim_state: hdx.SimState):
        vis.log_simulation(sim_state)
        vis.log_sdf_boundary(
            sdf_logic=chain_sdf,
            sdf_state=sim_state.forces[sdf_collider_idx],
        )

    def loop_body(i, sim_state):
        # sim_state = mpm_solver.step(sim_state)

        time = sim_state.time

        sim_forces = list(sim_state.forces)
        sdf_state = sim_forces[sdf_collider_idx]
        next_sdf_state = update_morph(time, sdf_state)
        sim_forces[sdf_collider_idx] = next_sdf_state

        sim_state = eqx.tree_at(lambda s: s.forces, sim_state, tuple(sim_forces))
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
