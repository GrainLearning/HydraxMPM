
# from hydraxmpm.sdf.sdfobject import SDFObjectState





# class MorphSDFState(SDFObjectState):
#     """
#     State for a morphing object.
#     blend_factor: 0.0 = Shape A, 1.0 = Shape B
#     """
#     blend_factor: float | Float[Array, ""]
#     hist_dist: Float[Array, "num_nodes"]

# class MorphSDF(SDFObjectBase):
#     """
#     Linearly interpolates between two SDFs.
    
#     Usage:
#         morph = MorphSDF(start_shape=BoxSDF(...), end_shape=SphereSDF(...))
#         state = morph.create_state(center, rotation, blend_factor=0.5)
#     """
#     shape_a: SDFObjectBase
#     shape_b: SDFObjectBase

#     def __init__(self, start_shape: SDFObjectBase, end_shape: SDFObjectBase):
#         self.shape_a = start_shape
#         self.shape_b = end_shape

#     def create_state(
#         self, 
#         center_of_mass, 
#         rotation=None, 
#         blend_factor: float = 0.0
#     ) -> MorphSDFState:
#         # call parent to get base fields
#         base = super().create_state(center_of_mass, rotation)
        
#         return MorphSDFState(
#             center_of_mass=base.center_of_mass,
#             rotation=base.rotation,
#             blend_factor=jnp.asarray(blend_factor),
#             hist_dist=0.0
#         )

#     def signed_distance_local(
#         self, 
#         state, 
#         p_local: Float[Array, "dim"]
#     ) -> Float[Array, ""]:
        
#         # 1. Compute distance to both shapes in the SAME local frame
#         # (They share the center_of_mass and rotation of the parent MorphSDF)
#         # We pass the 'state' down, but shape_a/b likely ignore the extra 'blend_factor' field 
#         # due to duck-typing in JAX, or we can construct a temporary base state if needed.
#         # Assuming shape_a/b are standard SDFs that expect SDFObjectState:
        
#         d_a = self.shape_a.signed_distance_local(state, p_local)
#         d_b = self.shape_b.signed_distance_local(state, p_local)
        
#         # 2. Linear Interpolation
#         # t=0 -> d_a, t=1 -> d_b
#         t = jnp.clip(state.blend_factor, 0.0, 1.0)
        
#         return (1.0 - t) * d_a + t * d_b
    
#     # def update_history(self, state: MorphSDFState, pos_world ) -> MorphSDFState:
#     #     p_local = pos_world - state.center_of_mass
#     #     dist_next = self.signed_distance_local(state, p_local)
#     #     return 
#     #     eqx.tree_at(
#     #         lambda s: s.hist_dist,
#     #         state,
#     #         dist_next
#     #     )
    
#     def get_velocity(self, sdf_state: MorphSDFState, pos_world: Float[Array, "dim"],dt) -> Float[Array, "dim"]:
#         """
#         Calculates kinematic velocity for a single point.
        
#         vwall​(x)=vlinear​+ω×(x−xcenter​)
        
#         """
#         #  Velocity is calculated in World Frame relative to COM
#         # we assume angular_velocity is in World/Body frame
#         p_local = pos_world - sdf_state.center_of_mass
        
#         # v = v_lin + w x r
#         # Handle 2D vs 3D cross product
#         if p_local.shape[0] == 2:
#             # 2D Cross product: omega is scalar, r is vector
#             # [-w * ry, w * rx]
#             cross = jnp.array([-sdf_state.angular_velocity * p_local[1], sdf_state.angular_velocity * p_local[0]])
#         else:
#             # 3D Cross product
#             cross = jnp.cross(sdf_state.angular_velocity, p_local)

#         v_body = sdf_state.velocity + cross

#         dist_curr = sdf_state.hist_dist
#         dist_next = self.signed_distance_local(sdf_state, p_local)
    
#         # Rate of change of the field value
#         d_phi_dt = (dist_next - dist_curr) / dt
        
#         # Get Local Normal
#         # (We can approximate world normal and rotate, or calc local grad)
#         # World normal is easier since we have the helper
#         n_world = self.get_normal(sdf_state, pos_world)
        
#         # v_expansion = - (d_phi/dt) * n
#         # The negative sign is because if Distance decreases (dist_next < dist_curr),
#         # the surface is moving OUTWARDS towards the point (positive velocity).
#         v_morph = -d_phi_dt * n_world
        
#         return v_body + v_morph



#     sdf_box = hdx.BoxSDF(size=(1.0, 4.0))
#     sdf_star = hdx.StarSDF(points=5, inner_radius=1, outer_radius=2)

#     mdf_object = hdx.MorphSDF(
#         start_shape=sdf_box,
#         end_shape=sdf_star,
#     )

#     mdf_state = hdx.MorphSDFState(
#         center_of_mass=jnp.array([5.0, 2.5]),
#         velocity=jnp.array([0.0, 0.0]),
#         rotation= 30.0 * jnp.pi / 180.0,
#         angular_velocity=0.0,
#         blend_factor=0.0,
#         hist_dist=0.0,
#     )

#     def update_morph(sim_time, sdf_state):
#         # Oscillate between Box and Sphere every 1 second
#         # t goes 0 -> 1 -> 0
#         # 1.0 = 2.0 second cycle
#         # 5.0 = 0.4 second cycle (5x faster)
#         speed_factor = 5.0 
    
#         # Angular frequency omega = pi * speed_factor
#         t = 0.5 * (1.0 + jnp.sin(sim_time * jnp.pi * speed_factor))
#         new_sdf_state = sdf_state.update_history(sdf_state, sdf_state.center_of_mass)
#         return eqx.tree_at(
#             lambda s: s.blend_factor, 
#             new_sdf_state, 
#             t
#         )


#     sdf_collider_idx = sim_builder.add_sdf_collider(
#         sdf_object = mdf_object,
#         f_state = mdf_state,
#         gap = 1e-4,
#     )


    # vis = hdx.RerunVisualizer(
    #     origin=origin, end=end, cell_size=cell_size, ppc=ppc
    # )

    # # vis.log_sdf_boundary(
    # #     sdf_logic=sdf_block,
    # #     sdf_state=sdf_state,    
    # # )
    # def log_simulation(sim_state: hdx.SimState):
    #     vis.log_simulation(sim_state)
    #     vis.log_sdf_boundary(
    #         sdf_logic=mdf_object,
    #         sdf_state=sim_state.forces[sdf_collider_idx],    
    #     )
    # def loop_body(i, sim_state):
    #     # sim_state = mpm_solver.step(sim_state)

    #     time = sim_state.time

    #     sim_forces = list(sim_state.forces)
    #     sdf_state = sim_forces[sdf_collider_idx]
    #     next_sdf_state = update_morph(
    #         time,
    #         sdf_state
    #     )
    #     sim_forces[sdf_collider_idx] = next_sdf_state

    #     sim_state = eqx.tree_at(
    #         lambda s: s.forces,
    #         sim_state,
    #         tuple(sim_forces)
    #     )
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
