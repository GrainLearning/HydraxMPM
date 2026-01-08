# import pytest
# import jax
# import jax.numpy as jnp
# import equinox as eqx

# import hydraxmpm as hdx



# class TestRigidBodyContact:
#     @pytest.fixture
#     def shape_map(self):
#         return hdx.ShapeFunctionMapping("linear", 2, 2)

#     def test_create_state(self, shape_map):

#         contact = hdx.RigidBodyContact(shape_map)
        
#         pos = jnp.array([[0.0, 0.0], [1.0, 0.0]])
#         vel = jnp.array([[0.0, 0.0], [0.0, 0.0]])
#         normals = jnp.array([[1.0, 0.0], [0.0, 2.0]]) 


#         state = contact.create_state(pos, vel, normals)
        
#         assert state.position.shape == (2, 2)
#         assert state.mass.shape == (2,)
#         assert jnp.allclose(state.mass, 1.0) # Default mass
        
#         # Check normalization
#         assert jnp.allclose(jnp.linalg.norm(state.normals, axis=1), 1.0)
#         assert jnp.allclose(state.normals[1], jnp.array([0.0, 1.0]))
    

#     def test_apply_kinematics(self, shape_map):
#         def callback(time, state):
#             return state.position + 1.0, state.velocity + 1.0

#         contact = hdx.RigidBodyContact(
#             shape_map,
#             kinematic_callback=callback)
        
#         pos = jnp.array([[0.0, 0.0]])
#         vel = jnp.array([[0.0, 0.0]])
#         normals = jnp.array([[1.0, 0.0]])
#         state = contact.create_state(pos, vel, normals)
        
#         f_states = [state]
        
#         new_f_states = contact.apply_kinematics(None, None, f_states, None, 0.1, 0.0)
#         new_state = new_f_states[0]
        
#         assert jnp.allclose(new_state.position, pos + 1.0)
#         assert jnp.allclose(new_state.velocity, vel + 1.0)

#     def test_apply_grid_moments_penetration(self, shape_map):
#         # Setup:
#         # Node 0: MP Mass=1, MP Vel=(1,0) -> Mom=(1,0)
#         # RB Particle 0: Mass=1, Vel=(0,0), Normal=(-1,0) (Surface on right, normal points left)
#         # Rel Vel = (1,0) - (0,0) = (1,0)
#         # Normal = (-1,0)
#         # v_n = dot((1,0), (-1,0)) = -1.0 (Penetrating)
#         # Tangential = 0
#         # Friction = 0.5
#         # Result: Normal velocity removed. Tangential remains 0.
#         # Final MP Vel = (0,0) -> Mom = (0,0)

#         contact = hdx.RigidBodyContact( shape_map, mu=0.5)
        
#         # RB State
#         rb_pos = jnp.array([[0.0, 0.0]]) # Maps to Node 0
#         rb_vel = jnp.array([[0.0, 0.0]])
#         rb_norm = jnp.array([[-1.0, 0.0]])
#         rb_state = contact.create_state(rb_pos, rb_vel, rb_norm)
        
#         # Grid State
#         # We need to manually create a GridState with data
#         # GridTopology has 4 nodes (2x2)
#         grid_state = hdx.GridState.create(
#             origin=jnp.array([0.0, 0.0]),
#             end=jnp.array([1.0, 1.0]),
#             cell_size=1.0

#         )
#         num_nodes = grid_state.num_cells
#         mass_stack = jnp.zeros(num_nodes).at[0].set(1.0)
#         moment_nt_stack = jnp.zeros((num_nodes, 2)).at[0].set(jnp.array([1.0, 0.0]))
        
#         # Create dummy GridState
#         grid_state = eqx.tree_at(lambda g: g.mass_stack, grid_state, mass_stack)
#         grid_state = eqx.tree_at(lambda g: g.moment_nt_stack, grid_state, moment_nt_stack)
        
#         grid_states = [grid_state]
#         f_states = [rb_state]
        
#         new_grid_states, _ = contact.apply_grid_moments(None, grid_states, f_states, None, 0.1, 0.0)
        
#         new_mom = new_grid_states[0].moment_nt_stack[0]
#         assert jnp.allclose(new_mom, jnp.array([0.0, 0.0]))

#     def test_apply_grid_moments_friction(self, shape_map):
#         # Setup:
#         # Node 0: MP Mass=1, MP Vel=(-1, 1) -> Mom=(-1, 1)
#         # RB Particle 0: Mass=1, Vel=(0,0), Normal=(1,0) (Surface on left, normal points right)
#         # Rel Vel = (-1, 1)
#         # Normal = (1, 0)
#         # v_n = dot((-1, 1), (1, 0)) = -1.0 (Penetrating)
#         # Tangential Vec = (-1, 1) - (-1 * (1,0)) = (0, 1)
#         # Tangential Mag = 1.0
#         # Friction Limit = mu * |v_n| = 0.5 * 1.0 = 0.5
#         # New Tangential Mag = 1.0 - 0.5 = 0.5
#         # New Tangential Vec = (0, 0.5)
#         # New Rel Vel = (0, 0.5)
#         # New MP Vel = (0, 0.5) -> Mom = (0, 0.5)

#         contact = hdx.RigidBodyContact(shape_map, mu=0.5)
        
#         rb_pos = jnp.array([[0.0, 0.0]])
#         rb_vel = jnp.array([[0.0, 0.0]])
#         rb_norm = jnp.array([[1.0, 0.0]])
#         rb_state = contact.create_state(rb_pos, rb_vel, rb_norm)
        
#         grid_state = hdx.GridState.create(
#             origin=jnp.array([0.0, 0.0]),
#             end=jnp.array([1.0, 1.0]),
#             cell_size=1.0

#         )

#         mass_stack = jnp.zeros(grid_state.num_cells).at[0].set(1.0)
#         moment_nt_stack = jnp.zeros((grid_state.num_cells, 2)).at[0].set(jnp.array([-1.0, 1.0]))
        
#         grid_state = eqx.tree_at(lambda g: g.mass_stack, grid_state, mass_stack)
#         grid_state = eqx.tree_at(lambda g: g.moment_nt_stack, grid_state, moment_nt_stack)
        
#         grid_states = [grid_state]
#         f_states = [rb_state]
        
#         new_grid_states, _ = contact.apply_grid_moments(None, grid_states, f_states, None, 0.1, 0.0)
        
#         new_mom = new_grid_states[0].moment_nt_stack[0]
#         assert jnp.allclose(new_mom, jnp.array([0.0, 0.5]))

#     def test_apply_grid_moments_no_contact(self, shape_map):
#         # Setup:
#         # Node 0: MP Mass=1, MP Vel=(1,0)
#         # RB Particle 0: Mass=0 (No RB here)
#         # Result: No change

#         contact = hdx.RigidBodyContact(shape_map)
        
#         # RB State with 0 mass at node 0 (or just position elsewhere)
#         # But our mock maps index 0 to node 0.
#         # So we set mass to 0.
#         rb_pos = jnp.array([[0.0, 0.0]])
#         rb_vel = jnp.array([[0.0, 0.0]])
#         rb_norm = jnp.array([[1.0, 0.0]])
#         rb_state = contact.create_state(rb_pos, rb_vel, rb_norm, mass=jnp.array([0.0]))
        
#         grid_state = hdx.GridState.create(
#             origin=jnp.array([0.0, 0.0]),
#             end=jnp.array([1.0, 1.0]),
#             cell_size=1.0

#         )

#         mass_stack = jnp.zeros(grid_state.num_cells).at[0].set(1.0)
#         moment_nt_stack = jnp.zeros((grid_state.num_cells, 2)).at[0].set(jnp.array([1.0, 0.0]))
        
#         grid_state = eqx.tree_at(lambda g: g.mass_stack, grid_state, mass_stack)
#         grid_state = eqx.tree_at(lambda g: g.moment_nt_stack, grid_state, moment_nt_stack)
        
#         grid_states = [grid_state]
#         f_states = [rb_state]
        
#         new_grid_states, _ = contact.apply_grid_moments(None, grid_states, f_states, None, 0.1, 0.0)
     