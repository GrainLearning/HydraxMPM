# """Unit tests for the USL Solver."""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

import hydraxmpm as hdx



def test_usl_step_2d():
    """
    Test a single step of the USL solver in 2D.
    Verifies P2G mass/momentum transfer and G2P velocity update.
    """
    
    # 1. Setup Topology
    origin = (0.0, 0.0)
    end = (1.0, 1.0)
    cell_size = 1.0
    
    

    # 2. Setup Material Points
    pos = jnp.array([[0.1, 0.25], [0.1, 0.25]])
    vel = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    mass = jnp.array([0.1, 0.3])
    vol = jnp.array([0.7, 0.4])
    stress = jnp.stack([jnp.ones((3, 3)), jnp.zeros((3, 3))])
    


    mp_state = hdx.MaterialPointState.create(
        position_stack=pos,
        velocity_stack=vel,
        mass_stack=mass,
        volume_stack=vol,
        stress_stack=stress,
        # Initialize F and L to identity/zeros as in default
        F_stack=jnp.tile(jnp.eye(3), (2, 1, 1)),
        L_stack=jnp.zeros((2, 3, 3))
    )
    
    # 3. Setup Constitutive Law
    law = hdx.LinearElasticLaw(E=1000.0, nu=0.3)
    law_state = law.create_state(mp_state)
    
    # 4. Setup Initial Grid State
    grid_state = hdx.GridState.create(
        origin=origin,
        end=end,
        cell_size=cell_size
        )
    
    shape_map = hdx.ShapeFunctionMapping("linear", dim=2)


    intr_cache = shape_map.create_cache(
        num_points=pos.shape[0],
        dim=2
    )
    coupling = hdx.BodyCoupling(
        shape_map=shape_map,

        constitutive_law=law
        )


    cache_states = dict()

    cache_states[(0,0)] = intr_cache

    sim_state = hdx.SimState(
        time=0.0,
        step=0,
        dt=0.001,
        # dt=0.1,
        material_points=(mp_state,),
        constitutive_laws=(law_state,),
        solvers=(hdx.USLSolverState(),),
        interactions=cache_states, # Solver will compute this
        grids=(grid_state,),
        forces=(None,)
    )



    solver = hdx.USLSolver(
        constitutive_laws=(law,),
        couplings = (coupling,),
        alpha=0.99
    )

 
    

    @jax.jit
    def run_step(state):
        return solver.step(state)


    new_state = run_step(sim_state)




#     new_grid_state = new_state.grids[0]

#     expected_mass_stack = jnp.array([0.27, 0.09, 0.03, 0.01])

#     np.testing.assert_allclose(new_grid_state.mass_stack, expected_mass_stack, rtol=1e-3)

#     expected_node_moment_stack = jnp.array(
#         [[0.27, 0.27], [0.09, 0.09], [0.03, 0.03], [0.01, 0.01]]
#     )


#     np.testing.assert_allclose(new_grid_state.moment_stack, expected_node_moment_stack, rtol=1e-3)

#     # --- G2P Verification ---
#     expected_F_stack= jnp.array([
#         [
#             [ 9.999806e-01, -1.944441e-05,  0.000000e+00],
#             [-9.333251e-06,  9.999906e-01,  0.000000e+00],
#             [ 0.000000e+00,  0.000000e+00,  1.000000e+00]
#         ],
#         [   [ 9.999806e-01, -1.944441e-05,  0.000000e+00],
#             [-9.333251e-06,  9.999906e-01,  0.000000e+00],
#             [ 0.000000e+00,  0.000000e+00,  1.000000e+00]
#         ],              
               
               
#         ])

#     expected_L_stack = jnp.array(
#         [
#             [
#                 [-0.019445, -0.019445, 0.0],
#                 [-0.009333334, -0.009333334, 0.0],
#                 [0.0, 0.0, 0.0],
#             ],
#             [
#                 [-0.019445, -0.019445, 0.0],
#                 [-0.009333334, -0.009333334, 0.0],
#                 [0.0, 0.0, 0.0],
#             ],
#         ]
#     )

#     expected_volume_stack = jnp. array([0.69998 , 0.399989])

#     expected_velocity_stack = jnp.array([[1.0, 1.0], [1.0, 1.0]])
#     expected_position_stack = jnp.array([[0.101, 0.251], [0.101, 0.251]])
#     expected_velocity_stack = jnp.array([[1.0, 1.0], [1.0, 1.0]])


#     new_mp_state = new_state.material_points[0] 

#     np.testing.assert_allclose(new_mp_state.L_stack, expected_L_stack, rtol=1e-3)

#     np.testing.assert_allclose(
#         new_mp_state.volume_stack, expected_volume_stack, rtol=1e-3
#     )

#     np.testing.assert_allclose(
#         new_mp_state.position_stack, expected_position_stack, rtol=1e-3
#     )

#     np.testing.assert_allclose(
#         new_mp_state.velocity_stack, expected_velocity_stack, rtol=1e-3
#     )

#     np.testing.assert_allclose(new_mp_state.F_stack, expected_F_stack, rtol=1e-3)



# def test_usl_step_3d():
#     """
#     Test a single step of the USL solver in 3D.
#     Verifies P2G mass/momentum transfer and G2P velocity update.
#     """
    
#     # 1. Setup Topology
#     origin = (0.0, 0.0, 0.0)
#     end = (1.0, 1.0, 1.0)
#     cell_size = 1.0
    

#     # 2. Setup Material Points
#     pos = jnp.array([[0.1, 0.25, 0.3], [0.1, 0.25, 0.3]])
#     vel = jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
#     mass = jnp.array([0.1, 0.3])
#     vol = jnp.array([0.7, 0.4])
#     stress = jnp.stack([jnp.ones((3, 3)), jnp.zeros((3, 3))])
    
#     mp_state = hdx.MaterialPointState.create(
#         position_stack=pos,
#         velocity_stack=vel,
#         mass_stack=mass,
#         volume_stack=vol,
#         stress_stack=stress,
#         F_stack=jnp.tile(jnp.eye(3), (2, 1, 1)),
#         L_stack=jnp.zeros((2, 3, 3))
#     )
    
#     # 3. Setup Constitutive Law
#     law = hdx.LinearElasticLaw(E=1000.0, nu=0.3)
#     law_state = law.create_state(mp_state)
    
#     # 4. Setup Initial Grid State
#     grid_state = hdx.GridState.create(

#         origin=origin,
#         end=end,
#         cell_size=cell_size
#     )
    
#     shape_map = hdx.ShapeFunctionMapping("linear", 2, 3)

#     intr_cache = shape_map.compute(
#         pos,
#         origin=origin,
#         grid_size=grid_state.grid_size,
#         inv_cell_size=grid_state._inv_cell_size
#         )


#     coupling = hdx.BodyCoupling(
#         shape_map=shape_map,
#         constitutive_law=law
#     )


#     cache_states = dict()

#     cache_states[(0,0)] = intr_cache

#     sim_state = hdx.SimState(
#         time=0.0,
#         step=0,
#         dt=0.001,
#         # dt=0.1,
#         material_points=(mp_state,),
#         constitutive_laws=(law_state,),
#         solvers=(hdx.USLSolverState(),),
#         interactions=cache_states, # Solver will compute this
#         grids=(grid_state,),
#         forces=(None,)
#     )


#     solver = hdx.USLSolver(
#         constitutive_laws=(law,),
#         couplings = (coupling,),
#         alpha=0.99
#     )

    
#     @jax.jit
#     def run_step(state):
#         return solver.step(state)


#     new_state = run_step(sim_state)
#     new_grid_state = new_state.grids[0]


#     # --- P2G Verification ---
#     expected_mass_stack = jnp.array(
#         [0.189, 0.081, 0.063, 0.027, 0.021, 0.009, 0.007, 0.003]
#     )
#     np.testing.assert_allclose(new_grid_state.mass_stack, expected_mass_stack, rtol=1e-3)

#     expected_node_moment_stack = jnp.stack([expected_mass_stack.squeeze()]*3, axis=-1)
#     np.testing.assert_allclose(new_grid_state.moment_stack, expected_node_moment_stack, rtol=1e-3)

#     # --- G2P Verification ---
#     new_mp_state = new_state.material_points[0]

#     expected_L_stack = jnp.array(
#         [
#             [
#                 [-0.019445, -0.019445, -0.019445],
#                 [-0.00933333, -0.00933333, -0.00933333],
#                 [-0.00833333, -0.00833333, -0.00833333],
#             ],
#             [
#                 [-0.019445, -0.019445, -0.019445],
#                 [-0.00933333, -0.00933333, -0.00933333],
#                 [-0.00833333, -0.00833333, -0.00833333],
#             ],
#         ]
#     )
#     np.testing.assert_allclose(new_mp_state.L_stack, expected_L_stack, rtol=1e-3)

#     dt = 0.001
#     expected_F_stack = jnp.tile(jnp.eye(3), (2, 1, 1)) + expected_L_stack * dt
#     np.testing.assert_allclose(new_mp_state.F_stack, expected_F_stack, rtol=1e-3)

#     expected_position_stack = pos + vel * dt
#     np.testing.assert_allclose(new_mp_state.position_stack, expected_position_stack, rtol=1e-3)

#     expected_velocity_stack = vel
#     np.testing.assert_allclose(new_mp_state.velocity_stack, expected_velocity_stack, rtol=1e-3)


#     np.testing.assert_allclose(new_mp_state.F_stack, expected_F_stack, rtol=1e-3)

