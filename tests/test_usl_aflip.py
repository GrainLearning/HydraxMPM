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
    cell_size = 0.25
    

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
    
    shape_map = hdx.ShapeFunctionMapping("cubic", 2)

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

    solver = hdx.USLAFLIP(
        constitutive_laws=(law,),
        couplings = (coupling,),
        alpha=0.99,
        beta_min=0.0,
        beta_max=0.0,
        cell_size=cell_size,
    )

    solver_state = solver.create_state(mp_state)

    sim_state = hdx.SimState(
        time=0.0,
        step=0,
        dt=0.001,
        # dt=0.1,
        material_points=(mp_state,),
        constitutive_laws=(law_state,),
        solvers=(solver_state,),
        interactions=cache_states, # Solver will compute this
        grids=(grid_state,),
        forces=(None,)
    )

    @jax.jit
    def run_step(state):
        return solver.step(state)


    new_state = run_step(sim_state)

    new_grid_state = new_state.grids[0]

 

