"""This example is coming back soon"""


import jax
import jax.numpy as jnp
import numpy as np

import hydraxmpm as hdx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import sys
import os

import equinox as eqx
# Set this to the number of threads/cores you want to use as "devices"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def main():


    # >>> 1. Initial configuration <<<
    dx, dy = (10.0, 10.0) # 10x10 domain

    particles_per_cell = 2
    cell_size = (1 / 80.0) * dx  # 80 cells along x-axis

    # particle spacing
    spacing = cell_size / particles_per_cell

    def create_block(block_start, block_size, spacing):
        """Create a block of particles in 2D space."""
        block_end = (block_start[0] + block_size, block_start[1] + block_size)
        x = np.arange(block_start[0], block_end[0], spacing)
        y = np.arange(block_start[1], block_end[1], spacing)
        block = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
        return block


    # Create two blocks (cubes in 2D context)
    block1 = create_block((3, 1/4), 2, spacing)
    block2 = create_block((7.5, 6.3/4), 2, spacing)
    # block4 = create_block((5, 3.8/4), 2, spacing)

    # block3 = create_block((2.8, 7/4), 2, spacing)




    # Stack all the positions together
    # setup normally considered clay
    clay_pos = jnp.vstack([block1])

    mcc = hdx.ModifiedCamClay(
        nu=0.3,
        M=0.9,
        lam=0.2,
        kap=0.05,
        N=2.0,
        p_ref=1000.0
    )



    p_stack = jnp.full(
        (clay_pos.shape[0],),
        10_000.0 # kPa 
        )
        
    mcc_state, stress_ref_stack,density_stack = mcc.create_state_from_ocr(
        p_stack=p_stack,
        ocr_stack =1.0
    )

    print("Density stack:", density_stack.mean())
    print("Stress ref stack:", stress_ref_stack.mean(axis=0))
  
    
    print("Density stack:", density_stack.mean())


    sim_builder = hdx.StandardSimBuilder()

    sim_builder = sim_builder.add_constitutive_law(
        mcc,
        law_state = mcc_state
    )

    
    # g 0 clay
    sim_builder.add_grid(origin=(0.0, 0.0), end=(dx, dy), cell_size=cell_size)

    
    # p 0 clay
    sim_builder = sim_builder.add_material_points(
        position_stack=clay_pos,
        stress_stack=stress_ref_stack,
        density=density_stack,  # kg/m^3
        cell_size=cell_size,
        ppc=particles_per_cell,
    )
 
    # # c 1 jelly

    
    # >>> 4. couple grid, material points, constitutive law<<<
    sim_builder = sim_builder.couple(
        shapefunction="quadratic", p_idx=0, c_idx=0, g_idx=0
        )


    # >>> 5. add forces<<<
    # act on all grids (or particles by default)
    sim_builder = sim_builder.add_boundary(friction=0.7)

    sim_builder = sim_builder.add_gravity(gravity=jnp.array([0.0, -9.8]))


    # >>> 6. add forces<<<
    sim_builder = sim_builder.add_solver(scheme="usl_aflip",alpha=0.95, use_mls_update = True)


    dt = 1e-4
    total_steps = int(4.3/dt)
    output_step= (0.01/dt)

    mpm_solver,sim_state = sim_builder.build(
        dt=dt
    )

    vis = hdx.RerunVisualizer(
        origin=(0.0, 0.0),
        end=(dx, dy),
        cell_size=cell_size,
        ppc=particles_per_cell
        
        )

    def log_simulation(sim_state: hdx.SimState):
        vis.log_simulation(sim_state)


    def loop_body(i, sim_state):
        sim_state = mpm_solver.step(sim_state)

        jax.lax.cond(
            i % output_step == 0,
            lambda s: jax.debug.callback(log_simulation, s),
            lambda s: None,
            sim_state
        )
        return sim_state

    @jax.jit
    def run_sim(sim_state: hdx.SimState):
        sim_state = jax.lax.fori_loop(0, total_steps, loop_body, sim_state)
        return sim_state

    sim_state = run_sim(sim_state)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Cleaning up...")
        
        # 1. Force JAX to clear backend caches (helps with VRAM)
        jax.clear_caches()
        
        # 2. Explicitly exit
        sys.exit(0)
