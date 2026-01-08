"""This example is coming back soon"""

import os
# Set this to the number of threads/cores you want to use as "devices"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np

import hydraxmpm as hdx

import sys
import os

import equinox as eqx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

def main():
    # >>> 1. Initial configuration <<<
    dx, dy,dz = (10.0, 10.0, 10.0) # 10x10x10 domain

    particles_per_cell = 2
    cell_size = (1 / 80.0) * dx  # 80 cells along x-axis

    # particle spacing
    spacing = cell_size / particles_per_cell



    print("Creating simulation")


    def create_block(block_start, block_size, spacing):
        """Create a block of particles in 3D space."""
        block_end = (
            block_start[0] + block_size,
            block_start[1] + block_size,
            block_start[2   ] + block_size,
        )
        x = np.arange(block_start[0], block_end[0], spacing)
        y = np.arange(block_start[1], block_end[1], spacing)
        z = np.arange(block_start[2], block_end[2], spacing)
        block = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
        return block

    # Create two blocks (cubes in 2D context)
    block1 = create_block((1, 1, 1), 2, spacing)
    block2 = create_block((7.5, 6.3,1.0), 2, spacing)
    block3 = create_block((2.8, 7,3.0), 2, spacing)
    block4 = create_block((5, 3.8,4), 2, spacing)

#     # Stack all the positions together
    water_pos = jnp.vstack([block1, block2, block4])

    sim_builder = hdx.StandardSimBuilder()

    # >>> 1. add grids <<<

    # g 0 water
    sim_builder.add_grid(origin=(0.0, 0.0, 0.0), end=(dx, dy, dz), cell_size=cell_size)

    # >>> 2. add material points <<<

    # p 0 water
    sim_builder = sim_builder.add_material_points(
        position_stack=water_pos,
        velocity_stack=jnp.zeros_like(water_pos),
        density=1000.0,  # kg/m^3
        cell_size=cell_size,
        ppc=particles_per_cell,
    )
    # >>> 3. add constitutive laws <<<

    # c 0 water
    sim_builder = sim_builder.add_constitutive_law(
        hdx.NewtonFluid(
        K = 2e6, 
        viscosity = 1e-3, 
        beta = 7.0
        )
    )

    # >>> 4. couple grid, material points, constitutive law<<<
    sim_builder = sim_builder.couple(shapefunction="quadratic")




    # >>> 5. add forces<<<
    # act on all grids (or particles by default)
    sim_builder = sim_builder.add_boundary(friction=0.0)

    sim_builder = sim_builder.add_gravity(gravity=jnp.array([0.0, -9.8,0.0]))

    # >>> 6. add forces<<<
    sim_builder = sim_builder.add_solver(
        scheme="usl",
        alpha=0.9
    )


    mpm_solver,sim_state = sim_builder.build()

    
    # vis = hdx.RerunVisualizer(
    #     grid_topology=mpm_solver.couplings[0].grid_topology,
    #     ppc=particles_per_cell,
    #     )
    vtk_io = hdx.VTKVisualizer(
        output_dir="output_vtk/cube_fall",
        grid_topology=mpm_solver.couplings[0].grid_topology,
        ppc=particles_per_cell,
        relative_dir=__file__ 
    )

    # devices = jax.devices()
    # num_devices = len(devices)
    # print(f"🚀 Running on {num_devices} CPU Devices")

    # if num_devices > 1:


    #     mesh = Mesh(devices, axis_names=('ensemble',))
    #     sharding_particles = NamedSharding(mesh, P('ensemble', None))
   
    #     sharding_replicated = NamedSharding(mesh, P()) # Fully replicated
        
    #     def distribute_state(state: hdx.SimState):
    #             """
    #             Intelligently distributes the SimState.
    #             Particles -> Sharded across devices.
    #             Grids -> Replicated on all devices.
    #             """
                
    #             def _sharding_decision(leaf):
    #                 if not eqx.is_array(leaf):
    #                     return leaf
                    
    #                 is_particle_array = (leaf.shape[0] == len(water_pos)) and (leaf.ndim > 0)
                    
    #                 if is_particle_array:
    #                     return jax.device_put(leaf, sharding_particles)
    #                 else:
    #                     return jax.device_put(leaf, sharding_replicated)

    #             # Apply the distribution
    #             distributed_state = jax.tree.map(_sharding_decision, state)
    #             return distributed_state
    #     # Apply transformations to SimState
    #     # 1. Shard Particles
    #     print("⏳ Distributing state across devices...")
    #     sim_state = distribute_state(sim_state)
    #     print("✅ State distributed.")
    

    total_steps = int(10.3/0.001)

    def log_simulation(sim_state: hdx.SimState):
        # vis.log_simulation(sim_state)
        vtk_io.log_simulation(sim_state)

        jax.debug.print("Saved step: {i}", i=sim_state.step)


    def loop_body(i, sim_state):
        sim_state = mpm_solver.step(sim_state)
        
        jax.lax.cond(
            i % 100 == 0,
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
