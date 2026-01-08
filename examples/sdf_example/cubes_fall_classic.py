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
    block1 = create_block((1, 1), 2, spacing)
    block2 = create_block((7.5, 6.3), 2, spacing)
    block3 = create_block((3.5, 0.1), 2, spacing)
    block4 = create_block((5, 3.8), 2, spacing)

    # Stack all the positions together
    water_pos = jnp.vstack([block1, block2])
    jelly_pos = jnp.vstack([block3])



    sim_builder = hdx.StandardSimBuilder()

    # Create Newtonian Fluid Constitutive Law
    density_water = jnp.full((len(water_pos),), 1000.0)  # kg/m^3
    water = hdx.NewtonFluid(
        K = 2e6, 
        viscosity = 1e-3, 
        beta = 7.0
        )

    water_state = water.create_state_from_density(
        density_stack = density_water
    )

    sim_builder = sim_builder.add_constitutive_law(
        water,
        water_state
    )

    density_jelly = jnp.full((len(jelly_pos),), 1200.0)  # kg/m^3
    jelly_law = hdx.LinearElasticLaw(E=1e6, nu=0.2)

    sim_builder = sim_builder.add_constitutive_law(
            jelly_law
        )


    # >>> Add grid <<<

    # g 0 water
    sim_builder.add_grid(origin=(0.0, 0.0), end=(dx, dy), cell_size=cell_size)

    # g 1 jelly
    sim_builder.add_grid(origin=(0.0, 0.0), end=(dx, dy), cell_size=cell_size) 

    # >>> 2. add material points <<<

    # p 0 water
    sim_builder = sim_builder.add_material_points(
        position_stack=water_pos,
        density=density_water,  # kg/m^3
        cell_size=cell_size,
        ppc=particles_per_cell,
    )

    sim_builder = sim_builder.add_material_points(
        position_stack=jelly_pos,
        density=density_jelly,  # kg/m^3
        cell_size=cell_size,
        ppc=particles_per_cell,
    )

    # >>> 4. couple grid, material points, constitutive law<<<
    sim_builder = sim_builder.couple(
        shapefunction="quadratic", p_idx=0, c_idx=0, g_idx=0,s_idx=1
        )

    sim_builder = sim_builder.couple(
        shapefunction="quadratic",p_idx=1,c_idx=1,g_idx=1,s_idx=1
        )

    # >>> 5. add forces<<<
    # act on all grids (or particles by default)
    sim_builder = sim_builder.add_boundary(friction=0.0)

    sim_builder = sim_builder.add_gravity(gravity=jnp.array([0.0, -9.8]))

    # # contact need to explicitly specify grids
    sim_builder = sim_builder.add_grid_contact(
        g_master = 1,
        g_slave = 0,
        friction = 0.0
    )


    # >>> 6. add forces<<<
    sim_builder = sim_builder.add_solver(scheme="usl_aflip",alpha=0.90)

# #     # devices = jax.devices()
# #     # num_devices = len(devices)
# #     # print(f"🚀 Running on {num_devices} CPU Devices")

# #     # if num_devices > 1:
# #     #     # Define Sharding Strategy
# #     #     # Particles are split (Sharded) along axis 0
# #     #     # mesh = Mesh(devices, axis_names=('ensemble',))
# #     #     # sharding_particles = NamedSharding(mesh, P('ensemble', None))
# #     #     # sharding_replicated = NamedSharding(mesh, P()) # Fully replicated
        

# #     #     mesh = Mesh(devices, axis_names=('ensemble',))
# #     #     sharding_particles = NamedSharding(mesh, P('ensemble', None))
# #     #     sharding_particles_1d = NamedSharding(mesh, P('ensemble')) # Add 1D sharding spec
# #     #     sharding_replicated = NamedSharding(mesh, P()) # Fully replicated
        
# #     #     def distribute_state(state: hdx.SimState):
# #     #             """
# #     #             Intelligently distributes the SimState.
# #     #             Particles -> Sharded across devices.
# #     #             Grids -> Replicated on all devices.
# #     #             """
                
# #     #             def _sharding_decision(leaf):
# #     #                 if not eqx.is_array(leaf):
# #     #                     return leaf
                    
# #     #                 is_particle_array = (leaf.shape[0] == len(water_position_stack)) and (leaf.ndim > 0)
                    
# #     #                 if is_particle_array:
# #     #                     if leaf.ndim == 1:
# #     #                         return jax.device_put(leaf, sharding_particles_1d)
# #     #                     return jax.device_put(leaf, sharding_particles)
# #     #                 else:
# #     #                     return jax.device_put(leaf, sharding_replicated)

# #     #             # Apply the distribution
# #     #             distributed_state = jax.tree.map(_sharding_decision, state)
# #     #             return distributed_state
# #     #     # Apply transformations to SimState
# #     #     # 1. Shard Particles
# #     #     print("⏳ Distributing state across devices...")
# #     #     sim_state = distribute_state(sim_state)
# #     #     print("✅ State distributed.")
    


    vis = hdx.RerunVisualizer(
        origin=(0.0, 0.0),
        end=(dx, dy),
        cell_size=cell_size,
        ppc=particles_per_cell
    )

    dt = 1e-4
    total_steps = int(4.3/dt)
    output_step= (0.01/dt)

    total_steps = int(4.3/0.001)

    mpm_solver,sim_state = sim_builder.build(
        dt=dt
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
