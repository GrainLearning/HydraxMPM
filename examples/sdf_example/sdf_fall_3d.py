"""This example is coming back soon"""

import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import jax.numpy as jnp
import numpy as np

import hydraxmpm as hdx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import sys


import equinox as eqx


class HeartSDF(hdx.SDFObjectBase):
    """
    This is an example of an SDF heart Shape,
    based on Inigo Quilez's work (https://iquilezles.org/articles/distfunctions2d/).
    
    We extrude the 2D heart shape along Z to create a 3D volume.

    The 'tip' of the heart is at the local origin (0,0).
    """
    scale: float
    half_height: float

    def __init__(self, scale=1.0, height=1.0):
   
        self.scale = scale
        self.half_height = height / 2.0

    def signed_distance_local(self, state, p_local):
        
        # Normalize coordinate space

        p_xy = p_local[:2] / self.scale

        # Extract XY coordinates (Works for 2D and 3D)
        x = p_xy[0]
        y = p_xy[1]

        # 3. Apply Symmetry
        x = jnp.abs(x)

        # define Branch Logic (JAX-friendly)
        
        # Branch A (Upper Arcs) 
        dx_a = x - 0.25
        dy_a = y - 0.75
        dist_a = jnp.sqrt(dx_a**2 + dy_a**2) - jnp.sqrt(2.0)/4.0

        # Branch B: Lower Cusp

        d1 = x**2 + (y - 1.0)**2
        
        w = 0.5 * jnp.maximum(x + y, 0.0)
        d2 = (x - w)**2 + (y - w)**2
        
        dist_b = jnp.sqrt(jnp.minimum(d1, d2)) * jnp.sign(x - y)

        dist_unit = jnp.where(
            (y + x) > 1.0,
            dist_a,
            dist_b
        )

        # Rescale to world coordinates
        d_2d = dist_unit * self.scale

        # Extrude along Z for 3D
        z = p_local[2]
        d_z = jnp.abs(z) - self.half_height
        

        d_vec = jnp.array([d_2d, d_z])
        
        # Safe normal for automatic differentiation
        outside = jnp.sqrt(jnp.sum(jnp.maximum(d_vec, 0.0)**2) + 1e-12)
        
        inside = jnp.minimum(jnp.max(d_vec), 0.0)

        return outside + inside


def main():


    # >>> 1. Initial configuration <<<
    dx, dy,dz = (10.0, 10.0, 10.0) # 10x10x10 domain
    particles_per_cell = 2
    cell_size = (1 / 40.0) * dx  # 40 cells along x-axis

    # particle spacing
    spacing = cell_size / particles_per_cell

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

    
    sdf = HeartSDF(scale=3.0, height=2.0)
    # sdf = hdx.BoxSDF(size=jnp.array([1.5, 1.5]))
    # sdf = hdx.SphereSDF(radius=1.0)
    # sdf_state = sdf.create_state(center_of_mass=jnp.array([5.0, 2.0]))

    # sdf = hdx.PlaneSDF(normal=jnp.array([0.0, 1.0]))

    # sdf = hdx.CapsuleSDF(radius=0.5, height=2.0)

    sdf_state = sdf.create_state(
        center_of_mass=jnp.array([5.0, 1.0, 5.0]),
        # angular_velocity=angular_velocity,
        # rotation=30.0 * jnp.pi / 180.0
        )

    # sdf = hdx.CylinderSDF(radius=1.0, height=2.0)
    # sdf_state = sdf.create_state(center_of_mass=jnp.array([5.0, 2.0]))

    # sdf = hdx.TorusSDF(major_r=1.0, minor_r=0.3)
    # sdf_state = sdf.create_state(center_of_mass=jnp.array([5.0, 2.0]))

    # sdf = hdx.HollowCylinderSDF(height=2.5, outer_radius=1.0, inner_radius=0.75, is_2d_ring=True)
    # sdf_state = sdf.create_state(center_of_mass=jnp.array([5.0, 2.0]))

    box = generate_particles_in_sdf(
        sdf_obj = sdf, 
        sdf_state = sdf_state,
        bounds_min=jnp.array([0, 0, 0.]),
        bounds_max=jnp.array([dx,dy,dz]),
        cell_size=cell_size,
        ppc=particles_per_cell,
        mode="regular"
    )

    # def generate_particles_in_sdf(
    #         *,
    #         sdf_obj, 
    #         bounds_min= None,
    #         bounds_max = None,
    #         num_particles= None, # For 'random' mode
    #         sdf_state = None, 
    #         key = None,
    #         center_of_mass = None,
    #         rotation = None,
    #         shapefunction = "cubic",
    #         mode = "uniform",
    #         cell_size = None, # For 'regular'/'gauss_legendre' modes
    #         ppc = None # For 'regular'/'gauss_legendre' modes
    #         ):


    # Stack all the positions together
    water_pos = jnp.vstack([block1, block2])
    # jelly_pos = jnp.vstack([block3])
    jelly_pos = jnp.vstack([box])



    sim_builder = hdx.StandardSimBuilder()

    
    density_jelly = jnp.full((len(jelly_pos),), 12000.0)  # kg/m^3

    jelly_law = hdx.LinearElasticLaw(E=1e7, nu=0.2)

    sim_builder = sim_builder.add_constitutive_law(
            jelly_law
    )



    density_water = jnp.full((len(water_pos),), 1000.0)  # kg/m^3
    water = hdx.NewtonFluid(
        K = 2e6, 
        viscosity = 1e-3, 
        beta = 7.0
        )

    water_state = water.create_state_from_density(
        density_stack = density_water
    )

    sim_builder = sim_builder.add_constitutive_law(water,water_state)


    # >>> Add grid <<<
  # g 1 jelly
    sim_builder.add_grid(origin=(0.0, 0.0, 0.0), end=(dx, dy,dz), cell_size=cell_size,padding=2) 

    # g 0 water
    sim_builder.add_grid(origin=(0.0, 0.0, 0.0), end=(dx, dy,dz), cell_size=cell_size,padding=2)

  
    # >>> 2. add material points <<<

    # p 0 water
    sim_builder = sim_builder.add_material_points(
        position_stack=jelly_pos,
        density=density_jelly,  # kg/m^3
        cell_size=cell_size,
        ppc=particles_per_cell,
    )

    sim_builder = sim_builder.add_material_points(
        position_stack=water_pos,
        density=density_water,  # kg/m^3
        cell_size=cell_size,
        ppc=particles_per_cell,
    )


    # >>> 4. couple grid, material points, constitutive law<<<
    sim_builder = sim_builder.couple(
        shapefunction="quadratic", p_idx=0, c_idx=0, g_idx=0,s_idx=0
        )

    sim_builder = sim_builder.couple(
        shapefunction="quadratic",p_idx=1,c_idx=1,g_idx=1,s_idx=1
        )

    # >>> 5. add forces<<<
    # act on all grids (or particles by default)
    sim_builder = sim_builder.add_boundary(friction=0.0)

    sim_builder = sim_builder.add_gravity(gravity=jnp.array([0.0, -9.8, 0.0]))

    # # contact need to explicitly specify grids
    sim_builder = sim_builder.add_couple_contact(
        couple_idx_actor = 0, # jelly
        couple_idx_receiver = 1, # water
        friction = 0.0,
        is_rigid = True
    )


    # >>> 6. add forces<<<
    sim_builder = sim_builder.add_solver(
        
        scheme="usl_aflip",alpha=0.90,
        small_mass_cutoff=1e-4
        )



    vis = hdx.RerunVisualizer(
        origin=(0.0, 0.0, 0.0),
        end=(dx, dy,dz),
        cell_size=cell_size,
        ppc=particles_per_cell
    )

    vis.log_sdf_boundary(sdf, sdf_state)
    


    # vis.log_sdf_boundary(sdf, sdf_state)

    dt = 1.0e-3
    total_steps = int(4.3/dt)
    output_step= (0.01/dt)


    mpm_solver,sim_state = sim_builder.build(
        dt=dt
    )


    devices = jax.devices()
    num_devices = len(devices)
    print(f"🚀 Running on {num_devices} CPU Devices")

    # if num_devices > 1:
    #     # Define Sharding Strategy
    #     # Particles are split (Sharded) along axis 0
    #     # mesh = Mesh(devices, axis_names=('ensemble',))
    #     # sharding_particles = NamedSharding(mesh, P('ensemble', None))
    #     # sharding_replicated = NamedSharding(mesh, P()) # Fully replicated
        

    #     mesh = Mesh(devices, axis_names=('ensemble',))
    #     sharding_particles = NamedSharding(mesh, P('ensemble', None))
    #     sharding_particles_1d = NamedSharding(mesh, P('ensemble')) # Add 1D sharding spec
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
    #                     if leaf.ndim == 1:
    #                         return jax.device_put(leaf, sharding_particles_1d)
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
