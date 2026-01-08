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
    A Heart Shape.
    Based on Inigo Quilez's exact Heart SDF.
    
    Domain: 2D (Shape) or 3D (Infinite Column along Z)
    The 'tip' of the heart is at the local origin (0,0).
    """
    scale: float
    half_height: float

    def __init__(self, scale=1.0, height=1.0):
        # Determine dim based on usage, but typically SDFs are dim-agnostic 
        # in this architecture if we handle slicing inside signed_distance_local
        self.scale = scale
        self.half_height = height / 2.0

    def signed_distance_local(self, state, p_local):
        
        # 1. Normalize coordinate space
        # We divide by scale to perform math in "Unit Heart" space
        # Then multiply result by scale at the end.
        # p = p_local / self.scale
        p_xy = p_local[:2] / self.scale

        # 2. Extract XY coordinates (Works for 2D and 3D)
        # In 3D, this creates an infinite heart-shaped column along Z
        x = p_xy[0]
        y = p_xy[1]

        # 3. Apply Symmetry
        x = jnp.abs(x)

        # 4. Define Branch Logic (JAX-friendly)
        
        # Branch A: y + x > 1.0 (Upper Arcs)
        # return sqrt(dot2(p - vec2(0.25, 0.75))) - sqrt(2.0)/4.0;
        dx_a = x - 0.25
        dy_a = y - 0.75
        dist_a = jnp.sqrt(dx_a**2 + dy_a**2) - jnp.sqrt(2.0)/4.0

        # Branch B: Lower Cusp
        # return sqrt(min(dot2(p - vec2(0.00, 1.00)),
        #                 dot2(p - 0.5*max(x+y, 0.0)))) * sign(x-y);
        
        # Term 1: dot2(p - vec2(0.0, 1.0))
        d1 = x**2 + (y - 1.0)**2
        
        # Term 2: dot2(p - 0.5*max(x+y, 0.0))
        # Note: GLSL `p - scalar` implies `vec2(p.x - s, p.y - s)`
        w = 0.5 * jnp.maximum(x + y, 0.0)
        d2 = (x - w)**2 + (y - w)**2
        
        dist_b = jnp.sqrt(jnp.minimum(d1, d2)) * jnp.sign(x - y)

        # 5. Combine using Select (Branchless)
        # if (y + x > 1.0)
        dist_unit = jnp.where(
            (y + x) > 1.0,
            dist_a,
            dist_b
        )
        d_2d = dist_unit * self.scale
        # 6. Extrude along Z for 3D
        z = p_local[2]
        d_z = jnp.abs(z) - self.half_height
        
        # Combine 2D shape distance and Z height distance
        # This is the standard "Box/Extrusion" logic:
        # d = length(max(vec2(d_2d, d_z), 0.0)) + min(max(d_2d, d_z), 0.0)
        
        d_vec = jnp.array([d_2d, d_z])
        
        # Safe Norm for Outside (prevent NaN gradient at 0)
        outside = jnp.sqrt(jnp.sum(jnp.maximum(d_vec, 0.0)**2) + 1e-12)
        
        # Inside (Negative)
        inside = jnp.minimum(jnp.max(d_vec), 0.0)
        # 6. Re-scale distance back to world units
        return outside + inside

def generate_particles_in_sdf(
        *,
        sdf_obj, 
        bounds_min= None,
        bounds_max = None,
        num_particles= None, # For 'random' mode
        sdf_state = None, 
        key = None,
        center_of_mass = None,
        rotation = None,
        shapefunction = "cubic",
        mode = "uniform",
        cell_size = None, # For 'regular'/'gauss_legendre' modes
        ppc = None # For 'regular'/'gauss_legendre' modes
        ):
    """
    Generates particles inside the SDF using Rejection Sampling.

    mode options: 
    "rejection" - rejection sampling monte carlo
    "regular" - uniform grid sampling (requires cell_size and ppc)
    "gauss_legendre - gauss legendre quadrature sampling (requires cell_size and ppc)
    """
    dim = bounds_min.shape[0]

    if sdf_state is None:
        if center_of_mass is None:
            center_of_mass = (bounds_min + bounds_max) / 2.0
        sdf_state = sdf_obj.create_state(center_of_mass=center_of_mass, rotation=rotation)
    


    candidate_point_stack = None
    if mode == "random":
        if num_particles is None or key is None:
            raise ValueError("Mode 'random' requires 'num_particles' and 'key'.")
        
        key, subkey = jax.random.split(key)
        candidate_point_stack = jax.random.uniform(
            subkey, 
            (num_particles, dim), 
            minval=bounds_min, 
            maxval=bounds_max
        )
    elif mode in ["regular", "gauss_legendre"]:
        # integer grid indices that cover the bounding box
        start_idx = jnp.floor(bounds_min / cell_size).astype(int)
        end_idx = jnp.ceil(bounds_max / cell_size).astype(int)

        # grid coordinates in grid index space
        id_ranges = [jnp.arange(s, e) for s, e in zip(start_idx, end_idx)]
        meshes = jnp.meshgrid(*id_ranges, indexing='ij')

        grid_origins_flat = jnp.stack(meshes, axis=-1).reshape(-1, dim) * cell_size
        offsets = None

        if mode == "regular":
            # Linear subdivision: e.g., ppc=4 (2D) -> 2x2 grid inside cell
            pts_per_dim = int(np.ceil(ppc ** (1/dim)))

            # Linspace from 0 to 1, shifted to center points
            # e.g., 2 points -> [0.25, 0.75]
            lin = jnp.linspace(0, 1, pts_per_dim * 2 + 1)[1::2]

            offset_meshes = jnp.meshgrid(*([lin] * dim), indexing='ij')
            offsets = jnp.stack(offset_meshes, axis=-1).reshape(-1, dim)
        

        # C. Broadcast Add
        # grid_origins: (N_cells, 1, dim)
        # offsets:      (1, N_offsets, dim)
        # Result:       (N_cells * N_offsets, dim)
        candidate_point_stack = (grid_origins_flat[:, None, :] + offsets[None, :, :]).reshape(-1, dim)

              # WAIT: offsets above were 0..1 relative. Need to scale by cell_size.
        # Correct logic:
        # candidate_point_stack = (grid_origins_flat[:, None, :] + (offsets[None, :, :] * cell_size)).reshape(-1, dim)

    # 2. Check SDF
    # < 0 means inside
    dists = sdf_obj.get_signed_distance_stack(sdf_state, candidate_point_stack)
    mask = dists < 0.0
    
    # 3. Filter (This part usually requires boolean masking which changes shapes)
    # In JIT, this is tricky. Two options:
    # A. Return fixed size with valid_mask
    # B. Use BBox sampling if shape is simple
    
    # For now, simplistic Boolean indices (Non-JIT safe usually, runs on CPU init)
    valid_points = candidate_point_stack[mask]

    return valid_points

def main():


    # >>> 1. Initial configuration <<<
    dx, dy,dz = (10.0, 10.0, 10.0) # 10x10x10 domain
    particles_per_cell = 2
    cell_size = (1 / 40.0) * dx  # 40 cells along x-axis

    # particle spacing
    spacing = cell_size / particles_per_cell

    # def create_block(block_start, block_size, spacing):
    #     """Create a block of particles in 3D space."""
    #     block_end = (
    #         block_start[0] + block_size,
    #         block_start[1] + block_size,
    #         block_start[2   ] + block_size,
    #     )
    #     x = np.arange(block_start[0], block_end[0], spacing)
    #     y = np.arange(block_start[1], block_end[1], spacing)
    #     z = np.arange(block_start[2], block_end[2], spacing)
    #     block = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
    #     return block

    # # Create two blocks (cubes in 2D context)
    # block1 = create_block((1, 3, 1), 2, spacing)
    # block2 = create_block((7.5, 3,1.0), 2, spacing)
    # block3 = create_block((2.8, 7,3.0), 2, spacing)
    # block4 = create_block((5, 3.8,4), 2, spacing)





    sdf = hdx.SphereSDF(radius=2.0)
    water_pos1 = generate_particles_in_sdf(
        sdf_obj = sdf, 
        center_of_mass=jnp.array([3.5, 6.0, 3.5]),
        bounds_min=jnp.array([0, 0, 0.]),
        bounds_max=jnp.array([dx,dy,dz]),
        cell_size=cell_size,
        ppc=particles_per_cell,
        mode="regular"
    )

    jelly_pos1 = generate_particles_in_sdf(
        sdf_obj = sdf, 
        center_of_mass=jnp.array([4.5, 6.0, 7.5]),
        bounds_min=jnp.array([0, 0, 0.]),
        bounds_max=jnp.array([dx,dy,dz]),
        cell_size=cell_size,
        ppc=particles_per_cell,
        mode="regular"
    )
    
    sdf = HeartSDF(scale=12.0, height=20.0)


    # sdf = hdx.BoxSDF(size=jnp.array([1.5, 1.5]))
    # sdf_state = sdf.create_state(center_of_mass=jnp.array([5.0, 2.0]))

    # sdf = hdx.PlaneSDF(normal=jnp.array([0.0, 1.0]))

    # sdf = hdx.CapsuleSDF(radius=0.5, height=2.0)

    sdf_state = sdf.create_state(
        center_of_mass=jnp.array([5.0, -9.0, 5.0]),
        # angular_velocity=angular_velocity,
        # rotation=30.0 * jnp.pi / 180.0
        )

    # sdf = hdx.CylinderSDF(radius=1.0, height=2.0)
    # sdf_state = sdf.create_state(center_of_mass=jnp.array([5.0, 2.0]))

    # sdf = hdx.TorusSDF(major_r=1.0, minor_r=0.3)
    # sdf_state = sdf.create_state(center_of_mass=jnp.array([5.0, 2.0]))

    # sdf = hdx.HollowCylinderSDF(height=2.5, outer_radius=1.0, inner_radius=0.75, is_2d_ring=True)
    # sdf_state = sdf.create_state(center_of_mass=jnp.array([5.0, 2.0]))

    # box = generate_particles_in_sdf(
    #     sdf_obj = sdf, 
    #     sdf_state = sdf_state,
    #     bounds_min=jnp.array([0, 0, 0.]),
    #     bounds_max=jnp.array([dx,dy,dz]),
    #     cell_size=cell_size,
    #     ppc=particles_per_cell,
    #     mode="regular"
    # )

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
    water_pos = jnp.vstack([water_pos1])
    # jelly_pos = jnp.vstack([block3])
    jelly_pos = jnp.vstack([jelly_pos1])



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

    sim_builder = sim_builder.add_sdf_collider(
        sdf_logic=sdf,
        sdf_state=sdf_state,
        friction=0.5,
        gap = 1e-2

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
