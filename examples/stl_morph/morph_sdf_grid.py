


import trimesh
from mesh_to_sdf import mesh_to_voxels
import numpy as np

import equinox as eqx
import jax

import hydraxmpm as hdx

import jax.numpy as jnp

from jaxtyping import Array, Float

class MChainedMorphSDF(hdx.ChainMorphSDF):
    def signed_distance_local(self, state, p_local):
        # 1. Compute all distances
        dists = jnp.stack([s.signed_distance_local(state, p_local) for s in self.shapes])
        
        T = state.blend_factor
        num_shapes = len(self.shapes)
        
        # 2. Safety Clamp for T
        # Valid range is 0.0 to (N-1)
        # We clamp slightly below N-1 to ensure floor() doesn't jump to the last index
        # But actually, the robust way is to clamp the INDEX, not T.
        T = jnp.clip(T, 0.0, num_shapes - 1.0)
        
        # 3. Robust Indexing
        # If T=2.0 (Box), we want idx=1 (Cow), t=1.0. 
        # So we blend Cow->Box with weight 1.0.
        # We ensure idx never exceeds N-2.
        idx = jnp.floor(T).astype(int)
        idx = jnp.minimum(idx, num_shapes - 2)
        
        # 4. Local t
        local_t = T - idx
        
        d_start = dists[idx]
        d_end   = dists[idx + 1]

        # 5. Log-Sum-Exp Smoothing
        k = 32.0
        t_safe = jnp.clip(local_t, 1e-5, 1.0 - 1e-5)
        
        # LogSumExp Logic
        # d = -1/k * log( (1-t)e^-kA + t*e^-kB )
        #   = -1/k * logsumexp( [log(1-t) - kA, log(t) - kB] )
        
        log_w_a = jnp.log(1.0 - t_safe)
        log_w_b = jnp.log(t_safe)
        
        exponents = jnp.stack([
            log_w_a - k * d_start,
            log_w_b - k * d_end
        ])
        
        smooth_dist = -jax.nn.logsumexp(exponents, axis=0) / k
            
        return smooth_dist


class GyroidSDF(hdx.SDFObjectBase):
    scale: float # Zoom level
    thickness: float # Wall thickness

    def __init__(self, scale=1.0, thickness=0.5):
        self.scale = scale
        self.thickness = thickness

    def signed_distance_local(self, state, p):
        p_s = p * self.scale
        # Approximation of Gyroid Surface
        val = jnp.sin(p_s[0])*jnp.cos(p_s[1]) + \
              jnp.sin(p_s[1])*jnp.cos(p_s[2]) + \
              jnp.sin(p_s[2])*jnp.cos(p_s[0])
        
        # Create solid walls
        dist = jnp.abs(val) - self.thickness
        
        # Divide by 1.5 (approx gradient norm) to correct metric for physics
        return dist / 1.5 / self.scale


class HexPrismSDF(hdx.SDFObjectBase):
    h: Float[Array, "2"] # x = Radius (width), y = Height/2

    def __init__(self, radius=1.0, height=2.0):
        self.h = jnp.array([radius, height/2.0])

    def signed_distance_local(self, state, p):
        q = jnp.abs(p)
        
        # Hexagon Logic (Fold 3 times)
        k = jnp.array([-0.8660254, 0.5, 0.57735])
        
        # Fold 1
        p_xy = jnp.array([q[0], q[1]])
        p_xy = p_xy - 2.0 * jnp.minimum(jnp.dot(p_xy, k[:2]), 0.0) * k[:2]
        
        # Fold 2
        d_xy = jnp.linalg.norm(
            p_xy - jnp.array([jnp.clip(p_xy[0], -k[2]*self.h[0], k[2]*self.h[0]), self.h[0]])
        ) * jnp.sign(p_xy[1] - self.h[0])
        
        d_z = q[2] - self.h[1]
        
        return jnp.minimum(jnp.maximum(d_xy, d_z), 0.0) + jnp.linalg.norm(jnp.maximum(jnp.array([d_xy, d_z]), 0.0))


class NullSDF(hdx.SDFObjectBase):
    """
    Represents Nothing. 
    Distance is effectively infinite everywhere.
    Particles will never collide with this.
    Useful for 'disappearing' an object during a morph sequence.
    """
    def signed_distance_local(self, state, p):
        # Return a huge positive number (Outside)
        return 1e6 
    

class DisplacedSphereSDF(hdx.SDFObjectBase):
    radius: float
    frequency: float
    amplitude: float

    def __init__(self, radius=2.0, frequency=5.0, amplitude=0.2):
        self.radius = radius
        self.frequency = frequency
        self.amplitude = amplitude

    def signed_distance_local(self, state, p):
        # Base Sphere
        d1 = jnp.linalg.norm(p) - self.radius
        
        # Displacement (Sinusoidal egg carton pattern)
        d2 = jnp.sin(self.frequency * p[0]) * \
             jnp.sin(self.frequency * p[1]) * \
             jnp.sin(self.frequency * p[2])
             
        # Add displacement
        return d1 + self.amplitude * d2



class RandomParticleCloudSDF(hdx.SDFObjectBase):
    """
    A cloud of randomly positioned spheres.
    Great for simulating: Asteroids, Gravel, Foam, or Abstract Obstacles.
    """
    local_points: Float[Array, "n_points 3"]
    radii: Float[Array, "n_points"]
    smooth_k: float = eqx.field(static=True)

    def __init__(
        self, 
        key: jax.Array, 
        count: int = 20, 
        bounds: float = 1.0, 
        min_radius: float = 0.1, 
        max_radius: float = 0.3,
        smooth_k: float = 0.0
    ):
        """
        Args:
            key: JAX PRNGKey for randomness.
            count: Number of spheres.
            bounds: Spheres will be generated in box [-bounds, bounds].
            min_radius: Smallest sphere.
            max_radius: Largest sphere.
            smooth_k: Blending factor. 
                      0.0 = Hard rocks. 
                      >0.0 (e.g. 10.0) = Organic/Blobby fusion.
        """
        k_pos, k_rad = jax.random.split(key)
        
        # 1. Random Positions
        self.local_points = jax.random.uniform(
            k_pos, shape=(count, 3), minval=-bounds, maxval=bounds
        )
        
        # 2. Random Radii
        self.radii = jax.random.uniform(
            k_rad, shape=(count,), minval=min_radius, maxval=max_radius
        )
        
        self.smooth_k = float(smooth_k)

    def signed_distance_local(self, state, p):
        # 1. Vector from query point 'p' to ALL sphere centers
        # p: (3,)
        # local_points: (N, 3)
        diff = p - self.local_points
        
        # 2. Distance to surfaces
        # dists: (N,)
        dists = jnp.linalg.norm(diff, axis=-1) - self.radii
        
        # 3. Combine
        if self.smooth_k > 1e-5:
            # Smooth Minimum (Metaball blending)
            # This makes the random spheres fuse into a single organic shape
            res = -jax.nn.logsumexp(-self.smooth_k * dists) / self.smooth_k
            return res
        else:
            # Hard Minimum (Gravel / Debris)
            return jnp.min(dists)


if __name__ == "__main__":
  
    import os
    import jax.numpy as jnp
    dir_path = os.path.dirname(os.path.realpath(__file__))


    origin = jnp.array([-0.1, -0.1, -0.1])
    end = jnp.array([0.1, 0.1, 0.1])

    cell_size = ((end - origin)/64).min()

    sdf_bunny = hdx.GridSDF(f"{dir_path}/stanford-bunny.npz", thickness=0.05, scale=1.)  

    sdf_sphere = hdx.SphereSDF(radius=0.05)
    sdf_cow = hdx.GridSDF(f"{dir_path}/cow.npz", thickness=0.001, scale=0.015)  
    sdf_box = hdx.BoxSDF(size=(0.025, 0.025, 0.025))

    # sdf_fractal = FractalSDF(scale=3.0, iterations=10)
    sdf_helix = HexPrismSDF(radius=0.03, height=0.1)

    sdf_gyroid = GyroidSDF(scale=0.1, thickness=0.001)

    sdf_displaced = DisplacedSphereSDF(radius=0.04, frequency=150.0, amplitude=0.01)

    sdf_displaced2 = DisplacedSphereSDF(radius=0.04, frequency=150.0, amplitude=0.01)

    key = jax.random.PRNGKey(42)
    asteroid_sdf = RandomParticleCloudSDF(
        key=key,
        count=15,
        bounds=0.04,       # Spread +/- 0.04 (Fits in 0.1 box)
        min_radius=0.01,   # Size 1cm
        max_radius=0.025,  # Size 2.5cm
        smooth_k=200.0     # Smooth blend for small scale
    )
    sdf_asteroid_state = asteroid_sdf.create_state(
        center_of_mass=jnp.array([0.0, 0.0, 0.0])
    )
    
    # 3. The Debris Field (Hard Cloud)
    key, subkey = jax.random.split(key)
    debris_sdf = RandomParticleCloudSDF(
        key=subkey,
        count=60,
        bounds=0.09,
        min_radius=0.002,
        max_radius=0.008,
        smooth_k=0.0      # Sharp rocks
    )


    debris_sdf2 = RandomParticleCloudSDF(
        key=subkey,
        count=200,
        bounds=0.2,
        min_radius=0.002,
        max_radius=0.008,
        smooth_k=0.0      # Sharp rocks
    )

    sdf_debris_state = debris_sdf.create_state(
        center_of_mass=jnp.array([0.0, 0.0, 0.0])
    )
    

    sdf_shapes =[sdf_displaced, sdf_displaced2,debris_sdf, debris_sdf2, sdf_cow, sdf_helix]
    # sdf_shapes = [asteroid_sdf]
    chain_sdf = MChainedMorphSDF(shapes=sdf_shapes)

    # # Initial State
    mdf_state = chain_sdf.create_state(
        center_of_mass=jnp.array([0.0, -0.05, 0.0]), # Centered in domain
        velocity=jnp.array([0.0, 0.0, 0.0]),
        # rotation=0.0,
        # angular_velocity=0.0,
        blend_factor=0.0,
        blend_rate=0.0
    )


    def update_morph(sim_time, sdf_state, num_shapes):
        
        # Timing Configuration
        hold_time = 1.0   # Seconds to freeze shape
        morph_time = 0.5  # Seconds to transition
        step_duration = hold_time + morph_time
        
        max_idx = float(num_shapes - 1)
        
        def compute_trajectory(t):
            # 1. "Unfolded" Step Count
            # This counts 0, 1, 2, 3, 4, 5... forever
            step_idx_unfolded = jnp.floor(t / step_duration)
            
            # 2. Local Progress within step
            # 0.0 to 1.0 over 'step_duration'
            local_time = t % step_duration
            
            # 3. Apply Hold Logic (Staircase)
            # If local_time < hold_time -> fraction is 0
            # If local_time > hold_time -> fraction goes 0 to 1
            linear_frac = (local_time - hold_time) / morph_time
            linear_frac = jnp.clip(linear_frac, 0.0, 1.0)
            
            # Smoothstep for nice velocity
            smooth_frac = linear_frac * linear_frac * (3.0 - 2.0 * linear_frac)
            
            # 4. Unfolded Trajectory
            # Value goes: 0 (hold) -> 1 (hold) -> 2 (hold) -> 3 ...
            p_unfolded = step_idx_unfolded + smooth_frac
            
            # 5. Fold Back (Ping-Pong)
            # We want to map: 0->0, 1->1, 2->2, 3->1, 4->0, 5->1 ...
            # Cycle length is 2 * max_idx (e.g. 0-1-2-1-0 is length 4 segments)
            cycle_len = 2.0 * max_idx
            
            p_mod = p_unfolded % cycle_len
            
            # Triangle Logic:
            # If p < 2: val = p
            # If p > 2: val = 4 - p  (e.g. 2.5 -> 1.5)
            p_folded = jnp.where(p_mod > max_idx, cycle_len - p_mod, p_mod)
            
            return p_folded

        # Use AD to get Value and Rate
        b_factor, b_rate = jax.value_and_grad(compute_trajectory)(sim_time)

        # Optional: Spin based on rate
        omega = b_rate * 5.0 

        return eqx.tree_at(
            lambda s: (s.blend_factor, s.blend_rate, s.angular_velocity), 
            sdf_state, 
            (b_factor, b_rate, omega)
        )
        # exit()
    # print(sdf_cow.grid_min, sdf_cow.grid_size)

    # smin, smax = sdf.get_local_bounds()
    # print("Bunny SDF Local Bounds:", smin, smax)
    # sdf_state = sdf.create_state(

        # center_of_mass=jnp.array([0.,0.,0.])
    # )

    vis = hdx.RerunVisualizer(
        origin = origin,
        end = end,
        cell_size=cell_size
        # origin=origin, end=end, cell_size=cell_size, ppc=ppc,
        #   mode="save"
        
        )
    
    # vis.log_sdf_boundary(
    # sdf_logic=debris_sdf,
    # sdf_state=sdf_debris_state,
    # resolution=200,
    # )

    # exit()
    
    # def log_simulation(sim_state):
        # vis.log_simulation(sim_state)
        # vis.log_sdf_boundary(chain_sdf, sim_state.forces[sdf_idx])
    dt = 0.5e-3
    total_steps = int(10.0 / dt)
    output_step = int(0.05 / dt)

    for  i in range(total_steps):
        # Update Morph
        if i % output_step == 0:
            print(f"Step {i}/{total_steps}")
            mdf_state = update_morph(i * dt, mdf_state,num_shapes=len(sdf_shapes))

            vis.log_sdf_boundary(chain_sdf, mdf_state)
        # sim_state = hdx.MPMSolver.step_simulation_step(sim_state)
        
        # if i % output_step == 0:
            # log_simulation(sim_state)
    
    # def loop_body(i, sim_state):
    # #     # Update Morph
    #     forces = list(sim_state.forces)
    #     forces[sdf_idx] = update_morph(sim_state.time, forces[sdf_idx])
    #     sim_state = eqx.tree_at(lambda s: s.forces, sim_state, tuple(forces))
        
    #     sim_state = mpm_solver.step(sim_state)
        
    #     # jax.lax.cond(i % output_step == 0, lambda s: jax.debug.callback(log_simulation, s), lambda s: None, sim_state)
    # #     return sim_state

    # # @jax.jit
    # # def run_sim(state):
    # #     return jax.lax.fori_loop(0, total_steps, loop_body, state)

    # # run_sim(sim_state)
