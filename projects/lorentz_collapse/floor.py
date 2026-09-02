import jax
import hydraxmpm as hdx
import jax.numpy as jnp

import equinox as eqx

from typing import Optional


class SpectralSetup(eqx.Module):
    """
    A 2D Bumpy Floor SDF with sinusoidal geometry and spatially varying friction.
    The floor is oriented such that the 'inside' (safe) region is y > h(x).
    """

    amplitudes: jnp.ndarray
    wavenumbers: jnp.ndarray
    phases: jnp.ndarray
    scale_base: jnp.ndarray # scale signal to meters OR friction

    def __init__(
        self,
        num_harmonics=16,
        hurst=0.8,
        lambda_min=0.1,
        lambda_max=2.0,
        rms_height=0.05,
        scale_base=0.0,
        key_factor = 0,
    ):  

        
        self.scale_base = jnp.array(scale_base)

        # natural surfaces are self-affine
        # if we zoom in  the lambda_min and lambda_max define the range of wavelengths we want to include in our surface representation.
        k_min = 2.0 * jnp.pi / lambda_max
        k_max = 2.0 * jnp.pi / lambda_min
        
        self.wavenumbers = jnp.logspace(
            jnp.log10(k_min), jnp.log10(k_max), num_harmonics
        )

        # define amplitudes based on Hurst Exponent
        # Power law: A(k) ~ k^(-(H + 0.5))
        raw_amplitudes = self.wavenumbers ** (-(hurst + 0.5))

        # normalize amplitudes to match desired RMS height
        # RMS of sum of sines is sqrt(sum(A_i^2 / 2))
        current_rms = jnp.sqrt(jnp.sum(raw_amplitudes**2) / 2.0)
        self.amplitudes = raw_amplitudes * (rms_height / current_rms)

        # random phases
        self.phases = jnp.linspace(0, 2 * jnp.pi, num_harmonics) * key_factor

    
    def get_bump_waves(self, x):
        args = self.wavenumbers * x + self.phases
        return jnp.sum(self.amplitudes *  jnp.sin(args))


    
    def __call__(self,x):
        return self.scale_base + self.get_bump_waves(x)


class BumpyFloorSDFState(hdx.SDFObjectState):
    bumps: SpectralSetup
    frictions: SpectralSetup

class BumpyFloorSDF(hdx.SDFObjectBase):

    x_min: float  # friction/ bump transition start
    x_max: float  # friction/ bump transition end
    x_transition_width: float  # transition width for friction and bump (linear ramp)
  
    

    def __init__(self, x_min, x_max, x_transition_width):
        
        self.x_min = x_min
        self.x_max = x_max
        self.x_transition_width = x_transition_width

    def create_state(self, bumps=None, frictions=None):

        amplitudes = bumps.amplitudes
        y_base = bumps.scale_base

        max_h = jnp.sum(jnp.abs(amplitudes))

        bbox_min, bbox_max= jnp.array([-1e6, y_base - max_h]), jnp.array([1e6, 1e6])

   
        center_of_mass = 0.5 * (bbox_min + bbox_max)

        base_state = super().create_state(
            center_of_mass=center_of_mass
        )

        return BumpyFloorSDFState(
            center_of_mass=base_state.center_of_mass,
            rotation=base_state.rotation,
            velocity=base_state.velocity,
            angular_velocity=base_state.angular_velocity,
            bumps=bumps,
            frictions=frictions,
        )



    def signed_distance_local(self, state, p_local):
        x, y = p_local[0], p_local[1]
        
        x_rel = x - self.x_min
        
        # get only the bumps 
        bump_waves = state.bumps.get_bump_waves(x_rel)

        # create ramp 0 to 1 over transition width
        weight = jnp.clip(x_rel/ self.x_transition_width, 0.0, 1.0)
        

        # final height is base + weighted bumps
        h = state.bumps.scale_base + (bump_waves * weight)
        
        is_past_end = x > self.x_max

        h_end = state.bumps.scale_base + state.bumps.get_bump_waves(x - self.x_max)

        h = jnp.where(is_past_end, h_end, h)

        dist_vert = y - h

        return dist_vert

    def get_surface_friction_local(self, state, p_local):
            """
            Calculates the spatially varying friction coefficient using the 
            spectral signal defined in state.frictions.
            """

            if state.frictions is None:
                return 0.5 

            x = p_local[0]
            

            x_rel = x - self.x_min

            is_past_end = x > self.x_max
            eval_x = jnp.where(is_past_end, x - self.x_max, x_rel)
            
            # Evaluate the spectral signal
            #  returns scale_base + sum(A * sin(kx + phi))
            mu = state.frictions(eval_x)

     
            return jnp.maximum(1e-6, mu)

    def transform_to_local(self, state, p_world):
        return p_world

    def get_world_aabb(self, state=None):

        amplitudes = state.bumps.amplitudes
        y_base = state.bumps.scale_base

        max_h = jnp.sum(jnp.abs(amplitudes))

        return jnp.array([-1e6, y_base - max_h]), jnp.array([1e6, 1e6])