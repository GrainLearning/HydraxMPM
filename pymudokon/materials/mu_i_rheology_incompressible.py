"""Implementation, state and functions for isotropic linear elastic material."""

from functools import partial
from typing import Tuple
from typing_extensions import Self

import chex
import jax
import jax.numpy as jnp

from ..particles.particles import Particles
from ..utils.math_helpers import (
    get_sym_tensor_stack,
    get_scalar_shear_strain,
    get_dev_strain,
    get_dev_strain_stack,
    get_inertial_number )

from .material import Material


def get_mu_I(I, mu_s, mu_d, I0):
    return mu_s + (mu_d - mu_s) * (1 / (1 + I0 / I))



def get_mu_I_regularized_exp(I, mu_s, mu_d, I0,pen,dgamma_dt):
    # s = (1.0-jnp.exp(-dgamma_dt/pen))
    s = 1./jnp.sqrt(dgamma_dt**2 +pen**2)
    return mu_s*s + (mu_d - mu_s) * (1.0 / (1.0 + I0 / I))

def get_I_phi(phi, phi_c, I_phi):
    return -I_phi * jnp.log(phi / phi_c)


def get_pressure(dgammadt, I, d, rho_p):
    return rho_p * ((dgammadt * d) / I) ** 2


@chex.dataclass
class MuI_incompressible(Material):
    mu_s: jnp.float32
    mu_d: jnp.float32
    I_0: jnp.float32
    rho_p: jnp.float32
    d: jnp.float32
    dim: jnp.int32
    K: jnp.float32
    """
    (nearly) incompressible mu I
    
    Tensorial form similar to
    Jop, Pierre, Yoël Forterre, and Olivier Pouliquen. "A constitutive law for dense granular flows." Nature 441.7094 (2006): 727-730.
    
    mu I regularized by
    Franci, Alessandro, and Massimiliano Cremonesi. "3D regularized μ (I)-rheology for granular flows simulation." Journal of Computational Physics 378 (2019): 257-277.
    
    Pressure term by
    
    Salehizadeh, A. M., and A. R. Shafiei. "Modeling of granular column collapses with μ (I) rheology using smoothed particle hydrodynamic method." Granular Matter 21.2 (2019): 32.

    """

    @classmethod
    def create(
        cls: Self,
        mu_s: jnp.float32,
        mu_d: jnp.float32,
        I_0: jnp.float32,
        rho_p: jnp.float32,
        d: jnp.float32,
        K:  jnp.float32 =1.0,
        absolute_density: jnp.float32 = 0.0,
        dim: jnp.int32 =3
    ) -> Self:
        
        return cls(
            mu_s=mu_s,
            mu_d=mu_d,
            I_0=I_0,
            d=d,
            rho_p=rho_p,
            absolute_density=absolute_density,
            K=K,
            dim = dim
        )

    def update_from_particles(
        self: Self, particles: Particles, dt: jnp.float32
    ) -> Tuple[Particles, Self]:
        """Update the material state and particle stresses for MPM solver."""

        density_stack = particles.mass_stack/particles.volume_stack
        density0_stack = particles.mass_stack/particles.volume0_stack

        phi_stack = density_stack/density0_stack
        stress_stack, self = self.update(
            particles.stress_stack, particles.F_stack, particles.L_stack, phi_stack, dt
        )

        return particles.replace(stress_stack=stress_stack), self

    def update(
        self: Self,
        stress_prev_stack: chex.Array,
        F_stack: chex.Array,
        L_stack: chex.Array,
        phi_stack: chex.Array,
        dt: jnp.float32,
    ) -> Tuple[chex.Array, Self]:

        deps_dt_stack = get_sym_tensor_stack(L_stack)

        stress_next_stack = self.vmap_viscoplastic(deps_dt_stack, phi_stack,stress_prev_stack)

        return stress_next_stack, self


    @partial(jax.vmap, in_axes=(None, 0, 0,0), out_axes=(0))
    def vmap_viscoplastic(self, strain_rate: chex.Array, phi: chex.Array,stress_prev:chex.Array):
        
        p = jnp.maximum(self.K*(phi-1.0),1e-12)

        deps_dev_dt = get_dev_strain(strain_rate,dim =self.dim)
        
        dgamma_dt = get_scalar_shear_strain(strain_rate,dim = self.dim)
        
        
        
        I = get_inertial_number(p,dgamma_dt,self.d, self.rho_p)

        def flow():
            alpha = 0.0001
            eta_E_s = p*self.mu_s/jnp.sqrt(dgamma_dt*dgamma_dt + alpha*alpha)
            
            mu_I_delta = (self.mu_d - self.mu_s)/(1.0 + self.I_0 / I)
            
            eta_delta = p*mu_I_delta/dgamma_dt
            
            eta = eta_E_s+eta_delta
            
            stress_next = -p * jnp.eye(3) + eta * deps_dev_dt

            return stress_next
        
        def stop():
            return stress_prev

        return jax.lax.cond(dgamma_dt<1e-6,stop,flow )

    def get_p_ref(self, phi):
        return jnp.maximum(self.K*(phi-1.0),1e-12)

