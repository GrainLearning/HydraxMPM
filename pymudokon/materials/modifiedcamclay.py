"""Implementation, state and functions for isotropic linear elastic material."""

import dataclasses
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array
from typing_extensions import Self

from ..core.particles import Particles
from .material import Material


def accept_elastic_solution(package):
    update_history, eps_e, eps_e_prev, stress = package
    
    if update_history:
        return stress, eps_e
    
    return stress, eps_e_prev

def vmap_update(
    eps_e_prev: Array,
    deps: Array,
    stress_ref: Array,
    pc:Array,
    G: jnp.float32,
    K: jnp.float32,
    M: jnp.float32,
    lam: jnp.float32,
    kap: jnp.float32
) -> Tuple[Array, Array]:
    """Parallelized stress update for each particle.

    Tensors are mapped per particle via vmap (num_particles, dim) -> (dim,).

    Args:
        eps_e_prev (Array):
            Vectorized previous elastic strain tensor.
            Shape of full tensor is `(number of particles,dimension,dimension)`.
        deps (Array):
            Vectorized strain increment.
            Shape of full tensor is `(number of particles,dimension,dimension)`.
        G (jnp.float32):
            Shear modulus.
        K (jnp.float32):
            Bulk modulus.
        dt (jnp.float32):
            Time step.

    Returns:
        Tuple[Array, Array]:
            Updated stress and elastic strain tensors.

    Example:
        >>> import pymudokon as pm
        >>> import jax.numpy as jnp
        >>> eps_e = jnp.ones((2, 2, 2), dtype=jnp.float32)
        >>> vel_grad = jnp.ones((2, 2, 2), dtype=jnp.float32)
        >>> stress, eps_e = jax.vmap(vmap_update, in_axes=(0, 0, None, None, None))(
        >>> eps_e, vel_grad, 1000, 1000, 0.001)
    """
    dim = deps.shape[0]

    eps_e_tr = eps_e_prev + deps

    eps_e_v_tr = -jnp.trace(eps_e_tr)

    eps_e_d_tr = eps_e_tr + (eps_e_v_tr / dim) * jnp.eye(dim)

    s_tr = 2.0 * G * eps_e_d_tr

    # pad for 3D stress tensor
    if dim == 2:
        s_tr = jnp.pad(s_tr, ((0, 1), (0, 1)), mode="constant")

    p_tr =  K * eps_e_v_tr

    p_ref = -jnp.trace(stress_ref)/dim
    s_ref = stress_ref + p_ref * jnp.eye(3)
    
    p_tr = p_tr + p_ref
    s_tr = s_tr + s_ref
    
    q_tr = jnp.sqrt(1.5*(s_tr@s_tr.T).trace())
            
    ps_tr = 0.5*pc
        
    yield_function = lambda p, ps, q, M : (p-ps)**2 +(q/M)**2 -ps**2
        
    F = yield_function(p_tr,q_tr,ps_tr, M)
    
    s_tr, eps_e_tr, p_tr, q_tr = jax.lax.cond(
        F <= 0,
        lambda x: x,
        lambda x: x, 
        (update_history, eps_e, eps_e_prev, stress))
    
    
    
    return s_tr - p_tr* jnp.eye(3), eps_e_tr, p_tr,q_tr


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=False)
class ModifiedCamClay(Material):
    """State for the isotropic linear elastic material.

    Attributes:
        E (jnp.float32):
            Young's modulus.
        nu (Array):
            Poisson's ratio.
        G (Array):
            Shear modulus.
        K (Array):
            Bulk modulus.
        eps_e (Array):
            Elastic strain tensors.
            Shape is `(number of particles,dim,dim)`.
    """

    # TODO we can replace this with incremental form / or hyperelastic form
    E: jnp.float32
    nu: jnp.float32
    G: jnp.float32
    K: jnp.float32
    
    M: jnp.float32
    R: jnp.float32
    
    lam: jnp.float32
    kap: jnp.float32
    Vs: jnp.float32
    
    stress_ref: Array
    pc: Array
    eps_v_p: Array
    eps_e: Array

    @classmethod
    def register(
        cls: Self,
        E: jnp.float32,
        nu: jnp.float32,
        M: jnp.float32,
        R: jnp.float32,
        lam: jnp.float32,
        kap: jnp.float32,
        Vs: jnp.float32,
        stress_ref: Array,
        num_particles,
        dim: jnp.int16 = 3) -> Self:
        """Initialize the isotropic linear elastic material.

        Args:
            E (jnp.float32):
                Young's modulus.
            nu (jnp.float32):
                Poisson's ratio.
            num_particles (jnp.int16):
                Number of particles.
            dim (jnp.int16, optional):
                Dimension. Defaults to 3.

        Returns:
            LinearIsotropicElastic:
                Updated state for the isotropic linear elastic material.

        Example:
            >>> # Create a linear elastic material for plane strain and 2 particles
            >>> import pymudokon as pm
            >>> import jax.numpy as jnp
            >>> material = pm.LinearIsotropicElastic.register(1000, 0.2, 2, 2)
        """
        # TODO: How can we avoid the need for num_particles and dim?

        G = E / (2.0 * (1.0 + nu))
        K = E / (3.0 * (1.0 - 2.0 * nu))
        eps_e = jnp.zeros((num_particles, dim, dim), dtype=jnp.float32)
        
        p_ref = -jnp.trace(stress_ref, axis1=1, axis2=2)/dim
        pc = p_ref*R
        
        eps_v_p = jnp.zeros((num_particles), dtype=jnp.float32)
        
        return cls(E=E, nu=nu, G=G, K=K, M=M, R=R, lam=lam, kap=kap, Vs=Vs,
                   stress_ref=stress_ref, eps_e=eps_e, eps_v_p=eps_v_p, pc=pc)

    @jax.jit
    def update_stress(
        self: Self,
        particles: Particles,
        dt: jnp.float32,  # unused
    ) -> Tuple[Particles, Self]:
        """Update stress and strain for all particles.

        Called by the MPM solver (e.g., see :func:`~usl.update`).

        Args:
            self (LinearIsotropicElastic):
                self reference.
            particles (ParticlesContainer):
                State of the particles prior to the update.

            dt (jnp.float32):
                Time step.

        Returns:
            Tuple[ParticlesContainer, LinearIsotropicElastic]:
                Updated particles and material state.

        Example:
            >>> import pymudokon as pm
            >>> # ...  Assume particles and material are initialized
            >>> particles, material = material.update_stress(particles, 0.001)
        """
        raise NotImplementedError("This method is not implemented yet.")
        vel_grad = particles.velgrads
        
        vel_grad_T = jnp.transpose(vel_grad, axes=(0, 2, 1))
        
        deps = 0.5 * (vel_grad + vel_grad_T) * dt
        
        stress, eps_e = jax.vmap(vmap_update, in_axes=(0, 0, None, None), out_axes=(0, 0))(
            self.eps_e, deps, self.G, self.K
        )

        material = self.replace(eps_e=eps_e)

        particles = particles.replace(stresses=stress)
        
        

        return particles, material

    @jax.jit
    def update_stress_benchmark(
        self: Self,
        strain_rate: Array,
        dt: jnp.float32,
        update_history: bool = True,
    ) -> Self:
        deps = strain_rate* dt
        stress, eps_e, p_trail, q_trail = jax.vmap(vmap_update, in_axes=(0, 0, 0,0, None, None, None, None, None), out_axes=(0, 0, 0,0))(
            self.eps_e, deps, self.stress_ref, self.pc, self.G, self.K, self.M, self.lam, self.kap
        )
        
        
        return stress, self.replace(eps_e=eps_e)