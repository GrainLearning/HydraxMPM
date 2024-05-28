"""Implementation, state and functions for isotropic linear elastic material."""

import dataclasses
from typing import Tuple
from typing import NamedTuple
import jaxopt

import jax
import jax.numpy as jnp
from jax import Array
from typing_extensions import Self

from ..core.particles import Particles
from .material import Material

from functools import partial


def yield_function(p, p_c, q, M):
    """Compute the modified Cam Clay yield function."""
    p_s = 0.5*p_c
    return (p_s - p) ** 2 + (q / M) ** 2 - p_s**2
        
class MaterialProperties(NamedTuple):
    E: jnp.float32
    nu: jnp.float32
    G: jnp.float32
    K: jnp.float32
    M: jnp.float32
    R: jnp.float32
    lam: jnp.float32
    kap: jnp.float32
    Vs: jnp.float32

@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=False)
class ModifiedCamClay(Material):

    props : MaterialProperties
    stress_ref: Array
    p_c: Array
    eps_e: Array
    eps_v_p: Array


    @partial(jax.vmap, in_axes=(None, 0))
    def _count_value_in_array(
        array: jnp.array, value: float
    ) -> jnp.array:
        """Count how often a value occurs in an array"""
        return jnp.count_nonzero(array == value)
    
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
        dim: jnp.int16 = 3,
    ) -> Self:
        
        K = E / (3.0 * (1.0 - 2.0 * nu))
        
        G = E / (2.0 * (1.0 + nu))
        
        props = MaterialProperties(E=E, nu=nu, G=G, K=K, M=M, R=R, lam=lam, kap=kap, Vs=Vs)
        
        eps_e = jnp.zeros((num_particles, dim, dim), dtype=jnp.float32)

        p_ref = -jnp.trace(stress_ref, axis1=1, axis2=2) / dim
        
        p_c = p_ref * R

        eps_v_p = jnp.zeros((num_particles), dtype=jnp.float32)

        return cls(
            props=props,
            stress_ref=stress_ref,
            p_c=p_c,
            eps_e=eps_e,
            eps_v_p=eps_v_p,
        )


    @partial(jax.vmap, in_axes=(None,0, 0, 0), out_axes=(0, 0, 0,0,0))
    def vmap_elastic_trail_step(
        self,
        stress_ref: Array,
        eps_e_prev: Array,
        deps_next: Array
    ):

        dim = deps_next.shape[0]

        eps_e_tr = eps_e_prev + deps_next

        eps_e_v_tr = -jnp.trace(eps_e_tr)

        eps_e_d_tr = eps_e_tr + (eps_e_v_tr / dim) * jnp.eye(dim)

        s_tr = 2.0 * self.props.G * eps_e_d_tr

        # pad for 3D stress tensor
        if dim == 2:
            s_tr = jnp.pad(s_tr, ((0, 1), (0, 1)), mode="constant")

        p_tr = self.props.K * eps_e_v_tr

        p_ref = -jnp.trace(stress_ref) / dim
        s_ref = stress_ref + p_ref * jnp.eye(3)

        p_tr = p_tr + p_ref

        s_tr = s_tr + s_ref
        
        return s_tr, p_tr, eps_e_tr,s_ref,p_ref

    @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0,0, 0, 0, 0), out_axes=(0, 0, 0, 0))
    def vmap_plastic_return_mapping(
        self,
        s_tr: Array,
        p_tr: Array,
        p_c_prev: Array,
        eps_e_tr: Array,
        eps_v_p_prev: Array,
        volume: Array,
        s_ref: Array,
        p_ref: Array,
    ):  
        
        specific_volume = volume / self.props.Vs
            
        v_lam_tilde = specific_volume / (self.props.lam - self.props.kap)
        
        q_tr = jnp.sqrt(1.5 * (s_tr @ s_tr.T).trace())
        
        is_ep = yield_function(p_tr, p_c_prev, q_tr, self.props.M) > 0
        
        # The following variables are global and passed 
        # eps_v_p_prev = eps_v_p_prev
        # down to functions where used implicitly jitted
        # s_tr, p_tr, p_c_prev, eps_e_tr, eps_v_p_prev, volume, s_ref, p_ref, q_tr, is_ep
        # self,v_lam_tilde, is_ep
        
        def accept_elas():
            """ Accept elastic solutions."""
            stress = s_tr - p_tr * jnp.eye(3)
            return  stress, p_c_prev, eps_e_tr, eps_v_p_prev
        
        def reduced_equation_system(sol):
            
            pmultp, eps_p_v_next = sol
            
            deps_p_v = eps_p_v_next - eps_v_p_prev
            
            p_next = p_tr - self.props.K * deps_p_v
            
            q_next = (self.props.M**2 / (self.props.M**2 + 6.0 * self.props.G * pmultp)) * q_tr
            
            p_c_next = p_c_prev * (1.0 + v_lam_tilde * deps_p_v)
            
            p_s_next = 0.5 * p_c_next
            
            s_next = (self.props.M**2 / (self.props.M**2 + 6.0 * self.props.G * pmultp)) * s_tr
            
            R = jnp.array(
                [
                    yield_function(p_next, p_c_next, q_next, self.props.M),
                    eps_p_v_next - eps_v_p_prev + 2.0 * pmultp * (p_s_next - p_next)
                ]
                ,dtype=jnp.float64)
            
            aux_data = p_next, s_next, p_c_next, eps_p_v_next
            
             # normalize residual for convergence check
            R = R.at[0].set(R[0] / (self.props.E*p_c_prev))
            
            conv = jnp.linalg.norm(R)
            
            return R, aux_data

        def pull_to_yield_surface():
            
            sol = jnp.array([0.0, eps_v_p_prev],dtype=jnp.float64)
            
            R = jnp.array([1.0, 1.0],dtype=jnp.float64)
            
            aux_data = p_tr, s_tr, p_c_prev, eps_v_p_prev
            
            def body_loop(carry):
                R, sol, aux_data = carry
                
                R, aux_data = reduced_equation_system(sol)
                
                jac, *_ = jax.jacfwd(reduced_equation_system, has_aux=True)(sol)
                print(jac.shape)
                inv_jac = jnp.linalg.inv(jac)
                
                sol = sol - inv_jac @ R
                
                return R, sol, aux_data
            
            R, sol, aux_data = jax.lax.while_loop(
                lambda carry: is_ep & (jnp.abs(jnp.linalg.norm(carry[0])) > 1e-2),
                body_loop,
                (R, sol, aux_data)
            )

            p_next, s_next, p_c_next, eps_p_v_next = aux_data

            stress = s_next - p_next * jnp.eye(3)
                
            eps_e_next = (s_next - s_ref) / (2.0 * self.props.G) - (
                    p_next - p_ref
                ) / (3.0 * self.props.K) * jnp.eye(3)
            return  stress, p_c_next,  eps_e_next, eps_p_v_next


        return jax.lax.cond(
            is_ep,
            pull_to_yield_surface,
            accept_elas
        )
    
    @jax.jit
    def update_stress_benchmark(
        self: Self,
        strain_rate: Array,
        volumes: Array,
        dt: jnp.float32,
        update_history: bool = True,
    ) -> Self:
        """
        
        Solution strategy for the Modified Cam Clay mode.

        Args:
            self (Self): _description_
            strain_rate (Array): _description_
            volumes (Array): _description_
            dt (jnp.float32): _description_
            update_history (bool, optional): _description_. Defaults to True.

        Returns:
            Self: _description_
        """
        # Calculate strain rate
        deps = strain_rate * dt
    
    
        # Elastic trail predictor step
        s_tr, p_tr, eps_e_tr,s_ref,p_ref = self.vmap_elastic_trail_step(self.stress_ref, self.eps_e,  deps)

        stress_next, p_c_next, eps_e_next,  eps_v_p_next = self.vmap_plastic_return_mapping(
            s_tr, p_tr, self.p_c, eps_e_tr, self.eps_v_p, volumes, s_ref, p_ref
        )
        
        return stress_next, self.replace(eps_e=eps_e_next, p_c=p_c_next, eps_v_p=eps_v_p_next)
