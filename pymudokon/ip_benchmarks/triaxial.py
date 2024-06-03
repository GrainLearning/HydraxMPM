"""Base class for single integration point benchmark module"""


import jax.numpy as jnp
import jax.scipy.optimize as jsp
import jax
import optax
import jaxopt
from jax import Array

from typing import Callable
from ..materials.material import Material



def update_from_params(
    material: Material,
    radial_strains_11_22: Array,
    target_strain_tensor: Array,
    strain_prev: Array,
    volumes: Array,
    dt: jnp.float32,
    
):
    strain_trail = target_strain_tensor.at[(1,2),(1,2)].set(radial_strains_11_22)
    
    strain_increment = strain_trail - strain_prev

    next_volumes = volumes * (1 + jnp.trace(strain_increment))
    
    strain_rates = (strain_increment / dt).reshape((1,3,3))
    
    
    stress, material = material.update_stress_benchmark(
        strain_rates, 
        next_volumes, 
        dt)
    return stress, strain_trail, material,next_volumes
    
    
def triax_loss(
    strain_guess_11_22,
    confine_11_22,
    target_strain_tensor,
    strain_prev,
    dt,
    material,
    volumes
):

    stress, _, material,_ = update_from_params(
        material,
        strain_guess_11_22,
        target_strain_tensor,
        strain_prev,
        volumes,
        dt
    )
    
    stress_11_22 = stress.at[0,(1,2),(1,2)].get().flatten()
    loss = optax.losses.l2_loss(stress_11_22, confine_11_22)
    return jnp.mean(loss)

# @jax.jit
def triaxial_compression(
    material: Material,
    eps_path: Array,
    confine: Array,
    prestress: Array,
    prestrain: Array,
    volumes: Array,
    dt: jnp.float32 | Array,
    output_step= int,
    output_function:Callable =None,
    triax_learning_rate:float = 1e-3,
    num_opt_iter:int = 20
)-> Material:
    
    num_steps = eps_path.shape[0]

    eps_path = eps_path.reshape(-1)
    
    prestrain = prestrain.reshape((3,3))
    
    prestress = prestress.reshape((1,3,3))
    
    
    confine_11_22 = -jnp.ones(2)*confine
    
    strain_prev = prestrain
    
    stress_prev = prestress
    
    def body_loop(step,carry):
        prestrain, strain_prev, stress_prev, material, volumes, dt, output_function = carry
        
        target_strain_tensor = prestrain.at[0,0].set(eps_path[step])
        
        params = strain_prev.at[(1,2),(1,2)].get().flatten()
        
        
        solver = optax.adabelief(learning_rate=triax_learning_rate)
        opt_state = solver.init(params)
        
        def run_solver_(i, carry):
            params, opt_state = carry
            grad = jax.grad(triax_loss)(params, confine_11_22, target_strain_tensor, strain_prev, dt, material, volumes)
            updates, opt_state = solver.update(grad, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state

        
        params,opt_state = jax.lax.fori_loop(
            0,
            num_opt_iter,
            run_solver_,
            (params, opt_state)
        )
        
        stress, strain, material, volumes = update_from_params(
            material,
            params,
            target_strain_tensor,
            strain_prev,
            volumes,
            dt)
        

        jax.lax.cond(
            step % output_step == 0,
            lambda x: jax.experimental.io_callback(output_function, None,x),
            lambda x: None,
            (step, stress, material, strain.reshape((1,3,3)), num_steps, dt)
        )
        
        return prestrain, strain, stress, material, volumes, dt, output_function
        
        
    package = jax.lax.fori_loop(
        0,
        num_steps,
        body_loop,
        (prestrain, strain_prev, stress_prev, material, volumes, dt, output_function),
    )
    prestrain, strain_prev, stress_prev, material, volumes, dt, output_function = package
    return material
    
