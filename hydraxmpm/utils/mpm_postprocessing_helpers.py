


# def post_processes_stress_stack(
#     stress_stack,
#     mass_stack,
#     position_stack,
#     nodes,
#     shapefunctions
#     ):
#     """
#     Post processing to visualize stress results of mpm
    
#     see refs
#     - Andersen, Søren, and Lars Vabbersgaard Andersen. "Post-processing in the material-point method." (2012).
#     - Dunatunga, Sachith, and Ken Kamrin. "Continuum modelling and simulation of granular flows through their many phases." Journal of Fluid Mechanics 779 (2015): 483-513.
    

#     Args:
#         stress_stack (_type_): _description_
#         mass_stack (_type_): _description_
#         position_stack (_type_): _description_
#         nodes (_type_): _description_
#         shapefunctions (_type_): _description_

#     Returns:
#         _type_: _description_
#     """
    
#     # get shapefunctions
    
#     stencil_size, dim = shapefunctions.stencil.shape
#     shapefunctions, _ = shapefunctions.calculate_shapefunction(
#         origin=nodes.origin,
#         inv_node_spacing=nodes.inv_node_spacing,
#         grid_size=nodes.grid_size,
#         position_stack=position_stack,
#         species_stack = nodes.species_stack
#     )
    
#     # p2g
#     stencil_size, dim = shapefunctions.stencil.shape

#     @partial(jax.vmap, in_axes=(0, 0))
#     def vmap_p2g(intr_id, intr_shapef):
#         particle_id = (intr_id / stencil_size).astype(jnp.int32)

#         intr_masses = mass_stack.at[particle_id].get()
#         intr_stresses = stress_stack.at[particle_id].get()

#         scaled_stress = intr_shapef*intr_masses*intr_stresses
        
#         scaled_mass = intr_shapef * intr_masses
        
#         return scaled_mass, scaled_stress
    
#     scaled_mass_stack, scaled_stress_stack = vmap_p2g(
#             shapefunctions.intr_id_stack,
#             shapefunctions.intr_shapef_stack
#         )
    
#     # in case arrays are not empty from prior simulation run
#     zero_node_mass_stack = jnp.zeros_like(nodes.mass_stack)
#     zero_node_stress_stack = jnp.zeros((nodes.num_nodes_total, 3,3)).astype(jnp.float32)

#     nodes_mass_stack = zero_node_mass_stack.at[shapefunctions.intr_hash_stack].add(
#             scaled_mass_stack
#         )
    
#     nodes_stress_stack = zero_node_stress_stack.at[shapefunctions.intr_hash_stack].add(
#             scaled_stress_stack
#         )
    
#     # # g2p
#     @partial(jax.vmap, in_axes=(0, 0))
#     def vmap_intr_scatter(intr_hashes, intr_shapef):
#         intr_stresses = nodes_stress_stack.at[intr_hashes].get()
        
#         return intr_shapef*intr_stresses

#     @partial(jax.vmap, in_axes=(0))
#     def vmap_particles_update(intr_scaled_stresses_reshaped):
#         p_stresses= jnp.sum(intr_scaled_stresses_reshaped,axis=0)
#         return p_stresses

    
#     intr_scaled_stresses_stack = vmap_intr_scatter(
#         shapefunctions.intr_hash_stack,
#         shapefunctions.intr_shapef_stack
#     )
#     p_stress_next_stack = vmap_particles_update(
#         intr_scaled_stresses_stack.reshape(-1, stencil_size, 3, 3)
#     )

#     return p_stress_next_stack


# def post_processes_grid_gradient_stack(
#     x_stack,
#     mass_stack,
#     position_stack,
#     nodes,
#     shapefunctions
#     ):
#     """
#     Post processing to visualize stress results of mpm
    
#     see refs
#     - Andersen, Søren, and Lars Vabbersgaard Andersen. "Post-processing in the material-point method." (2012).
#     - Dunatunga, Sachith, and Ken Kamrin. "Continuum modelling and simulation of granular flows through their many phases." Journal of Fluid Mechanics 779 (2015): 483-513.
    

#     Args:
#         stress_stack (_type_): _description_
#         mass_stack (_type_): _description_
#         position_stack (_type_): _description_
#         nodes (_type_): _description_
#         shapefunctions (_type_): _description_

#     Returns:
#         _type_: _description_
#     """
    
#     # get shapefunctions
    
#     stencil_size, dim = shapefunctions.stencil.shape
#     shapefunctions, _ = shapefunctions.calculate_shapefunction(
#         origin=nodes.origin,
#         inv_node_spacing=nodes.inv_node_spacing,
#         grid_size=nodes.grid_size,
#         position_stack=position_stack,
#         species_stack = None
#     )
    
#     # p2g
#     stencil_size, dim = shapefunctions.stencil.shape
    
#     @partial(jax.vmap, in_axes=(0, 0))
#     def vmap_p2g(intr_id, intr_shapef):
#         particle_id = (intr_id / stencil_size).astype(jnp.int32)

#         intr_masses = mass_stack.at[particle_id].get()
#         intr_x = x_stack.at[particle_id].get()

#         scaled_x = intr_shapef*intr_masses*intr_x
        
#         scaled_mass = intr_shapef * intr_masses
        
#         return scaled_mass, scaled_x
    
#     scaled_mass_stack, scaled_x_stack = vmap_p2g(
#             shapefunctions.intr_id_stack,
#             shapefunctions.intr_shapef_stack
#         )
#     # print(scaled_x_stack.min(), scaled_x_stack.max())
    
#     # in case arrays are not empty from prior simulation run
#     zero_node_mass_stack = jnp.zeros_like(nodes.mass_stack)
    
#     out_shape = x_stack.shape[1:]
#     zero_node_x_stack = jnp.zeros((nodes.num_nodes_total, *out_shape)).astype(jnp.float32)

#     nodes_mass_stack = zero_node_mass_stack.at[shapefunctions.intr_hash_stack].add(
#             scaled_mass_stack
#         )
    
#     nodes_x_stack = zero_node_x_stack.at[shapefunctions.intr_hash_stack].add(
#             scaled_x_stack
#         )

#     def divide(X_generic,mass):
        
#         result = jax.lax.cond(
#                 mass > nodes.small_mass_cutoff,
#                 lambda x: x/ mass,
#                 lambda x: jnp.nan,
#                 X_generic,
#         )
#         return result
    
#     return jax.vmap(divide)(nodes_x_stack,nodes_mass_stack)

#     # return nodes_x_stack/nodes_mass_stack.reshape(-1,1,1)
    
