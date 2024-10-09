"""Module for imposing zero/non-zero boundaries via rigid particles."""

from functools import partial
from typing import Tuple
from typing_extensions import Self

import chex
import jax
import jax.numpy as jnp
from jax import Array

from ..nodes.nodes import Nodes
from ..particles.particles import Particles
from ..shapefunctions.shapefunctions import ShapeFunction
from ..shapefunctions.linear import LinearShapeFunction

@chex.dataclass
class NodeLevelSet:
    id_stack: Array
    velocity_stack: Array
    mu: jnp.float32

    @classmethod
    def create(
        cls: Self,
        id_stack: Array,
        velocity_stack: Array= None
    ) -> Self:
        """Initialize the rigid particles."""
        num_tagged_nodes, dim = velocity_stack.shape        
        if velocity_stack is None:
            velocity_stack = jnp.zeros((num_tagged_nodes,dim))

        return cls(
            id_stack=id_stack,
            velocity_stack=velocity_stack
        )
    @classmethod
    def create_domain_box(
        cls: Self,
        nodes: Nodes,
        thickness:jnp.int32 = 2,
        mu =0.0
        ):
        _, dim = nodes.moment_nt_stack.shape    
        
        all_id_stack = jnp.arange(nodes.num_nodes_total).reshape(nodes.grid_size).astype(jnp.int32)
        
        mask_id_stack = jnp.zeros_like(all_id_stack).astype(jnp.bool_)

        if dim ==2:
            # boundary layers
            mask_id_stack = mask_id_stack.at[0:thickness, :].set(True) #x0
            mask_id_stack = mask_id_stack.at[:, 0:thickness].set(True) #y0
            mask_id_stack = mask_id_stack.at[nodes.grid_size[0]-thickness:, :].set(True) #x1
            mask_id_stack = mask_id_stack.at[:,nodes.grid_size[1] - thickness:].set(True) #y0

        non_zero_ids = jnp.where(mask_id_stack.reshape(-1))[0]
        id_stack = all_id_stack.reshape(-1).at[non_zero_ids].get()
        
        velocity_stack = jnp.zeros((id_stack.shape[0],dim))
        
        return cls(
            id_stack=id_stack,
            velocity_stack=velocity_stack,
            mu = mu
        )
    def apply_on_nodes_moments(
        self: Self,
        nodes: Nodes,
        particles: Particles = None,
        shapefunctions: ShapeFunction = None,
        dt: jnp.float32 = 0.0,
        step: jnp.int32 =0
    ) -> Tuple[Nodes, Self]:
    # Get the normals of the non-rigid particles on the grid.
        stencil_size, dim = shapefunctions.stencil.shape
        
        @partial(jax.vmap, in_axes=(0, 0))
        def vmap_nr_p2g_grid_normals(
            intr_id: chex.ArrayBatched, intr_shapef_grad: chex.ArrayBatched
        ) -> chex.ArrayBatched:
            """Get the normals of the non-rigid particles on the grid."""
            particle_id = (intr_id / stencil_size).astype(jnp.int32)
            intr_masses = particles.mass_stack.at[particle_id].get()
            intr_normal = (intr_shapef_grad * intr_masses).at[:dim].get()
            return intr_normal

        intr_normal_stack = vmap_nr_p2g_grid_normals(
            shapefunctions.intr_id_stack, shapefunctions.intr_shapef_grad_stack
        )
        nodes_normal_stack = (
            jnp.zeros_like(nodes.moment_nt_stack)
            .at[shapefunctions.intr_hash_stack]
            .add(intr_normal_stack)
        )
        @partial(jax.vmap, in_axes=(0, 0))
        def vmap_nodes(
            hash_id,
            velocity
        ):

            node_moment_nt = nodes.moment_nt_stack.at[hash_id].get()
            node_mass = nodes.mass_stack.at[hash_id].get()
            node_normals = nodes_normal_stack.at[hash_id].get()
            
            # skip the nodes with small mass, due to numerical instability
            nodes_vel_nt = jax.lax.cond(
                node_mass > nodes.small_mass_cutoff,
                lambda x: x / node_mass,
                lambda x: jnp.zeros_like(x),
                node_moment_nt,
            )

            # normalize the normals
            node_normals = jax.lax.cond(
                node_mass > nodes.small_mass_cutoff,
                lambda x: x / jnp.linalg.vector_norm(x),
                lambda x: jnp.zeros_like(x),
                node_normals,
            )

            # check if the velocity direction of the normal and apply contact
            # dot product is 0 when the vectors are orthogonal
            # and 1 when they are parallel
            # if othogonal no contact is happening
            # if parallel the contact is happening
            delta_vel = nodes_vel_nt 
            # - velocity
            
            delta_vel_dot_normal = jnp.dot(delta_vel, node_normals)
            
            
            # get tangents
            
            if dim ==2:
                delta_vel_padded =jnp.pad(delta_vel,pad_width=(0,1))
                norm_padded =jnp.pad(node_normals,pad_width=(0,1))
                delta_vel_cross_normal = jnp.cross(
                        delta_vel_padded,
                        norm_padded
                        ) # works only for vectors of len 3
                norm_delta_vel_cross_normal = jnp.linalg.vector_norm(delta_vel_cross_normal)
                
                omega = delta_vel_cross_normal/norm_delta_vel_cross_normal
                mu_prime = jnp.minimum(self.mu, norm_delta_vel_cross_normal/ delta_vel_dot_normal)
                
                normal_cross_omega = jnp.cross(
                        norm_padded,
                        omega
                        ) # works only for vectors of len 3
                
                tangent = (norm_padded + mu_prime*normal_cross_omega).at[:2].get()
            
            # sometimes tangent become nan if velocity is zero at initialization
            # which causes problems
            tangent = jnp.nan_to_num(tangent) 
            
            new_nodes_vel_nt = jax.lax.cond(
                delta_vel_dot_normal > 0.0,
                lambda x: x - delta_vel_dot_normal * tangent,
                # lambda x: x - delta_vel_dot_normal*node_normals, # no friction debug
                lambda x: x,
                nodes_vel_nt,
            )

            node_moments_nt = new_nodes_vel_nt * node_mass
            return node_moments_nt
    
        levelset_moment_nt_stack= vmap_nodes(
            self.id_stack,
            self.velocity_stack
        )

        moment_nt_stack = nodes.moment_nt_stack.at[self.id_stack].set(
            levelset_moment_nt_stack
        )

        return nodes.replace(moment_nt_stack=moment_nt_stack), self
    
    
    def debug_plot_2d(
        self,
        nodes,
        particles,
        breaking =True
    ):
        import matplotlib.pyplot as plt

        aspect_x_y = nodes.grid_size.at[0].get()/ nodes.grid_size.at[1].get()
        fig,ax = plt.subplots(figsize=(4*aspect_x_y,4))
        nodes_position_stack = nodes.get_coordinate_stack(dim=2)
        X, Y = nodes_position_stack.T

        x, y= nodes_position_stack.at[self.id_stack].get().T
        
        p_x, p_y = particles.position_stack.T
        ax.scatter(
            X,
            Y,
            s=1
        )

        ax.scatter(
            x,
            y,
            s=2,
            marker ="s",
            c="red"
        )
        
        ax.scatter(
            p_x,
            p_y,
            s=2,
            marker ="s",
            c="green"
        )
        plt.show()
        if breaking:
            exit()