"""Module for imposing zero/non-zero boundaries via rigid material_points."""

from functools import partial
from typing import Optional, Self, Any

import equinox as eqx
import jax
import jax.numpy as jnp

from ..common.types import (
    TypeFloat,
    TypeInt,
    TypeUIntScalarAStack,
)
from ..grid.grid import Grid
from ..material_points.material_points import MaterialPoints
from .force import Force


class Boundary(Force):
    mu: TypeFloat
    thickness: TypeInt = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    id_stack: Optional[TypeUIntScalarAStack] = None

    _padding: tuple = eqx.field(init=False, static=True, repr=False)

    def __init__(
        self,
        mu: Optional[TypeFloat] = 0.0,
        thickness: Optional[TypeInt] = 2,
        **kwargs,
    ):
        self.mu = mu
        self.thickness = thickness

        self.dim = kwargs.get("dim", 3)
        self.id_stack = kwargs.get("id_stack", None)
        self._padding = (0, 3 - self.dim)

    def init_ids(self: Self, grid: Grid, dim: int, **kwargs):
        # grid.num_cells = jnp.prod(jnp.array(grid_size))

        all_id_stack = (
            jnp.arange(grid.num_cells).reshape(grid.grid_size).astype(jnp.uint32)
        )

        mask_stack = jnp.zeros_like(all_id_stack).astype(jnp.bool_)

        # for BSMPM shapefunctions

        # 0th index is middle
        # 1st index is boundary 0 or N
        # 3rd index is left side of closes boundary 0 + h
        # 4th index is right side of closes boundary N -h
        type_stack = grid.type_stack.reshape(grid.grid_size).at[:].set(0)

        if dim == 2:
            # boundary layers
            mask_stack = mask_stack.at[0 : self.thickness, :].set(True)  # x0

            mask_stack = mask_stack.at[:, 0 : self.thickness].set(True)  # y0

            mask_stack = mask_stack.at[grid.grid_size[0] - self.thickness :, :].set(
                True
            )  # x1

            mask_stack = mask_stack.at[:, grid.grid_size[1] - self.thickness :].set(
                True
            )  # y1
            # left boundary +h
            type_stack = type_stack.at[self.thickness - 1, :].set(3)
            type_stack = type_stack.at[:, self.thickness - 1].set(3)
            # right boundary -h
            type_stack = type_stack.at[grid.grid_size[0] - self.thickness, :].set(4)
            type_stack = type_stack.at[:, grid.grid_size[1] - self.thickness].set(4)

            # boundary
            type_stack = type_stack.at[self.thickness - 2, :].set(1)
            type_stack = type_stack.at[:, self.thickness - 2].set(1)
            type_stack = type_stack.at[grid.grid_size[0] - self.thickness + 1, :].set(1)
            type_stack = type_stack.at[:, grid.grid_size[1] - self.thickness + 1].set(1)

            # # degenerate
            # type_stack = type_stack.at[:, 0 : self.thickness - 1].set(0)
            # type_stack = type_stack.at[0 : self.thickness - 1, :].set(0)
            # type_stack = type_stack.at[grid.grid_size[0] - self.thickness + 1 :, :].set(
            #     0
            # )
            # type_stack = type_stack.at[:, grid.grid_size[1] - self.thickness + 1 :].set(
            #     0
            # )
        else:
            mask_stack = mask_stack.at[0 : self.thickness, :, :].set(True)  # x0
            mask_stack = mask_stack.at[:, 0 : self.thickness, :].set(True)  # y0
            mask_stack = mask_stack.at[:, :, 0 : self.thickness].set(True)  # z0
            mask_stack = mask_stack.at[grid.grid_size[0] - self.thickness :, :, :].set(
                True
            )  # x1
            mask_stack = mask_stack.at[:, grid.grid_size[1] - self.thickness :, :].set(
                True
            )  # y1
            mask_stack = mask_stack.at[:, :, grid.grid_size[2] - self.thickness :].set(
                True
            )  # z1

        non_zero_ids = jnp.where(mask_stack.reshape(-1))[0]

        id_stack = all_id_stack.reshape(-1).at[non_zero_ids].get()

        new_self = Boundary(
            mu=self.mu,
            thickness=self.thickness,
            id_stack=id_stack,
            dim=dim,
        )

        new_grid = eqx.tree_at(
            lambda state: (state.type_stack,),
            grid,
            (type_stack.reshape(-1),),
        )
        return new_self, new_grid

    def apply_on_grid(
        self: Self,
        material_points: Optional[MaterialPoints] = None,
        grid: Optional[Grid] = None,
        step: Optional[TypeInt] = 0,
        dt: Optional[TypeFloat] = 0.01,
        dim: TypeInt = 3,
        **kwargs: Any,
    ):
        @partial(jax.vmap, in_axes=0, out_axes=0)
        def vmap_selected_grid(moment_nt, moment, mass, normal):
            # normal = grid.normal_stack.at[n_id].get()
            # scalar_norm = jnp.linalg.vector_norm(normal)

            def calculate_velocity(mom):
                # check if the velocity direction of the normal and apply contact
                # dot product is 0 when the vectors are orthogonal
                # and 1 when they are parallel
                # if othogonal no contact is happening
                # if parallel the contact is happening

                # vel = mom / mass

                vel = mom / (mass + 1e-12)

                # vel = jax.lax.cond(
                #     mass > grid.small_mass_cutoff,
                #     lambda x: x / mass,
                #     lambda x: jnp.zeros_like(x),
                #     mom,
                # )
                norm = jnp.linalg.vector_norm(normal)
                normal_hat = normal / (norm + 1e-12)
                # normal_hat = jax.lax.cond(
                #     mass > grid.small_mass_cutoff,
                #     lambda x: x /(norm),
                #     lambda x: jnp.zeros_like(x),
                #     normal,
                # )
                # normal_hat = jnp.nan_to_num(normal_hat)

                # normal_hat =normal/scalar_norm

                norm_padded = jnp.pad(
                    normal_hat,
                    self._padding,
                    mode="constant",
                    constant_values=0,
                )
                # delta_vel = vel - levelset_vel
                delta_vel = vel

                delta_vel_padded = jnp.pad(
                    delta_vel,
                    self._padding,
                    mode="constant",
                    constant_values=0,
                )

                delta_vel_dot_normal = jnp.dot(delta_vel, normal_hat)

                delta_vel_cross_normal = jnp.cross(
                    delta_vel_padded, norm_padded
                )  # works only for vectors of len

                norm_delta_vel_cross_normal = jnp.linalg.vector_norm(
                    delta_vel_cross_normal
                )

                omega = delta_vel_cross_normal / norm_delta_vel_cross_normal

                mu_prime = jnp.minimum(
                    self.mu, norm_delta_vel_cross_normal / delta_vel_dot_normal
                )

                normal_cross_omega = jnp.cross(
                    norm_padded, omega
                )  # works only for vectors of len 3

                tangent = (
                    (norm_padded + mu_prime * normal_cross_omega).at[: self.dim].get()
                )

                # sometimes tangent become nan if velocity is zero at initialization
                # which causes problems
                tangent = jnp.nan_to_num(tangent)

                return jax.lax.cond(
                    (delta_vel_dot_normal > 0.0),
                    lambda x: x - delta_vel_dot_normal * tangent,
                    # lambda x: x
                    # - delta_vel_dot_normal
                    # * normal_hat,  # uncomment for debug, no friction
                    lambda x: x,
                    vel,
                )

            vel = calculate_velocity(moment)
            node_moment = vel * mass

            vel_nt = calculate_velocity(moment_nt)
            node_moment_nt = vel_nt * mass

            return node_moment, node_moment_nt

        levelset_moment_stack, levelset_moment_nt_stack = vmap_selected_grid(
            # self.id_stack,
            grid.moment_nt_stack.at[self.id_stack].get(),
            grid.moment_stack.at[self.id_stack].get(),
            grid.mass_stack.at[self.id_stack].get(),
            grid.normal_stack.at[self.id_stack].get(),
        )
        new_moment_nt_stack = grid.moment_nt_stack.at[self.id_stack].set(
            levelset_moment_nt_stack
        )

        new_grid = eqx.tree_at(
            lambda state: (state.moment_nt_stack),
            grid,
            (new_moment_nt_stack),
        )
        return new_grid, self
