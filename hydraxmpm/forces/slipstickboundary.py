# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-

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


def create_boundary_slice(axis, side, grid_size, thickness):
    if axis == 0:  # x-axis
        if side == 0:
            return (slice(0, thickness), slice(None))
        else:
            return (slice(grid_size[0] - thickness, None), slice(None))
    else:  # y-axis
        if side == 0:
            return (slice(None), slice(0, thickness))
        else:
            return (slice(None), slice(grid_size[1] - thickness - 1, None))


def stick_all(moment_nt, index):
    """Stick all directions."""
    moment_nt = moment_nt.at[index].set(0.0)
    return moment_nt


def slip_positive_normal(moment_nt, index):
    """Slip in min direction of inward normal."""
    moment_nt = moment_nt.at[index].min(0.0)
    return moment_nt


def slip_negative_normal(moment_nt, index):
    """Slip in max direction of outward normal."""
    moment_nt = moment_nt.at[index].max(0.0)
    return moment_nt


class SlipStickBoundary(Force):
    thickness: TypeInt = eqx.field(static=True)

    x0: Optional[str] = eqx.field(static=True)
    x1: Optional[str] = eqx.field(static=True)
    y0: Optional[str] = eqx.field(static=True)
    y1: Optional[str] = eqx.field(static=True)
    z0: Optional[str] = eqx.field(static=True)
    z1: Optional[str] = eqx.field(static=True)

    x0_slice: Optional[Any] = eqx.field(static=True, default=None)
    x1_slice: Optional[Any] = eqx.field(static=True, default=None)
    y0_slice: Optional[Any] = eqx.field(static=True, default=None)
    y1_slice: Optional[Any] = eqx.field(static=True, default=None)
    z0_slice: Optional[Any] = eqx.field(static=True, default=None)
    z1_slice: Optional[Any] = eqx.field(static=True, default=None)

    # dim: int = eqx.field(static=True)
    # _padding: tuple = eqx.field(init=False, static=True, repr=False)

    def __init__(
        self,
        x0: Optional[str] = None,
        x1: Optional[str] = None,
        y0: Optional[str] = None,
        y1: Optional[str] = None,
        z0: Optional[str] = None,
        z1: Optional[str] = None,
        thickness: Optional[TypeInt] = 2,
        **kwargs,
    ):
        self.thickness = thickness
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.z0 = z0
        self.z1 = z1

        self.x0_slice = kwargs.get("x0_slice", None)
        self.x1_slice = kwargs.get("x1_slice", None)
        self.y0_slice = kwargs.get("y0_slice", None)
        self.y1_slice = kwargs.get("y1_slice", None)
        self.z0_slice = kwargs.get("z0_slice", None)
        self.z1_slice = kwargs.get("z1_slice", None)

    def init_ids(self: Self, grid: Grid, dim: int, **kwargs):
        assert dim < 3, "SlipStickBoundary only supports 2D grids. (For now)"
        # TODO ADD SLIP STICK FOR 3D
        # TODO modify node type
        x0_slice = create_boundary_slice(
            axis=0, side=0, grid_size=grid.grid_size, thickness=self.thickness
        )
        x1_slice = create_boundary_slice(
            axis=0, side=1, grid_size=grid.grid_size, thickness=self.thickness
        )

        y0_slice = create_boundary_slice(
            axis=1, side=0, grid_size=grid.grid_size, thickness=self.thickness
        )
        y1_slice = create_boundary_slice(
            axis=1, side=1, grid_size=grid.grid_size, thickness=self.thickness
        )
        new_self = SlipStickBoundary(
            x0=self.x0,
            x1=self.x1,
            y0=self.y0,
            y1=self.y1,
            z0=self.z0,
            z1=self.z1,
            thickness=self.thickness,
            x0_slice=x0_slice,
            x1_slice=x1_slice,
            y0_slice=y0_slice,
            y1_slice=y1_slice,
        )

        return new_self, grid

    def apply_on_grid(
        self: Self,
        material_points: Optional[MaterialPoints] = None,
        grid: Optional[Grid] = None,
        step: Optional[TypeInt] = 0,
        dt: Optional[TypeFloat] = 0.01,
        dim: TypeInt = 3,
        **kwargs: Any,
    ):
        new_moment_nt_stack = grid.moment_nt_stack.reshape((*grid.grid_size, dim))

        if self.x0 == "stick":
            new_moment_nt_stack = stick_all(new_moment_nt_stack, self.x0_slice)
        elif self.x0 == "slip":
            index = (*self.x0_slice, 0)
            new_moment_nt_stack = slip_negative_normal(new_moment_nt_stack, index)  #

        if self.x1 == "stick":
            new_moment_nt_stack = stick_all(new_moment_nt_stack, self.x1_slice)
        elif self.x1 == "slip":
            index = (*self.x1_slice, 0)
            new_moment_nt_stack = slip_positive_normal(new_moment_nt_stack, index)

        if self.y0 == "stick":
            new_moment_nt_stack = stick_all(new_moment_nt_stack, self.y0_slice)
        elif self.y0 == "slip":
            index = (*self.y0_slice, 1)
            new_moment_nt_stack = slip_negative_normal(new_moment_nt_stack, index)  #

        if self.y1 == "stick":
            new_moment_nt_stack = stick_all(new_moment_nt_stack, self.y1_slice)
        elif self.y1 == "slip":
            index = (*self.y1_slice, 1)
            new_moment_nt_stack = slip_positive_normal(new_moment_nt_stack, index)

        new_moment_nt_stack = new_moment_nt_stack.reshape(-1, dim)

        new_grid = eqx.tree_at(
            lambda state: (state.moment_nt_stack),
            grid,
            (new_moment_nt_stack),
        )
        return new_grid, self
