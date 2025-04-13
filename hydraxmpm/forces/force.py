# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-

from typing import Optional, Self, Tuple, Any

from ..common.base import Base
from ..common.types import TypeFloat, TypeInt
from ..grid.grid import Grid
from ..material_points.material_points import MaterialPoints


class Force(Base):
    """Force state for the material properties."""

    def apply_on_grid(
        self: Self,
        material_points: Optional[MaterialPoints] = None,
        grid: Optional[Grid] = None,
        step: Optional[TypeInt] = 0,
        dt: Optional[TypeFloat] = 0.01,
        dim: TypeInt = 3,
        **kwargs: Any,
    ) -> Tuple[Grid, Self]:
        return grid, self

    def apply_on_points(
        self: Self,
        material_points: Optional[MaterialPoints] = None,
        grid: Optional[Grid] = None,
        step: Optional[TypeInt] = 0,
        dt: Optional[TypeFloat] = 0.01,
        dim: TypeInt = 3,
        **kwargs: Any,
    ) -> Tuple[MaterialPoints, Self]:
        return material_points, self
