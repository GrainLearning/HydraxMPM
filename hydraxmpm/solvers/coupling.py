# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM
# -*- coding: utf-8 -*-
"""
Explanation:
    This module defines the `BodyCoupling` used for coupling different states.
    
    The `BodyCoupling` couples material point, grid, and solver (or numeric) states.
    
    It also includes the shape function mapping logic for mapping between these states.

    This allows for flexible coupling in multiphysics simulations. e.g., 
    rigid bodies, multi material contact, multi grid contact, fluid-structure interaction, and
    in future probably partioned solvers or thermal coupling.

"""

from ..shapefunctions.mapping import ShapeFunctionMapping
from typing import Optional
import equinox as eqx

class BodyCoupling(eqx.Module):
    """
    Body coupling static (stateless) logic between material points, grids, and solvers. This used used within the
    solver classes and looped over MPM operations (such as particle-to-grid, grid-to-particle, constitutive updates)

    The id's ending with _idx refer to the indices in the globl SimState tuples (See `SimState`).

    Attributes:
        shape_map: ShapeFunctionMapping instance for mapping between material points and grid nodes.
        p_idx: material points index in SimState.material_points
        g_idx: grid index in SimState.grids
        s_idx: solver or numerical state index in SimState.solvers (e.g., affine arrays in APIC)
        c_idx: constitutive law index in SimState.constitutive_laws
        skip_mpm_logic: If true, skips standard MPM logic within this coupling,but
            is still available for custom operations in force updates.
            (useful for rigid bodies or custom coupling implementations).
    """

    shape_map: ShapeFunctionMapping

    p_idx: int = eqx.field(static=True, default =0) 
    g_idx: int = eqx.field(static=True, default =0)

    # Optional in case of rigid bodies or custom solvers it may not be needed
    c_idx: Optional[int] = eqx.field(static=True, default =0) 
    s_idx: Optional[int] = eqx.field(static=True, default =0) 

    skip_mpm_logic: bool = eqx.field(static=True, default =False)

