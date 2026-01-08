# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-
"""
Explaination:
    This module defines base state (`BaseSolverState`) and logic (`BaseSolver`) classes 
    which is inherited by other solvers.

    Mainly used for consistent type checking and dependency injection.
"""

import equinox as eqx


class BaseSolverState(eqx.Module):
    pass

class BaseSolver(eqx.Module):
    pass