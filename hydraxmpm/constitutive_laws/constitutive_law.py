# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-

import equinox as eqx
import jax.numpy as jnp

import jax
from jaxtyping import Float, Array
from typing import Tuple, Optional, Self


class Convergence(eqx.Module):
    residuals: Float[Array, "..."]
    unknowns: Float[Array, "..."]
    jacobians: Float[Array, "..."]


class ConstitutiveLawState(eqx.Module):
    pass


def dummy_convergence(
    debug_convergence: bool,
    NUM_NEWTON_ITERS: int,
    NUM_RESIDUALS: int,
    NUM_UNKNOWNS: int,
) -> Optional[Convergence]:

    if not debug_convergence:
        return None

    convergence = Convergence(
        residuals=jnp.zeros((NUM_NEWTON_ITERS, NUM_RESIDUALS)),
        unknowns=jnp.zeros((NUM_NEWTON_ITERS, NUM_UNKNOWNS)),
        jacobians=jnp.zeros((NUM_NEWTON_ITERS, NUM_UNKNOWNS, NUM_UNKNOWNS)),
    )
    return convergence


class ConstitutiveLaw(eqx.Module):
    debug_convergence: bool = eqx.field(static=True)
