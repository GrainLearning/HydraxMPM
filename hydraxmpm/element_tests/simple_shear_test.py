# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-

import jax.numpy as jnp
import equinox as eqx
from .driver import ElementTestDriver
import jax
from typing import Optional
from jaxtyping import Float, Array


class SimpleShearTest(eqx.Module):
    solver: ElementTestDriver
    confine: float | Float[Array, ""]
    shear_rate: float | Float[Array, "..."]
    dt: float | Float[Array, "..."]
    num_steps: Optional[int] = 2000
    stride: int = 1
    start_static: bool = False

    @property
    def shear_strain_stack(self):
        # Handle scalar vs array case for strain calculation
        if hasattr(self.dt, "shape") and self.dt.shape:
            # Array case: Cumulative sum of increments
            incr_strain = self.shear_rate * self.dt
            return jnp.cumsum(incr_strain)[:: self.stride]
        else:
            # Scalar case
            return jnp.linspace(
                0, self.shear_rate * self.dt * self.num_steps, self.num_steps
            )[:: self.stride]

    def run(self, mp_init, law_init):

        def shear_strain_input(idx):
            rate = (
                self.shear_rate[idx]
                if (hasattr(self.shear_rate, "ndim") and self.shear_rate.ndim > 0)
                else self.shear_rate
            )
            d_t = (
                self.dt[idx]
                if (hasattr(self.dt, "ndim") and self.dt.ndim > 0)
                else self.dt
            )
            return rate, d_t

        def scan_outer(carry, idx):
            mp, law = carry

            curr_rate, curr_dt = shear_strain_input(idx)

            # Strictly isochoric (constant volume) simple shear velocity gradient tensor:
        
            L_next = jnp.zeros((1, 3, 3))
            L_next = L_next.at[0, 0, 2].set(curr_rate)

            mp_next, law_next = self.solver.step(mp, law, L_next, curr_dt)

            return (mp_next, law_next), (mp_next, law_next)

        # Runs strides and evaluates static initialization phases if flagged
        def static_run(carry, idx):
            if self.start_static:
                return jax.lax.cond(
                    idx == 0,
                    lambda: (carry, carry),
                    lambda: scan_outer(carry, idx),
                )
            return scan_outer(carry, idx)

        def run_inner(carry, indices_chunk):
            segment_final_state, _ = jax.lax.scan(static_run, carry, indices_chunk)
            return segment_final_state, segment_final_state

        num_steps = (
            len(self.shear_rate)
            if (hasattr(self.shear_rate, "ndim") and self.shear_rate.ndim > 0)
            else self.num_steps
        )
        all_indices = jnp.arange(num_steps)

        num_snapshots = num_steps // self.stride
        reshaped_indices = all_indices.reshape((num_snapshots, self.stride))

        _, trajectory = jax.lax.scan(run_inner, (mp_init, law_init), reshaped_indices)

        if mp_init.num_points == 1:

            def _squeeze_axis1(x):
                try:
                    if hasattr(x, "ndim") and x.ndim >= 2 and x.shape[1] == 1:
                        return jnp.squeeze(x, axis=1)
                except Exception:
                    pass
                return x

            trajectory = jax.tree.map(_squeeze_axis1, trajectory)

        return trajectory