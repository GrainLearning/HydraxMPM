# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-

import equinox as eqx
import jax.numpy as jnp

import jax

class ConstitutiveLawState(eqx.Module):
    pass


class ConstitutiveLaw(eqx.Module):
    requires_F_reset: bool = eqx.field(static=True, default=False)
    

    def remove_accumulated_shear(self, mp_state):
        """
        This is necessary for stability in MPM

        When shear components accumulate in F over time,
        the condition number of F worsens, leading to
        numerical issues when computing volume change within g2p update

        Remove shear component from deformation gradient F.
        
        - Called within MPM update if `requires_F_reset` is True.
        - Used for fluids to avoid shear history accumulation.
        - Also hypoelastic solids that do not depend on F to compute stress
        
        """
        if not self.requires_F_reset:
            return mp_state
        
        dim = mp_state.dim
        def compute_cbar(F):
            J = jnp.linalg.det(F)
            if dim == 2:

                scale = jnp.sqrt(J)
                return jnp.diag(jnp.array([scale, scale, 0.0]))
            else:
                # 3D cbar element-wise cube root
                scale = jnp.cbrt(J)
                return scale * jnp.eye(3)
            

        new_F_stack = jax.vmap(compute_cbar)(mp_state.F_stack)

        if mp_state.F_store_stack is not None:
            new_F_store_stack = mp_state.F_stack
        else:
            new_F_store_stack = None

        new_mp = eqx.tree_at(lambda m: (m.F_stack,m.F_store_stack), mp_state, (new_F_stack, new_F_store_stack))


        return new_mp
