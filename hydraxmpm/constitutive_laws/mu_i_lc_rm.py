# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-

"""Implementation, state and functions for isotropic linear elastic material."""

from functools import partial
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from typing_extensions import Optional, Self, Union
import optimistix as optx
from ..common.types import TypeFloat, TypeFloatScalarPStack, TypeInt
from ..material_points.material_points import MaterialPoints
from ..utils.math_helpers import (
    get_dev_strain,
    get_inertial_number,
    get_scalar_shear_strain,
    get_sym_tensor,
    get_scalar_shear_stress
)
from .constitutive_law import ConstitutiveLaw
import equinox.internal as eqxi
import pickle

from ..utils.jax_helpers import debug_state

def yield_function(tau, p, mu_I):
    return tau - mu_I * p

def get_mu_I(I, mu_s, mu_d, I0):
    return mu_s + (mu_d - mu_s) * (1 / (1 + I0 / I))


def get_mu_I_regularized_exp(I, mu_s, mu_d, I0, pen, dgamma_dt):
    s = 1.0 / jnp.sqrt(dgamma_dt**2 + pen**2)
    return mu_s * s + (mu_d - mu_s) * (1.0 / (1.0 + I0 / I))


def get_I_phi(phi, phi_c, I_phi):
    return -I_phi * jnp.log(phi / phi_c)


def get_pressure(dgammadt, I, d, rho_p):
    return rho_p * ((dgammadt * d) / I) ** 2


def give_p(K, rho, rho_0):
    return K * (rho / rho_0 - 1.0)


def give_rho(K, rho_0, p):
    return rho_0 * ((p / K) + 1)


class MuI_LC_RM(ConstitutiveLaw):
    mu_s: TypeFloat
    mu_d: TypeFloat
    I_0: TypeFloat
    K: TypeFloat
    alpha_2: TypeFloat

    """
    (nearly) incompressible mu I

    Tensorial form similar to
    Jop, Pierre, Yoël Forterre, and Olivier Pouliquen. "A constitutive law for dense granular flows." Nature 441.7094 (2006): 727-730.

    mu I regularized by
    Franci, Alessandro, and Massimiliano Cremonesi. "3D regularized μ (I)-rheology for granular flows simulation." Journal of Computational Physics 378 (2019): 257-277.

    Pressure term by

    Salehizadeh, A. M., and A. R. Shafiei. "Modeling of granular column collapses with μ (I) rheology using smoothed particle hydrodynamic method." Granular Matter 21.2 (2019): 32.

    """

    def __init__(
        self: Self,
        mu_s: TypeFloat,
        mu_d: TypeFloat,
        I_0: TypeFloat,
        K: TypeFloat = 1.0,
        alpha: TypeFloat = 1e-4,
        **kwargs,
    ) -> Self:
        self.mu_s = mu_s

        self.mu_d = mu_d

        self.I_0 = I_0

        self.K = K
        self.alpha_2 = alpha * alpha
        super().__init__(**kwargs)

    def init_state(self: Self, material_points: MaterialPoints):
        p_0_stack = material_points.p_stack

        vmap_give_rho_ref = partial(
            jax.vmap,
            in_axes=(None, None, 0),
        )(give_rho)

        rho = vmap_give_rho_ref(self.K, self.rho_0, p_0_stack)

        rho_rho_0_stack = rho / self.rho_0

        vmap_update_ip = jax.vmap(fun=self.update_ip, in_axes=0)
        deps_dt_stack = material_points.deps_dt_stack

        new_stress_stack = vmap_update_ip(
            material_points.stress_stack,
            material_points.F_stack,
            deps_dt_stack,
            rho_rho_0_stack,
        )

        material_points = material_points.replace(stress_stack=new_stress_stack)

        return self.post_init_state(material_points, rho=rho, rho_0=self.rho_0)

    def update(
        self: Self,
        material_points: MaterialPoints,
        dt: TypeFloat,
        dim: Optional[TypeInt] = 3,
    ) -> Tuple[MaterialPoints, Self]:
        """Update the material state and particle stresses for MPM solver."""

        rho_stack = material_points.rho_stack

        rho_rho_0_stack = rho_stack / self.rho_0

        deps_dt_stack = material_points.deps_dt_stack

        vmap_update_ip = jax.vmap(fun=self.update_ip, in_axes=
                                  (0, 0, 0, 0))

        new_stress_stack = vmap_update_ip(
            material_points.stress_stack,
            material_points.F_stack,
            deps_dt_stack,
            rho_rho_0_stack
        )

        new_material_points = eqx.tree_at(
            lambda state: (state.stress_stack),
            material_points,
            (new_stress_stack),
        )

        return new_material_points, self

    def update_ip(
        self: Self,
        stress_prev,
        F,
        deps_dt,
        rho_rho_0
    ):
        rho_rho_0 = jnp.clip(rho_rho_0, 1.0, None)
        dgamma_dt = get_scalar_shear_strain(deps_dt)
        deps_dev_dt = get_dev_strain(deps_dt)
        p = jnp.log(rho_rho_0) * self.K
        p_next = jnp.clip(p, 1e-12, None)
        def residuals_cone(sol, args):
            hat_dgamma_dt = sol

            
            I = get_inertial_number( p_next,hat_dgamma_dt, self.d, self.rho_p)
            
            # eta_s = (self.mu_s * p)/ (dgamma_dt*dgamma_dt + self.alpha_2)
            mu_I = get_mu_I(I, self.mu_s, self.mu_d, self.I_0)
            eta = (mu_I * p)/ hat_dgamma_dt

            s_next = eta*deps_dev_dt
            tau_next = get_scalar_shear_stress(s_next)

            R  = yield_function(tau_next, p_next, mu_I)
            aux = (s_next,p_next)
            return R, aux

        def find_roots_cone():
            solver = optx.Newton(rtol=1e-1, atol=1e-3)

            sol = optx.root_find(
                residuals_cone,
                solver,
                jnp.clip(dgamma_dt, 1e-22, None),
                throw=False,
                has_aux=True,
                max_steps=20,
                options=dict(lower=1e-22,upper=None),
            )

            return sol.value
        
        def pull_to_cone():
            pmulti = find_roots_cone()

            R, aux = residuals_cone(pmulti, None)
            return aux, pmulti
        
        (s_next, p_next), pmulti = pull_to_cone()
        stress_next = -p_next * jnp.eye(3) + s_next
        return stress_next

    #     dgamma_dt = get_scalar_shear_strain(deps_dt)
    #     I_min = 1e-5
    #     I_max = 1e3
    #     # dgamma_dt = jnp.clip(dgamma_dt, 1e-22, None)

 

    #     alpha_r = 0.01

    #     alpha_s = 0.000001


    #     p = jnp.clip(p, 1e-22, None)
    #     dgamma_dt_min = (I_min / self.d) * jnp.sqrt(p / self.rho_p)
    #     dgamma_dt_max = (I_max / self.d) * jnp.sqrt(p / self.rho_p)
    #     dgamma_dt = jnp.clip(dgamma_dt, dgamma_dt_min, dgamma_dt_max)
    #     # p_max = self.rho_p * ((dgamma_dt * self.d) / I_min) ** 2
    #     # p_min = self.rho_p * ((dgamma_dt * self.d) / I_max) ** 2
    #     # p = jnp.clip(p, p_min, p_max)


    #     # eta_s = ((self.mu_s * p) * (1 - jnp.exp(-dgamma_dt / alpha_r)) )/ (dgamma_dt)

    #     eta_s = (self.mu_s * p)/ (dgamma_dt*dgamma_dt + self.alpha_2)
    #     delta_mu = self.mu_d - self.mu_s
        


    #     # eta_d = (p * delta_mu * dgamma_dt) / (
    #     #     (xi * jnp.sqrt(p) + dgamma_dt)
    #     #     * jnp.sqrt(dgamma_dt * dgamma_dt + alpha_s * alpha_s)
    #     # )
    # #    xi = self.I_0 * (p/self.rho_p ) ** (-0.5)
    
    #     xi = self.I_0 *jnp.sqrt(p/ self.rho_p)
       
    #     eta_d =  (self.d*p*delta_mu) / (
    #             xi + self.d*dgamma_dt
    #         )
       
    #     if self.error_check:
    #         eta_s = eqx.error_if(eta_s, ~jnp.isfinite(eta_s), "eta_s is non finite")
    #         eta_d = eqx.error_if(eta_d, ~jnp.isfinite(eta_d), "eta_d is non finite ")

    #     eta = eta_s + eta_d
        
    #     # eta = jnp.clip(eta, 1e-22, None)
    #     stress_next = -p * jnp.eye(3) + eta * deps_dev_dt

    #     if self.error_check:
    #         stress_next = eqx.error_if(
    #             stress_next,
    #             ~jnp.isfinite(stress_next).any(),
    #             "stress_next is non finite",
    #         )
        # return stress_next

    def get_dt_crit(self, material_points, cell_size, dt_alpha=0.5):
        """Get critical timestep of material poiints for stability."""

        def vmap_dt_crit(rho, vel):
            cdil = jnp.sqrt(self.K / rho)

            c = jnp.abs(vel) + cdil * jnp.ones_like(vel)
            return c

        c_stack = jax.vmap(vmap_dt_crit)(
            material_points.rho_stack,
            material_points.velocity_stack,
        )
        return (dt_alpha * cell_size) / jnp.max(c_stack)

