# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-


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
    get_inertial_number_stack,
    get_scalar_shear_strain,
    get_sym_tensor,
    get_pressure,
)
from .constitutive_law import ConstitutiveLaw


def give_p_implicit(phi_phi_0, dgamma_dt, K, I_phi, d, rho_p, p_prev=1e-12):


    def give_p_correction(p, args):
        I = get_inertial_number(p, dgamma_dt, d, rho_p)
 
        R = p / K - I / I_phi - jnp.log(phi_phi_0)
        return R
        # return R**2

    solver = optx.Newton(rtol=1e-1, atol=1e-4)
    # sol = optx.minimise(
    sol = optx.root_find(
        give_p_correction,
        solver,
        p_prev,
        throw=False,
        has_aux=False,
        max_steps=100,
        options=dict(lower=1e-8),
    )
    return sol.value


def phi_linear_comp_I(p, K, I, I_phi, phi_0):
    return phi_0 * jnp.exp(-I / I_phi) * jnp.exp(p / K)


class MuI_LC_PhiI(ConstitutiveLaw):
    mu_s: TypeFloat
    mu_d: TypeFloat
    I_0: TypeFloat
    K: TypeFloat
    I_phi: TypeFloat
    alpha_2: TypeFloat = 1e-4

    def __init__(
        self: Self,
        mu_s: TypeFloat,
        mu_d: TypeFloat,
        I_0: TypeFloat,
        I_phi: TypeFloat,
        K: TypeFloat = 1.0,
        alpha: TypeFloat = 1e-4,
        **kwargs,
    ) -> Self:
        self.mu_s = mu_s

        self.mu_d = mu_d

        self.I_0 = I_0

        self.I_phi = I_phi
        self.K = K
        self.alpha_2 = alpha * alpha

        super().__init__(**kwargs)

    def init_state(self: Self, material_points: MaterialPoints):
        p_0_stack = material_points.p_stack

        # vmap_give_rho_ref = partial(
        #     jax.vmap,
        #     in_axes=(None, None, 0),
        # )(give_rho)
        p_0_stack = jnp.clip(p_0_stack, 1e-12, None)

        I = get_inertial_number_stack(
            p_0_stack, material_points.dgamma_dt_stack, self.d, self.rho_p
        )

        phi_0 = self.rho_0 / self.rho_p

        phi_stack = phi_0 * jnp.exp(p_0_stack / self.K) * jnp.exp(-I / self.I_phi)

        rho_stack = phi_stack * self.rho_p

        vmap_update_ip = jax.vmap(fun=self.update_ip, in_axes=0)

        deps_dt_stack = material_points.deps_dt_stack

        new_stress_stack = vmap_update_ip(
            material_points.stress_stack,
            material_points.F_stack,
            deps_dt_stack,
            phi_stack,
        )

        material_points = material_points.replace(stress_stack=new_stress_stack)

        return self.post_init_state(material_points, rho=rho_stack, rho_0=self.rho_0)

    def update(
        self: Self,
        material_points: MaterialPoints,
        dt: TypeFloat,
        dim: Optional[TypeInt] = 3,
    ) -> Tuple[MaterialPoints, Self]:
        """Update the material state and particle stresses for MPM solver."""

        deps_dt_stack = material_points.deps_dt_stack

        vmap_update_ip = jax.vmap(fun=self.update_ip, in_axes=0)

        new_stress_stack = vmap_update_ip(
            material_points.stress_stack,
            material_points.F_stack,
            deps_dt_stack,
            material_points.phi_stack(self.rho_p),
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
        phi,
    ):
        
        deps_dev_dt = get_dev_strain(deps_dt)

        dgamma_dt = get_scalar_shear_strain(deps_dt)
        
        phi_0 = self.rho_0 / self.rho_p
        phi_phi_0 = phi / phi_0
        
        # is connected and sheared
        is_connected = (phi_phi_0 > 1.0)* (dgamma_dt > 1e-22)
        # used for initial gues of p
        p_prev = get_pressure(stress_prev)
        p_prev = jnp.clip(p_prev, 1e-12)
        def viscoplastic_update():
            # https://github.com/patrick-kidger/optimistix/issues/132
            phi_phi_0_safe = jnp.where(~is_connected, 1.0, phi_phi_0)
            
            p = give_p_implicit(
                phi_phi_0_safe, dgamma_dt, self.K, self.I_phi, self.d, self.rho_p, p_prev
            )
            eta_s = (self.mu_s * p)/ jnp.sqrt(dgamma_dt*dgamma_dt + self.alpha_2)
            
            delta_mu = self.mu_d - self.mu_s
            
            xi = self.I_0 *jnp.sqrt(p/ self.rho_p)
       
            eta_d =  (self.d*p*delta_mu) / (
                    xi + self.d*dgamma_dt
                )
            
            eta = eta_s + eta_d
        
            # eta = jnp.clip(eta, 1e-22, None)
            stress_next = -p * jnp.eye(3) + eta * deps_dev_dt
            return stress_next
        
        ptol = 0.0
        stress_next = jax.lax.cond(
            is_connected,  # partciles disconnect (called stress-free assumption)
            lambda: viscoplastic_update(),
            lambda: -ptol * jnp.ones((3, 3)),
        )
        return stress_next
        # 
        # # jax.debug.print("phi_0: {}", self.I_phi)


        # xi = self.I_0 * (self.rho_p * self.d * self.d) ** (-0.5)

        # alpha_r = 0.01

        # alpha_s = 0.000001

        # p = jnp.clip(p, 1e-22, None)
        # dgamma_dt = jnp.clip(dgamma_dt, 1e-22, None)

        # eta_s = (self.mu_s * p) * ((1 - jnp.exp(-dgamma_dt / alpha_r)) / (dgamma_dt))

        # delta_mu = self.mu_d - self.mu_s

        # eta_d = (p * delta_mu * dgamma_dt) / (
        #     (xi * jnp.sqrt(p) + dgamma_dt)
        #     * jnp.sqrt(dgamma_dt * dgamma_dt + alpha_s * alpha_s)
        # )

        # eta = eta_s + eta_d
        # stress_next = -p * jnp.eye(3) + eta * deps_dev_dt

        # if self.error_check:
        #     stress_next = eqx.error_if(
        #         stress_next, jnp.isnan(stress_next).any(), "stress_next is nan"
        #     )
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

    def compute_phi_p_I(self, p, I):
        return phi_linear_comp_I(
            p=p,
            K=self.K,
            I=I,
            I_phi=self.I_phi,
            phi_0=self.rho_0 / self.rho_p,
        )
