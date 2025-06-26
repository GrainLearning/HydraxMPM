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

from ..common.types import TypeFloat, TypeFloatScalarPStack, TypeInt
from ..material_points.material_points import MaterialPoints
from ..utils.math_helpers import (
    get_dev_strain,
    get_inertial_number,
    get_scalar_shear_strain,
    get_sym_tensor,
)
from .constitutive_law import ConstitutiveLaw


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


class MuI_LC(ConstitutiveLaw):
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
        alpha: TypeFloat = 1e-2,
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

        rho_rho_0_stack = material_points.rho_stack / self.rho_0

        # jax.debug.print(
        #     "rho_rho_0_stack: {} {}", rho_rho_0_stack.min(), rho_rho_0_stack.max()
        # )
        # jax.debug.breakpoint()

        deps_dt_stack = material_points.deps_dt_stack
        if self.error_check:
            deps_dt_stack = eqx.error_if(
                deps_dt_stack,
                jnp.isnan(deps_dt_stack).any(),
                "deps_dt_stack is nan",
                # on_error="breakpoint",
            )
        vmap_update_ip = jax.vmap(fun=self.update_ip, in_axes=0)

        new_stress_stack = vmap_update_ip(
            material_points.stress_stack,
            material_points.F_stack,
            deps_dt_stack,
            rho_rho_0_stack,
        )

        new_material_points = eqx.tree_at(
            lambda state: (state.stress_stack),
            material_points,
            (new_stress_stack),
        )

        return new_material_points, self

    # def update_ip(
    #     self: Self,
    #     stress_prev,
    #     F,
    #     deps_dt,
    #     rho_rho_0,
    # ):
    #     is_comp = rho_rho_0 > 1.0 + 1e-12

    #     if self.error_check:
    #         deps_dt = eqx.error_if(deps_dt, jnp.isnan(deps_dt).any(), "deps_dt is nan")
    #     # dgamma_dt = jnp.clip(dgamma_dt, 1e-12, None)
    #     # p = self.K * (rho_rho_0 - 1.0)

    #     # ensure numerical stability
    #     # p = jnp.clip(p, 1e-12, None)

    #     # is_flow = p > 1e-12

    #     def stress_update():
    #         rho_rho_0_ = jnp.where(~is_comp, 1.1, rho_rho_0)
    #         deps_dev_dt = get_dev_strain(deps_dt)

    #         dgamma_dt = get_scalar_shear_strain(deps_dt)
    #         # dgamma_dt = 0.0
    #         rho_rho_0_ = jnp.clip(rho_rho_0_, 1.0 + 1e-7, None)
    #         p = jnp.log(rho_rho_0_) * self.K

    #         # dgamma_dt = jnp.clip(dgamma_dt, 0.0, None)

    #         # dgamma_dt = jnp.nan_to_num(
    #         #     dgamma_dt,
    #         #     posinf=0.0,
    #         #     neginf=0.0,
    #         #     nan=0.0,
    #         # )
    #         # p = jnp.nan_to_num(
    #         #     p,
    #         #     posinf=0.0,
    #         #     neginf=0.0,
    #         #     nan=0.0,
    #         # )
    #         # eq (12) https://www.sciencedirect.com/science/article/pii/S0021999118307290

    #         delta_mu = self.mu_d - self.mu_s

    #         # avoid 0/0
    #         eta_d = (p * delta_mu * self.d) / (
    #             self.I_0 * jnp.sqrt(p / self.rho_p) + self.d * dgamma_dt
    #         )

    #         eta_s = (p * self.mu_s) / (jnp.sqrt(dgamma_dt * dgamma_dt + self.alpha_2))

    #         if self.error_check:
    #             eta_s = eqx.error_if(eta_s, jnp.isnan(eta_s).any(), "eta_s is nan")
    #             eta_d = eqx.error_if(eta_d, jnp.isnan(eta_d).any(), "eta_d is nan")

    #         eta = eta_s + eta_d
    #         stress_next = -p * jnp.eye(3) + eta * deps_dev_dt

    #         if self.error_check:
    #             stress_next = eqx.error_if(
    #                 stress_next, jnp.isnan(stress_next).any(), "stress_next is nan"
    #             )

    #         return stress_next

    #     # return jax.lax.cond(is_comp, stress_update, lambda: jnp.zeros(0, 0))
    #     return jax.lax.cond(is_comp, stress_update, lambda: jnp.zeros((3, 3)))
    def update_ip(
        self: Self,
        stress_prev,
        F,
        deps_dt,
        rho_rho_0,
    ):
        # is_comp = rho_rho_0 > 1.0 + 1e-12
        # rho_rho_0 = jnp.clip(rho_rho_0, 1.0 + 1e-6, None)

        if self.error_check:
            deps_dt = eqx.error_if(deps_dt, jnp.isnan(deps_dt).any(), "deps_dt is nan")

        deps_dev_dt = get_dev_strain(deps_dt)

        dgamma_dt = get_scalar_shear_strain(deps_dt)
        # eps = 1e-6
        # dgamma_dt = jnp.sqrt(dgamma_dt * dgamma_dt + eps * eps) - eps

        # eq (12) https://www.sciencedirect.com/science/article/pii/S0021999118307290
        rho_rho_0 = jnp.clip(rho_rho_0, 1.0, None)
        p = jnp.log(rho_rho_0) * self.K
        # dgamma_dt = jnp.clip(dgamma_dt, 1e-22, None)
        I_min = 1e-3
        I_max = 1000.0
        # p_max = self.rho_p * ((dgamma_dt * self.d) / I_min) ** 2
        # p_min = self.rho_p * ((dgamma_dt * self.d) / I_max) ** 2

        # # jax.debug.breakpoint()
        # p = jnp.clip(
        #     p, p_min, p_max
        # )  # make sure pressure is not -0, numerical artifact

        # p = eqx.error_if(p, p < p_min, " p < p_min")
        # p = eqx.error_if(p, p > p_max, " p > p_max")
        p = jnp.clip(p, 1e-22, None)  # make sure pressure is not -0, numerical artifact

        if self.error_check:
            p = eqx.error_if(p, jnp.isnan(p).any(), "dgamma_dt is nan")

        # p = jnp.abs(p)
        dgamma_dt_min = (I_min / self.d) * jnp.sqrt(p / self.rho_p)
        dgamma_dt_max = (I_max / self.d) * jnp.sqrt(p / self.rho_p)
        dgamma_dt = jnp.clip(dgamma_dt, 0.0, dgamma_dt_max)
        # dgamma_dt = jnp.clip(dgamma_dt, dgamma_dt_min, dgamma_dt_max)
        if self.error_check:
            dgamma_dt = eqx.error_if(
                dgamma_dt, jnp.isnan(dgamma_dt).any(), "dgamma_dt is nan"
            )
        delta_mu = self.mu_d - self.mu_s

        eta_d = (p * delta_mu * self.d) / (
            self.I_0 * jnp.sqrt(p / self.rho_p) + self.d * dgamma_dt
        )

        eta_s = (p * self.mu_s) / (jnp.sqrt(dgamma_dt * dgamma_dt + self.alpha_2))

        if self.error_check:
            eta_s = eqx.error_if(eta_s, jnp.isnan(eta_s).any(), "eta_s is nan")
            eta_d = eqx.error_if(eta_d, jnp.isnan(eta_d).any(), "eta_d is nan")

        eta = eta_s + eta_d
        stress_next = -p * jnp.eye(3) + eta * deps_dev_dt

        if self.error_check:
            stress_next = eqx.error_if(
                stress_next, jnp.isnan(stress_next).any(), "stress_next is nan"
            )
        return stress_next
        # dgamma_dt = jnp.clip(dgamma_dt, 1e-12, None)
        # p = self.K * (rho_rho_0 - 1.0)

        # ensure numerical stability
        # p = jnp.clip(p, 1e-12, None)

        # is_flow = p > 1e-12

        # def stress_update():

        # dgamma_dt = 0.0

        # dgamma_dt = jnp.clip(dgamma_dt, 0.0, None)

        # dgamma_dt = jnp.nan_to_num(
        #     dgamma_dt,
        #     posinf=0.0,
        #     neginf=0.0,
        #     nan=0.0,
        # )
        # p = jnp.nan_to_num(
        #     p,
        #     posinf=0.0,
        #     neginf=0.0,
        #     nan=0.0,
        # )

        # avoid 0/0

        # return stress_next

        # return jax.lax.cond(is_comp, stress_update, lambda: jnp.zeros(0, 0))
        # return jax.lax.cond(is_

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
