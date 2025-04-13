# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-

"""Implementation, non-associated Drucker-Prager model with isotropic linear elasticity,
and linear hardening.

[1] de Souza Neto, E.A., Peric, D. and Owen, D.R., 2011. Computational methods for
plasticity.
"""

from functools import partial
from typing import Tuple, Optional, TypedDict, Unpack


import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from typing_extensions import Self, Union


from ..utils.math_helpers import (
    get_dev_strain,
    get_J2,
    get_sym_tensor_stack,
    get_volumetric_strain,
)


from .constitutive_law import ConstitutiveLaw, ConvergenceControlConfig


from ..common.types import (
    TypeFloat,
    TypeFloatMatrix3x3,
    TypeFloatMatrix3x3PStack,
    TypeFloatScalarPStack,
    TypeInt,
)

from ..material_points.material_points import MaterialPoints


def yield_function(sqrt_J2_tr, p, mu_1, mu_2, c):
    return sqrt_J2_tr - mu_1 * p - mu_2 * c


def give_rho(K, rho_0, p):
    return rho_0 * ((p / K) + 1)


def get_lin_elas_dev(eps_e_dev, G):
    return 2.0 * G * eps_e_dev


def get_lin_elas_vol(eps_e_vol, K):
    return K * eps_e_vol


class DruckerPrager(ConstitutiveLaw):
    r"""Non-associated Drucker-Prager model.

    The Drucker-Prager model is a smooth approximation to the Mohr-Coulomb model.

    This formulation is in small strain and elastic law is  isotropic linear elasticity.

    The implementation follows [1] with the exception that pressure and
    volumetric strain are positive in compression.

    [1] de Souza Neto, E.A., Peric, D. and Owen, D.R., 2011. Computational methods for
    plasticity.


    DP yield function
        q + M*p +M2*c

    plastic potential
        q + M_hat*p


    deps_v = pmulti*M_hat

    deps_dev = s/sqrt(2)J_2

    Attributes:
        E: Young's modulus.
        nu: Poisson's ratio.
        G: Shear modulus.
        K: Bulk modulus.
        M: Mohr-Coulomb friction parameter.
        M_hat: Mohr-Coulomb dilatancy parameter.
        M2: Mohr-Coulomb cohesion parameter.
        c0: Initial cohesion parameter.
        eps_acc_stack: Accumulated plastic strain for linear hardening
        eps_e_stack: Elastic strain tensor.
        H: Hardening modulus
    """

    E: TypeFloat
    nu: TypeFloat
    G: TypeFloat
    K: TypeFloat

    mu_1: TypeFloat
    mu_2: TypeFloat
    c0: TypeFloat
    mu_1_hat: TypeFloat
    H: TypeFloat

    eps_e_stack: Optional[TypeFloatMatrix3x3PStack] = None
    p_0_stack: Optional[TypeFloatScalarPStack] = None
    eps_p_acc_stack: Optional[TypeFloatScalarPStack] = None

    def __init__(
        self: Self,
        K: TypeFloat,
        nu: TypeFloat,
        mu_1: TypeFloat,
        mu_2: TypeFloat = 0.0,
        c0: TypeFloat = 0.0,
        mu_1_hat: TypeFloat = 0.0,
        H: TypeFloat = 0.0,
        **kwargs,
    ) -> Self:
        """Create a non-associated Drucker-Prager material model."""

        self.K = K
        self.E = 3 * K * (1 - 2 * nu)
        self.G = self.E / (2.0 * (1.0 + nu))
        self.nu = nu

        self.c0 = c0

        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.H = H

        self.mu_1_hat = mu_1_hat

        self.eps_e_stack = kwargs.get("eps_e_stack")

        self.eps_p_acc_stack = kwargs.get("eps_p_acc_stack")

        self.p_0_stack = kwargs.get("p_0_stack")
        super().__init__(**kwargs)

    def init_state(self: Self, material_points: MaterialPoints):
        p_0_stack = material_points.p_stack

        vmap_give_rho_ref = partial(
            jax.vmap,
            in_axes=(None, None, 0),
        )(give_rho)

        rho = vmap_give_rho_ref(self.K, self.rho_0, p_0_stack)

        eps_e_stack = jnp.zeros((material_points.num_points, 3, 3))
        eps_p_acc_stack = jnp.zeros_like(p_0_stack)

        return self.post_init_state(
            material_points,
            rho=rho,
            rho_0=self.rho_0,
            p_0_stack=p_0_stack,
            eps_e_stack=eps_e_stack,
            eps_p_acc_stack=eps_p_acc_stack,
        )

    def update(
        self: Self,
        material_points: MaterialPoints,
        dt: TypeFloat,
        dim: Optional[TypeInt] = 3,
    ) -> Tuple[MaterialPoints, Self]:
        """Update the material state and particle stresses for MPM solver."""

        deps_dt_stack = material_points.deps_dt_stack

        new_stress_stack, new_eps_e_stack, new_eps_p_acc_stack = (
            self.vmap_update_stress(
                deps_dt_stack * dt,
                self.p_0_stack,
                self.eps_e_stack,
                self.eps_p_acc_stack,
                material_points.rho_stack,
                dim,
            )
        )
        new_self = eqx.tree_at(
            lambda state: (state.eps_e_stack, state.eps_p_acc_stack),
            self,
            (new_eps_e_stack, new_eps_p_acc_stack),
        )
        new_material_points = eqx.tree_at(
            lambda state: (state.stress_stack),
            material_points,
            (new_stress_stack),
        )

        return new_material_points, new_self

    @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0, None), out_axes=(0, 0, 0))
    def vmap_update_stress(self, deps_next, p_0, eps_e_prev, eps_p_acc_prev, rho, dim):
        eps_e_tr = eps_e_prev + deps_next

        eps_e_v_tr = get_volumetric_strain(eps_e_tr)

        eps_e_d_tr = get_dev_strain(eps_e_tr, eps_e_v_tr)

        s_tr = get_lin_elas_dev(eps_e_d_tr, self.G)

        p_tr = get_lin_elas_vol(eps_e_v_tr, self.K)

        p_tr = p_tr + p_0

        # linear hardening
        c = self.c0 + self.H * eps_p_acc_prev

        sqrt_J2_tr = jnp.sqrt(get_J2(dev_stress=s_tr))

        yf = yield_function(sqrt_J2_tr, p_tr, self.mu_1, self.mu_2, c)

        # work around
        # we need to multiply residuals with this condition
        # jax lax traces both blocks
        # equinox sometimes does not converge on tracing

        # is_compression = p_tr > 0.0
        is_ep = yf > 0.0

        def elastic_update():
            """If yield function is negative, return elastic solution."""
            stress = s_tr - p_tr * jnp.eye(3)
            return stress, eps_e_tr, eps_p_acc_prev

        def pull_to_ys():
            """If yield function is negative, return elastic solution."""
            J2_non_zer0 = sqrt_J2_tr > 0.0

            def residuals_cone(sol, args):
                """Reduced system for non-associated flow rule."""
                pmulti = sol

                # solve non associated flow rules
                # volumetric plastic strain increment
                deps_p_v = pmulti * self.mu_1_hat

                # Trail isotropic linear elastic law
                p_next = p_tr - self.K * deps_p_v
                s_next = s_tr * (1.0 - (pmulti * self.G) / sqrt_J2_tr)

                sqrt_J2_next = sqrt_J2_tr - self.G * pmulti

                eps_p_acc_next = eps_p_acc_prev + self.mu_2 * pmulti

                c_next = self.c0 + self.H * eps_p_acc_next

                aux = p_next, s_next, eps_p_acc_next, sqrt_J2_next

                R = (
                    yield_function(sqrt_J2_next, p_next, self.mu_1, self.mu_2, c_next)
                    * is_ep
                    * J2_non_zer0
                )

                return R, aux

            def find_roots_cone():
                solver = optx.Newton(rtol=1e-1, atol=1e3)

                sol = optx.root_find(
                    residuals_cone,
                    solver,
                    0.0,
                    throw=True,
                    has_aux=True,
                    max_steps=256,
                    options=dict(
                        lower=0.0,
                        upper=1000.0,
                    ),
                )

                return sol.value

            def pull_to_cone():
                pmulti = find_roots_cone()

                R, aux = residuals_cone(pmulti, None)
                return aux, pmulti

            (p_next, s_next, eps_p_acc_next, sqrt_J2_next), pmulti = pull_to_cone()
            alpha = self.mu_2 / self.mu_1

            beta = jnp.clip(self.mu_2 / self.mu_1_hat, 0.0, 1.0)
            # beta = jnp.nan_to_num(self.mu_2 / self.mu_1_hat, posinf=0.0, neginf=0.0)
            J2_cone_negzero = sqrt_J2_next <= 0.0

            def residuals_apex(sol, args):
                """Reduced system for non-associated flow rule."""
                deps_p_v = sol

                eps_p_acc_next = eps_p_acc_prev + alpha * deps_p_v

                p_next = p_tr - self.K * deps_p_v

                c_next = self.c0 + self.H * eps_p_acc_next

                # ensure no division by zero when no hardening is present, & non associative flow rule
                R = beta * c_next + p_next
                R = p_next

                aux = p_next, jnp.zeros((3, 3)), eps_p_acc_next

                return R * is_ep * J2_cone_negzero, aux

            def find_roots_apex():
                solver = optx.Newton(rtol=1e-12, atol=1e-5)

                sol = optx.root_find(
                    residuals_apex,
                    solver,
                    0.0,
                    throw=True,
                    has_aux=True,
                    max_steps=512,
                    options=dict(
                        lower=-1.0,
                        upper=1.0,
                    ),
                )

                return sol.value

            def pull_to_apex():
                deps_v_p = find_roots_apex()

                R, aux = residuals_apex(deps_v_p, None)
                return aux

            p_next, s_next, eps_p_acc_next = jax.lax.cond(
                J2_cone_negzero,
                pull_to_apex,
                lambda: (p_next, s_next, eps_p_acc_next),
            )

            stress_next = s_next - p_next * jnp.eye(3)

            eps_e_v_next = (p_next - p_0) / self.K

            eps_e_d_next = s_next / (2.0 * self.G)

            eps_e_next = eps_e_d_next - (1.0 / 3) * eps_e_v_next * jnp.eye(3)

            return stress_next, eps_e_next, eps_p_acc_next

        ptol = 0.0
        stress_next, eps_e_next, eps_p_acc_next = jax.lax.cond(
            (rho >= self.rho_0),  # partciles disconnect (called stress-free assumption)
            lambda: jax.lax.cond(is_ep, pull_to_ys, elastic_update),
            lambda: (-ptol * jnp.ones((3, 3)), jnp.zeros((3, 3)), eps_p_acc_prev),
        )
        # stress_next, eps_e_next, eps_p_acc_next = jax.lax.cond(
        #     is_ep * , pull_to_ys, elastic_update
        # )

        return stress_next, eps_e_next, eps_p_acc_next

    @property
    def M(self):
        return self.mu_1 * jnp.sqrt(3)

    def get_dt_crit(self, material_points, cell_size, dt_alpha=0.5):
        """Get critical timestep of material poiints for stability."""

        def vmap_dt_crit(p, rho, vel):
            cdil = jnp.sqrt((self.K + (4 / 3) * self.G) / rho)

            c = jnp.abs(vel) + cdil * jnp.ones_like(vel)
            return c

        c_stack = jax.vmap(vmap_dt_crit)(
            material_points.p_stack,
            material_points.rho_stack,
            material_points.velocity_stack,
        )
        return (dt_alpha * cell_size) / jnp.max(c_stack)
