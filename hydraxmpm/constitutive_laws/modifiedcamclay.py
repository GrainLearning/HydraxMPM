# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp

import optimistix as optx
from typing_extensions import Optional, Self, Tuple

from ..common.types import (
    TypeFloat,
    TypeFloatScalarPStack,
    TypeFloatMatrixPStack,
    TypeInt,
)
from ..material_points.material_points import MaterialPoints
from ..utils.math_helpers import (
    get_dev_strain,
    get_dev_strain_stack,
    get_dev_stress,
    get_pressure,
    get_q_vm,
    get_volumetric_strain,
)
from .constitutive_law import ConstitutiveLaw, ConvergenceControlConfig


def yield_function(p_hat, px_hat, q, M):
    return ((q * q) / (M * M)) + p_hat * p_hat - px_hat * p_hat


def get_p_hat(deps_e_v, kap, p_hat_prev):
    """Compute non-linear pressure."""

    p_hat = p_hat_prev / (1.0 - (1.0 / kap) * deps_e_v + 1e-12)
    return jnp.clip(p_hat, 1.0, None)


def get_px_hat_mcc(px_hat_prev, cp, deps_p_v):
    """Compute non-linear pressure."""

    px_hat = px_hat_prev / (1.0 - (1.0 / cp) * deps_p_v + 1e-12)
    return jnp.clip(px_hat, 1.0, None)


def get_s(deps_e_d, G, s_prev):
    return 2.0 * G * deps_e_d + s_prev


def get_K(kap, p_hat, K_min=None, K_max=None):
    p_hat = jnp.clip(p_hat, 1.0, None)
    K = (1.0 / kap) * (p_hat)
    K = jnp.clip(K, K_min, K_max)
    return K


def get_G(nu, K):
    G = (3 * (1 - 2 * nu) / (2 * (1 + nu))) * K
    return G


class ModifiedCamClay(ConstitutiveLaw):
    nu: TypeFloat
    M: TypeFloat
    R: TypeFloat
    lam: TypeFloat
    kap: TypeFloat
    p_t: TypeFloat = 0.0
    ln_N: Optional[TypeFloat] = None

    K_min: Optional[TypeFloat] = None
    K_max: Optional[TypeFloat] = None

    px_hat_stack: Optional[TypeFloatScalarPStack] = None
    stress_0_stack: Optional[TypeFloatMatrixPStack] = None

    settings: ConvergenceControlConfig

    def __init__(
        self: Self,
        nu: TypeFloat,
        M: TypeFloat,
        R: TypeFloat,
        lam: TypeFloat,
        kap: TypeFloat,
        K_min: TypeFloat = None,
        K_max: TypeFloat = None,
        ln_N: Optional[TypeFloat] = None,
        p_t: Optional[TypeFloat] = 0.0,
        settings: Optional[dict | ConvergenceControlConfig] = None,
        **kwargs,
    ) -> Self:
        self.nu = nu

        self.M = M

        self.R = R

        self.lam = lam

        self.kap = kap

        self.p_t = p_t

        self.ln_N = ln_N

        self.K_min = K_min

        self.K_max = K_max

        self.eps_e_stack = kwargs.get("eps_e_stack")

        self.px_hat_stack = kwargs.get("px_hat_stack")

        self.stress_0_stack = kwargs.get("stress_0_stack")

        # settings used for convergence control
        if settings is None:
            settings = dict()
        if isinstance(settings, dict):
            self.settings = ConvergenceControlConfig(
                rtol=settings.get("rtol", 1e-3),
                atol=settings.get("atol", 1e3),
                max_iter=settings.get("max_iter", 1000),
                throw=settings.get("throw", True),
                # plastic multiplier and volumetric strain, respectively
                lower_bound=settings.get("lower_bound", (0, -1.0)),
                upper_bound=settings.get("upper_bound", (1000.0, 1.0)),
            )
        else:
            self.settings = settings
        del settings

        super().__init__(**kwargs)

    def init_state(self: Self, material_points: MaterialPoints):
        # tension cuttoff
        # at p_0=px = 1 Pa
        # then ln_N = ln_sl
        # i.e., normall consolidation line and swelling line both are zero

        # pressure has to be at least 1 Pa
        def clip_(p):
            return jnp.clip(p, 1.0, None)

        stress_0_stack = material_points.stress_stack
        p_0_stack = material_points.p_stack

        ln_N = self.ln_N

        rho_0 = self.rho_0
        if self.ln_N is None:
            phi_N = self.rho_0 / self.rho_p
            N = 1.0 / phi_N
            ln_N = jnp.log(N)
        else:
            N = jnp.exp(self.ln_N)
            phi_N = 1.0 / N
            rho_0 = self.rho_p * phi_N

        ln_v0 = jax.vmap(self.get_ln_v0, in_axes=(0, None))(stress_0_stack, ln_N)

        rho = self.rho_p / jnp.exp(ln_v0)

        px_hat_stack = p_0_stack * self.R

        eps_e_stack = jnp.zeros((material_points.num_points, 3, 3))
        return self.post_init_state(
            material_points,
            rho_0=rho_0,
            rho=rho,
            ln_N=ln_N,
            stress_0_stack=stress_0_stack,
            px_hat_stack=px_hat_stack,
            eps_e_stack=eps_e_stack,
        )

    def update(
        self: Self,
        material_points: MaterialPoints,
        dt: TypeFloat,
        dim: Optional[TypeInt] = 3,
    ) -> Tuple[MaterialPoints, Self]:
        """Update the material state and particle stresses for MPM solver."""

        deps_dt_stack = material_points.deps_dt_stack

        new_stress_stack, new_eps_e_stack, new_px_hat_stack = self.vmap_update_ip(
            deps_dt_stack * dt,
            self.eps_e_stack,
            material_points.stress_stack,
            self.px_hat_stack,
            self.stress_0_stack,
            material_points.rho_stack,
        )
        new_self = eqx.tree_at(
            lambda state: (state.eps_e_stack, state.px_hat_stack),
            self,
            (new_eps_e_stack, new_px_hat_stack),
        )

        new_material_points = eqx.tree_at(
            lambda state: (state.stress_stack),
            material_points,
            (new_stress_stack),
        )

        # new_self = new_self.post_update(new_stress_stack, deps_dt_stack, dt)
        return new_material_points, new_self

    @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0, 0), out_axes=(0, 0, 0))
    def vmap_update_ip(
        self: Self,
        deps_next,
        eps_e_prev,
        stress_prev,
        px_hat_prev,
        stress_0,
        rho,
    ):
        # reference stress
        p_0 = get_pressure(stress_0)
        s_0 = get_dev_stress(stress_0, pressure=p_0)

        p_prev = get_pressure(stress_prev)

        p_hat_prev = p_prev + self.p_t

        s_prev = get_dev_stress(stress_prev, pressure=p_prev)

        deps_e_v_tr = get_volumetric_strain(deps_next)

        p_hat_tr = get_p_hat(deps_e_v_tr, self.kap, p_hat_prev)

        deps_e_d_tr = get_dev_strain(deps_next, deps_e_v_tr)

        K_tr = get_K(self.kap, p_hat_tr, self.K_min, self.K_max)

        G_tr = get_G(self.nu, K_tr)

        s_tr = get_s(deps_e_d_tr, G_tr, s_prev)

        q_tr = get_q_vm(dev_stress=s_tr)

        yf = yield_function(p_hat_tr, px_hat_prev, q_tr, self.M)
        is_ep = yf > 0.0

        def elastic_update():
            stress_next = s_tr - (p_hat_tr - self.p_t) * jnp.eye(3)
            eps_e_tr = eps_e_prev + deps_next
            return stress_next, eps_e_tr, px_hat_prev

        def pull_to_ys():
            def residuals(sol, args):
                pmulti, deps_p_v = sol

                p_hat_next = get_p_hat(deps_e_v_tr - deps_p_v, self.kap, p_hat_prev)

                K_next = get_K(self.kap, p_hat_next, self.K_min, self.K_max)

                G_next = get_G(self.nu, K_next)

                factor = 1 / (1 + 6.0 * G_next * pmulti)

                s_next = s_tr * factor

                q_next = q_tr * factor

                px_hat_next = get_px_hat_mcc(px_hat_prev, self._cp, deps_p_v)

                deps_v_p_fr = pmulti * (2.0 * p_hat_next - px_hat_next) * self.M**2
                yf_next = yield_function(p_hat_next, px_hat_next, q_next, self.M)

                yf_next = yf_next

                deps_scale = p_hat_tr * self.M**2

                yf_scale = p_hat_tr * p_hat_tr
                R = jnp.array(
                    [yf_next / yf_scale, (deps_v_p_fr - deps_p_v) / deps_scale]
                )

                aux = (p_hat_next, s_next, px_hat_next, G_next, K_next)

                return R, aux

            def find_roots():
                """Find roots of the residuals function."""

                init_val = jnp.array([0.0, 0.0])

                solver = optx.Newton(
                    rtol=self.settings.rtol,
                    atol=self.settings.atol,
                )

                sol = optx.root_find(
                    residuals,
                    solver,
                    init_val,
                    throw=self.settings.throw,
                    has_aux=True,
                    max_steps=self.settings.max_iter,
                    options=dict(
                        lower=jnp.array(self.settings.lower_bound),
                        upper=jnp.array(self.settings.upper_bound),
                    ),
                )
                return sol.value

            pmulti_curr, deps_p_v_next = find_roots()

            R, aux = residuals([pmulti_curr, deps_p_v_next], None)

            p_hat_next, s_next, px_hat_next, G_next, K_next = aux

            p_next = p_hat_next - self.p_t

            stress_next = s_next - (p_next) * jnp.eye(3)

            eps_e_v_next = (p_next - p_0) / K_next

            eps_e_d_next = (s_next - s_0) / (2.0 * G_next)

            eps_e_next = eps_e_d_next - (1.0 / 3) * eps_e_v_next * jnp.eye(3)

            return stress_next, eps_e_next, px_hat_next

        stress_next, eps_e_next, px_hat_next = jax.lax.cond(
            (rho >= self.rho_0),
            lambda: jax.lax.cond(is_ep, pull_to_ys, elastic_update),
            lambda: (0.0 * jnp.eye(3), eps_e_prev, px_hat_prev),
        )

        return stress_next, eps_e_next, px_hat_next

    @property
    def GAMMA(self):
        """Reference (natural) logarithmic specific volume of critical state line (CSL) at 1kPa

        #     Returns ln_GAMMA
        #"""

        return self.ln_N - (self.lam - self.kap) * jnp.log(2)

    @property
    def _cp(self):
        return self.lam - self.kap

    def CSL(self, p):
        """Equation for critical state line (CSL) in double log specific volume/pressure space (ln v - ln p) space.

        Returns specific volume (not logaritm)
        """
        return jnp.exp(self.GAMMA - self.lam * jnp.log(p))

    def CSL_q_p(self, p):
        """Equation for critical state line (CSL) in scalar shear stress- pressure (q - p) space.

        Returns specific volume (not logaritm)
        """
        return p * self.M

    def ICL(self, p):
        """Equation for isotropic compression line (ICL) in double log specific volume/pressure space (ln v - ln p) space.

        Returns specific volume (not logaritm)
        """
        return jnp.exp(self.ln_N - self.lam * jnp.log(p))

    def get_p_0(self, ln_v0):
        return self.px_hat_stack * jnp.exp(
            (ln_v0, self.ln_N + self.lam * self.px_hat_stack) / self.kap
        ) ** (-1)

    def get_ln_v0(self, stress, ln_N=None):
        p = get_pressure(stress)

        if ln_N is None:
            ln_N = self.ln_N

        ln_v = ln_N - self.lam * jnp.log(p) - (self.lam - self.kap) * jnp.log(self.R)

        return ln_v

    def SL(self, p, ln_v0, p_0, return_ln=False):
        ln_v_sl = ln_v0 + self.kap * jnp.log(p_0)
        ln_v = ln_v_sl - self.kap * jnp.log(p)

        if return_ln:
            return ln_v
        else:
            return jnp.exp(ln_v)

    def get_dt_crit(self, material_points, cell_size, dt_alpha=0.5):
        """Get critical timestep of material poiints for stability."""

        def vmap_dt_crit(p, rho, vel):
            K = get_K(self.kap, p, self.K_min, self.K_max)
            G = get_G(self.nu, K)

            cdil = jnp.sqrt((K + (4 / 3) * G) / rho)

            c = jnp.abs(vel) + cdil * jnp.ones_like(vel)
            return c

        c_stack = jax.vmap(vmap_dt_crit)(
            material_points.p_stack,
            material_points.rho_stack,
            material_points.velocity_stack,
        )
        return (dt_alpha * cell_size) / jnp.max(c_stack)
