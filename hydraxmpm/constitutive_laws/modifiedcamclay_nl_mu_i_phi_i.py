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
    get_inertial_number,
    get_scalar_shear_strain,
)
from .constitutive_law import ConstitutiveLaw, ConvergenceControlConfig


# def yield_function(p, px, q, M, chi=0.0):
# return ((q * q) / (M * M)) + p * p - px * p
def yield_function(p, px, q, M, chi=0.0):
    return (q * q) + (p * p - px * p) * (M * M)


def get_M_I(I, M, M_d, I0):
    return M + I * ((M_d - M) / (I0 + I))


def get_p_hat(deps_e_v, kap, p_hat_prev):
    """Compute non-linear pressure."""
    p_hat = p_hat_prev / (1.0 - (1.0 / kap) * deps_e_v + 1e-12)
    return jnp.clip(p_hat, 1.0, None)


def get_px_hat_mcc(px_hat_prev, cp, deps_p_v):
    """Compute non-linear pressure."""
    px_hat = px_hat_prev / (1.0 - (1.0 / cp) * deps_p_v + 1e-12)
    return jnp.clip(px_hat, 1.0, None)


def get_px_hat_mcc_mu_i(px_hat_prev, cp, deps_p_v, dI, I_phi):
    """Compute non-linear pressure."""
    deps_p_v_I = deps_p_v + dI / I_phi
    px_hat = px_hat_prev / (1.0 - (1.0 / cp) * deps_p_v_I + 1e-12)
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


def get_reloadingline(p, px, ps, kap, lam, phi_0):
    p_hat = p + ps
    pc_hat = px + ps
    phi = phi_0 * ((pc_hat / ps) ** lam) * ((p_hat / pc_hat) ** kap)

    return phi


class ModifiedCamClayNLMuI(ConstitutiveLaw):
    nu: TypeFloat
    M: TypeFloat
    OCR: TypeFloat
    lam: TypeFloat
    kap: TypeFloat

    I_0: TypeFloat
    I_phi: TypeFloat
    M_d: TypeFloat

    p_t: TypeFloat = 0.0
    ln_N: Optional[TypeFloat] = None

    K_min: Optional[TypeFloat] = None
    K_max: Optional[TypeFloat] = None

    ln_Z: Optional[TypeFloat] = None
    ps: Optional[TypeFloat] = None
    ln_v0: Optional[TypeFloat] = None
    chi: TypeFloat = 0.0

    px_hat_stack: Optional[TypeFloatScalarPStack] = None
    I_stack: Optional[TypeFloatScalarPStack] = None
    stress_0_stack: Optional[TypeFloatMatrixPStack] = None

    settings: ConvergenceControlConfig

    def __init__(
        self: Self,
        nu: TypeFloat,
        M: TypeFloat,
        OCR: TypeFloat,
        lam: TypeFloat,
        kap: TypeFloat,
        I_0: TypeFloat,
        I_phi: TypeFloat,
        M_d: TypeFloat,
        K_min: TypeFloat = None,
        K_max: TypeFloat = None,
        ln_N: Optional[TypeFloat] = None,
        ln_Z: Optional[TypeFloat] = None,
        ln_v0: Optional[TypeFloat] = None,
        ps: Optional[TypeFloat] = None,
        chi: TypeFloat = 0.0,
        settings: Optional[dict | ConvergenceControlConfig] = None,
        **kwargs,
    ) -> Self:
        self.nu = nu

        self.M = M

        self.OCR = OCR

        self.lam = lam

        self.kap = kap

        self.chi = chi

        self.ln_N = ln_N

        self.K_min = K_min

        self.K_max = K_max

        self.I_0 = I_0
        self.I_phi = I_phi
        self.M_d = M_d

        rho_0 = kwargs.get("rho_0", None)
        rho_p = kwargs.get("rho_p", None)

        if ps == 0.0:
            self.ps = 0
            self.ln_Z = ln_N
            self.ln_v0 = ln_N
        elif ln_v0 is not None:
            self.ln_v0 = ln_v0
            self.ps = jnp.exp((ln_N - self.ln_v0) / lam)
            self.ln_Z = self.ln_v0 - lam * jnp.log((1.0 + self.ps) / self.ps)
        elif ln_Z is not None:
            self.ln_Z = ln_Z
            self.ps = jnp.exp((ln_N - ln_Z) / lam) - 1.0
            self.ln_v0 = ln_Z + lam * jnp.log((1 + self.ps) / self.ps)
            self.rho_0 = rho_p / jnp.exp(self.ln_v0)
            kwargs["rho_0"] = self.rho_0
        elif rho_0 is not None:
            self.ln_v0 = jnp.log(1.0 / (rho_0 / rho_p))
            self.ps = jnp.exp((ln_N - self.ln_v0) / lam)
            self.ln_Z = self.ln_v0 - lam * jnp.log((1.0 + self.ps) / self.ps)

        self.eps_e_stack = kwargs.get("eps_e_stack")

        self.px_hat_stack = kwargs.get("px_hat_stack")

        self.stress_0_stack = kwargs.get("stress_0_stack")

        self.I_stack = kwargs.get("I_stack")

        # settings used for convergence control
        if settings is None:
            settings = dict()
        if isinstance(settings, dict):
            self.settings = ConvergenceControlConfig(
                rtol=settings.get("rtol", 1e-6),
                atol=settings.get("atol", 1e-6),
                max_iter=settings.get("max_iter", 20),
                throw=settings.get("throw", False),
                # plastic multiplier and volumetric strain, respectively
                lower_bound=settings.get("lower_bound", (0.0, -100.0)),
            )
        else:
            self.settings = settings
        del settings

        super().__init__(**kwargs)

    def init_state(self: Self, material_points: MaterialPoints):
        stress_0_stack = material_points.stress_stack
        p_0_stack = material_points.p_stack
        px_hat_stack = p_0_stack * self.OCR

        rho_0 = self.rho_0

        phi = get_reloadingline(
            p_0_stack, px_hat_stack, self.ps, self.kap, self.lam, self.phi_0
        )

        rho = self.rho_p * phi

        # jax.debug.breakpoint()
        eps_e_stack = jnp.zeros((material_points.num_points, 3, 3))
        I_stack = jnp.zeros(material_points.num_points)
        return self.post_init_state(
            material_points,
            rho_0=rho_0,
            rho=rho,
            stress_0_stack=stress_0_stack,
            px_hat_stack=px_hat_stack,
            eps_e_stack=eps_e_stack,
            I_stack=I_stack,
        )

    def update(
        self: Self,
        material_points: MaterialPoints,
        dt: TypeFloat,
        dim: Optional[TypeInt] = 3,
    ) -> Tuple[MaterialPoints, Self]:
        """Update the material state and particle stresses for MPM solver."""

        deps_dt_stack = material_points.deps_dt_stack

        new_stress_stack, new_eps_e_stack, new_px_hat_stack, new_I_stack = (
            self.vmap_constitutive_update(
                dt,
                deps_dt_stack,
                self.eps_e_stack,
                material_points.stress_stack,
                self.px_hat_stack,
                self.stress_0_stack,
                material_points.specific_volume_stack(self.rho_p),
                material_points.isactive_stack,
                self.I_stack,
            )
        )
        new_self = eqx.tree_at(
            lambda state: (state.eps_e_stack, state.px_hat_stack, state.I_stack),
            self,
            (new_eps_e_stack, new_px_hat_stack, new_I_stack),
        )

        new_material_points = eqx.tree_at(
            lambda state: (state.stress_stack),
            material_points,
            (new_stress_stack),
        )
        # new_self = new_self.post_update(new_stress_stack, deps_dt_stack, dt)
        return new_material_points, new_self

    @partial(
        jax.vmap, in_axes=(None, None, 0, 0, 0, 0, 0, 0, 0, 0), out_axes=(0, 0, 0, 0)
    )
    def vmap_constitutive_update(
        self: Self,
        dt,
        deps_dt_next,
        eps_e_prev,
        stress_prev,
        px_hat_prev,
        stress_0,
        specific_volume,
        isactive,
        I_prev,
    ):
        default_values = (stress_prev, eps_e_prev, px_hat_prev, I_prev)

        def update(_):
            return self.update_ip(
                dt,
                deps_dt_next,
                eps_e_prev,
                stress_prev,
                px_hat_prev,
                stress_0,
                specific_volume,
                I_prev,
            )

        return jax.lax.cond(
            isactive,
            update,
            lambda _: default_values,
            operand=None,  # No additional operand needed
        )

    def update_ip(
        self: Self,
        dt,
        deps_dt_next,
        eps_e_prev,
        stress_prev,
        px_hat_prev,
        stress_0,
        specific_volume,
        I_prev,
    ):
        deps_next = deps_dt_next * dt
        # reference stresses
        p_0 = get_pressure(stress_0)
        s_0 = get_dev_stress(stress_0, pressure=p_0)

        # previous stresses
        p_prev = get_pressure(stress_prev)
        s_prev = get_dev_stress(stress_prev, pressure=p_prev)

        # previous pressure in transformed space
        p_hat_prev = p_prev + self.ps
        px_hat_prev = px_hat_prev + self.ps

        # trail elastic volumetric strain
        deps_e_v_tr = get_volumetric_strain(deps_next)

        # trail pressure in transformed space
        p_hat_tr = get_p_hat(deps_e_v_tr, self.kap, p_hat_prev)

        # trail bulk and shear modulus in transformed space
        K_tr = get_K(self.kap, p_hat_tr, self.K_min, self.K_max)
        G_tr = get_G(self.nu, K_tr)

        # trail elastic deviatoric strain tensor
        deps_e_d_tr = get_dev_strain(deps_next, deps_e_v_tr)

        # trail elastic deviatoric stress tensor
        s_tr = get_s(deps_e_d_tr, G_tr, s_prev)

        # trail von Mises stress
        q_tr = get_q_vm(dev_stress=s_tr)

        ######
        # dgamma_dt = get_scalar_shear_strain(deps_dt_next)

        # I = get_inertial_number(p_hat_tr - self.ps, dgamma_dt, self.d, self.rho_p)

        # M_I = get_M_I(I, self.M, self.M_d, self.I_0)

        yf = yield_function(
            p_hat_tr - self.ps,
            px_hat_prev - self.ps,
            # px_hat_prev,
            q_tr,
            self.M,
            self.chi,
        )

        is_ep = yf > 0.0

        def elastic_update():
            stress_next = s_tr - (p_hat_tr - self.ps) * jnp.eye(3)
            eps_e_tr = eps_e_prev + deps_next
            return stress_next, eps_e_tr, px_hat_prev - self.ps, 0.0

        def pull_to_ys():
            # https://github.com/patrick-kidger/optimistix/issues/132
            deps_e_v_tr_ = jnp.where(~is_ep, 0.0, deps_e_v_tr)
            p_hat_prev_ = jnp.where(~is_ep, p_0 + self.ps, p_hat_prev)
            px_hat_prev_ = jnp.where(~is_ep, p_0 + self.ps, px_hat_prev)
            q_tr_ = jnp.where(~is_ep, 0.0, q_tr)

            deps_e_v_tr_scale = jnp.where(~is_ep, 1.0, deps_e_v_tr)

            def residuals(sol, args):
                pmulti, deps_p_v = sol

                # next pressure in transformed space
                p_hat_next = get_p_hat(deps_e_v_tr_ - deps_p_v, self.kap, p_hat_prev_)

                # next bulk and shear modulus in transformed space
                K_next = get_K(self.kap, p_hat_next, self.K_min, self.K_max)

                G_next = get_G(self.nu, K_next)

                # next deviatoric strain tensor end Von Mises stress tensor
                factor = 1 / (1 + 6.0 * G_next * pmulti)

                s_next = s_tr * factor
                q_next = q_tr_ * factor

                # next consolidation pressure in transformed space
                # px_hat_next = get_px_hat_mcc(px_hat_prev_, self._cp, deps_p_v)

                # dgamma_dt = get_scalar_shear_strain(deps_dt_next)
                dgamma_p_dt = (2 * pmulti * q_next) / dt
                I_next = get_inertial_number(
                    jnp.clip(p_hat_next - self.ps, 1, None),
                    dgamma_p_dt,
                    self.d,
                    self.rho_p,
                )
                # I_next = jnp.nanmax(I_next, 10)
                # I_next = jnp.clip(I_next, 1e-22, 10)
                dI = I_next - I_prev
                px_hat_next = get_px_hat_mcc_mu_i(
                    px_hat_prev, self._cp, deps_p_v, dI, self.I_phi
                )
                # px_hat_next = get_px_hat_mcc(px_hat_prev_, self._cp, deps_p_v)

                M_I = get_M_I(I_next, self.M, self.M_d, self.I_0)

                deps_v_p_fr = (
                    pmulti
                    * (2 * (p_hat_next - self.ps) - (px_hat_next - self.ps))
                    * M_I**2
                    # * self.M**2
                )

                yf_next = yield_function(
                    p_hat_next - self.ps,
                    px_hat_next - self.ps,
                    q_next,
                    M_I,
                    self.chi,
                )

                deps_scale = 1.0

                # yf_scale = K_tr
                yf_scale = 1.0
                yf_scale = p_hat_prev_ * p_hat_prev_ * M_I
                # R = jnp.array([yf_next / yf_scale, px_hat_next_fr - px_hat_next])
                R = jnp.array(
                    [yf_next / yf_scale, (deps_v_p_fr - deps_p_v) / deps_scale]
                )
                # jax.debug.print("R: {}", R)

                aux = (
                    I_next,
                    p_hat_next,
                    s_next,
                    px_hat_next,
                    G_next,
                    K_next,
                    deps_v_p_fr,
                )

                return R, aux

            def find_roots():
                """Find roots of the residuals function."""
                # Provide a better initial guess based on trial state
                # init_pmulti = yf / (
                #     2 * G_tr + K_tr * self.M**2
                # )  # Approximate initial plastic multiplier
                # init_deps_p_v = (
                #     init_pmulti
                #     * self.M**2
                #     * (2 * (p_hat_tr - self.ps) - (px_hat_prev - self.ps))
                # )  # Approximate volumetric plastic strain

                # init_val = jnp.array([init_pmulti, init_deps_p_v])
                init_val = jnp.array([0.0, 0.0])

                solver = optx.Newton(
                    rtol=self.settings.rtol, atol=self.settings.atol, norm=optx.rms_norm
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
                    ),
                )
                return sol.value

            pmulti_curr, deps_p_v_next = find_roots()

            R, aux = residuals([pmulti_curr, deps_p_v_next], None)

            I_next, p_hat_next, s_next, px_hat_next, G_next, K_next, deps_v_p_fr = aux

            p_next = p_hat_next - self.ps

            px_next = px_hat_next - self.ps

            # G_next =

            stress_next = s_next - (p_next) * jnp.eye(3)

            eps_e_v_next = (p_next - p_0) / K_next

            # eps_e_v_next = (p_hat_next - p_0) / K_next

            # jax.debug.print(
            #     " eps_e_v_next: {} deps_p_v_next {}  | deps_v_p_fr {}",
            #     deps_e_v_tr - deps_p_v_next,
            #     deps_p_v_next,
            #     deps_v_p_fr,
            # )
            # jax.debug.print(
            #     "deps_p_v_next: {}",
            # )

            eps_e_d_next = (s_next - s_0) / (2.0 * G_next)

            eps_e_next = eps_e_d_next - (1.0 / 3) * eps_e_v_next * jnp.eye(3)

            return stress_next, eps_e_next, px_next, I_next

        ln_v = jnp.log(specific_volume)

        stress_next, eps_e_next, px_next, I_next = jax.lax.cond(
            ln_v <= self.ln_v0,
            lambda: jax.lax.cond(is_ep, pull_to_ys, elastic_update),
            lambda: (0.0 * jnp.eye(3), eps_e_prev, px_hat_prev - self.ps, 0.0),
        )

        return stress_next, eps_e_next, px_next, I_next

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

        ln_v = ln_N - self.lam * jnp.log(p) - (self.lam - self.kap) * jnp.log(self.OCR)

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

    def compute_reloadingline(self, p=None, px=None, phi_0=None):
        """Wrapper that passes instance attributes to the external function."""

        if p is None:
            p = self.px_hat_stack
        if px is None:
            px = self.px_hat_stack
        if phi_0 is None:
            phi_0 = self.rho_0 / self.rho_p

        return get_reloadingline(p, px, self.ps, self.kap, self.lam, phi_0)
