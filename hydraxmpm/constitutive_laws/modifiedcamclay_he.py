# # Copyright (c) 2024, Retiefasuarus
# # SPDX-License-Identifier: BSD-3-Clause
# #
# # Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# # -*- coding: utf-8 -*-

# from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp

# import optimistix as optx
from typing import Optional, Self, Tuple, Any

# Math utility functions
from ..utils.math_helpers import (
    get_dev_strain,
    get_dev_strain_stack,
    get_dev_stress,
    get_pressure,
    get_q_vm,
    get_volumetric_strain,
    get_pressure_stack,
    safe_inv_scalar_clamped,
    inv_2x2_robust,
    reconstruct_stress_from_triaxial,
    get_spin_tensor,
    get_sym_tensor,
    get_jaumann_increment,
)
from ..material_points.material_points import MaterialPointState

import warnings

from .constitutive_law import (
    ConstitutiveLawState,
    ConstitutiveLaw,
    Convergence,
    dummy_convergence,
)

from jaxtyping import Float, Array

def yield_function(p, p_c, q, M):
    return ((q * q) / (M * M)) + p * p - p_c * p


def get_pressure_mcc(deps_e_v, kap, p_prev):
    """Compute non-linear pressure incrementally."""
    return p_prev * jnp.exp(deps_e_v / kap)


def get_nc_pressure_mcc(deps_p_v, cp, p_c_prev):
    """Compute non-linear normal consolidation pressure incrementally.

    cp = lambda - kappa
    """
    return p_c_prev * jnp.exp(deps_p_v / cp)


def get_s(deps_e_d, G, s_prev):
    """Compute deviatoric stress incrementally."""
    return 2.0 * G * deps_e_d + s_prev


def get_K(kap, p, K_min=None, K_max=None):
    """Get pressure dependent bulk modulus"""
    p = jnp.clip(p, 1.0, None)
    K = (1.0 / kap) * (p)
    K = jnp.clip(K, K_min, K_max)
    return K


def get_G(nu, K):
    """Get shear modulus"""
    G = (3 * (1 - 2 * nu) / (2 * (1 + nu))) * K
    return G


def get_v_ncl(p_c, p_ref, N, lam):
    """Critical state specific volume"""
    return N * (p_ref / (p_c)) ** lam


def get_v_csl(p, p_ref, N, lam, kap):
    """
    Compute specific volume on the Critical State Line (CSL) at pressure p.

    """
    gamma = N * jnp.power(2.0, kap - lam)
    return gamma * (p_ref / p) ** lam


def get_v_f_kappa(p_c, p, kap):
    """Critical state specific volume elastic state"""
    return (p_c / p) ** kap


def get_v_sl(p_c, p, p_ref, N, lam, kap):
    """Swelling line specific volume"""
    v_ncl = get_v_ncl(p_c, p_ref, N, lam)
    v_elas = get_v_f_kappa(p_c, p, kap)
    return v_ncl * v_elas


def get_qp_ys(p_c, p, M):
    return M * jnp.sqrt((p_c / p) - 1)


def get_pressure_from_eps_v_e(eps_v_e, p_0, kap):
    return p_0 * (jnp.expm1(eps_v_e / kap) + 1.0)


def get_p_c_from_eps_v_p(eps_v_p, p_c0, cp):
    return p_c0 * (jnp.expm1(eps_v_p / cp) + 1.0)


def get_eps_v_p_hr(p_c, p_c0, cp):
    return cp * jnp.log(p_c / p_c0)


def get_eps_v_e(p, p_0, kap):
    return kap * jnp.log(p / p_0)


def invert_stress(tau, p_ref, kap, nu):

    p = jnp.trace(tau) / 3.0
    s_tau = tau - p * jnp.eye(3)

    # non linear bulk and shear moduli
    K = get_K(kap=kap, p=p)
    G = get_G(nu=nu, K=K)

    # compute strains

    eps_v_e = get_eps_v_e(p, p_ref, kap)

    eps_q_e_tensor = s_tau / (2.0 * G)

    # Total strain tensor
    eps_total = eps_q_e_tensor + (1.0 / 3.0) * eps_v_e * jnp.eye(3)

    return jax.scipy.linalg.expm(-2.0 * eps_total), p


class ModifiedCamClayHEState(ConstitutiveLawState):
    be_stack: Float[Array, "num_points 3 3"]
    eps_v_p_stack: Float[Array, "num_points"]
    convergence: Optional[Convergence] = None

    def get_p_c(self, lam, kap, p_ref):
        return p_ref * jnp.exp(self.eps_v_p_stack / (lam - kap))


class ModifiedCamClayHE(ConstitutiveLaw):
    # Return mapping
    NUM_NEWTON_ITERS = 10
    NUM_RESIDUALS = 2
    NUM_UNKNOWNS = 2

    nu: float | Float[Array, ""]
    M: float | Float[Array, ""]
    lam: float | Float[Array, ""]
    kap: float | Float[Array, ""]
    N: Optional[float | Float[Array, ""]] = None  # Specific volume at p=1 kPa on NCL
    p_ref: float | Float[Array, ""] = 1_000.0  # 1 kPa

    # Stability
    K_min: Optional[float | Float[Array, ""]] = None
    K_max: Optional[float | Float[Array, ""]] = None

    p_min_calc: Optional[float | Float[Array, ""]] = 10.0

    rho_p: Optional[float | Float[Array, ""]] = 2650.0

    # Derived
    _cp: float | Float[Array, ""]  # (Lambda - Kappa)

    """
    This formulation uses bi-logarithmic space for the volumetric behaviour,

    - Purpose: Prevents numerical instability near the free surface
    where pressure approaches zero. It ensures particles in a "vacuum" state have non-zero stiffness and can re-pressurize if compressed.

    Hypo elastic but avoids issues with large elastic strains
    does not use objective stress rates 

    (e.g., Hencky strains, incrementally)    
    """

    def __init__(
        self,
        *,
        nu: float | Float[Array, ""],
        M: float | Float[Array, ""],
        lam: float | Float[Array, ""],
        kap: float | Float[Array, ""],
        N: float | Float[Array, ""],
        p_ref: float | Float[Array, ""] = 1_000,  # 1 kPa
        K_min: float | Float[Array, ""] = None,
        K_max: float | Float[Array, ""] = None,
        p_t: Optional[float | Float[Array, ""]] = 0.0,
        p_min_calc: float | Float[Array, ""] = 10.0,
        rho_p: Optional[float | Float[Array, ""]] = 2650.0,
        debug_convergence: bool = False,
    ):
        # Model parameters, material properties
        self.nu = nu
        self.M = M
        self.lam = lam
        self.kap = kap
        self.p_ref = p_ref
        self.rho_p = rho_p
        self.N = N

        # stability
        self.K_min = K_min
        self.K_max = K_max
        self.p_min_calc = p_min_calc

        # derived
        self._cp = lam - kap

        self.debug_convergence = debug_convergence

    @classmethod
    def give_N_from_density0_stack(cls, density0_stack, rho_p):
        """
        Assume all density0_stack corresponds to the same N,
        and compute N from density0_stack and rho_p.
        """
        # return float for user API
        return rho_p / density0_stack

    def give_density0_stack(self, v_ref_stack):
        return jnp.atleast_1d(self.rho_p / self.N)

    def give_density_stack(self, p_stack, p_c_stack=None):

        if p_c_stack is None:
            p_c_stack = p_stack

        v = get_v_sl(p_c_stack, p_stack, self.p_ref, self.N, self.lam, self.kap)
        return self.rho_p / v

    def give_stress_stack(self, p_stack, p_c_stack=None, q_stack=None):
        """ "
        Calculate current stress stack from p, p_c, q. If q is not provided,
        check elastic state (q=0), otherwise pull to yield surface.
        """
        if p_c_stack is None:
            p_c_stack = p_stack

        if q_stack is None:
            # Elastic state, q=0
            q_stack = jnp.zeros_like(p_stack)
        else:
            q_p_stack = get_qp_ys(p_c_stack, p_stack, self.M)

            q_calc_stack = q_p_stack * p_stack

            # check YS
            q_stack = jnp.where(
                jnp.abs(q_stack - q_calc_stack) > 0, q_calc_stack, q_stack
            )

        stress_stack = reconstruct_stress_from_triaxial(
            p_stack=p_stack, q_stack=q_stack
        )
        return stress_stack

    def create_state(
        self,
        stress_stack: Float[Array, "num_points 3 3"],
        p_c_stack: Float[Array, "num_points"] = None,
    ) -> ModifiedCamClayHEState:

        be_initial, p_stack = jax.vmap(invert_stress, in_axes=(0, None, None, None))(
            stress_stack, self.p_ref, self.kap, self.nu
        )

        # normally consolidated state
        if p_c_stack is None:
            p_c_stack = p_stack

        eps_v_p_stack = get_eps_v_p_hr(p_c_stack, self.p_ref, self._cp)

        convergence = None
        if self.debug_convergence:

            jacobians = jnp.zeros(
                (
                    p_stack.shape[0],
                    self.NUM_NEWTON_ITERS,
                    self.NUM_UNKNOWNS,
                    self.NUM_UNKNOWNS,
                )
            )
            residuals = jnp.zeros(
                (p_stack.shape[0], self.NUM_NEWTON_ITERS, self.NUM_RESIDUALS)
            )
            unknowns = jnp.zeros(
                (p_stack.shape[0], self.NUM_NEWTON_ITERS, self.NUM_UNKNOWNS)
            )

            convergence = Convergence(
                residuals=residuals,
                unknowns=unknowns,
                jacobians=jacobians,
            )

        return ModifiedCamClayHEState(
            be_stack=be_initial, eps_v_p_stack=eps_v_p_stack, convergence=convergence
        )

    def update(self, mp_state, law_state, dt):

        stress_next_stack, be_next_stack, eps_v_p_next_stack, convergence = jax.vmap(
            self._update_stress, in_axes=(0, 0, 0, 0, None)
        )(
            mp_state.F_inc_stack,
            law_state.be_stack,
            law_state.eps_v_p_stack,
            mp_state.density_stack,
            dt,
        )

        new_mp = eqx.tree_at(lambda m: m.stress_stack, mp_state, stress_next_stack)
        new_law = eqx.tree_at(
            lambda l: (l.be_stack, l.eps_v_p_stack, l.convergence),
            law_state,
            (be_next_stack, eps_v_p_next_stack, convergence),
        )

        return new_mp, new_law

    def _update_stress(self, F_inc, be_prev, eps_v_p_prev, rho, dt):

        # Elastic predictor
        be_tr = F_inc @ be_prev @ F_inc.T

        eigvals, V = jnp.linalg.eigh(be_tr)
        eigvals = jnp.maximum(eigvals, 1e-12)

        # Hencky Strain (Compression Positive)
        eps_e_tr = -0.5 * jnp.log(eigvals)
        eps_e_v_tr = jnp.sum(eps_e_tr)
        eps_e_q_tr = eps_e_tr - (eps_e_v_tr / 3.0)

        eps_e_v_tr = jnp.maximum(eps_e_v_tr, 0.0)

        p_tr = get_pressure_from_eps_v_e(eps_e_v_tr, self.p_ref, self.kap)

        K_tr = get_K(self.kap, p_tr, self.K_min, self.K_max)
        G_tr = get_G(self.nu, K_tr)

        s_tr = 2.0 * G_tr * eps_e_q_tr
        sqrt_J2_tr = jnp.sqrt(jnp.maximum(0.5 * jnp.sum(s_tr**2), 1e-12))
        q_tr = jnp.sqrt(3.0) * sqrt_J2_tr

        p_c_tr = get_p_c_from_eps_v_p(eps_v_p_prev, self.p_ref, self._cp)

        yf = yield_function(p_tr, p_c_tr, q_tr, self.M)
        is_ep = yf > 0.0

        def elastic_update():
            s_next = s_tr * (q_tr / jnp.maximum(q_tr, 1e-12))
            stress_principal = s_next + p_tr
            tau_next = (V * stress_principal) @ V.T
            be_next = (V * jnp.exp(-2.0 * eps_e_tr)) @ V.T

            convergence = None
            if self.debug_convergence:
                convergence = dummy_convergence(
                    self.debug_convergence,
                    self.NUM_NEWTON_ITERS,
                    self.NUM_RESIDUALS,
                    self.NUM_UNKNOWNS,
                )
            return tau_next, be_next, eps_v_p_prev, convergence

        def pull_to_ys():
            # https://github.com/patrick-kidger/optimistix/issues/132
            # Here we have safe values to avoid nan during compile time
            safe_mask = lambda x, default: jnp.where(is_ep, x, default)
            q_tr_safe = safe_mask(q_tr, 1e-6)
            u_p_c_prev = jnp.log(safe_mask(p_c_tr, self.p_min_calc))

            def residuals(sol, args):
                pmulti, u_p_c = sol
                p_c_next = jnp.exp(u_p_c)

                eps_p_v_hr = get_eps_v_p_hr(p_c_next, self.p_ref, self._cp)

                deps_p_v_hr = eps_p_v_hr - eps_v_p_prev

                p_next = get_pressure_from_eps_v_e(
                    eps_e_v_tr - deps_p_v_hr, self.p_ref, self.kap
                )

                K_next = get_K(self.kap, p_next, self.K_min, self.K_max)

                G_next = get_G(self.nu, K_next)

                factor = 1 / (1 + 6.0 * G_next * pmulti)

                s_next = s_tr * factor

                q_next = q_tr_safe * factor

                deps_v_p_fr = pmulti * (2.0 * p_next - p_c_next) * self.M**2

                yf_next = yield_function(p_next, p_c_next, q_next, self.M)

                # maybe volumetric strains need rescaling for the residuals?
                # ... from tests, tests show that the residuals are very small,
                #  so maybe not necessary
                deps_scale = 1.0

                yf_scale = K_tr
                R = jnp.array(
                    [yf_next / yf_scale, (deps_v_p_fr - deps_p_v_hr) / deps_scale]
                )

                aux = (p_next, s_next, G_next, K_next, eps_p_v_hr)

                return R, aux

            # Newton-Raphson solver for the 2x2 system
            # replacing optimistix root finder...
            def step_fn(carry, _):
                x = carry

                # Here we compute the Jacobian via AD
                # TODO can be optimized further via analytical Jacobian...
                R_func = lambda v: residuals(v, None)[0]

                # Using forward-mode AD for Jacobian (fast for 2x2)
                J = jax.jacfwd(R_func)(x)
                R = R_func(x)

                # inversion in helper file, safe for AD
                inv_J = inv_2x2_robust(J)

                dx = -(inv_J @ R)

                # Damping (0.8 is safer... could be a parameter?)
                x_new = x + 0.8 * dx

                # --- NAN GUARD ---
                # If Jacobian was singular or update exploded, dx might be NaN.
                # In that case, ignore the update (x_new = x).
                is_bad = jnp.any(jnp.isnan(x_new)) | jnp.any(jnp.isinf(x_new))
                x_new = jnp.where(is_bad, x, x_new)

                # Prevent negative plastic multiplier i
                x_new = x_new.at[0].set(jnp.maximum(x_new[0], 0.0))
                x_new = x_new.at[1].set(jnp.maximum(x_new[1], 0.0))
                convergence = None
                if self.debug_convergence:
                    convergence = Convergence(
                        residuals=R,
                        unknowns=x,
                        jacobians=J,
                    )
                return x_new, convergence

            x_init = jnp.array([0.0, u_p_c_prev])

            # JAX unrolls the loops here
            x_final, convergence = jax.lax.scan(
                step_fn, x_init, None, length=self.NUM_NEWTON_ITERS
            )

            pmulti_curr, u_p_c_final = x_final[0], x_final[1]

            _, aux = residuals(x_final, None)

            p_next, s_next, G_next, K_next, eps_p_v_hr = aux

            eps_e_v_next = get_eps_v_e(p_next, self.p_ref, self.kap)

            K_next = get_K(self.kap, p_next, self.K_min, self.K_max)
            G_next = get_G(self.nu, K_next)
            eps_e_q_next = s_next / (2.0 * G_next)

            eps_e_next = eps_e_q_next + eps_e_v_next / 3.0

            stress_principal = s_next + p_next

            tau_next = (V * stress_principal) @ V.T

            be_next = (V * jnp.exp(-2.0 * eps_e_next)) @ V.T

            return tau_next, be_next, eps_p_v_hr, convergence

        # We treat it as disconnected at low pressures
        stress_next, be_next, eps_v_p_next, convergence = jax.lax.cond(
            p_tr > 0.0,
            lambda: jax.lax.cond(is_ep, pull_to_ys, elastic_update),
            lambda: (
                jnp.zeros((3, 3)),
                jnp.eye(3),
                eps_v_p_prev,
                dummy_convergence(
                    self.debug_convergence,
                    self.NUM_NEWTON_ITERS,
                    self.NUM_RESIDUALS,
                    self.NUM_UNKNOWNS,
                ),
            ),
        )
        return stress_next, be_next, eps_v_p_next, convergence

    def get_dt_crit(
            self,
            mp_state: MaterialPointState,
            cell_size: float,
            alpha: float = 0.5
        ) -> Float[Array, ""]:
            density = mp_state.mass_stack / mp_state.volume_stack
            p = jnp.maximum(mp_state.pressure_stack, self.p_min_calc)
            K = get_K(self.kap, p, self.K_min, self.K_max)
            G = get_G(self.nu, K)

            c_p = jnp.sqrt((K + (4.0 / 3.0) * G) / density)
            vel_mag = jnp.linalg.norm(mp_state.velocity_stack, axis=1)
            max_speed = jnp.max(c_p + vel_mag)

            return (alpha * cell_size) / (max_speed + 1e-9)

# This code makes available helper functions on the class for user-facing API
_helpers = (
    "yield_function",
    "get_pressure_mcc",
    "get_nc_pressure_mcc",
    "get_s",
    "get_K",
    "get_G",
    "get_v_ncl",
    "get_v_f_kappa",
    "get_v_sl",
    "get_qp_ys",
    "get_v_csl",
)

for _name in _helpers:
    _fn = globals().get(_name)
    if _fn is not None:
        setattr(ModifiedCamClayHE, _name, staticmethod(_fn))


# def create_state_from_stress(
#     self,
#     p_stack: Float[Array, "num_points"],
#     q_stack: Float[Array, "num_points"]
# ) -> Tuple[ModifiedCamClayHEState, Float[Array, "num_points"]]:
#     """
#     Case 2: Critical/Limit State Initialization.
#     We assume the current stress state (p, q) lies EXACTLY on the Yield Surface.
#     We derive p_s and specific volume.
#     """
#     # 1. Solve for p_s from Yield Function
#     # f = (q/M)^2 + (p - ps)^2 - ps^2 = 0gamma
#     # This implies: p_s = p * [ (q/Mp)^2 + 1 ]

#     q_p = q_stack / p_stack
#     term = (q_p / self.M)**2 + 1.0
#     p_c_stack = p_stack * term

#     # 2. Derive Specific Volume
#     # Since it's NC (on yield surface), we technically use the NCL at p_s
#     # or the swelling line from p_s back to p. They meet at the yield surface.
#     specific_volume_stack = get_v_sl(
#         p_c_stack, p_stack, self.p_ref, self.N, self.lam, self.kap
#     )

#     density_stack = self.rho_p / specific_volume_stack

#     eps_e = jnp.zeros((p_stack.shape[0], 3, 3))
#     return ModifiedCamClayHEState(p_c_stack=p_c_stack, eps_e_stack=eps_e),density_stack

# def create_state_from_density(
#         self,
#             p_stack: Float[Array, "num_points"],
#             density_stack: Float[Array, "num_points"]
#     ) -> Tuple[ModifiedCamClayHEState, Float[Array, "num_points"]]:
#         """

#         We know pressure and density (v). We derive the internal hardening state (p_s).

#         WARNING: This can result in 'Impossible' states if (p, v) is outside the NCL.
#         """
#         specific_volume_stack = self.rho_p/density_stack

#         # 1. Invert Swelling Line Equation to find p_s
#         # v = v_csl * v_elas
#         # v = [Gamma * (pref/ps)^lam] * [(ps/p)^kap]
#         # v = Gamma * pref^lam * p^-kap * ps^(kap-lam)
#         # ps^(lam-kap) = (Gamma * pref^lam * p^-kap) / v
#         # Let A = lam - kap
#         # ps = [ (Gamma * pref^lam) / (v * p^kap) ] ^ (1/A)

#         numerator = self.N * (self.p_ref**self.lam)
#         denominator = specific_volume_stack * (p_stack**self.kap)
#         exponent = 1.0 / (self.lam - self.kap)

#         p_c_stack = (numerator / denominator) ** exponent

#         # 2. Validation / Clamping
#         # In MCC, we usually assume the state is elastic or on yield surface.
#         # If the derived p_s is such that the current p is way outside the yield surface (p > 2*p_s),
#         # it means the provided density is "too loose" for this pressure under this model.
#         # You might want to clamp p_s or warn.

#         # Check yield condition for q=0 case: p must be <= 2*p_s
#         # p_s_stack = jnp.maximum(p_s_stack, p_stack / 2.0)

#         eps_e = jnp.zeros((p_stack.shape[0], 3, 3))

#         # Return same specific volume back confirms it was used
#         return ModifiedCamClayHEState(p_c_stack=p_c_stack, eps_e_stack=eps_e), specific_volume_stack

# def create_state(
#     self,
#     mp_state: MaterialPointState =None,
# ) -> ModifiedCamClayHEState:

#     mcc_state,*_ = self.create_state_from_ocr(
#         p_stack=mp_state.pressure_stack,
#         ocr_stack =jnp.full((mp_state.position_stack.shape[0], 1.0))
#     )

#     return mcc_state
