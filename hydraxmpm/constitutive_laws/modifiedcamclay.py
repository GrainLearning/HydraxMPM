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
    get_jaumann_increment
)
from ..material_points.material_points import MaterialPointState

import warnings

from .constitutive_law import ConstitutiveLawState, ConstitutiveLaw

from jaxtyping import Float, Array





####################

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


class ModifiedCamClayState(ConstitutiveLawState):
    p_c_stack: Float[Array, "num_points"] = None
    stress_ref_stack: Float[Array, "num_points 3 3"] = None

    # # We store it, but not used for calculations
    # # TODO remove if not needed
    eps_e_stack: Float[Array, "num_points 3 3"] = None

    @property
    def p_s_stack(self):
        return self.p_c_stack / 2.0


class ModifiedCamClay(ConstitutiveLaw):
    nu: float | Float[Array, ""]
    M: float | Float[Array, ""]
    lam: float | Float[Array, ""]
    kap: float | Float[Array, ""]
    N: Optional[float |Float[Array, ""]] = None  # Specific volume at p=1 kPa on NCL
    p_ref: float |Float[Array, ""] = 1_000.0  # 1 kPa

    # Stability
    K_min: Optional[float |Float[Array, ""]] = None    
    K_max: Optional[float |Float[Array, ""]] = None


    p_min_calc: Optional[float | Float[Array, ""]]  = 10.0 
    
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
        M: float |Float[Array, ""],
        lam: float | Float[Array, ""],
        kap: float |Float[Array, ""],
        N: float | Float[Array, ""],
        p_ref: float | Float[Array, ""] = 1_000,  # 1 kPa
        K_min: float | Float[Array, ""] = None,
        K_max: float | Float[Array, ""] = None,
        p_t: Optional[float | Float[Array, ""]] = 0.0,
        p_min_calc:float | Float[Array, ""] = 10.0,
        rho_p: Optional[float |Float[Array, ""]] = 2650.0,
        requires_F_reset: bool = True,
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

        # inherited from base class for bvps
        self.requires_F_reset = requires_F_reset 

        # derived
        self._cp = lam - kap




    def create_state_from_ocr(
        self,
        p_stack: Float[Array, "num_points"],
        ocr_stack:  float | Float[Array, "num_points"] | Float[Array, ""],
        q_stack: Optional[Float[Array, "num_points"]] = None,
    ) -> Tuple[
        ModifiedCamClayState, Float[Array, "num_points 3 3"], Float[Array, "num_points"]
    ]:
        """
        Initialization given the stress history (OCR),
        and derive the required specific volume (density).

        If q_stack is not provided we assume isotropic (q=)
        otherwise, we check if its within the yield surface.
        If outside, we project it back to the yield surface.
        """


        if isinstance(ocr_stack, float | Float[Array, ""]):
            ocr_stack = jnp.full((p_stack.shape[0],), ocr_stack)

        # Normal consolidation pressure
        p_c_stack = p_stack * ocr_stack

        # Derive specific volume from swelling line
        specific_volume_stack = get_v_sl(
            p_c_stack, p_stack, self.p_ref, self.N, self.lam, self.kap
        )

        density_stack = self.rho_p / specific_volume_stack


        q_p_stack = get_qp_ys(p_c_stack, p_stack, self.M)

        eps_e_stack = jnp.zeros((p_stack.shape[0], 3, 3))

        if q_stack is None:
            q_stack = jnp.zeros_like(p_stack)
        else:
            q_p_inp_stack = q_stack / p_stack

            q_p_ys_stack = get_qp_ys(p_c_stack, p_stack, self.M)
            
            yf_check = jnp.abs(q_p_inp_stack - q_p_ys_stack)

            q_p_stack = jnp.where(yf_check > 0, q_p_ys_stack, q_p_inp_stack)

            q_stack = q_p_stack * p_stack

        stress_ref_stack = reconstruct_stress_from_triaxial(
            p_stack=p_stack, q_stack=q_stack
        )

        mcc_state = ModifiedCamClayState(
            stress_ref_stack=stress_ref_stack,
            p_c_stack=p_c_stack,
            eps_e_stack=eps_e_stack,
        )

        return (mcc_state, stress_ref_stack, density_stack)

    
    def create_state_from_stress(
        self,
        p_stack: Float[Array, "num_points"],
        q_stack: Float[Array, "num_points"]
    ) -> Tuple[ModifiedCamClayState, Float[Array, "num_points"]]:
        """
        Case 2: Critical/Limit State Initialization.
        We assume the current stress state (p, q) lies EXACTLY on the Yield Surface.
        We derive p_s and specific volume.
        """
        # 1. Solve for p_s from Yield Function
        # f = (q/M)^2 + (p - ps)^2 - ps^2 = 0gamma
        # This implies: p_s = p * [ (q/Mp)^2 + 1 ]
        
        q_p = q_stack / p_stack
        term = (q_p / self.M)**2 + 1.0
        p_c_stack = p_stack * term
        
        # 2. Derive Specific Volume
        # Since it's NC (on yield surface), we technically use the NCL at p_s
        # or the swelling line from p_s back to p. They meet at the yield surface.
        specific_volume_stack = get_v_sl(
            p_c_stack, p_stack, self.p_ref, self.N, self.lam, self.kap
        )

        density_stack = self.rho_p / specific_volume_stack

        eps_e = jnp.zeros((p_stack.shape[0], 3, 3))
        return ModifiedCamClayState(p_c_stack=p_c_stack, eps_e_stack=eps_e),density_stack


    def create_state_from_density(
            self,
                p_stack: Float[Array, "num_points"],
                density_stack: Float[Array, "num_points"]
        ) -> Tuple[ModifiedCamClayState, Float[Array, "num_points"]]:
            """

            We know pressure and density (v). We derive the internal hardening state (p_s).
            
            WARNING: This can result in 'Impossible' states if (p, v) is outside the NCL.
            """
            specific_volume_stack = self.rho_p/density_stack

            # 1. Invert Swelling Line Equation to find p_s
            # v = v_csl * v_elas
            # v = [Gamma * (pref/ps)^lam] * [(ps/p)^kap]
            # v = Gamma * pref^lam * p^-kap * ps^(kap-lam)
            # ps^(lam-kap) = (Gamma * pref^lam * p^-kap) / v
            # Let A = lam - kap
            # ps = [ (Gamma * pref^lam) / (v * p^kap) ] ^ (1/A)
            
            numerator = self.N * (self.p_ref**self.lam)
            denominator = specific_volume_stack * (p_stack**self.kap)
            exponent = 1.0 / (self.lam - self.kap)
            
            p_c_stack = (numerator / denominator) ** exponent
            
            # 2. Validation / Clamping
            # In MCC, we usually assume the state is elastic or on yield surface.
            # If the derived p_s is such that the current p is way outside the yield surface (p > 2*p_s),
            # it means the provided density is "too loose" for this pressure under this model.
            # You might want to clamp p_s or warn.
            
            # Check yield condition for q=0 case: p must be <= 2*p_s
            # p_s_stack = jnp.maximum(p_s_stack, p_stack / 2.0) 

            eps_e = jnp.zeros((p_stack.shape[0], 3, 3))
            
            # Return same specific volume back confirms it was used
            return ModifiedCamClayState(p_c_stack=p_c_stack, eps_e_stack=eps_e), specific_volume_stack

    def create_state(
        self,
        mp_state: MaterialPointState =None,
    ) -> ModifiedCamClayState:
        

        mcc_state,*_ = self.create_state_from_ocr(
            p_stack=mp_state.pressure_stack,
            ocr_stack =jnp.full((mp_state.position_stack.shape[0], 1.0))
        )

        return mcc_state

    def update(
        self,
        mp_state,
        law_state,
        dt
    ):
        specific_volume_stack = self.rho_p/mp_state.density_stack

        new_stress, new_eps_e, new_p_c = jax.vmap(self._update_stress,
            in_axes=(0,0,0,0,0,0,None))(
            mp_state.L_stack,
            law_state.eps_e_stack,
            mp_state.stress_stack,
            law_state.p_c_stack,
            law_state.stress_ref_stack,
            specific_volume_stack,
            dt
        )


        new_mp = eqx.tree_at(lambda m: m.stress_stack, mp_state, new_stress)
        new_law = eqx.tree_at(
            lambda l: (l.p_c_stack, l.eps_e_stack),
            law_state,
            (new_p_c, new_eps_e)
        )

        return new_mp, new_law


    def _update_stress(
        self,
        L: Float[Array, "3 3"],
        eps_e_prev,
        stress_prev,
        p_c_prev,
        stress_ref,
        specific_volume,
        dt
    ):

    
        D = get_sym_tensor(L) # symmetric part
        W = get_spin_tensor(L) # skew-symmetric part


        deps_next = D * dt

        ### Apply objective stress rate (Jaumann) ###
        
        stress_prev_rot = get_jaumann_increment(stress_prev, W, dt)
        eps_e_prev_rot  = get_jaumann_increment(eps_e_prev, W, dt)
        
        # Use these rotated values for the rest of the calculation
        stress_prev = stress_prev_rot
        eps_e_prev = eps_e_prev_rot 


        # reference stresses
        p_0 = get_pressure(stress_ref)
        s_0 = get_dev_stress(stress_ref, pressure=p_0)

        # previous stresses
        p_prev = get_pressure(stress_prev)

        # Handle particle that was in a vacuum or
        # free surface so particle recovers
        # ensure stable wave speed that bulk modulus
        # never drops bellow sqrt(K_min/rho)
        # preventing exploding small values 
        # in internal force calculations of MPM
        p_prev = jnp.maximum(p_prev, self.p_min_calc)
        deps_e_v_tr = get_volumetric_strain(deps_next)
        p_tr = get_pressure_mcc(deps_e_v_tr, self.kap, p_prev)
        K_tr = get_K(self.kap, jnp.maximum(p_tr, self.p_min_calc), self.K_min, self.K_max)
        
        # deviatoric shear trial stress invariant and tensor
        s_prev = get_dev_stress(stress_prev, pressure=p_prev)
        deps_e_d_tr = get_dev_strain(deps_next, deps_e_v_tr)
        G_tr = get_G(self.nu, K_tr)

        s_tr = get_s(deps_e_d_tr, G_tr, s_prev)

        q_tr = get_q_vm(dev_stress=s_tr)

        yf = yield_function(p_tr, p_c_prev, q_tr, self.M)
        is_ep = yf > 0.0


        def elastic_update():
            stress_next = s_tr + p_tr * jnp.eye(3)
            eps_e_tr = eps_e_prev + deps_next

            return stress_next, eps_e_tr, p_c_prev

        def pull_to_ys():
            # https://github.com/patrick-kidger/optimistix/issues/132
            # Here we have safe values to avoid nan during compile time
            safe_mask = lambda x, default: jnp.where(is_ep, x, default)

            deps_e_v_tr_safe = safe_mask(deps_e_v_tr, 0.0)
            p_prev_safe = safe_mask(p_prev, 1.0) #
            p_c_prev_safe = safe_mask(p_c_prev, 2.0)
            q_tr_safe = safe_mask(q_tr, 1.0)
            q_tr_safe = jnp.maximum(q_tr_safe, 1e-6)

            # using logarithmic critical state pressure for better numerical stability
            u_p_c_prev =jnp.log(p_c_prev_safe)
            
            def residuals(sol, args):
                pmulti, u_p_c = sol
                p_c_next = jnp.exp(u_p_c)

                deps_p_v =  self._cp * (u_p_c - u_p_c_prev)

                p_next = get_pressure_mcc(deps_e_v_tr_safe - deps_p_v, self.kap, p_prev_safe)

                K_next = get_K(self.kap, p_next, self.K_min, self.K_max)

                G_next = get_G(self.nu, K_next)

                factor = 1 / (1 + 6.0 * G_next * pmulti)

                s_next = s_tr * factor

                q_next = q_tr_safe * factor

                deps_v_p_fr = pmulti * (2.0 * p_next - p_c_next) * self.M**2

                yf_next = yield_function(p_next, p_c_next, q_next, self.M)


                # maybe volumetric strains need rescaling for the residuals?
                # ... found not needed in tests
                deps_scale = 1.0

                yf_scale = K_tr
                R = jnp.array(
                    [yf_next / yf_scale, (deps_v_p_fr - deps_p_v) / deps_scale]
                )

                aux = (p_next, s_next, G_next, K_next)

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
                return x_new, None

            x_init = jnp.array([0.0, u_p_c_prev])

            # JAX unrolls the loops here
            x_final, _ = jax.lax.scan(step_fn, x_init, None, length=10)


            pmulti_curr, u_p_c_final = x_final[0], x_final[1]

            _, aux = residuals(x_final, None)

            p_c_next = jnp.exp(u_p_c_final)

            p_next, s_next, G_next, K_next = aux

            # Handles unconverged values, which may occur at low pressures
            # close to the free surface
            s_next = jnp.where(s_next, s_next, s_tr)
            p_next = jnp.where(p_next, p_next, p_prev_safe)
            p_c_next = jnp.where(p_c_next, p_c_next, p_c_prev_safe)


            stress_next = s_next + (p_next) * jnp.eye(3)

            eps_e_v_next = (p_next - p_0) / K_next

            eps_e_d_next = (s_next - s_0) / (2.0 * G_next)

            eps_e_next = eps_e_d_next + (1.0 / 3) * eps_e_v_next * jnp.eye(3)

            return stress_next, eps_e_next, p_c_next


        # We treat it as disconnected at low pressures
        stress_next, eps_e_next, p_c_next = jax.lax.cond(
            specific_volume <= self.N*0.999999,
            lambda: jax.lax.cond(is_ep, pull_to_ys, elastic_update),
            lambda: (0.0 * jnp.eye(3), eps_e_prev, p_c_prev),
        )


        return stress_next, eps_e_next, p_c_next

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
    "get_v_csl"
)

for _name in _helpers:
    _fn = globals().get(_name)
    if _fn is not None:
        setattr(ModifiedCamClay, _name, staticmethod(_fn))
