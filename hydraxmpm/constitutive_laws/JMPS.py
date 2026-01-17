import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Any
from jaxtyping import Float, Array

from ..material_points.material_points import MaterialPointState
from .constitutive_law import ConstitutiveLawState, ConstitutiveLaw
from ..utils.math_helpers import (
    get_dev_strain,
    get_dev_stress,
    get_pressure,
    get_q_vm,
    get_volumetric_strain,
    safe_inv_scalar_clamped,
    reconstruct_stress_from_triaxial,
)


def inv_3x3_robust(m, gradient_clip_val=1e6):
    """Inverts a 3x3 matrix using universal robust scalar inverse."""
    a, b, c = m[0, 0], m[0, 1], m[0, 2]
    d, e, f = m[1, 0], m[1, 1], m[1, 2]
    g, h, i = m[2, 0], m[2, 1], m[2, 2]

    A = e * i - f * h
    B = -(d * i - f * g)
    C = d * h - e * g
    D = -(b * i - c * h)
    E = a * i - c * g
    F = -(a * h - b * g)
    G = b * f - c * e
    H = -(a * f - c * d)
    I_ = a * e - b * d

    det = a * A + b * B + c * C
    inv_det = safe_inv_scalar_clamped(det, gradient_clip_val)

    inv = jnp.array([[A, D, G], [B, E, H], [C, F, I_]]) * inv_det
    return inv


def get_xi(p_s, p):
    return p_s / p


def get_A(xi, xi_0):
    return 1.0 - 2.0 * xi_0 + xi


def get_B(xi, beta):
    return 1.0 - beta + beta * xi


def get_eta_g_xi_2(B, A, xi):
    # (B * sqrt(1 - ((1-xi)/A)^2))^2
    return B * B * (1.0 - ((1.0 - xi) / A) ** 2)


def get_eta_g_I_2(I, I_eta, M, M_d):
    return (1.0 + I * ((M_d / M - 1.0) / (I_eta + I))) ** 2


def get_eta_g_xi(B, A, xi):
    return B * jnp.sqrt(1.0 - ((1.0 - xi) / A) ** 2)


def get_eta_g_I(I, I_eta, M, M_d):
    return 1.0 + I * ((M_d / M - 1.0) / (I_eta + I))


def get_ys(p, q, M, g_xi_2, g_I_2):
    return (q / (p * M)) ** 2 - g_xi_2 * g_I_2


def get_qp_ys(p, p_s, I, xi_0, beta, M, M_d, I_eta):

    xi = get_xi(p_s, p)
    A = get_A(xi, xi_0)
    B = get_B(xi, beta)

    g_I = get_eta_g_I(I, I_eta, M, M_d)
    g_xi = get_eta_g_xi(B, A, xi)
    return M * g_xi * g_I


def get_qp_I_csl(I, M, M_d, I_eta):
    g_I = get_eta_g_I(I, I_eta, M, M_d)
    return M  * g_I


def get_inertial_number(p, deps_s_p_dt, d, rho_p):
    p = jnp.maximum(p, 1e-6)
    gamma_dot = deps_s_p_dt
    return gamma_dot * d * jnp.sqrt(rho_p / p)


def get_deps_v_p_hardening(p_s, p_s_prev, dI, I_v, lam, kap):
    ratio = jnp.maximum(p_s / p_s_prev, 1e-12)
    return (lam - kap) * jnp.log(ratio) - dI / I_v


def get_v_ncl(p_ref, lam, kap, xi_0, gamma=None, N=None, p_c=None, p_s=None):
    """Critical state specific volume"""
    if p_c is None:
        p_c =  p_s*xi_0
    if N is None:
        N = gamma / ((1.0 / xi_0) ** (kap - lam))
    
    return N * (p_ref / p_c) ** lam


def get_v_csl(p_s, p_ref, gamma, lam):
    """
    Compute specific volume on the Critical State Line (CSL) at pressure p.

    """
    return gamma * (p_ref / p_s) ** lam

def get_v_I(I, I_v):
    """
    Compute specific volume on the Critical State Line (CSL) at pressure p.

    """
    return jnp.exp(I/I_v)

def get_v_I_csl(p_s, p_ref, gamma, lam, I, I_v):
    """Critical state specific volume with inertial correction"""
    v_csl = get_v_csl(p_s, p_ref, gamma, lam)
    v_I = get_v_I(I, I_v)
    return v_csl * v_I

def get_v_f_kappa(p_s, p, kap):
    """Critical state specific volume elastic state"""
    return (p_s / p) ** kap


def get_v_sl(p_s, p, p_ref, gamma, lam, kap):
    """Swelling line specific volume"""
    v_csl = get_v_csl(p_s, p_ref, gamma, lam)
    v_elas = get_v_f_kappa(p_s, p, kap)
    return v_csl * v_elas


class ParamMCCInertiaState(ConstitutiveLawState):
    p_s_stack: Float[Array, "num_points"] = None
    I_stack: Float[Array, "num_points"] = None
    stress_ref_stack: Float[Array, "num_points 3 3"] = None

    eps_e_stack: Float[Array, "num_points 3 3"] = None


class ParamMCCInertia(ConstitutiveLaw):
    nu: Float[Array, ""]
    lam: Float[Array, ""]
    kap: Float[Array, ""]
    M: Float[Array, ""]
    M_d: Float[Array, ""]

    # Inertial / Fluidity parameters
    I_v: Float[Array, ""]
    I_eta: Float[Array, ""]
    beta: Float[Array, ""]
    xi_0: Float[Array, ""]
    d: Float[Array, ""]  # grain diameter

    # Reference
    p_ref: Float[Array, ""] = 1000.0
    gamma: Float[Array, ""]
    rho_p: Float[Array, ""] = 2650.0

    # Stability
    p_min_calc: Float[Array, ""] = 10.0
    K_min: Float[Array, ""] = 1e6

    def __init__(
        self,
        *,
        nu: Float[Array, ""],
        M: Float[Array, ""],
        lam: Float[Array, ""],
        kap: Float[Array, ""],
        xi_0: Float[Array, ""] = 0.5,
        beta: Float[Array, ""] = 1.0,
        d: Float[Array, ""] = 1e-3,
        I_v: Float[Array, ""] = 1e20,
        I_eta: Float[Array, ""] = 1e20,
        M_d: Optional[Float[Array, ""]] = None,
        gamma: Optional[Float[Array, ""]] = None,
        N: Optional[Float[Array, ""]] = None,
        p_ref: Float[Array, ""] = 1000.0,
        rho_p: Float[Array, ""] = 2650.0,
        requires_F_reset: bool = True,
        **kwargs: Any,
    ):
        self.nu = nu
        self.M = M
        self.lam = lam
        self.kap = kap
        self.xi_0 = xi_0
        self.beta = beta
        self.d = d
        self.I_v = I_v
        self.I_eta = I_eta
        self.M_d = M * (1 + 1e-12) if M_d is None else M_d

        if gamma is None:
            gamma = N * (1.0 / xi_0) ** (kap - lam)

        self.gamma = gamma

        self.p_ref = p_ref
        self.rho_p = rho_p

        self.requires_F_reset = requires_F_reset

    def create_state_from_ocr(
        self,
        p_stack: Float[Array, "num_points"],
        ocr_stack: Float[Array, "num_points"] | Float[Array, ""],
        q_stack: Optional[Float[Array, "num_points"]] = None,
    ) -> Tuple[
        ParamMCCInertiaState, Float[Array, "num_points 3 3"], Float[Array, "num_points"]
    ]:
        """
        Initialization given the stress history (OCR),
        and derive the required specific volume (density).

        If q_stack is not provided we assume isotropic (q=)
        otherwise, we check if its within the yield surface.
        If outside, we project it back to the yield surface.

        Triaxial initiation

        We assume the relationship:
        ρ = ρₚ / v

        Where:
        v  = Specific Volume (1 + void_ratio)
        ρ  = Bulk Density (kg/m³)
        ρₚ = Particle Density (self.rho_p)


        Args:
            p_stack: Current hydrostatic pressure.
            ocr_stack: Over-Consolidation Ratio (p_c / p).
            q_stack: (Optional) Current shear stress.

        Returns:
            (LawState, specific_volume_stack)
        """

        if isinstance(ocr_stack, Float[Array, ""]):
            ocr_stack = jnp.full((p_stack.shape[0],), ocr_stack)

        I_stack = jnp.ones(p_stack.shape[0]) * 1e-22
        # Normal consolidation pressure
        p_c_stack = p_stack * ocr_stack

        p_s_stack = self.xi_0 * p_c_stack

        # Derive specific volume from swelling line
        specific_volume_stack = get_v_sl(
            p_s_stack, p_stack, self.p_ref, self.gamma, self.lam, self.kap
        )

        density_stack = self.rho_p / specific_volume_stack

        eps_e_stack = jnp.zeros((p_stack.shape[0], 3, 3))

        if q_stack is None:
            q_stack = jnp.zeros_like(p_stack)
        else:
            q_p_inp_stack = q_stack / p_stack

            q_p_ys_stack = get_qp_ys(
                p_stack,
                p_s_stack,
                0.0,
                self.xi_0,
                self.beta,
                self.M,
                self.M_d,
                self.I_eta,
            )

            yf_check = jnp.abs(q_p_inp_stack - q_p_ys_stack)

            q_p_stack = jnp.where(yf_check > 0, q_p_ys_stack, q_p_inp_stack)

            q_stack = q_p_stack * p_stack

        stress_ref_stack = reconstruct_stress_from_triaxial(
            p_stack=p_stack, q_stack=q_stack
        )

        law_state = ParamMCCInertiaState(
            stress_ref_stack=stress_ref_stack,
            p_s_stack=p_s_stack,
            eps_e_stack=eps_e_stack,
            I_stack=I_stack,
        )

        return (law_state, stress_ref_stack, density_stack)

    def create_state(self, mp_state: MaterialPointState) -> ParamMCCInertiaState:
        """Initializes state. Uses OCR=1 assumption by default."""
        p_0_stack = mp_state.pressure_stack

        # Simple Initialization: Assume Normal Consolidation (OCR=1)
        # p_s = xi_0 * p_c -> p_c = p
        p_s_stack = self.xi_0 * p_0_stack

        I_stack = jnp.zeros_like(p_0_stack)
        eps_e_stack = jnp.zeros_like(mp_state.stress_stack)

        # Initialize Reference Stress (Hypoelastic Anchor)
        stress_ref_stack = mp_state.stress_stack

        return ParamMCCInertiaState(
            p_s_stack=p_s_stack,
            I_stack=I_stack,
            stress_ref_stack=stress_ref_stack,
            eps_e_stack=eps_e_stack,
        )

    def update(
        self, mp_state: MaterialPointState, law_state: ParamMCCInertiaState, dt
    ) -> Tuple[MaterialPointState, ParamMCCInertiaState]:
        # return mp_state, law_state
        specific_volume_stack = self.rho_p / mp_state.density_stack

        # Vectorize over particles
        new_stress, new_eps_e, new_p_s, new_I = jax.vmap(
            self._update_stress, in_axes=(0, 0, 0, 0, 0, 0, 0, None)
        )(
            mp_state.L_stack,
            law_state.eps_e_stack,
            mp_state.stress_stack,
            law_state.stress_ref_stack,
            law_state.p_s_stack,
            law_state.I_stack,
            specific_volume_stack,
            dt,
        )

        new_mp = eqx.tree_at(lambda m: m.stress_stack, mp_state, new_stress)

        # Note: We keep stress_ref_stack constant during the step (it only updates at step boundaries)
        # But usually in MPM, 'mp_state.stress_stack' becomes the new 'stress_ref' for the NEXT step.
        # Here we just return the updated history variables.
        new_law = eqx.tree_at(
            lambda l: (l.p_s_stack, l.I_stack, l.eps_e_stack),
            law_state,
            (new_p_s, new_I, new_eps_e),
        )

        return new_mp, new_law

    def _update_stress(
        self,
        L,
        eps_e_prev,
        stress_prev,
        stress_ref,
        p_s_prev,
        I_prev,
        specific_volume,
        dt,
    ):
        # ---------------------------------------------------------------------
        # 1. SETUP & REFERENCE STATE
        # ---------------------------------------------------------------------
        # Strain increment via symmetric part of velocity gradient
        deps_next = 0.5 * (L + L.T) * dt

        # Get Reference State (Start of Step)
        p_0 = get_pressure(stress_ref)
        p_0 = jnp.maximum(p_0, self.p_min_calc)
        s_0 = get_dev_stress(stress_ref, pressure=p_0)

        # Get Previous State (Current Integration Point)
        # Usually p_prev ~ p_0 at start, but distinct for sub-stepping
        p_prev = get_pressure(stress_prev)
        p_prev = jnp.maximum(p_prev, self.p_min_calc)
        s_prev = get_dev_stress(stress_prev, pressure=p_prev)

        # ---------------------------------------------------------------------
        # 2. TRIAL STEP (ELASTIC PREDICTOR)
        # ---------------------------------------------------------------------
        deps_e_v_tr = get_volumetric_strain(deps_next)

        # Hypoelastic Volumetric Update (Bi-Logarithmic)
        # p_tr = p_prev * exp(deps_v / kappa)
        p_tr = p_prev * jnp.exp(deps_e_v_tr / self.kap)

        # Compute Stiffness at Midpoint (approx) or Trial State
        p_avg = 0.5 * (p_prev + p_tr)
        K_const = p_avg / self.kap
        K_const = jnp.maximum(K_const, self.K_min)
        G_const = (3.0 * (1.0 - 2.0 * self.nu) / (2.0 * (1.0 + self.nu))) * K_const

        # Hypoelastic Deviatoric Update
        deps_e_d_tr = get_dev_strain(deps_next, deps_e_v_tr)
        s_tr = 2.0 * G_const * deps_e_d_tr + s_prev
        q_tr = get_q_vm(dev_stress=s_tr)

        # Yield Surface Check
        xi_tr = get_xi(p_s_prev, p_tr)
        A_tr = get_A(xi_tr, self.xi_0)
        B_tr = get_B(xi_tr, self.beta)

        # Assuming I ~ I_prev (or 0) for yield check
        g_xi_tr_2 = get_eta_g_xi_2(B_tr, A_tr, xi_tr)
        g_I_tr_2 = get_eta_g_I_2(I_prev, self.I_eta, self.M, self.M_d)

        yf = get_ys(p_tr, q_tr, self.M, g_xi_tr_2, g_I_tr_2)

        is_ep = yf > 0.0

        # ---------------------------------------------------------------------
        # 3. RETURN MAPPING (CORRECTOR)
        # ---------------------------------------------------------------------

        def elastic_update():
            # Stress update is just the trial
            stress_next = s_tr - p_tr * jnp.eye(3)
            eps_e_tr = eps_e_prev + deps_next
            # History vars unchanged (I returns to 0 or stays? Old code implied 0)
            return stress_next, eps_e_tr, p_s_prev, 0.0

        def pull_to_ys():
            # Safe variables for AD (prevent NaN in masked branches)
            p_prev_safe = jnp.maximum(p_prev, 1.0)
            p_s_prev_safe = jnp.maximum(p_s_prev, 1.0)
            q_tr_safe = jnp.maximum(q_tr, 1e-6)

            # Variables to solve: [pmulti, log(p_s), I]
            u_p_s_prev = jnp.log(p_s_prev_safe)

            def residuals(sol, _):
                pmulti, u_p_s_next, I_next = sol
                p_s_next = jnp.exp(u_p_s_next)

                # A. Consistency: Hardening
                dI_next = I_next - I_prev
                deps_p_v = get_deps_v_p_hardening(
                    p_s_next, p_s_prev_safe, dI_next, self.I_v, self.lam, self.kap
                )

                # B. Consistency: Volumetric Stress
                # p_next = p_prev * exp((deps_tot - deps_p) / kap)
                deps_e_v = deps_e_v_tr - deps_p_v
                p_next = p_prev_safe * jnp.exp(deps_e_v / self.kap)

                # C. Consistency: Deviatoric Stress (Radial Return)
                factor = 1.0 / (1.0 + 6.0 * G_const * pmulti)
                q_next = q_tr_safe * factor
                deps_s_p = 2.0 * pmulti * q_next

                # D. Consistency: Inertial Number
                deps_s_p_dt = deps_s_p / dt
                I_next_fr = get_inertial_number(
                    p_s_next, deps_s_p_dt, self.d, self.rho_p
                )

                # E. Consistency: Flow Rule / Yield Surface
                xi_next = get_xi(p_s_next, p_next)
                A_next = get_A(xi_next, self.xi_0)
                B_next = get_B(xi_next, self.beta)

                g_xi_2 = get_eta_g_xi_2(B_next, A_next, xi_next)
                g_I_2 = get_eta_g_I_2(I_next, self.I_eta, self.M, self.M_d)

                yf_next = get_ys(p_next, q_next, self.M, g_xi_2, g_I_2)

                deps_v_p_fr = (
                    2.0
                    * pmulti
                    * (p_next - p_s_next)
                    * ((self.M * B_next) / A_next) ** 2
                )

                R = jnp.array(
                    [
                        yf_next,  # Yield Surface = 0
                        (deps_v_p_fr - deps_p_v),  # Plastic Flow consistency
                        (I_next - I_next_fr),  # Inertial Law consistency
                    ]
                )

                aux = (p_next, factor, I_next)
                return R, aux

            def step_fn(carry, _):
                x = carry
                R_func = lambda v: residuals(v, None)[0]

                # Forward Mode AD for Jacobian
                J = jax.jacfwd(R_func)(x)
                R = R_func(x)
                inv_J = inv_3x3_robust(J)
                dx = -(inv_J @ R)

                x_new = x + 0.8 * dx  # Damping

                # Nan Guard
                is_bad = jnp.any(jnp.isnan(x_new)) | jnp.any(jnp.isinf(x_new))
                x_new = jnp.where(is_bad, x, x_new)

                # Constraints
                x_new = x_new.at[0].set(jnp.maximum(x_new[0], 0.0))  # pmulti >= 0

                return x_new, None

            # Initial Guess
            x_init = jnp.array([1e-12, u_p_s_prev, I_prev])

            # Solve
            x_final, _ = jax.lax.scan(step_fn, x_init, None, length=12)

            # Extract Results
            pmulti_f, u_p_s_f, I_f = x_final
            _, aux = residuals(x_final, None)
            p_next, factor, I_next = aux

            # Reconstruct State
            s_next = s_tr * factor
            stress_next = s_next - p_next * jnp.eye(3)
            p_s_next = jnp.exp(u_p_s_f)

            # Reconstruct Elastic Strain (Tracking) using Reference State
            # eps_e = (sigma - sigma_0) / C ??
            # Here we just use the simplified bulk relation for tracking
            eps_e_v_next = (p_next - p_0) / K_const
            eps_e_d_next = (s_next - s_0) / (2.0 * G_const)
            eps_e_next = eps_e_d_next - (1.0 / 3.0) * eps_e_v_next * jnp.eye(3)

            return stress_next, eps_e_next, p_s_next, I_next

        stress_next, eps_e_next, p_s_next, I_next = jax.lax.cond(
            is_ep, pull_to_ys, elastic_update
        )

        return stress_next, eps_e_next, p_s_next, I_next


# Export helper names for API access
_helpers = (
    "get_xi",
    "get_ys",
    "get_v_ncl",
    "get_v_csl",
    "get_v_sl",
    "get_inertial_number",
    "get_deps_v_p_hardening",
    "get_v_I_csl",
    "get_qp_I_csl"
)
for _name in _helpers:
    _fn = globals().get(_name)
    if _fn is not None:
        setattr(ParamMCCInertia, _name, staticmethod(_fn))
