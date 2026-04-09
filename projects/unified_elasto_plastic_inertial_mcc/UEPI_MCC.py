import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Any
from jaxtyping import Float, Array


import hydraxmpm as hdx

###########################################################################
# Modified Cam Clay specific helpers
###########################################################################


def get_K(kap, p):
    """Get pressure dependent bulk modulus"""
    K = p / kap
    return K


def get_G(nu, K):
    """Get shear modulus"""
    G = (3 * (1 - 2 * nu) / (2 * (1 + nu))) * K
    return G


def get_xi(p_s, p):
    """Get state parameter with shift pressure / pressure"""
    return p_s / p


def get_M_xi(xi):
    """Get modified cam clay product correction function"""
    return jnp.sqrt(2.0 * xi - 1.0)


def get_M_xi_2(xi):
    return 2.0 * xi - 1.0


def get_mcc_M(xi, M_csl):
    """Get yield surface as stress ratio for mcc"""
    return M_csl * get_M_xi(xi)


def get_v_ncl(p_c, p_ref, N, lam):
    """Get specific volume on the normal consolidation line"""
    return N * (p_ref / p_c) ** lam


def get_v_csl(p_s, p_ref, gamma, lam):
    """Get specific volume on the critical state line"""
    return gamma * (p_ref / p_s) ** lam


def get_v_xi(xi, kap):
    """Compressibility specific volume correction function"""
    return xi**kap


def get_mcc_sl(xi, p_s, p_ref, gamma, lam, kap):
    """Compressibility specific volume correction function"""
    v_csl = get_v_csl(p_s=p_s, p_ref=p_ref, gamma=gamma, lam=lam)
    v_xi = get_v_xi(xi=xi, kap=kap)
    return v_csl * v_xi


def convert_gamma_N(gamma, lam, kap):
    """Convert between gamma and N for linear swelling line"""
    return gamma * 2 ** -(kap - lam)


def get_deps_v_p(p, xi, pmulti):
    direction = 2 * (p - p * xi) / (p * xi) ** 2
    return pmulti * direction


def get_deps_s_p(p, xi, q, M, pmulti):
    direction = 2 * q / (M * p * xi) ** 2
    return pmulti * direction


###########################################################################
# Inertial Steady State helper functions
###########################################################################


def get_plastic_inertial_number(p, dot_epsq_p, d, rho_p):
    """Get plastic inertial number I_p"""
    p = jnp.maximum(p, 1e-12)
    return dot_epsq_p * d * jnp.sqrt(rho_p / p)


def solid_volume_fraction_to_specific_volume(v):
    """Get solid volume fraction from specific volume"""
    return 1 / v


def get_M_I(I, I_M, M_csl, M_inf):
    """Get inertial steady state product correction function"""
    return 1.0 + I * ((M_inf / M_csl - 1.0) / (I_M + I))


def get_M_I_2(I, I_M, M_csl, M_inf):
    return (1.0 + I * ((M_inf / M_csl - 1.0) / (I_M + I))) ** 2


def get_iss_M(I, I_M, M_csl, M_inf):
    """Get inertial steady state product correction function"""
    return M_csl * get_M_I(I, I_M, M_csl, M_inf)


def get_v_I(I, I_v):
    """
    Compute specific volume on the Critical State Line (CSL).
    """
    return jnp.exp(I / I_v)


def get_iss_v(v_csl, I, I_v):
    """Inertial steady state specific volume with inertial correction"""
    v_I = get_v_I(I, I_v)
    return v_csl * v_I


###########################################################################
# Get unified elasto-plastic-inertial product function
###########################################################################


def get_unified_M(xi, I, M_csl, M_inf, I_M):
    M_xi = get_M_xi(xi)
    M_I = get_M_I(I, I_M, M_csl, M_inf)
    return M_csl * M_xi * M_I


def get_unified_M_2(xi, I, M_csl, M_inf, I_M):
    M_xi_2 = get_M_xi_2(xi)
    M_I_2 = get_M_I_2(I, I_M, M_csl, M_inf)
    return (M_csl**2) * M_xi_2 * M_I_2


def get_unified_iys(p, q, M_csl, M_xi_2, M_I_2):
    return (q / (p * M_csl + 1e-12)) ** 2 - M_xi_2 * M_I_2


def get_unified_isl(xi, p, p_ref, gamma, lam, kap):
    p_s = xi * p
    v_csl = get_v_csl(p_s, p_ref, gamma, lam)
    v_xi = get_v_xi(xi, kap)
    return v_csl * v_xi


def get_deps_v_p_hardening(p_s, p_s_prev, dI, I_v, lam, kap):
    ratio = jnp.maximum(p_s / p_s_prev, 1e-12)
    return (lam - kap) * jnp.log(ratio) - dI / I_v


def get_ps_I(p_s, I, I_v):
    return p_s * jnp.exp(I / I_v)


def get_unified_iys_res(p, p_s, q, M_csl, M_I):
    """This form is useful for residuals in return mapping"""
    M_iss = M_csl * M_I
    p_c = p_s * 2.0
    return ((q * q) / (M_iss * M_iss)) + p * p - p_c * p


class UEPI_MCCState(hdx.ConstitutiveLawState):
    p_s_stack: Float[Array, "num_points"] = None
    stress_ref_stack: Float[Array, "num_points 3 3"] = None
    I_p_stack: Float[Array, "num_points"] = None
    eps_e_stack: Float[Array, "num_points 3 3"] = None


class UEPI_MCC(hdx.ConstitutiveLaw):

    # Modified Cam Clay
    p_ref: float | Float[Array, ""] = 1000.0
    nu: float | Float[Array, ""]
    lam: float | Float[Array, ""]
    kap: float | Float[Array, ""]
    M_csl: float | Float[Array, ""]
    gamma: float | Float[Array, ""]

    # Inertial Steady State
    M_inf: float | Float[Array, ""]
    I_v: float | Float[Array, ""]
    I_M: float | Float[Array, ""]
    rho_p: float | Float[Array, ""] = 2650.0
    d: float | Float[Array, ""]  # grain diameter

    def __init__(
        self,
        *,
        nu: float | Float[Array, ""],
        M_csl: float | Float[Array, ""],
        lam: float | Float[Array, ""],
        kap: float | Float[Array, ""],
        d: float | Float[Array, ""] = 1e-3,
        I_v: float | Float[Array, ""] = 1e20,
        I_M: float | Float[Array, ""] = 1e20,
        M_inf: Optional[float | Float[Array, ""]] = None,
        gamma: Optional[float | Float[Array, ""]] = None,
        N: Optional[float | Float[Array, ""]] = None,
        p_ref: float | Float[Array, ""] = 1000.0,
        rho_p: float | Float[Array, ""] = 2650.0,
        **kwargs: Any,
    ):
        self.p_ref = p_ref
        self.nu = nu
        self.M_csl = M_csl
        self.lam = lam
        self.kap = kap

        if gamma is None:
            gamma = N * 2.0 ** (self.kap - self.lam)

        self.gamma = gamma

        self.d = d
        self.rho_p = rho_p
        self.I_v = I_v
        self.I_M = I_M
        self.M_inf = M_csl * (1 + 1e-12) if M_inf is None else M_inf

    @property
    def N(self):
        return self.gamma * 2.0 ** -(self.kap - self.lam)

    def create_state_from_ocr(
        self,
        p_stack: Float[Array, "num_points"],
        ocr_stack: Float[Array, "num_points"] | Float[Array, ""],
        q_stack: Optional[Float[Array, "num_points"]] = None,
    ) -> Tuple[
        UEPI_MCCState, Float[Array, "num_points 3 3"], Float[Array, "num_points"]
    ]:
        """
        Initialization given pressure, OCR and deviatoric shear

        """

        if isinstance(ocr_stack, Float[Array, ""]):
            ocr_stack = jnp.full((p_stack.shape[0],), ocr_stack)

        I_p_stack = jnp.ones(p_stack.shape[0]) * 1e-22

        p_c_stack = p_stack * ocr_stack

        p_s_stack = p_c_stack / 2.0

        # Derive specific volume from swelling line
        # Assume plastic inertial number is negligible at initialization
        xi_stack = get_xi(p_s_stack, p_stack)
        specific_volume_stack = get_mcc_sl(
            xi_stack, p_s_stack, self.p_ref, self.gamma, self.lam, self.kap
        )

        density_stack = self.rho_p / specific_volume_stack

        eps_e_stack = jnp.zeros((p_stack.shape[0], 3, 3))

        q_stack = jnp.zeros_like(p_stack)

        stress_ref_stack = hdx.reconstruct_stress_from_triaxial(
            p_stack=p_stack, q_stack=q_stack
        )

        law_state = UEPI_MCCState(
            stress_ref_stack=stress_ref_stack,
            p_s_stack=p_s_stack,
            eps_e_stack=eps_e_stack,
            I_p_stack=I_p_stack,
        )

        return (law_state, stress_ref_stack, density_stack)

    def update(
        self, mp_state: hdx.MaterialPointState, law_state: UEPI_MCCState, dt
    ) -> Tuple[hdx.MaterialPointState, UEPI_MCCState]:

        specific_volume_stack = self.rho_p / mp_state.density_stack

        new_stress, new_eps_e, new_p_s, new_I_p = jax.vmap(
            self._update_stress, in_axes=(0, 0, 0, 0, 0, 0, 0, None)
        )(
            mp_state.L_stack,
            law_state.eps_e_stack,
            mp_state.stress_stack,
            law_state.stress_ref_stack,
            law_state.p_s_stack,
            law_state.I_p_stack,
            specific_volume_stack,
            dt,
        )

        new_mp = eqx.tree_at(lambda m: m.stress_stack, mp_state, new_stress)

        new_law = eqx.tree_at(
            lambda l: (l.p_s_stack, l.I_p_stack, l.eps_e_stack),
            law_state,
            (new_p_s, new_I_p, new_eps_e),
        )

        return new_mp, new_law

    def _update_stress(
        self,
        L,
        eps_e_prev,
        stress_prev,
        stress_ref,
        p_s_prev,
        I_p_prev,
        specific_volume,
        dt,
    ):
        ###########################################################################
        # Setup reference state
        ###########################################################################

        D = hdx.get_sym_tensor(L)
        W = hdx.get_spin_tensor(L)

        # Strain increment
        deps_next = D * dt

        stress_prev_rot = hdx.get_jaumann_increment(stress_prev, W, dt)
        eps_e_prev_rot = hdx.get_jaumann_increment(eps_e_prev, W, dt)

        # Use these rotated values for the rest of the calculation
        stress_prev = stress_prev_rot
        eps_e_prev = eps_e_prev_rot

        # Get reference state
        p_0 = hdx.get_pressure(stress_ref)
        s_0 = hdx.get_dev_stress(stress_ref, pressure=p_0)

        # Get previous state
        p_prev = hdx.get_pressure(stress_prev)
        s_prev = hdx.get_dev_stress(stress_prev, pressure=p_prev)

        ###########################################################################
        # Compute trial state (Elastic Predictor)
        ###########################################################################
        deps_e_v_tr = hdx.get_volumetric_strain(deps_next)

        # Hypoelastic Volumetric Update (bi-Logarithmic)
        p_tr = p_prev * jnp.exp(deps_e_v_tr / self.kap)

        # Compute Stiffness at Midpoint (approx) or Trial State
        p_avg = 0.5 * (p_prev + p_tr)

        K_const = get_K(self.kap, p_avg)
        G_const = get_G(self.nu, K_const)

        deps_e_d_tr = hdx.get_dev_strain(deps_next, deps_e_v_tr)

        s_tr = 2.0 * G_const * deps_e_d_tr + s_prev
        q_tr = hdx.get_q_vm(dev_stress=s_tr)

        yf = get_unified_iys_res(p_tr, p_s_prev, q_tr, self.M_csl, 1.0)

        is_ep = yf > 0.0

        ###########################################################################
        # Return mapping
        ###########################################################################

        def elastic_update():
            stress_next = s_tr + p_tr * jnp.eye(3)
            eps_e_tr = eps_e_prev + deps_next

            return stress_next, eps_e_tr, p_s_prev, 1e-22

        def pull_to_ys():
            # Safe variables for AD (prevent NaN in masked branches)
            p_prev_safe = jnp.maximum(p_prev, 1.0)
            p_s_prev_safe = jnp.maximum(p_s_prev, 1.0)
            q_tr_safe = jnp.maximum(q_tr, 1e-6)

            # Variables to solve: [pmulti, log(p_s)]
            u_p_s_prev = jnp.log(p_s_prev_safe)

            def residuals(sol, _):
                pm_pmulti, u_p_s_next = sol
                p_s_next = jnp.exp(u_p_s_next)

                pmulti = pm_pmulti

                factor = 1.0 / (1.0 + 6.0 * G_const * pmulti)

                q_next = q_tr_safe * factor

                deps_s_p = 2.0 * pmulti * q_next

                deps_s_p_dt = deps_s_p / dt

                I_p_next = get_plastic_inertial_number(
                    jnp.maximum(p_s_next, self.p_ref), deps_s_p_dt, self.d, self.rho_p
                )

                dI = I_p_next - I_p_prev

                deps_p_v = (self.lam - self.kap) * (u_p_s_next - u_p_s_prev) - (
                    dI / self.I_v
                )

                deps_e_v = deps_e_v_tr - deps_p_v

                p_next = p_prev_safe * jnp.exp(deps_e_v / self.kap)

                deps_v_p_fr = pmulti * (2.0 * (p_next - p_s_next)) * (self.M_csl) ** 2


                M_I = get_M_I(I_p_next, self.I_M, self.M_csl, self.M_inf)

                yf_next = get_unified_iys_res(p_next, p_s_next, q_next, self.M_csl, M_I)

                # scaling residuals
                R = jnp.array([yf_next, (deps_v_p_fr - deps_p_v) * 1e4])

                aux = (p_next, factor, I_p_next)
                return R, aux

            def step_fn(carry, _):
                x = carry
                R_func = lambda v: residuals(v, None)[0]

                #Forward mode AD for jacobian
                J = jax.jacfwd(R_func)(x)
                R = R_func(x)
                inv_J = hdx.inv_2x2_robust(J)
                dx = -(inv_J @ R)
                dx = dx.at[1].set(jnp.clip(dx[1], -0.1, 0.1))
                x_new = x + 0.4 * dx 

                # nan guard
                is_bad = jnp.any(jnp.isnan(x_new)) | jnp.any(jnp.isinf(x_new))
                x_new = jnp.where(is_bad, x, x_new)

                # constraints
                x_new = x_new.at[0].set(jnp.maximum(x_new[0], 0.0)) 

                return x_new, None

            # initial guess
            x_init = jnp.array([0.0, u_p_s_prev])

            # solve
            x_final, _ = jax.lax.scan(step_fn, x_init, None, length=20)

            # extract results
            pmulti_f, u_p_s_f = x_final
            _, aux = residuals(x_final, None)
            p_next, factor, I_p_next = aux

            # reconstruct state
            s_next = s_tr * factor
            p_s_next = jnp.exp(u_p_s_f)

            stress_next = s_next + p_next * jnp.eye(3)

            # Reconstruct Elastic Strain using reference state for tracking
            eps_e_v_next = (p_next - p_0) / K_const
            eps_e_d_next = (s_next - s_0) / (2.0 * G_const)
            eps_e_next = eps_e_d_next + (1.0 / 3.0) * eps_e_v_next * jnp.eye(3)

            return stress_next, eps_e_next, p_s_next, I_p_next

        stress_next, eps_e_next, p_s_next, I_p_next = jax.lax.cond(
            is_ep, pull_to_ys, elastic_update
        )
        return stress_next, eps_e_next, p_s_next, I_p_next
