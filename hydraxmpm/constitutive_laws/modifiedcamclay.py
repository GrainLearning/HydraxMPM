from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp

import optimistix as optx
from typing_extensions import Optional, Self, Tuple

from ..common.types import (
    TypeFloat,
    TypeFloatScalarPStack,
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
from .constitutive_law import ConstitutiveLaw


def yield_function(p_hat, px_hat, q, M):
    return (q * q) / (M * M) + p_hat * p_hat - px_hat * p_hat


def get_state_boundary_layer(p_hat, q, M, cp, lam, p_t, ln_N):
    return (
        ln_N
        - lam * jnp.log(p_hat / (1.0 + p_t))
        - cp * jnp.log(1.0 + (q / p_hat) ** 2 / M**2)
    )


def get_p_hat(deps_e_v, kap, p_hat_prev):
    """Compute non-linear pressure."""
    p_hat = p_hat_prev / (1.0 - (1.0 / kap) * deps_e_v)
    return jnp.nanmax(jnp.array([p_hat, 1e-12]))
    # return jnp.nanmax(jnp.array([p_hat, 2.0]))
    # return p_hat


def get_px_hat_mcc(px_hat_prev, cp, deps_p_v):
    """Compute non-linear pressure."""
    px_hat = px_hat_prev / (1.0 - (1.0 / cp) * deps_p_v)
    return jnp.nanmax(jnp.array([px_hat, 1e-12]))
    # return jnp.nanmax(jnp.array([px_hat, 2.0]))
    # return px_hat


def get_s(deps_e_d, G, s_prev):
    return 2.0 * G * deps_e_d + s_prev


def get_K(kap, p_hat):
    return (1.0 / kap) * (p_hat)


def get_G(nu, K):
    return (3.0 * (1.0 - 2.0 * nu)) / (2.0 * (1.0 + nu)) * K


def give_phi_0(p, R, ln_N, lam, kap, p_t):
    """Assume q =0, give reference solid volume fraction"""
    p_hat = p + p_t

    ln_v = ln_N - lam * jnp.log(R * p) + kap * jnp.log(R)
    phi = 1.0 / jnp.exp(ln_v)
    # ln_eta = get_state_boundary_layer(p_hat, 0.0, 1, (lam - kap), lam, p_t, ln_N)
    # ln_v = (lam - kap) * jnp.log(R) + ln_eta

    return phi


def give_p_0(phi, R, ln_N, lam, kap, p_t):
    ln_v = jnp.log(1.0 / phi)

    ln_eta = ln_v - (lam - kap) * jnp.log(R)

    p_hat = (1.0 + p_t) * jnp.exp((ln_N - ln_eta) / lam)

    return p_hat - p_t


class ModifiedCamClay(ConstitutiveLaw):
    nu: TypeFloat
    M: TypeFloat
    R: TypeFloat
    lam: TypeFloat
    kap: TypeFloat
    p_t: TypeFloat = 0.0
    ln_N: TypeFloat

    _cp: TypeFloat  # ?

    px_hat_stack: Optional[TypeFloatScalarPStack] = None
    p_0_stack: Optional[TypeFloatScalarPStack] = None

    def __init__(
        self: Self,
        nu: TypeFloat,
        M: TypeFloat,
        R: TypeFloat,
        lam: TypeFloat,
        kap: TypeFloat,
        ln_N: TypeFloat,
        p_t: Optional[TypeFloat] = 0.0,
        **kwargs,
    ) -> Self:
        self.nu = nu

        self.M = M

        self.R = R

        self.lam = lam

        self.kap = kap

        self.p_t = p_t

        self.ln_N = ln_N

        self._cp = lam - kap

        self.eps_e_stack = kwargs.get("eps_e_stack")

        self.px_hat_stack = kwargs.get("px_hat_stack")

        self.p_0_stack = kwargs.get("p_0_stack")

        super().__init__(**kwargs)

    def init_state(self: Self, material_points: MaterialPoints):
        # There are two ways to initialize via a reference pressure or reference density
        # these can be given as a scalar or array

        p_0 = self.p_0
        if p_0 is None:
            p_0 = material_points.p_stack

        phi_0 = self.phi_0

        if self.init_by_density:
            if eqx.is_array(phi_0):
                vmap_give_p_ref = partial(
                    jax.vmap, in_axes=(0, None, None, None, None, None)
                )(give_p_0)
            else:
                vmap_give_p_ref = give_p_0

            p_0 = vmap_give_p_ref(
                phi_0, self.R, self.ln_N, self.lam, self.kap, self.p_t
            )
        else:
            if eqx.is_array(p_0):
                vmap_give_ln_v0 = partial(jax.vmap)(self.get_ln_v0)
            else:
                vmap_give_ln_v0 = self.get_ln_v0

            ln_v0 = vmap_give_ln_v0(p_0)

            phi_0 = 1.0 / jnp.exp(ln_v0)

        material_points = material_points.init_stress_from_p_0(p_0)

        rho_0 = phi_0 * self.rho_p
        material_points = material_points.init_mass_from_rho_0(rho_0)
        params = self.__dict__

        # can we handle making this scalar or array, both ?
        p_0_stack = jnp.zeros(material_points.num_points).at[:].set(p_0)
        px_hat_stack = p_0_stack * self.R

        eps_e_stack = jnp.zeros((material_points.num_points, 3, 3))

        W_stack = None
        if self.approx_strain_energy_density:
            W_stack = jnp.zeros(material_points.num_points)

        P_stack = None
        if self.approx_stress_power:
            P_stack = jnp.zeros(material_points.num_points)

        params.update(
            rho_0=rho_0,
            p_0=p_0,
            p_0_stack=p_0_stack,
            px_hat_stack=px_hat_stack,
            eps_e_stack=eps_e_stack,
            W_stack=W_stack,
            P_stack=P_stack,
        )
        return self.__class__(**params), material_points

    def update(
        self: Self,
        material_points: MaterialPoints,
        dt: TypeFloat,
        dim: Optional[TypeInt] = 3,
    ) -> Tuple[MaterialPoints, Self]:
        """Update the material state and particle stresses for MPM solver."""

        deps_dt_stack = material_points.depsdt_stack
        new_stress_stack, new_eps_e_stack, new_px_hat_stack = self.vmap_update_ip(
            deps_dt_stack * dt,
            self.eps_e_stack,
            material_points.stress_stack,
            self.px_hat_stack,
            self.p_0_stack,
            material_points.specific_volume_stack(self.rho_p),
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

        new_self = new_self.post_update(new_stress_stack, deps_dt_stack, dt)
        return new_material_points, new_self

    @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0, 0), out_axes=(0, 0, 0))
    def vmap_update_ip(
        self: Self,
        deps_next,
        eps_e_prev,
        stress_prev,
        px_hat_prev,
        p_ref,
        specific_volume,
    ):
        p_prev = get_pressure(stress_prev)

        p_hat_prev = p_prev + self.p_t

        s_prev = get_dev_stress(stress_prev, pressure=p_prev)

        deps_e_v_tr = get_volumetric_strain(deps_next)

        p_hat_tr = get_p_hat(deps_e_v_tr, self.kap, p_hat_prev)

        deps_e_d_tr = get_dev_strain(deps_next, deps_e_v_tr)

        K_tr = get_K(self.kap, p_hat_tr)

        G_tr = get_G(self.nu, K_tr)

        s_tr = get_s(deps_e_d_tr, G_tr, s_prev)

        q_tr = get_q_vm(dev_stress=s_tr)

        yf = yield_function(p_hat_tr, px_hat_prev, q_tr, self.M)

        def elastic_update():
            stress_next = s_tr - (p_hat_tr - self.p_t) * jnp.eye(3)
            eps_e_tr = eps_e_prev + deps_next
            return stress_next, eps_e_tr, px_hat_prev

        def pull_to_ys():
            def residuals(sol, args):
                pmulti, deps_p_v = sol

                pmulti = jnp.nanmax(jnp.array([pmulti, 0.0]))

                p_hat_next = get_p_hat(deps_e_v_tr - deps_p_v, self.kap, p_hat_prev)

                K_next = get_K(self.kap, p_hat_next)

                G_next = get_G(self.nu, K_next)

                factor = 1.0 / (1.0 + 6.0 * G_next * pmulti)

                s_next = s_tr * factor

                q_next = q_tr * factor

                px_hat_next = get_px_hat_mcc(px_hat_prev, self._cp, deps_p_v)

                deps_v_p_fr = pmulti * (2.0 * p_hat_next - px_hat_next) * self.M**2

                yf_next = yield_function(p_hat_next, px_hat_next, q_next, self.M)

                R = jnp.array([yf_next, deps_v_p_fr - deps_p_v])

                aux = (p_hat_next, s_next, px_hat_next, G_next, K_next)

                return R, aux

            def find_roots():
                """Find roots of the residuals function."""

                # avoiding non-finite values

                init_val = jnp.array([0.0, 0.0])

                solver = optx.Newton(
                    rtol=1e-8,
                    atol=1e-8,
                )
                sol = optx.root_find(
                    residuals, solver, init_val, throw=False, has_aux=True, max_steps=20
                )
                return sol.value

            pmulti_curr, deps_p_v_next = jax.lax.stop_gradient(find_roots())
            R, aux = residuals([pmulti_curr, deps_p_v_next], None)

            p_hat_next, s_next, px_hat_next, G_next, K_next = aux

            s_next = eqx.error_if(s_next, jnp.isnan(s_next).any(), "s_next is nan")

            p_next = p_hat_next - self.p_t

            stress_next = s_next - (p_next) * jnp.eye(3)

            eps_e_v_next = (p_next - p_ref) / K_next

            eps_e_d_next = s_next / (2.0 * G_next)

            eps_e_next = eps_e_d_next - (1.0 / 3) * eps_e_v_next * jnp.eye(3)

            return stress_next, eps_e_next, px_hat_next

        stress_next, eps_e_next, px_hat_next = jax.lax.cond(
            ((p_hat_tr >= 0.0) & (jnp.log(specific_volume) < self.ln_N)),
            lambda: jax.lax.cond(yf > 0.0, pull_to_ys, elastic_update),
            lambda: (jnp.zeros((3, 3)), jnp.zeros((3, 3)), px_hat_prev),
        )

        return stress_next, eps_e_next, px_hat_next

    @property
    def GAMMA(self):
        """Reference (natural) logarithmic specific volume of critical state line (CSL) at 1kPa

        #     Returns ln_GAMMA
        #"""

        return self.ln_N - (self.lam - self.kap) * jnp.log(2)

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

    # def SL(p):

    def get_p_0(self, ln_v0):
        return self.px_hat_stack * jnp.exp(
            (ln_v0, self.ln_N + self.lam * self.px_hat_stack) / self.kap
        ) ** (-1)

    def get_ln_v0(self, p_0):
        pc0 = self.R * p_0
        return self.ln_N - self.lam * jnp.log(pc0) + self.kap * jnp.log(self.R)

    # def ln_Vk(self, ln_v, p_0):
    #     """Reference (natural) logarithmic specific volume of swelling line (SL) at 1kPa
    #     (input current specific volume/ pressure)
    #     Returns ln_GAMMA
    #     """
    # return ln_v + self.kap * jnp.log(p_0 / self.R)

    def SL(self, p, ln_v0, p_0, return_ln=False):
        ln_v_sl = ln_v0 + self.kap * jnp.log(p_0)
        ln_v = ln_v_sl - self.kap * jnp.log(p)

        if return_ln:
            return ln_v
        else:
            return jnp.exp(ln_v)

    def get_critical_time(self, material_points, cell_size, alpha=0.5):
        p_stack = material_points.p_stack
        rho_stack = material_points.rho_stack
        K = get_K(self.kap, p_stack)
        G = get_G(self.nu, K)

        # dilation timestep
        cdil = jnp.sqrt((K + (4 / 3) * G) / rho_stack)

        dt = alpha * jnp.min(cell_size / cdil)
        return dt
