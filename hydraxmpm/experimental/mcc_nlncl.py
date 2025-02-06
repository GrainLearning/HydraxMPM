from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from typing_extensions import Optional, Self, Tuple, Union

from ..common.types import (
    TypeFloat,
    TypeFloatMatrix3x3,
    TypeFloatMatrixPStack,
    TypeFloatScalarPStack,
    TypeInt,
)
from ..materials.material import Material
from ..particles.particles import Particles
from ..utils.math_helpers import (
    # get_dev_strain,
    # get_hencky_strain_stack,
    get_dev_strain,
    get_dev_stress,
    get_hencky_strain,
    get_pressure,
    get_q_vm,
    get_scalar_shear_strain,
    get_sym_tensor_stack,
    get_volumetric_strain,
)


def yield_function(p_hat, px_hat, q, M, ps):
    p_hat = p_hat - ps
    px_hat = px_hat
    return (q * q) / (M * M) + p_hat * p_hat - px_hat * p_hat


def get_state_boundary_layer(p_hat, q, M, cp, lam, ps, ln_N):
    return (
        ln_N
        - lam * jnp.log(p_hat / (1.0 + ps))
        - cp * jnp.log(1.0 + (q / p_hat) ** 2 / M**2)
    )


def get_p_hat(deps_e_v, kap, p_hat_prev):
    """Compute non-linear pressure."""
    p_hat = p_hat_prev / (1.0 - (1.0 / kap) * deps_e_v)
    return jnp.nanmax(jnp.array([p_hat, 1e-12]))
    # return p_hat


def get_px_hat_mcc(px_hat_prev, cp, deps_p_v):
    """Compute non-linear pressure."""
    px_hat = px_hat_prev / (1.0 - (1.0 / cp) * deps_p_v)
    return jnp.nanmax(jnp.array([px_hat, 1e-12]))
    # return px_hat


def get_s(deps_e_d, G, s_prev):
    return 2.0 * G * deps_e_d + s_prev


def get_K(kap, p_hat):
    return (1.0 / kap) * (p_hat)


def get_G(nu, K):
    return (3.0 * (1.0 - 2.0 * nu)) / (2.0 * (1.0 + nu)) * K


def give_phi(p_hat, ln_Z, lam, ps):
    """Assume q =0, give reference solid volume fraction"""
    phi = ((p_hat / (1.0 + ps)) ** lam) * jnp.exp(-ln_Z)
    return phi


def give_p_hat(phi, ln_Z, lam):
    ln_v = jnp.log(1.0 / phi)

    p_hat = jnp.exp((ln_Z - ln_v) / lam)

    return p_hat


class MCC_NLNCL(Material):
    """

    ln_Z, ln_N at 1kPa

    Args:
        Material: _description_

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _description_
    """

    nu: TypeFloat
    M: TypeFloat
    R: TypeFloat
    lam: TypeFloat
    kap: TypeFloat
    ps: TypeFloat = 0.0

    ln_Z: TypeFloat = None
    ln_N: TypeFloat = None
    ln_vc0: TypeFloat = None

    _cp: TypeFloat  # ?

    eps_e_stack: Optional[TypeFloatMatrixPStack] = None
    px_hat_stack: Optional[TypeFloatScalarPStack] = None
    p_ref_stack: Optional[TypeFloatScalarPStack] = None

    dt: TypeFloat = eqx.field(static=True)
    dim: TypeInt = eqx.field(static=True)

    def __init__(
        self: Self,
        nu: TypeFloat,
        M: TypeFloat,
        R: TypeFloat,
        lam: TypeFloat,
        kap: TypeFloat,
        ln_Z: Optional[TypeFloat] = None,
        ln_N: Optional[TypeFloat] = None,
        ln_vc0: Optional[TypeFloat] = None,
        ps: Optional[TypeFloat] = None,
        **kwargs,
    ) -> Self:
        self.nu = nu

        self.M = M

        self.R = R

        # either ln_Z or ln_vc0 is required
        self.ln_Z = ln_Z
        self.ln_vc0 = ln_vc0

        if (ln_Z is None) and (ln_vc0 is None):
            raise ValueError("Either ln_Z or ln_vc0 is necessary")

        self.lam = lam

        self.kap = kap

        self.ps = ps
        self.ln_N = ln_N

        self._cp = lam - kap

        self.eps_e_stack = kwargs.get("eps_e_stack", None)

        self.px_hat_stack = kwargs.get("px_hat_stack", None)

        self.p_ref_stack = kwargs.get("p_ref_stack", None)

        self._setup_done = kwargs.get("_setup_done", False)

        self.dim = kwargs.get("dim", 3)
        self.dt = kwargs.get("dt", 0.001)

    @property
    def phi_c0(self):
        return 1.0 / jnp.exp(self.ln_vc0)

    @property
    def phi_N(self):
        return 1.0 / jnp.exp(self.ln_N)

    @property
    def phi_Z(self):
        return 1.0 / jnp.exp(self.ln_Z)

    def setup(
        self,
        p_ref: Optional[Union[TypeFloatScalarPStack, TypeFloat]] = None,
        density_ref: Optional[Union[TypeFloatScalarPStack, TypeFloat]] = None,
        rho_p: Optional[TypeFloat] = 1,
        num_points: TypeInt = 1,
        dt: TypeFloat = 0.001,
        dim: TypeInt = 3,
        **kwargs,
    ) -> Tuple[
        Self,
        Optional[TypeFloatScalarPStack | TypeFloat],
        Optional[TypeFloatScalarPStack | TypeFloat],
    ]:
        """Setup

        note p_ref is not  reference pressure at Ln_Z, ln_N but instead
        initial pressure of the model

        Get $\\ln Z$ given $\\ln v_{c0}$
        $$
        \\ln Z = \\ln N -\\lambda \\ln \\left(
        \exp \\left ( \\frac{ \\ln N - \\ln v_{c0}}{\\lambda}\\right)
        +1
        \\right)
        $$

        Args:
            p_ref: _description_. Defaults to None.
            density_ref: _description_. Defaults to None.
            rho_p: _description_. Defaults to 1.
            num_points: _description_. Defaults to 1.
            dt: _description_. Defaults to 0.001.
            dim: _description_. Defaults to 3.


        """
        # There are two ways to initialize via a reference pressure or reference density
        # these can be given as a scalar or array
        if self._setup_done:
            return self

        ln_Z = self.ln_Z

        ps = self.ps

        ln_vc0 = self.ln_vc0

        if (ln_vc0 is None) and (ln_Z is None):
            raise ValueError("Either ln_vc0 or ln_Z should be defined")

        if ln_Z is None:
            ln_Z = self.ln_N - self.lam * jnp.log(
                jnp.exp((self.ln_N - self.ln_vc0) / self.lam) + 1
            )

        if ps is None:
            ps = jnp.exp((self.ln_N - ln_Z) / self.lam) - 1

        if ln_vc0 is None:
            ln_vc0 = ln_Z + self.lam * jnp.log((1 + ps) / ps)

        if density_ref is not None:
            density_ref = jnp.array(density_ref).flatten()

            if density_ref.shape[0] != num_points:
                phi_ref = jnp.ones(num_points) * density_ref / rho_p
            else:
                phi_ref = density_ref / rho_p

            vmap_give_p_hat = partial(jax.vmap, in_axes=(0, None, None))(give_p_hat)

            p_hat = vmap_give_p_hat(phi_ref, ln_Z, self.lam)

            p_ref = p_hat - ps  # not reference pressure at ln_Z, but pressure at start

        elif p_ref is not None:
            p_ref = jnp.array(p_ref).flatten()

            if p_ref.shape[0] != num_points:
                p_ref = jnp.ones(num_points) * p_ref

            vmap_give_phi_ref = partial(jax.vmap, in_axes=(0, None, None, None))(
                give_phi
            )

            phi_ref = vmap_give_phi_ref(p_ref + ps, ln_Z, self.lam, ps)

            density_ref = phi_ref * rho_p

        else:
            raise ValueError("Reference density or pressure not given")

        eps_e_stack = jnp.zeros((num_points, 3, 3))
        # R = px_hat/p_hat = (px +ps)/ (p + ps)
        px_hat_stack = self.R * (p_ref + ps) - ps

        params = self.__dict__

        params.update(
            eps_e_stack=eps_e_stack,
            px_hat_stack=px_hat_stack,
            p_ref_stack=p_ref,
            dt=dt,
            dim=dim,
            ps=ps,
            ln_Z=ln_Z,
            ln_vc0=ln_vc0,
        )
        return self.__class__(**params), p_ref, density_ref

    def update_from_particles(
        self: Self, particles: Particles
    ) -> Tuple[Particles, Self]:
        """Update the material state and particle stresses for MPM solver."""

        deps_stack = get_sym_tensor_stack(particles.L_stack) * self.dt

        new_stress_stack, new_eps_e_stack, new_px_hat_stack = self.vmap_update_ip(
            deps_stack,
            self.eps_e_stack,
            particles.stress_stack,
            self.px_hat_stack,
            self.p_ref_stack,
        )

        new_self = eqx.tree_at(
            lambda state: (state.eps_e_stack, state.px_hat_stack),
            self,
            (new_eps_e_stack, new_px_hat_stack),
        )

        new_particles = eqx.tree_at(
            lambda state: (state.stress_stack),
            particles,
            (new_stress_stack),
        )

        return new_particles, new_self

    def update(self: Self, stress_prev_stack, F_stack, L_stack, phi_stack):
        deps_stack = get_sym_tensor_stack(L_stack) * self.dt

        new_stress_stack, new_eps_e_stack, new_px_hat_stack = self.vmap_update_ip(
            deps_stack,
            self.eps_e_stack,
            stress_prev_stack,
            self.px_hat_stack,
            self.p_ref_stack,
        )

        new_self = eqx.tree_at(
            lambda state: (state.eps_e_stack, state.px_hat_stack),
            self,
            (new_eps_e_stack, new_px_hat_stack),
        )

        return (new_stress_stack, new_self)

    @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0), out_axes=(0, 0, 0))
    def vmap_update_ip(
        self: Self, deps_next, eps_e_prev, stress_prev, px_hat_prev, p_ref
    ):
        p_prev = get_pressure(stress_prev)
        p_hat_prev = p_prev + self.ps
        # jax.debug.print("p_prev = {} p_hat_prev {}", p_prev, p_hat_prev)

        s_prev = get_dev_stress(stress_prev, pressure=p_prev)

        deps_e_v_tr = get_volumetric_strain(deps_next)

        p_hat_tr = get_p_hat(deps_e_v_tr, self.kap, p_hat_prev)

        deps_e_d_tr = get_dev_strain(deps_next, deps_e_v_tr)

        K_tr = get_K(self.kap, p_hat_tr)

        G_tr = get_G(self.nu, K_tr)

        s_tr = get_s(deps_e_d_tr, G_tr, s_prev)

        q_tr = get_q_vm(dev_stress=s_tr)

        yf = yield_function(p_hat_tr, px_hat_prev, q_tr, self.M, self.ps)

        def elastic_update():
            stress_next = s_tr - (p_hat_tr - self.ps) * jnp.eye(3)
            eps_e_tr = eps_e_prev + deps_next
            return stress_next, eps_e_tr, px_hat_prev

        def pull_to_ys():
            def residuals(sol, args):
                pmulti, deps_p_v = sol

                pmulti = jnp.nanmax(jnp.array([pmulti, 0.0]))
                # pmulti = jnp.nanmax(jnp.array([pmulti, 0.0]))
                p_hat_next = get_p_hat(deps_e_v_tr - deps_p_v, self.kap, p_hat_prev)

                K_next = get_K(self.kap, p_hat_next)

                G_next = get_G(self.nu, K_next)

                # factor = 1.0 / (1.0 + 6.0 * G_next * pmulti)
                factor = self.M**2 / (self.M**2 + 6.0 * G_next * pmulti)
                s_next = s_tr * factor

                q_next = q_tr * factor

                px_hat_next = get_px_hat_mcc(px_hat_prev, self._cp, deps_p_v)

                # deps_v_p_fr = pmulti * (2.0 * p_hat_next - px_hat_next) * self.M**2
                # deps_v_p_fr = pmulti * (2.0 * p_hat_next - px_hat_next)
                deps_v_p_fr = pmulti * (2.0 * (p_hat_next - self.ps) - px_hat_next)
                yf_next = yield_function(
                    p_hat_next, px_hat_next, q_next, self.M, self.ps
                )

                R = jnp.array([yf_next, deps_v_p_fr - deps_p_v])

                aux = (p_hat_next, s_next, px_hat_next, G_next, K_next)

                return R, aux

            def find_roots():
                """Find roots of the residuals function."""

                # avoiding non-finite values

                init_val = jnp.array([0.0, 0.0])

                solver = optx.Newton(
                    rtol=1e-10,
                    atol=1e-10,
                )
                sol = optx.root_find(
                    residuals, solver, init_val, throw=False, has_aux=True, max_steps=10
                )
                return sol.value

            pmulti_curr, deps_p_v_next = jax.lax.stop_gradient(find_roots())
            R, aux = residuals([pmulti_curr, deps_p_v_next], None)

            p_hat_next, s_next, px_hat_next, G_next, K_next = aux

            s_next = eqx.error_if(s_next, jnp.isnan(s_next).any(), "s_next is nan")

            p_next = p_hat_next - self.ps

            stress_next = s_next - (p_next) * jnp.eye(3)

            eps_e_v_next = (p_next - p_ref) / K_next

            eps_e_d_next = s_next / (2.0 * G_next)

            eps_e_next = eps_e_d_next - (1.0 / 3) * eps_e_v_next * jnp.eye(3)

            return stress_next, eps_e_next, px_hat_next

        stress_next, eps_e_next, px_hat_next = jax.lax.cond(
            (p_hat_tr >= 0.0),
            # True,
            lambda: jax.lax.cond(yf > 0.0, pull_to_ys, elastic_update),
            lambda: (0.0 * jnp.eye(3), jnp.zeros((3, 3)), 0.0),
        )

        return stress_next, eps_e_next, px_hat_next
