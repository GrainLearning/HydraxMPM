"""Implementation, state and functions for isotropic linear elastic material."""

from typing import Tuple
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from typing_extensions import Optional, Self, Union

from ..common.types import TypeFloat, TypeFloatScalarPStack, TypeInt
from ..materials.material import Material
from ..particles.particles import Particles
from ..utils.math_helpers import (
    get_dev_strain,
    get_inertial_number,
    get_scalar_shear_strain,
    get_sym_tensor,
)


def get_mu_I(I, mu_s, mu_d, I0):
    return mu_s + (mu_d - mu_s) * (1 / (1 + I0 / I))


def get_mu_I_regularized_exp(I, mu_s, mu_d, I0, pen, dgamma_dt):
    s = 1.0 / jnp.sqrt(dgamma_dt**2 + pen**2)
    return mu_s * s + (mu_d - mu_s) * (1.0 / (1.0 + I0 / I))


def get_I_phi(phi, phi_c, I_phi):
    return -I_phi * jnp.log(phi / phi_c)


def get_pressure(dgammadt, I, d, rho_p):
    return rho_p * ((dgammadt * d) / I) ** 2


def give_p_hat(phi, ln_Z, lam):
    ln_v = jnp.log(1.0 / phi)

    p_hat = jnp.exp((ln_Z - ln_v) / lam)

    return p_hat


def give_phi(p_hat, ln_Z, lam, ps):
    """Assume q =0, give reference solid volume fraction"""
    phi = ((p_hat / (1.0 + ps)) ** lam) * jnp.exp(-ln_Z)
    return phi


class MuI_NLNCL(Material):
    mu_s: TypeFloat
    mu_d: TypeFloat
    I_0: TypeFloat
    d: TypeFloat

    ln_Z: TypeFloat = None
    ln_N: TypeFloat = None
    ln_vc0: TypeFloat = None
    phi_c0: TypeFloat = None

    ps: TypeFloat = 0.0
    lam: TypeFloat

    dim: TypeInt = eqx.field(static=True)
    rho_p: TypeFloat = eqx.field(default=1.0)

    def __init__(
        self: Self,
        mu_s: TypeFloat,
        mu_d: TypeFloat,
        I_0: TypeFloat,
        d: TypeFloat,
        lam: TypeFloat,
        ln_Z: Optional[TypeFloat] = None,
        ln_N: Optional[TypeFloat] = None,
        ln_vc0: Optional[TypeFloat] = None,
        ps: Optional[TypeFloat] = None,
        **kwargs,
    ) -> Self:
        self.mu_s = mu_s

        self.mu_d = mu_d

        self.I_0 = I_0

        self.d = d

        # either ln_Z or ln_vc0 is required
        self.ln_Z = ln_Z
        self.ln_vc0 = ln_vc0

        if (ln_Z is None) and (ln_vc0 is None):
            raise ValueError("Either ln_Z or ln_vc0 is necessary")

        self.ps = ps
        self.ln_N = ln_N
        self.lam = lam

        if self.ln_vc0:
            self.phi_c0 = 1.0 / jnp.exp(self.ln_vc0)

        self.dim = kwargs.get("dim", 3)

        self.rho_p = kwargs.get("rho_p", 1.0)

        self._setup_done = kwargs.get("_setup_done", False)

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
        num_points: Optional[TypeInt] = 1,
        dt: TypeFloat = 0.001,
        dim: TypeInt = 3,
        **kwargs,
    ) -> Tuple[
        Self,
        Optional[TypeFloatScalarPStack | TypeFloat],
        Optional[TypeFloatScalarPStack | TypeFloat],
    ]:
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

        params = self.__dict__

        params.update(
            dt=dt,
            dim=dim,
            rho_p=rho_p,
            ps=ps,
            ln_Z=ln_Z,
            ln_vc0=ln_vc0,
        )
        return self.__class__(**params), p_ref, density_ref

    def update_from_particles(
        self: Self, particles: Particles
    ) -> Tuple[Particles, Self]:
        """Update the material state and particle stresses for MPM solver."""

        # density_stack = particles.mass_stack / particles.volume_stack
        # density0_stack = particles.mass_stack / particles.volume0_stack

        # phi_ref_phi_stack = density_stack / density0_stack

        phi_stack = particles.phi_stack

        vmap_update_ip = jax.vmap(fun=self.update_ip, in_axes=0)

        new_stress_stack = vmap_update_ip(
            particles.stress_stack,
            particles.F_stack,
            particles.L_stack,
            phi_stack,
        )

        new_particles = eqx.tree_at(
            lambda state: (state.stress_stack),
            particles,
            (new_stress_stack),
        )

        return new_particles, self

    def update(self: Self, stress_prev_stack, F_stack, L_stack, phi_stack):
        vmap_update_ip = jax.vmap(fun=self.update_ip, in_axes=0)

        new_stress_stack = vmap_update_ip(
            stress_prev_stack,  # unused
            F_stack,  # unused
            L_stack,
            phi_stack,
        )

        return (new_stress_stack, self)

    def update_ip(
        self: Self,
        stress_prev,
        F,
        L,
        phi,
    ):
        deps_dt = get_sym_tensor(L)

        p = self.ps * ((phi / self.phi_c0) ** (1 / self.lam) - 1.0)
        # p = jnp.maximum(self.K * (phi - 1.0), 1.0e-12)
        p = jnp.nanmax(jnp.array([p, 1.0e-12]))

        # deps_dev_dt = get_dev_strain(deps_dt, dim=self.dim)

        # dgamma_dt = get_scalar_shear_strain(deps_dt, dim=self.dim)

        deps_dev_dt = get_dev_strain(deps_dt)

        dgamma_dt = get_scalar_shear_strain(deps_dt)

        dgamma_dt = jnp.maximum(dgamma_dt, 1.0e-12)

        I = get_inertial_number(p, dgamma_dt, self.d, self.rho_p)

        # regularize I
        I = jnp.maximum(I, 1e-9)

        alpha = 0.000001
        eta_E_s = p * self.mu_s / jnp.sqrt(dgamma_dt * dgamma_dt + alpha * alpha)

        mu_I_delta = (self.mu_d - self.mu_s) / (1.0 + self.I_0 / I)

        eta_delta = p * mu_I_delta / dgamma_dt

        eta = eta_E_s + eta_delta

        stress_next = -p * jnp.eye(3) + eta * deps_dev_dt

        return stress_next
