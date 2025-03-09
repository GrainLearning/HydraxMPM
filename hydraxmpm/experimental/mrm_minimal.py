"""Implementation, state and functions for isotropic linear elastic material."""

from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from typing_extensions import Optional, Self, Union

from ..common.types import TypeFloat, TypeFloatScalarPStack, TypeInt
from ..particles.particles import Particles  # type: ignore
from ..utils.math_helpers import (
    get_dev_strain,
    get_inertial_number,
    get_scalar_shear_strain,
    get_sym_tensor,
)
from .material import Material


def get_mu_I(I, mu_s, mu_d, I0):
    return mu_s + (mu_d - mu_s) * (1 / (1 + I0 / I))


def get_mu_I_regularized_exp(I, mu_s, mu_d, I0, pen, dgamma_dt):
    s = 1.0 / jnp.sqrt(dgamma_dt**2 + pen**2)
    return mu_s * s + (mu_d - mu_s) * (1.0 / (1.0 + I0 / I))


def get_I_phi(phi, phi_c, I_phi):
    return -I_phi * jnp.log(phi / phi_c)


def get_pressure(dgammadt, I, d, rho_p):
    return rho_p * ((dgammadt * d) / I) ** 2


class MRMMinimal(Material):
    mu_s: TypeFloat
    mu_d: TypeFloat
    I_0: TypeFloat
    d: TypeFloat
    phi_c: TypeFloat
    ps: TypeFloat
    lam: TypeFloat
    I_phi_c: TypeFloat

    dim: TypeInt = eqx.field(static=True)
    rho_p: TypeFloat = eqx.field(default=1.0)
    """
    (nearly) incompressible mu I

    Tensorial form similar to
    Jop, Pierre, Yoël Forterre, and Olivier Pouliquen. "A constitutive law for dense granular flows." Nature 441.7094 (2006): 727-730.

    mu I regularized by
    Franci, Alessandro, and Massimiliano Cremonesi. "3D regularized μ (I)-rheology for granular flows simulation." Journal of Computational Physics 378 (2019): 257-277.

    Pressure term by

    Salehizadeh, A. M., and A. R. Shafiei. "Modeling of granular column collapses with μ (I) rheology using smoothed particle hydrodynamic method." Granular Matter 21.2 (2019): 32.

    """

    def __init__(
        self: Self,
        mu_s: TypeFloat,
        mu_d: TypeFloat,
        phi_c: TypeFloat,
        I_0: TypeFloat,
        I_phi_c: TypeFloat,
        d: TypeFloat,
        lam: TypeFloat,
        ps: TypeFloat,
        **kwargs,
    ) -> Self:
        self.mu_s = mu_s

        self.mu_d = mu_d

        self.I_0 = I_0

        self.d = d

        self.ps = ps
        self.lam = lam
        self.phi_c = phi_c
        self.I_phi_c = I_phi_c

        self.dim = kwargs.get("dim", 3)

        self.rho_p = kwargs.get("rho_p", 1.0)

        self._setup_done = kwargs.get("_setup_done", False)

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
        params = self.__dict__

        params.update(dt=dt, dim=dim, rho_p=rho_p)
        return self.__class__(**params), p_ref, density_ref

    def update_from_particles(
        self: Self, particles: Particles
    ) -> Tuple[Particles, Self]:
        """Update the material state and particle stresses for MPM solver."""

        phi_stack = particles.get_solid_volume_fraction_stack()

        # jax.debug.print("phi_stack {}", phi_stack)

        vmap_update_ip = jax.vmap(fun=self.update_ip, in_axes=0)

        new_stress_stack = vmap_update_ip(
            particles.stress_stack,
            particles.F_stack,
            particles.L_stack,
            phi_stack,
        )
        # jax.debug.print("{}", new_stress_stack)

        new_particles = eqx.tree_at(
            lambda state: (state.stress_stack),
            particles,
            (new_stress_stack),
        )

        return new_particles, self

    def update_ip(
        self: Self,
        stress_prev,
        F,
        L,
        phi,
    ):
        deps_dt = get_sym_tensor(L)

        deps_dev_dt = get_dev_strain(deps_dt)

        dgamma_dt = get_scalar_shear_strain(deps_dt)

        dgamma_dt = jnp.nanmax(jnp.array([dgamma_dt, 1e-12]))

        def get_p(sol, args):
            p_guess = jnp.nanmax(jnp.array([sol, 1e-12]))

            lhs = phi / self.phi_c

            I = get_inertial_number(p_guess, dgamma_dt, self.d, self.rho_p)

            I = jnp.nanmax(jnp.array([I, 1e-12]))

            inertial_part = jnp.exp(-I / self.I_phi_c)

            solid_part = (1.0 + p_guess / self.ps) ** (self.lam)

            rhs = inertial_part * solid_part

            aux = I
            # jax.debug.print("I {} p_guess {} res {}", I, p_guess, lhs - rhs)
            # d =

            return lhs - rhs, aux

        def find_root():
            p_init = 1e-12
            # jax.debug.print(" phi {} phi_c {}", phi, self.phi_c)
            p_init = jax.lax.cond(
                phi >= self.phi_c,
                lambda phi: self.ps * ((phi / self.phi_c) ** (1.0 / self.lam) - 1),
                lambda phi: ((dgamma_dt**2) * (self.d**2) * self.rho_p)
                / (self.I_phi_c * jnp.log(self.phi_c / phi)) ** 2,
                phi,
            )

            solver = optx.Newton(rtol=1e-10, atol=1e-10)
            sol = optx.minimise(
                get_p, solver, p_init, throw=False, has_aux=True, max_steps=20
            )
            return sol.value

        p = jax.lax.stop_gradient(find_root())

        R, aux = get_p(p, None)

        I = aux

        mu_I = self.mu_s + (self.mu_d - self.mu_s) / (1.0 + self.I_0 / I)
        eta = p * (mu_I / dgamma_dt)
        # jax.debug.print("mu_I {} I {} p {} eta {}", mu_I, I, p, eta)

        stress_next = -p * jnp.eye(3) + eta * deps_dev_dt
        # stress_next = -p * jnp.eye(3)

        return stress_next

        # alpha = 0.000001
        # eta_E_s = p * self.mu_s / jnp.sqrt(dgamma_dt * dgamma_dt + alpha * alpha)

        #

        # eta_delta = p * mu_I_delta / dgamma_dt

        # eta = eta_E_s + eta_delta

        # stress_next = -p * jnp.eye(3) + eta * deps_dev_dt

        # return stress_next
