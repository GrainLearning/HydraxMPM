"""Implementation, state and functions for isotropic linear elastic material."""

from functools import partial
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from typing_extensions import Optional, Self, Union

from ..common.types import TypeFloat, TypeFloatScalarPStack, TypeInt
from ..material_points.material_points import MaterialPoints
from ..utils.math_helpers import (
    get_dev_strain,
    get_inertial_number,
    get_inertial_number_stack,
    get_scalar_shear_strain,
    get_sym_tensor,
    get_pressure_stack,
)
from .constitutive_law import ConstitutiveLaw


def get_pressure(dgammadt, I, d, rho_p):
    return rho_p * ((dgammadt * d) / I) ** 2


def give_p(phi, dgamma_dt, phi_c, a, d, rho_p):
    return rho_p * ((d * a * dgamma_dt) / (phi_c - phi)) ** 2


def give_phi_I(phi_c, a, I):
    return phi_c - a * I


class ExpMuICSL(ConstitutiveLaw):
    mu_s: TypeFloat
    mu_d: TypeFloat
    I_0: TypeFloat
    ln_Gamma: TypeFloat
    lam: TypeFloat
    init_by_density: bool = True

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
        I_0: TypeFloat,
        ln_Gamma: TypeFloat = 1.0,
        lam: TypeFloat = 1.0,
        init_by_density: bool = True,
        **kwargs,
    ) -> Self:
        self.mu_s = mu_s

        self.mu_d = mu_d

        self.I_0 = I_0

        self.ln_Gamma = ln_Gamma

        self.lam = lam

        self.init_by_density = init_by_density

        # init d, dim, rho_p, _setup_done
        super().__init__(**kwargs)

    def init_state(self: Self, material_points: MaterialPoints):
        # There are two ways to initialize via a reference pressure or reference density
        # these can be given as a scalar or array

        p_0 = self.p_0
        if p_0 is None:
            p_0 = material_points.p_stack

        rho_0 = self.rho_0

        if self.init_by_density:
            raise ValueError("Not supported")
        else:
            p_0_stack = p_0

            dgamma_dt_stack = material_points.dgammadt_stack
            if not eqx.is_array(p_0_stack):
                p_0_stack = p_0_stack * jnp.ones(material_points.num_points)
            # I_stack = get_inertial_number_stack(
            #     p_0_stack, dgamma_dt_stack, self.d, self.rho_p
            # )

            ln_specific_volume_stack = self.ln_Gamma - self.lam * jnp.log(p_0_stack)
            # jax.debug.print(
            #     # f"{self.lam=} {ln_specific_volume_stack=} {p_0_stack=} { self.ln_Gamma=}",
            # )
            phi_stack = 1.0 / jnp.exp(ln_specific_volume_stack)
            # phi_stack = self.phi_c - self.a * I_stack

            rho_0 = phi_stack * self.rho_p
            # print("dgamma", dgamma_dt_stack, p_0_stack, p_0_stack)
        material_points = material_points.init_mass_from_rho_0(rho_0)
        phi_stack = rho_0 / self.rho_p

        vmap_update_ip = jax.vmap(fun=self.update_ip, in_axes=0)

        new_stress_stack = vmap_update_ip(
            material_points.stress_stack,
            material_points.F_stack,
            material_points.L_stack,
            phi_stack,
        )
        pred = get_pressure_stack(new_stress_stack)
        # print("predicted pressure ", pred, ln_specific_volume_stack)
        material_points = material_points.replace(stress_stack=new_stress_stack)

        params = self.__dict__
        params.update(rho_0=rho_0, p_0=p_0)
        return self.__class__(**params), material_points
        # return self, material_points

    def update(
        self: Self,
        material_points: MaterialPoints,
        dt: TypeFloat,
        dim: Optional[TypeInt] = 3,
    ) -> Tuple[MaterialPoints, Self]:
        """Update the material state and particle stresses for MPM solver."""

        # rho_rho_0_stack = material_points.rho_stack / self.rho_0
        phi_stack = material_points.phi_stack(self.rho_p)

        vmap_update_ip = jax.vmap(fun=self.update_ip, in_axes=0)

        new_stress_stack = vmap_update_ip(
            material_points.stress_stack,
            material_points.F_stack,
            material_points.L_stack,
            phi_stack,
        )

        new_material_points = eqx.tree_at(
            lambda state: (state.stress_stack),
            material_points,
            (new_stress_stack),
        )
        pred = get_pressure_stack(new_stress_stack)
        # jax.debug.print("oredicted pressure {}", pred)
        return new_material_points, self

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

        # regularize p and dgamma_dt to avoid division by zero

        # p = jnp.nanmax(jnp.array([self.K * (rho_rho_0 - 1.0), 1.0e-12]))

        ln_specific_volume = jnp.log(1.0 / phi)

        p = jnp.exp((self.ln_Gamma - ln_specific_volume) / self.lam)

        p = jnp.nanmax(jnp.array([p, 1.0e-12]))
        # p = give_p(phi, dgamma_dt, self.phi_c, self.a, self.d, self.rho_p)

        dgamma_dt = jnp.nanmax(jnp.array([dgamma_dt, 1.0e-12]))

        I = get_inertial_number(p, dgamma_dt, self.d, self.rho_p)

        # this part is needed to regularize instabilithy at low-I
        alpha = 0.000001
        eta_E_s = p * self.mu_s / jnp.sqrt(dgamma_dt * dgamma_dt + alpha * alpha)

        mu_I_delta = (self.mu_d - self.mu_s) / (1.0 + self.I_0 / I)

        eta_delta = p * mu_I_delta / dgamma_dt

        eta = eta_E_s + eta_delta

        stress_next = -p * jnp.eye(3) + eta * deps_dev_dt

        return stress_next
