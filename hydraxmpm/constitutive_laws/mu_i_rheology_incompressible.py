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
    get_scalar_shear_strain,
    get_sym_tensor,
)
from .constitutive_law import ConstitutiveLaw


def get_mu_I(I, mu_s, mu_d, I0):
    return mu_s + (mu_d - mu_s) * (1 / (1 + I0 / I))


def get_mu_I_regularized_exp(I, mu_s, mu_d, I0, pen, dgamma_dt):
    s = 1.0 / jnp.sqrt(dgamma_dt**2 + pen**2)
    return mu_s * s + (mu_d - mu_s) * (1.0 / (1.0 + I0 / I))


def get_I_phi(phi, phi_c, I_phi):
    return -I_phi * jnp.log(phi / phi_c)


def get_pressure(dgammadt, I, d, rho_p):
    return rho_p * ((dgammadt * d) / I) ** 2


def give_p(K, rho, rho_0):
    return K * (rho / rho_0 - 1.0)


def give_rho(K, rho_0, p):
    return rho_0 * ((p / K) + 1)


class MuI_incompressible(ConstitutiveLaw):
    mu_s: TypeFloat
    mu_d: TypeFloat
    I_0: TypeFloat
    K: TypeFloat

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
        K: TypeFloat = 1.0,
        **kwargs,
    ) -> Self:
        self.mu_s = mu_s

        self.mu_d = mu_d

        self.I_0 = I_0

        self.K = K

        # init d, dim, rho_p, _setup_done
        super().__init__(**kwargs)

    def init_state(self: Self, material_points: MaterialPoints):
        # There are two ways to initialize via a reference pressure or reference density
        # these can be given as a scalar or array

        p_0 = self.p_0
        if p_0 is None:
            p_0 = material_points.p_stack

        rho_0 = self.rho_0

        rho = self.rho_0

        if self.init_by_density:
            if eqx.is_array(rho_0):
                vmap_give_p_ref = partial(
                    jax.vmap,
                    in_axes=(None, 0, None),
                )(give_p)
            else:
                vmap_give_p_ref = give_p

            p_0 = vmap_give_p_ref(self.K, rho_0, rho_0)
        else:
            p_0_stack = p_0
            if not eqx.is_array(p_0_stack):
                p_0_stack = p_0_stack * jnp.ones(material_points.num_points)

            vmap_give_rho_ref = partial(
                jax.vmap,
                in_axes=(None, None, 0),
            )(give_rho)

            rho = vmap_give_rho_ref(self.K, self.rho_0, p_0_stack)

        rho_rho_0_stack = rho / self.rho_0
        # print(rho_rho_0_stack)
        vmap_update_ip = jax.vmap(fun=self.update_ip, in_axes=0)

        new_stress_stack = vmap_update_ip(
            material_points.stress_stack,
            material_points.F_stack,
            material_points.L_stack,
            rho_rho_0_stack,
        )
        material_points = material_points.replace(stress_stack=new_stress_stack)
        # material_points = material_points.init_stress_from_p_0(p_0)
        # if there is pressure, then density is not on reference density
        material_points = material_points.init_mass_from_rho_0(rho)

        W_stack = None
        if self.approx_strain_energy_density:
            W_stack = jnp.zeros(material_points.num_points)

        P_stack = None
        if self.approx_stress_power:
            P_stack = jnp.zeros(material_points.num_points)

        params = self.__dict__
        params.update(
            rho_0=rho_0,
            p_0=p_0,
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

        rho_rho_0_stack = material_points.rho_stack / self.rho_0
        # rho_stack = material_points.rho_stack

        deps_dt_stack = material_points.depsdt_stack

        vmap_update_ip = jax.vmap(fun=self.update_ip, in_axes=0)

        new_stress_stack = vmap_update_ip(
            material_points.stress_stack,
            material_points.F_stack,
            deps_dt_stack,
            rho_rho_0_stack,
        )

        new_material_points = eqx.tree_at(
            lambda state: (state.stress_stack),
            material_points,
            (new_stress_stack),
        )

        new_self = self.post_update(new_stress_stack, deps_dt_stack, dt)

        return new_material_points, new_self

    def update_ip(
        self: Self,
        stress_prev,
        F,
        deps_dt,
        rho_rho_0,
    ):
        deps_dev_dt = get_dev_strain(deps_dt)

        if self.error_check:
            rho_rho_0 = eqx.error_if(
                rho_rho_0, jnp.isnan(rho_rho_0).any(), "rho_rho_0 is nan"
            )

            deps_dev_dt = eqx.error_if(
                deps_dev_dt, jnp.isnan(deps_dev_dt).any(), "deps_dev_dt is nan"
            )

        dgamma_dt = get_scalar_shear_strain(deps_dt)
        # dgamma_dt = jnp.nanmax(jnp.array([dgamma_dt, 1.0e-6]))

        if self.error_check:
            dgamma_dt = eqx.error_if(
                dgamma_dt, jnp.isnan(dgamma_dt).any(), "dgamma_dt is nan"
            )

        # stress free condition...
        # rho_rho_0 = jnp.nanmax(jnp.array([rho_rho_0 - 1.0, 1e-6])) + 1.0

        p = self.K * (rho_rho_0 - 1.0)

        if self.error_check:
            p = eqx.error_if(p, jnp.isnan(p).any(), "p is nan")

        p = jnp.nanmax(jnp.array([p, 1e-12]))

        def stress_update(_):
            # correction for viscosity diverges
            # r = 1e-10
            r = 0.001
            # r = 0.01
            # eq (12) https://www.sciencedirect.com/science/article/pii/S0021999118307290

            delta_mu = self.mu_d - self.mu_s

            eta_d = (p * delta_mu * self.d) / (
                self.I_0 * jnp.sqrt(p / self.rho_p) + self.d * dgamma_dt
            )
            # eta_d = jnp.nanmax(jnp.array([eta_d, 0.0]))

            eta_s = (p * self.mu_s) / jnp.sqrt(dgamma_dt * dgamma_dt + r * r)
            # eta_s = jnp.nanmax(jnp.array([eta_s, 0.0]))
            if self.error_check:
                eta_s = eqx.error_if(eta_s, jnp.isnan(eta_s).any(), "eta_s is nan")
                eta_d = eqx.error_if(eta_d, jnp.isnan(eta_d).any(), "eta_d is nan")

            eta = eta_s + eta_d
            stress_next = -p * jnp.eye(3) + eta * deps_dev_dt

            if self.error_check:
                stress_next = eqx.error_if(
                    stress_next, jnp.isnan(stress_next).any(), "stress_next is nan"
                )

            return stress_next

        return stress_update(None)
        # def stress_free_assump(_):
        #     # adding a very small number such that inertial numbers are not infinite
        #     return jnp.eye(3) * -1e-12

        # return jax.lax.cond(
        #     (rho_rho_0 - 1.0 > 1e-12), stress_update, stress_free_assump, operand=False
        # )
