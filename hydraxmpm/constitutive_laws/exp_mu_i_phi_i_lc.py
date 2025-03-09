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
    get_pressure,
)
from .constitutive_law import ConstitutiveLaw

import optimistix as optx


def give_phi(p, I, Ic, phi_c, K):
    return phi_c * (1 - I / Ic) * (1 + p / K)


class ExpMuIPhiILC(ConstitutiveLaw):
    mu_s: TypeFloat
    mu_d: TypeFloat
    I_0: TypeFloat
    K: TypeFloat
    Ic: TypeFloat
    phi_c: TypeFloat
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
        Ic: TypeFloat,
        phi_c: TypeFloat,
        K: TypeFloat = 1.0,
        init_by_density: bool = True,
        **kwargs,
    ) -> Self:
        self.mu_s = mu_s

        self.mu_d = mu_d

        self.I_0 = I_0

        self.K = K

        self.Ic = Ic

        self.phi_c = phi_c

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

        rho = self.rho_0
        if self.init_by_density:
            raise ValueError("Not supported")
        else:
            p_0_stack = p_0

            dgamma_dt_stack = material_points.dgammadt_stack
            if not eqx.is_array(p_0_stack):
                p_0_stack = p_0_stack * jnp.ones(material_points.num_points)
            I_stack = get_inertial_number_stack(
                p_0_stack, dgamma_dt_stack, self.d, self.rho_p
            )

            phi_stack = give_phi(p_0, I_stack, self.Ic, self.phi_c, self.K)
            jax.debug.print("phi_stack {}", phi_stack)
            rho_0 = phi_stack * self.rho_p

        vmap_update_ip = jax.vmap(fun=self.update_ip, in_axes=0)

        phi_stack = rho_0 / self.rho_p

        new_stress_stack = vmap_update_ip(
            material_points.stress_stack,
            material_points.F_stack,
            material_points.L_stack,
            phi_stack,
        )
        material_points = material_points.replace(stress_stack=new_stress_stack)
        # material_points = material_points.init_stress_from_p_0(p_0)
        # if there is pressure, then density is not on reference density
        material_points = material_points.init_mass_from_rho_0(rho_0)

        params = self.__dict__
        params.update(rho_0=rho_0, p_0=p_0)
        return self.__class__(**params), material_points

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

        if self.error_check:
            phi = eqx.error_if(phi, jnp.isnan(phi).any(), "phi is nan")

            deps_dev_dt = eqx.error_if(
                deps_dev_dt, jnp.isnan(deps_dev_dt).any(), "deps_dev_dt is nan"
            )

        dgamma_dt = get_scalar_shear_strain(deps_dt)

        if self.error_check:
            dgamma_dt = eqx.error_if(
                dgamma_dt, jnp.isnan(dgamma_dt).any(), "dgamma_dt is nan"
            )

        # rho_rho_0 = jnp.nanmax(jnp.array([rho_rho_0 - 1.0, 1e-6])) + 1.0

        # regularize p and dgamma_dt to avoid division by zero
        # p = jnp.nanmax(jnp.array([self.K * (rho_rho_0 - 1.0), 1.0e-12]))
        # p = self.K * (rho_rho_0 - 1.0)

        def residuals(sol, args):
            I = get_inertial_number(sol, dgamma_dt, self.d, self.rho_p)

            phi_guess = give_phi(sol, I, self.Ic, self.phi_c, self.K)

            aux = I
            return phi - phi_guess, aux

        def find_roots():
            solver = optx.Newton(
                rtol=1e-10,
                atol=1e-10,
            )

            sol = optx.root_find(
                residuals,
                solver,
                get_pressure(stress_prev),
                throw=False,
                has_aux=True,
                max_steps=10,
                options=dict(lower=1.0),
            )
            return sol.value

        p = jax.lax.stop_gradient(find_roots())
        R, I = residuals(p, None)

        if self.error_check:
            p = eqx.error_if(p, jnp.isnan(p).any(), "p is nan")

        def stress_update(_):
            # I = get_inertial_number(p, dgamma_dt, self.d, self.rho_p)

            # I = jnp.nanmax(jnp.array([I, 1e-10]))

            # correction for viscosity diverges
            # r = 1e-10
            r = 0.001
            # eq (12) https://www.sciencedirect.com/science/article/pii/S0021999118307290

            # dgamma_dt = jnp.nanmax(jnp.array([dgamma_dt, 1.0e-12]))

            delta_mu = self.mu_d - self.mu_s

            eta_d = (p * delta_mu * self.d) / (
                self.I_0 * jnp.sqrt(p / self.rho_p) + self.d * dgamma_dt
            )
            eta_d = jnp.nanmax(jnp.array([eta_d, 0.0]))

            eta_s = (p * self.mu_s) / jnp.sqrt(dgamma_dt * dgamma_dt + r * r)

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
