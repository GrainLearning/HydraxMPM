"""Implementation, non-associated Drucker-Prager model with isotropic linear elasticity,
and linear hardening.

[1] de Souza Neto, E.A., Peric, D. and Owen, D.R., 2011. Computational methods for
plasticity.
"""

from functools import partial
from typing import Tuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from typing_extensions import Self, Union

from ..config.ip_config import IPConfig
from ..config.mpm_config import MPMConfig
from ..particles.particles import Particles
from ..utils.math_helpers import (
    get_dev_strain,
    get_J2,
    get_sym_tensor_stack,
    get_volumetric_strain,
)
from .common import (
    get_bulk_modulus,
    get_lin_elas_dev,
    get_lin_elas_vol,
    get_shear_modulus,
)
from .material import Material


def yield_function(
    sqrt_J2_tr: jnp.float32,
    p: jnp.float32,
    mu_1: jnp.float32,
    mu_2: jnp.float32,
    c: jnp.float32,
):
    return sqrt_J2_tr - mu_1 * p - mu_2 * c


class DruckerPragerEP(Material):
    r"""Non-associated Drucker-Prager model.

    The Drucker-Prager model is a smooth approximation to the Mohr-Coulomb model.

    This formulation is in small strain and elastic law is  isotropic linear elasticity.

    The implementation follows [1] with the exception that pressure and
    volumetric strain are positive in compression.

    [1] de Souza Neto, E.A., Peric, D. and Owen, D.R., 2011. Computational methods for
    plasticity.


    DP yield function
        q + M*p +M2*c

    plastic potential
        q + M_hat*p


    deps_v = pmulti*M_hat

    deps_dev = s/sqrt(2)J_2

    Attributes:
        E: Young's modulus.
        nu: Poisson's ratio.
        G: Shear modulus.
        K: Bulk modulus.
        M: Mohr-Coulomb friction parameter.
        M_hat: Mohr-Coulomb dilatancy parameter.
        M2: Mohr-Coulomb cohesion parameter.
        c0: Initial cohesion parameter.
        eps_acc_stack: Accumulated plastic strain for linear hardening
        eps_e_stack: Elastic strain tensor.
        H: Hardening modulus
    """

    E: jnp.float32
    nu: jnp.float32
    G: jnp.float32
    K: jnp.float32

    eps_e_stack: chex.Array
    p_ref_stack: chex.Array

    mu_1: jnp.float32
    mu_2: jnp.float32
    c0: jnp.float32

    mu_1_hat: jnp.float32

    eps_p_acc_stack: chex.Array
    H: jnp.float32

    rho_p: jnp.float32

    def __init__(
        self: Self,
        config: Union[MPMConfig, IPConfig],
        E: jnp.float32,
        nu: jnp.float32,
        mu_1: jnp.float32,
        mu_2: jnp.float32 = 0.0,
        c0: jnp.float32 = 0.0,
        p_ref_stack: chex.Array = 1.0,
        mu_1_hat: jnp.float32 = 0.0,
        H: jnp.float32 = 0.0,
        rho_p: jnp.float32 = 0.0,
    ) -> Self:
        """Create a non-associated Drucker-Prager material model."""

        self.K = get_bulk_modulus(E, nu)
        self.G = get_shear_modulus(E, nu)
        self.E = E
        self.nu = nu

        self.eps_e_stack = jnp.zeros((config.num_points, 3, 3))

        self.eps_p_acc_stack = jnp.zeros(config.num_points)

        self.p_ref_stack = p_ref_stack
        self.c0 = c0
        self.config = config

        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.H = H

        self.mu_1_hat = mu_1_hat

        self.rho_p = rho_p
        self.config = config

    def update_from_particles(
        self: Self, particles: Particles
    ) -> Tuple[Particles, Self]:
        """Update the material state and particle stresses for MPM solver."""

        deps_stack = get_sym_tensor_stack(particles.L_stack) * self.config.dt

        new_stress_stack, new_eps_e_stack, new_eps_p_acc_stack = (
            self.vmap_update_stress(
                deps_stack, self.p_ref_stack, self.eps_e_stack, self.eps_p_acc_stack
            )
        )

        new_particles = eqx.tree_at(
            lambda state: (state.stress_stack),
            particles,
            (new_stress_stack),
        )
        new_self = eqx.tree_at(
            lambda state: (state.eps_e_stack, state.eps_p_acc_stack),
            self,
            (new_eps_e_stack, new_eps_p_acc_stack),
        )

        return new_particles, new_self

    def update(
        self: Self,
        stress_prev_stack: chex.Array,
        F_stack: chex.Array,
        L_stack: chex.Array,
        phi_stack: chex.Array,
    ) -> Tuple[chex.Array, Self]:
        """Update stress using the Drucker-Prager model"""

        deps_stack = get_sym_tensor_stack(L_stack) * self.config.dt

        new_stress_stack, new_eps_e_stack, new_eps_p_acc_stack = (
            self.vmap_update_stress(
                deps_stack, self.p_ref_stack, self.eps_e_stack, self.eps_p_acc_stack
            )
        )

        new_self = eqx.tree_at(
            lambda state: (state.eps_e_stack, state.eps_p_acc_stack),
            self,
            (new_eps_e_stack, new_eps_p_acc_stack),
        )

        return (new_stress_stack, new_self)

    @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0), out_axes=(0, 0, 0))
    def vmap_update_stress(self, deps_next, p_ref, eps_e_prev, eps_p_acc_prev):
        eps_e_tr = eps_e_prev + deps_next

        eps_e_v_tr = get_volumetric_strain(eps_e_tr)

        eps_e_d_tr = get_dev_strain(eps_e_tr, eps_e_v_tr)

        s_tr = get_lin_elas_dev(eps_e_d_tr, self.G)

        p_tr = get_lin_elas_vol(eps_e_v_tr, self.K)

        p_tr = p_tr + p_ref

        # linear hardening
        c = self.c0 + self.H * eps_p_acc_prev

        sqrt_J2_tr = jnp.sqrt(get_J2(dev_stress=s_tr))

        yf = yield_function(sqrt_J2_tr, p_tr, self.mu_1, self.mu_2, c=c)

        is_ep = yf > 0.0

        def elastic_update() -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
            """If yield function is negative, return elastic solution."""
            stress = s_tr - p_tr * jnp.eye(3)
            return stress, eps_e_tr, eps_p_acc_prev

        def pull_to_ys() -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
            """If yield function is negative, return elastic solution."""

            def residuals_cone(sol, args):
                """Reduced system for non-associated flow rule."""
                pmulti = sol

                # solve non associated flow rules
                # volumetric plastic strain increment
                # deps_p_v = -pmulti * self.mu_1_hat
                # deps_p_v = 0.0
                deps_p_v = pmulti * self.mu_1_hat

                # deviatoric plastic strain increment
                # flow vector is coaxial to deviatoric stress

                # deps_p_dev = pmulti * (self.G / ) * s_tr

                # Trail isotropic linear elastic law

                p_next = p_tr - self.K * deps_p_v
                s_next = s_tr * (1.0 - (pmulti * self.G) / sqrt_J2_tr)

                sqrt_J2_next = sqrt_J2_tr - self.G * pmulti
                eps_p_acc_next = eps_p_acc_prev + self.mu_2 * pmulti

                c_next = self.c0 + self.H * eps_p_acc_next

                aux = p_next, s_next, eps_p_acc_next, sqrt_J2_next

                R = yield_function(sqrt_J2_next, p_next, self.mu_1, self.mu_2, c=c_next)

                return R, aux

            def find_roots_cone():
                solver = optx.Newton(rtol=1e-8, atol=1e-8)

                sol = optx.root_find(
                    residuals_cone,
                    solver,
                    0.0,
                    throw=False,
                    has_aux=True,
                    max_steps=5,
                )

                return sol.value

            def pull_to_cone():
                pmulti = jax.lax.stop_gradient(find_roots_cone())

                R, aux = residuals_cone(pmulti, None)
                return aux, pmulti

            (p_next, s_next, eps_p_acc_next, sqrt_J2_next), pmulti = jax.lax.cond(
                sqrt_J2_tr > 0.0,
                pull_to_cone,
                lambda: ((p_tr, jnp.zeros((3, 3)), 0.0, sqrt_J2_tr), 0.0),
            )

            alpha = self.mu_2 / self.mu_1

            beta = jnp.nan_to_num(self.mu_2 / self.mu_1_hat, posinf=0.0, neginf=0.0)

            def residuals_apex(sol, args):
                """Reduced system for non-associated flow rule."""
                deps_p_v = sol

                eps_p_acc_next = eps_p_acc_prev + alpha * deps_p_v

                p_next = p_tr - self.K * deps_p_v

                c_next = self.c0 + self.H * eps_p_acc_next

                # ensure no division by zero when no hardening is present, & non associative flow rule
                R = beta * c_next + p_next

                aux = p_next, jnp.zeros((3, 3)), 0.0
                return R, aux

            def find_roots_apex():
                solver = optx.Newton(rtol=1e-12, atol=1e-12)

                sol = optx.root_find(
                    residuals_apex,
                    solver,
                    0.0,
                    throw=False,
                    has_aux=True,
                    max_steps=5,
                )

                return sol.value

            def pull_to_apex():
                deps_v_p = jax.lax.stop_gradient(find_roots_apex())

                R, aux = residuals_apex(deps_v_p, None)
                return aux

            p_next, s_next, eps_p_acc_next = jax.lax.cond(
                sqrt_J2_next <= 0.0,
                pull_to_apex,
                lambda: (p_next, s_next, eps_p_acc_next),
            )

            stress_next = s_next - p_next * jnp.eye(3)

            eps_e_v_next = (p_next - p_ref) / self.K

            eps_e_d_next = s_next / (2.0 * self.G)

            eps_e_next = eps_e_d_next - (1.0 / 3) * eps_e_v_next * jnp.eye(3)

            return stress_next, eps_e_next, eps_p_acc_next

        stress_next, eps_e_next, eps_p_acc_next = jax.lax.cond(
            is_ep, pull_to_ys, elastic_update
        )

        return stress_next, eps_e_next, eps_p_acc_next
