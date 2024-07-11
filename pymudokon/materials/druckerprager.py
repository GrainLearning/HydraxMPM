"""Implementation, non-associated Drucker-Prager model with isotropic linear elasticity, and linear hardening.

[1] de Souza Neto, E.A., Peric, D. and Owen, D.R., 2011. Computational methods for plasticity.
"""

import warnings
from functools import partial
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from typing_extensions import Self

from .material import Material

from ..core.particles import Particles


def yield_function(J2: jnp.float32, p: jnp.float32, eta: jnp.float32, xi: jnp.float32, c: jnp.float32):
    """Drucker-Prager yield function."""
    return jnp.sqrt(J2) - eta * p - xi * c


@chex.dataclass
class DruckerPrager(Material):
    r"""Non-associated Drucker-Prager model.

    The Drucker-Prager model is a smooth approximation to the Mohr-Coulomb model.

    .. math::
        f = \\sqrt{J_2} - \\eta p - \\xi c

    where :math:`J_2` is the second invariant of the deviatoric stress tensor,
    :math:`p` is the pressure,
    :math:`\\eta` is the friction parameter, :math:`\\xi` is the cohesion parameter
    and :math:`c` is the cohesion.

    The non-associated plastic potential is adopted
    .. math::
        g = \\sqrt{J_2} - \\overline \\eta p

    where :math:`\\overline \\eta` is the dilatancy parameter. Friction and dilatancy angles are in radians.
    The values for \\eta \\overline \\eta and \\xi are determined by the cone approximation:

    0 - Outer cone approximation (compressive Mohr-Coulomb cone) approaximation matches uniaxial compression
        and biaxial tension.
    1 - Inner cone approximation (tension Mohr-Coulomb cone) approximation matches uniaxial tension
        and biaxial compression.
    2 - Plain strain approximation

    This formulation is in small strain and elastic law is  isotropic linear elasticity.

    The implementation follows [1] with the exception that pressure and volumetric strain are positive in compression.

    [1] de Souza Neto, E.A., Peric, D. and Owen, D.R., 2011. Computational methods for plasticity.

    Usage in MPM simulatios:
    >>> import pymudokon as pm
    >>> material = pm.DruckerPrager.create(E=1.0e6, nu=0.3, friction_angle=30.0, dilatancy_angle=30.0, cohesion=1.0e3)
    >>> # Add material to the simulation ...


    Attributes:
        E: Young's modulus.
        nu: Poisson's ratio.
        G: Shear modulus.
        K: Bulk modulus.
        eta: Mohr-Coulomb friction parameter.
        eta_hat: Mohr-Coulomb dilatancy parameter.
        xi: Mohr-Coulomb cohesion parameter.
        c0: Initial cohesion parameter.
        eps_acc: Accumulated plastic strain for linear hardening
        eps_e: Elastic strain tensor.
        H: Hardening modulus
    """

    E: jnp.float32
    nu: jnp.float32
    G: jnp.float32
    K: jnp.float32

    eps_e: chex.Array
    eps_p_acc: chex.Array

    eta: jnp.float32 = 0.0
    eta_hat: jnp.float32 = 0.0

    xi: jnp.float32 = 0.0
    c0: jnp.float32 = 0.0

    H: jnp.float32 = 0.0

    @classmethod
    def create(
        cls: Self,
        E: jnp.float32,
        nu: jnp.float32,
        friction_angle: jnp.float32,
        dilatancy_angle: jnp.float32,
        cohesion: jnp.float32,
        H: jnp.float32 = 0.0,
        cone_approximation: jnp.int32 = 0,
        stress_ref: chex.Array = None,
        num_particles: jnp.int32 = 1,
        dim: jnp.int16 = 3,
    ) -> Self:
        """Create a non-associated Drucker-Prager material model."""

        def eta_outer(phi):
            return 6 * jnp.sin(phi) / (jnp.sqrt(3) * (3 - jnp.sin(phi)))

        def eta_inner(phi):
            return 6 * jnp.sin(phi) / (jnp.sqrt(3) * (3 + jnp.sin(phi)))

        def eta_plain_strain(phi):
            return 3 * jnp.tan(phi) / jnp.sqrt(9 + 12 * jnp.tan(phi) ** 2)

        def xi_outer(phi):
            return 6 * jnp.cos(phi) / (jnp.sqrt(3) * (3 - jnp.sin(phi)))

        def xi_inner(phi):
            return 6 * jnp.cos(phi) / (jnp.sqrt(3) * (3 + jnp.sin(phi)))

        def xi_plain_strain(phi):
            return 3 / (9 + 12 * jnp.tan(phi) ** 2)

        eta = jax.lax.switch(cone_approximation, [eta_outer, eta_inner, eta_plain_strain], friction_angle)

        eta_hat = jax.lax.switch(cone_approximation, [eta_outer, eta_inner, eta_plain_strain], dilatancy_angle)

        xi = jax.lax.switch(cone_approximation, [xi_outer, xi_inner, xi_plain_strain], friction_angle)

        K = E / (3.0 * (1.0 - 2.0 * nu))
        G = E / (2.0 * (1.0 + nu))

        eps_e = jnp.zeros((num_particles, dim, dim), dtype=jnp.float32)

        eps_p_acc = jnp.zeros(num_particles, dtype=jnp.float32)

        if stress_ref is None:
            stress_ref = jnp.zeros((num_particles, dim, dim), dtype=jnp.float32)

        return cls(
            stress_ref=stress_ref,
            eps_e=eps_e,
            E=E,
            nu=nu,
            G=G,
            K=K,
            eta=eta,
            eta_hat=eta_hat,
            xi=xi,
            H=H,
            c0=cohesion,
            eps_p_acc=eps_p_acc,
        )

    def update_stress(self: Self, particles: Particles, dt: jnp.float32) -> Tuple[Particles, Self]:
        """Update stress and strain for all particles."""
        vel_grad = particles.velgrads

        vel_grad_T = jnp.transpose(vel_grad, axes=(0, 2, 1))

        deps = 0.5 * (vel_grad + vel_grad_T) * dt
        # Elastic step
        s_tr, p_tr, eps_e_tr, s_ref, p_ref = self.vmap_elastic_trail_step(self.stress_ref, self.eps_e, deps)

        # Plastic return mapping
        stress_next, eps_p_acc_next, eps_e_next = self.vmap_plastic_return_mapping(
            s_tr, p_tr, eps_e_tr, self.eps_p_acc, s_ref, p_ref
        )

        particles = particles.replace(stresses=stress_next)

        return particles, self

    @jax.jit
    def update_stress_benchmark(
        self: Self,
        strain_rate: chex.Array,
        volume_fraction: chex.Array,
        dt: jnp.float32,
    ) -> Tuple[Self, chex.Array]:
        """Update stress using the Drucker-Prager model with single integration point benchmarks."""
        strain_rate = strain_rate.reshape(-1, 3, 3)
        deps = strain_rate * dt

        # Elastic step
        s_tr, p_tr, eps_e_tr, s_ref, p_ref = self.vmap_elastic_trail_step(self.stress_ref, self.eps_e, deps)

        # Plastic return mapping
        stress_next, eps_p_acc_next, eps_e_next = self.vmap_plastic_return_mapping(
            s_tr, p_tr, eps_e_tr, self.eps_p_acc, s_ref, p_ref
        )

        return self.replace(eps_e=eps_e_next, eps_p_acc=eps_p_acc_next), stress_next

    @partial(jax.vmap, in_axes=(None, 0, 0, 0), out_axes=(0, 0, 0, 0, 0))
    def vmap_elastic_trail_step(
        self, stress_ref: chex.ArrayBatched, eps_e_prev: chex.ArrayBatched, deps_next: chex.ArrayBatched
    ) -> Tuple[chex.ArrayBatched, chex.ArrayBatched, chex.ArrayBatched, chex.ArrayBatched, chex.ArrayBatched]:
        dim = deps_next.shape[0]
        # Reference stress and pressure
        p_ref = -jnp.trace(stress_ref) / dim

        s_ref = stress_ref + p_ref * jnp.eye(3)

        eps_e_tr = eps_e_prev + deps_next

        eps_e_v_tr = -jnp.trace(eps_e_tr)

        eps_e_d_tr = eps_e_tr + (eps_e_v_tr / dim) * jnp.eye(dim)

        s_tr = 2.0 * self.G * eps_e_d_tr

        p_tr = self.K * eps_e_v_tr

        p_tr = p_tr + p_ref

        s_tr = s_tr + s_ref

        return s_tr, p_tr, eps_e_tr, s_ref, p_ref

    @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0, 0), out_axes=(0, 0, 0))
    def vmap_plastic_return_mapping(
        self,
        s_tr: chex.ArrayBatched,
        p_tr: chex.ArrayBatched,
        eps_e_tr: chex.ArrayBatched,
        eps_p_acc: chex.ArrayBatched,
        s_ref: chex.ArrayBatched,
        p_ref: chex.ArrayBatched,
    ) -> Tuple[chex.ArrayBatched, chex.ArrayBatched, chex.ArrayBatched]:
        """Plastic implicit return mapping algorithm."""
        J2_tr = jnp.sqrt(0.5 * (s_tr @ s_tr.T).trace())

        c = self.c0 + self.H * eps_p_acc  # linear hardening

        yf = yield_function(J2_tr, p_tr, self.eta, self.xi, c=c)

        is_ep = yf > 0

        def elastic_update() -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
            """If yield function is negative, return elastic solution."""
            stress = s_tr - p_tr * jnp.eye(3)
            return stress, eps_p_acc, eps_e_tr

        def return_mapping():
            """If yield function is positive, pull back to the yield surface."""
            sol = 0.0

            R = 1.0

            aux_data = p_tr, s_tr, eps_p_acc

            def reduced_system_cone(sol):
                """Reduced system for non-associated flow rule."""
                p_multi = sol
                # solve non associated flow rules
                # volumetric plastic strain increment
                deps_p_v = -p_multi * self.eta_hat

                # deviatoric plastic strain increment
                # flow vector is coaxial to deviatoric stress
                deps_p_gamma = p_multi * s_tr / (2 * jnp.sqrt(J2_tr))

                # Trail isotropic linear elastic law
                p_next = p_tr - self.K * deps_p_v
                s_next = s_tr - 2.0 * self.G * deps_p_gamma

                # linear hardening
                eps_p_acc_next = eps_p_acc + self.xi * deps_p_v
                c = self.c0 + self.H * eps_p_acc_next

                aux_data = p_next, s_next, eps_p_acc_next
                R = yield_function(J2_tr, p_next, self.eta, self.xi, c=c)

                # R = R / (self.E * p_next)

                return R, aux_data

            def reduced_system_apex(sol):
                p_multi = sol
                deps_p_v = -p_multi * self.eta_hat
                p_next = p_tr - self.K * deps_p_v
                eps_p_acc_next = eps_p_acc + self.xi * deps_p_v
                c = self.c0 + self.H * eps_p_acc_next
                R = yield_function(0, p_next, self.eta, self.xi, c=c)

                # R = R / (self.E * p_next)

                s_next = jnp.zeros((3, 3), dtype=jnp.float32)

                aux_data = p_next, s_next, eps_p_acc_next

                return R, aux_data

            def NewtonRhapson(step, carry):
                """Newton-Raphson iteration cone."""
                R, sol, aux_data = carry

                R, aux_data = jax.lax.cond(
                    is_cone,
                    reduced_system_cone,
                    reduced_system_apex,
                    sol,
                )

                d, *_ = jax.lax.cond(
                    is_cone,
                    jax.jacfwd(reduced_system_cone, has_aux=True),
                    jax.jacfwd(reduced_system_apex, has_aux=True),
                    sol,
                )

                # d, *_ = jax.jacfwd(reduced_system_cone, has_aux=True)(sol)

                sol = sol - R / d

                return R, sol, aux_data

            is_cone = True
            R, sol, aux_data = jax.lax.fori_loop(0, 30, NewtonRhapson, (R, sol, aux_data))

            is_cone = False
            R, sol, aux_data = jax.lax.cond(
                J2_tr - self.G * sol >= 0,  # noqa: E999
                lambda carry: carry,
                lambda carry: jax.lax.fori_loop(0, 30, NewtonRhapson, carry),
                (R, sol, aux_data),
            )

            p_next, s_next, eps_p_acc_next = aux_data

            stress_next = s_next - p_next * jnp.eye(3)

            eps_e_next = (s_next - s_ref) / (2.0 * self.G) - (p_next - p_ref) / (3.0 * self.K) * jnp.eye(3)

            return stress_next, eps_p_acc_next, eps_e_next

        return jax.lax.cond(is_ep, return_mapping, elastic_update)
