"""
Implementation of the Unified Hardening (UH) Model for normally consolidated clays [1,2]. This model is implemented in double natural log specific volume and pressure space ($\ln v$ - $\ln p$) space instead of void ratio and natural log pressure space
$e$-$\ln p$ space as the original. benefits are given by ref [2].

**Current and reference yield surface**

The model links two yield surfaces, named the current yield surface and a reference yield. The current yield surface, passes through all stress states during yielding, and is defined by the equation:

\\begin{align}
F=\ln\\left(\\frac{p}{p_{x0}}\\right)
+\\ln\\left(1+\\frac{q^{2}}{M^{2}p^{2}}\\right)-\\frac{H}{c_{p}},
\\end{align}
where $q$ is the von-Mises shear stress, $p$ is the pressure,
$p_{x0}$ is the initial reference pressure, i.e., initial intersection of the
current yield surface with the hydrostatic axis; $M$ is the slope of the critical
state line; The constant $c_{p}=\\lambda - \\kappa$
composes of the slope of the normal compression line (NCL) $\\lambda$ and unloading line $\\kappa$.

The hardening parameter $H$, controls the size of the yield surface and is defined as:
\\begin{align}
H=\int\\frac{M_f^4 - m^4}{M^4 - m^4}d\\varepsilon_{p}^{v}
\\end{align}

where $m= q/p$, and $M_f$ is the potential failure stress ratio and be found from the parabolic Hvorslev envelope:

\\begin{align}
M_f = 6\\left(\\sqrt{\\frac{12(3-M)}{M^2}R + 1} + 1\\right)^{-1}.
\\end{align}

The variable R is an overconsolidation parameter and links the current and refrence yield surface $R=\\frac{p}{\\overline p}=\\frac{q}{\\overline q}$. The quantity $R$ can also be determined by linking both yield surfaces in the $\\ln v$$-\\ln p$ space. The distance between the NCL and anisotropic compression line (ACL) in volumetric space is given by [2,3]:

\\begin{align}
\ln v_m = \ln N - \\lambda \ln p - (\\lambda - \\kappa) \ln(1 + \\frac{m^2}{M^2}).
\\end{align}

The overconsolidation parameter is then defined in terms of distance between the NCL and ACL $\\xi = \ln v - \ln v_m=\ln \\frac{v}{v_m}$, and written as:
\\begin{align}
R = \\exp\\left(- \\frac{ \\xi }{\\lambda - \\kappa} \\right),
\\end{align}

The reference yield surface takes a similar form to the current yield surface with the
exceptions: that it is defined at stress points $\overline p, \overline q$; and
the hardening parameter is replaced with the plastic volumetric strain $\\varepsilon_{p}^{v}$.

**Plastic potential function**
The flow rule is associated with the current yield surface, which is defined as:
\\begin{align}
g=\ln\\left(\\frac{p}{p_{x}}\\right)
+\\ln\\left(1+\\frac{q^{2}}{M^{2}p^{2}}\\right)=0,
\\end{align}
where $p_{x}$ is the intersection of the current yield surface with the hydrostatic axis:

\\begin{align}
p_x = p_{x0} \\exp\\left(-\\frac{H}{c_{p}}\\right).
\\end{align}

The flow rules and elastic law are taken the same as in the ,modified Cam-Clay model.


1. Yao, Y-P., Wei Hou, and A-N. Zhou. "UH model: three-dimensional unified hardening
model for overconsolidated clays." Géotechnique 59.5 (2009): 451-469.
2. Yao, Yang-Ping, et al. "Unified hardening (UH) model for clays and sands." Computers and Geotechnics 110 (2019): 326-343.
3. Houlsby, Guy Tinmouth. Study of plasticity theories and their applicability to soils. PhD Diss. 1981.

"""

from functools import partial
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array
from typing_extensions import Self

from ...particles.particles import Particles
from ...utils.jax_helpers import simple_warning
from ...utils.math_helpers import (
    get_dev_strain,
    get_dev_stress,
    get_pressure,
    get_pressure_stack,
    get_q_vm,
    get_sym_tensor_stack,
    get_volumetric_strain,
)
from ..constitutive_law import Material


def plot_yield_surface(
    ax, p_range: Tuple, M: jnp.float32, p_c: jnp.float32, color="black", linestyle="--"
):
    """Plot the yield surface in q - p space."""
    raise NotImplementedError  # pragma: no cover


def plot_ncl_csl(
    ax,
    p_range: Tuple,
    M: jnp.float32,
    ln_N: jnp.float32,
    lam: jnp.float32,
    kap: jnp.float32,
    ncl_color="black",
    csl_color="red",
    linestyle="--",
    is_lnv_lnp=True,
):
    """Plot NCL and CSL in ln v - ln p OR v - ln p space."""
    p_stack = jnp.arange(p_range[0], p_range[1], p_range[2])

    ln_v_ncl = jax.vmap(get_ln_v_acl, in_axes=(0, None, None, None, None, None))(
        p_stack, 0.0, ln_N, lam, kap, M
    )

    v_ncl_stack = jnp.exp(ln_v_ncl)
    ax.plot(p_stack, v_ncl_stack, color=ncl_color, linestyle=linestyle)
    ax.set_xscale("log")

    if is_lnv_lnp:
        ax.set_yscale("log")

    ln_v_csl = jax.vmap(get_ln_v_acl, in_axes=(0, 0, None, None, None, None))(
        p_stack, p_stack * M, ln_N, lam, kap, M
    )

    v_csl_stack = jnp.exp(ln_v_csl)
    ax.plot(p_stack, v_csl_stack, color=csl_color, linestyle=linestyle)
    ax.set_xscale("log")

    if is_lnv_lnp:
        ax.set_yscale("log")

    return ax


def get_elas_non_linear_pressure(deps_e_v, kap, p_prev):
    """Compute non-linear pressure."""
    out = p_prev / (1.0 - (1.0 / kap) * deps_e_v)
    return out


def get_K(kap, p):
    """Get pressure dependent bulk modulus."""
    return (1.0 / kap) * p


def get_G(nu, K):
    """Get pressure dependent shear modulus via bulk modulus K(p)."""
    return (3.0 * (1.0 - 2.0 * nu)) / (2.0 * (1.0 + nu)) * K


def get_elas_dev_stress(eps_e_d, s_ref, G):
    """Get elastic deviatoric stress tensor."""
    return 2.0 * G * eps_e_d + s_ref


def get_ln_v_acl(p, q, ln_N, lam, kap, M):
    """Distance from ACL to NCL in natural log specific volume space / pressure space."""
    m = q / p
    ln_v = ln_N - lam * jnp.log(p) - (lam - kap) * jnp.log(1 + m**2 / M**2)
    return ln_v


def yield_functions(p, q, H, cp, px0, M):
    """Unified Hardening current yield function."""
    return jnp.log(p / px0) + jnp.log(1.0 + (q**2) / (p**2 * M**2)) - H / cp


def plastic_potential(stress, H, cp, px0, M):
    """Unified Hardening plastic potential function."""
    q = get_q_vm(stress)
    p = get_pressure(stress)
    return yield_functions(p, q, H, cp, px0, M)


def get_flattened_triu_3x3(vals):
    """Convert flattened upper triangular values to 3x3 symmetric tensor."""
    new = jnp.zeros((3, 3))
    inds = jnp.triu_indices_from(new)
    new = new.at[inds].set(vals)
    new = new.at[inds[1], inds[0]].set(vals)
    return new


def get_flattened_triu(A):
    """Get flattened upper triangular components of a 3x3 symmetric tensor."""
    return A.at[jnp.triu_indices(A.shape[0])].get()


def get_state_variable(ln_N, ln_v):
    """Get state variable specific volume on NCL - current specific volume."""
    return ln_N - ln_v


def get_R(xi, cp):
    """Get overconsolidation parameter."""
    return jnp.exp(-xi / cp)


def get_Mf(M, R):
    """Get potential failure stress ratio from Hvorslev envelope."""
    term_sqrt = jnp.sqrt((12 * (3 - M) / M**2) * R + 1)
    return 6.0 / (term_sqrt + 1)


def get_H(q, p, Mf, M, deps_p_v, H_prev):
    """Get hardening parameter"""
    m = q / p
    curr = (Mf**4 - m**4) / (M**4 - m**4)
    # denominator may be zero due to numerical issues
    return jnp.nan_to_num(curr, posinf=0.0) * deps_p_v + H_prev


@chex.dataclass
class UHModel(Material):
    """Unified Hardening (UH) Model for normally consolidated clays.

    Attributes:
        nu: Young's modulus.
        M: Slope of critical state line.
        lam: Slope of compression curve.
        kap: Slope of decompression curve.
        ln_N: Natural logarithm of specific volume on the NCL at 1 KPa.
        R: Overconsolidation parameter $0 \le R \leq 1$.
        cp: Constant $c_p = \lambda - \kappa$.
        H_stack: Hardening parameter stack.
        px0_stack: Reference pressure stack.
        eps_e_stack: Elastic strain stack.
        stress_ref_stack: Reference stress stack.
    """

    nu: jnp.float32
    M: jnp.float32
    lam: jnp.float32
    kap: jnp.float32
    ln_N: jnp.float32
    R: jnp.float32
    cp: jnp.float32
    H_stack: chex.Array
    px0_stack: chex.Array
    eps_e_stack: chex.Array
    stress_ref_stack: chex.Array

    def get_phi_ref(self, stress_ref: chex.Array, dim=3):
        """Get reference solid volume fraction. Over consolidation parameter is
        $$
        R = p_{x0} / p_c = \exp(-\\xi / c_p),
        $$
        where $c_p=\\lambda - \\kappa$ and state parameter $\\xi = \ln v - \ln v_m$.

        We can find the current specific volume from the distance of the NCL to ACL
        $$
        \\ln v = \\ln v_m + c_p \\ln R
        $$
        where $\\ln v_m$ is obtained from the distance of the NCL to ACL in
        linear natural logarithm $\\ln v - \\ln p$ space:
        $$
        \\ln v = \\ln N - \\lambda \\ln p - (\\lambda - \\kappa) \\ln(1 + \\frac{m^2}{M^2})
        $$
        This function is only applicable when $q=0$.

        Args:
            stress_ref (chex.Array): cauchy stress tensor
            dim (int, optional): _description_. Defaults to 3.

        Returns:
            (chex.Array): current specific volume
        """

        ln_v_m = get_ln_v_acl(
            get_pressure(stress_ref, dim=dim),
            get_q_vm(stress_ref, dim=dim),
            self.ln_N,
            self.lam,
            self.kap,
            self.M,
        )

        ln_v = ln_v_m + self.cp * jnp.log(self.R)
        v = jnp.exp(ln_v)

        return 1.0 / v

    @classmethod
    def create(
        cls: Self,
        nu: jnp.float32,
        M: jnp.float32,
        lam: jnp.float32,
        kap: jnp.float32,
        ln_N: jnp.float32,
        R: jnp.float32,
        stress_ref_stack: Array = None,
        num_particles: jnp.int32 = 1,
        dim: jnp.int16 = 3,
    ) -> Self:
        """Create a new instance UH model

        Args:
            cls (Self): Self type reference
            nu (jnp.float32): Poisson's ratio.
            M (jnp.float32): Slope of Critcal state line.
            lam (jnp.float32): Compression index.
            kap (jnp.float32): Decompression index.
            ln_N (jnp.float32): Natural logarithm of normal compression curve $\ln N$.
            R (jnp.float32): Overconsolidation parameter.
            stress_ref_stack (Array): Reference stress tensor.
            num_particles (_type_): Number of particles.
            dim (jnp.int16, optional): Dimension of the domain. Defaults to 3.
        """

        eps_e_stack = jnp.zeros((num_particles, 3, 3), dtype=jnp.float32)

        if stress_ref_stack is None:
            stress_ref_stack = jnp.zeros((num_particles, 3, 3), dtype=jnp.float32)

        H_stack = jnp.zeros(num_particles, dtype=jnp.float32)

        px0_stack = get_pressure_stack(stress_ref_stack, dim)

        cp = lam - kap

        return cls._check(
            stress_ref_stack=stress_ref_stack,
            px0_stack=px0_stack,
            H_stack=H_stack,
            eps_e_stack=eps_e_stack,
            nu=nu,
            M=M,
            lam=lam,
            kap=kap,
            ln_N=ln_N,
            R=R,
            cp=cp,
            absolute_density=1.0,
        )

    def update_from_particles(
        self: Self, particles: Particles, dt: jnp.float32
    ) -> Tuple[Particles, Self]:
        """Update the material state and particle stresses for MPM solver."""

        stress_stack, self = self.update(
            particles.stress_stack, particles.F_stack, particles.L_stack, None, dt
        )

        return particles.replace(stress_stack=stress_stack), self

    def update(
        self: Self,
        stress_prev_stack: chex.Array,
        F_stack: chex.Array,
        L_stack: chex.Array,
        phi_stack: chex.Array,
        dt: jnp.float32,
    ) -> Tuple[chex.Array, Self]:
        deps_stack = get_sym_tensor_stack(L_stack) * dt

        stress_next_stack, eps_e_next_stack, H_next_stack = self.vmap_update_stress(
            deps_stack,
            self.stress_ref_stack,
            stress_prev_stack,
            self.eps_e_stack,
            self.px0_stack,
            self.H_stack,
            phi_stack,
        )

        return (
            stress_next_stack,
            self.replace(eps_e_stack=eps_e_next_stack, H_stack=H_next_stack),
        )

    @partial(
        jax.vmap,
        in_axes=(
            None,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ),
        out_axes=(0, 0, 0),
    )
    def vmap_update_stress(
        self,
        deps_next,
        stress_ref,
        stress_prev,
        eps_e_prev,
        px0,
        H_prev,
        phi_curr,
    ):
        dim = deps_next.shape[0]
        p_prev = get_pressure(stress_prev, dim)

        # Reference pressure and deviatoric stress
        p_ref = get_pressure(stress_ref, dim)

        s_ref = get_dev_stress(stress_ref, p_ref, dim)

        # Trail Elastic strain
        eps_e_tr = eps_e_prev + deps_next

        eps_e_v_tr = get_volumetric_strain(eps_e_tr)

        eps_e_d_tr = get_dev_strain(eps_e_tr, eps_e_v_tr)

        deps_e_v = get_volumetric_strain(deps_next)

        p_tr = get_elas_non_linear_pressure(deps_e_v, self.kap, p_prev)

        K_tr = get_K(self.kap, p_tr)

        G_tr = get_G(self.nu, K_tr)

        s_tr = get_elas_dev_stress(eps_e_d_tr, s_ref, G_tr)

        q_tr = get_q_vm(dev_stress=s_tr)

        H_tr = H_prev
        yf_prev = yield_functions(p_tr, q_tr, H_tr, self.cp, px0, self.M)

        def elastic_update():
            """Do elastic update if yield function is negative"""
            stress_next = s_tr - p_tr * jnp.eye(3)
            return stress_next, eps_e_tr, H_tr

        def pull_to_ys():
            """Do return mapping if yield function is positive"""

            def residuals(sol, args):
                """Return mapping is performed on plastic multiplier and upper triangular
                components of the plastic strain tensor."""
                pmulti_curr, *deps_p_flat = sol

                deps_p = get_flattened_triu_3x3(deps_p_flat)

                deps_p_v = get_volumetric_strain(deps_p)

                deps_p_d = get_dev_strain(deps_p, deps_p_v)

                p_next = get_elas_non_linear_pressure(
                    deps_e_v - deps_p_v, self.kap, p_prev
                )

                K_next = get_K(self.kap, p_next)

                G_next = get_G(self.nu, K_next)

                s_next = s_tr - get_elas_dev_stress(deps_p_d, jnp.zeros((3, 3)), G_next)

                q_next = get_q_vm(dev_stress=s_next, dim=dim)

                ln_v_m = get_ln_v_acl(
                    p_next, q_next, self.ln_N, self.lam, self.kap, self.M
                )

                ln_v = jnp.log(1.0 / phi_curr)
                xi = get_state_variable(ln_v_m, ln_v)

                R_param = get_R(xi, self.cp)

                Mf = get_Mf(self.M, R_param)

                H_next = get_H(
                    q_next, p_next, Mf, self.M, deps_p_v, H_prev
                )  # integrate hardening parameter

                yf_curr = yield_functions(p_next, q_next, H_next, self.cp, px0, self.M)

                stress_next = s_next - p_next * jnp.eye(3)

                N = jax.grad(plastic_potential, argnums=0)(
                    stress_next, H_next, self.cp, px0, self.M
                )  # flow vector is associated with respect to current yield surface

                deps_p_fr = pmulti_curr * N

                deps_p_fr_flat = get_flattened_triu(deps_p - deps_p_fr)

                R = jnp.array([yf_curr, *deps_p_fr_flat])

                R = R.at[0].set(R[0] / (K_tr * self.kap) / 2)

                aux = (p_next, s_next, H_next)

                return R, aux

            def find_roots():
                """Find roots of the residuals function."""

                # 1st components plastic multiplier,
                # other 6 components of the plastic strain tensor
                init_val = jnp.zeros(7)

                solver = optx.Newton(rtol=1e-8, atol=1e-8)
                sol = optx.root_find(
                    residuals, solver, init_val, throw=False, has_aux=True, max_steps=20
                )
                return sol.value

            pmulti_curr, *deps_p_flat = jax.lax.stop_gradient(find_roots())

            R, aux = residuals([pmulti_curr, *deps_p_flat], None)

            p_next, s_next, H_next = aux

            stress_next = s_next - p_next * jnp.eye(3)

            eps_e_next = eps_e_tr - get_flattened_triu_3x3(deps_p_flat)

            return stress_next, eps_e_next, H_next

        return jax.lax.cond(yf_prev > 0.0, pull_to_ys, elastic_update)

    @classmethod
    def _check(cls, **kwargs):
        """Check the input arguments and calculated arguments."""

        def assert_scalar_positive(val):
            return chex.assert_scalar_positive(jnp.float32(val).item())

        assert_scalar_positive(kwargs["nu"])
        assert_scalar_positive(kwargs["M"])
        assert_scalar_positive(kwargs["lam"])
        assert_scalar_positive(kwargs["kap"])
        assert_scalar_positive(kwargs["ln_N"])
        assert_scalar_positive(kwargs["R"])
        assert_scalar_positive(kwargs["cp"])

        chex.assert_shape(kwargs["stress_ref_stack"], (..., 3, 3))
        simple_warning(
            kwargs["R"] < 0.0 or kwargs["R"] > 1.0,
            "Warning! Overconsolidation parameter R must be in [0,1] range",
            cls.__name__,
        )

        simple_warning(
            kwargs["lam"] < kwargs["kap"],
            "Warning! Lambda must be greater than kappa",
            cls.__name__,
        )

        return cls(**kwargs)
