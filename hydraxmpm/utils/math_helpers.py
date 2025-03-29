"""Math helper functions for the pymudokon package."""

import jax
import jax.numpy as jnp

from ..common.types import (
    TypeFloat,
    TypeFloatMatrix3x3,
    TypeFloatMatrix3x3AStack,
    TypeFloatMatrixAStack,
    TypeFloatScalarAStack,
    TypeFloatVector,
    TypeFloatVectorAStack,
)


def get_double_contraction(A, B):
    return jnp.trace(A @ B.T)


def get_double_contraction_stack(A_stack, B_stack):
    return jax.vmap(get_double_contraction)(A_stack, B_stack)


def get_pressure(stress: TypeFloatMatrix3x3, dim: int = 3) -> TypeFloat:
    """Get compression positive pressure from the cauchy stress tensor.

    $$
    p = -\\mathrm{trace} ( \\boldsymbol \\sigma ) / \\mathrm{dim}
    $$

    Args:
        stress: Cauchy stress tensor
        dim: Dimension. Defaults to 3.

    Returns:
        pressure
    """
    return -(1 / dim) * jnp.trace(stress)


def get_pressure_stack(
    stress_stack: TypeFloatMatrix3x3AStack, dim: int = 3
) -> TypeFloatScalarAStack:
    """Vectorized version of [get_pressure][utils.math_helpers.get_pressure]
    for a stack of stress tensors.

    Args:
        stress_stack: stack of cauchy stress tensors
        dim: dimension. Defaults to 3.

    Returns:
        stack of pressures
    """

    vmap_get_pressure = jax.vmap(get_pressure, in_axes=(0, None))
    return vmap_get_pressure(stress_stack, dim)


def get_dev_stress(stress, pressure=None, dim=3):
    """Get deviatoric part of the cauchy stress tensor.

    $$
    \\boldsymbol s = \\boldsymbol \\sigma - p \\mathbf{I}
    $$

    Args:
        stress: cauchy stress tensor
        pressure: pressure. Defaults to None.
        dim: dimension. Defaults to 3.

    Returns:
        deviatoric stress tensor
    """
    if pressure is None:
        pressure = get_pressure(stress, dim)
    return stress + jnp.eye(3) * pressure


def get_dev_stress_stack(stress_stack, pressure_stack=None, dim=3):
    """Vectorized version of [get_dev_stress][utils.math_helpers.get_dev_stress]
    for a  stress tensors.

    Args:
        stress_stack: stack of cauchy stress tensors
        dim: dimension. Defaults to 3.

    Returns:
        stack of deviatoric stress tensors
    """
    if pressure_stack is None:
        pressure_stack = get_pressure_stack(stress_stack, dim)
    vmap_get_dev_stress = jax.vmap(get_dev_stress, in_axes=(0, 0, None))
    return vmap_get_dev_stress(stress_stack, pressure_stack, dim)


def get_q_vm(stress=None, dev_stress=None, pressure=None, dim=3):
    """Get the scalar von-Mises shear stress from the cauchy stress tensor.

    $$
    q = \\sqrt{3/2 J_2}
    $$
    where  $J_2 = \\frac{1}{2} \\mathrm{trace} ( \\boldsymbol s     \\boldsymbol s^T)$
    is the second invariant of the deviatoric stress tensor.

    Args:
        stress: cauchy stress tensor. Defaults to None.
        dev_stress: deviatoric stress tensor. Defaults to None.
        pressure: input pressure. Defaults to None.
        dim: dimension. Defaults to 3.

    Returns:
        scalar von-Mises shear stress
    """

    if dev_stress is None:
        dev_stress = get_dev_stress(stress, pressure, dim)
    return jnp.sqrt(3 * 0.5 * jnp.trace(dev_stress @ dev_stress.T))


def get_q_vm_stack(
    stress_stack,
    dev_stress_stack=None,
    pressure_stack=None,
    dim=3,
):
    """Vectorized version of [get_q_vm][utils.math_helpers.get_q_vm]
    for a stack of stress tensors.

    Args:
        stress_stack: stack of cauchy stress tensors.
        dev_stress_stack: stack of deviatoric stress tensors tensors.
        dev_stress_stack: stack of pressures.
        dim: dimension. Defaults to 3.

    Returns:
        stack of scalar von-Mises stresses
    """
    if dev_stress_stack is None:
        dev_stress_stack = get_dev_stress_stack(stress_stack, pressure_stack, dim)
    vmap_get_q_vm = jax.vmap(get_q_vm, in_axes=(0, 0, None, None))

    return vmap_get_q_vm(stress_stack, dev_stress_stack, pressure_stack, dim)


def get_J2(stress=None, dev_stress=None, pressure=None, dim=3):
    """Get the second invariant of the deviatoric stress tensor."""
    if dev_stress is None:
        dev_stress = get_dev_stress(stress, pressure, dim)
    return 0.5 * jnp.trace(dev_stress @ dev_stress.T)


def get_J2_stack(
    stress_stack: jax.Array, dev_stress_stack=None, pressure_stack=None, dim=3
):
    """Get the J2 from a stack of stress (or its deviatoric) tensors."""
    if dev_stress_stack is None:
        dev_stress_stack = get_dev_stress_stack(stress_stack, pressure_stack, dim)
    vmap_get_J2 = jax.vmap(get_J2, in_axes=(0, 0, None, None))

    return vmap_get_J2(stress_stack, dev_stress_stack, pressure_stack, dim)


def get_scalar_shear_stress(stress, dev_stress=None, pressure=None, dim=3):
    """Get the shear stress tau=sqrt(1/2 J2)."""
    if dev_stress is None:
        dev_stress = get_dev_stress(stress, pressure, dim)
    return jnp.sqrt(0.5 * jnp.trace(dev_stress @ dev_stress.T))


def get_scalar_shear_stress_stack(
    stress_stack, dev_stress_stack=None, pressure_stack=None, dim=3
):
    """Get the shear stress tau=sqrt(1/2 J2) from a stack of stress tensors."""
    if dev_stress_stack is None:
        dev_stress_stack = get_dev_stress_stack(stress_stack, pressure_stack, dim)
    vmap_get_scalar_shear_stress = jax.vmap(
        get_scalar_shear_stress, in_axes=(0, 0, None, None)
    )

    return vmap_get_scalar_shear_stress(
        stress_stack, dev_stress_stack, pressure_stack, dim
    )


def get_volumetric_strain(strain):
    "Get compressive positive volumetric strain."
    return -jnp.trace(strain)


def get_volumetric_strain_stack(strain_stack: jax.Array):
    """Get compressive positive volumetric strain from a stack strain tensors."""
    vmap_get_volumetric_strain = jax.vmap(get_volumetric_strain, in_axes=(0))
    return vmap_get_volumetric_strain(strain_stack)


def get_dev_strain(strain, volumetric_strain=None, dim=3):
    """Get deviatoric strain tensor."""
    if volumetric_strain is None:
        volumetric_strain = get_volumetric_strain(strain)
    return strain + (1.0 / dim) * jnp.eye(3) * volumetric_strain


def get_dev_strain_stack(strain_stack, volumetric_strain_stack=None, dim=3):
    """Get deviatoric strain tensor from a stack of strain tensors."""
    if volumetric_strain_stack is None:
        volumetric_strain_stack = get_volumetric_strain_stack(strain_stack)
    vmap_get_dev_strain = jax.vmap(get_dev_strain, in_axes=(0))
    return vmap_get_dev_strain(strain_stack, volumetric_strain_stack)


def get_scalar_shear_strain(
    strain=None, dev_strain=None, volumetric_strain=None, dim=3
):
    """Get scalar shear strain gamma = sqrt(1/2 trace(e_dev @ e_dev.T))."""
    if dev_strain is None:
        dev_strain = get_dev_strain(strain, volumetric_strain, dim)

    return jnp.sqrt(0.5 * jnp.trace(dev_strain @ dev_strain.T))


def get_scalar_shear_strain_stack(
    strain_stack: jax.Array, dev_strain_stack=None, volumetric_strain_stack=None, dim=3
):
    """Get scalar shear strain from a stack of strain tensors."""
    vmap_get_scalar_shear_strain = jax.vmap(
        get_scalar_shear_strain, in_axes=(0, 0, 0, None)
    )

    return vmap_get_scalar_shear_strain(
        strain_stack, dev_strain_stack, volumetric_strain_stack, dim
    )


def get_KE(mass: TypeFloat, velocity: TypeFloatVector) -> TypeFloat:
    """Get kinetic energy."""
    return 0.5 * mass * jnp.dot(velocity, velocity)


def get_KE_stack(
    masses: TypeFloatScalarAStack, velocities: TypeFloatVectorAStack
) -> TypeFloatScalarAStack:
    """Get kinetic energy from a stack of masses and velocities."""
    vmap_get_KE = jax.vmap(get_KE, in_axes=(0, 0))
    return vmap_get_KE(masses, velocities)


def get_inertial_number(pressure, dgamma_dt, p_dia, rho_p):
    """Get MiDi inertial number.

    Microscopic pressure time scale over macroscopic shear rate timescale

    I=Tp/T_dot_gamma

    Args:
        pressure: hydrostatic pressure
        dgamma_dt: scalar shear strain rate
        p_dia: particle diameter
        rho_p: particle density [kg/m^3]
    """
    return (dgamma_dt * p_dia) / jnp.sqrt(pressure / rho_p)


def get_inertial_number_stack(pressure_stack, dgamma_dt_stack, p_dia, rho_p):
    """Get the inertial number from a stack of pressures and shear strain rates."""
    vmap_get_inertial_number = jax.vmap(get_inertial_number, in_axes=(0, 0, None, None))
    return vmap_get_inertial_number(
        pressure_stack,
        dgamma_dt_stack,
        p_dia,
        rho_p,
    )


def get_plastic_strain(
    strain,
    elastic_strain,
):
    """Get the plastic strain."""
    return strain - elastic_strain


def get_plastic_strain_stack(strain_stack, elastic_strain_stack):
    """Get the plastic strain from a stack of strain tensors."""
    vmap_get_plastic_strain = jax.vmap(get_plastic_strain, in_axes=(0, 0))
    return vmap_get_plastic_strain(strain_stack, elastic_strain_stack)


def get_small_strain(F):
    """Get small strain tensor from deformation gradient."""
    return 0.5 * (F.T + F) - jnp.eye(3)


def get_small_strain_stack(F_stack):
    """Get small strain tensor from a stack of deformation gradients."""
    vmap_get_small_strain = jax.vmap(get_small_strain)
    return vmap_get_small_strain(F_stack)


def get_strain_rate_from_L(L):
    """Get strain rate tensor from velocity gradient."""
    return 0.5 * (L + L.T)


def get_strain_rate_from_L_stack(L_stack):
    """Get strain rate tensor from a stack of velocity gradients."""
    vmap_get_strain_rate_from_L = jax.vmap(get_strain_rate_from_L)
    return vmap_get_strain_rate_from_L(L_stack)


def phi_to_e(phi):
    """Solid volume fraction to void ratio."""
    return (1.0 - phi) / phi


def phi_to_e_stack(phi_stack):
    """Vectorized version of [phi_to_e][utils.math_helpers.phi_to_e]
    for a stack of solid volume fractions."""
    vmap_phi_to_e = jax.vmap(phi_to_e)
    return vmap_phi_to_e(phi_stack)


def e_to_phi(e):
    """
    Convert void ratio to solid volume fraction, assuming gradients are zero.
    $$
    \\phi = \\frac{1}{1+e}
    $$
    where $e$ is the void ratio and $\\phi$ is the volume fraction.

    Args:
        e: void ratio

    Returns:
        (jnp.float32): solid volume fraction
    """

    return 1.0 / (1.0 + e)


def e_to_phi_stack(e_stack):
    """Vectorized version of [e_to_phi][utils.math_helpers.e_to_phi]
    for a solid volume fraction.

    Args:
        e_stack (chex.Array): void ratio stack

    Returns:
        (chex.Array): solid volume fraction stack
    """
    vmap_e_to_phi = jax.vmap(e_to_phi)
    return vmap_e_to_phi(e_stack)


def get_sym_tensor(A):
    """Get symmetric part of a tensor.

    $$
    B = \\frac{1}{2}(A + A^T)
    $$

    Args:
        A (chex.Array): input tensor

    Returns:
        chex.Array: Symmetric part of the tensor
    """
    return 0.5 * (A + A.T)


def get_sym_tensor_stack(A_stack):
    """Vectorized version of [get_sym_tensor][utils.math_helpers.get_sym_tensor]
    for a stack of tensors.

    Args:
        A_stack (chex.Array): stack of input tensors

    Returns:
        (chex.Array): stack of symmetric tensors.
    """
    vmap_get_sym_tensor = jax.vmap(get_sym_tensor)
    return vmap_get_sym_tensor(A_stack)


def get_skew_tensor(A):
    """Get skew-symmetric part of a tensor.

    $$
    B = \\frac{1}{2}(A - A^T)
    $$

    Args:
        A (chex.Array): input tensor

    Returns:
        (chex.Array): Skew-symmetric part of the tensor
    """
    return 0.5 * (A - A.T)


def get_skew_tensor_stack(A_stack):
    """Vectorized version of  [get_skew_tensor][utils.math_helpers.get_skew_tensor]
    for a stack of tensors.

    Args:
        A_stack (chex.Array): stack of input tensors

    Returns:
        (chex.Array): stack of symmetric tensors.
    """
    vmap_get_skew_tensor = jax.vmap(get_skew_tensor)
    return vmap_get_skew_tensor(A_stack)


def get_phi_from_L(L, phi_prev, dt):
    """Get solid volume fraction from velocity gradient using the mass balance.

    Args:
        L (chex.Array): velocity gradient
        phi_prev (jnp.float32): previous solid volume fraction
        dt (jnp.float32): time step

    Returns:
        jnp.float32: Solid volume fraction
    """
    deps = get_sym_tensor(L) * dt
    deps_v = get_volumetric_strain(deps)
    phi_next = phi_prev / (1.0 - deps_v)
    return phi_next


def get_e_from_bulk_density(absolute_density, bulk_density):
    """Get void ratio from absolute and bulk density."""
    return absolute_density / bulk_density


def get_phi_from_bulk_density(absolute_density, bulk_density):
    """Get solid volume fraction from absolute and bulk density."""
    e = get_e_from_bulk_density(absolute_density, bulk_density)
    return e_to_phi(e)


def get_phi_from_bulk_density_stack(absolute_density_stack, bulk_density_stack):
    """Get volume fraction from a stack of absolute and bulk densities.

    See [get_phi_from_bulk_density][utils.math_helpers.get_phi_from_bulk_density] for
    more details.
    """
    vmap_get_phi_from_bulk_density = jax.vmap(
        get_phi_from_bulk_density, in_axes=(None, 0)
    )
    return vmap_get_phi_from_bulk_density(absolute_density_stack, bulk_density_stack)


def get_hencky_strain(F):
    """Get Hencky strain from the deformation gradient.

    Do Singular Value Decomposition (SVD) of the deformation gradient $F$ to get the
    singular values, left stretch tensor $U$ and right stretch tensor $V^T$. After, take
    the matrix logarithm of the singular values to get the Hencky strain.

    issues with forward mode AD.
    https://github.com/jax-ml/jax/issues/2011

    Args:
        F (chex.Array): deformation gradient

    Returns:
        Tuple[chex.Array, chex.Array, chex.Array]: strain tensor, left stretch tensor,
        right stretch tensor
    """
    u, s, vh = jnp.linalg.svd(F, full_matrices=False)

    eps = jnp.zeros((3, 3)).at[[0, 1, 2], [0, 1, 2]].set(jnp.log(s))
    return eps, u, vh


def get_hencky_strain_stack(F_stack):
    """Vectorized version of get Hencky strain from a stack of deformation gradients.

    See [get_hencky_strain][utils.math_helpers.get_hencky_strain] for more details.

    Args:
        F_stack: deformation gradient stack

    Returns:
        strain tensor, left stretch tensor (stacked)
    """
    vmap_get_hencky = jax.vmap(get_hencky_strain, in_axes=(0))
    return vmap_get_hencky(F_stack)


def get_k0_stress(
    height: jnp.float32,
    gravity: jnp.float32,
    rho_0: jnp.float32,
    mu: jnp.float32,
    axis_vertical: jnp.int32 = 3,
) -> jnp.float32:
    """Get k0 stress tensor.

    ??? warning
        This function is still being developed and may not work as expected.

    """
    import warnings

    warnings.warn(
        "This function is still being developed and may not work as expected."
    )

    fric_angle = jnp.arctan(mu)

    K0 = 1 - jnp.sin(fric_angle)

    factor = height * gravity * rho_0

    stress = jnp.zeros((3, 3))
    stress = jnp.zeros((3, 3)).at[[0, 1, 2], [0, 1, 2]].set(factor * K0)

    stress = stress.at[axis_vertical, axis_vertical].set(factor)

    p = get_pressure(stress, dim=2)

    q = get_q_vm(stress, dim=2)

    mu = (q / p) / jnp.sqrt(3)

    return stress
