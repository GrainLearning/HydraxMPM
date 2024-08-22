"""Module containing helper functions for mathematical operations.

To be used for post-processing and analysis of the simulation results.
"""

import chex
import jax
import jax.numpy as jnp


def get_pressure(stress: jax.Array, dim=3) -> jnp.float32:
    """Compression positive pressure from stress tensor."""
    return -(1 / dim) * jnp.trace(stress)


def get_pressure_stack(stress_stack: jax.Array, dim=3) -> chex.Array:
    """Get compression positive pressure from a stack of stress tensors."""
    vmap_get_pressure = jax.vmap(get_pressure, in_axes=(0, None))
    return vmap_get_pressure(stress_stack, dim)


def get_dev_stress(stress: jax.Array, pressure=None, dim=3) -> chex.ArrayBatched:
    """Get deviatoric stress tensor."""
    if pressure is None:
        pressure = get_pressure(stress, dim)
    return stress + jnp.eye(3) * pressure


def get_dev_stress_stack(
    stress_stack: jax.Array, pressure_stack=None, dim=3
) -> chex.Array:
    """Get deviatoric stress tensor from a stack of stress tensors."""
    if pressure_stack is None:
        pressure_stack = get_pressure_stack(stress_stack, dim)
    vmap_get_dev_stress = jax.vmap(get_dev_stress, in_axes=(0, 0, None))
    return vmap_get_dev_stress(stress_stack, pressure_stack, dim)


def get_q_vm(
    stress: jax.Array = None, dev_stress=None, pressure=None, dim=3
) -> jnp.float32:
    """Get the von Mises stress from the stress tensor sqrt(3/2*J2)."""
    if dev_stress is None:
        dev_stress = get_dev_stress(stress, pressure, dim)
    return jnp.sqrt(3 * 0.5 * jnp.trace(dev_stress @ dev_stress.T))


def get_q_vm_stack(
    stress_stack: jax.Array, dev_stress_stack=None, pressure_stack=None, dim=3
) -> chex.Array:
    """Get the von Mises stress from a stack of stress tensors."""
    if dev_stress_stack is None:
        dev_stress_stack = get_dev_stress_stack(stress_stack, pressure_stack, dim)
    vmap_get_q_vm = jax.vmap(get_q_vm, in_axes=(0, 0, None, None))

    return vmap_get_q_vm(stress_stack, dev_stress_stack, pressure_stack, dim)


def get_J2(stress: jax.Array, dev_stress=None, pressure=None, dim=3) -> jnp.float32:
    """Get the second invariant of the deviatoric stress tensor."""
    if dev_stress is None:
        dev_stress = get_dev_stress(stress, pressure, dim)
    return 0.5 * jnp.trace(dev_stress @ dev_stress.T)


def get_J2_stack(
    stress_stack: jax.Array, dev_stress_stack=None, pressure_stack=None, dim=3
) -> chex.Array:
    """Get the J2 from a stack of stress (or its deviatoric) tensors."""
    if dev_stress_stack is None:
        dev_stress_stack = get_dev_stress_stack(stress_stack, pressure_stack, dim)
    vmap_get_J2 = jax.vmap(get_J2, in_axes=(0, 0, None, None))

    return vmap_get_J2(stress_stack, dev_stress_stack, pressure_stack, dim)


def get_scalar_shear_stress(
    stress: jax.Array, dev_stress=None, pressure=None, dim=3
) -> jnp.float32:
    """Get the shear stress tau=sqrt(1/2 J2)."""
    if dev_stress is None:
        dev_stress = get_dev_stress(stress, pressure, dim)
    return jnp.sqrt(0.5 * jnp.trace(dev_stress @ dev_stress.T))


def get_scalar_shear_stress_stack(
    stress_stack: jax.Array, dev_stress_stack=None, pressure_stack=None, dim=3
) -> chex.Array:
    """Get the shear stress tau=sqrt(1/2 J2) from a stack of stress tensors."""
    if dev_stress_stack is None:
        dev_stress_stack = get_dev_stress_stack(stress_stack, pressure_stack, dim)
    vmap_get_scalar_shear_stress = jax.vmap(
        get_scalar_shear_stress, in_axes=(0, 0, None, None)
    )

    return vmap_get_scalar_shear_stress(
        stress_stack, dev_stress_stack, pressure_stack, dim
    )


def get_volumetric_strain(strain: jax.Array) -> jnp.float32:
    "Get compressive positive volumetric strain."
    return -jnp.trace(strain)


def get_volumetric_strain_stack(strain_stack: jax.Array) -> chex.Array:
    """Get compressive positive volumetric strain from a stack strain tensors."""
    vmap_get_volumetric_strain = jax.vmap(get_volumetric_strain, in_axes=(0))
    return vmap_get_volumetric_strain(strain_stack)


def get_dev_strain(strain: jax.Array, volumetric_strain=None, dim=3) -> jnp.float32:
    """Get deviatoric strain tensor."""
    if volumetric_strain is None:
        volumetric_strain = get_volumetric_strain(strain)
    return strain + (1.0 / dim) * jnp.eye(3) * volumetric_strain


def get_dev_strain_stack(
    strain_stack: jax.Array, volumetric_strain_stack=None, dim=3
) -> chex.Array:
    """Get deviatoric strain tensor from a stack of strain tensors."""
    if volumetric_strain_stack is None:
        volumetric_strain_stack = get_volumetric_strain_stack(strain_stack)
    vmap_get_dev_strain = jax.vmap(get_dev_strain, in_axes=(0))
    return vmap_get_dev_strain(strain_stack, volumetric_strain_stack)


def get_scalar_shear_strain(
    strain: jax.Array = None, dev_strain=None, volumetric_strain=None, dim=3
) -> jnp.float32:
    """Get scalar shear strain gamma = sqrt(1/2 trace(e_dev @ e_dev.T))."""
    if dev_strain is None:
        dev_strain = get_dev_strain(strain, volumetric_strain, dim)

    return jnp.sqrt(0.5 * jnp.trace(dev_strain @ dev_strain.T))


def get_scalar_shear_strain_stack(
    strain_stack: jax.Array, dev_strain_stack=None, volumetric_strain_stack=None, dim=3
) -> chex.Array:
    """Get scalar shear strain from a stack of strain tensors."""
    vmap_get_scalar_shear_strain = jax.vmap(
        get_scalar_shear_strain, in_axes=(0, 0, 0, None)
    )

    return vmap_get_scalar_shear_strain(
        strain_stack, dev_strain_stack, volumetric_strain_stack, dim
    )


def get_KE(mass: jnp.float32, velocity: jnp.float32) -> jnp.float32:
    """Get kinetic energy."""
    return 0.5 * mass * jnp.sum(velocity**2)


def get_KE_stack(masses: jax.Array, velocities: jax.Array) -> chex.Array:
    """Get kinetic energy from a stack of masses and velocities."""
    vmap_get_KE = jax.vmap(get_KE, in_axes=(0, 0))
    return vmap_get_KE(masses, velocities)


def get_inertial_number(pressure, dgamma_dt, p_dia, rho_p) -> jnp.float32:
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
    strain: jax.Array,
    elastic_strain: jax.Array,
):
    """Get the plastic strain."""
    return strain - elastic_strain


def get_plastic_strain_stack(
    strain_stack: jax.Array,
    elastic_strain_stack: jax.Array,
):
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
    """Volume fraction to void ratio."""
    return (1.0 - phi) / phi


def phi_to_e_stack(phi_stack):
    """Volume fraction to void ratio from a stack of volume fractions."""
    vmap_phi_to_e = jax.vmap(phi_to_e)
    return vmap_phi_to_e(phi_stack)


def e_to_phi(e):
    """Void ratio to volume fraction."""
    return 1.0 / (1.0 + e)


def e_to_phi_stack(e_stack):
    """Void ratio to volume fraction from a stack of void ratios."""
    vmap_e_to_phi = jax.vmap(e_to_phi)
    return vmap_e_to_phi(e_stack)


def get_sym_tensor(A):
    """Get symmetric part of a tensor."""
    return 0.5 * (A + A.T)


def get_sym_tensor_stack(A_stack):
    """Get symmetric part of a stack of tensors."""
    vmap_get_sym_tensor = jax.vmap(get_sym_tensor)
    return vmap_get_sym_tensor(A_stack)


def get_skew_tensor(A):
    """Get skew-symmetric part of a tensor."""
    return 0.5 * (A - A.T)

def get_skew_tensor_stack(A_stack):
    """Get skew-symmetric part of a stack of tensors."""
    vmap_get_skew_tensor = jax.vmap(get_skew_tensor)
    return vmap_get_skew_tensor(A_stack)


def get_phi_from_L(L, phi_prev, dt):
    """Get volume fraction from velocity gradient."""
    deps = get_sym_tensor(L)*dt
    deps_v = get_volumetric_strain(deps)
    phi_next = phi_prev / (1.0 - deps_v)
    return phi_next

def get_e_from_bulk_density(absolute_density, bulk_density):
    """Get void ratio from absolute and bulk density."""
    return absolute_density / bulk_density


def get_phi_from_bulk_density(absolute_density, bulk_density):
    """Get volume fraction from absolute and bulk density."""
    e = get_e_from_bulk_density(absolute_density, bulk_density)
    return e_to_phi(e)


def get_phi_from_bulk_density_stack(absolute_density_stack, bulk_density_stack):
    """Get volume fraction from a stack of absolute and bulk densities."""
    vmap_get_phi_from_bulk_density = jax.vmap(
        get_phi_from_bulk_density, in_axes=(None, 0)
    )
    return vmap_get_phi_from_bulk_density(absolute_density_stack, bulk_density_stack)

def get_hencky_strain(F):
    u,s,vh = jnp.linalg.svd(F)

    eps = jnp.zeros((3,3)).at[[0,1,2],[0,1,2]].set(jnp.log(s))
    return eps, u, vh

def get_hencky_strain_stack(F_stack):
    vmap_get_hencky = jax.vmap(get_hencky_strain, in_axes=(0))
    return vmap_get_hencky(F_stack)
    