# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp

from jaxtyping import Float, Array
from typing import Any, Tuple

def get_double_contraction(A, B):
    return jnp.trace(A @ B.T)


def get_double_contraction_stack(A_stack, B_stack):
    return jax.vmap(get_double_contraction)(A_stack, B_stack)


def get_pressure(stress):
    """Get compression positive pressure from the cauchy stress tensor.
    (Plane strain)

    $$
    p = -\\mathrm{trace} ( \\boldsymbol \\sigma ) / \\mathrm{dim}
    $$

    Args:
        stress: Cauchy stress tensor
        dim: Dimension. Defaults to 3.

    Returns:
        pressure
    """
    return -(1 / 3) * jnp.trace(stress)


def get_pressure_stack(stress_stack, dim: int = 3):
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
    """Get the scalar trx shear stress from the Cauchy stress tensor.

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


def get_q_trx_stack(
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


def get_KE(mass, velocity):
    """Get kinetic energy."""
    return 0.5 * mass * jnp.dot(velocity, velocity)


def get_KE_stack(masses, velocities):
    """Get kinetic energy from a stack of masses and velocities."""
    vmap_get_KE = jax.vmap(get_KE, in_axes=(0, 0))
    return vmap_get_KE(masses, velocities)


def get_inertial_number(pressure, dgamma_dt, p_dia, rho_p):
    """Get inertial number (e.g., mu(I) rheology).

    Microscopic pressure time scale over macroscopic shear rate timescale

    I=Tp/T_dot_gamma

    Args:
        pressure: hydrostatic pressure
        dgamma_dt: scalar shear strain rate
        p_dia: particle diameter
        rho_p: particle density [kg/m^3]
    """
    # Add small value to pressure to avoid sqrt(0) and division by zero
    p_safe = jnp.maximum(pressure, 1e-6)
    return (dgamma_dt * p_dia) / jnp.sqrt(p_safe / rho_p)


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
    """Get the plastic strain from additive decomposition."""
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
    v = 1.0/phi
    return  v-1


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
    v = 1.0 + e
    return 1.0 / v


def e_to_phi_stack(e_stack):
    """Vectorized version of [e_to_phi][utils.math_helpers.e_to_phi]
    for a solid volume fraction.

    Args:
        e_stack (chex.Array): void ratio stack

    Returns:
        (jnp.float32): solid volume fraction stack
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
    
    # Avoid log(0) numerical issues
    s = jnp.clip(s, 1e-12, None)
    
    # 1. Principal Strains
    log_s = jnp.log(s)
    
    # 2. Rotate back to global frame
    # Spatial Hencky Strain: eps = U @ diag(ln(s)) @ U.T
    eps_principal = jnp.diag(log_s)
    eps_spatial = u @ eps_principal @ u.T

    return eps_spatial


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


def precondition_from_lithostatic(
        density_stack: Float[Array, "num_points 1"],
        depth_stack: Float[Array, "num_points 1"],
        gravity,
        k0 = 0.5,

    ):
    """
    Initializes the state of the model based on lithostatic lithostatic gravity loading and input density.
    """
    density_stack = density_stack.squeeze()
    depth_stack = depth_stack.squeeze()

    # calculate vertical stress (lithostatic)
    # sigma_v = rho * g * z
    sigma_v_stack = density_stack * gravity * depth_stack
    
    # calculate horizontal stress (K0 assumption)
    sigma_h_stack = k0 * sigma_v_stack
    
    # convert to invariants (p, q)
    # assuming triaxial symmetry (sigma_x = sigma_y = sigma_h)
    p_stack = (sigma_v_stack + 2.0 * sigma_h_stack) / 3.0
    q_stack = jnp.abs(sigma_v_stack - sigma_h_stack)
    
    return p_stack.reshape(-1,1), q_stack.reshape(-1,1)


def reconstruct_stress_from_triaxial(
        p_stack,
        q_stack,
    ):
    """
    Reconstructs the stress tensor from triaxial stress invariants (p, q).
    """
    p_stack = p_stack.squeeze()
    q_stack = q_stack.squeeze()


    sig_v = p_stack + (2.0/3.0) * q_stack
    sig_h = p_stack - (1.0/3.0) * q_stack

    def make_tensor(sv, sh):
        return jnp.diag(jnp.array([-sh, -sh, -sv])) # Assuming Z is vertical
            
    stress0_stack = jax.vmap(make_tensor)(sig_v, sig_h)

    return stress0_stack




def safe_inv_scalar_clamped(d, gradient_clip_val =1e6):
    """
    Computes 1/d with gradient clipping.
    Works for both Forward (jacfwd) and Reverse (grad) modes automatically.
    """
    # 1. Primal Safety: Avoid division by zero
    d_safe = d + 1e-20 * jnp.where(d >= 0, 1.0, -1.0)
    inv_d = 1.0 / d_safe
    
    # 2. Gradient Clipping Trick so the derivative does not explode.
    true_grad = -1.0 / (d_safe * d_safe)
 
    grad_mag = jnp.abs(true_grad)
    scale = jnp.minimum(1.0, gradient_clip_val / (grad_mag + 1e-12))
    
    effective_grad = true_grad * scale
    
    # 3. Construct the output with modified gradients
    # Use stop_gradient to treat 'effective_grad' as a constant slope.
    # Formula: y = x * m + (y_true - x * m)
    # Value: y_true
    # Derivative: m
    slope = jax.lax.stop_gradient(effective_grad)
    res = d * slope + jax.lax.stop_gradient(inv_d - d * slope)
    
    return res

def inv_2x2_robust(m, gradient_clip_val=1e6):
    """
    Inverts a 2x2 matrix using the universal robust scalar inverse.
    """
    a, b = m[0, 0], m[0, 1]
    c, d = m[1, 0], m[1, 1]
    
    det = a * d - b * c
    
    # Use the safe scalar inverse for the determinant
    inv_det = safe_inv_scalar_clamped(det, gradient_clip_val)
    
    inv = jnp.array([
        [d, -b],
        [-c, a]
    ]) * inv_det
    return inv

def safe_norm(x, eps=1e-12):
    """
    Computes euclidean norm safely for AutoDiff.
    Prevents NaN gradients when x is the zero vector.
    """
    return jnp.sqrt(jnp.sum(x**2) + eps)


def quaternion_rotate(q, v):
    """
    Rotate vector v by quaternion q.
    q: [w, x, y, z]
    v: [x, y, z]
    """
    q_vec = q[1:]
    uv = jnp.cross(q_vec, v)
    uuv = jnp.cross(q_vec, uv)
    return v + 2 * (q[0] * uv + uuv)

def quaternion_inv(q):
    """Inverse of unit quaternion."""
    return jnp.array([q[0], -q[1], -q[2], -q[3]])

def rotation_2d(theta,v):
    """Returns a 2D rotation matrix for angle theta."""
    c, s = jnp.cos(theta), jnp.sin(theta)
    x_new =  c * v[0] - s * v[1]
    y_new =  s * v[0] + c * v[1]
    return jnp.stack([x_new, y_new])


def rotation_2d_inv(theta,v):
    """Returns the inverse 2D rotation matrix for angle theta."""
    c, s = jnp.cos(theta), jnp.sin(theta)
    x_new =  c * v[0] + s * v[1]
    y_new = -s * v[0] + c * v[1]
    return jnp.stack([x_new, y_new])


def integrate_quaternion(q, w, dt):
    """
    Integrates a unit quaternion with an angular velocity vector.

    """
    # create quaternion from angular velocity: [0, wx, wy, wz]
    w_quat = jnp.concatenate([jnp.array([0.0]), w])
    
    # Quaternion multiplication: dq/dt = 0.5 * w_quat * q

    # q_new = q + 0.5 * w_quat * q * dt
    # Explicit multiplication for w_quat * q
    # w_quat = [0, bx, by, bz], q = [aw, ax, ay, az]    
    bx, by, bz = w
    aw, ax, ay, az = q
    
    dq = 0.5 * dt * jnp.array([
        -bx*ax - by*ay - bz*az,
         bx*aw + by*az - bz*ay,
        -bx*az + by*aw + bz*ax,
         bx*ay - by*ax + bz*aw
    ])

    q_new = q + dq
    
    # Normalize to prevent drift
    return q_new / (jnp.linalg.norm(q_new) + 1e-12)

def rotate_2d_update(theta, omega, dt):
    return theta + omega * dt