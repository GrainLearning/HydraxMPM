"""Unit tests for math helper functions."""

import jax.numpy as jnp
import numpy as np

import pymudokon as pm


def test_strain_helpers():
    strain = -jnp.eye(3) * 0.01
    strain = strain.at[0, 1].set(0.02)
    strain_stack = jnp.stack([strain, strain, strain])

    eps_v = pm.get_volumetric_strain(strain)

    eps_v_stack = pm.get_volumetric_strain_stack(strain_stack)

    eps_dev = pm.get_dev_strain(strain)

    eps_dev_stack = pm.get_dev_strain_stack(strain_stack)

    gamma = pm.get_scalar_shear_strain(strain)

    gamma_stack = pm.get_scalar_shear_strain_stack(strain_stack)

    expected_eps_v = 0.03
    expected_eps_dev = jnp.array([[0.0, 0.02, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    expected_gamma = 0.014142135
    np.testing.assert_allclose(eps_v, expected_eps_v)
    np.testing.assert_allclose(eps_dev, expected_eps_dev)
    np.testing.assert_allclose(gamma, expected_gamma)

    # test stacked versions that use vmap

    np.testing.assert_allclose(eps_v_stack, [expected_eps_v] * 3)
    np.testing.assert_allclose(eps_dev_stack, [expected_eps_dev] * 3)
    np.testing.assert_allclose(gamma_stack, [expected_gamma] * 3)


def test_stress_helpers():
    stress = -jnp.eye(3) * 10000
    stress = stress.at[0, 1].set(20000)
    stress_stack = jnp.stack([stress, stress, stress])

    pressure = pm.get_pressure(stress)

    pressure_stack = pm.get_pressure_stack(stress_stack)

    dev_stress = pm.get_dev_stress(stress)
    dev_stress_stack = pm.get_dev_stress_stack(stress_stack)

    q_vm = pm.get_q_vm(stress)

    q_vm_stack = pm.get_q_vm_stack(stress_stack)

    J2 = pm.get_J2(stress)

    tau = pm.get_scalar_shear_stress(stress)

    expected_pressure = 10000.0
    expected_dev_stress = jnp.array(
        [
            [0.0, 20000.0, 0.0],
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
            ],
        ]
    )
    expected_q_vm = 24494.896
    expected_J2 = 200000000.0
    expected_tau = 14142.136
    np.testing.assert_allclose(pressure, expected_pressure)
    np.testing.assert_allclose(dev_stress, expected_dev_stress)
    np.testing.assert_allclose(q_vm, expected_q_vm)
    np.testing.assert_allclose(J2, expected_J2)
    np.testing.assert_allclose(expected_tau, tau)

    # test stacked versions that use vmap
    np.testing.assert_allclose(pressure_stack, [expected_pressure] * 3)
    np.testing.assert_allclose(dev_stress_stack, [expected_dev_stress] * 3)
    np.testing.assert_allclose(q_vm_stack, [expected_q_vm] * 3)
    np.testing.assert_allclose(J2, [expected_J2] * 3)
    np.testing.assert_allclose(tau, [expected_tau] * 3)


# %%
def test_energy_helpers():
    """Test the helper functions."""
    mass = 1.5
    vel = jnp.array([2.0, 1.0])

    ke = pm.get_KE(mass, vel)

    np.testing.assert_allclose(ke, 3.75)

    mass_stack = jnp.stack([mass, mass, mass])
    vel_stack = jnp.stack([vel, vel, vel])

    pm.get_KE_stack(mass_stack, vel_stack)
    np.testing.assert_allclose(ke, [3.75, 3.75, 3.75])


def test_mu_I_helpers():
    pressure = 10.0
    dgamma_dt = 10
    p_dia = 1
    rho_p = 2000

    I = pm.get_inertial_number(pressure, dgamma_dt, p_dia, rho_p)

    expected_I = 141.42137

    np.testing.assert_allclose(I, expected_I)

    pressure_stack = jnp.stack([pressure, pressure, pressure])
    dgamma_dt_stack = jnp.stack([dgamma_dt, dgamma_dt, dgamma_dt])

    I_stack = pm.get_inertial_number_stack(pressure_stack, dgamma_dt_stack, p_dia, rho_p)

    np.testing.assert_allclose(I_stack, [expected_I] * 3, rtol=1e-2)


test_mu_I_helpers()
# %%


def test_elastoplastic_helpers():
    strain = -jnp.eye(3) * 0.01
    elastic_strain = -jnp.eye(3) * 0.001

    eps_p = pm.get_plastic_strain(strain, elastic_strain)

    eps_p_stack = pm.get_plastic_strain_stack(
        jnp.stack([strain] * 3), jnp.array([elastic_strain] * 3)
    )

    expected_plastic_strain = jnp.array(
        [[-0.009, 0.0, 0.0], [0.0, -0.009, 0.0], [0.0, 0.0, -0.009]]
    )
    np.testing.assert_allclose(eps_p, expected_plastic_strain)
    np.testing.assert_allclose(eps_p_stack, [expected_plastic_strain] * 3)


def test_volumetric_converters():
    F = jnp.eye(3) * 1.1
    L = jnp.eye(3) * 1.1
    phi = 0.5
    e = 0.5

    eps = pm.get_small_strain(F)

    eps_stack = pm.get_small_strain_stack(jnp.stack([F] * 3))

    expected_eps = jnp.array(
        [
            [
                0.10000002,
                0.0,
                0,
            ],
            [
                0.0,
                0.10000002,
                0.0,
            ],
            [0.0, 0.0, 0.10000002],
        ]
    )
    np.testing.assert_allclose(eps, expected_eps)
    np.testing.assert_allclose(eps_stack, [expected_eps] * 3)

    deps = pm.get_strain_rate_from_L(L)
    deps_stack = pm.get_strain_rate_from_L_stack(jnp.stack([L] * 3))

    expected_deps = jnp.array([[1.1, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 1.1]])

    np.testing.assert_allclose(deps, expected_deps)
    np.testing.assert_allclose(deps_stack, [expected_deps] * 3)

    phi2e = pm.phi_to_e(e)
    phi2e_stack = pm.phi_to_e_stack(jnp.array([e] * 3))

    phi2e_expected = 1.0

    np.testing.assert_allclose(phi2e, phi2e_expected)
    np.testing.assert_allclose(phi2e_stack, [phi2e_expected] * 3)

    e2phi = pm.e_to_phi(phi)
    e2phi_stack = pm.e_to_phi_stack(jnp.array([phi] * 3))

    expected_e2phi = 0.6666666666666666
    np.testing.assert_allclose(e2phi, expected_e2phi)
    np.testing.assert_allclose(e2phi_stack, [expected_e2phi] * 3, rtol=1e-3, atol=1e-3)

    D = pm.get_sym_tensor(L)
    D_stack = pm.get_sym_tensor_stack(jnp.stack([L] * 3))

    epected_D_stack = jnp.array([[1.1, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 1.1]])

    np.testing.assert_allclose(D, epected_D_stack)
    np.testing.assert_allclose(D_stack, [epected_D_stack] * 3)

    W = pm.get_skew_tensor(L)
    W_stack = pm.get_skew_tensor_stack(jnp.stack([L] * 3))

    expected_W = jnp.zeros((3, 3))

    np.testing.assert_allclose(W, expected_W)
    np.testing.assert_allclose(W_stack, [expected_W] * 3)

    L2phi = pm.get_phi_from_L(L, phi, 0.1)

    expected_L2phi = 0.5

    np.testing.assert_allclose(L2phi, expected_L2phi)

    bulkdensity2e = pm.get_e_from_bulk_density(1.0, 0.5)

    expected_bulkdensity2e = 2.0

    np.testing.assert_allclose(bulkdensity2e, expected_bulkdensity2e)

    bulkdensity2phi = pm.get_phi_from_bulk_density(1.0, 0.5)
    bulkdensity2phi_stack = pm.get_phi_from_bulk_density_stack(1.0, jnp.array([0.5] * 3))
    expected_bulkdensity2phi = 0.3333333333333333

    np.testing.assert_allclose(bulkdensity2phi, expected_bulkdensity2phi)
    np.testing.assert_allclose(bulkdensity2phi_stack, [expected_bulkdensity2phi] * 3)
