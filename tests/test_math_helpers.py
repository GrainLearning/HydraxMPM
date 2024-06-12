"""Unit tests for math helper functions."""
import jax.numpy as jnp
import numpy as np

import pymudokon as pm


def test_helpers_energy():
    """Test the helper functions."""
    mass = np.ones(10)
    vel = np.ones((10, 3))

    ke = pm.get_KE(mass, vel)

    np.testing.assert_allclose(ke, np.array([1.5] * 10))


def test_helpers_strain():
    """Test the helper functions."""
    strain = jnp.eye(3)

    strain = strain.at[0, 1].set(1.0) * 2

    volumetric_strain = pm.get_volumetric_strain(strain)

    np.testing.assert_allclose(volumetric_strain, np.array([-6.0]))

    dev_strain = pm.get_dev_strain(strain)

    np.testing.assert_allclose(dev_strain, [[[0.0, 2.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])

    gamma = pm.get_gamma(strain)

    np.testing.assert_allclose(gamma, np.array([4.0]))


def test_helpers_stress():
    """Test the helper functions."""
    stress = jnp.eye(3)

    stress = stress.at[0, 1].set(1.0) * 2

    pressures = pm.get_pressure(stress)

    np.testing.assert_allclose(pressures, np.array([-2.0]))

    s = pm.get_dev_stress(stress)

    np.testing.assert_allclose(s, [[[0.0, 2.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])

    q = pm.get_q_vm(stress)

    np.testing.assert_allclose(q, np.array([2.4494896]))

    tau = pm.get_tau(stress)

    np.testing.assert_allclose(tau, np.array([2.0]))
