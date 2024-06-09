"""Unit tests for the isotropic linear elastic material module.

Test and examples on how to use the isotropic linear elastic material
module to setup/update material state

The module contains the following main components:

- TestLinearElastic.test_init:
    Unit test to initialize the isotropic linear elastic material.
- TestLinearElastic.test_vmap_update:
    Unit test for vectorized update of the isotropic linear elastic material.
- TestLinearElastic.test_update_stress:
    Unit test for updating stress and strain for all particles.
- TestLinearElastic.test_solve:
    Unit test for solving the linear elastic material.

"""

import jax
import jax.numpy as jnp
import numpy as np

import pymudokon as pm


def test_create():
    """Test the initialization of the isotropic linear elastic material."""
    material = pm.LinearIsotropicElastic.create(E=1000.0, nu=0.2, num_particles=2, dim=3)

    assert isinstance(material, pm.LinearIsotropicElastic)
    np.testing.assert_allclose(material.E, 1000.0)
    np.testing.assert_allclose(material.nu, 0.2)
    np.testing.assert_allclose(material.G, 416.666667)
    np.testing.assert_allclose(material.K, 555.5555555555557)
    np.testing.assert_allclose(material.eps_e, jnp.zeros((2, 3, 3), dtype=jnp.float32))

def test_vmap_update():
    """Test the vectorized update of the isotropic linear elastic material."""
    # 2D
    material = pm.LinearIsotropicElastic.create(E=1000.0, nu=0.2, num_particles=2, dim=2)

    transform = jnp.eye(2, dtype=jnp.float32) * 0.001
    deps = jnp.stack([transform, transform])
    stress, eps_e = material.vmap_update(
        material.eps_e,  deps, jnp.zeros((2,3,3)), material.G, material.K
    )
    np.testing.assert_allclose(
        stress,
        jnp.array(
            [
                [
                    [1.11111111111, 0.0, 0.0],
                    [0.0, 1.11111111111, 0.0],
                    [0.0, 0.0, 1.11111111111],
                ],
                [
                    [1.11111111111, 0.0, 0.0],
                    [0.0, 1.11111111111, 0.0],
                    [0.0, 0.0, 1.11111111111],
                ],
            ]
        ),
    )

    np.testing.assert_allclose(
        eps_e,
        jnp.array(
            [
                [[0.001, 0.0], [0.0, 0.001]],
                [[0.001, 0.0], [0.0, 0.001]],
            ]
        ),
    )
    # 3D

    material = pm.LinearIsotropicElastic.create(E=1000.0, nu=0.2, num_particles=2, dim=3)
    
    transform = jnp.eye(3, dtype=jnp.float32) * 0.001
    
    deps = jnp.stack([transform, transform])
    
    stress, eps_e = material.vmap_update(
        material.eps_e,  deps, jnp.zeros((2,3,3)), material.G, material.K
    )
    np.testing.assert_allclose(
        stress,
        jnp.array(
            [
                [
                    [1.6666666666, 0.0, 0.0],
                    [0.0, 1.6666666666, 0.0],
                    [0.0, 0.0, 1.6666666666],
                ],
                [
                    [1.6666666666, 0.0, 0.0],
                    [0.0, 1.6666666666, 0.0],
                    [0.0, 0.0, 1.6666666666],
                ],
            ]
        ),
    )

    np.testing.assert_allclose(
        eps_e,
        jnp.array(
            [
                [[0.001, 0.0, 0.0], [0.0, 0.001, 0.0], [0.0, 0.0, 0.001] ],
                [[0.001, 0.0, 0.0], [0.0, 0.001, 0.0], [0.0, 0.0, 0.001] ]
            ]
        ),
    rtol=1e-3
    )

def test_update_stress():
    """Test the update of stress and strain for all particles."""
    particles = pm.Particles.create(positions=jnp.array([[0.0, 0.0], [1.0, 1.0]]))

    particles = particles.replace(velgrads=jnp.stack([jnp.eye(2), jnp.eye(2)]))

    material = pm.LinearIsotropicElastic.create(E=1000.0, nu=0.2, num_particles=2, dim=2)

    particle, material = material.update_stress(particles, 0.001)

    # Runs!

def test_update_benchmark():
    """Test the vectorized update of the isotropic linear elastic material."""
    material = pm.LinearIsotropicElastic.create(E=1000.0, nu=0.2, num_particles=2, dim=2)

    dt = 0.01
    strain_rate_transform = jnp.eye(2, dtype=jnp.float32) * 0.001 / dt
    strain_rate = jnp.stack([strain_rate_transform, strain_rate_transform])
    volumes = np.ones(2)
    stress, material = material.update_stress_benchmark(strain_rate, volumes, dt)
    # Runs!

