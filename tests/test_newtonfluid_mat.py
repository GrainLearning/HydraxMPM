import equinox as eqx
import jax.numpy as jnp
import numpy as np

import hydraxmpm as hdx


def test_create():
    """Unit test the initialization of the Newton Fluid material."""
    model = hdx.NewtonFluid(K=2.0 * 10**6, viscosity=0.2)
    assert isinstance(model, hdx.NewtonFluid)

def test_create_state():

    material_points = hdx.MaterialPointState.create(
        position_stack=jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        mass_stack =jnp.array([1000.0, 2000.0]), 
        volume_stack=jnp.array([1.0, 2.0]),
        pressure_stack=jnp.array([1000, 1000])
    )

    model = hdx.NewtonFluid(K=2.0 * 10**6, viscosity=0.2, beta=7.0)
    
    # Test with 0 initial pressure
    law_state = model.create_state_from_density(
        density_stack=material_points.mass_stack / material_points.volume_stack,
        pressure_stack=jnp.array([0.0, 0.0])
        )
    
    expected_density = material_points.mass_stack / material_points.volume_stack
    np.testing.assert_allclose(law_state.density_ref_stack, expected_density)
    

    p_init = 1e5
    law_state = model.create_state_from_density(
        density_stack=material_points.mass_stack / material_points.volume_stack,
        pressure_stack=jnp.array([p_init, p_init])
        )
    
    # Test with non-zero initial pressure
    law_state = model.create_state_from_density(
        density_stack=material_points.mass_stack / material_points.volume_stack,
        pressure_stack=jnp.array([p_init, p_init])
        )
    
    factor = (p_init / model.K + 1.0) ** (1.0 / model.beta)
    expected_density_p = expected_density / factor
    
    np.testing.assert_allclose(law_state.density_ref_stack, expected_density_p)

def test_update_stress_3d():
    model = hdx.NewtonFluid(K=2.0 * 10**6, viscosity=0.001, beta=7.0)

    # rho_rho_0 = 1400 / 1390  # volume0/volume
    # We want ratio = density / density_ref = 1400 / 1390
    
    mass = 1400.0
    volume = 1390.0
    density_ref = 1.0
    
    L = jnp.eye(3)
    dt = 0.0 # Unused in _update_stress

    stress = model._update_stress(L, mass, volume, density_ref, dt)

    expected_stress = jnp.eye(3) * -102920.05

    np.testing.assert_allclose(stress, expected_stress, rtol=1e-3)

