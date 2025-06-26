import jax.numpy as jnp
import equinox as eqx
from hydraxmpm.forces.particle_damping import ParticleDamping
from hydraxmpm.material_points.material_points import MaterialPoints


def test_particle_damping_basic():
    # Create a dummy MaterialPoints with known velocity_stack
    velocity_stack = jnp.array([[1.0, -2.0], [0.5, 0.5]])
    force_stack = jnp.zeros_like(velocity_stack)
    # Minimal required fields for MaterialPoints
    mp = MaterialPoints(
        position_stack=jnp.zeros_like(velocity_stack),
        velocity_stack=velocity_stack,
        force_stack=force_stack,
        mass_stack=jnp.ones(velocity_stack.shape[0]),
        stress_stack=jnp.zeros((velocity_stack.shape[0], 3, 3)),
    )
    alpha = 0.3
    damping = ParticleDamping(alpha=alpha)
    mp_new, _ = damping.apply_on_points(material_points=mp)
    # The force_stack should be -alpha * velocity_stack
    expected = -alpha * velocity_stack
    assert jnp.allclose(mp_new.force_stack, expected), f"Expected {expected}, got {mp_new.force_stack}"
