import jax.numpy as jnp

import pymudokon as pm


def test_mix_control():
    L = jnp.eye(3) * 0.1

    L_stack = jnp.array([L] * 4)
    stress_ref = jnp.array([jnp.eye(3)]) * -1e5
    material = pm.ModifiedCamClay.create(
        nu=0.2, M=1.2, R=2.0, lam=0.8, kap=0.1, Vs=2, stress_ref_stack=stress_ref
    )

    carry, accumulate = pm.mix_control(
        material=material,
        dt=0.001,
        L_control_stack=L_stack,
        phi_ref=0.8,
        stress_ref=stress_ref.squeeze(),
        output=("stress",),
    )
    # test passes if the following line does not raise an exception


# %%
