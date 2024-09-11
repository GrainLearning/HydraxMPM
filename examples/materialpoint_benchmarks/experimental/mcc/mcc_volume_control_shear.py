import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import pymudokon as pm
import time

jax.config.update("jax_platform_name", "cpu")


start_time = time.time()


stress_ref = -1e5 * jnp.eye(3).reshape(1, 3, 3)

material = pm.ModifiedCamClay.create(
    nu=0.2, M=0.5, R=1, lam=0.01, kap=0.005, Vs=1.0, stress_ref_stack=stress_ref
)


# Reference conditions
phi_ref = 0.8
dgamma_dt_ref = 0.001


dt = 0.00001
store_every = 500

benchmark = pm.MPBenchmark.create_volume_control_shear(
    material=material,
    total_time=5,
    dt=dt,
    x_range=(dgamma_dt_ref, dgamma_dt_ref),
    y_range=(0, 0.05),
    stress_ref=stress_ref.reshape(3, 3),
    phi_ref=phi_ref,
    store_every=store_every,
    output=("stress", "F", "L", "phi"),
)
benchmark = benchmark.run()

(
    stress_stack,
    F_stack,
    L_stack,
    phi_stack
) = benchmark.accumulated
print("--- %s seconds ---" % (time.time() - start_time))

fig_ax_set1 = pm.plot_set1(
    stress_stack,
    phi_stack,
    L_stack)
fig_ax_set2 = pm.plot_set2(
    stress_stack,
    L_stack,
    F_stack
)
fig_ax_set3 = pm.plot_set3(
    stress_stack,
    phi_stack,
    L_stack,
    F_stack,
    benchmark.get_time_stack()
)

plt.show()
