import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import pymudokon as pm
import time

jax.config.update("jax_platform_name", "cpu")


start_time = time.time()

material = pm.MuI.create(
    mu_s=0.3819,
    mu_d=0.645,
    I_0=0.279,
    phi_c=0.648,
    I_phi=0.5,
    rho_p=2000,
    d=0.0053,
)

# Reference conditions
phi_ref = 0.64

dgamma_dt_ref = 1.0

I = pm.materials.mu_i_rheology.get_I_phi(
    phi_ref,
    material.phi_c,
    material.I_phi,
)

p_ref = pm.materials.mu_i_rheology.get_pressure(
    dgamma_dt_ref,
    I,
    material.d,
    material.rho_p,
)

stress_ref = -p_ref * jnp.eye(3)

dt = 0.00001
store_every = 500

benchmark = pm.MPBenchmark.create_pressure_control_shear(
    material=material,
    total_time=2,
    dt=dt,
    x_range=(dgamma_dt_ref, dgamma_dt_ref),
    y_range=(p_ref, p_ref*100),
    stress_ref=stress_ref,
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
