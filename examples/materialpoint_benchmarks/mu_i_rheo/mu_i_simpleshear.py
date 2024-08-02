import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import pymudokon as pm
import time

jax.config.update("jax_platform_name", "cpu")


start_time = time.time()

dgamma_dt_start = 0.0001

phi_ref = 0.67

material = pm.MuI.create(
    mu_s=jnp.tan(jnp.deg2rad(20.9)),  # mu_s 0.381..
    mu_d=jnp.tan(jnp.deg2rad(32.76)),  # mu_d 0.643..
    I_0=0.279,  # I0
    phi_c=0.648,  # phi_c
    I_phi=0.5,  # I_phi
    rho_p=2000,  # rho p (kg/m^3)
    d=0.0053,  # d (m)
)

pressure = material.get_p_ref(phi_ref, dgamma_dt_start)

stress_ref = pressure * jnp.eye(3).reshape(1, 3, 3)

dt = 0.0001
store_every = 500

benchmark = pm.MPBenchmark.create_isochoric_shear_rate(
    material=material,
    total_time=200,
    dt=dt,
    target_range=(dgamma_dt_start, 2.0),
    phi_ref=phi_ref,
    output=("stress", "F", "L", "phi"),
    # pressure_control=3e5,
    store_every=store_every,
)

benchmark = benchmark.run()

(stress_stack, F_stack, L_stack, phi_stack) = benchmark.accumulated
print("--- %s seconds ---" % (time.time() - start_time))

strain_stack = pm.get_small_strain_stack(F_stack)
strain_rate_stack = pm.get_strain_rate_from_L_stack(L_stack)
shear_strain_rate_stack = pm.get_scalar_shear_strain_stack(strain_rate_stack)
pressure_stack = pm.get_pressure_stack(stress_stack)
q_stack = pm.get_q_vm_stack(stress_stack)


I_stack = pm.get_inertial_number_stack(
    pressure_stack, shear_strain_rate_stack, material.d, material.rho_p
)


fig, ax, color = pm.make_plot_set1(
    stress_stack=stress_stack,
    strain_stack=strain_stack,
    volume_fraction_stack=phi_stack,
    internal_variables=(
        pressure_stack,
        I_stack,
    ),
    internal_variables_labels=(r"$p$ [Pa]", r"$I$ "),
    eps_e_stack=None,
)


plt.tight_layout()
plt.show()
