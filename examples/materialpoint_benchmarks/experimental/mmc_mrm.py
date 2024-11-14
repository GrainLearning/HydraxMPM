import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pymudokon as pm

jax.config.update("jax_platform_name", "cpu")


start_time = time.time()

pressure = 0
stress_ref = -pressure * jnp.eye(3).reshape(1, 3, 3)

material = pm.MCC_MRM.create(
    nu=0.2,
    M=0.5,
    R=1,
    lam=0.01,
    kap=0.005,
    Vs=1.0,
    phi_c=0.634,
    I_phi=3.28,
    d0=1,
    rho_p=1,
    M_d=0.9,
    I_0=0.2,
    # I_0=0.0001,
    stress_ref_stack=stress_ref,
)

phi_ref = 0.634 * (1 + pressure / 2) ** 0.01

print(f"phi_ref: {phi_ref}")

dt = 0.0001
store_every = 500

benchmark = pm.MPBenchmark.create_isochoric_shear(
    material=material,
    total_time=200,
    # total_time=10,
    dt=dt,
    target_range=(0.0, 0.1),
    phi_ref=phi_ref,
    output=(
        "stress",
        "F",
        "L",
        "phi",
        "eps_e_stack",
        "p_c_stack",
    ),
    # pressure_control=3e5,
    store_every=store_every,
)

benchmark = benchmark.run()

(
    stress_stack,
    F_stack,
    L_stack,
    phi_stack,
    eps_e_stack,
    p_c_stack,
) = benchmark.accumulated
print("--- %s seconds ---" % (time.time() - start_time))


# q-p plot
pressure_stack = pm.get_pressure_stack(stress_stack)
q_stack = pm.get_q_vm_stack(stress_stack)

qp_plot = pm.PlotHelper(
    x=pressure_stack,
    y=q_stack,
    xlabel="$p$ [Pa]",
    ylabel="$q$ [Pa]",
    xlim=[0, None],
    ylim=[0, None],
)

# q-gamma plot
strain_stack = pm.get_small_strain_stack(F_stack)

gamma_stack = pm.get_scalar_shear_strain_stack(strain_stack)

q_gamma_plot = pm.PlotHelper(
    x=gamma_stack,
    y=q_stack,
    xlabel="$\gamma$ [-]",
    ylabel="$q$ [Pa]",
)

# deps_v_p vs dgamma_p
eps_p_stack = strain_stack - eps_e_stack
deps_p_stack = jnp.diff(eps_p_stack, axis=0)

deps_p_v_stack = pm.get_volumetric_strain_stack(deps_p_stack)

dgamma_p_stack = pm.get_scalar_shear_strain_stack(deps_p_stack)

deps_v_p_dgamma_p_plot = pm.PlotHelper(
    x=dgamma_p_stack,
    y=deps_p_v_stack,
    xlabel="$d\gamma^p$ [-]",
    ylabel="$d\epsilon_v^p$ [-]",
)
pm.make_plots([qp_plot, q_gamma_plot, deps_v_p_dgamma_p_plot])

plt.show()
