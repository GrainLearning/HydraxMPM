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
phi_ref = 0.63
dgamma_dt_ref = 0.0001

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

print(stress_ref)

dt = 0.0001
store_every = 500

benchmark = pm.MPBenchmark.create_isochoric_shear_rate(
    material=material,
    total_time=2,
    dt=dt,
    target_range=(dgamma_dt_ref, 0.1),
    phi_ref=phi_ref,
    output=("stress", "F", "L", "phi"),
    # pressure_control=3e5,
    store_every=store_every,
    stress_ref=stress_ref,
)

benchmark = benchmark.run()

(
    stress_stack,
    F_stack,
    L_stack,
    phi_stack,
    # eps_e_stack,
    # p_c_stack,
) = benchmark.accumulated
print("--- %s seconds ---" % (time.time() - start_time))


# # q-p plot
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


eps_v = pm.get_volumetric_strain_stack(strain_stack)

# print(L_stack[-10:])
eps_v_gamma_plot = pm.PlotHelper(
    x=gamma_stack,
    y=eps_v,
    xlabel="$\gamma$ [-]",
    ylabel="$\epsilon_v$ [-]",
)

e_stack = pm.phi_to_e_stack(phi_stack)
# print(e_stack)
e_lnp_plot = pm.PlotHelper(
    x=pressure_stack, y=e_stack, xlabel="ln $p$ [-]", ylabel="$e$ [-]", xlogscale=True
)


fig, axes = pm.make_plots(
    [
        qp_plot,
        q_gamma_plot,
        eps_v_gamma_plot,
        e_lnp_plot,
    ]
)
# # plot first and last yield surface
# pc0 = p_c_stack.at[0].get()
# axes[0] = pm.materials.modifiedcamclay.plot_yield_surface(
#     axes[0], p_range=(1, pc0, 100), M=material.M, p_c=pc0
# )

# pcx = p_c_stack.at[-1].get()
# axes[0] = pm.materials.modifiedcamclay.plot_yield_surface(
#     axes[0], p_range=(1, pcx, 100), M=material.M, p_c=pcx
# )


# pc_max = max(pc0, pcx)

# csl_x = [0, pc_max]
# csl_y = [0, (lambda p: material.M * p)(pc_max)]
# axes[0].plot(
#     [0, pc_max], [0, (lambda p: material.M * p)(pc_max)], ls=(0, (5, 1)), color="red"
# )

# axes[0].set_xlim(csl_x[0], csl_x[1])

# axes[0].set_ylim(csl_y[0], csl_y[1])

plt.tight_layout()
plt.show()
