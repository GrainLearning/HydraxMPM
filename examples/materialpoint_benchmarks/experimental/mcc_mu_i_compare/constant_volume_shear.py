import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pymudokon as pm

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# loading conditions

load_steps = 50000


# total_time_list = [50]
total_time_list = [100, 200]

phi_list = [0.65, 0.655, 0.66]
# phi_list = [0.636, 0.638, 0.640, 0.642, 0.644,0.646
# phi_list = jnp.linspace(0.63,0.66,5)

total_shear_strain = 1.0

store_every = 10

# common params
phi_c = 0.648
mu_s = 0.3819

# mu I
mu_d = 0.645
rho_p = 2000
d = 0.0053
I_phi = 0.3
# I_phi = 0.01

I_0 = 0.279

mu_d = 0.645

# modified cam clay
M_s = jnp.sqrt(3) * mu_s
M_d = jnp.sqrt(3) * mu_d
nu = 0.2
R = 1
lam = 0.01
kap = 0.005

start_time = time.time()
fig_ax_set1 = pm.make_plots()
fig_ax_set2 = pm.make_plots()
fig_ax_set3 = pm.make_plots()

colors = ["blue", "red", "black"]
linestyles = ["--", "-"]


for phi_ref in phi_list:
    # modified cam clay
    p_ref_mcc = pm.ModifiedCamClay.get_p_ref_phi(phi_ref, phi_c, lam, kap)

    mrm = pm.MCC_MRM.create(
        nu=nu,
        M=M_s,
        R=1,
        lam=lam,
        kap=kap,
        Vs=1.0,
        phi_c=phi_c,
        I_phi=I_phi,
        d0=d,
        rho_p=rho_p,
        M_d=M_d,
        I_0=I_0,
        stress_ref_stack=-p_ref_mcc * jnp.eye(3).reshape((1, 3, 3)),
    )

    # mu I rheology
    mu_i = pm.MuI.create(
        mu_s=mu_s,
        mu_d=mu_d,
        I_0=I_0,
        phi_c=phi_c,
        I_phi=I_phi,
        rho_p=rho_p,
        d=d,
    )

    mcc = pm.ModifiedCamClay.create(
        nu=nu,
        M=M_s,
        R=1,
        lam=lam,
        kap=kap,
        Vs=1.0,
        stress_ref_stack=-p_ref_mcc * jnp.eye(3).reshape((1, 3, 3)),
    )

    I_ref = pm.materials.mu_i_rheology.get_I_phi(
        phi_ref,
        phi_c,
        I_phi,
    )

    for mi, material in enumerate([mcc, mrm]):
        for ti, total_time in enumerate(total_time_list):
            plot_helper_args = {"ls": linestyles[ti], "color": colors[mi]}

            dt = total_time / load_steps
            dgamma_dt_ref = (total_shear_strain / load_steps) / dt
            print(type(material))

            if isinstance(material, pm.MuI):
                p_ref = pm.materials.mu_i_rheology.get_pressure(
                    dgamma_dt_ref,
                    I_ref,
                    material.d,
                    material.rho_p,
                )
            else:
                p_ref = pm.get_pressure_stack(material.stress_ref_stack).at[0].get()

            print(f"{dt=} {dgamma_dt_ref=} {p_ref=}")

            stress_ref = -p_ref * jnp.eye(3)

            benchmark = pm.MPBenchmark.create_volume_control_shear(
                material=material,
                total_time=total_time,
                dt=dt,
                x_range=(dgamma_dt_ref, dgamma_dt_ref),
                y_range=(0.0, 0.0),
                stress_ref=stress_ref,
                phi_ref=phi_ref,
                store_every=store_every,
                output=("stress", "F", "L", "phi"),
            )
            benchmark = benchmark.run()

            (stress_stack, F_stack, L_stack, phi_stack) = benchmark.accumulated
            print("--- %s seconds ---" % (time.time() - start_time))

            fig_ax_set1 = pm.plot_set1(
                stress_stack,
                phi_stack,
                L_stack,
                fig_ax=fig_ax_set1,
                plot_helper_args=plot_helper_args,
            )
            fig_ax_set2 = pm.plot_set2(
                stress_stack,
                L_stack,
                F_stack,
                fig_ax=fig_ax_set2,
                plot_helper_args=plot_helper_args,
            )
            fig_ax_set3 = pm.plot_set3(
                stress_stack,
                phi_stack,
                L_stack,
                F_stack,
                benchmark.get_time_stack(),
                fig_ax=fig_ax_set3,
                plot_helper_args=plot_helper_args,
            )


plt.show()
