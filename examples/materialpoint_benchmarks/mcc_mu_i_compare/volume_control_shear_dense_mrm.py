import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import pymudokon as pm
import time

jax.config.update("jax_platform_name", "cpu")

# loading conditions

load_steps = 500000

total_time_list = [1.0,2.0,4.0]

total_shear_strain =1.0

store_every = 500

total_volume_strain = 0.03


# common params
phi_c = 0.648
mu_s = 0.3819

# mu I
mu_d = 0.645
rho_p = 2000
d = 0.0053
I_phi=0.3
# I_phi = 0.01
I_0 = 0.279
mu_d = 0.645

# modified cam clay
M_s = jnp.sqrt(3)*mu_s
M_d = jnp.sqrt(3)*mu_d

nu=0.2
R=1
lam=0.01
kap=0.005

# Reference conditions and create material
phi_ref = 0.63


start_time = time.time()

p_ref_mcc = pm.ModifiedCamClay.get_p_ref_phi(
    phi_ref,phi_c,lam,kap
)

mcc = pm.MCC_MRM.create(
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
    stress_ref_stack=-p_ref_mcc*jnp.eye(3).reshape((1,3,3))
)

mu_i = pm.MuI.create(
    mu_s=mu_s,
    mu_d=mu_d,
    I_0=I_0,
    phi_c=phi_c,
    I_phi=I_phi,
    rho_p=rho_p,
    d=d,
)

I_ref = pm.materials.mu_i_rheology.get_I_phi(
    phi_ref,
    phi_c,
    I_phi,
)

fig_ax_set1 = pm.make_plots()
fig_ax_set2 = pm.make_plots()
fig_ax_set3 = pm.make_plots()

colors = ["blue","green","red"]
linestyles = ["--","-"]

for mi,material in enumerate([mcc,mu_i]):
    for ti, total_time in enumerate(total_time_list):
        
        plot_helper_args={"ls":linestyles[mi],"color":colors[ti]}
        
        dt = total_time/load_steps
        dgamma_dt_ref = (total_shear_strain/load_steps)/dt
        deps_v_dt_ref = (total_volume_strain/load_steps)/dt
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

        print(f"{dt=} {dgamma_dt_ref=} {deps_v_dt_ref=} {p_ref=}")
        
        stress_ref = -p_ref * jnp.eye(3)

        benchmark = pm.MPBenchmark.create_volume_control_shear(
            material=material,
            total_time=total_time,
            dt=dt,
            x_range=(dgamma_dt_ref, dgamma_dt_ref),
            y_range=(deps_v_dt_ref, deps_v_dt_ref),
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
            L_stack,
            fig_ax= fig_ax_set1,
            plot_helper_args = plot_helper_args
            )
        fig_ax_set2 = pm.plot_set2(
            stress_stack,
            L_stack,
            F_stack,
            fig_ax= fig_ax_set2,
            plot_helper_args = plot_helper_args
        )
        fig_ax_set3 = pm.plot_set3(
            stress_stack,
            phi_stack,
            L_stack,
            F_stack,
            benchmark.get_time_stack(),
            fig_ax= fig_ax_set3,
            plot_helper_args = plot_helper_args
        )


plt.show()
