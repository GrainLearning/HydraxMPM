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

dt = 0.0001
store_every = 500

benchmark = pm.MPBenchmark.create_isochoric_shear(
    material=material,
    total_time=200,
    dt=dt,
    target_range=(0.0, 0.1),
    phi_ref=0.8,
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


eps_v = pm.get_volumetric_strain_stack(strain_stack)


eps_v_gamma_plot = pm.PlotHelper(
    x=gamma_stack,
    y=eps_v,
    xlabel="$\gamma$ [-]",
    ylabel="$\epsilon_v$ [-]",
)

e_stack = pm.phi_to_e_stack(phi_stack)
e_lnp_plot = pm.PlotHelper(
    x=pressure_stack, y=e_stack, xlabel="ln $p$ [-]", ylabel="$e$ [-]", xlogscale=True
)


p_c_p_plot = pm.PlotHelper(
    x=pressure_stack,
    y=p_c_stack,
    xlabel="$p$ [Pa]",
    ylabel="$p_c$ [Pa]",
    xlim=[0, None],
    ylim=[0, None],
)

fig, axes = pm.make_plots(
    [
        qp_plot,
        q_gamma_plot,
        deps_v_p_dgamma_p_plot,
        eps_v_gamma_plot,
        e_lnp_plot,
        p_c_p_plot,
    ]
)
# plot first and last yield surface
pc0 = p_c_stack.at[0].get()
axes[0] = pm.materials.modifiedcamclay.plot_yield_surface(
    axes[0], p_range=(1, pc0, 100), M=material.M, p_c=pc0
)

pcx = p_c_stack.at[-1].get()
axes[0] = pm.materials.modifiedcamclay.plot_yield_surface(
    axes[0], p_range=(1, pcx, 100), M=material.M, p_c=pcx
)


pc_max = max(pc0, pcx)

csl_x = [0, pc_max]
csl_y = [0, (lambda p: material.M * p)(pc_max)]
axes[0].plot(
    [0, pc_max], [0, (lambda p: material.M * p)(pc_max)], ls=(0, (5, 1)), color="red"
)

axes[0].set_xlim(csl_x[0], csl_x[1])

axes[0].set_ylim(csl_y[0], csl_y[1])

plt.tight_layout()
plt.show()
