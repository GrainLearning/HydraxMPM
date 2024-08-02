import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import pymudokon as pm
import time

jax.config.update("jax_platform_name", "cpu")


start_time = time.time()

stress_ref = -1e5 * jnp.eye(3).reshape(1, 3, 3)

material = pm.ModifiedCamClay.create(
    nu=0.2, M=0.5, R=5, lam=0.01, kap=0.005, Vs=1.0, stress_ref=stress_ref
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
        "eps_e",
        "p_c",
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

strain_stack = pm.get_small_strain_stack(F_stack)
pressure_stack = pm.get_pressure_stack(stress_stack)
q_stack = pm.get_q_vm_stack(stress_stack)

fig, ax, color = pm.make_plot_set1(
    stress_stack=stress_stack,
    strain_stack=strain_stack,
    volume_fraction_stack=phi_stack,
    internal_variables=(
        pressure_stack,
        p_c_stack,
    ),
    internal_variables_labels=(r"$p$ [Pa]", r"$p_c$ [Pa]"),
    eps_e_stack=eps_e_stack,
)

pc0 = p_c_stack.at[0].get()
ax[0, 0] = pm.materials.modifiedcamclay.plot_yield_surface(
    ax[0, 0], p_range=(1, pc0, 100), M=material.M, p_c=pc0
)


pcx = p_c_stack.at[-1].get()
ax[0, 0] = pm.materials.modifiedcamclay.plot_yield_surface(
    ax[0, 0], p_range=(1, pcx, 100), M=material.M, p_c=pcx
)

pc_max = max(pc0, pcx)
ax[0, 0] = pm.plot_csl(ax[0, 0], (0, pc_max), material.M)
ax[0, 0].set_xlim(0, pc_max)

plt.tight_layout()
plt.show()
