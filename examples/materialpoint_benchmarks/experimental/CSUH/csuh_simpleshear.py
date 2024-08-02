import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import pymudokon as pm
import time


raise NotImplementedError("This example is not working yet")

jax.config.update("jax_platform_name", "cpu")


start_time = time.time()

OCR = 1
stress_ref = -89020 / OCR * jnp.eye(3).reshape(1, 3, 3)

# Toyoura sand
material = pm.CSUHModel.create(
    nu=0.3,
    M=1.25,
    R=OCR,
    lam=0.135,
    kap=0.04,
    V0=1.934,
    N=1.973,
    # X=0.4,
    X=1.8,
    m_state=1.8,
    # m_state=9,
    stress_ref=stress_ref,
)


dt = 0.0001
store_every = 500

benchmark = pm.MPBenchmark.create_simpleshear(
    material=material,
    total_time=200,
    dt=dt,
    target_range=(0.0, 0.1),
    volume_fraction=0.8,
    output=(
        "stress",
        "strain",
        "strain_rate",
        "volume_fraction",
        "eps_e",
        "H",
    ),
    store_every=store_every,
)

benchmark = benchmark.run()

(
    stress_stack,
    strain_stack,
    strain_rate_stack,
    volume_fraction_stack,
    eps_e_stack,
    H_stack,
) = benchmark.accumulated
print("--- %s seconds ---" % (time.time() - start_time))

pressure_stack = pm.get_pressure_stack(stress_stack)
q_stack = pm.get_q_vm_stack(stress_stack)

eps_p_stack = strain_stack - eps_e_stack
deps_p_stack = jnp.diff(eps_p_stack, axis=0)
deps_p_v_stack = pm.get_volumetric_strain_stack(deps_p_stack)
fig, ax, color = pm.make_plot_set1(
    stress_stack=stress_stack,
    strain_stack=strain_stack,
    volume_fraction_stack=volume_fraction_stack,
    internal_variables=(
        deps_p_v_stack,
        H_stack.at[:-1].get(),
    ),
    internal_variables_labels=(r"$d\epsilon_v^p$ [-]", r"$H$ [-]"),
    eps_e_stack=eps_e_stack,
)

# pc0 = p_c_stack.at[0].get()
# ax[0, 0] = pm.materials.modifiedcamclay.plot_yield_surface(
#     ax[0, 0], p_range=(1, pc0, 100), M=material.M, p_c=pc0
# )


# pcx = p_c_stack.at[-1].get()
# ax[0, 0] = pm.materials.modifiedcamclay.plot_yield_surface(
#     ax[0, 0], p_range=(1, pcx, 100), M=material.M, p_c=pcx
# )

pc_max = max(0, 150000)
ax[0, 0] = pm.plot_csl(ax[0, 0], (0, pc_max), material.M)
ax[0, 0].set_xlim(0, pc_max)

plt.tight_layout()
plt.show()
