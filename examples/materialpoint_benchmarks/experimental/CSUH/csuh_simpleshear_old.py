import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import pymudokon as pm
import time


raise NotImplementedError("This example is not working yet")

import scienceplots

jax.config.update("jax_platform_name", "cpu")

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"font.size": 8})

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
    X=0.4,
    m_state=1.8,
    # m_state=9,
    stress_ref=stress_ref,
)


dt = 0.0001
store_every = 500

benchmark = pm.MPBenchmark.preset_create_simpleshear(
    material=material,
    total_time=200,
    dt=dt,
    target=0.3,
    target_start=0.0,
    volume_fraction=0.8,
    keys=(
        "stress",
        "strain",
        "strain_rate",
        "volume_fraction",
        "H",
        "eps_e",
    ),
    store_every=store_every,
)

carry, accumulated = benchmark.run()
(
    stress_stack,
    strain_stack,
    strain_rate_stack,
    volume_fraction_stack,
    p_c_stack,
    eps_e_stack,
) = accumulated
print("--- %s seconds ---" % (time.time() - start_time))


fig, ax = plt.subplots(2, 2, figsize=(6, 3), dpi=300)

# q-p plot
q_stack = pm.get_q_vm_stack(stress_stack)
p_stack = pm.get_pressure_stack(stress_stack)

ax[0, 0].plot(p_stack, q_stack, ls="-", marker=None)
ax[0, 0].set_xlabel("$p$ [Pa]")
ax[0, 0].set_ylabel("$q$ [Pa]")
# ax[0, 0].set_xlim(0, 2e5)
# ax[0, 0].set_xlim(0, 2e5)

CSL = lambda p, M: M * p

CSL_range = jnp.array([0, 1e5])
ax[0, 0].plot(CSL_range, CSL(CSL_range, material.M), ls="dashdot")

# q-gamma plot
gamma_stack = pm.get_scalar_shear_strain_stack(strain_stack)

ax[0, 1].plot(gamma_stack, q_stack, ls="-", marker=None)
ax[0, 1].set_xlabel("$\gamma$ [-]")
ax[0, 1].set_ylabel("$q$ [Pa]")

eps_p_stack = strain_stack - eps_e_stack

deps_p_stack = jnp.diff(eps_p_stack, axis=0)

deps_p_v_stack = pm.get_volumetric_strain_stack(deps_p_stack)
dgamma_p_stack = pm.get_scalar_shear_strain_stack(deps_p_stack)


ax[1, 0].plot(dgamma_p_stack, deps_p_v_stack)
ax[1, 0].set_ylabel("$d\epsilon_v^p$ [-]")
ax[1, 0].set_xlabel("$d\gamma^p$ [-]")


ax[1, 1].plot(p_c_stack, gamma_stack)
ax[1, 1].set_ylabel("$H$ [-]")
ax[1, 1].set_xlabel("$\gamma$ [-]")

# debugging
# pm.plot_tensor(stress_stack)
# pm.plot_tensor(strain_stack)

plt.tight_layout()
plt.show()
