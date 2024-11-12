import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import pymudokon as pm


raise NotImplementedError("This example is not working yet")

jax.config.update("jax_platform_name", "cpu")
plt.style.use("ggplot")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "mathtext.default": "regular",
        "figure.dpi": 100,
        "savefig.dpi": 100,
        "axes.grid": True,
    }
)

start_time = time.time()
OCR = 1
stress_ref = -689020 / OCR * jnp.eye(3).reshape(1, 3, 3)

material = pm.UHModel.create(
    nu=0.2,
    M=0.896,
    R=OCR,
    lam=0.240,
    kap=0.045,
    V0=2.27,
    N=1.973,
    stress_ref=stress_ref,
)


dt = 0.0001
store_every = 500

benchmark = pm.MPBenchmark.preset_create_simpleshear(
    material=material,
    total_time=100.0,
    # total_time=dt * 9,
    dt=dt,
    target=0.5,
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
    # keys=None,
    store_every=store_every,
)
carry, accumulated = benchmark.run()
(
    stress_stack,
    strain_stack,
    strain_rate_stack,
    volume_fraction_stack,
    H_stack,
    eps_e_stack,
) = accumulated
print("--- %s seconds ---" % (time.time() - start_time))


fig, ax = plt.subplots(2, 2, figsize=(8, 4))

# q-p plot
q_stack = pm.get_q_vm_stack(stress_stack)
p_stack = pm.get_pressure_stack(stress_stack)

ax[0, 0].plot(p_stack, q_stack)
ax[0, 0].set_xlabel("$p$")
ax[0, 0].set_ylabel("$q$")
# ax[0, 0].set_xlim(0, 1e5)
# ax[0, 0].set_xlim(0, 1e5)

CSL = lambda p, M: M * p

ax[0, 0].plot(jnp.arange(0, 1e5, 1e4), CSL(jnp.arange(0, 1e5, 1e4), material.M))

# q-gamma plot
gamma_stack = pm.get_scalar_shear_strain_stack(strain_stack)


ax[0, 1].plot(gamma_stack, q_stack, ls="-", marker="s")
ax[0, 1].set_xlabel("$\gamma$")
ax[0, 1].set_ylabel("$q$")
# deps_p - deps_v plot

eps_p_stack = strain_stack - eps_e_stack

deps_p_stack = jnp.diff(eps_p_stack, axis=0)

deps_p_v_stack = pm.get_volumetric_strain_stack(deps_p_stack)
dgamma_p_stack = pm.get_scalar_shear_strain_stack(deps_p_stack)


ax[1, 0].plot(dgamma_p_stack, deps_p_v_stack)
ax[1, 0].set_xlabel("$d\gamma^p$")
ax[1, 0].set_ylabel("$d\epsilon_v^p$")


plt.tight_layout()
plt.show()
