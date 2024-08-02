import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import pymudokon as pm


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


material = pm.DruckerPrager.create(E=5e5, nu=0.2, M=0.4, M2=0.4, M_hat=0.6, c0=1e4, H=6e5)

dt = 0.0001
store_every = 500

benchmark = pm.MPBenchmark.preset_create_simpleshear(
    material=material,
    total_time=1.0,
    # total_time=dt * 9,
    dt=dt,
    target=0.5,
    target_start=0.0,
    volume_fraction=0.8,
    stress_ref=-1e5 * jnp.eye(3).reshape(3, 3),
    keys=(
        "stress",
        "strain",
        "strain_rate",
        "volume_fraction",
        "c0",
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


fig, ax = plt.subplots(2, 2, figsize=(8, 4))

# q-p plot
q_stack = pm.get_q_vm_stack(stress_stack)
p_stack = pm.get_pressure_stack(stress_stack)

ax[0, 0].plot(p_stack, q_stack)
ax[0, 0].set_xlabel("$p$")
ax[0, 0].set_ylabel("$q$")

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
