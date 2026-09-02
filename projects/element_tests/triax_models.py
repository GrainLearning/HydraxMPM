import matplotlib.pyplot as plt
from enum import Enum


import jax

jax.config.update("jax_platform_name", "cpu")

import os

base_path = os.path.dirname(os.path.abspath(__file__)) + "/plots"

import hydraxmpm as hdx

import jax.numpy as jnp

###########################################################################
# Settings
###########################################################################

confine = 10_000.0  # kPa
axial_rate = 0.02  # 2/s
num_steps = 10_000
dt = 0.01  # s
is_undrained = True
ocr = 1.0  # overconsolidation ratio (OCR) for initial state
kPa = 1_000

p_stack = jnp.array([confine])

stress_stack = p_stack[:, None, None] * jnp.eye(3)


# Share these parameters
density0_stack = jnp.atleast_1d(1000.0)  # density at p->0
rho_p = 2000.0
K = 100_000
M = 0.9
v_0 = float(rho_p / density0_stack.at[0].get())

###########################################################################
# Setup MCC logic and state
###########################################################################


class Model(Enum):
    DRUCKER_PRAGER = "dp"
    MODIFIED_CAM_CLAY = "mmc"
    MU_I = "mu_i"
    NEWTON_FLUID = "newton"
    LINEAR_ELASTIC = "linear_elastic"
    

model = Model.MODIFIED_CAM_CLAY

if model == Model.DRUCKER_PRAGER:

    law_logic = hdx.DruckerPragerHE(
        nu=0.3,
        M=M,
        K=K,
        rho_p=rho_p,
        debug_convergence=True,
    )

    density_stack = law_logic.give_density_stack(p_stack, density0_stack)

    law_state = law_logic.create_state(
        stress_stack=stress_stack,
    )
elif model == Model.MODIFIED_CAM_CLAY:

    law_logic = hdx.ModifiedCamClayHE(
        nu=0.3,
        M=M,
        lam=0.2,
        kap=0.05,
        N=float(v_0),
        p_ref=1000.0,
        debug_convergence=True,
    )

    density_stack = law_logic.give_density_stack(p_stack)

    law_state = law_logic.create_state(
        stress_stack=stress_stack,
    )
elif model == Model.MU_I:

    law_logic = hdx.MuI_LC(
        mu_s=M / (3.0**0.5),
        mu_d=1.8 * M / (3.0**0.5),
        I_0=1e-2,
        d_p=1e-3,
        K=K,
        rho_p=rho_p,
    )
    density_stack = law_logic.give_density_stack(p_stack, density0_stack)
    law_state = law_logic.create_state()
elif model == Model.NEWTON_FLUID:
    law_logic = hdx.NewtonFluid(
        K=K,
        viscosity=1e2,
        beta=7.0,
    )
    density_stack = law_logic.give_density_stack(p_stack, density0_stack)

    law_state = law_logic.create_state()
elif model == Model.LINEAR_ELASTIC:
    law_logic = hdx.LinearElasticHE(
        K=K,
        nu=0.3,
    )
    density_stack = law_logic.give_density_stack(p_stack, density0_stack)
    law_state = law_logic.create_state(
        stress_stack=stress_stack,
    )
elif model == Model.LINEAR_ELASTIC:
    law_logic = hdx.LinearElasticHE(
        K=K,
        nu=0.3,
    )
    density_stack = law_logic.give_density_stack(p_stack, density0_stack)
    law_state = law_logic.create_state(
        stress_stack=stress_stack,
    )

elif model == Model.UNIFIED_MCC_INERTIAL:

    law_logic = hdx.UnifiedRM(
        nu=0.3,
        M_csl=M,
        lam=0.2,
        kap=0.05,
        gamma=float(v_0),
        p_ref=1000.0,
        I_v=1e-4,
        I_M=1e-4,
        M_inf=1.5 * M,
        debug_convergence=True,
    )

    density_stack = law_logic.give_density_stack(p_stack)

    law_state = law_logic.create_state(
        stress_stack=stress_stack,
    )

elif model == Model.DRUCKER_PRAGER_INERTIAL:

    law_logic = hdx.DruckerPragerHEI(
        nu=0.3,
        M=M,
        M_inf=1.5 * M,
        K=K,
        rho_p=rho_p,
        I_v=0.75,
        I_M=1e-6,
        d=1e-4,
        debug_convergence=True,
    )

    density_stack = law_logic.give_density_stack(p_stack, density0_stack)

    law_state = law_logic.create_state(
        stress_stack=stress_stack,
    )


plot_file_path = os.path.join(base_path, f"{model.value}_triax.png")
convergence_file_path = os.path.join(base_path, f"{model.value}_convergence.png")


###########################################################################
# Setup element test driver and triaxial test
###########################################################################

driver = hdx.ElementTestDriver(law_logic)


triaxial_test = hdx.TriaxialTest(
    solver=driver,
    confine=confine,
    is_undrained=is_undrained,
    axial_rate=axial_rate,
    num_steps=num_steps,
    dt=dt,
)


mp_state = hdx.MaterialPointState.create(
    stress_stack=stress_stack,
    density_stack=density_stack,
    density0_stack=density0_stack,
    volume_stack=jnp.array([1.0]),
)

print("Setup complete. Ready to run triaxial test.")

# ###########################################################################
# # Run the triaxial test
# ###########################################################################


jitted_triax = jax.jit(triaxial_test.run)

mp_traj, law_traj = jitted_triax(mp_init=mp_state, law_init=law_state)

# verify Δq/Δp=0
# uncomment this to verify
# q_diff = jnp.diff(mp_traj.q_stack)
# p_diff = jnp.diff(mp_traj.pressure_stack)
# plt.plot(q_diff/p_diff,'-o')
# plt.ylim(0,4)
# plt.show()

# ###########################################################################
# # Plot results and reference curves
# ###########################################################################


fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# deviatoric stress vs pressure - (p,q)
color = "blue"
axs[0, 0].plot(mp_traj.pressure_stack / kPa, mp_traj.q_stack / kPa, "-", color=color)
axs[0, 0].plot(
    mp_traj.pressure_stack[-1] / kPa, mp_traj.q_stack[-1] / kPa, "x", color=color, ms=10
)
axs[0, 0].plot(
    mp_traj.pressure_stack[0] / kPa, mp_traj.q_stack[0] / kPa, "o", color=color, ms=10
)
axs[0, 0].set_xlabel("Pressure p (kPa)")
axs[0, 0].set_ylabel("Deviatoric stress q (kPa)")
axs[0, 0].plot([0, 17], [0, M * 17], "r--", label="Failure Envelope q=Mp")
axs[0, 0].grid(True)
plt.xlim(0, 15)
plt.ylim(0, 15)

# deviatoric strain vs pressure - (eps_q,q)
axs[0, 1].plot(
    triaxial_test.axial_strain_stack, mp_traj.q_stack / kPa, "-o", markevery=100
)
axs[0, 1].set_xlabel("Deviatoric Strain")
axs[0, 1].set_ylabel("Deviatoric Stress q (kPa)")
axs[0, 1].grid(True)


# bilogarithmic pressure--specific volume - (p,v)
axs[1, 0].plot(
    mp_traj.pressure_stack / kPa, rho_p / mp_traj.density_stack, "-o", markevery=100
)
# axs[1, 0].plot(mp_traj.pressure_stack[-2:]/kPa, (dp.rho_p/mp_traj.density_stack)[-2:], marker="o",color="r")
axs[1, 0].set_xlabel("Pressure p (kPa)")
axs[1, 0].set_ylabel("Specific Volume v")
axs[1, 0].set_xscale("log")
axs[1, 0].set_yscale("log")
axs[1, 0].set_ylim(v_0 * 0.5, v_0 * 1.05)

axs[1, 0].set_xlim(1, 100)
axs[1, 0].grid(True)

if triaxial_test.is_undrained:
    axs[1, 1].plot(
        triaxial_test.axial_strain_stack,
        mp_traj.pressure_stack / kPa,
        "-o",
        markevery=100,
    )
    axs[1, 1].set_ylim(0, 10)
    axs[1, 1].set_xlim(0.0, 0.2)
    axs[1, 1].set_xlabel("Pressure p (kPa)")
    axs[1, 1].set_ylabel("Volumetric Strain εᵥ")
else:
    axs[1, 1].plot(
        triaxial_test.axial_strain_stack, mp_traj.eps_v_stack, "-o", markevery=100
    )
    axs[1, 1].set_xlabel("Axial Strain εₐ")
    axs[1, 1].set_ylabel("Volumetric Strain εᵥ")
    axs[1, 1].grid(True)
    axs[1, 1].set_ylim(-0.2, 0.2)
    axs[1, 1].set_xlim(0.0, 0.2)

axs[1, 1].set_xlabel("Axial Strain εₐ")

plt.savefig(os.path.join(base_path, plot_file_path), dpi=300)


# ###########################################################################
# Convergence plots
# ###########################################################################

has_debug_convergence = hasattr(law_logic, "debug_convergence")
debug_convergence_enabled = getattr(law_logic, "debug_convergence", False)

if not has_debug_convergence or not debug_convergence_enabled:
    print("Convergence data not available or not enabled. Skipping convergence plots.")
    exit(0)

res = law_traj.convergence.residuals
unk = law_traj.convergence.unknowns

steps_to_plot = [0, num_steps // 4, num_steps // 2, -1]

# Compute residual norm per iteration


fig2, ax2 = plt.subplots(figsize=(10, 6))
for step in steps_to_plot:
    if step < 0:
        step = num_steps + step

    # if law_logic.NUM_UNKNOWNS == 1:
    # ax2.semilogy(res[:, step], "o-", label=f"Step {step}")
    # else:
    ax2.semilogy(jnp.abs(res[step]), "o-", label=f"Step {step}")


ax2.set_xlabel("Newton iteration")
ax2.set_ylabel("Residual norm")
ax2.set_title("Convergence history at selected time steps")
ax2.grid(True)
ax2.legend()

plt.savefig(os.path.join(base_path, convergence_file_path), dpi=300)
