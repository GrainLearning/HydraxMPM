import matplotlib.pyplot as plt


import jax

# Use CPU for this example
jax.config.update("jax_platform_name", "cpu")

import os

base_path = os.path.dirname(os.path.abspath(__file__))

import hydraxmpm as hdx

import jax.numpy as jnp


###########################################################################
# Settings
###########################################################################

confine = 10_000.0  # kPa
axial_rate = 0.02  # 2/s
num_steps = 10000
dt = 0.01  # s
is_undrained = False 
ocr = 1.0  # overconsolidation ratio (OCR) for initial state
kPa = 1_000 

###########################################################################
# Setup MCC logic and state
###########################################################################


mcc = hdx.ModifiedCamClay(nu=0.3, M=0.9, lam=0.2, kap=0.05, N=2.0, p_ref=1000.0)


p_stack = jnp.array([confine])  # kPa

law_state, stress_ref_stack, density_stack = mcc.create_state_from_ocr(
    p_stack=p_stack, ocr_stack=ocr
)

###########################################################################
# Setup element test driver and triaxial test
###########################################################################


driver = hdx.ElementTestDriver(mcc)


triaxial_test = hdx.TriaxialTest(
    solver=driver,
    confine=confine, 
    is_undrained=is_undrained,
    axial_rate=axial_rate, 
    num_steps=num_steps,
    dt=dt,  
)


mp_state = hdx.MaterialPointState.create(
    stress_stack=stress_ref_stack, density_stack=density_stack
)


###########################################################################
# Run the triaxial test 
###########################################################################


jitted_triax = jax.jit(triaxial_test.run)

mp_traj, law_traj = jitted_triax(mp_init=mp_state, law_init=law_state)

# DEBUG
# verify Δq/Δp=0
# uncomment this to verify
# q_diff = jnp.diff(mp_traj.q_stack)
# p_diff = jnp.diff(mp_traj.pressure_stack)
# plt.plot(q_diff/p_diff,'-o')
# plt.ylim(0,4)
# plt.show()

###########################################################################
# Plot results and reference curves
###########################################################################


fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# deviatoric stress vs pressure - (p,q)
axs[0, 0].plot(mp_traj.pressure_stack / kPa, mp_traj.q_stack / kPa, "-o", markevery=100)
axs[0, 0].set_xlabel("Pressure p (kPa)")
axs[0, 0].set_ylabel("Deviatoric stress q (kPa)")
axs[0, 0].plot([0, 17], [0, mcc.M * 17], "r--", label="Failure Envelope q=Mp")
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
    mp_traj.pressure_stack / kPa, mcc.rho_p / mp_traj.density_stack, "-o", markevery=100
)
# axs[1, 0].plot(mp_traj.pressure_stack[-2:]/kPa, (mcc.rho_p/mp_traj.density_stack)[-2:], marker="o",color="r")
axs[1, 0].set_xlabel("Pressure p (kPa)")
axs[1, 0].set_ylabel("Specific Volume v")
axs[1, 0].set_xscale("log")
axs[1, 0].set_yscale("log")
axs[1, 0].set_ylim(mcc.N * 0.5, mcc.N)

v_ncl = hdx.ModifiedCamClay.get_v_ncl(
    jnp.array([1000, 100_000]),
    mcc.p_ref,
    mcc.N,
    mcc.lam,
)
axs[1, 0].plot([1, 100], v_ncl, "b--", label="Normal Consolidation Line")

v_csl = hdx.ModifiedCamClay.get_v_csl(
    jnp.array([1000, 100_000]),
    mcc.p_ref,
    mcc.N,
    mcc.lam,
    mcc.kap,
)
axs[1, 0].plot([1, 100], v_csl, "r--", label="Critical State Line")

v_sl = hdx.ModifiedCamClay.get_v_sl(
    10_000, jnp.array([10, law_state.p_c_stack[0]]), mcc.p_ref, mcc.N, mcc.lam, mcc.kap
)
axs[1, 0].plot([1, 10], v_sl, "g--", label="Swelling Line")


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
else:
    axs[1, 1].plot(
        triaxial_test.axial_strain_stack, mp_traj.eps_v_stack, "-o", markevery=100
    )
    axs[1, 1].set_xlabel("Axial Strain εₐ")
    axs[1, 1].set_ylabel("Volumetric Strain εᵥ")
    axs[1, 1].grid(True)
    axs[1, 1].set_ylim(-0.2, 0.2)
    axs[1, 1].set_xlim(0.0, 0.2)



plt.savefig(os.path.join(base_path, "mcc_triax.png"), dpi=300)

