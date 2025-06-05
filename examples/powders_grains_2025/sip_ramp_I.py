import jax
import jax.numpy as jnp

import os


import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots
import math

# --- Configuration ---
# JAX Configuration
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import hydraxmpm as hdx

# Matplotlib Configuration
plt.style.use(["science", "no-latex"])
mpl.rcParams["lines.linewidth"] = 2.5
mpl.rcParams["font.size"] = 14
mpl.rcParams["figure.dpi"] = 300
cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Current directory
dir_path = os.path.dirname(os.path.realpath(__file__))


# --- Configuration ---

p_0 = 1000.0  # initial pressure [Pa]

# Shared parameters (used by MCC and mu(I)-LC)

rho_p = 2450.0  # particle density [kg/m^3]
d = 0.005  # grain diameter [m]
M = 1.2  # slope of CSL in q/p space [-]
mu_0 = M / jnp.sqrt(3)  # static bulk friction [-]


# Modified Cam-Clay
lam = 0.04  # [-]
kap = 0.008  # [-]
nu = 0.2  # [-]
ln_N = 0.021  # [-]

# mu(I)-LC
mu_d = 5.4 * mu_0
K = 392000.0  # [Pa]
I_0 = 0.279

# We choose a rho_0,  near the CSL in ln v-ln p space.
# for illustrative purposes
v_cs = 0.7573950354136967
phi_ref_cs = 1.0 / (v_cs * (p_0 / K + 1))
rho_0_cs = rho_p * phi_ref_cs


constitutive_laws = (
    hdx.ModifiedCamClay(
        name="MCC OCR=1",
        other=dict(label="MCC OCR=1", ls="-", zorder=-1, color=cycle[0]),
        R=1,
        nu=nu,
        M=M,
        lam=lam,
        kap=kap,
        d=d,
        ln_N=ln_N,
        rho_p=rho_p,
    ),
    hdx.ModifiedCamClay(
        name="MCC OCR=4",
        other=dict(label="MCC OCR=4", ls="--", zorder=1, color=cycle[0]),
        R=4,
        nu=nu,
        M=M,
        lam=lam,
        kap=kap,
        d=d,
        ln_N=ln_N,
        rho_p=rho_p,
    ),
    # initial density/ pressure is selected to match steady state of mcc
    hdx.MuI_incompressible(
        name="MU I LC",
        other=dict(label="$ \\mu (I)$-LC", ls="-", zorder=0, color=cycle[1]),
        K=K,
        d=d,
        I_0=I_0,
        mu_d=mu_d,
        mu_s=mu_0,
        rho_0=rho_0_cs,
        rho_p=rho_p,
    ),
)

# --- Simulation Setup ---

# Simulation Stages (based on Inertial Number I)
# Stage 1: Slow shearing (I = I_slow)
# Stage 2: Transition from slow to fast shearing
# Stage 3: Fast shearing (I = I_fast)

I_slow = 0.001
I_fast = 0.1

num_steps_s1 = 200000
num_steps_s2 = 200000
num_steps_s3 = 200000
total_num_steps = num_steps_s1 + num_steps_s2 + num_steps_s3

# Shear rate calculation (dot gamma = I * sqrt(p/rho_p) / d)
dgamma_dt_slow = (I_slow / d) * jnp.sqrt(p_0 / rho_p)
dgamma_dt_fast = (I_fast / d) * jnp.sqrt(p_0 / rho_p)


dgamma_dt_s1_stack = jnp.full(num_steps_s1, dgamma_dt_slow)
dgamma_dt_s2_stack = jnp.linspace(dgamma_dt_slow, dgamma_dt_fast, num_steps_s2)
dgamma_dt_s3_stack = jnp.full(num_steps_s3, dgamma_dt_fast)

dgamma_dt_all_stack = jnp.concat(
    (dgamma_dt_s1_stack, dgamma_dt_s2_stack, dgamma_dt_s3_stack)
)

sip_benchmark = hdx.ConstantPressureShear(
    deps_xy_dt=dgamma_dt_all_stack,
    num_steps=total_num_steps,
    p0=p_0,
    # initialize values of the material point
    init_material_points=True,
)

dt = 1e-5  # time Step [s]

# time vector for plotting
time_stack = jnp.arange(0, total_num_steps) * dt

fig, axes = plt.subplots(
    ncols=4, nrows=1, figsize=(14, 3), dpi=150, layout="constrained"
)

for ci, model in enumerate(constitutive_laws):
    solver = hdx.SIPSolver(
        # Specify desired outputs
        output_vars=(
            "p_stack",
            "q_stack",
            "gamma_stack",
            "dgamma_dt_stack",
            "rho_stack",
            "specific_volume_stack",
            "inertial_number_stack",
            "viscosity_stack",
        ),
        constitutive_law=model,
        sip_benchmarks=sip_benchmark,
    )

    # Assigns internal variables etc.
    solver = solver.setup()

    # Run the simulation
    (
        p_stack,
        q_stack,
        gamma_stack,
        dgamma_dt_stack,
        rho_stack,
        specific_volume_stack,
        inertial_number_stack,
        viscosity_stack,
    ) = solver.run(dt=dt)

    # Plot 1: q/p vs time
    q_p_stack = q_stack / p_stack

    hdx.make_plot(
        axes.flat[0],
        time_stack,
        q_p_stack,
        xlabel="t [s]",
        ylabel="$q/p$ [-]",
        label=model.other.get("label", None),
        color=model.other.get("color", "black"),
        ls=model.other.get("ls", "-"),
        zorder=model.other.get("zorder", 1),
        start_end_markersize=10,
    )

    # Plot 2: Viscosity vs time (log scale)
    hdx.make_plot(
        axes.flat[1],
        time_stack,
        viscosity_stack,
        xlabel="t [s]",
        ylabel="$\\eta$ [Pa.s] (log-scale)",
        color=model.other["color"],
        ls=model.other["ls"],
        zorder=model.other["zorder"],
        start_end_markersize=10,
        ylogscale=True,
    )

    # Plot 3: Specific Volume vs Pressure (log-log scale)
    hdx.make_plot(
        axes.flat[2],
        p_stack,
        specific_volume_stack,
        xlabel="$p$ [Pa] (log-scale)",
        ylabel="$v=\\phi^{-1}$ [-] (log-scale)",
        color=model.other["color"],
        ls=model.other["ls"],
        zorder=model.other["zorder"],
        xlogscale=True,
        ylogscale=True,
        start_end_markersize=10,
    )

    # Plot 4: q vs p plane
    hdx.make_plot(
        axes.flat[3],
        p_stack,
        q_stack,
        xlabel="$p$ [Pa]",
        ylabel="$q$ [Pa]",
        color=model.other["color"],
        ls=model.other["ls"],
        zorder=model.other["zorder"],
        start_end_markersize=10,
    )


# --- Plot Enhancements ---
model_ref = constitutive_laws[0]


# Plot 1: q/p vs time extras
# Add CSL
axes.flat[0].axhline(
    model_ref.M,
    color="r",
    ls="-",
    lw=1.25,
    zorder=2,
    label="CSL",
)
axes.flat[0].set_xticks([0, 2, 4, 6])

axes.flat[0].set_yticks(jnp.linspace(0, 2.4, 5))

# Plot 2: Viscosity vs time (log scale) extras

# Add reference viscosity line (eta = q / dgamma_dt = p * M / dgamma_dt)
# Note: This eta_csl depends on dgamma_dt, so it varies over time

eta_csl_ref = (jnp.ones(total_num_steps) * model_ref.M * jnp.sqrt(3) * p_0) / (
    dgamma_dt_all_stack / 2
)
axes.flat[1].plot(time_stack, eta_csl_ref, "r-", lw=1.25, zorder=1)
axes.flat[1].set_ylim(bottom=10)
axes.flat[1].set_yticks([1, 1e1, 1e2, 1e3, 1e4, 1e5], minor=True)


axes.flat[1].yaxis.set_minor_formatter(
    mpl.ticker.FuncFormatter(lambda v, _: ("$10^{%d}$" % math.log(v, 10)))
)
axes.flat[1].yaxis.set_major_formatter(
    mpl.ticker.FuncFormatter(lambda v, _: ("$10^{%d}$" % math.log(v, 10)))
)
axes.flat[1].set_xticks([0, 2, 4, 6])

# Plot 3: Specific Volume vs Pressure (log-log scale) extras
p_range_swelling = jnp.arange(500, model_ref.R * 1500, 10)
p_range_csl = jnp.arange(0, 1500, 10)
v_csl = constitutive_laws[0].CSL(p_range_csl)

# CSL
axes.flat[2].plot(p_range_csl, v_csl, "r-", lw=1.25, zorder=-1)

# OCL
v_ocl = constitutive_laws[0].SL(
    p_range_swelling,
    jnp.log(0.7407794446688467),  # initial specific volume of MCC OCR=4
    1000,
)
axes.flat[2].plot(
    p_range_swelling,
    v_ocl,
    color="dimgray",
    ls="-",
    lw=1.25,
    label="OCL",
    zorder=-1,
)
v_icl = constitutive_laws[0].ICL(p_range_csl)
axes.flat[2].plot(
    p_range_csl,
    v_icl,
    color="black",
    ls="-",
    lw=1.25,
    label="ICL",
    zorder=-1,
)

axes.flat[2].set_xticks([500, 750, 1000, 1500], minor=True)
axes.flat[2].set_yticks([0.74, 0.76, 0.78, 0.8], minor=True)
# axes.flat[2].set_ylim(0.73, 0.8)
axes.flat[2].set_ylim(0.73, 0.8)
axes.flat[2].set_xlim([500, 1500])

# Plot 4: q vs p plane
axes.flat[3].set_xticks([0, 500, 1000, 1500])
axes.flat[3].set_yticks([0, 500, 1000, 1500, 2000, 2500])
axes.flat[3].set_xlim([0, 1500])
axes.flat[3].set_ylim([0, 2500])


q_csl = model_ref.CSL_q_p(p_range_csl)

axes.flat[3].plot(p_range_csl, q_csl, "r-", lw=1.5, zorder=-1)

# --- Plot Annotations ---
# Draw vertical lines, and annotate Inertial Number stages
stage_labels = [
    # "$I_1=0.001$",
    # "$I_2: $ \n $ 0.001$ \n $ \\rightarrow $ \n $0.1$",
    # "$I_3=0.1$",
    "$I_1$",
    "$I_2$",
    "$I_3$",
]
stage_times = [
    num_steps_s1 * dt,
    (num_steps_s1 + num_steps_s2) * dt,
    (num_steps_s1 + num_steps_s2 + num_steps_s3) * dt,
]
times_prev = 0.0
for times, lables in zip(stage_times, stage_labels):
    x_coords = [0.1, 0.01]
    for i in range(2):
        axes.flat[i].axvline(x=times, color="black", zorder=-1, ls="--", lw=1.0)
        axes.flat[i].text(
            (times - times_prev) / 2 + times_prev,
            # x_coords[i],
            0.2,
            lables,
            color="black",
            ha="center",
            va="center",
            rotation=0,
            fontsize=14,
            transform=axes.flat[i].get_xaxis_transform(),
        )
    times_prev = times


for i, label in enumerate(["(a)", "(b)", "(c)", "(d)"]):
    axes.flat[i].set_title(label, y=0, pad=-42, verticalalignment="top")


def create_legend(fig, ncols=4):
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(
        lines,
        labels,
        ncols=ncols,
        loc="outside lower center",
        # numpoints=1,
        # handlelength=4.0,
    )
    return fig


create_legend(fig, 6)
fig.savefig(dir_path + "/plots/sip_pressure_control.png")
