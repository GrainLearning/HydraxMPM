import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import matplotlib.pyplot as plt


import hydraxmpm as hdx
import plotting as ps

import os

from enum import Enum

import jax.numpy as jnp

import pandas as pd


from sip_setup import SIPConfig, LondonClayProcedure



dir_path = os.path.dirname(os.path.realpath(__file__))

plot_dir = os.path.join(dir_path, "./figures")


kPa = 1000.0


###########################################################################
# Load data
###########################################################################


file_path = f"{dir_path}/data/S1LCrA2 - shearing.xls"
# L: Time [hr], N: q [kPa], O: p' [kPa], S: eps_a [%], X: rate [%/hr]
hour = 3600.0
column_names = ["time", "q", "p", "eps_a", "rate"]

df = pd.read_excel(
    file_path,
    usecols="A,N,O,S,X",
    skiprows=14,
    names=column_names,
    sheet_name="Data",
)
df = df.dropna()

# Preprocess Data

df["time"] = df["time"].values - df["time"].values[0]

# Stress Conversions (kPa -> Pa)
time_steps = df["time"].diff().fillna(1e-16).values  # convert hours to seconds
dt = jnp.array(time_steps)

rates = jnp.array(
    df["rate"].abs().values / 100.0 / hour
)  # (converts to strain/s)
q_target = (
    jnp.array(df["q"].values) * 1000.0
)  # convert to Pa

q_target = q_target - q_target[0]


###########################################################################
# Loading procedure for numerical element simulations
###########################################################################

loading_procedure = LondonClayProcedure(
    is_undrained=True,
)

exp_config = SIPConfig(
    I_M= 1e20,
    M_inf=0.89*1.0000000001,
    confine = 300_000.0 # 300 kPa
)

###########################################################################
# Loading procedure for numerical element simulations
###########################################################################

jitted_run = jax.jit(loading_procedure)

# Result shapes will be (Batch_Size, Time_Steps, ...)
mp_traj, law_traj, axial_strain = jitted_run(exp_config,rates,dt)

print("Simulation Complete.")


###########################################################################
# Plot results
###########################################################################


fig = plt.figure(figsize=ps.FIG_SIZES["single_col"])
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel(ps.LABELS["axial_strain_perc"])
ax.set_ylabel(ps.LABELS["deviatoric_stress"])

ax.plot(axial_strain * 100, mp_traj.q_stack/kPa, ls="-", label="Simulation", lw=2., color="tab:orange")


ax.plot(
    axial_strain * 100.0,
    q_target / kPa,
    marker="o",
    markeredgecolor="black",
    color="tab:green",
    markevery=10,
    alpha=0.6,
    markersize=4,
    ls="",
    label="S{\o}rensen et al., 2007",
    zorder=-1,
)


# Annotate stress jumps
ps.annotate_point(
    ax,
    label=r"$18\dot{\varepsilon}_{a,0}$",
    xy=[1.2, 200.0], 
    arrow=False,

)

ps.annotate_point(
    ax,
    label="",
    xy=[4.0, 185.0],
    arrow=True,
    xytext=[1.85, 200.0],
)

ps.annotate_point(
    ax,
    label="",
    xy=[2.2, 150.0],
    arrow=True,
    xytext=[1.4, 185.0],
)

ps.annotate_point(
    ax,
    label="",
    xy=[1.2, 140.0],
    arrow=True,
    xytext=[1.2, 187.0],
)

ps.annotate_point(
    ax,
    label=r"$\dot{\varepsilon}_{a,0}=0.05~\%/\mathrm{hour}$",
    xy=[6.5, 70.0], 
    arrow=False,
)


ps.annotate_point(
    ax,
    label="",
    xy=[3.5, 150.0],
    arrow=True,
    xytext=[4.25, 80.0],
)

ps.annotate_point(
    ax,
    label="",
    xy=[2.0, 130.0],
    arrow=True,
    xytext=[4.0, 80.0],
)

ps.annotate_point(
    ax,
    label="",
    xy=[0.6, 80.0],
    arrow=True,
    xytext=[3.75, 80.0],
)


plt.tight_layout()

ax.legend(frameon=True, loc="lower right")

output_file = f"{plot_dir}/results_london_clay_undrained.pdf"
print(f"Saving to {output_file}")
plt.savefig(output_file, dpi=300)
